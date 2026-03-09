import argparse
import json
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import save_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from cube3d.inference.utils import (
    load_config,
    load_model_weights,
    select_device,
)
from cube3d.model.gpt.block_diffusion_dit import BlockDiffusionDiT
from cube3d.train.data.block_diffusion_dataset import BlockDiffusionDataset
from cube3d.train.noise.bd3_schedule import (
    LogLinearSchedule,
    q_xt,
    restrict_logits_to_codes_and_mask,
    subs_parameterization,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train Block Diffusion text-to-shape model")
    parser.add_argument(
        "--config",
        type=str,
        default="cube3d/configs/block_diffusion_t2s.yaml",
        help="Path to block diffusion training config.",
    )
    return parser.parse_args()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _resolve_amp_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    return torch.bfloat16


def _autocast_context(device: torch.device, amp_dtype: torch.dtype):
    return torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=device.type in {"cuda", "mps"},
    )


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _ddp_world_size() -> int:
    return dist.get_world_size() if _is_distributed() else 1


def _ddp_rank() -> int:
    return dist.get_rank() if _is_distributed() else 0


def _reduce_mean(value: float, device: torch.device) -> float:
    if not _is_distributed():
        return float(value)
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= _ddp_world_size()
    return float(tensor.item())


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_clip_intervals(
    eps_min: float,
    eps_max: float,
    clip_search_delta: float,
    clip_search_widths: list,
) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = [(float(eps_min), float(eps_max))]
    if clip_search_delta <= 0 or len(clip_search_widths) == 0:
        return intervals

    for width in clip_search_widths:
        w = float(width)
        if w <= 0 or w > 1:
            continue
        i = 0.0
        limit = max(0.0, 1.0 - w)
        while i <= limit + 1e-8:
            lo = max(float(eps_min), i)
            hi = max(float(eps_min), i + w)
            if hi <= 1.0 + 1e-8:
                intervals.append((round(lo, 6), round(min(1.0, hi), 6)))
            i += clip_search_delta

    seen: set[tuple[float, float]] = set()
    deduped: list[tuple[float, float]] = []
    for pair in intervals:
        if pair in seen:
            continue
        seen.add(pair)
        deduped.append(pair)
    return deduped


def _build_optimizer(
    cfg: DictConfig,
    model: torch.nn.Module,
    is_main: bool,
) -> torch.optim.Optimizer:
    optimizer_name = str(OmegaConf.select(cfg, "train.optimizer", default="adamw")).lower()
    lr = float(cfg.train.lr)
    betas = (float(cfg.train.beta1), float(cfg.train.beta2))
    weight_decay = float(cfg.train.weight_decay)

    if optimizer_name == "adamw_8bit":
        try:
            import bitsandbytes as bnb

            if is_main:
                print("[info] using bitsandbytes AdamW8bit optimizer")
            return bnb.optim.AdamW8bit(
                model.parameters(),
                lr=lr,
                betas=betas,
                weight_decay=weight_decay,
            )
        except ImportError:
            if is_main:
                print("[warn] bitsandbytes not found, fallback to torch.optim.AdamW")

    if optimizer_name == "adamw_zero":
        if _is_distributed():
            from torch.distributed.optim import ZeroRedundancyOptimizer

            if is_main:
                print("[info] using ZeroRedundancyOptimizer(AdamW)")
            return ZeroRedundancyOptimizer(
                model.parameters(),
                optimizer_class=torch.optim.AdamW,
                lr=lr,
                betas=betas,
                weight_decay=weight_decay,
            )
        if is_main:
            print("[warn] adamw_zero requested without DDP; fallback to torch.optim.AdamW")

    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
    )


def _prepare_cfg_unconditional(
    text_hidden: torch.Tensor,
    text_attention_mask: torch.Tensor,
    bbox_xyz: torch.Tensor,
    drop_prob: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if drop_prob <= 0:
        return text_hidden, text_attention_mask, bbox_xyz

    bsz = int(text_hidden.shape[0])
    drop_mask = torch.rand((bsz,), device=text_hidden.device) < drop_prob
    if not drop_mask.any():
        return text_hidden, text_attention_mask, bbox_xyz

    text_hidden = text_hidden.clone()
    text_attention_mask = text_attention_mask.clone()
    bbox_xyz = bbox_xyz.clone()

    text_hidden[drop_mask] = 0
    text_attention_mask[drop_mask] = False
    bbox_xyz[drop_mask] = 0
    return text_hidden, text_attention_mask, bbox_xyz


def _forward_bd_training(
    model: torch.nn.Module,
    shape_ids: torch.Tensor,
    noisy_ids: torch.Tensor,
    text_hidden: torch.Tensor,
    text_attention_mask: torch.Tensor,
    bbox_xyz: torch.Tensor,
    block_size: int,
    sigma: torch.Tensor,
) -> torch.Tensor:
    x_input = torch.cat([noisy_ids, shape_ids], dim=1)
    return model(
        x_input,
        sigma=sigma,
        attention_mode="bd_training",
        block_size=block_size,
        text_hidden=text_hidden,
        text_attention_mask=text_attention_mask,
        bbox_xyz=bbox_xyz,
        sample_mode=False,
        store_kv=False,
    )


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    base_model: BlockDiffusionDiT,
    loader: DataLoader,
    schedule: LogLinearSchedule,
    num_codes: int,
    block_size: int,
    amp_dtype: torch.dtype,
    device: torch.device,
    max_batches: int,
    eps_min: float,
    eps_max: float,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_count = 0.0
    total_acc = 0.0
    block_nll_means: list[float] = []

    mask_token_id = int(base_model.ensure_mask_token())

    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        shape_ids = batch["shape_ids"].to(device, non_blocking=True)
        text_hidden = batch["text_hidden"].to(device, non_blocking=True)
        text_attention_mask = batch["text_attention_mask"].to(device, non_blocking=True)
        bbox_xyz = batch["bbox_xyz"].to(device, non_blocking=True)

        bsz, seq_len = shape_ids.shape
        t = schedule.sample_t(
            batch_size=bsz,
            seq_len=seq_len,
            block_size=block_size,
            device=device,
            eps_min=eps_min,
            eps_max=eps_max,
        )
        loss_scale, move_chance = schedule.compute_loss_scaling_and_move_chance(t)
        sigma = schedule.sigma_from_move_chance(move_chance[:, :1])
        noisy_ids, _ = q_xt(
            shape_ids,
            move_chance=move_chance,
            mask_token_id=mask_token_id,
            block_size=block_size,
            resample=schedule.resample,
            eps_min=eps_min,
            eps_max=eps_max,
        )

        with _autocast_context(device, amp_dtype):
            logits = _forward_bd_training(
                model=model,
                shape_ids=shape_ids,
                noisy_ids=noisy_ids,
                text_hidden=text_hidden,
                text_attention_mask=text_attention_mask,
                bbox_xyz=bbox_xyz,
                block_size=block_size,
                sigma=sigma,
            )

        logits, mask_local_idx = restrict_logits_to_codes_and_mask(
            logits=logits,
            num_codes=num_codes,
            mask_token_id=mask_token_id,
        )
        log_probs = F.log_softmax(logits.float(), dim=-1)
        noisy_local = noisy_ids.clone()
        noisy_local[noisy_local == mask_token_id] = mask_local_idx
        log_probs = subs_parameterization(log_probs, noisy_local, mask_local_idx)

        nll = -torch.gather(log_probs, -1, shape_ids.unsqueeze(-1)).squeeze(-1)
        weighted_nll = nll * loss_scale
        loss = weighted_nll.mean()

        masked = noisy_ids.eq(mask_token_id)
        if masked.any():
            preds = logits.argmax(dim=-1)
            acc = (preds[masked] == shape_ids[masked]).float().mean().item()
        else:
            acc = 0.0

        block_nll = weighted_nll.view(bsz, -1, block_size).mean(dim=-1)
        block_nll_means.extend(block_nll.detach().float().cpu().reshape(-1).tolist())

        total_loss += float(loss.item())
        total_acc += float(acc)
        total_count += 1.0

    stats = torch.tensor([total_loss, total_acc, total_count], device=device, dtype=torch.float64)
    if _is_distributed():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    total_count = float(stats[2].item())
    if total_count == 0:
        return {"val_loss": 0.0, "val_mask_acc": 0.0, "val_block_var": 0.0}

    local_var = float(np.var(block_nll_means)) if block_nll_means else 0.0
    var_mean = _reduce_mean(local_var, device)
    return {
        "val_loss": float(stats[0].item() / total_count),
        "val_mask_acc": float(stats[1].item() / total_count),
        "val_block_var": float(var_mean),
    }


def _init_distributed(cfg: DictConfig) -> tuple[bool, int, int]:
    requested = bool(OmegaConf.select(cfg, "runtime.distributed", default=False))
    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    has_rank_env = "RANK" in os.environ and "LOCAL_RANK" in os.environ
    use_distributed = world_size_env > 1 or has_rank_env
    if requested and not use_distributed:
        return False, 0, 0
    if not use_distributed:
        return False, 0, 0

    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    return True, rank, local_rank


def _build_model_cfg(cfg: DictConfig, base_cfg: DictConfig) -> BlockDiffusionDiT.Config:
    model_cfg = OmegaConf.select(cfg, "diffusion.model", default={})

    def _pick(path: str, default):
        value = OmegaConf.select(model_cfg, path, default=None)
        return default if value is None else value

    default_layers = int(base_cfg.gpt_model.n_layer + base_cfg.gpt_model.n_single_layer)
    n_layer = int(_pick("n_layer", default_layers))
    n_head = int(_pick("n_head", int(base_cfg.gpt_model.n_head)))
    n_embd = int(_pick("n_embd", int(base_cfg.gpt_model.n_embd)))
    cond_dim = int(_pick("cond_dim", n_embd))
    text_cond_dim = int(
        _pick("text_cond_dim", int(base_cfg.gpt_model.text_model_embed_dim))
    )
    rope_theta = float(_pick("rope_theta", float(base_cfg.gpt_model.rope_theta)))
    use_bbox = bool(_pick("use_bbox", bool(base_cfg.gpt_model.use_bbox)))

    return BlockDiffusionDiT.Config(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        eps=float(_pick("eps", float(base_cfg.gpt_model.eps))),
        rope_theta=rope_theta,
        shape_model_vocab_size=int(base_cfg.shape_model.num_codes),
        cond_dim=cond_dim,
        text_cond_dim=text_cond_dim,
        use_bbox=use_bbox,
        time_conditioning=bool(_pick("time_conditioning", False)),
        dropout=float(_pick("dropout", 0.0)),
        cross_attn_dropout=float(_pick("cross_attn_dropout", 0.0)),
        attn_backend=str(_pick("attn_backend", "flash_attn")),
        max_seqlen=int(
            _pick("max_seqlen", int(base_cfg.shape_model.num_encoder_latents))
        ),
        add_mask_token=bool(_pick("add_mask_token", True)),
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    assert isinstance(cfg, DictConfig)

    use_distributed, rank, local_rank = _init_distributed(cfg)
    if bool(OmegaConf.select(cfg, "runtime.distributed", default=False)) and not use_distributed:
        print("[warn] runtime.distributed=true but torchrun env not found; running single process.")

    if use_distributed and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = (
            select_device()
            if cfg.runtime.device == "auto"
            else torch.device(str(cfg.runtime.device))
        )
        if device.type == "cuda":
            cuda_idx = 0 if device.index is None else int(device.index)
            torch.cuda.set_device(cuda_idx)
            device = torch.device("cuda", cuda_idx)

    is_main = rank == 0
    world_size = _ddp_world_size()
    amp_dtype = _resolve_amp_dtype(str(cfg.train.amp_dtype))

    base_seed = int(OmegaConf.select(cfg, "runtime.seed", default=42))
    _set_seed(base_seed + rank)

    base_cfg = load_config(cfg.model.base_config_path)
    output_dir = Path(str(cfg.runtime.output_dir))
    _ensure_dir(output_dir)
    if is_main:
        with open(output_dir / "resolved_train_config.json", "w", encoding="utf-8") as f:
            resolved = OmegaConf.to_container(cfg, resolve=True)
            f.write(json.dumps(resolved, indent=2))

    train_ds = BlockDiffusionDataset(
        manifest_path=str(cfg.data.manifest_path),
        split="train",
        val_ratio=float(cfg.data.val_ratio),
        seed=int(cfg.data.seed),
    )
    val_ds = BlockDiffusionDataset(
        manifest_path=str(cfg.data.manifest_path),
        split="val",
        val_ratio=float(cfg.data.val_ratio),
        seed=int(cfg.data.seed),
    )
    if len(train_ds) == 0:
        raise RuntimeError("No training samples found. Check manifest and split setup.")

    micro_batch_size = int(
        OmegaConf.select(cfg, "train.micro_batch_size_per_gpu", default=int(cfg.train.batch_size))
    )
    grad_accum_steps = int(OmegaConf.select(cfg, "train.grad_accum_steps", default=1))
    if grad_accum_steps < 1:
        raise ValueError("train.grad_accum_steps must be >= 1")

    if is_main:
        effective_batch = micro_batch_size * grad_accum_steps * world_size
        print(
            "[info] training setup: "
            f"micro_batch_per_gpu={micro_batch_size}, grad_accum_steps={grad_accum_steps}, "
            f"world_size={world_size}, effective_batch={effective_batch}"
        )

    train_sampler: Optional[DistributedSampler] = None
    val_sampler: Optional[DistributedSampler] = None
    if use_distributed:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )
        val_sampler = DistributedSampler(
            val_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=micro_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=int(cfg.train.num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=micro_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=int(cfg.train.num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    dit_cfg = _build_model_cfg(cfg, base_cfg)
    base_model = BlockDiffusionDiT(dit_cfg).to(device)

    load_ckpt = bool(OmegaConf.select(cfg, "diffusion.model.load_ckpt", default=False))
    if load_ckpt:
        load_model_weights(base_model, str(cfg.model.gpt_ckpt_path))

    base_model.ensure_mask_token()
    base_model.set_gradient_checkpointing(
        bool(OmegaConf.select(cfg, "train.grad_checkpoint", default=False))
    )
    base_model = base_model.train()

    model: torch.nn.Module = base_model
    if use_distributed:
        model = DDP(
            base_model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    num_codes = int(base_cfg.shape_model.num_codes)
    block_size = int(cfg.diffusion.block_size)
    num_steps = int(cfg.train.max_steps)
    grad_clip = float(cfg.train.grad_clip)

    schedule = LogLinearSchedule(
        eps_min=float(OmegaConf.select(cfg, "diffusion.eps_min", default=1.0e-3)),
        eps_max=float(OmegaConf.select(cfg, "diffusion.eps_max", default=1.0)),
        antithetic_sampling=bool(
            OmegaConf.select(cfg, "diffusion.antithetic_sampling", default=True)
        ),
        resample=bool(OmegaConf.select(cfg, "diffusion.resample", default=False)),
    )

    cond_drop_prob = float(OmegaConf.select(cfg, "diffusion.cond_drop_prob", default=0.1))
    var_min_enabled = bool(OmegaConf.select(cfg, "diffusion.var_min", default=False))
    fix_clipping = bool(OmegaConf.select(cfg, "diffusion.fix_clipping", default=False))
    clip_search_delta = float(OmegaConf.select(cfg, "diffusion.clip_search_delta", default=0.05))
    clip_search_widths = list(OmegaConf.select(cfg, "diffusion.clip_search_widths", default=[]))
    current_eps_min = float(schedule.eps_min)
    current_eps_max = float(schedule.eps_max)

    optimizer = _build_optimizer(cfg, model, is_main)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda" and amp_dtype == torch.float16),
    )

    writer = None
    if is_main and bool(OmegaConf.select(cfg, "logging.tensorboard", default=True)):
        tb_log_dir = Path(str(OmegaConf.select(cfg, "logging.tb_log_dir", default=output_dir / "tb")))
        _ensure_dir(tb_log_dir)
        writer = SummaryWriter(log_dir=str(tb_log_dir))
        print(f"[info] tensorboard log_dir={tb_log_dir}")

    step = 0
    running_loss = 0.0
    running_acc = 0.0
    running_count = 0.0
    running_iter_time = 0.0

    train_epoch = 0
    if train_sampler is not None:
        train_sampler.set_epoch(train_epoch)
    train_iter = iter(train_loader)
    start_time = time.time()

    pbar = tqdm(total=num_steps, desc="train_block_diffusion", disable=not is_main)
    try:
        while step < num_steps:
            model.train()
            optimizer.zero_grad(set_to_none=True)

            step_loss_sum = 0.0
            step_acc_sum = 0.0
            step_count = 0.0
            iter_start = time.time()

            for micro_idx in range(grad_accum_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_epoch += 1
                    if train_sampler is not None:
                        train_sampler.set_epoch(train_epoch)
                    train_iter = iter(train_loader)
                    batch = next(train_iter)

                shape_ids = batch["shape_ids"].to(device, non_blocking=True)
                text_hidden = batch["text_hidden"].to(device, non_blocking=True)
                text_attention_mask = batch["text_attention_mask"].to(device, non_blocking=True)
                bbox_xyz = batch["bbox_xyz"].to(device, non_blocking=True)

                text_hidden, text_attention_mask, bbox_xyz = _prepare_cfg_unconditional(
                    text_hidden=text_hidden,
                    text_attention_mask=text_attention_mask,
                    bbox_xyz=bbox_xyz,
                    drop_prob=cond_drop_prob,
                )

                bsz, seq_len = shape_ids.shape
                if seq_len % block_size != 0:
                    raise RuntimeError(
                        f"Sequence length {seq_len} not divisible by block_size {block_size}"
                    )

                t = schedule.sample_t(
                    batch_size=bsz,
                    seq_len=seq_len,
                    block_size=block_size,
                    device=device,
                    eps_min=current_eps_min,
                    eps_max=current_eps_max,
                )
                loss_scale, move_chance = schedule.compute_loss_scaling_and_move_chance(t)
                sigma = schedule.sigma_from_move_chance(move_chance[:, :1])

                noisy_ids, _ = q_xt(
                    shape_ids,
                    move_chance=move_chance,
                    mask_token_id=base_model.ensure_mask_token(),
                    block_size=block_size,
                    resample=schedule.resample,
                    eps_min=current_eps_min,
                    eps_max=current_eps_max,
                )

                sync_context = (
                    model.no_sync()
                    if use_distributed and micro_idx < grad_accum_steps - 1
                    else nullcontext()
                )
                with sync_context:
                    with _autocast_context(device, amp_dtype):
                        logits = _forward_bd_training(
                            model=model,
                            shape_ids=shape_ids,
                            noisy_ids=noisy_ids,
                            text_hidden=text_hidden,
                            text_attention_mask=text_attention_mask,
                            bbox_xyz=bbox_xyz,
                            block_size=block_size,
                            sigma=sigma,
                        )

                    logits, mask_local_idx = restrict_logits_to_codes_and_mask(
                        logits=logits,
                        num_codes=num_codes,
                        mask_token_id=base_model.ensure_mask_token(),
                    )
                    log_probs = F.log_softmax(logits.float(), dim=-1)
                    noisy_local = noisy_ids.clone()
                    noisy_local[noisy_local == base_model.ensure_mask_token()] = mask_local_idx
                    log_probs = subs_parameterization(
                        log_probs=log_probs,
                        xt_local=noisy_local,
                        mask_index_local=mask_local_idx,
                    )

                    nll = -torch.gather(log_probs, -1, shape_ids.unsqueeze(-1)).squeeze(-1)
                    loss = (nll * loss_scale).mean()
                    loss_to_backward = loss / grad_accum_steps

                    if scaler.is_enabled():
                        scaler.scale(loss_to_backward).backward()
                    else:
                        loss_to_backward.backward()

                with torch.no_grad():
                    masked = noisy_ids.eq(base_model.ensure_mask_token())
                    if masked.any():
                        pred = logits.argmax(dim=-1)
                        acc = (pred[masked] == shape_ids[masked]).float().mean().item()
                    else:
                        acc = 0.0

                step_loss_sum += float(loss.item())
                step_acc_sum += float(acc)
                step_count += 1.0

            if scaler.is_enabled():
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                else:
                    grad_norm = torch.tensor(0.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                if grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                else:
                    grad_norm = torch.tensor(0.0)
                optimizer.step()

            step_loss = step_loss_sum / max(step_count, 1.0)
            step_acc = step_acc_sum / max(step_count, 1.0)
            iter_time = time.time() - iter_start
            step_loss = _reduce_mean(step_loss, device)
            step_acc = _reduce_mean(step_acc, device)
            iter_time = _reduce_mean(iter_time, device)
            grad_norm_val = _reduce_mean(float(grad_norm), device)

            running_loss += step_loss
            running_acc += step_acc
            running_iter_time += iter_time
            running_count += 1.0
            step += 1
            pbar.update(1)

            if step % int(cfg.train.log_every) == 0 and is_main:
                avg_loss = running_loss / max(running_count, 1.0)
                avg_acc = running_acc / max(running_count, 1.0)
                avg_iter_time = running_iter_time / max(running_count, 1.0)
                elapsed = time.time() - start_time
                print(
                    f"[train] step={step} loss={avg_loss:.4f} mask_acc={avg_acc:.4f} "
                    f"iter_s={avg_iter_time:.3f} elapsed_s={elapsed:.1f}"
                )
                if writer is not None:
                    writer.add_scalar("train/loss", avg_loss, step)
                    writer.add_scalar("train/mask_acc", avg_acc, step)
                    writer.add_scalar("train/iter_time_sec", avg_iter_time, step)
                    writer.add_scalar("train/lr", float(optimizer.param_groups[0]["lr"]), step)
                    writer.add_scalar("train/grad_norm", grad_norm_val, step)
                    if device.type == "cuda":
                        writer.add_scalar(
                            "system/gpu_mem_allocated_gb",
                            torch.cuda.memory_allocated(device) / (1024**3),
                            step,
                        )
                running_loss = 0.0
                running_acc = 0.0
                running_iter_time = 0.0
                running_count = 0.0

            if step % int(cfg.train.eval_every) == 0 and len(val_ds) > 0:
                if val_sampler is not None:
                    val_sampler.set_epoch(step)
                metrics = evaluate(
                    model=model,
                    base_model=base_model,
                    loader=val_loader,
                    schedule=schedule,
                    num_codes=num_codes,
                    block_size=block_size,
                    amp_dtype=amp_dtype,
                    device=device,
                    max_batches=int(cfg.train.val_max_batches),
                    eps_min=current_eps_min,
                    eps_max=current_eps_max,
                )

                best_interval = (current_eps_min, current_eps_max)
                if var_min_enabled:
                    intervals = _build_clip_intervals(
                        eps_min=current_eps_min,
                        eps_max=current_eps_max,
                        clip_search_delta=clip_search_delta,
                        clip_search_widths=clip_search_widths,
                    )
                    best_var = float("inf")
                    for eps_min_i, eps_max_i in intervals:
                        clip_metrics = evaluate(
                            model=model,
                            base_model=base_model,
                            loader=val_loader,
                            schedule=schedule,
                            num_codes=num_codes,
                            block_size=block_size,
                            amp_dtype=amp_dtype,
                            device=device,
                            max_batches=int(OmegaConf.select(cfg, "train.val_var_batches", default=20)),
                            eps_min=float(eps_min_i),
                            eps_max=float(eps_max_i),
                        )
                        clip_var = float(clip_metrics["val_block_var"])
                        if is_main and writer is not None:
                            writer.add_scalar(
                                f"val/clip_var_{eps_min_i:.3f}_{eps_max_i:.3f}",
                                clip_var,
                                step,
                            )
                        if clip_var < best_var:
                            best_var = clip_var
                            best_interval = (float(eps_min_i), float(eps_max_i))
                    if not fix_clipping:
                        current_eps_min, current_eps_max = best_interval

                if is_main:
                    print(
                        f"[val] step={step} val_loss={metrics['val_loss']:.4f} "
                        f"val_mask_acc={metrics['val_mask_acc']:.4f} "
                        f"val_block_var={metrics['val_block_var']:.6f} "
                        f"eps=[{current_eps_min:.3f},{current_eps_max:.3f}]"
                    )
                    if writer is not None:
                        writer.add_scalar("val/loss", metrics["val_loss"], step)
                        writer.add_scalar("val/mask_acc", metrics["val_mask_acc"], step)
                        writer.add_scalar("val/block_var", metrics["val_block_var"], step)
                        writer.add_scalar("diffusion/eps_min", current_eps_min, step)
                        writer.add_scalar("diffusion/eps_max", current_eps_max, step)

            if step % int(cfg.train.save_every) == 0 or step == num_steps:
                if is_main:
                    ckpt_path = output_dir / f"block_diffusion_step_{step}.safetensors"
                    save_model(base_model, str(ckpt_path))
                    meta = {
                        "step": step,
                        "backbone": "single_stream_dit_official_aligned",
                        "num_codes": num_codes,
                        "block_size": block_size,
                        "shape_mask_id": base_model.shape_mask_id,
                        "eps_min": current_eps_min,
                        "eps_max": current_eps_max,
                        "world_size": world_size,
                        "grad_accum_steps": grad_accum_steps,
                        "micro_batch_size_per_gpu": micro_batch_size,
                        "cond_drop_prob": cond_drop_prob,
                        "cfg_scale_default": float(
                            OmegaConf.select(cfg, "diffusion.cfg_scale", default=1.0)
                        ),
                        "attn_backend": base_model.attn_backend,
                    }
                    with open(
                        output_dir / f"block_diffusion_step_{step}.json",
                        "w",
                        encoding="utf-8",
                    ) as f:
                        json.dump(meta, f, indent=2)
                    print(f"[ckpt] saved {ckpt_path}")
                if _is_distributed():
                    dist.barrier()
    finally:
        pbar.close()
        if writer is not None:
            writer.close()
        if _is_distributed():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
