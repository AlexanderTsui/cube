import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import save_model
from torch.utils.data import DataLoader
from tqdm import tqdm

from cube3d.inference.utils import (
    load_config,
    load_model_weights,
    parse_structured,
    select_device,
)
from cube3d.model.autoencoder.one_d_autoencoder import OneDAutoEncoder
from cube3d.model.gpt.block_diffusion_roformer import BlockDiffusionRoformer
from cube3d.train.data.block_diffusion_dataset import BlockDiffusionDataset
from cube3d.train.noise.masked_schedule import ClippedMaskSchedule, mask_one_block_per_sample


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


def _prepare_condition(
    gpt_model: BlockDiffusionRoformer,
    text_hidden: torch.Tensor,
    bbox_xyz: torch.Tensor,
) -> torch.Tensor:
    cond = gpt_model.encode_text(text_hidden)
    if hasattr(gpt_model, "bbox_proj"):
        cond_bbox = gpt_model.bbox_proj(bbox_xyz).unsqueeze(1)
        cond = torch.cat([cond, cond_bbox], dim=1)
    return cond


def _autocast_context(device: torch.device, amp_dtype: torch.dtype):
    return torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=device.type in {"cuda", "mps"},
    )


@torch.no_grad()
def evaluate(
    model: BlockDiffusionRoformer,
    loader: DataLoader,
    schedule: ClippedMaskSchedule,
    num_codes: int,
    block_size: int,
    amp_dtype: torch.dtype,
    device: torch.device,
    max_batches: int,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    total_acc = 0.0

    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        shape_ids = batch["shape_ids"].to(device)
        text_hidden = batch["text_hidden"].to(device)
        bbox_xyz = batch["bbox_xyz"].to(device)

        bsz, seq_len = shape_ids.shape
        num_blocks = seq_len // block_size
        block_indices = torch.randint(0, num_blocks, (bsz,), device=device)
        ratios = schedule.sample_ratio(bsz, device=device)
        noisy_ids, masked = mask_one_block_per_sample(
            shape_ids=shape_ids,
            block_indices=block_indices,
            mask_ratios=ratios,
            block_size=block_size,
            mask_token_id=model.ensure_mask_token(),
        )

        with _autocast_context(device, amp_dtype):
            cond = _prepare_condition(model, text_hidden, bbox_xyz)
            logits = model(model.encode_token(noisy_ids), cond)[..., :num_codes]

        if not masked.any():
            continue
        target = shape_ids[masked]
        masked_logits = logits[masked]
        ce = F.cross_entropy(masked_logits.float(), target, reduction="none")

        weights = schedule.weight_from_ratio(ratios)
        token_weights = (weights[:, None].expand_as(masked))[masked]
        weighted_loss = (ce * token_weights).mean()

        preds = masked_logits.argmax(dim=-1)
        acc = (preds == target).float().mean().item()

        total_loss += float(weighted_loss.item())
        total_acc += acc
        total_count += 1

    if total_count == 0:
        return {"val_loss": 0.0, "val_mask_acc": 0.0}
    return {
        "val_loss": total_loss / total_count,
        "val_mask_acc": total_acc / total_count,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    assert isinstance(cfg, DictConfig)

    base_cfg = load_config(cfg.model.base_config_path)
    device = (
        select_device()
        if cfg.runtime.device == "auto"
        else torch.device(str(cfg.runtime.device))
    )
    amp_dtype = _resolve_amp_dtype(str(cfg.train.amp_dtype))

    output_dir = Path(str(cfg.runtime.output_dir))
    _ensure_dir(output_dir)
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

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=int(cfg.train.num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=False,
        num_workers=int(cfg.train.num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    gpt_cfg = parse_structured(
        BlockDiffusionRoformer.Config,
        base_cfg.gpt_model,
    )
    model = BlockDiffusionRoformer(gpt_cfg)
    load_model_weights(model, str(cfg.model.gpt_ckpt_path))
    model = model.to(device)

    shape_model = OneDAutoEncoder(
        parse_structured(OneDAutoEncoder.Config, base_cfg.shape_model)
    )
    load_model_weights(shape_model, str(cfg.model.shape_ckpt_path))
    shape_model = shape_model.eval().to(device)

    # Align shape token embeddings with tokenizer codebook for stable initialization.
    with torch.no_grad():
        codebook = shape_model.bottleneck.block.get_codebook()
        codebook = model.shape_proj(codebook).detach()
        codebook = codebook.to(
            model.transformer.wte.weight.device, dtype=model.transformer.wte.weight.dtype
        )
        model.transformer.wte.weight.data[: codebook.shape[0]] = codebook
        model.lm_head.weight.data[: codebook.shape[0]] = codebook

    model.ensure_mask_token()
    model = model.train()
    del shape_model

    num_codes = int(base_cfg.shape_model.num_codes)
    block_size = int(cfg.diffusion.block_size)
    num_steps = int(cfg.train.max_steps)
    grad_clip = float(cfg.train.grad_clip)

    schedule = ClippedMaskSchedule(
        beta_low=float(cfg.diffusion.beta_low),
        beta_high=float(cfg.diffusion.beta_high),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train.lr),
        betas=(float(cfg.train.beta1), float(cfg.train.beta2)),
        weight_decay=float(cfg.train.weight_decay),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and amp_dtype == torch.float16))

    step = 0
    running_loss = 0.0
    running_acc = 0.0
    running_count = 0
    train_iter = iter(train_loader)
    start_time = time.time()

    pbar = tqdm(total=num_steps, desc="train_block_diffusion")
    while step < num_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        model.train()
        shape_ids = batch["shape_ids"].to(device)
        text_hidden = batch["text_hidden"].to(device)
        bbox_xyz = batch["bbox_xyz"].to(device)

        if float(cfg.diffusion.cfg_drop_prob) > 0:
            drop_mask = (
                torch.rand(shape_ids.shape[0], device=device) < float(cfg.diffusion.cfg_drop_prob)
            )
            text_hidden = text_hidden.clone()
            text_hidden[drop_mask] = 0

        bsz, seq_len = shape_ids.shape
        if seq_len % block_size != 0:
            raise RuntimeError(f"Sequence length {seq_len} not divisible by block_size {block_size}")
        num_blocks = seq_len // block_size

        block_indices = torch.randint(0, num_blocks, (bsz,), device=device)
        ratios = schedule.sample_ratio(bsz, device=device)
        noisy_ids, masked = mask_one_block_per_sample(
            shape_ids=shape_ids,
            block_indices=block_indices,
            mask_ratios=ratios,
            block_size=block_size,
            mask_token_id=model.ensure_mask_token(),
        )

        optimizer.zero_grad(set_to_none=True)

        with _autocast_context(device, amp_dtype):
            cond = _prepare_condition(model, text_hidden, bbox_xyz)
            logits = model(model.encode_token(noisy_ids), cond)[..., :num_codes]

        target = shape_ids[masked]
        masked_logits = logits[masked]
        ce = F.cross_entropy(masked_logits.float(), target, reduction="none")
        weights = schedule.weight_from_ratio(ratios)
        token_weights = (weights[:, None].expand_as(masked))[masked]
        loss = (ce * token_weights).mean()

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        with torch.no_grad():
            pred = masked_logits.argmax(dim=-1)
            acc = (pred == target).float().mean().item()

        running_loss += float(loss.item())
        running_acc += float(acc)
        running_count += 1
        step += 1
        pbar.update(1)

        if step % int(cfg.train.log_every) == 0:
            avg_loss = running_loss / max(running_count, 1)
            avg_acc = running_acc / max(running_count, 1)
            elapsed = time.time() - start_time
            print(
                f"[train] step={step} loss={avg_loss:.4f} "
                f"mask_acc={avg_acc:.4f} elapsed_s={elapsed:.1f}"
            )
            running_loss = 0.0
            running_acc = 0.0
            running_count = 0

        if step % int(cfg.train.eval_every) == 0 and len(val_ds) > 0:
            metrics = evaluate(
                model=model,
                loader=val_loader,
                schedule=schedule,
                num_codes=num_codes,
                block_size=block_size,
                amp_dtype=amp_dtype,
                device=device,
                max_batches=int(cfg.train.val_max_batches),
            )
            print(
                f"[val] step={step} val_loss={metrics['val_loss']:.4f} "
                f"val_mask_acc={metrics['val_mask_acc']:.4f}"
            )

        if step % int(cfg.train.save_every) == 0 or step == num_steps:
            ckpt_path = output_dir / f"block_diffusion_step_{step}.safetensors"
            save_model(model, str(ckpt_path))
            meta = {
                "step": step,
                "num_codes": num_codes,
                "block_size": block_size,
                "shape_mask_id": model.shape_mask_id,
            }
            with open(output_dir / f"block_diffusion_step_{step}.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            print(f"[ckpt] saved {ckpt_path}")

    pbar.close()


if __name__ == "__main__":
    main()
