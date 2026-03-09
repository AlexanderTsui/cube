from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast

from cube3d.inference.utils import (
    load_config,
    load_model_weights,
    parse_structured,
)
from cube3d.model.autoencoder.one_d_autoencoder import OneDAutoEncoder
from cube3d.model.gpt.block_diffusion_dit import BlockDiffusionDiT


def _sample_categorical(probs: torch.Tensor) -> torch.Tensor:
    gumbel = 1.0e-10 - (torch.rand_like(probs) + 1.0e-10).log()
    return (probs / gumbel).argmax(dim=-1)


def _nucleus_on_probs(probs: torch.Tensor, top_p: Optional[float]) -> torch.Tensor:
    if top_p is None or top_p >= 1.0:
        return probs
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    cumsum = sorted_probs.cumsum(dim=-1)
    nucleus_mask = cumsum <= top_p
    nucleus_mask[..., 0] = True
    sorted_probs = sorted_probs * nucleus_mask
    filtered = torch.zeros_like(probs)
    filtered.scatter_(-1, sorted_idx, sorted_probs)
    return filtered / torch.clamp(filtered.sum(dim=-1, keepdim=True), min=1.0e-12)


class EngineBlockDiffusion:
    def __init__(
        self,
        config_path: str,
        gpt_ckpt_path: str,
        shape_ckpt_path: str,
        device: torch.device,
        block_size: int = 32,
        num_denoise_steps: int = 8,
    ):
        self.cfg = load_config(config_path)
        self.base_cfg = self._resolve_base_config(self.cfg)
        self.device = device

        # Constructor arguments take precedence for CLI compatibility.
        self.block_size = int(block_size)
        self.num_denoise_steps = int(num_denoise_steps)
        self.first_hitting = bool(
            OmegaConf.select(self.cfg, "diffusion.first_hitting", default=True)
        )
        self.kv_cache = bool(
            OmegaConf.select(self.cfg, "diffusion.kv_cache", default=False)
        )
        self.context_size = int(
            OmegaConf.select(
                self.cfg,
                "diffusion.context_size",
                default=int(self.base_cfg.shape_model.num_encoder_latents),
            )
        )
        self.default_cfg_scale = float(
            OmegaConf.select(self.cfg, "diffusion.cfg_scale", default=1.0)
        )
        self.default_top_p = float(
            OmegaConf.select(self.cfg, "diffusion.nucleus_p", default=1.0)
        )

        dit_cfg = self._build_model_cfg(self.cfg, self.base_cfg)
        self.gpt_model = BlockDiffusionDiT(dit_cfg)
        load_model_weights(self.gpt_model, gpt_ckpt_path)
        self.gpt_model.ensure_mask_token()
        self.gpt_model = self.gpt_model.eval().to(self.device)

        self.shape_model = OneDAutoEncoder(
            parse_structured(OneDAutoEncoder.Config, self.base_cfg.shape_model)
        )
        load_model_weights(self.shape_model, shape_ckpt_path)
        self.shape_model = self.shape_model.eval().to(self.device)

        self.text_model = CLIPTextModelWithProjection.from_pretrained(
            self.base_cfg.text_model_pretrained_model_name_or_path,
            force_download=False,
            device_map=self.device,
        ).eval()
        self.text_tokenizer = CLIPTokenizerFast.from_pretrained(
            self.base_cfg.text_model_pretrained_model_name_or_path
        )

        self.max_new_tokens = int(self.shape_model.cfg.num_encoder_latents)
        self.min_id = 0
        self.max_id = int(self.shape_model.cfg.num_codes)
        self.mask_id = int(self.gpt_model.ensure_mask_token())

    @staticmethod
    def _build_model_cfg(cfg, base_cfg) -> BlockDiffusionDiT.Config:
        model_cfg = OmegaConf.select(cfg, "diffusion.model", default={})
        default_layers = int(base_cfg.gpt_model.n_layer + base_cfg.gpt_model.n_single_layer)

        def _pick(path: str, default):
            value = OmegaConf.select(model_cfg, path, default=None)
            return default if value is None else value

        return BlockDiffusionDiT.Config(
            n_layer=int(_pick("n_layer", default_layers)),
            n_head=int(_pick("n_head", int(base_cfg.gpt_model.n_head))),
            n_embd=int(_pick("n_embd", int(base_cfg.gpt_model.n_embd))),
            eps=float(_pick("eps", float(base_cfg.gpt_model.eps))),
            rope_theta=float(
                _pick("rope_theta", float(base_cfg.gpt_model.rope_theta))
            ),
            shape_model_vocab_size=int(base_cfg.shape_model.num_codes),
            cond_dim=int(_pick("cond_dim", int(base_cfg.gpt_model.n_embd))),
            text_cond_dim=int(_pick("text_cond_dim", int(base_cfg.gpt_model.text_model_embed_dim))),
            use_bbox=bool(_pick("use_bbox", bool(base_cfg.gpt_model.use_bbox))),
            time_conditioning=bool(_pick("time_conditioning", False)),
            dropout=float(_pick("dropout", 0.0)),
            cross_attn_dropout=float(_pick("cross_attn_dropout", 0.0)),
            attn_backend=str(_pick("attn_backend", "flash_attn")),
            max_seqlen=int(
                _pick("max_seqlen", int(base_cfg.shape_model.num_encoder_latents))
            ),
            add_mask_token=bool(_pick("add_mask_token", True)),
        )

    @staticmethod
    def _resolve_base_config(cfg):
        base_path = OmegaConf.select(cfg, "model.base_config_path", default=None)
        if base_path is None:
            return cfg
        return load_config(str(base_path))

    def _autocast_context(self):
        return torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type in {"cuda", "mps"},
        )

    @torch.inference_mode()
    def _encode_text_hidden(self, text_inputs: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        tokenized = self.text_tokenizer(
            text_inputs,
            max_length=self.text_tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        attention_mask = tokenized["attention_mask"].to(self.device).bool()
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
        with torch.autocast(device_type=self.device.type, enabled=False):
            encoded = self.text_model(**tokenized)
        return encoded.last_hidden_state, attention_mask

    def _prepare_bbox_tensor(
        self,
        batch_size: int,
        bounding_box_xyz: Optional[Tuple[float]],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if bounding_box_xyz is None:
            return torch.zeros((batch_size, 3), device=self.device, dtype=dtype)
        bbox = torch.as_tensor(bounding_box_xyz, dtype=dtype, device=self.device)
        if bbox.ndim == 1:
            bbox = bbox.unsqueeze(0)
        if bbox.shape[0] == 1 and batch_size > 1:
            bbox = bbox.expand(batch_size, -1)
        if bbox.shape != (batch_size, 3):
            raise ValueError(
                f"bounding_box_xyz must be broadcastable to [{batch_size},3], got shape={tuple(bbox.shape)}"
            )
        return bbox

    @staticmethod
    def _sigma_from_t(t: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
        p = torch.clamp(t, min=eps, max=1.0 - eps)
        return -torch.log1p(-p)

    def _make_unconditional(
        self,
        text_hidden: torch.Tensor,
        text_attention_mask: torch.Tensor,
        bbox_xyz: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.zeros_like(text_hidden),
            torch.zeros_like(text_attention_mask, dtype=torch.bool),
            torch.zeros_like(bbox_xyz),
        )

    def _restrict_logits(self, logits: torch.Tensor) -> torch.Tensor:
        code_logits = logits[..., self.min_id : self.max_id]
        mask_logit = logits[..., self.mask_id : self.mask_id + 1]
        return torch.cat([code_logits, mask_logit], dim=-1)

    def _subs_log_probs(self, logits: torch.Tensor, shape_ids: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits.float(), dim=-1)
        local_mask_idx = self.max_id
        local_shape_ids = shape_ids.clone()
        local_shape_ids[local_shape_ids == self.mask_id] = local_mask_idx
        unmasked = local_shape_ids != local_mask_idx
        log_probs[unmasked] = -1.0e6
        log_probs[unmasked, local_shape_ids[unmasked]] = 0.0
        return log_probs

    @torch.inference_mode()
    def _model_log_probs(
        self,
        *,
        shape_ids: torch.Tensor,
        t_vec: torch.Tensor,
        text_hidden: torch.Tensor,
        text_attention_mask: torch.Tensor,
        bbox_xyz: torch.Tensor,
        cfg_scale: float,
        use_kv_cache: bool,
    ) -> torch.Tensor:
        sigma = self._sigma_from_t(t_vec)

        with self._autocast_context():
            logits_cond = self.gpt_model(
                shape_ids,
                sigma=sigma,
                attention_mode="block_causal",
                block_size=self.block_size,
                text_hidden=text_hidden,
                text_attention_mask=text_attention_mask,
                bbox_xyz=bbox_xyz,
                sample_mode=use_kv_cache,
                store_kv=False,
            )
            logits_cond = self._restrict_logits(logits_cond)

            if abs(cfg_scale - 1.0) > 1.0e-6:
                uncond_hidden, uncond_mask, uncond_bbox = self._make_unconditional(
                    text_hidden=text_hidden,
                    text_attention_mask=text_attention_mask,
                    bbox_xyz=bbox_xyz,
                )
                logits_uncond = self.gpt_model(
                    shape_ids,
                    sigma=sigma,
                    attention_mode="block_causal",
                    block_size=self.block_size,
                    text_hidden=uncond_hidden,
                    text_attention_mask=uncond_mask,
                    bbox_xyz=uncond_bbox,
                    sample_mode=False,
                    store_kv=False,
                )
                logits_uncond = self._restrict_logits(logits_uncond)
                logits = logits_uncond + cfg_scale * (logits_cond - logits_uncond)
            else:
                logits = logits_cond

        return self._subs_log_probs(logits=logits, shape_ids=shape_ids)

    @torch.inference_mode()
    def _store_kv_block(
        self,
        *,
        block_ids: torch.Tensor,
        t_vec: torch.Tensor,
        text_hidden: torch.Tensor,
        text_attention_mask: torch.Tensor,
        bbox_xyz: torch.Tensor,
    ) -> None:
        sigma = self._sigma_from_t(t_vec)
        with self._autocast_context():
            _ = self.gpt_model(
                block_ids,
                sigma=sigma,
                attention_mode="block_causal",
                block_size=self.block_size,
                text_hidden=text_hidden,
                text_attention_mask=text_attention_mask,
                bbox_xyz=bbox_xyz,
                sample_mode=True,
                store_kv=True,
            )

    @torch.inference_mode()
    def _ddpm_caching_update(
        self,
        *,
        x: torch.Tensor,
        t_vec: torch.Tensor,
        dt: float,
        p_x0: Optional[torch.Tensor],
        text_hidden: torch.Tensor,
        text_attention_mask: torch.Tensor,
        bbox_xyz: torch.Tensor,
        cfg_scale: float,
        top_p: Optional[float],
        use_kv_cache: bool,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        move_t = torch.clamp(t_vec, min=1.0e-6)
        move_s = torch.clamp(t_vec - dt, min=1.0e-6)
        mask_prob = torch.clamp(move_s / move_t, 0.0, 1.0)

        if p_x0 is None:
            if use_kv_cache:
                log_probs = self._model_log_probs(
                    shape_ids=x[:, -self.block_size :],
                    t_vec=t_vec,
                    text_hidden=text_hidden,
                    text_attention_mask=text_attention_mask,
                    bbox_xyz=bbox_xyz,
                    cfg_scale=cfg_scale,
                    use_kv_cache=True,
                )
            else:
                log_probs = self._model_log_probs(
                    shape_ids=x,
                    t_vec=t_vec,
                    text_hidden=text_hidden,
                    text_attention_mask=text_attention_mask,
                    bbox_xyz=bbox_xyz,
                    cfg_scale=cfg_scale,
                    use_kv_cache=False,
                )[:, -self.block_size :]
            p_x0 = _nucleus_on_probs(log_probs.exp(), top_p=top_p)

        local_mask_idx = self.max_id
        x_curr = x[:, -self.block_size :]
        block_mask = x_curr.eq(self.mask_id)

        if self.first_hitting:
            x_block = _sample_categorical(p_x0)
            for i in range(x_block.shape[0]):
                mask_pos = torch.nonzero(block_mask[i], as_tuple=False).squeeze(-1)
                if mask_pos.numel() == 0:
                    continue
                chosen = mask_pos[
                    torch.randint(0, mask_pos.numel(), (1,), device=x_block.device)
                ]
                keep = x_curr[i].clone()
                val = x_block[i, chosen]
                if val == local_mask_idx:
                    val = self.mask_id
                keep[chosen] = val
                x_block[i] = keep
        else:
            q_xs = p_x0 * (1.0 - mask_prob[:, None, :])
            q_xs[:, :, local_mask_idx] = mask_prob.squeeze(-1).unsqueeze(-1)
            x_block = _sample_categorical(q_xs)
            x_block = x_block.long()
            x_block[x_block == local_mask_idx] = self.mask_id

        x_block = torch.where(block_mask, x_block, x_curr)
        x_new = torch.cat((x[:, :-self.block_size], x_block), dim=-1)

        if use_kv_cache and (x_block != self.mask_id).all():
            self._store_kv_block(
                block_ids=x_block,
                t_vec=t_vec,
                text_hidden=text_hidden,
                text_attention_mask=text_attention_mask,
                bbox_xyz=bbox_xyz,
            )

        if torch.equal(x_new, x):
            return p_x0, x_new
        return None, x_new

    @torch.inference_mode()
    def run_gpt(
        self,
        prompts: list[str],
        top_p: Optional[float] = None,
        bounding_box_xyz: Optional[Tuple[float]] = None,
        cfg_scale: Optional[float] = None,
    ) -> torch.Tensor:
        bsz = len(prompts)
        text_hidden, text_attention_mask = self._encode_text_hidden(prompts)
        bbox_xyz = self._prepare_bbox_tensor(
            batch_size=bsz,
            bounding_box_xyz=bounding_box_xyz,
            dtype=text_hidden.dtype,
        )

        if top_p is None:
            top_p = self.default_top_p
        if cfg_scale is None:
            cfg_scale = self.default_cfg_scale

        use_kv_cache = bool(self.kv_cache and abs(cfg_scale - 1.0) <= 1.0e-6)
        if self.kv_cache and not use_kv_cache:
            print("[warn] cfg_scale != 1.0 with kv_cache requested: fallback to no-cache sampling")

        if use_kv_cache:
            self.gpt_model.reset_kv_cache(
                batch_size=bsz,
                max_seqlen=self.context_size,
                device=self.device,
                dtype=self.gpt_model.wte.weight.dtype,
            )
        else:
            self.gpt_model.clear_kv_cache()

        seqlen = self.max_new_tokens
        if seqlen % self.block_size != 0:
            raise RuntimeError(
                f"shape sequence length ({seqlen}) must be divisible by block_size ({self.block_size})"
            )
        num_strides = seqlen // self.block_size

        x_accum = torch.full(
            (bsz, self.block_size),
            fill_value=self.mask_id,
            dtype=torch.long,
            device=self.device,
        )

        dt = 1.0 / max(self.num_denoise_steps, 1)
        ones = torch.ones((bsz, 1), device=self.device, dtype=torch.float32)

        for stride_num in tqdm(range(num_strides), desc="block diffusion"):
            if stride_num > 0:
                x_accum = torch.cat(
                    [
                        x_accum,
                        torch.full(
                            (bsz, self.block_size),
                            fill_value=self.mask_id,
                            dtype=torch.long,
                            device=self.device,
                        ),
                    ],
                    dim=1,
                )

            end_idx = (stride_num + 1) * self.block_size
            start_idx = max(end_idx - self.context_size, 0)
            fwd_idx = torch.arange(start_idx, end_idx, device=self.device)

            p_x0_cache = None
            t_vec = ones.clone()

            for denoise_step in range(self.num_denoise_steps):
                x_view = x_accum[:, fwd_idx]
                if not x_view.eq(self.mask_id).any():
                    break

                if self.first_hitting:
                    num_masked = x_view[:, -self.block_size :].eq(self.mask_id).sum(dim=-1).clamp(min=1).float()
                    u = torch.rand((bsz,), device=self.device).clamp_(1.0e-6, 1.0)
                    t_vec = t_vec * torch.pow(u.unsqueeze(-1), 1.0 / num_masked.unsqueeze(-1))
                else:
                    t_scalar = max(1.0 - denoise_step * dt, 1.0e-3)
                    t_vec = torch.full((bsz, 1), t_scalar, device=self.device, dtype=torch.float32)

                p_x0_cache, x_next = self._ddpm_caching_update(
                    x=x_view,
                    t_vec=t_vec,
                    dt=dt,
                    p_x0=p_x0_cache,
                    text_hidden=text_hidden,
                    text_attention_mask=text_attention_mask,
                    bbox_xyz=bbox_xyz,
                    cfg_scale=cfg_scale,
                    top_p=top_p,
                    use_kv_cache=use_kv_cache,
                )
                x_accum[:, fwd_idx] = x_next

            # last fill to avoid remaining masks
            block_slice = x_accum[:, end_idx - self.block_size : end_idx]
            block_mask = block_slice.eq(self.mask_id)
            if block_mask.any():
                if use_kv_cache:
                    log_probs = self._model_log_probs(
                        shape_ids=x_accum[:, fwd_idx][:, -self.block_size :],
                        t_vec=ones,
                        text_hidden=text_hidden,
                        text_attention_mask=text_attention_mask,
                        bbox_xyz=bbox_xyz,
                        cfg_scale=cfg_scale,
                        use_kv_cache=True,
                    )
                else:
                    log_probs = self._model_log_probs(
                        shape_ids=x_accum[:, fwd_idx],
                        t_vec=ones,
                        text_hidden=text_hidden,
                        text_attention_mask=text_attention_mask,
                        bbox_xyz=bbox_xyz,
                        cfg_scale=cfg_scale,
                        use_kv_cache=False,
                    )[:, -self.block_size :]

                probs = log_probs.exp()
                probs[:, :, self.max_id] = 0.0
                probs = probs / torch.clamp(probs.sum(dim=-1, keepdim=True), min=1.0e-12)
                x_block = _sample_categorical(probs)
                x_block[x_block == self.max_id] = self.mask_id
                x_accum[:, end_idx - self.block_size : end_idx] = torch.where(
                    block_mask,
                    x_block,
                    block_slice,
                )

        self.gpt_model.clear_kv_cache()
        return x_accum[:, :seqlen]

    @torch.inference_mode()
    def run_shape_decode(
        self,
        output_ids: torch.Tensor,
        resolution_base: float = 8.0,
        chunk_size: int = 100_000,
    ):
        shape_ids = (
            output_ids[:, : self.shape_model.cfg.num_encoder_latents, ...]
            .clamp_(0, self.shape_model.cfg.num_codes - 1)
            .view(-1, self.shape_model.cfg.num_encoder_latents)
        )
        latents = self.shape_model.decode_indices(shape_ids)
        mesh_v_f, _ = self.shape_model.extract_geometry(
            latents,
            resolution_base=resolution_base,
            chunk_size=chunk_size,
            use_warp=True,
        )
        return mesh_v_f

    @torch.inference_mode()
    def t2s(
        self,
        prompts: list[str],
        use_kv_cache: bool = False,  # kept for API compatibility; ignored
        resolution_base: float = 8.0,
        chunk_size: int = 100_000,
        top_p: float = None,
        bounding_box_xyz: Optional[Tuple[float]] = None,
    ):
        del use_kv_cache
        output_ids = self.run_gpt(
            prompts,
            top_p=top_p,
            bounding_box_xyz=bounding_box_xyz,
            cfg_scale=self.default_cfg_scale,
        )
        with self._autocast_context():
            mesh_v_f = self.run_shape_decode(output_ids, resolution_base, chunk_size)
        return mesh_v_f
