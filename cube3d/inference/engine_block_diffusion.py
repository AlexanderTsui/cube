from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast

from cube3d.inference.utils import load_config, load_model_weights, parse_structured
from cube3d.model.autoencoder.one_d_autoencoder import OneDAutoEncoder
from cube3d.model.gpt.block_diffusion_roformer import BlockDiffusionRoformer


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
        self.device = device
        self.block_size = block_size
        self.num_denoise_steps = num_denoise_steps

        gpt_cfg = parse_structured(BlockDiffusionRoformer.Config, self.cfg.gpt_model)
        self.gpt_model = BlockDiffusionRoformer(gpt_cfg)
        try:
            # Case 1: AR checkpoint (no mask token yet)
            load_model_weights(self.gpt_model, gpt_ckpt_path)
            self.gpt_model.ensure_mask_token()
        except Exception:
            # Case 2: Block diffusion checkpoint already contains mask token row.
            gpt_cfg = parse_structured(BlockDiffusionRoformer.Config, self.cfg.gpt_model)
            gpt_cfg.add_mask_token = True
            self.gpt_model = BlockDiffusionRoformer(gpt_cfg)
            load_model_weights(self.gpt_model, gpt_ckpt_path)
            self.gpt_model.ensure_mask_token()

        self.gpt_model = self.gpt_model.eval().to(self.device)

        self.shape_model = OneDAutoEncoder(
            parse_structured(OneDAutoEncoder.Config, self.cfg.shape_model)
        )
        load_model_weights(self.shape_model, shape_ckpt_path)
        self.shape_model = self.shape_model.eval().to(self.device)

        self.text_model = CLIPTextModelWithProjection.from_pretrained(
            self.cfg.text_model_pretrained_model_name_or_path,
            force_download=False,
            device_map=self.device,
        ).eval()
        self.text_tokenizer = CLIPTokenizerFast.from_pretrained(
            self.cfg.text_model_pretrained_model_name_or_path
        )

        self.max_new_tokens = int(self.shape_model.cfg.num_encoder_latents)
        self.min_id = 0
        self.max_id = int(self.shape_model.cfg.num_codes)
        self.mask_id = int(self.gpt_model.ensure_mask_token())
        self.first_hitting = bool(
            getattr(getattr(self.cfg, "diffusion", {}), "first_hitting", False)
        )

    def _autocast_context(self):
        return torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type in {"cuda", "mps"},
        )

    @torch.inference_mode()
    def run_clip(self, text_inputs):
        tokenized = self.text_tokenizer(
            text_inputs,
            max_length=self.text_tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
        with torch.autocast(device_type=self.device.type, enabled=False):
            encoded = self.text_model(**tokenized)
        hidden = encoded.last_hidden_state
        return self.gpt_model.encode_text(hidden)

    @torch.inference_mode()
    def prepare_conditions_with_bbox(
        self,
        cond: torch.Tensor,
        bounding_box_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not hasattr(self.gpt_model, "bbox_proj"):
            return cond
        if bounding_box_tensor is None:
            bounding_box_tensor = torch.zeros(
                (cond.shape[0], 3), dtype=cond.dtype, device=self.device
            )
        bbox_emb = self.gpt_model.bbox_proj(bounding_box_tensor).unsqueeze(1)
        return torch.cat([cond, bbox_emb], dim=1)

    @torch.inference_mode()
    def prepare_inputs(
        self,
        prompts: list[str],
        bounding_box_xyz: Optional[Tuple[float]] = None,
    ) -> torch.Tensor:
        prompt_embeds = self.run_clip(prompts)
        if bounding_box_xyz is not None:
            cond_bbox = torch.atleast_2d(torch.tensor(bounding_box_xyz)).to(self.device)
        else:
            cond_bbox = None
        return self.prepare_conditions_with_bbox(prompt_embeds, cond_bbox)

    @torch.inference_mode()
    def _predict_log_probs(self, shape_ids: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        with self._autocast_context():
            embed = self.gpt_model.encode_token(shape_ids)
            logits = self.gpt_model(
                embed,
                cond,
                attention_mode="block_causal",
                block_size=self.block_size,
            )
        code_logits = logits[..., self.min_id : self.max_id]
        mask_logit = logits[..., self.mask_id : self.mask_id + 1]
        restricted = torch.cat([code_logits, mask_logit], dim=-1)
        log_probs = F.log_softmax(restricted.float(), dim=-1)

        local_mask_idx = self.max_id
        local_shape_ids = shape_ids.clone()
        local_shape_ids[local_shape_ids == self.mask_id] = local_mask_idx
        unmasked = local_shape_ids != local_mask_idx
        log_probs[unmasked] = -1.0e6
        log_probs[unmasked, local_shape_ids[unmasked]] = 0.0
        return log_probs

    @torch.inference_mode()
    def run_gpt(
        self,
        prompts: list[str],
        top_p: Optional[float] = None,
        bounding_box_xyz: Optional[Tuple[float]] = None,
    ) -> torch.Tensor:
        cond = self.prepare_inputs(prompts, bounding_box_xyz)
        bsz = len(prompts)
        seq_len = self.max_new_tokens
        assert seq_len % self.block_size == 0, "seq_len must be divisible by block_size"
        num_blocks = seq_len // self.block_size

        shape_ids = torch.full(
            (bsz, seq_len),
            fill_value=self.mask_id,
            dtype=torch.long,
            device=self.device,
        )

        dt = 1.0 / max(self.num_denoise_steps, 1)
        ones = torch.ones((bsz, 1), device=self.device, dtype=torch.float32)
        local_mask_idx = self.max_id

        for block_idx in tqdm(range(num_blocks), desc="block diffusion"):
            start = block_idx * self.block_size
            end = start + self.block_size

            block_ids = shape_ids[:, start:end]
            if self.mask_id not in block_ids:
                continue

            t_scalar = 1.0
            for denoise_step in range(self.num_denoise_steps):
                block_mask = shape_ids[:, start:end].eq(self.mask_id)
                if not block_mask.any():
                    break
                if self.first_hitting:
                    num_masked = block_mask.sum(dim=-1).clamp(min=1).float()
                    u = torch.rand((bsz,), device=self.device).clamp_(1.0e-6, 1.0)
                    t_vec = (t_scalar * torch.pow(u, 1.0 / num_masked)).view(-1, 1)
                    t_scalar = float(t_vec.mean().item())
                else:
                    t_scalar = max(1.0 - denoise_step * dt, 1.0e-3)
                    t_vec = t_scalar * ones
                s_vec = torch.clamp(t_vec - dt, min=1.0e-6)
                mask_prob = torch.clamp(s_vec / torch.clamp(t_vec, min=1.0e-6), 0.0, 1.0)

                probs = self._predict_log_probs(shape_ids, cond).exp()
                probs = _nucleus_on_probs(probs, top_p=top_p)
                p_block = probs[:, start:end, :]

                if self.first_hitting:
                    x_block = _sample_categorical(p_block)
                    for i in range(bsz):
                        mask_pos = torch.nonzero(block_mask[i], as_tuple=False).squeeze(-1)
                        if mask_pos.numel() == 0:
                            continue
                        chosen = mask_pos[torch.randint(0, mask_pos.numel(), (1,), device=self.device)]
                        keep = shape_ids[i, start:end].clone()
                        keep[chosen] = x_block[i, chosen]
                        x_block[i] = keep
                else:
                    q_xs = p_block * (1.0 - mask_prob[:, None, :])
                    q_xs[:, :, local_mask_idx] = mask_prob.squeeze(-1).unsqueeze(-1)
                    x_block = _sample_categorical(q_xs)

                x_block_global = x_block.clone()
                x_block_global[x_block_global == local_mask_idx] = self.mask_id
                copy_flag = (~block_mask).to(x_block_global.dtype)
                x_block_global = copy_flag * shape_ids[:, start:end] + (1 - copy_flag) * x_block_global
                shape_ids[:, start:end] = x_block_global

            block_mask = shape_ids[:, start:end].eq(self.mask_id)
            if block_mask.any():
                probs = self._predict_log_probs(shape_ids, cond).exp()[:, start:end, :]
                probs[:, :, local_mask_idx] = 0.0
                probs = probs / torch.clamp(probs.sum(dim=-1, keepdim=True), min=1.0e-12)
                x_block = _sample_categorical(probs)
                x_block_global = x_block.clone()
                x_block_global[x_block_global == local_mask_idx] = self.mask_id
                for i in range(bsz):
                    mask_pos = torch.nonzero(block_mask[i], as_tuple=False).squeeze(-1)
                    if mask_pos.numel() > 0:
                        shape_ids[i, start + mask_pos] = x_block_global[i, mask_pos]

        return shape_ids

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
        use_kv_cache: bool = False,  # kept for compatibility with Engine API
        resolution_base: float = 8.0,
        chunk_size: int = 100_000,
        top_p: float = None,
        bounding_box_xyz: Optional[Tuple[float]] = None,
    ):
        output_ids = self.run_gpt(prompts, top_p=top_p, bounding_box_xyz=bounding_box_xyz)
        with self._autocast_context():
            mesh_v_f = self.run_shape_decode(output_ids, resolution_base, chunk_size)
        return mesh_v_f
