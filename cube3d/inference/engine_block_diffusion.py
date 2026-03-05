from math import ceil
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast

from cube3d.inference.utils import load_config, load_model_weights, parse_structured
from cube3d.model.autoencoder.one_d_autoencoder import OneDAutoEncoder
from cube3d.model.gpt.block_diffusion_roformer import BlockDiffusionRoformer


def _sample_from_logits(logits: torch.Tensor, top_p: Optional[float]) -> torch.Tensor:
    """
    Args:
        logits: [N, V]
    Returns:
        sampled token ids [N]
    """
    if top_p is None:
        return torch.argmax(logits, dim=-1)

    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    cumsum = sorted_probs.cumsum(dim=-1)
    to_remove = cumsum > top_p
    to_remove[..., 0] = False
    sorted_probs = sorted_probs.masked_fill(to_remove, 0.0)
    sorted_probs = sorted_probs / torch.clamp(sorted_probs.sum(dim=-1, keepdim=True), min=1e-12)
    chosen = torch.multinomial(sorted_probs, num_samples=1).squeeze(-1)
    return sorted_idx.gather(-1, chosen.unsqueeze(-1)).squeeze(-1)


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
    def _predict_logits(self, shape_ids: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        with self._autocast_context():
            embed = self.gpt_model.encode_token(shape_ids)
            logits = self.gpt_model(embed, cond)
        return logits[..., self.min_id : self.max_id]

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

        for block_idx in tqdm(range(num_blocks), desc="block diffusion"):
            start = block_idx * self.block_size
            end = start + self.block_size

            block_mask = shape_ids[:, start:end].eq(self.mask_id)
            for denoise_step in range(self.num_denoise_steps):
                logits = self._predict_logits(shape_ids, cond)
                block_logits = logits[:, start:end, :]
                probs = F.softmax(block_logits.float(), dim=-1)
                confidence, _ = probs.max(dim=-1)
                candidates = _sample_from_logits(block_logits.reshape(-1, block_logits.shape[-1]), top_p)
                candidates = candidates.view(bsz, self.block_size)

                remaining_counts = block_mask.sum(dim=1)
                steps_left = self.num_denoise_steps - denoise_step
                update_counts = torch.clamp(
                    torch.ceil(remaining_counts.float() / max(steps_left, 1)).long(),
                    min=1,
                )

                for i in range(bsz):
                    if remaining_counts[i] == 0:
                        continue
                    mask_pos = torch.nonzero(block_mask[i], as_tuple=False).squeeze(-1)
                    k = int(min(update_counts[i].item(), mask_pos.numel()))
                    masked_conf = confidence[i, mask_pos]
                    _, top_idx = torch.topk(masked_conf, k=k, largest=True)
                    chosen_rel = mask_pos[top_idx]
                    shape_ids[i, start + chosen_rel] = candidates[i, chosen_rel]
                    block_mask[i, chosen_rel] = False

                if not block_mask.any():
                    break

            if block_mask.any():
                # Safety fallback to ensure no [MASK] remains in the block.
                logits = self._predict_logits(shape_ids, cond)[:, start:end, :]
                fill = torch.argmax(logits, dim=-1)
                for i in range(bsz):
                    mask_pos = torch.nonzero(block_mask[i], as_tuple=False).squeeze(-1)
                    if mask_pos.numel() > 0:
                        shape_ids[i, start + mask_pos] = fill[i, mask_pos]

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
