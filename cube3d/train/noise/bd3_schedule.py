from dataclasses import dataclass

import torch


@dataclass
class LogLinearSchedule:
    eps_min: float = 1.0e-3
    eps_max: float = 1.0
    antithetic_sampling: bool = True
    resample: bool = False

    def sample_t(
        self,
        batch_size: int,
        seq_len: int,
        block_size: int,
        device: torch.device,
        eps_min: float | None = None,
        eps_max: float | None = None,
    ) -> torch.Tensor:
        if seq_len % block_size != 0:
            raise ValueError(f"seq_len={seq_len} must be divisible by block_size={block_size}")
        eps_min = self.eps_min if eps_min is None else float(eps_min)
        eps_max = self.eps_max if eps_max is None else float(eps_max)
        if eps_max < eps_min:
            eps_min, eps_max = eps_max, eps_min
        num_blocks = seq_len // block_size
        eps_b = torch.rand((batch_size, num_blocks), device=device)
        if self.antithetic_sampling:
            offset = torch.arange(batch_size * num_blocks, device=device, dtype=torch.float32)
            offset = offset.view(batch_size, num_blocks) / float(batch_size * num_blocks)
            eps_b = (eps_b / float(batch_size * num_blocks) + offset) % 1.0
        t = eps_b * (eps_max - eps_min) + eps_min
        return t.repeat_interleave(block_size, dim=-1)

    @staticmethod
    def loss_scale(t: torch.Tensor, min_t: float = 1.0e-6) -> torch.Tensor:
        t = torch.clamp(t, min=min_t)
        return 1.0 / t

    @staticmethod
    def move_chance(t: torch.Tensor) -> torch.Tensor:
        return torch.clamp(t, min=0.0, max=1.0)


def q_xt(
    x0: torch.Tensor,
    move_chance: torch.Tensor,
    mask_token_id: int,
    block_size: int | None = None,
    resample: bool = False,
    eps_min: float = 1.0e-3,
    eps_max: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x0.shape != move_chance.shape:
        raise ValueError("x0 and move_chance must have the same shape")
    move_indices = torch.rand_like(move_chance) <= move_chance
    if resample:
        if block_size is None:
            raise ValueError("block_size is required when resample=True")
        if x0.shape[1] % block_size != 0:
            raise ValueError("sequence length must be divisible by block_size")
        n_blocks = x0.shape[1] // block_size
        while True:
            ratio = move_indices.view(x0.shape[0], n_blocks, block_size).float().mean(dim=-1)
            bad = (ratio < eps_min) | (ratio > eps_max)
            if not bad.any():
                break
            bad_flat = bad.repeat_interleave(block_size, dim=-1)
            regen = torch.rand_like(move_chance) <= move_chance
            move_indices[bad_flat] = regen[bad_flat]
    xt = torch.where(move_indices, torch.full_like(x0, mask_token_id), x0)
    return xt, move_indices


def restrict_logits_to_codes_and_mask(
    logits: torch.Tensor,
    num_codes: int,
    mask_token_id: int,
) -> tuple[torch.Tensor, int]:
    # Keep only [0..num_codes-1] and [mask_token_id] to match BD3-LM absorbing-state setup.
    code_logits = logits[..., :num_codes]
    mask_logit = logits[..., mask_token_id : mask_token_id + 1]
    restricted = torch.cat([code_logits, mask_logit], dim=-1)
    return restricted, num_codes


def subs_parameterization(
    log_probs: torch.Tensor,
    xt_local: torch.Tensor,
    mask_index_local: int,
    neg_infinity: float = -1.0e6,
) -> torch.Tensor:
    # Follows the official "subs" parameterization: unmasked positions become one-hot at x_t.
    if log_probs.shape[:-1] != xt_local.shape:
        raise ValueError("Shape mismatch between log_probs and xt_local")
    out = log_probs.clone()
    unmasked = xt_local != mask_index_local
    out[unmasked] = neg_infinity
    out[unmasked, xt_local[unmasked]] = 0.0
    return out
