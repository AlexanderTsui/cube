from dataclasses import dataclass

import torch


@dataclass
class ClippedMaskSchedule:
    beta_low: float = 0.3
    beta_high: float = 0.8

    def sample_ratio(self, batch_size: int, device: torch.device) -> torch.Tensor:
        low = min(self.beta_low, self.beta_high)
        high = max(self.beta_low, self.beta_high)
        return low + (high - low) * torch.rand(batch_size, device=device)

    @staticmethod
    def weight_from_ratio(mask_ratio: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        # Positive proxy of alpha'_t / (1 - alpha_t) used for weighting.
        return 1.0 / torch.clamp(mask_ratio, min=eps)


def mask_one_block_per_sample(
    shape_ids: torch.Tensor,
    block_indices: torch.Tensor,
    mask_ratios: torch.Tensor,
    block_size: int,
    mask_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        shape_ids: [B, L]
        block_indices: [B]
        mask_ratios: [B] in [0,1]
    Returns:
        noisy_ids: [B, L]
        masked_positions: [B, L] bool
    """
    bsz, seq_len = shape_ids.shape
    noisy_ids = shape_ids.clone()
    masked_positions = torch.zeros_like(shape_ids, dtype=torch.bool)
    num_blocks = seq_len // block_size
    assert num_blocks > 0 and seq_len % block_size == 0

    for i in range(bsz):
        block_idx = int(torch.clamp(block_indices[i], 0, num_blocks - 1).item())
        ratio = float(mask_ratios[i].item())
        start = block_idx * block_size
        end = start + block_size

        n_mask = max(1, int(round(block_size * ratio)))
        perm = torch.randperm(block_size, device=shape_ids.device)
        selected = perm[:n_mask] + start
        noisy_ids[i, selected] = mask_token_id
        masked_positions[i, selected] = True

    return noisy_ids, masked_positions
