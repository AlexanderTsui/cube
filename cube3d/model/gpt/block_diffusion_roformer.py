from dataclasses import dataclass, fields
from typing import Optional

import torch
from torch import nn

from cube3d.model.gpt.dual_stream_roformer import DualStreamRoformer


class BlockDiffusionRoformer(DualStreamRoformer):
    @dataclass
    class Config(DualStreamRoformer.Config):
        add_mask_token: bool = False

    def __init__(self, cfg: Config) -> None:
        # DualStreamRoformer.Config does not know add_mask_token.
        base_kwargs = {f.name: getattr(cfg, f.name) for f in fields(DualStreamRoformer.Config)}
        super().__init__(DualStreamRoformer.Config(**base_kwargs))
        self.shape_mask_id: Optional[int] = None
        if cfg.add_mask_token:
            self.add_mask_token()

    def add_mask_token(self, init_from_padding: bool = True) -> int:
        """Append one [MASK] token to the model vocabulary."""
        if self.shape_mask_id is not None:
            return self.shape_mask_id

        old_vocab_size = self.vocab_size
        new_vocab_size = old_vocab_size + 1

        old_wte = self.transformer.wte
        new_wte = nn.Embedding(
            new_vocab_size,
            self.cfg.n_embd,
            padding_idx=self.padding_id,
        ).to(old_wte.weight.device, dtype=old_wte.weight.dtype)
        with torch.no_grad():
            new_wte.weight[:old_vocab_size].copy_(old_wte.weight)
            if init_from_padding:
                new_wte.weight[old_vocab_size].copy_(old_wte.weight[self.padding_id])
            else:
                nn.init.normal_(new_wte.weight[old_vocab_size], std=0.02)
        self.transformer.wte = new_wte

        old_lm_head = self.lm_head
        new_lm_head = nn.Linear(self.cfg.n_embd, new_vocab_size, bias=False).to(
            old_lm_head.weight.device, dtype=old_lm_head.weight.dtype
        )
        with torch.no_grad():
            new_lm_head.weight[:old_vocab_size].copy_(old_lm_head.weight)
            if init_from_padding:
                new_lm_head.weight[old_vocab_size].copy_(
                    old_lm_head.weight[self.padding_id]
                )
            else:
                nn.init.normal_(new_lm_head.weight[old_vocab_size], std=0.02)
        self.lm_head = new_lm_head

        self.vocab_size = new_vocab_size
        self.shape_mask_id = old_vocab_size
        return self.shape_mask_id

    def ensure_mask_token(self) -> int:
        return self.add_mask_token()
