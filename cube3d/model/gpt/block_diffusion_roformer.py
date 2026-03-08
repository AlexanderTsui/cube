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
        self._mask_cache: dict[tuple[int, int, int, str, torch.device], torch.Tensor] = {}
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

    @staticmethod
    def _shape_block_ids(length: int, block_size: int, device: torch.device) -> torch.Tensor:
        if length % block_size != 0:
            raise ValueError(f"length={length} must be divisible by block_size={block_size}")
        return torch.arange(length, device=device, dtype=torch.long) // block_size

    @classmethod
    def _build_block_causal_shape_mask(
        cls,
        length: int,
        block_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        block_q = cls._shape_block_ids(length, block_size, device).view(-1, 1)
        block_k = cls._shape_block_ids(length, block_size, device).view(1, -1)
        return block_q >= block_k

    @classmethod
    def _build_bd_training_shape_mask(
        cls,
        length: int,
        block_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        # Official BD3-LM training mask on [x_t | x_0]:
        # 1) block-diagonal for x_t<->x_t and x_0<->x_0 in the same block
        # 2) offset block-causal from x_t queries to previous x_0 blocks
        # 3) block-causal from x_0 queries to previous/current x_0 blocks
        n = length
        idx = torch.arange(2 * n, device=device, dtype=torch.long)
        q = idx.view(-1, 1)
        k = idx.view(1, -1)

        q_is_x0 = q >= n
        k_is_x0 = k >= n

        block_q = torch.where(q_is_x0, (q - n) // block_size, q // block_size)
        block_k = torch.where(k_is_x0, (k - n) // block_size, k // block_size)

        block_diagonal = (block_q == block_k) & (q_is_x0 == k_is_x0)
        offset_block_causal = (block_q > block_k) & (~q_is_x0) & k_is_x0
        block_causal = (block_q >= block_k) & q_is_x0 & k_is_x0
        return block_diagonal | offset_block_causal | block_causal

    @staticmethod
    def _compose_cond_shape_mask(
        cond_len: int,
        shape_mask: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        # Full mask on [cond | shape]:
        # - cond tokens attend causal within cond only
        # - shape tokens attend to all cond tokens
        # - shape<->shape attention follows supplied shape mask
        shape_len = int(shape_mask.shape[0])
        full = torch.zeros(
            cond_len + shape_len,
            cond_len + shape_len,
            dtype=torch.bool,
            device=device,
        )
        cond_idx = torch.arange(cond_len, device=device)
        full[:cond_len, :cond_len] = cond_idx.view(-1, 1) >= cond_idx.view(1, -1)
        full[cond_len:, :cond_len] = True
        full[cond_len:, cond_len:] = shape_mask
        return full

    def _get_or_build_mask(
        self,
        *,
        cond_len: int,
        shape_len: int,
        block_size: int,
        mode: str,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (cond_len, shape_len, block_size, mode, device)
        full_mask = self._mask_cache.get(key)
        if full_mask is not None:
            shape_mask = full_mask[cond_len:, cond_len:]
            return full_mask, shape_mask

        if mode == "block_causal":
            shape_mask = self._build_block_causal_shape_mask(
                length=shape_len,
                block_size=block_size,
                device=device,
            )
        elif mode == "bd_training":
            if shape_len % 2 != 0:
                raise ValueError("bd_training mode expects shape_len to be even (x_t || x_0)")
            shape_mask = self._build_bd_training_shape_mask(
                length=shape_len // 2,
                block_size=block_size,
                device=device,
            )
        else:
            raise ValueError(f"Unsupported mask mode: {mode}")

        full_mask = self._compose_cond_shape_mask(
            cond_len=cond_len,
            shape_mask=shape_mask,
            device=device,
        )
        self._mask_cache[key] = full_mask
        return full_mask, shape_mask

    def forward(
        self,
        embed: torch.Tensor,
        cond: torch.Tensor,
        kv_cache=None,
        curr_pos_id: Optional[torch.Tensor] = None,
        decode: bool = False,
        attention_mode: str = "causal",
        block_size: Optional[int] = None,
    ):
        if attention_mode == "causal":
            return super().forward(
                embed=embed,
                cond=cond,
                kv_cache=kv_cache,
                curr_pos_id=curr_pos_id,
                decode=decode,
            )

        if decode:
            raise ValueError(f"{attention_mode} mode does not support decode=True in this implementation")
        if block_size is None:
            raise ValueError(f"{attention_mode} mode requires block_size")

        cond_len = int(cond.shape[1])
        shape_len = int(embed.shape[1])
        full_mask, shape_mask = self._get_or_build_mask(
            cond_len=cond_len,
            shape_len=shape_len,
            block_size=int(block_size),
            mode=attention_mode,
            device=embed.device,
        )
        return super().forward(
            embed=embed,
            cond=cond,
            kv_cache=kv_cache,
            curr_pos_id=curr_pos_id,
            decode=decode,
            dual_attn_mask=full_mask,
            single_attn_mask=shape_mask,
            dual_is_causal=False,
            single_is_causal=False,
        )
