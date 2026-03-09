from dataclasses import dataclass
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from flash_attn import flash_attn_func

    FLASH_ATTN_AVAILABLE = True
except Exception:
    flash_attn_func = None
    FLASH_ATTN_AVAILABLE = False


def block_diff_mask(
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    block_size: int,
    n: int,
) -> torch.Tensor:
    """Official BD3-LM training mask on [x_t | x_0]."""
    x0_flag_q = q_idx >= n
    x0_flag_kv = kv_idx >= n

    block_q = torch.where(x0_flag_q, (q_idx - n) // block_size, q_idx // block_size)
    block_kv = torch.where(x0_flag_kv, (kv_idx - n) // block_size, kv_idx // block_size)

    block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)
    offset_block_causal = (block_q > block_kv) & x0_flag_kv & (~x0_flag_q)
    block_causal = (block_q >= block_kv) & x0_flag_q & x0_flag_kv
    return block_diagonal | offset_block_causal | block_causal


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class LayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            y = F.layer_norm(x.float(), [self.dim])
        return y * self.weight[None, None, :]


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor,
        dim: int,
        max_period: int = 10000,
    ) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(float(max_period), device=t.device))
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class Rotary:
    def __init__(self, head_dim: int, base: float = 10000.0):
        if head_dim % 2 != 0:
            raise ValueError(f"Rotary head_dim must be even, got {head_dim}")
        self.head_dim = int(head_dim)
        self.base = float(base)
        self._cache: dict[tuple[int, torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}

    def get(
        self,
        seqlen: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (int(seqlen), device, dtype)
        if key in self._cache:
            return self._cache[key]
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32) / self.head_dim)
        )
        pos = torch.arange(seqlen, device=device, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1).to(dtype=dtype)
        cos = emb.cos()[None, :, None, :]
        sin = emb.sin()[None, :, None, :]
        self._cache[key] = (cos, sin)
        return cos, sin


class DDiTFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()
        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x: torch.Tensor, cond_global: torch.Tensor) -> torch.Tensor:
        x = self.norm_final(x)
        shift, scale = self.adaLN_modulation(cond_global)[:, None].chunk(2, dim=-1)
        x = modulate(x, shift, scale)
        return self.linear(x)


class DDiTBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        n_heads: int,
        dropout: float,
        cond_dim: int,
        cross_attn_dropout: float,
        attn_backend: str,
    ) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by n_heads={n_heads}")
        self.dim = int(dim)
        self.n_heads = int(n_heads)
        self.head_dim = self.dim // self.n_heads
        self.self_attn_dropout = float(dropout)
        self.cross_attn_dropout = float(cross_attn_dropout)
        self.attn_backend = str(attn_backend)

        self.norm1 = LayerNorm(self.dim)
        self.attn_qkv = nn.Linear(self.dim, 3 * self.dim, bias=False)
        self.attn_out = nn.Linear(self.dim, self.dim, bias=False)

        self.norm_cross = LayerNorm(self.dim)
        self.cross_q = nn.Linear(self.dim, self.dim, bias=False)
        self.cross_k = nn.Linear(self.dim, self.dim, bias=False)
        self.cross_v = nn.Linear(self.dim, self.dim, bias=False)
        self.cross_out = nn.Linear(self.dim, self.dim, bias=False)

        self.norm2 = LayerNorm(self.dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, 4 * self.dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * self.dim, self.dim, bias=True),
        )

        # (shift, scale, gate) for self-attn / cross-attn / mlp
        self.adaLN_modulation = nn.Linear(cond_dim, 9 * self.dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

        self.key_cache: Optional[torch.Tensor] = None
        self.value_cache: Optional[torch.Tensor] = None
        self.cache_idx: int = 0
        self.cache_max_seqlen: int = 0

    def set_attn_backend(self, backend: str) -> None:
        self.attn_backend = str(backend)

    def reset_kv_cache(
        self,
        *,
        batch_size: int,
        max_seqlen: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.cache_max_seqlen = int(max_seqlen)
        self.key_cache = torch.zeros(
            (batch_size, self.cache_max_seqlen, self.n_heads, self.head_dim),
            device=device,
            dtype=dtype,
        )
        self.value_cache = torch.zeros(
            (batch_size, self.cache_max_seqlen, self.n_heads, self.head_dim),
            device=device,
            dtype=dtype,
        )
        self.cache_idx = 0

    def clear_kv_cache(self) -> None:
        self.key_cache = None
        self.value_cache = None
        self.cache_idx = 0
        self.cache_max_seqlen = 0

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)
        return q, k

    def _qkv_with_rope(
        self,
        x: torch.Tensor,
        rotary: Rotary,
        rope_len: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seqlen, _ = x.shape
        qkv = self.attn_qkv(x).view(bsz, seqlen, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        use_len = int(seqlen if rope_len is None else rope_len)
        cos, sin = rotary.get(use_len, x.device, q.dtype)
        q, k = self._apply_rope(q, k, cos, sin)
        return q, k, v

    def _attend(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        dropout_p: float,
        prefer_flash: bool,
    ) -> torch.Tensor:
        if (
            prefer_flash
            and FLASH_ATTN_AVAILABLE
            and attn_mask is None
            and q.is_cuda
            and q.dtype in {torch.float16, torch.bfloat16}
        ):
            out = flash_attn_func(  # type: ignore[misc]
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                dropout_p=(dropout_p if self.training else 0.0),
                causal=False,
            )
            return out

        qh = q.transpose(1, 2)
        kh = k.transpose(1, 2)
        vh = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(
            query=qh,
            key=kh,
            value=vh,
            attn_mask=attn_mask,
            dropout_p=(dropout_p if self.training else 0.0),
            is_causal=False,
        )
        return out.transpose(1, 2)

    def _append_cache(self, k: torch.Tensor, v: torch.Tensor) -> None:
        if self.key_cache is None or self.value_cache is None:
            return
        if k.shape[1] > self.cache_max_seqlen:
            k = k[:, -self.cache_max_seqlen :]
            v = v[:, -self.cache_max_seqlen :]

        need = int(k.shape[1])
        end = self.cache_idx + need
        if end > self.cache_max_seqlen:
            shift = end - self.cache_max_seqlen
            if shift >= self.cache_idx:
                self.cache_idx = 0
            else:
                keep = self.cache_idx - shift
                self.key_cache[:, :keep] = self.key_cache[:, shift : self.cache_idx]
                self.value_cache[:, :keep] = self.value_cache[:, shift : self.cache_idx]
                self.cache_idx = keep
        end = self.cache_idx + need
        self.key_cache[:, self.cache_idx : end].copy_(k)
        self.value_cache[:, self.cache_idx : end].copy_(v)
        self.cache_idx = end

    def forward(
        self,
        x: torch.Tensor,
        rotary: Rotary,
        cond_global: torch.Tensor,
        cond_tokens: torch.Tensor,
        cond_kv_mask: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        bd_training: bool = False,
        sample_mode: bool = False,
        store_kv: bool = False,
    ) -> torch.Tensor:
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_xattn,
            scale_xattn,
            gate_xattn,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(cond_global)[:, None].chunk(9, dim=-1)

        residual = x
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)

        if bd_training and (not sample_mode):
            if x.shape[1] % 2 != 0:
                raise ValueError("bd_training expects [x_t|x_0] with even length")
            n = x.shape[1] // 2
            q1, k1, v1 = self._qkv_with_rope(x_norm[:, :n], rotary, rope_len=n)
            q2, k2, v2 = self._qkv_with_rope(x_norm[:, n:], rotary, rope_len=n)
            q = torch.cat([q1, q2], dim=1)
            k = torch.cat([k1, k2], dim=1)
            v = torch.cat([v1, v2], dim=1)
        else:
            q, k, v = self._qkv_with_rope(x_norm, rotary)

        k_full = k
        v_full = v
        if sample_mode and self.key_cache is not None and self.value_cache is not None:
            cache_len = int(self.cache_idx)
            if cache_len > 0:
                k_full = torch.cat([self.key_cache[:, :cache_len], k], dim=1)
                v_full = torch.cat([self.value_cache[:, :cache_len], v], dim=1)

        prefer_flash = self.attn_backend == "flash_attn"
        self_out = self.attn_out(
            self._attend(
                q=q,
                k=k_full,
                v=v_full,
                attn_mask=attn_mask,
                dropout_p=self.self_attn_dropout,
                prefer_flash=prefer_flash,
            ).contiguous().view(x.shape[0], q.shape[1], self.dim)
        )
        x = residual + gate_msa * self_out

        residual = x
        x_cross = modulate(self.norm_cross(x), shift_xattn, scale_xattn)
        q_cross = self.cross_q(x_cross).view(x.shape[0], x.shape[1], self.n_heads, self.head_dim)
        k_cross = self.cross_k(cond_tokens).view(
            cond_tokens.shape[0], cond_tokens.shape[1], self.n_heads, self.head_dim
        )
        v_cross = self.cross_v(cond_tokens).view(
            cond_tokens.shape[0], cond_tokens.shape[1], self.n_heads, self.head_dim
        )
        cross_mask = cond_kv_mask[:, None, None, :].bool()
        cross_out = self.cross_out(
            self._attend(
                q=q_cross,
                k=k_cross,
                v=v_cross,
                attn_mask=cross_mask,
                dropout_p=self.cross_attn_dropout,
                prefer_flash=False,
            ).contiguous().view(x.shape[0], x.shape[1], self.dim)
        )
        x = residual + gate_xattn * cross_out

        residual = x
        x_mlp = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(x_mlp)
        x = residual + gate_mlp * mlp_out

        if sample_mode and store_kv:
            self._append_cache(k.detach(), v.detach())

        return x


class BlockDiffusionDiT(nn.Module):
    @dataclass
    class Config:
        n_layer: int = 24
        n_head: int = 12
        n_embd: int = 1536
        eps: float = 1.0e-6
        rope_theta: float = 10000.0
        shape_model_vocab_size: int = 16384

        cond_dim: int = 1536
        text_cond_dim: int = 768
        use_bbox: bool = True
        time_conditioning: bool = False

        dropout: float = 0.0
        cross_attn_dropout: float = 0.0
        attn_backend: str = "flash_attn"  # flash_attn | sdpa
        max_seqlen: int = 1024

        add_mask_token: bool = True

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.gradient_checkpointing = False

        backend_req = str(cfg.attn_backend).lower()
        if backend_req not in {"flash_attn", "sdpa"}:
            raise ValueError("attn_backend must be one of {'flash_attn', 'sdpa'}")
        if backend_req == "flash_attn" and not FLASH_ATTN_AVAILABLE:
            self.attn_backend = "sdpa"
            print("[warn] flash_attn not available, fallback to sdpa")
        else:
            self.attn_backend = backend_req

        self.vocab_size = int(cfg.shape_model_vocab_size)
        self.shape_mask_id: Optional[int] = None
        self._mask_cache: dict[tuple[str, int, int, torch.device], torch.Tensor] = {}

        self.wte = nn.Embedding(self.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList(
            [
                DDiTBlock(
                    dim=int(cfg.n_embd),
                    n_heads=int(cfg.n_head),
                    dropout=float(cfg.dropout),
                    cond_dim=int(cfg.cond_dim),
                    cross_attn_dropout=float(cfg.cross_attn_dropout),
                    attn_backend=self.attn_backend,
                )
                for _ in range(int(cfg.n_layer))
            ]
        )

        self.final_layer = DDiTFinalLayer(
            hidden_size=int(cfg.n_embd),
            out_channels=self.vocab_size,
            cond_dim=int(cfg.cond_dim),
        )

        self.rotary = Rotary(head_dim=int(cfg.n_embd // cfg.n_head), base=float(cfg.rope_theta))
        self.sigma_map = TimestepEmbedder(int(cfg.cond_dim))

        self.text_global_proj = nn.Linear(int(cfg.text_cond_dim), int(cfg.cond_dim), bias=True)
        self.text_token_proj = nn.Linear(int(cfg.text_cond_dim), int(cfg.n_embd), bias=True)
        self.bbox_global_proj = (
            nn.Linear(3, int(cfg.cond_dim), bias=True) if bool(cfg.use_bbox) else None
        )
        self.bbox_token_proj = (
            nn.Linear(3, int(cfg.n_embd), bias=True) if bool(cfg.use_bbox) else None
        )
        self.null_cond_token = nn.Parameter(torch.zeros(1, 1, int(cfg.n_embd)))

        if cfg.add_mask_token:
            self.add_mask_token()

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.gradient_checkpointing = bool(enabled)

    def set_attention_backend(self, backend: str) -> None:
        backend = str(backend).lower()
        if backend not in {"flash_attn", "sdpa"}:
            raise ValueError("backend must be one of {'flash_attn', 'sdpa'}")
        if backend == "flash_attn" and not FLASH_ATTN_AVAILABLE:
            backend = "sdpa"
            print("[warn] flash_attn not available, keep sdpa backend")
        self.attn_backend = backend
        for block in self.blocks:
            block.set_attn_backend(backend)

    def reset_kv_cache(
        self,
        *,
        batch_size: int,
        max_seqlen: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if max_seqlen is None:
            max_seqlen = int(self.cfg.max_seqlen)
        if device is None:
            device = self.wte.weight.device
        if dtype is None:
            dtype = self.wte.weight.dtype
        for block in self.blocks:
            block.reset_kv_cache(
                batch_size=int(batch_size),
                max_seqlen=int(max_seqlen),
                device=device,
                dtype=dtype,
            )

    def clear_kv_cache(self) -> None:
        for block in self.blocks:
            block.clear_kv_cache()

    def _has_kv_cache(self) -> bool:
        return len(self.blocks) > 0 and self.blocks[0].key_cache is not None

    def add_mask_token(self) -> int:
        if self.shape_mask_id is not None:
            return self.shape_mask_id

        old_vocab_size = self.vocab_size
        new_vocab_size = old_vocab_size + 1

        old_wte = self.wte
        new_wte = nn.Embedding(new_vocab_size, self.cfg.n_embd).to(
            old_wte.weight.device,
            dtype=old_wte.weight.dtype,
        )
        with torch.no_grad():
            new_wte.weight[:old_vocab_size].copy_(old_wte.weight)
            nn.init.normal_(new_wte.weight[old_vocab_size], std=0.02)
        self.wte = new_wte

        old_linear = self.final_layer.linear
        new_linear = nn.Linear(self.cfg.n_embd, new_vocab_size, bias=True).to(
            old_linear.weight.device,
            dtype=old_linear.weight.dtype,
        )
        with torch.no_grad():
            new_linear.weight[:old_vocab_size].copy_(old_linear.weight)
            if old_linear.bias is not None:
                new_linear.bias[:old_vocab_size].copy_(old_linear.bias)
            nn.init.normal_(new_linear.weight[old_vocab_size], std=0.02)
            nn.init.zeros_(new_linear.bias[old_vocab_size : old_vocab_size + 1])
        self.final_layer.linear = new_linear

        self.vocab_size = new_vocab_size
        self.shape_mask_id = old_vocab_size
        return self.shape_mask_id

    def ensure_mask_token(self) -> int:
        return self.add_mask_token()

    def _process_sigma(
        self,
        sigma: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if sigma is None:
            sigma_1d = torch.zeros((batch_size,), device=device, dtype=torch.float32)
        else:
            sigma = sigma.to(device=device)
            if sigma.ndim == 2:
                sigma_1d = sigma.mean(dim=-1).float()
            elif sigma.ndim == 1:
                sigma_1d = sigma.float()
            else:
                sigma_1d = sigma.view(batch_size, -1).mean(dim=-1).float()
            if sigma_1d.numel() == 1 and batch_size > 1:
                sigma_1d = sigma_1d.expand(batch_size)
        if not bool(self.cfg.time_conditioning):
            sigma_1d = torch.zeros_like(sigma_1d)
        return F.silu(self.sigma_map(sigma_1d))

    @staticmethod
    def _validate_text_inputs(
        text_hidden: Optional[torch.Tensor],
        text_attention_mask: Optional[torch.Tensor],
    ) -> None:
        if text_hidden is None:
            raise ValueError("text_hidden is required")
        if text_attention_mask is None:
            raise ValueError("text_attention_mask is required")
        if text_hidden.ndim != 3:
            raise ValueError(f"text_hidden must be [B,S,D], got {tuple(text_hidden.shape)}")
        if text_attention_mask.ndim != 2:
            raise ValueError(
                f"text_attention_mask must be [B,S], got {tuple(text_attention_mask.shape)}"
            )
        if text_hidden.shape[:2] != text_attention_mask.shape:
            raise ValueError(
                "text_hidden and text_attention_mask mismatch: "
                f"{tuple(text_hidden.shape[:2])} vs {tuple(text_attention_mask.shape)}"
            )

    @staticmethod
    def _prepare_bbox(
        bbox_xyz: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if bbox_xyz is None:
            return torch.zeros((batch_size, 3), device=device, dtype=dtype)
        if bbox_xyz.ndim == 1:
            if bbox_xyz.shape[0] != 3:
                raise ValueError(f"bbox_xyz 1D tensor must have length 3, got {bbox_xyz.shape}")
            bbox_xyz = bbox_xyz.unsqueeze(0)
        if bbox_xyz.ndim != 2 or bbox_xyz.shape[1] != 3:
            raise ValueError(f"bbox_xyz must be [B,3], got shape={tuple(bbox_xyz.shape)}")
        if bbox_xyz.shape[0] == 1 and batch_size > 1:
            bbox_xyz = bbox_xyz.expand(batch_size, -1)
        if bbox_xyz.shape[0] != batch_size:
            raise ValueError(
                f"bbox_xyz batch={bbox_xyz.shape[0]} does not match text batch={batch_size}"
            )
        return bbox_xyz.to(device=device, dtype=dtype)

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.to(dtype=x.dtype).unsqueeze(-1)
        denom = torch.clamp(mask_f.sum(dim=1), min=1.0)
        return (x * mask_f).sum(dim=1) / denom

    def _build_conditions(
        self,
        *,
        sigma: Optional[torch.Tensor],
        text_hidden: torch.Tensor,
        text_attention_mask: torch.Tensor,
        bbox_xyz: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz = int(text_hidden.shape[0])

        sigma_cond = self._process_sigma(
            sigma=sigma,
            batch_size=bsz,
            device=text_hidden.device,
        )

        text_hidden = text_hidden.to(dtype=self.wte.weight.dtype)
        text_attention_mask = text_attention_mask.bool()

        text_global = self.text_global_proj(text_hidden)
        cond_global = self._masked_mean(text_global, text_attention_mask)
        cond_global = cond_global + sigma_cond

        cond_tokens = self.text_token_proj(text_hidden)
        cond_mask = text_attention_mask.clone()

        if self.bbox_global_proj is not None or self.bbox_token_proj is not None:
            bbox = self._prepare_bbox(
                bbox_xyz=bbox_xyz,
                batch_size=bsz,
                device=text_hidden.device,
                dtype=text_hidden.dtype,
            )
            if self.bbox_global_proj is not None:
                cond_global = cond_global + self.bbox_global_proj(bbox)
            if self.bbox_token_proj is not None:
                bbox_token = self.bbox_token_proj(bbox).unsqueeze(1)
                cond_tokens = torch.cat([cond_tokens, bbox_token], dim=1)
                bbox_mask = torch.ones((bsz, 1), device=cond_mask.device, dtype=torch.bool)
                cond_mask = torch.cat([cond_mask, bbox_mask], dim=1)

        # CFG uncond rows can end up with all-false mask. Inject one learned null token.
        no_token = ~cond_mask.any(dim=1)
        if no_token.any():
            cond_tokens = cond_tokens.clone()
            cond_mask = cond_mask.clone()
            cond_tokens[no_token, 0:1, :] = self.null_cond_token.to(
                device=cond_tokens.device,
                dtype=cond_tokens.dtype,
            )
            cond_mask[no_token, 0] = True

        return cond_global, cond_tokens, cond_mask

    @staticmethod
    def _build_block_causal_mask(
        length: int,
        block_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if length % block_size != 0:
            raise ValueError(f"length={length} must be divisible by block_size={block_size}")
        block_ids = torch.arange(length, device=device) // block_size
        return block_ids.view(-1, 1) >= block_ids.view(1, -1)

    @staticmethod
    def _build_bd_training_mask(
        length: int,
        block_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        idx = torch.arange(2 * length, device=device, dtype=torch.long)
        return block_diff_mask(
            q_idx=idx.view(-1, 1),
            kv_idx=idx.view(1, -1),
            block_size=block_size,
            n=length,
        )

    def _get_mask(
        self,
        mode: str,
        seqlen: int,
        block_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        key = (mode, int(seqlen), int(block_size), device)
        if key in self._mask_cache:
            return self._mask_cache[key]

        if mode == "causal":
            idx = torch.arange(seqlen, device=device)
            mask = idx.view(-1, 1) >= idx.view(1, -1)
        elif mode == "block_causal":
            mask = self._build_block_causal_mask(
                length=seqlen,
                block_size=block_size,
                device=device,
            )
        elif mode == "bd_training":
            if seqlen % 2 != 0:
                raise ValueError("bd_training expects [x_t|x_0] with even sequence length")
            mask = self._build_bd_training_mask(
                length=seqlen // 2,
                block_size=block_size,
                device=device,
            )
        else:
            raise ValueError(f"Unsupported attention mode: {mode}")

        self._mask_cache[key] = mask
        return mask

    def forward(
        self,
        input_ids: torch.Tensor,
        sigma: Optional[torch.Tensor] = None,
        attention_mode: str = "causal",
        block_size: Optional[int] = None,
        text_hidden: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        bbox_xyz: Optional[torch.Tensor] = None,
        sample_mode: bool = False,
        store_kv: bool = False,
    ) -> torch.Tensor:
        if input_ids.dtype != torch.long:
            raise ValueError("BlockDiffusionDiT.forward expects token ids (dtype=torch.long)")
        if attention_mode in {"block_causal", "bd_training"} and block_size is None:
            raise ValueError(f"{attention_mode} requires block_size")
        if sample_mode and attention_mode == "bd_training":
            raise ValueError("sample_mode=True is incompatible with attention_mode='bd_training'")

        self._validate_text_inputs(
            text_hidden=text_hidden,
            text_attention_mask=text_attention_mask,
        )
        assert text_hidden is not None and text_attention_mask is not None

        bsz, seqlen = input_ids.shape
        text_hidden = text_hidden.to(device=input_ids.device, dtype=self.wte.weight.dtype)
        text_attention_mask = text_attention_mask.to(device=input_ids.device).bool()

        cond_global, cond_tokens, cond_kv_mask = self._build_conditions(
            sigma=sigma,
            text_hidden=text_hidden,
            text_attention_mask=text_attention_mask,
            bbox_xyz=bbox_xyz,
        )

        x = self.wte(input_ids)
        use_block = 1 if block_size is None else int(block_size)

        attn_mask: Optional[torch.Tensor] = None
        if not (sample_mode and self._has_kv_cache()):
            attn_mask = self._get_mask(
                mode=attention_mode,
                seqlen=seqlen,
                block_size=use_block,
                device=input_ids.device,
            )

        bd_training = attention_mode == "bd_training"
        use_checkpointing = self.gradient_checkpointing and self.training and not sample_mode

        for block in self.blocks:
            if use_checkpointing:
                x = checkpoint(
                    partial(
                        block,
                        rotary=self.rotary,
                        cond_global=cond_global,
                        cond_tokens=cond_tokens,
                        cond_kv_mask=cond_kv_mask,
                        attn_mask=attn_mask,
                        bd_training=bd_training,
                        sample_mode=sample_mode,
                        store_kv=store_kv,
                    ),
                    x,
                    use_reentrant=False,
                )
            else:
                x = block(
                    x=x,
                    rotary=self.rotary,
                    cond_global=cond_global,
                    cond_tokens=cond_tokens,
                    cond_kv_mask=cond_kv_mask,
                    attn_mask=attn_mask,
                    bd_training=bd_training,
                    sample_mode=sample_mode,
                    store_kv=store_kv,
                )

        logits = self.final_layer(x, cond_global)
        if bd_training and not sample_mode:
            logits = logits[:, : (seqlen // 2)]
        return logits
