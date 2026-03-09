import logging
from typing import Any, Optional, Tuple

import torch
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file, load_model

BOUNDING_BOX_MAX_SIZE = 1.925


def normalize_bbox(bounding_box_xyz: Tuple[float]):
    max_l = max(bounding_box_xyz)
    return [BOUNDING_BOX_MAX_SIZE * elem / max_l for elem in bounding_box_xyz]


def load_config(cfg_path: str) -> Any:
    """
    Load and resolve a configuration file.
    Args:
        cfg_path (str): The path to the configuration file.
    Returns:
        Any: The loaded and resolved configuration object.
    Raises:
        AssertionError: If the loaded configuration is not an instance of DictConfig.
    """

    cfg = OmegaConf.load(cfg_path)
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    return cfg


def parse_structured(cfg_type: Any, cfg: DictConfig) -> Any:
    """
    Parses a configuration dictionary into a structured configuration object.
    Args:
        cfg_type (Any): The type of the structured configuration object.
        cfg (DictConfig): The configuration dictionary to be parsed.
    Returns:
        Any: The structured configuration object created from the dictionary.
    """

    scfg = OmegaConf.structured(cfg_type(**cfg))
    return scfg


def load_model_weights(model: torch.nn.Module, ckpt_path: str) -> None:
    """
    Load a safetensors checkpoint into a PyTorch model.
    The model is updated in place.

    Args:
        model: PyTorch model to load weights into
        ckpt_path: Path to the safetensors checkpoint file

    Returns:
        None
    """
    assert ckpt_path.endswith(
        ".safetensors"
    ), f"Checkpoint path '{ckpt_path}' is not a safetensors file"

    load_model(model, ckpt_path)


def load_model_weights_flexible(
    model: torch.nn.Module,
    ckpt_path: str,
    *,
    is_main: bool = True,
    max_report: int = 6,
) -> dict[str, int]:
    """
    Flexible safetensors loader for architecture evolution.

    Rules:
    - Exact-shape parameters are loaded directly.
    - Prefix-compatible tensors (same trailing dims, smaller first dim) are
      copied into the beginning of target tensor.
    - Others are skipped.
    """
    assert ckpt_path.endswith(
        ".safetensors"
    ), f"Checkpoint path '{ckpt_path}' is not a safetensors file"

    ckpt_state = load_file(ckpt_path)
    model_state = model.state_dict()
    patched_state: dict[str, torch.Tensor] = {}
    partial_keys: list[str] = []
    skipped_keys: list[str] = []

    for key, value in ckpt_state.items():
        if key not in model_state:
            skipped_keys.append(key)
            continue
        target = model_state[key]
        if value.shape == target.shape:
            patched_state[key] = value.to(device=target.device, dtype=target.dtype)
            continue
        if (
            value.ndim == target.ndim
            and value.shape[1:] == target.shape[1:]
            and value.shape[0] <= target.shape[0]
        ):
            patched = target.clone()
            patched[: value.shape[0]].copy_(
                value.to(device=target.device, dtype=target.dtype)
            )
            patched_state[key] = patched
            partial_keys.append(key)
            continue
        skipped_keys.append(key)

    missing_keys, unexpected_keys = model.load_state_dict(patched_state, strict=False)

    if is_main:
        print(
            "[info] flexible load: "
            f"loaded={len(patched_state)} partial={len(partial_keys)} "
            f"missing={len(missing_keys)} unexpected={len(unexpected_keys)} "
            f"skipped={len(skipped_keys)}"
        )
        if len(partial_keys) > 0:
            print(f"[info] partial keys (first {max_report}): {partial_keys[:max_report]}")
        if len(missing_keys) > 0:
            print(f"[info] missing keys (first {max_report}): {missing_keys[:max_report]}")
        if len(unexpected_keys) > 0:
            print(
                f"[info] unexpected keys (first {max_report}): "
                f"{unexpected_keys[:max_report]}"
            )
        if len(skipped_keys) > 0:
            print(f"[info] skipped keys (first {max_report}): {skipped_keys[:max_report]}")

    return {
        "loaded": len(patched_state),
        "partial": len(partial_keys),
        "missing": len(missing_keys),
        "unexpected": len(unexpected_keys),
        "skipped": len(skipped_keys),
    }


def select_device() -> Any:
    """
    Selects the appropriate PyTorch device for tensor allocation.

    Returns:
        Any: The `torch.device` object.
    """
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
