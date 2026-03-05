#!/usr/bin/env python3
"""Build paired text/shape features for Block Diffusion training.

This script consumes Objaverse subset pairs and writes per-asset feature files
containing:
  - CLIP tokenized text + hidden-state text embeddings
  - shape tokenizer discrete indices (shape tokens)
  - normalized bbox xyz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Iterator, Tuple

import numpy as np
import torch
import trimesh
from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast

# Enable running the script directly from the repo without package install.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cube3d.inference.utils import (
    BOUNDING_BOX_MAX_SIZE,
    load_config,
    load_model_weights,
    parse_structured,
    select_device,
)
from cube3d.model.autoencoder.one_d_autoencoder import OneDAutoEncoder

MESH_SCALE = 0.96


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build paired text embedding + shape token dataset for Block Diffusion."
    )
    parser.add_argument(
        "--pairs-jsonl",
        type=Path,
        default=Path("/root/autodl-tmp/objaverse_subset/manifests/pairs.jsonl"),
        help="Input jsonl with uid/glb_path/text.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/root/autodl-tmp/bdcube_dataset"),
        help="Output directory for processed dataset files.",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="cube3d/configs/open_model_v0.5.yaml",
        help="Cube config path (used for text model name and shape model cfg).",
    )
    parser.add_argument(
        "--shape-ckpt-path",
        type=str,
        default="model_weights/shape_tokenizer.safetensors",
        help="Path to shape tokenizer checkpoint (.safetensors).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device for feature extraction.",
    )
    parser.add_argument(
        "--num-surface-samples",
        type=int,
        default=8192,
        help="Number of surface samples for shape tokenizer input.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for processed samples (useful for smoke testing).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild samples even if uid is already in completed_uids.txt.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_line_set(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.add(line)
    return out


def iter_pairs(path: Path) -> Iterator[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            uid = item.get("uid")
            text = item.get("text")
            glb_path = item.get("glb_path")
            if not isinstance(uid, str) or not uid:
                continue
            if not isinstance(text, str):
                text = ""
            if not isinstance(glb_path, str) or not glb_path:
                continue
            item["text"] = text
            item["uid"] = uid
            item["glb_path"] = glb_path
            yield item


def rescale(vertices: np.ndarray, mesh_scale: float = MESH_SCALE) -> np.ndarray:
    bbmin = vertices.min(0)
    bbmax = vertices.max(0)
    center = (bbmin + bbmax) * 0.5
    scale = 2.0 * mesh_scale / max((bbmax - bbmin).max(), 1e-6)
    return (vertices - center) * scale


def load_clean_mesh(mesh_path: str) -> trimesh.Trimesh:
    loaded = trimesh.load(mesh_path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        meshes = []
        for geom in loaded.geometry.values():
            if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) and len(geom.faces):
                meshes.append(geom)
        if not meshes:
            raise ValueError("no mesh geometry in scene")
        mesh = trimesh.util.concatenate(meshes)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise ValueError(f"unsupported mesh object type: {type(loaded)}")

    mesh.remove_infinite_values()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("mesh empty after cleaning")
    return mesh


def compute_normalized_bbox_xyz(mesh: trimesh.Trimesh) -> np.ndarray:
    bbox_xyz = (mesh.bounds[1] - mesh.bounds[0]).astype(np.float32)
    bbox_xyz = np.maximum(bbox_xyz, 1e-6)
    return BOUNDING_BOX_MAX_SIZE * bbox_xyz / max(float(np.max(bbox_xyz)), 1e-6)


def build_point_cloud(mesh: trimesh.Trimesh, n_samples: int) -> torch.Tensor:
    positions, face_indices = trimesh.sample.sample_surface(mesh, n_samples)
    normals = mesh.face_normals[face_indices]
    point_cloud = np.concatenate([positions, normals], axis=1)
    return torch.from_numpy(point_cloud.reshape(1, -1, 6)).float()


@torch.inference_mode()
def encode_text(
    tokenizer: CLIPTokenizerFast,
    text_model: CLIPTextModelWithProjection,
    text: str,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tokenized = tokenizer(
        [text],
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokenized["input_ids"].cpu().numpy().astype(np.int32)[0]
    attention_mask = tokenized["attention_mask"].cpu().numpy().astype(np.int8)[0]
    tokenized = {k: v.to(device) for k, v in tokenized.items()}

    with torch.autocast(device_type=device.type, enabled=False):
        outputs = text_model(**tokenized)

    text_hidden = outputs.last_hidden_state.detach().cpu().numpy().astype(np.float16)[0]
    text_pooled = outputs.text_embeds.detach().cpu().numpy().astype(np.float16)[0]
    return input_ids, attention_mask, text_hidden, text_pooled


@torch.inference_mode()
def encode_shape_ids(
    shape_model: OneDAutoEncoder,
    mesh: trimesh.Trimesh,
    n_samples: int,
    device: torch.device,
) -> np.ndarray:
    scaled_mesh = mesh.copy()
    scaled_mesh.vertices = rescale(scaled_mesh.vertices)
    point_cloud = build_point_cloud(scaled_mesh, n_samples).to(device)
    output = shape_model.encode(point_cloud)
    indices = output[3]["indices"]
    return indices.detach().cpu().numpy().astype(np.int32)[0]


def save_progress(progress_path: Path, total_done: int, total_failed: int, start_ts: float) -> None:
    data = {
        "completed": total_done,
        "failed": total_failed,
        "elapsed_sec": round(time.time() - start_ts, 2),
        "timestamp": int(time.time()),
    }
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)


def main() -> None:
    args = parse_args()
    start_ts = time.time()

    if args.device == "auto":
        device = select_device()
    else:
        device = torch.device(args.device)
    print(f"[info] using device: {device}")

    cfg = load_config(args.config_path)
    text_model_name = cfg.text_model_pretrained_model_name_or_path
    print(f"[info] loading text model: {text_model_name}")

    tokenizer = CLIPTokenizerFast.from_pretrained(text_model_name)
    text_model = CLIPTextModelWithProjection.from_pretrained(
        text_model_name,
        force_download=False,
    ).eval().to(device)

    print(f"[info] loading shape tokenizer: {args.shape_ckpt_path}")
    shape_model = OneDAutoEncoder(parse_structured(OneDAutoEncoder.Config, cfg.shape_model))
    load_model_weights(shape_model, args.shape_ckpt_path)
    shape_model = shape_model.eval().to(device)

    output_root = args.output_root
    features_dir = output_root / "features"
    manifests_dir = output_root / "manifests"
    ensure_dir(features_dir)
    ensure_dir(manifests_dir)

    completed_path = manifests_dir / "completed_uids.txt"
    failed_path = manifests_dir / "failed_uids.txt"
    pairs_out_path = manifests_dir / "pairs_bdcube.jsonl"
    progress_path = manifests_dir / "progress.json"

    completed = set() if args.overwrite else load_line_set(completed_path)
    total_done = 0
    total_failed = 0

    print(
        f"[info] start build, existing completed={len(completed)}, "
        f"max_samples={args.max_samples}"
    )

    for item in iter_pairs(args.pairs_jsonl):
        if args.max_samples is not None and total_done >= args.max_samples:
            break

        uid = item["uid"]
        glb_path = Path(item["glb_path"])
        text = item["text"]

        if not args.overwrite and uid in completed:
            continue
        if not glb_path.exists():
            total_failed += 1
            with open(failed_path, "a", encoding="utf-8") as f:
                f.write(f"{uid}\tmissing_glb\t{glb_path}\n")
            continue

        shard = uid[:3] if len(uid) >= 3 else "misc"
        out_dir = features_dir / shard
        ensure_dir(out_dir)
        feature_path = out_dir / f"{uid}.npz"

        try:
            mesh = load_clean_mesh(str(glb_path))
            bbox_xyz = compute_normalized_bbox_xyz(mesh)
            shape_ids = encode_shape_ids(
                shape_model=shape_model,
                mesh=mesh,
                n_samples=args.num_surface_samples,
                device=device,
            )
            input_ids, attention_mask, text_hidden, text_pooled = encode_text(
                tokenizer=tokenizer,
                text_model=text_model,
                text=text,
                device=device,
            )

            np.savez_compressed(
                feature_path,
                shape_ids=shape_ids,
                text_input_ids=input_ids,
                text_attention_mask=attention_mask,
                text_hidden=text_hidden,
                text_pooled=text_pooled,
                bbox_xyz=bbox_xyz.astype(np.float32),
            )

            record = {
                "uid": uid,
                "text": text,
                "glb_path": str(glb_path),
                "feature_path": str(feature_path),
                "bbox_xyz": [round(float(x), 6) for x in bbox_xyz.tolist()],
                "shape_token_len": int(shape_ids.shape[0]),
                "text_seq_len": int(text_hidden.shape[0]),
                "text_hidden_dim": int(text_hidden.shape[1]),
            }
            with open(pairs_out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=True) + "\n")
            with open(completed_path, "a", encoding="utf-8") as f:
                f.write(uid + "\n")
            completed.add(uid)
            total_done += 1

            if total_done % 5 == 0:
                save_progress(progress_path, total_done, total_failed, start_ts)
            print(f"[ok] uid={uid} done={total_done} failed={total_failed}")
        except Exception as exc:  # noqa: BLE001
            total_failed += 1
            err = f"{type(exc).__name__}: {exc}"
            with open(failed_path, "a", encoding="utf-8") as f:
                f.write(f"{uid}\t{err}\t{glb_path}\n")
            print(f"[fail] uid={uid} error={err}")
            tb_path = manifests_dir / "last_error_traceback.txt"
            with open(tb_path, "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())

    save_progress(progress_path, total_done, total_failed, start_ts)
    print(
        f"[done] completed={total_done}, failed={total_failed}, "
        f"output_root={output_root}"
    )


if __name__ == "__main__":
    main()
