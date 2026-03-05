import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class Sample:
    uid: str
    feature_path: str
    bbox_xyz: list[float]


def uid_to_split(uid: str, val_ratio: float, seed: int) -> str:
    val_threshold = int(max(0.0, min(1.0, val_ratio)) * 10_000)
    digest = hashlib.md5(f"{uid}:{seed}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 10_000
    return "val" if bucket < val_threshold else "train"


class BlockDiffusionDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        split: str,
        val_ratio: float = 0.02,
        seed: int = 0,
    ) -> None:
        assert split in {"train", "val"}
        self.split = split
        self.samples: list[Sample] = []

        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                uid = d.get("uid")
                feature_path = d.get("feature_path")
                if not isinstance(uid, str) or not isinstance(feature_path, str):
                    continue
                if uid_to_split(uid, val_ratio, seed) != split:
                    continue
                bbox = d.get("bbox_xyz")
                if not isinstance(bbox, list) or len(bbox) != 3:
                    bbox = [1.0, 1.0, 1.0]
                self.samples.append(
                    Sample(uid=uid, feature_path=feature_path, bbox_xyz=bbox)
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        data = np.load(Path(sample.feature_path))

        shape_ids = torch.from_numpy(data["shape_ids"].astype(np.int64))
        text_hidden = torch.from_numpy(data["text_hidden"].astype(np.float32))
        text_attention_mask = torch.from_numpy(
            data["text_attention_mask"].astype(np.bool_)
        )
        bbox_xyz = torch.from_numpy(data["bbox_xyz"].astype(np.float32))

        return {
            "uid": sample.uid,
            "shape_ids": shape_ids,
            "text_hidden": text_hidden,
            "text_attention_mask": text_attention_mask,
            "bbox_xyz": bbox_xyz,
        }
