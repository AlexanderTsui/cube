#!/usr/bin/env python3
import argparse
import gzip
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


BASE_URL = "https://huggingface.co/datasets/allenai/objaverse/resolve/main"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a resumable Objaverse subset with model-text pairs."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/root/autodl-tmp"),
        help="Root directory for downloaded dataset files.",
    )
    parser.add_argument(
        "--target-gb",
        type=float,
        default=38.0,
        help="Target size (GB) for downloaded GLB files.",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=6.0,
        help="Stop early when free space is lower than this threshold.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=8,
        help="Max retries per file download.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def bytes_to_gb(num_bytes: int) -> float:
    return num_bytes / (1024**3)


def log(message: str) -> None:
    print(message, flush=True)


def walk_file_sizes(directory: Path, suffix: str) -> int:
    if not directory.exists():
        return 0
    total = 0
    for root, _, files in os.walk(directory):
        for file_name in files:
            if file_name.endswith(suffix):
                fp = Path(root) / file_name
                try:
                    total += fp.stat().st_size
                except FileNotFoundError:
                    pass
    return total


def download_with_resume(url: str, dest: Path, max_retries: int) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return

    ensure_dir(dest.parent)
    part = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, max_retries + 1):
        try:
            _download_once(url, dest, part)
            return
        except Exception as exc:  # noqa: BLE001
            wait_s = min(60, 2**attempt)
            if attempt == max_retries:
                raise RuntimeError(f"failed to download {url}: {exc}") from exc
            log(
                f"[retry {attempt}/{max_retries}] {url} -> {dest} ({exc}), sleep {wait_s}s"
            )
            time.sleep(wait_s)


def _download_once(url: str, dest: Path, part: Path) -> None:
    offset = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset > 0 else {}
    req = urllib.request.Request(url, headers=headers)
    start = time.time()
    last_report = start
    downloaded = 0

    with urllib.request.urlopen(req, timeout=120) as resp:
        status = getattr(resp, "status", None)
        if offset > 0 and status == 200:
            offset = 0
        mode = "ab" if offset > 0 else "wb"
        with open(part, mode) as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                if now - last_report >= 3:
                    elapsed = max(0.001, now - start)
                    speed_mb_s = downloaded / elapsed / (1024**2)
                    total_mb = (offset + downloaded) / (1024**2)
                    log(
                        f"Downloading {dest.name}: {total_mb:.1f} MB written, {speed_mb_s:.2f} MB/s"
                    )
                    last_report = now

    if not part.exists() or part.stat().st_size == 0:
        raise RuntimeError(f"empty download for {url}")
    part.replace(dest)
    elapsed = max(0.001, time.time() - start)
    speed_mb_s = downloaded / elapsed / (1024**2)
    log(f"Finished {dest.name}: +{downloaded / (1024**2):.1f} MB in {elapsed:.1f}s ({speed_mb_s:.2f} MB/s)")


def load_json_gz(path: Path) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def load_line_set(path: Path, uid_key: str = "") -> set:
    if not path.exists():
        return set()
    out = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if uid_key:
                try:
                    out.add(json.loads(line)[uid_key])
                except Exception:  # noqa: BLE001
                    continue
            else:
                out.add(line)
    return out


def build_text(metadata: Dict[str, Any]) -> str:
    parts = []
    name = metadata.get("name")
    desc = metadata.get("description")
    if isinstance(name, str) and name.strip():
        parts.append(name.strip())
    if isinstance(desc, str) and desc.strip():
        parts.append(desc.strip())

    tags = metadata.get("tags")
    if isinstance(tags, list):
        cleaned = []
        for item in tags:
            if isinstance(item, dict):
                val = item.get("name")
            else:
                val = item
            if isinstance(val, str) and val.strip():
                cleaned.append(val.strip())
        if cleaned:
            parts.append("tags: " + ", ".join(cleaned[:20]))

    cats = metadata.get("categories")
    if isinstance(cats, list):
        cleaned = []
        for item in cats:
            if isinstance(item, dict):
                val = item.get("name")
            else:
                val = item
            if isinstance(val, str) and val.strip():
                cleaned.append(val.strip())
        if cleaned:
            parts.append("categories: " + ", ".join(cleaned[:20]))

    return " | ".join(parts).strip()


def iter_object_paths(object_paths: Dict[str, str]) -> Iterable[Tuple[str, str]]:
    return iter(sorted(object_paths.items(), key=lambda kv: kv[1]))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    args = parse_args()
    root = args.output_root / "objaverse_subset"
    version_root = root / "hf-objaverse-v1"
    manifests = root / "manifests"
    glb_dir = version_root / "glbs"
    metadata_dir = version_root / "metadata"
    ensure_dir(manifests)

    object_paths_path = version_root / "object-paths.json.gz"
    download_with_resume(
        f"{BASE_URL}/object-paths.json.gz", object_paths_path, args.max_retries
    )
    object_paths = load_json_gz(object_paths_path)

    completed_path = manifests / "completed_uids.txt"
    pairs_path = manifests / "pairs.jsonl"
    failed_path = manifests / "failed_uids.txt"
    progress_path = manifests / "progress.json"
    completed = load_line_set(completed_path)
    paired = load_line_set(pairs_path, uid_key="uid")

    downloaded_glb_bytes = walk_file_sizes(glb_dir, ".glb")
    initial_glb_bytes = downloaded_glb_bytes
    target_bytes = int(args.target_gb * (1024**3))
    min_free_bytes = int(args.min_free_gb * (1024**3))

    current_shard = None
    current_metadata: Dict[str, Any] = {}
    blocked_shards = set()
    appended = 0
    started = time.time()

    log(
        f"Start: GLB={bytes_to_gb(downloaded_glb_bytes):.2f}GB, "
        f"target={args.target_gb:.2f}GB, completed={len(completed)}"
    )

    for uid, rel_path in iter_object_paths(object_paths):
        if uid in completed:
            continue

        usage = shutil.disk_usage(args.output_root)
        if usage.free < min_free_bytes:
            log(
                f"Stop: free disk {bytes_to_gb(usage.free):.2f}GB < min-free {args.min_free_gb:.2f}GB"
            )
            break

        if downloaded_glb_bytes >= target_bytes:
            log(
                f"Stop: GLB target reached {bytes_to_gb(downloaded_glb_bytes):.2f}GB / {args.target_gb:.2f}GB"
            )
            break

        shard = rel_path.split("/")[1]
        if shard in blocked_shards:
            continue

        glb_path = version_root / rel_path
        existed = glb_path.exists() and glb_path.stat().st_size > 0
        if not existed:
            try:
                download_with_resume(
                    f"{BASE_URL}/{rel_path}", glb_path, args.max_retries
                )
            except Exception as exc:  # noqa: BLE001
                with open(failed_path, "a", encoding="utf-8") as f:
                    f.write(f"{uid}\t{rel_path}\t{exc}\n")
                log(f"Skip uid {uid}: failed glb download ({exc})")
                continue
            downloaded_glb_bytes += glb_path.stat().st_size

        if shard != current_shard:
            shard_path = metadata_dir / f"{shard}.json.gz"
            try:
                download_with_resume(
                    f"{BASE_URL}/metadata/{shard}.json.gz", shard_path, args.max_retries
                )
                current_metadata = load_json_gz(shard_path)
            except Exception as exc:  # noqa: BLE001
                blocked_shards.add(shard)
                with open(failed_path, "a", encoding="utf-8") as f:
                    f.write(f"{uid}\tmetadata/{shard}.json.gz\t{exc}\n")
                log(f"Skip shard {shard}: failed metadata download ({exc})")
                continue
            current_shard = shard

        if uid not in paired:
            meta = current_metadata.get(uid, {})
            text = build_text(meta)
            if not text:
                text = f"uid: {uid}"
            record = {
                "uid": uid,
                "glb_rel_path": rel_path,
                "glb_path": str(glb_path),
                "text": text,
            }
            with open(pairs_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=True) + "\n")
            paired.add(uid)

        with open(completed_path, "a", encoding="utf-8") as f:
            f.write(uid + "\n")
        completed.add(uid)
        appended += 1

        elapsed_s = max(0.001, time.time() - started)
        avg_speed_mb_s = (downloaded_glb_bytes - initial_glb_bytes) / elapsed_s / (1024**2)
        log(
            f"Progress: total={len(completed)}, run+={appended}, "
            f"GLB={bytes_to_gb(downloaded_glb_bytes):.2f}/{args.target_gb:.2f}GB, "
            f"avg={avg_speed_mb_s:.2f} MB/s, uid={uid}"
        )

        if appended % 10 == 0:
            with open(progress_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "completed": len(completed),
                        "downloaded_glb_gb": round(bytes_to_gb(downloaded_glb_bytes), 4),
                        "target_gb": args.target_gb,
                        "timestamp": int(time.time()),
                    },
                    f,
                    ensure_ascii=True,
                    indent=2,
                )

    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "completed": len(completed),
                "downloaded_glb_gb": round(bytes_to_gb(downloaded_glb_bytes), 4),
                "target_gb": args.target_gb,
                "timestamp": int(time.time()),
            },
            f,
            ensure_ascii=True,
            indent=2,
        )

    log(
        f"Done: total={len(completed)}, pairs={len(paired)}, GLB={bytes_to_gb(downloaded_glb_bytes):.2f}GB"
    )


if __name__ == "__main__":
    main()
