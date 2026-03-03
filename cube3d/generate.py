import argparse
import gc
import os
import sys

import torch
import trimesh

from cube3d.inference.engine import Engine, EngineFast
from cube3d.inference.utils import normalize_bbox, select_device
from cube3d.mesh_utils.postprocessing import (
    PYMESHLAB_AVAILABLE,
    create_pymeshset,
    postprocess_mesh,
    save_mesh,
)
from cube3d.renderer import renderer


def is_cuda_oom(exc: Exception) -> bool:
    message = str(exc).lower()
    return "out of memory" in message and "cuda" in message


def clear_cuda_memory(device: torch.device) -> None:
    if device.type != "cuda":
        return
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def relaunch_without_fast_inference() -> None:
    new_args = [arg for arg in sys.argv[1:] if arg != "--fast-inference"]
    new_env = os.environ.copy()
    new_env["CUBE3D_FAST_FALLBACK"] = "1"
    print("Relaunching without --fast-inference due to CUDA OOM.")
    os.execvpe(
        sys.executable, [sys.executable, "-m", "cube3d.generate", *new_args], new_env
    )


def generate_mesh(
    engine,
    prompt,
    output_dir,
    output_name,
    resolution_base=8.0,
    disable_postprocess=False,
    top_p=None,
    bounding_box_xyz=None,
):
    mesh_v_f = engine.t2s(
        [prompt],
        use_kv_cache=True,
        resolution_base=resolution_base,
        top_p=top_p,
        bounding_box_xyz=bounding_box_xyz,
    )
    vertices, faces = mesh_v_f[0][0], mesh_v_f[0][1]
    obj_path = os.path.join(output_dir, f"{output_name}.obj")
    if PYMESHLAB_AVAILABLE:
        ms = create_pymeshset(vertices, faces)
        if not disable_postprocess:
            target_face_num = max(10000, int(faces.shape[0] * 0.1))
            print(f"Postprocessing mesh to {target_face_num} faces")
            postprocess_mesh(ms, target_face_num, obj_path)

        save_mesh(ms, obj_path)
    else:
        print(
            "WARNING: pymeshlab is not available, using trimesh to export obj and skipping optional post processing."
        )
        mesh = trimesh.Trimesh(vertices, faces)
        mesh.export(obj_path)

    return obj_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cube shape generation script")
    parser.add_argument(
        "--config-path",
        type=str,
        default="cube3d/configs/open_model_v0.5.yaml",
        help="Path to the configuration YAML file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/",
        help="Path to the output directory to store .obj and .gif files",
    )
    parser.add_argument(
        "--gpt-ckpt-path",
        type=str,
        required=True,
        help="Path to the main GPT checkpoint file.",
    )
    parser.add_argument(
        "--shape-ckpt-path",
        type=str,
        required=True,
        help="Path to the shape encoder/decoder checkpoint file.",
    )
    parser.add_argument(
        "--fast-inference",
        help="Use optimized inference",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for generating a 3D mesh",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Float < 1: Keep smallest set of tokens with cumulative probability ≥ top_p. Default None: deterministic generation.",
    )
    parser.add_argument(
        "--bounding-box-xyz",
        nargs=3,
        type=float,
        help="Three float values for x, y, z bounding box",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--render-gif",
        help="Render a turntable gif of the mesh",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--disable-postprocessing",
        help="Disable postprocessing on the mesh. This will result in a mesh with more faces.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--resolution-base",
        type=float,
        default=8.0,
        help="Resolution base for the shape decoder.",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = select_device()
    print(f"Using device: {device}")
    fast_fallback_mode = os.environ.get("CUBE3D_FAST_FALLBACK") == "1"
    # Initialize engine based on fast_inference flag
    using_fast_engine = False
    if args.fast_inference and not fast_fallback_mode:
        print(
            "Using cuda graphs, this will take some time to warmup and capture the graph."
        )
        try:
            engine = EngineFast(
                args.config_path,
                args.gpt_ckpt_path,
                args.shape_ckpt_path,
                device=device,
            )
            using_fast_engine = True
            print("Compiled the graph.")
        except RuntimeError as exc:
            if not is_cuda_oom(exc):
                raise
            print(
                "WARNING: Fast inference failed with CUDA OOM during graph setup. "
                "Relaunching without fast inference."
            )
            clear_cuda_memory(device)
            relaunch_without_fast_inference()
    else:
        if args.fast_inference and fast_fallback_mode:
            print("Fast-inference fallback mode active, running standard inference.")
        engine = Engine(
            args.config_path, args.gpt_ckpt_path, args.shape_ckpt_path, device=device
        )

    if args.bounding_box_xyz is not None:
        args.bounding_box_xyz = normalize_bbox(tuple(args.bounding_box_xyz))

    # Generate meshes based on input source
    try:
        obj_path = generate_mesh(
            engine,
            args.prompt,
            args.output_dir,
            "output",
            args.resolution_base,
            args.disable_postprocessing,
            args.top_p,
            args.bounding_box_xyz,
        )
    except RuntimeError as exc:
        if not (using_fast_engine and is_cuda_oom(exc)):
            raise
        print(
            "WARNING: Fast inference failed with CUDA OOM during generation. "
            "Relaunching without fast inference."
        )
        clear_cuda_memory(device)
        relaunch_without_fast_inference()
    if args.render_gif:
        gif_path = renderer.render_turntable(obj_path, args.output_dir)
        print(f"Rendered turntable gif for {args.prompt} at `{gif_path}`")
    print(f"Generated mesh for {args.prompt} at `{obj_path}`")
