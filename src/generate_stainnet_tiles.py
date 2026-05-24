#!/usr/bin/env python
"""Generate StainNet-normalized tile images for stainmix training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = ROOT_DIR / "dataset" / "processed" / "ubc-mini-tiles-2048-scale-0-5" / "train_images"
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "dataset" / "ubc-tilesx1024-stain3-data"
DEFAULT_STAINNET_DIR = ROOT_DIR / "dataset" / "stainnet"
DEFAULT_CHECKPOINT = DEFAULT_STAINNET_DIR / "StainNet-Public_layer3_ch32.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stainnet-dir", type=Path, default=DEFAULT_STAINNET_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_model(stainnet_dir: Path, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    sys.path.insert(0, str(stainnet_dir))
    from models import StainNet  # type: ignore

    model = StainNet().to(device)
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def normalize_image(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    arr = np.where(arr == 0, 255, arr).astype(np.float32)
    arr = arr.transpose((2, 0, 1))
    arr = ((arr / 255.0) - 0.5) / 0.5
    return torch.from_numpy(arr).unsqueeze(0)


def unnormalize_image(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu().squeeze(0)
    tensor = torch.clamp(tensor * 0.5 + 0.5, 0.0, 1.0)
    arr = (tensor.numpy() * 255.0).astype(np.uint8).transpose((1, 2, 0))
    return Image.fromarray(arr)


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    stainnet_dir = args.stainnet_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    device = torch.device(args.device)

    if not input_root.is_dir():
        raise FileNotFoundError(f"Missing input root: {input_root}")
    if not stainnet_dir.is_dir():
        raise FileNotFoundError(f"Missing StainNet code directory: {stainnet_dir}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing StainNet checkpoint: {checkpoint}")

    model = load_model(stainnet_dir, checkpoint, device)
    image_paths = sorted(input_root.glob("*/*.png"))
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    print(f"input_root={input_root}")
    print(f"output_root={output_root}")
    print(f"checkpoint={checkpoint}")
    print(f"device={device}")
    print(f"images={len(image_paths)}")

    with torch.inference_mode():
        for image_path in tqdm(image_paths):
            rel_path = image_path.relative_to(input_root)
            out_path = output_root / rel_path
            if out_path.exists() and not args.overwrite:
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)

            image = Image.open(image_path)
            tensor = normalize_image(image).to(device)
            normalized = model(tensor)
            unnormalize_image(normalized).save(out_path)


if __name__ == "__main__":
    main()
