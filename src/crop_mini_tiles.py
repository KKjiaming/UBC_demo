#!/usr/bin/env python
"""Crop a small local UBC-OCEAN subset into 1024px tiles.

This mirrors the core tiling idea in image-crop.ipynb, but does not require
supplemental masks. It reads raw WSI PNGs and writes:

    dataset/processed/ubc-mini-tiles-2048-scale-0-5/train_images/<image_id>/*.png
    dataset/processed/ubc-mini-tiles-2048-scale-0-5/train.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

ROOT_DIR = Path(__file__).resolve().parents[1]


def try_import_pyvips():
    try:
        import pyvips  # type: ignore

        return pyvips
    except Exception:
        return None


def tile_indices(width: int, height: int, size: int) -> Iterable[tuple[int, int, int, int]]:
    for y in range(0, height, size):
        for x in range(0, width, size):
            yield y, min(y + size, height), x, min(x + size, width)


def keep_tile(tile: np.ndarray, drop_thr: float, white_thr: int) -> bool:
    rgb = tile[..., :3]
    black_bg = np.sum(rgb, axis=2) == 0
    white_bg = np.mean(rgb, axis=2) > white_thr
    bg = black_bg | white_bg
    return np.mean(bg) < drop_thr


def pad_resize_save(tile: np.ndarray, out_path: Path, size: int, scale: float) -> None:
    h = w = size
    if tile.shape[:2] != (h, w):
        padded = np.zeros((h, w, tile.shape[2]), dtype=tile.dtype)
        padded[: tile.shape[0], : tile.shape[1], :] = tile
        tile = padded

    new_size = (int(size * scale), int(size * scale))
    Image.fromarray(tile[..., :3]).resize(new_size, Image.Resampling.LANCZOS).save(out_path)


def crop_with_pyvips(img_path: Path, out_dir: Path, size: int, scale: float, drop_thr: float, white_thr: int, max_tiles: int | None) -> int:
    pyvips = try_import_pyvips()
    if pyvips is None:
        raise RuntimeError("pyvips is not available")

    os.environ.setdefault("VIPS_DISC_THRESHOLD", "9gb")
    im = pyvips.Image.new_from_file(str(img_path), access="random")

    written = 0
    for y, y2, x, x2 in tile_indices(im.width, im.height, size):
        tile = im.crop(x, y, x2 - x, y2 - y).numpy()[..., :3]
        if not keep_tile(tile, drop_thr=drop_thr, white_thr=white_thr):
            continue
        out_path = out_dir / f"{written:05}_{int(x2 / size)}-{int(y2 / size)}.png"
        pad_resize_save(tile, out_path, size=size, scale=scale)
        written += 1
        if max_tiles is not None and written >= max_tiles:
            break
    return written


def crop_with_pil(img_path: Path, out_dir: Path, size: int, scale: float, drop_thr: float, white_thr: int, max_tiles: int | None) -> int:
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        width, height = im.size
        written = 0
        for y, y2, x, x2 in tile_indices(width, height, size):
            tile = np.array(im.crop((x, y, x2, y2)))
            if not keep_tile(tile, drop_thr=drop_thr, white_thr=white_thr):
                continue
            out_path = out_dir / f"{written:05}_{int(x2 / size)}-{int(y2 / size)}.png"
            pad_resize_save(tile, out_path, size=size, scale=scale)
            written += 1
            if max_tiles is not None and written >= max_tiles:
                break
        return written


def read_selection(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_selected_train_csv(rows: list[dict[str, str]], out_path: Path) -> None:
    fieldnames = ["image_id", "label", "image_width", "image_height", "is_tma", "selection_note"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default=ROOT_DIR / "dataset" / "raw" / "UBC-OCEAN-mini-kaggle")
    parser.add_argument("--selection", default=ROOT_DIR / "dataset" / "mini_train_selection.csv")
    parser.add_argument("--out-root", default=ROOT_DIR / "dataset" / "processed" / "ubc-mini-tiles-2048-scale-0-5")
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--drop-thr", type=float, default=0.6)
    parser.add_argument("--white-thr", type=int, default=240)
    parser.add_argument("--max-tiles-per-image", type=int, default=64)
    parser.add_argument("--backend", choices=["auto", "pyvips", "pil"], default="auto")
    parser.add_argument("--include-tma", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    selection = Path(args.selection)
    out_root = Path(args.out_root)
    out_images = out_root / "train_images"

    rows = read_selection(selection)
    if not args.include_tma:
        rows = [row for row in rows if row["is_tma"] == "False"]

    if args.overwrite and out_root.exists():
        shutil.rmtree(out_root)
    out_images.mkdir(parents=True, exist_ok=True)

    use_pyvips = args.backend == "pyvips" or (args.backend == "auto" and try_import_pyvips() is not None)
    backend_name = "pyvips" if use_pyvips else "pil"
    print(f"backend={backend_name}")
    print(f"images={len(rows)} out_root={out_root}")

    summary: list[dict[str, str | int]] = []
    for row in rows:
        image_id = row["image_id"]
        img_path = raw_root / "train_images" / f"{image_id}.png"
        if not img_path.exists():
            print(f"[skip] missing {img_path}")
            continue
        image_out = out_images / image_id
        image_out.mkdir(parents=True, exist_ok=True)

        print(f"[crop] {image_id} label={row['label']} path={img_path.name}")
        if use_pyvips:
            n_tiles = crop_with_pyvips(
                img_path,
                image_out,
                size=args.size,
                scale=args.scale,
                drop_thr=args.drop_thr,
                white_thr=args.white_thr,
                max_tiles=args.max_tiles_per_image,
            )
        else:
            n_tiles = crop_with_pil(
                img_path,
                image_out,
                size=args.size,
                scale=args.scale,
                drop_thr=args.drop_thr,
                white_thr=args.white_thr,
                max_tiles=args.max_tiles_per_image,
            )
        summary.append({"image_id": image_id, "label": row["label"], "tiles": n_tiles})
        print(f"       saved_tiles={n_tiles}")

    write_selected_train_csv(rows, out_root / "train.csv")
    with (out_root / "tile_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "label", "tiles"])
        writer.writeheader()
        writer.writerows(summary)

    print(f"wrote {out_root / 'train.csv'}")
    print(f"wrote {out_root / 'tile_summary.csv'}")


if __name__ == "__main__":
    main()
