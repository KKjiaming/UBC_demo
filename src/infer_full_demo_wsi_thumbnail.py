import argparse
import os
import re

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
ROOT_DIR = os.path.join(DATASET_DIR, "processed", "ubc-mini-tiles-2048-scale-0-5")
TRAIN_IMAGES_DIR = os.path.join(ROOT_DIR, "train_images")
DEFAULT_CHECKPOINT = os.path.join(
    BASE_DIR,
    "save",
    "full_demo",
    "tf_efficientnetv2_l.in21k_ft_in1k_bestTrainLoss_imgsize_1024_full.pt",
)
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "inference_demo")

LABELS = ["CC", "EC", "HGSC", "LGSC", "MC"]
COLORS = {
    "CC": (70, 170, 255),
    "EC": (255, 190, 70),
    "HGSC": (255, 90, 110),
    "LGSC": (130, 230, 110),
    "MC": (185, 120, 255),
}


def read_gt_table() -> pd.DataFrame:
    train = pd.read_csv(os.path.join(ROOT_DIR, "train.csv"))
    train.loc[train["image_id"] == 15583, "label"] = "MC"
    return train


def default_image_id() -> int:
    train = read_gt_table()
    return int(train.iloc[0]["image_id"])


def tile_paths_for_image(image_id: int) -> list[str]:
    image_dir = os.path.join(TRAIN_IMAGES_DIR, str(image_id))
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"No tile directory: {image_dir}")
    paths = sorted(
        os.path.join(image_dir, name)
        for name in os.listdir(image_dir)
        if name.lower().endswith(".png")
    )
    if not paths:
        raise FileNotFoundError(f"No png tile found under {image_dir}")
    return paths


def gt_for_image(image_id: int) -> str:
    train = read_gt_table()
    return str(train.set_index("image_id").loc[image_id, "label"])


def parse_tile_xy(path: str) -> tuple[int, int]:
    name = os.path.splitext(os.path.basename(path))[0]
    match = re.search(r"_(\d+)-(\d+)$", name)
    if match is None:
        return 0, 0
    return int(match.group(1)), int(match.group(2))


class TileDataset(Dataset):
    def __init__(self, paths: list[str]):
        self.paths = paths
        self.transform = A.Compose(
            [
                A.Resize(1024, 1024),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image_bgr = cv2.imread(self.paths[idx])
        if image_bgr is None:
            raise FileNotFoundError(self.paths[idx])
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return self.transform(image=image_rgb)["image"]


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    model = timm.create_model(
        "tf_efficientnetv2_l.in21k_ft_in1k",
        pretrained=False,
        checkpoint_path=None,
        num_classes=len(LABELS),
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def infer_tiles(paths: list[str], checkpoint_path: str, batch_size: int) -> np.ndarray:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device)
    loader = DataLoader(TileDataset(paths), batch_size=batch_size, shuffle=False, num_workers=2)

    probs = []
    with torch.inference_mode():
        for images in loader:
            images = images.to(device, dtype=torch.float32)
            logits = model(images)
            probs.append(F.softmax(logits, dim=1).detach().cpu().numpy())
    return np.concatenate(probs, axis=0)


def font(size: int, bold: bool = False):
    try:
        name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)
    except OSError:
        return ImageFont.load_default()


def make_thumbnail(
    image_id: int,
    tile_paths: list[str],
    probs: np.ndarray,
    gt_label: str,
    output_path: str,
    cell: int,
) -> None:
    coords = [parse_tile_xy(path) for path in tile_paths]
    min_x = min(x for x, _ in coords)
    min_y = min(y for _, y in coords)
    max_x = max(x for x, _ in coords)
    max_y = max(y for _, y in coords)
    cols = max_x - min_x + 1
    rows = max_y - min_y + 1

    panel_h = 210
    margin = 16
    w = max(cols * cell + margin * 2, 760)
    h = rows * cell + panel_h + margin * 2
    canvas = Image.new("RGB", (w, h), (245, 245, 245))

    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.rectangle([0, 0, w, panel_h], fill=(22, 24, 28, 255))

    mean_probs = probs.mean(axis=0)
    pred_idx = int(np.argmax(mean_probs))
    pred_label = LABELS[pred_idx]
    status = "MATCH" if pred_label == gt_label else "MISMATCH"
    status_color = (90, 230, 130, 255) if pred_label == gt_label else (255, 110, 90, 255)

    draw.text((24, 18), f"WSI {image_id}  Prediction: {pred_label} ({mean_probs[pred_idx] * 100:.2f}%)", font=font(28, True), fill=(255, 255, 255, 255))
    draw.text((24, 56), f"GT: {gt_label}    Result: {status}    Tiles: {len(tile_paths)}", font=font(21), fill=status_color)

    bar_x = 118
    bar_y = 92
    bar_w = 360
    for i, label in enumerate(LABELS):
        y = bar_y + i * 22
        color = COLORS[label]
        prob = float(mean_probs[i])
        draw.text((24, y - 3), label, font=font(16), fill=(255, 255, 255, 255))
        draw.rectangle([bar_x, y, bar_x + bar_w, y + 14], fill=(255, 255, 255, 38))
        draw.rectangle([bar_x, y, bar_x + int(bar_w * prob), y + 14], fill=(*color, 255))
        draw.text((bar_x + bar_w + 10, y - 4), f"{prob * 100:5.2f}%", font=font(16), fill=(255, 255, 255, 255))

    legend_x = 535
    for i, label in enumerate(LABELS):
        y = 92 + i * 22
        draw.rectangle([legend_x, y, legend_x + 18, y + 14], fill=(*COLORS[label], 220))
        draw.text((legend_x + 26, y - 4), label, font=font(16), fill=(255, 255, 255, 255))

    grid_y0 = panel_h + margin
    for path, prob in zip(tile_paths, probs):
        x, y = parse_tile_xy(path)
        gx = margin + (x - min_x) * cell
        gy = grid_y0 + (y - min_y) * cell
        tile = Image.open(path).convert("RGB").resize((cell, cell), Image.BILINEAR)
        canvas.paste(tile, (gx, gy))

        idx = int(np.argmax(prob))
        label = LABELS[idx]
        confidence = float(prob[idx])
        color = COLORS[label]
        alpha = int(70 + 120 * confidence)
        draw.rectangle([gx, gy, gx + cell, gy + cell], fill=(*color, alpha))
        draw.rectangle([gx, gy, gx + cell - 1, gy + cell - 1], outline=(*color, 255), width=3)
        draw.text((gx + 5, gy + 4), f"{label} {confidence * 100:.0f}%", font=font(max(11, cell // 9), True), fill=(255, 255, 255, 255))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-id", type=int, default=None)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cell", type=int, default=128)
    args = parser.parse_args()

    image_id = args.image_id if args.image_id is not None else default_image_id()
    tile_paths = tile_paths_for_image(image_id)
    gt_label = gt_for_image(image_id)
    checkpoint_path = os.path.abspath(args.checkpoint)
    output_path = args.output or os.path.join(DEFAULT_OUTPUT_DIR, f"{image_id}_wsi_thumbnail_heatmap.png")

    probs = infer_tiles(tile_paths, checkpoint_path, args.batch_size)
    make_thumbnail(image_id, tile_paths, probs, gt_label, output_path, args.cell)

    mean_probs = probs.mean(axis=0)
    pred_idx = int(np.argmax(mean_probs))
    print(f"image_id={image_id}")
    print(f"tiles={len(tile_paths)}")
    print(f"checkpoint={checkpoint_path}")
    print(f"gt={gt_label}")
    print(f"pred={LABELS[pred_idx]}")
    for label, prob in zip(LABELS, mean_probs):
        print(f"{label}={prob:.6f}")
    print(f"output={os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
