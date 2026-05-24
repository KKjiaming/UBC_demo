import argparse
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

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
LABEL2INDEX = {label: idx for idx, label in enumerate(LABELS)}


def pick_default_image() -> str:
    for image_id in sorted(os.listdir(TRAIN_IMAGES_DIR)):
        image_dir = os.path.join(TRAIN_IMAGES_DIR, image_id)
        if not os.path.isdir(image_dir):
            continue
        images = sorted(
            os.path.join(image_dir, name)
            for name in os.listdir(image_dir)
            if name.lower().endswith(".png")
        )
        if images:
            return images[0]
    raise FileNotFoundError(f"No png image found under {TRAIN_IMAGES_DIR}")


def gt_for_image(image_path: str) -> str:
    image_id = int(os.path.basename(os.path.dirname(image_path)))
    train = pd.read_csv(os.path.join(ROOT_DIR, "train.csv"))
    train.loc[train["image_id"] == 15583, "label"] = "MC"
    label = train.set_index("image_id").loc[image_id, "label"]
    return str(label)


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


def preprocess(image_rgb: np.ndarray) -> torch.Tensor:
    transform = A.Compose(
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
    return transform(image=image_rgb)["image"].unsqueeze(0)


def draw_overlay(image_rgb: np.ndarray, probs: np.ndarray, gt_label: str, out_path: str) -> None:
    pred_idx = int(np.argmax(probs))
    pred_label = LABELS[pred_idx]
    correct = pred_label == gt_label

    img = Image.fromarray(image_rgb).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    panel_w = min(520, img.width - 24)
    panel_h = 265
    x0, y0 = 18, 18
    draw.rounded_rectangle(
        [x0, y0, x0 + panel_w, y0 + panel_h],
        radius=8,
        fill=(0, 0, 0, 185),
        outline=(255, 255, 255, 130),
        width=2,
    )

    try:
        font_big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except OSError:
        font_big = font = font_small = ImageFont.load_default()

    status = "MATCH" if correct else "MISMATCH"
    status_color = (90, 230, 130, 255) if correct else (255, 110, 90, 255)
    draw.text((x0 + 18, y0 + 14), f"Prediction: {pred_label}  ({probs[pred_idx] * 100:.2f}%)", font=font_big, fill=(255, 255, 255, 255))
    draw.text((x0 + 18, y0 + 50), f"GT: {gt_label}    Result: {status}", font=font, fill=status_color)

    bar_x = x0 + 110
    bar_y = y0 + 88
    bar_w = panel_w - 145
    bar_h = 18
    for i, label in enumerate(LABELS):
        y = bar_y + i * 32
        prob = float(probs[i])
        color = (80, 170, 255, 255)
        if label == gt_label:
            color = (90, 230, 130, 255)
        if i == pred_idx and label != gt_label:
            color = (255, 150, 70, 255)

        draw.text((x0 + 18, y - 3), label, font=font_small, fill=(255, 255, 255, 255))
        draw.rectangle([bar_x, y, bar_x + bar_w, y + bar_h], fill=(255, 255, 255, 40))
        draw.rectangle([bar_x, y, bar_x + int(bar_w * prob), y + bar_h], fill=color)
        draw.text((bar_x + bar_w + 8, y - 3), f"{prob * 100:5.2f}%", font=font_small, fill=(255, 255, 255, 255))

    annotated = Image.alpha_composite(img, overlay).convert("RGB")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    annotated.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None, help="Path to a 1024 tile png. Defaults to the first train tile.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    image_path = os.path.abspath(args.image or pick_default_image())
    checkpoint_path = os.path.abspath(args.checkpoint)
    image_id = os.path.basename(os.path.dirname(image_path))
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    out_path = args.output or os.path.join(DEFAULT_OUTPUT_DIR, f"{image_id}_{image_name}_annotated.png")

    gt_label = gt_for_image(image_path)
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device)
    x = preprocess(image_rgb).to(device, dtype=torch.float32)
    with torch.inference_mode():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()

    draw_overlay(image_rgb, probs, gt_label, out_path)

    pred_idx = int(np.argmax(probs))
    print(f"image_path={image_path}")
    print(f"checkpoint={checkpoint_path}")
    print(f"gt={gt_label}")
    print(f"pred={LABELS[pred_idx]}")
    for label, prob in zip(LABELS, probs):
        print(f"{label}={prob:.6f}")
    print(f"output={os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
