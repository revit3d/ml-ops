import argparse
import os
import sys
import torch
import pandas as pd
import numpy as np
import yaml
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .model import UNet
from .utils import heatmaps_to_coords, get_device


def get_prediction_transforms(
    img_size: tuple[int, int], mean: list, std: list
) -> A.Compose:
    return A.Compose(
        [
            A.Resize(*img_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def load_model_for_inference(model_path: str, device: torch.device) -> nn.Module:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    try:
        model = UNet.from_pretrained(model_path)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        raise IOError(f"Error loading model from {model_path}: {e}")


def predict_keypoints(
    model: nn.Module, image: np.ndarray, transforms: A.Compose, device: torch.device
) -> list[float]:
    original_size = image.shape[:2]

    transformed = transforms(image=image)
    input_tensor = transformed["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        heatmaps = model(input_tensor)

    upsampled_heatmaps = nn.functional.interpolate(
        heatmaps,
        size=original_size,
        mode="bilinear",
        align_corners=False,
    )

    coords = heatmaps_to_coords(upsampled_heatmaps.cpu()).squeeze(0)
    return coords.round().flatten().tolist()


def predict_batch(model, image_dir, output_path, device, img_size, mean, std):
    transforms = get_prediction_transforms(img_size, mean, std)

    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    image_files = [
        f for f in os.listdir(image_dir) if f.lower().endswith(valid_extensions)
    ]

    print(f"[INFO] Found {len(image_files)} images in {image_dir}", flush=True)
    if len(image_files) == 0:
        return

    results = []

    for img_name in tqdm(image_files, desc="Running inference", file=sys.stdout):
        img_path = os.path.join(image_dir, img_name)
        try:
            image = np.array(Image.open(img_path).convert("RGB"))

            coords_flat = predict_keypoints(model, image, transforms, device)

            row = {"filename": img_name}
            for i, val in enumerate(coords_flat):
                row[f"p{i}"] = val
            results.append(row)

        except Exception as e:
            print(f"[ERROR] processing {img_name}: {e}", flush=True)

    if results:
        df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[SUCCESS] Saved predictions to {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--model_path", type=str, default="./saved_models/face_keypoint_detector"
    )
    parser.add_argument("--config_path", type=str, default="configs/training.yaml")

    args = parser.parse_args()

    if os.path.exists(args.config_path):
        with open(args.config_path, "r") as f:
            config = yaml.safe_load(f)
            img_size = tuple(config["data_processing"]["img_size"])
            mean = config["data_processing"]["dataset_mean"]
            std = config["data_processing"]["dataset_std"]
    else:
        print("[INFO] Config not found, using defaults.", flush=True)
        img_size = (120, 120)
        mean = [0.5364, 0.4303, 0.3750]
        std = [0.2378, 0.2182, 0.2084]

    device = get_device("auto")

    try:
        model = load_model_for_inference(args.model_path, device)
    except Exception as e:
        print(f"[ERROR] {e}", flush=True)
        sys.exit(1)

    predict_batch(model, args.input_path, args.output_path, device, img_size, mean, std)


if __name__ == "__main__":
    main()
