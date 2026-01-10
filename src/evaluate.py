import argparse
import yaml
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .model import UNet
from .data_loader import FaceImageDataset
from .utils import get_device, heatmaps_to_coords

def evaluate(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = get_device(config["training"]["device"])
    img_size = tuple(config["data_processing"]["img_size"])
    dataset_mean = config["data_processing"]["dataset_mean"]
    dataset_std = config["data_processing"]["dataset_std"]

    val_transforms = A.Compose(
        [
            A.Resize(*img_size),
            A.Normalize(mean=dataset_mean, std=dataset_std),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )

    val_dataset = FaceImageDataset(
        image_dir=config["data"]["image_dir"],
        csv_file=config["processed_data"]["val_path"],
        img_size=img_size,
        transform=val_transforms,
        sigma=config["data_processing"]["sigma"],
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        shuffle=False,
    )

    model_path = config["output"]["model_save_dir"]
    model = UNet.from_pretrained(model_path).to(device)
    model.eval()

    total_mse = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            target_coords = heatmaps_to_coords(targets.to(device))
            output_coords = heatmaps_to_coords(outputs)
            
            mse = nn.functional.mse_loss(target_coords, output_coords)
            total_mse += mse.item()

    avg_mse = total_mse / len(val_loader)
    print(f"Evaluation MSE: {avg_mse:.4f}")

    metrics = {"val_mse": avg_mse}
    with open(config["output"]["metrics_file"], "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    evaluate(args.config)
