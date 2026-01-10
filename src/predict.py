import argparse
import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import yaml

from .model import UNet
from .utils import heatmaps_to_coords, get_device

def predict_batch(model, image_dir, device, img_size, mean, std):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    transform = A.Compose([
        A.Resize(*img_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

    results = []
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    model.eval()

    for img_name in tqdm(image_files, desc="Running inference"):
        img_path = os.path.join(image_dir, img_name)
        try:
            image = np.array(Image.open(img_path).convert("RGB"))
            original_size = image.shape[:2]

            transformed = transform(image=image)["image"].unsqueeze(0).to(device)

            with torch.no_grad():
                heatmaps = model(transformed)

            upsampled_heatmaps = torch.nn.functional.interpolate(
                heatmaps,
                size=original_size,
                mode="bilinear",
                align_corners=False,
            )
            
            coords = heatmaps_to_coords(upsampled_heatmaps.cpu()).squeeze(0)
            coords_flat = coords.round().flatten().tolist()
            
            row = {"filename": img_name}
            for i, val in enumerate(coords_flat):
                row[f"p{i}"] = val
            results.append(row)

        except Exception as e:
            print(f"Error processing {img_name}: {e}")

    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description="Offline Inference for Face Keypoints")
    parser.add_argument("--input_path", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save result CSV")
    parser.add_argument("--model_path", type=str, default="./saved_models/face_keypoint_detector", help="Path to saved model folder")
    parser.add_argument("--config_path", type=str, default="configs/training.yaml", help="Path to config file for preprocessing params")
    
    args = parser.parse_args()

    if os.path.exists(args.config_path):
        with open(args.config_path, "r") as f:
            config = yaml.safe_load(f)
            img_size = tuple(config["data_processing"]["img_size"])
            mean = config["data_processing"]["dataset_mean"]
            std = config["data_processing"]["dataset_std"]
    else:
        print("Warning: Config not found, using defaults.")
        img_size = (120, 120)
        mean = [0.5364, 0.4303, 0.3750]
        std = [0.2378, 0.2182, 0.2084]

    device = get_device("auto")
    print(f"Loading model from {args.model_path} to {device}...")
    
    try:
        model = UNet.from_pretrained(args.model_path).to(device)
    except OSError:
        raise FileNotFoundError(f"Model not found at {args.model_path}. Make sure to build the image WITH the model or mount it.")

    print(f"Processing images from {args.input_path}...")
    df = predict_batch(model, args.input_path, device, img_size, mean, std)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    df.to_csv(args.output_path, index=False)
    print(f"Saved predictions to {args.output_path}")

if __name__ == "__main__":
    main()
