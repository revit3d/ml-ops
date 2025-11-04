import torch
import torch.nn as nn
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2

from .model import UNet
from .utils import heatmaps_to_coords


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
    original_size = image.shape[:2]  # (h, w)

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
