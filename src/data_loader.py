import os
import pandas as pd
import numpy as np
import torch
import albumentations as A

from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from .utils import generate_heatmap, remap_keypoints


class FaceImageDataset(Dataset):
    def __init__(self, image_dir, transform: A.Compose, img_size, sigma, gt=None):
        self.image_dir = image_dir
        self.image_files = os.listdir(image_dir)
        self.transform = transform
        self.targets = pd.read_csv(gt, index_col="filename") if gt else None
        self.img_size = img_size
        self.sigma = sigma

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_file)
        image = np.array(Image.open(img_path).convert("RGB"))
        raw_img_size = image.shape[:2]

        if self.targets is not None:
            keypoints = self.targets.loc[img_file].to_numpy().reshape(-1, 2)
            transformed = self.transform(image=image, keypoints=keypoints)
            target_kpts = np.array(transformed["keypoints"])
            target_heatmap = generate_heatmap(
                self.img_size, target_kpts, sigma=self.sigma
            )
            return transformed["image"], torch.from_numpy(target_heatmap)
        else:
            transformed = self.transform(image=image)
            return transformed["image"], torch.tensor(raw_img_size, dtype=int)


def prepare_dataloaders(config: dict) -> tuple[DataLoader, DataLoader]:
    img_size = tuple(config["data_processing"]["img_size"])
    dataset_mean = config["data_processing"]["dataset_mean"]
    dataset_std = config["data_processing"]["dataset_std"]

    train_transforms = A.Compose(
        [
            A.OneOf(
                [
                    A.Compose(
                        [
                            A.HorizontalFlip(p=1.0),
                            A.Lambda(keypoints=remap_keypoints),
                        ],
                        p=0.5,
                    ),
                    A.NoOp(p=0.5),
                ]
            ),
            A.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                rotate=(-20, 20),
                p=0.8,
            ),
            A.RandomResizedCrop(size=img_size, scale=(0.8, 1), p=1.0),
            A.OneOf(
                [
                    A.ColorJitter(
                        brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.8
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.3, contrast_limit=0.3, p=0.8
                    ),
                ],
                p=0.7,
            ),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.GaussNoise(std_range=(0.03, 0.07), p=0.3),
            A.Normalize(mean=dataset_mean, std=dataset_std),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )

    val_transforms = A.Compose(
        [
            A.Resize(*img_size),
            A.Normalize(mean=dataset_mean, std=dataset_std),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )

    full_dataset = FaceImageDataset(
        image_dir=config["data"]["image_dir"],
        gt=config["data"]["gt_path"],
        img_size=img_size,
        transform=train_transforms,
        sigma=config["data_processing"]["sigma"],
    )

    generator = torch.Generator().manual_seed(config["training"]["seed"])
    train_size = int(config["data_processing"]["train_ratio"] * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=generator
    )
    val_dataset.dataset.transform = val_transforms

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        shuffle=False,
    )
    return train_loader, val_loader
