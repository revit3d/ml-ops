from src.data_loader import FaceImageDataset, prepare_dataloaders
from src.predict import get_prediction_transforms
import torch


def test_dataset_item(test_config):
    image_dir = test_config["data"]["image_dir"]
    gt_path = test_config["data"]["gt_path"]
    img_size = test_config["data_processing"]["img_size"]

    transforms = get_prediction_transforms(
        img_size=img_size,
        mean=test_config["data_processing"]["dataset_mean"],
        std=test_config["data_processing"]["dataset_std"],
    )

    dataset = FaceImageDataset(
        image_dir=image_dir,
        csv_file=gt_path,
        transform=transforms,
        img_size=img_size,
        sigma=test_config["data_processing"]["sigma"],
    )

    assert len(dataset) == 10

    image, heatmap = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert isinstance(heatmap, torch.Tensor)
    assert image.shape == (3, *img_size)
    assert heatmap.shape == (14, *img_size)


def test_prepare_dataloaders(test_config, prepared_data):
    train_loader, val_loader = prepare_dataloaders(test_config)

    assert len(train_loader.dataset) == 8
    assert len(val_loader.dataset) == 2

    train_batch_images, train_batch_heatmaps = next(iter(train_loader))
    batch_size = test_config["training"]["batch_size"]
    img_size = test_config["data_processing"]["img_size"]

    assert train_batch_images.shape == (batch_size, 3, *img_size)
    assert train_batch_heatmaps.shape == (batch_size, 14, *img_size)
