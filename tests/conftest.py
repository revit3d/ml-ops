import pytest
import pandas as pd
import numpy as np
from PIL import Image
import os
from src.model import UNet, UNetConfig


@pytest.fixture(scope="session")
def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def temp_data_dir(tmpdir):
    image_dir = tmpdir.mkdir("images")
    gt_data = []

    for i in range(5):
        img_array = np.random.randint(0, 256, (120, 120, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img_name = f"test_{i}.jpg"
        img.save(image_dir.join(img_name))

        keypoints = np.random.uniform(10, 110, 28).tolist()
        gt_data.append([img_name] + keypoints)

    columns = ["filename"] + [f"p{i}" for i in range(28)]
    gt_df = pd.DataFrame(gt_data, columns=columns)
    gt_path = tmpdir.join("gt.csv")
    gt_df.to_csv(gt_path, index=False)

    return str(image_dir), str(gt_path)


@pytest.fixture
def test_config(temp_data_dir):
    image_dir, gt_path = temp_data_dir
    return {
        "data": {"image_dir": image_dir, "gt_path": gt_path},
        "data_processing": {
            "img_size": [120, 120],
            "sigma": 3.0,
            "train_ratio": 0.8,
            "dataset_mean": [0.5, 0.5, 0.5],
            "dataset_std": [0.5, 0.5, 0.5],
        },
        "training": {
            "seed": 42,
            "fast_train": False,
            "batch_size": 2,
            "num_workers": 0,
        },
    }


@pytest.fixture(scope="session")
def uninitialized_model():
    config = UNetConfig(in_ch=3, out_ch=14, ch_mul=4)
    model = UNet(config)
    return model
