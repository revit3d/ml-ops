import os
import pandas as pd
import yaml
from src.prepare import prepare


def test_prepare_script(test_config, temp_data_dir):
    _, _, tmpdir = temp_data_dir
    config_path = tmpdir.join("test_config.yaml")

    with open(config_path, "w") as f:
        yaml.dump(test_config, f)

    prepare(str(config_path))

    assert os.path.exists(test_config["processed_data"]["train_path"])
    assert os.path.exists(test_config["processed_data"]["val_path"])

    train_df = pd.read_csv(test_config["processed_data"]["train_path"])
    val_df = pd.read_csv(test_config["processed_data"]["val_path"])

    assert len(train_df) == 8
    assert len(val_df) == 2

    train_imgs = set(train_df["filename"])
    val_imgs = set(val_df["filename"])
    assert train_imgs.isdisjoint(val_imgs)
