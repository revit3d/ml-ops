import os
import yaml
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def prepare(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    gt_path = config["data"]["gt_path"]
    train_ratio = config["data_processing"]["train_ratio"]
    seed = config["training"]["seed"]

    os.makedirs(config["processed_data"]["dir"], exist_ok=True)

    df = pd.read_csv(gt_path)
    train_df, val_df = train_test_split(
        df, train_size=train_ratio, random_state=seed, shuffle=True
    )

    train_path = config["processed_data"]["train_path"]
    val_path = config["processed_data"]["val_path"]

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"Data prepared. Train: {len(train_df)}, Val: {len(val_df)}")
    print(f"Saved to {train_path} and {val_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    prepare(args.config)
