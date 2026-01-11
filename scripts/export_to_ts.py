import torch
import os
import sys

from src.model import UNet


def export_model(model_path, output_path):
    print(f"Loading model from {model_path}...")
    model = UNet.from_pretrained(model_path)
    model.eval()

    torch.save(model.state_dict(), output_path)
    print(f"Model state_dict saved to {output_path}")


if __name__ == "__main__":
    input_dir = "./saved_models/face_keypoint_detector"
    output_file = "./saved_models/face_kpts.pt"

    if not os.path.exists(input_dir):
        print(f"Error: {input_dir} does not exist. Train the model first.")
        sys.exit(1)

    export_model(input_dir, output_file)
