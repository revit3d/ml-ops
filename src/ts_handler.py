import io
import torch
import numpy as np
from PIL import Image
from ts.torch_handler.base_handler import BaseHandler
import albumentations as A
from albumentations.pytorch import ToTensorV2

try:
    from utils import heatmaps_to_coords, get_device
    from model import UNet, UNetConfig
except ImportError:
    from src.utils import heatmaps_to_coords, get_device
    from src.model import UNet, UNetConfig


class FaceKeypointsHandler(BaseHandler):
    def __init__(self):
        super(FaceKeypointsHandler, self).__init__()
        self.img_size = (120, 120)
        self.transform = A.Compose(
            [
                A.Resize(*self.img_size),
                A.Normalize(
                    mean=[0.5364, 0.4303, 0.3750], std=[0.2378, 0.2182, 0.2084]
                ),
                ToTensorV2(),
            ]
        )

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")

        self.device = get_device("auto")

        config = UNetConfig(in_ch=3, out_ch=14, ch_mul=32)
        self.model = UNet(config)

        serialized_file = self.manifest["model"]["serializedFile"]
        model_pt_path = f"{model_dir}/{serialized_file}"

        state_dict = torch.load(model_pt_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.initialized = True

    def preprocess(self, data):
        images = []
        for row in data:
            image = row.get("data") or row.get("body")

            if isinstance(image, (bytearray, bytes)):
                image = Image.open(io.BytesIO(image)).convert("RGB")
            else:
                image = Image.open(io.BytesIO(image)).convert("RGB")

            image_np = np.array(image)
            self.original_size = image_np.shape[:2]

            transformed = self.transform(image=image_np)["image"]
            images.append(transformed)

        return torch.stack(images).to(self.device)

    def inference(self, input_batch):
        with torch.no_grad():
            output = self.model(input_batch)
        return output

    def postprocess(self, inference_output):
        upsampled = torch.nn.functional.interpolate(
            inference_output,
            size=self.original_size,
            mode="bilinear",
            align_corners=False,
        )

        coords = heatmaps_to_coords(upsampled.cpu()).squeeze(0)
        coords_list = coords.round().flatten().tolist()

        return [{"keypoints": coords_list}]
