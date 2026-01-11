import io
import pytest
import torch
import numpy as np
from src.ts_handler import FaceKeypointsHandler
from PIL import Image


@pytest.fixture
def handler():
    h = FaceKeypointsHandler()
    h.device = torch.device("cpu")
    h.initialized = True
    return h


def test_preprocess(handler):
    img = Image.new("RGB", (200, 200), color="red")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_bytes = img_byte_arr.getvalue()

    data = [{"data": img_bytes}]

    input_tensor = handler.preprocess(data)

    assert input_tensor.shape == (1, 3, 120, 120)
    assert handler.original_size == (200, 200)


def test_postprocess(handler):
    heatmap = torch.zeros(1, 14, 120, 120)

    heatmap[:, :, 60, 60] = 100.0

    handler.original_size = (240, 240)

    result = handler.postprocess(heatmap)

    assert isinstance(result, list)
    assert len(result) == 1
    keypoints = result[0]["keypoints"]
    assert len(keypoints) == 28

    assert np.isclose(keypoints[0], 120.0, atol=2.0)
    assert np.isclose(keypoints[1], 120.0, atol=2.0)
