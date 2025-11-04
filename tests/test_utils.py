import torch
import numpy as np
from src.utils import generate_heatmap, heatmaps_to_coords

def test_generate_heatmap():
    size = (100, 100)
    keypoints = np.array([[20.0, 30.0]]) # x=20, y=30
    sigma = 3.0

    heatmap = generate_heatmap(size, keypoints, sigma)
    assert heatmap.shape == (1, 100, 100)

    max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)

    assert max_idx[1] == 30, "Пик по оси Y не в той точке"
    assert max_idx[2] == 20, "Пик по оси X не в той точке"
    assert np.isclose(heatmap.max(), 1.0)

def test_heatmaps_to_coords():
    heatmap = torch.zeros(1, 1, 10, 10)
    heatmap[0, 0, 3, 7] = 100.0

    coords = heatmaps_to_coords(heatmap)

    assert coords.shape == (1, 1, 2)
    assert torch.allclose(coords[0, 0, 0], torch.tensor(7.0), atol=1e-3)
    assert torch.allclose(coords[0, 0, 1], torch.tensor(3.0), atol=1e-3)
