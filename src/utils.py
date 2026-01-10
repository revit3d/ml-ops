import logging
import torch
import numpy as np
import sys

from collections.abc import MutableMapping


def get_device(device_config: str) -> torch.device:
    if device_config == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_config)


def setup_logging(log_file: str):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )


def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_dvc_hash(filepath):
    try:
        import yaml
        dvc_file = filepath + ".dvc"
        with open(dvc_file, 'r') as f:
            data = yaml.safe_load(f)
        return data['outs'][0]['md5']
    except Exception:
        return "unknown"


def generate_heatmap(size, keypoints, sigma):
    h, w = size
    num_points = keypoints.shape[0]
    heatmaps = np.zeros((num_points, h, w), dtype=np.float32)
    for i, (x, y) in enumerate(keypoints):
        if not (0 <= x < w and 0 <= y < h):
            continue
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        heatmaps[i] = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
    return heatmaps


def heatmaps_to_coords(heatmaps, beta=100.0):
    b, n, h, w = heatmaps.shape
    heatmaps = heatmaps.view(b, n, -1)
    prob = torch.softmax(heatmaps * beta, dim=2)
    ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    xs = xs.reshape(-1).to(prob.device)
    ys = ys.reshape(-1).to(prob.device)
    exp_x = (prob * xs).sum(dim=2)
    exp_y = (prob * ys).sum(dim=2)
    coords = torch.stack([exp_x, exp_y], dim=2)
    return coords


def remap_keypoints(keypoints, **_):
    flip_mapping = [3, 2, 1, 0, 9, 8, 7, 6, 5, 4, 10, 13, 12, 11]
    return np.array(keypoints)[flip_mapping]


def weighted_mse_loss(pred, target, weight_factor=10.0):
    weight = torch.ones_like(target)
    weight[target > 0.1] = weight_factor
    loss = torch.mean(weight * (pred - target) ** 2)
    return loss
