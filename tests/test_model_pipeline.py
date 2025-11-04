import torch
import numpy as np
from src.trainer import Trainer
from src.utils import weighted_mse_loss
from src.predict import (
    load_model_for_inference,
    predict_keypoints,
    get_prediction_transforms,
)


def test_model_forward_pass(uninitialized_model, test_config):
    img_size = test_config["data_processing"]["img_size"]

    model = uninitialized_model
    dummy_input = torch.randn(2, 3, *img_size)
    output = model(dummy_input)

    assert output.shape == (2, 14, *img_size)
    assert output.min() >= 0.0 and output.max() <= 1.0


def test_training_step(uninitialized_model, test_config):
    img_size = test_config["data_processing"]["img_size"]

    model = uninitialized_model
    device = torch.device("cpu")
    model.to(device)

    weights_before = [p.clone().detach() for p in model.parameters()]

    optimizer = torch.optim.Adam(model.parameters())
    trainer = Trainer(model, weighted_mse_loss, optimizer, device)

    dummy_inputs = torch.randn(2, 3, *img_size, device=device)
    dummy_targets = torch.rand(2, 14, *img_size, device=device)

    trainer.optimizer.zero_grad()
    outputs = trainer.model(dummy_inputs)
    loss = trainer.criterion(outputs, dummy_targets)
    loss.backward()
    trainer.optimizer.step()

    weights_after = list(model.parameters())

    weight_changed = False
    for p_before, p_after in zip(weights_before, weights_after):
        if not torch.equal(p_before, p_after):
            weight_changed = True
            break

    assert weight_changed, "Model weights didn't change after training step"


def test_prediction_pipeline(uninitialized_model, tmpdir, test_config):
    img_size = test_config["data_processing"]["img_size"]
    mean = test_config["data_processing"]["dataset_mean"]
    std = test_config["data_processing"]["dataset_std"]

    device = torch.device("cpu")
    model = uninitialized_model.to(device)

    model_path = tmpdir.mkdir("test_model")
    model.save_pretrained(model_path)
    loaded_model = load_model_for_inference(str(model_path), device)

    fake_image = np.random.randint(0, 256, (200, 150, 3), dtype=np.uint8)

    transforms = get_prediction_transforms(
        img_size=img_size,
        mean=mean,
        std=std,
    )

    coords = predict_keypoints(loaded_model, fake_image, transforms, device)

    assert isinstance(coords, list)
    assert len(coords) == 28

    for val in coords:
        assert isinstance(val, float)

    xs = coords[0::2]
    ys = coords[1::2]

    assert all(0 <= x < 150 for x in xs)
    assert all(0 <= y < 200 for y in ys)
