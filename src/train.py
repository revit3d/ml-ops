import argparse
import yaml
import logging
import random
import numpy as np
import os
import torch
from torch.optim.swa_utils import AveragedModel, update_bn

from .utils import setup_logging, get_device, weighted_mse_loss
from .data_loader import prepare_dataloaders
from .model import UNet, UNetConfig
from .trainer import Trainer


def main(config_path: str, verbose: bool = False):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    os.makedirs(os.path.dirname(config["output"]["log_file"]), exist_ok=True)
    setup_logging(config["output"]["log_file"])
    logger = logging.getLogger(__name__)
    if verbose:
        logger.setLevel(logging.DEBUG)

    logger.info("--- Starting Training ---")
    logger.info(f"Loaded configuration from {config_path}")

    seed = config["training"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if config["training"]["fast_train"]:
        logger.warning("!!! Running in fast_train mode !!!")
        config["training"]["epochs"] = 1
        config["swa"]["swa_epochs"] = 1
        config["training"]["batch_size"] = 4
        config["training"]["num_workers"] = 0
        config["training"]["device"] = "cpu"

    device = get_device(config["training"]["device"])
    logger.info(f"Using device: {device}")

    train_loader, val_loader = prepare_dataloaders(config)
    logger.info(
        f"Train loader: {len(train_loader)} batches, Val loader: {len(val_loader)} batches"
    )

    model_config = UNetConfig(**config["model"])
    model = UNet(model_config).to(device)
    logger.info(
        f"Model created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters"
    )

    if config["optimizer"]["name"] == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["optimizer"]["lr"],
            weight_decay=config["optimizer"]["weight_decay"],
        )
    elif config["optimizer"]["name"] == "Adam":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["optimizer"]["lr"],
            weight_decay=config["optimizer"]["weight_decay"],
        )
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']['name']}")

    if config["scheduler"]["name"] == "OneCycleLR":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config["scheduler"]["max_lr"],
            steps_per_epoch=len(train_loader),
            epochs=config["training"]["epochs"],
        )
    else:
        raise ValueError(f"Unknwon scheduler: {config['scheduler']['name']}")

    trainer = Trainer(
        model=model,
        criterion=weighted_mse_loss,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        clip_grad_value=config["training"]["clip_grad_value"],
        fast_train=config["training"]["fast_train"],
    )

    for epoch in range(config["training"]["epochs"]):
        logger.info(f"--- Epoch {epoch + 1}/{config['training']['epochs']} ---")
        trainer.train_epoch(train_loader)
        trainer.validate(val_loader)

    if config["swa"]["use_swa"]:
        logger.info("--- Starting SWA Phase ---")
        swa_model = AveragedModel(model)
        swa_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=len(train_loader) * config["swa"]["swa_epochs"]
        )

        trainer.scheduler = swa_scheduler

        for epoch in range(config["swa"]["swa_epochs"]):
            logger.info(f"--- SWA Epoch {epoch + 1}/{config['swa']['swa_epochs']} ---")
            trainer.train_epoch(train_loader)
            swa_model.update_parameters(model)

        logger.info("Updating SWA batch norm statistics...")
        update_bn(train_loader, swa_model, device=device)
        final_model = swa_model.module
    else:
        final_model = model

    logger.info("--- Final Validation on the resulting model ---")
    trainer.model = final_model.to(device)
    final_mse = trainer.validate(val_loader)
    logger.info(f"Final Validation MSE: {final_mse:.4f}")

    save_path = config["output"]["model_save_dir"]
    os.makedirs(save_path, exist_ok=True)
    final_model.save_pretrained(save_path)
    logger.info(f"Model saved to {save_path} in Hugging Face format.")
    logger.info("--- Training Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a face keypoint detection model."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the training configuration file (e.g., configs/train_config.yaml)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging."
    )
    args = parser.parse_args()
    main(args.config, args.verbose)
