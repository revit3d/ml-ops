import argparse
import yaml
import logging
import random
import numpy as np
import os
import torch
import mlflow
import mlflow.pytorch
from torch.optim.swa_utils import AveragedModel, update_bn

from .utils import setup_logging, get_device, weighted_mse_loss, flatten_dict, get_dvc_hash
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

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    mlflow.pytorch.autolog(log_models=False)

    with mlflow.start_run(run_name=config["mlflow"].get("run_name")) as run:
        logger.info(f"MLflow Run ID: {run.info.run_id}")

        mlflow.log_params(flatten_dict(config))

        data_hash = get_dvc_hash("data/gt.csv") 
        mlflow.set_tag("dvc_gt_hash", data_hash)

        if os.path.exists("dvc.yaml"):
            mlflow.log_artifact("dvc.yaml")
        if os.path.exists("dvc.lock"):
            mlflow.log_artifact("dvc.lock")

        seed = config["training"]["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if config["training"]["fast_train"]:
            logger.warning("!!! running in fast_train mode !!!")
            config["training"]["epochs"] = 1
            config["swa"]["swa_epochs"] = 1
            config["training"]["batch_size"] = 4
            config["training"]["device"] = "cpu"

        device = get_device(config["training"]["device"])
        
        train_loader, val_loader = prepare_dataloaders(config)
        
        model_config = UNetConfig(**config["model"])
        model = UNet(model_config).to(device)
        
        if config["optimizer"]["name"] == "AdamW":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config["optimizer"]["lr"],
                weight_decay=config["optimizer"]["weight_decay"],
            )
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["optimizer"]["lr"])

        if config["scheduler"]["name"] == "OneCycleLR":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=config["scheduler"]["max_lr"],
                steps_per_epoch=len(train_loader),
                epochs=config["training"]["epochs"],
            )
        else:
            scheduler = None

        trainer = Trainer(
            model=model,
            criterion=weighted_mse_loss,
            optimizer=optimizer,
            device=device,
            scheduler=scheduler,
            clip_grad_value=config["training"]["clip_grad_value"],
            fast_train=config["training"]["fast_train"],
        )

        extra_epochs = config["training"]["epochs"]

        for epoch in range(extra_epochs):
            logger.info(f"--- Epoch {epoch + 1}/{extra_epochs} ---")
            trainer.train_epoch(train_loader, epoch)
            trainer.validate(val_loader, epoch)

        if config["swa"]["use_swa"]:
            logger.info("--- Starting SWA Phase ---")
            swa_model = AveragedModel(model)
            swa_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=len(train_loader) * config["swa"]["swa_epochs"]
            )
            trainer.scheduler = swa_scheduler

            for i in range(config["swa"]["swa_epochs"]):
                current_swa_epoch = extra_epochs + i
                logger.info(f"--- SWA Epoch {i + 1}/{config['swa']['swa_epochs']} ---")
                trainer.train_epoch(train_loader, current_swa_epoch)
                swa_model.update_parameters(model)
            
            update_bn(train_loader, swa_model, device=device)
            final_model = swa_model.module
        else:
            final_model = model

        logger.info("--- Final Validation ---")
        trainer.model = final_model.to(device)
        final_mse = trainer.validate(val_loader, epoch=extra_epochs + config["swa"]["swa_epochs"])

        mlflow.log_metric("final_mse", final_mse)

        save_path = config["output"]["model_save_dir"]
        os.makedirs(save_path, exist_ok=True)
        final_model.save_pretrained(save_path)
        
        logger.info(f"Model saved to {save_path}")

        mlflow.log_artifacts(save_path, artifact_path="model")

        mlflow.log_artifact(config["output"]["log_file"])

        logger.info("--- Training Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    main(args.config, args.verbose)
