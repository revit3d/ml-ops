import torch
import torch.nn as nn
import logging
from tqdm.auto import tqdm

from .utils import heatmaps_to_coords

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler = None,
        clip_grad_value: float = 0.05,
        fast_train: bool = False,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.logger = logging.getLogger(__name__)
        self.clip_grad_value = clip_grad_value
        self.fast_train = fast_train

    def train_epoch(self, loader: torch.utils.data.DataLoader):
        self.model.train()
        total_loss = 0.0
        total_steps = 2 if self.fast_train else len(loader)
        for i, (inputs, targets) in tqdm(enumerate(loader), desc="Training epoch", total=total_steps):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_value)
            self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

            total_loss += loss.item()

            if self.fast_train and i > 1:
                break

        avg_loss = total_loss / total_steps
        self.logger.info(f"Train Loss: {avg_loss:.4f}")
        return avg_loss

    @torch.no_grad()
    def validate(self, loader: torch.utils.data.DataLoader):
        self.model.eval()
        total_mse = 0.0
        for inputs, targets in tqdm(loader, desc="Validation"):
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            target_coords = heatmaps_to_coords(targets)
            output_coords = heatmaps_to_coords(outputs.cpu())

            mse = nn.functional.mse_loss(target_coords, output_coords)
            total_mse += mse.item()

        avg_mse = total_mse / len(loader)
        self.logger.info(f"Validation coordinate MSE: {avg_mse:.4f}")
        return avg_mse
