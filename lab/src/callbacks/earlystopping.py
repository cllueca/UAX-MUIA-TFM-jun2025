# Builtins
import os
# Installed
import numpy as np
import torch
import torch.nn as nn
# Local
from .logger import LogFileLogger
# Types
from typing import (
    Dict,
    Any
)

# Custom early stopping callback
class EarlyStopping:
    def __init__(
        self,
        model_config: Dict[str, Any],
        verbose: bool=False,
        delta: float=0.0,
        restore_best_weights: bool=True,
        logger: LogFileLogger=None
    ):
        self.model_config = model_config
        self.patience = self.model_config['early_stopping_patience']
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.filepath = os.path.join(os.path.abspath('..'), 'models', self.model_config['name'], 'checkpoints', f'{self.model_config["name"]}.pt')
        self.restore_best_weights = restore_best_weights
        self.best_model_state = None
        self.logger = logger

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.filepath) if os.path.dirname(self.filepath) else '.', exist_ok=True)

        print(self.__str__())
        self.logger.log_message(self.__str__())

    def __str__(self):
        return (f"EarlyStopping(patience={self.patience}, verbose={self.verbose}, "
            f"delta={self.delta}, filepath='{self.filepath}', "
            f"restore_best_weights={self.restore_best_weights})")

    def __call__(self, val_loss: float, model: nn.Module):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.logger:
                self.logger.log_early_stopping(self.counter, self.patience)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: nn.Module) -> None:
        '''Save model when validation loss decreases.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        
        if self.logger:
            self.logger.log_val_improvement(None, "Validation loss", self.val_loss_min, val_loss, self.filepath)

        if self.restore_best_weights:
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        torch.save(model.state_dict(), self.filepath)
        self.val_loss_min = val_loss

    def restore_model(self, model: nn.Module) -> None:
        '''Restore best model weights when early stopping is triggered'''
        if self.restore_best_weights and self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            if self.verbose:
                print('Restoring best model weights')
            if self.logger:
                self.logger.log_message("Restoring best model weights")

