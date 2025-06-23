# Builtins
import os
# Installed
import torch
import torch.nn as nn
# Local
from .logger import LogFileLogger
# Types
from typing import (
    Dict,
    Any
)

# Model checkpoint equivalent
class ModelCheckpoint:
    def __init__(
        self,
        model_config: Dict[str, Any],
        monitor: str='val_loss',
        verbose: bool=False,
        save_best_only: bool=False,
        mode: str='min',
        logger: LogFileLogger=None
    ):
        self.model_config = model_config
        self.filepath = os.path.join(os.path.abspath('..'), 'models', self.model_config['name'], 'models', f'{self.model_config["name"]}.pth')
        self.save_method = self.model_config['save_method']

        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.mode = mode
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.logger = logger

        if self.save_method not in ['complete', 'state_dict', 'with_params']:
            raise Exception(f"Save method not allowed, choose one from: {', '.join(['complete', 'state_dict', 'with_params'])}")

        if self.save_method == 'with_params' and not self.filepath.endswith('.pt'):
            self.filepath = self.filepath.replace('.pth', '.pt')

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.filepath) if os.path.dirname(self.filepath) else '.', exist_ok=True)

        print(self.__str__())
        self.logger.log_message(self.__str__())

    def __str__(self):
        return (f"ModelCheckpoint(filepath={self.filepath}, monitor={self.monitor}, "
            f"verbose={self.verbose}, save_best_only='{self.save_best_only}', "
            f"mode={self.mode}, save_method={self.save_method})")
        
    def _save_model(self, model: nn.Module, epoch: int):
        """Save the model in the specified format"""
        model.eval()

        if self.save_method == 'complete':
            # Save the entire model (architecture + weights)
            torch.save(model, self.filepath)
        elif self.save_method == 'state_dict':
            # Save only the state dict
            torch.save(model.state_dict(), self.filepath)
        elif self.save_method == 'with_params':
            # Save model class, init parameters, and state_dict together
            model_info = {
                'class_name': model.__class__.__name__,
                'init_params': {
                    'input_shape': self.model_config['input_shape'],
                    'model_name': self.model_config['name']
                },
                'state_dict': model.state_dict(),
                'config': self.model_config,  # Save config for reference
                'epoch': epoch + 1
            }
            torch.save(model_info, self.filepath)

    def check_save(self, epoch: int, model: nn.Module, logs: dict[str, Any]) -> None:
        current = logs.get(self.monitor)
        if current is None:
            return
        
        should_save = False
        if self.save_best_only:
            if (self.mode == 'min' and current < self.best) or (self.mode == 'max' and current > self.best):
                if self.verbose:
                    print(f'\nEpoch {epoch+1}: {self.monitor} improved from {self.best:.5f} to {current:.5f}, saving model to {self.filepath}')
                if self.logger:
                    self.logger.log_val_improvement(epoch+1, self.monitor, self.best, current, self.filepath)
                self.best = current
                should_save = True
        else:
            if self.verbose:
                print(f'\nEpoch {epoch+1}: saving model to {self.filepath}')
            if self.logger:
                self.logger.log_message(f'Epoch {epoch+1}: saving model to {self.filepath}')
            should_save = True
            
        if should_save:
            self._save_model(model, epoch)

