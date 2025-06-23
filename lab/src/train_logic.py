# Builtins
import time
# Installed
from tqdm import tqdm
import torch
import torch.nn as nn
# Local
from .dataloader import TorchDataLoader
from .callbacks.earlystopping import EarlyStopping
from .callbacks.reducelr import ReduceLROnPlateau
from .callbacks.modelcheckpoint import ModelCheckpoint
from .callbacks.logger import (
    LogFileLogger,
    CSVLogger
)
# Types
from typing import (
    Dict,
    List,
    Any
)


def train_model(
    model: nn.Module,
    train_loader: TorchDataLoader,
    val_loader: TorchDataLoader,
    optimizer: torch.optim.Optimizer, 
    criterion: nn.Module,
    device: torch.device,
    config: Dict,
    callbacks: List = None
) -> dict[str, Any]:
    """Defines the logic durint the training phase of any NN"""
    model.to(device)
    
    # Initialize callbacks
    early_stopping = None
    reduce_lr = None
    checkpoint = None
    csv_logger = None
    log_file_logger = None
    if callbacks:
        for callback in callbacks:
            if isinstance(callback, EarlyStopping):
                early_stopping = callback
            elif isinstance(callback, ReduceLROnPlateau):
                reduce_lr = callback
            elif isinstance(callback, ModelCheckpoint):
                checkpoint = callback
            elif isinstance(callback, CSVLogger):
                csv_logger = callback
            elif isinstance(callback, LogFileLogger):
                log_file_logger = callback
    
    history = {
        'train_loss': [],
        'train_mae': [],
        'val_loss': [],
        'val_mae': []
    }

    # Log training start
    if log_file_logger:
        log_file_logger.log_message(f"Starting training with {config['num_epochs']} epochs")
        log_file_logger.log_message(f"Model: {config['name']}")
        log_file_logger.log_message(f"Device: {device}")
        log_file_logger.log_message(f"Training samples: {len(train_loader.dataset)}")
        log_file_logger.log_message(f"Validation samples: {len(val_loader.dataset)}")
        log_file_logger.log_message("-" * 100)
    
    # Training loop
    for epoch in range(config['num_epochs']):
        model.train()
        running_loss = 0.0
        running_mae = 0.0
        
        epoch_start = time.time()
        total_batches = len(train_loader)
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Training phase
        for i, (inputs, targets) in progress_bar:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            mae = torch.mean(torch.abs(outputs - targets))

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            running_mae += mae.item()

            # Calculate batch metrics
            batch_loss = running_loss / (i + 1)
            batch_mae = running_mae / (i + 1)

            # Update progress bar
            progress_bar.set_postfix({
                'loss': running_loss / (i + 1),
                'mae': running_mae / (i + 1),
                'lr': current_lr
            })

            # Log batch metrics
            if log_file_logger and (i % 10 == 0 or i == total_batches - 1):  # Log every 10 batches or the last batch
                log_file_logger.log_batch(
                    epoch + 1,
                    i + 1,
                    total_batches,
                    batch_loss,
                    batch_mae,
                    current_lr
                )
        
        # Calculate training metrics
        train_loss = running_loss / len(train_loader)
        train_mae = running_mae / len(train_loader)
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_mae = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                mae = torch.mean(torch.abs(outputs - targets))
                
                val_running_loss += loss.item()
                val_running_mae += mae.item()
        
        # Calculate validation metrics
        val_loss = val_running_loss / len(val_loader)
        val_mae = val_running_mae / len(val_loader)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_mae'].append(train_mae)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start

        # Print epoch summary
        print(f'Epoch {epoch+1}/{config["num_epochs"]} - {epoch_end - epoch_start:.2f}s - loss: {train_loss:.4f} - mae: {train_mae:.4f} - val_loss: {val_loss:.4f} - val_mae: {val_mae:.4f} - learning_rate: {current_lr}')
        
        # Log epoch metrics
        if log_file_logger:
            log_file_logger.log_epoch(
                epoch + 1,
                epoch_time,
                train_loss,
                train_mae,
                val_loss,
                val_mae,
                current_lr
            )

        # Handle callbacks
        logs = {
            'loss': train_loss,
            'mae': train_mae,
            'val_loss': val_loss,
            'val_mae': val_mae,
            'learning_rate': current_lr
        }
        
        if csv_logger:
            csv_logger.log(epoch, logs)
            csv_logger.save()  # Save after each epoch
            
        if checkpoint:
            checkpoint.check_save(epoch, model, logs)
            
        if reduce_lr:
            reduce_lr.step(val_loss)
            
        if early_stopping:
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                if log_file_logger:
                    log_file_logger.log_message("Early stopping triggered")
                if early_stopping.restore_best_weights:
                    early_stopping.restore_model(model)
                break

    # Log training completion
    if log_file_logger:
        log_file_logger.log_message(f"Training completed after {epoch+1} epochs")
        log_file_logger.close()
    
    return history
