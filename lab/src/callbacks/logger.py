# Builtins
import os
import datetime
# Installed
import pandas as pd
# Local
# Types
from typing import (
    Dict,
    Any
)


class LogFileLogger:
    def __init__(
        self,
        model_config: Dict[str, Any],
        flush_interval: int=1
    ):
        self.model_config = model_config
        self.filepath = os.path.join(os.path.abspath('..'), 'models', self.model_config['name'], 'logs', 'model_train.log')
        self.file = None
        self.flush_interval = flush_interval  # How often to flush to disk (in batches)
        self.batch_counter = 0
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.filepath) if os.path.dirname(self.filepath) else '.', exist_ok=True)

        print(self.__str__())

        # Open the log file and write the header
        self.open_file()
        self.log_message(self.__str__())

    def open_file(self) -> None:
        self.file = open(self.filepath, 'w')
        self.file.write("# Training Log\n")
        self.file.write("# Format: [timestamp] epoch/total_epochs - time - metrics\n\n")
        self.file.flush()  # Ensure header is written immediately

    def log_batch(
        self,
        epoch: int,
        batch: int,
        total_batches: int,
        train_loss: float,
        train_mae: float,
        learning_rate: float
    ) -> None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] Batch {batch}/{total_batches} of Epoch {epoch}/{self.model_config['num_epochs']} - "
        log_message += f"loss: {train_loss:.4f} - mae: {train_mae:.4f} - lr: {learning_rate:.6f}\n"
        
        self.file.write(log_message)
        
        # Flush to disk periodically to ensure logs are saved even if the program crashes
        self.batch_counter += 1
        if self.batch_counter % self.flush_interval == 0:
            self.file.flush()
    
    def log_epoch(
        self,
        epoch: int,
        time_taken: float,
        train_loss: float,
        train_mae: float,
        val_loss: float,
        val_mae: float,
        learning_rate: float
    ) -> None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] Epoch {epoch}/{self.model_config['num_epochs']} - {time_taken:.2f}s - "
        log_message += f"loss: {train_loss:.4f} - mae: {train_mae:.4f} - val_loss: {val_loss:.4f} - val_mae: {val_mae:.4f} - lr: {learning_rate:.6f}\n"
        
        # Add a separator line after each epoch for better readability
        separator = "-" * 100 + "\n"
        
        self.file.write(log_message)
        self.file.write(separator)
        self.file.flush()
    
    def log_early_stopping(self, counter: int, patience: int) -> None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] EarlyStopping counter: {counter} out of {patience}\n"
        self.file.write(log_message)
        self.file.flush()
    
    def log_lr_reduction(self, old_lr: float, new_lr: float) -> None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] Reducing learning rate from {old_lr:.6f} to {new_lr:.6f}\n"
        self.file.write(log_message)
        self.file.flush()
    
    def log_val_improvement(
        self,
        epoch: int,
        metric_name: str,
        old_value: float,
        new_value: float,
        checkpoint_path: str
    ) -> None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] Epoch {epoch}: {metric_name} improved from {old_value:.5f} to {new_value:.5f}, saving model to {checkpoint_path}\n"
        self.file.write(log_message)
        self.file.flush()
    
    def log_message(self, message: str) -> None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        self.file.write(log_message)
        self.file.flush()
    
    def close(self):
        if self.file:
            end_message = f"\nTraining completed at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            self.file.write(end_message)
            self.file.close()
    
    def __del__(self):
        self.close()  # Ensure the file is closed when the object is deleted
    
    def __str__(self):
        return f"LogFileLogger(filepath={self.filepath}, flush_interval={self.flush_interval})"
    



# CSV Logger equivalent
class CSVLogger:
    def __init__(
        self,
        model_config: Dict[str, Any],
        logger: LogFileLogger
    ):
        self.model_config = model_config
        self.filepath = os.path.join(os.path.abspath('..'), 'models', self.model_config['name'], 'logs', 'model_train.csv')
        self.logger = logger
        self.data = []

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.filepath) if os.path.dirname(self.filepath) else '.', exist_ok=True)

        print(self.__str__())
        self.logger.log_message(self.__str__())

    def __str__(self):
        return (f"CSVLogger(filepath={self.filepath})")
        
    def log(self, epoch: int, logs: dict[str, Any]) -> None:
        self.data.append({**{'epoch': epoch}, **logs})
        
    def save(self) -> None:
        df = pd.DataFrame(self.data)
        df.to_csv(self.filepath, index=False)

