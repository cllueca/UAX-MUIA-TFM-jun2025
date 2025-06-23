# Local
from .logger import LogFileLogger

# Custom reduce learning rate on plateau
class ReduceLROnPlateau:
    def __init__(
        self,
        optimizer,
        model_config,
        mode: str='min',
        verbose: bool=False,
        logger: LogFileLogger=None,
    ):
        self.optimizer = optimizer
        self.model_config = model_config
        self.mode = mode
        self.factor = self.model_config['lr_reduce_factor']
        self.patience = self.model_config['reduce_lr_patience']
        self.min_lr = self.model_config['min_lr']
        self.verbose = verbose
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.cooldown_counter = 0
        self.waiting = 0
        self.logger = logger

        print(self.__str__())
        self.logger.log_message(self.__str__())

    def __str__(self):
        return (f"ReduceLROnPlateau(optimizer={self.optimizer}, mode={self.mode}, "
            f"factor={self.factor}, patience='{self.patience}', "
            f"verbose={self.verbose}, min_lr={self.min_lr})")

    def step(self, metrics: float) -> None:
        current = metrics
        
        if (self.mode == 'min' and current < self.best) or (self.mode == 'max' and current > self.best):
            self.best = current
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            for param_group in self.optimizer.param_groups:
                old_lr = param_group['lr']
                new_lr = max(old_lr * self.factor, self.min_lr)
                param_group['lr'] = new_lr
                if self.verbose:
                    print(f'Reducing learning rate from {old_lr:.6f} to {new_lr:.6f}')
                if self.logger:
                    self.logger.log_lr_reduction(old_lr, new_lr)
            self.counter = 0

