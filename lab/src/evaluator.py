# Builtins
import os
# Installed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
# Local
# Types
from typing import (
    Dict,
    Tuple,
    Any
)



class ModelEvaluator:
    def __init__(
        self,
        model_config: Dict[str, Any]
    ):
        self.model_config = model_config

    # Evaluation metrics
    def _calculate_psnr(self, y_true: torch.Tensor, y_pred: torch.Tensor, max_val: float=1.0) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio between two images.
        
        Args:
            y_true: Ground truth tensor
            y_pred: Predicted tensor
            max_val: Maximum value of the signal
        
        Returns:
            PSNR value
        """
        # Convert to numpy if they are torch tensors
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        
        # Calculate MSE
        mse = np.mean((y_true - y_pred) ** 2)
        if mse == 0:  # If MSE is zero (perfect prediction)
            return float('inf')

        # Calculate PSNR
        return 20 * np.log10(max_val / np.sqrt(mse))

    def _calculate_mae(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """
        Calculate Mean Absolute Error between two images.
        
        Args:
            y_true: Ground truth tensor
            y_pred: Predicted tensor
        
        Returns:
            MAE value
        """
        # Convert to numpy if they are torch tensors
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        
        return np.mean(np.abs(y_true - y_pred))

    def _calculate_mse(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """
        Calculate Mean Squared Error between two images.
        
        Args:
            y_true: Ground truth tensor
            y_pred: Predicted tensor
        
        Returns:
            MSE value
        """
        # Convert to numpy if they are torch tensors
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        
        return np.mean((y_true - y_pred) ** 2)

    def _calculate_ssim(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        """
        Calculate Structural Similarity Index between two images.
        
        Args:
            y_true: Ground truth tensor
            y_pred: Predicted tensor
        
        Returns:
            SSIM value
        """
        # Convert to numpy if they are torch tensors
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        
        # Reshape if needed
        y_true = np.squeeze(y_true)
        y_pred = np.squeeze(y_pred)
        
        # Make sure inputs have the same shape
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
        
        # For 3D volumes, calculate SSIM for each 2D slice along the first dimension
        if len(y_true.shape) == 3:  # 3D volume
            ssim_values = []
            for i in range(y_true.shape[0]):
                slice_true = y_true[i]
                slice_pred = y_pred[i]
                # Data range is 1.0 for normalized images
                ssim_val = ssim(slice_true, slice_pred, data_range=1.0)
                ssim_values.append(ssim_val)
            return np.mean(ssim_values)
        else:  # 2D image
            return ssim(y_true, y_pred, data_range=1.0)

    def evaluate_prediction_quality(
        self,
        model,
        test_dataloader: DataLoader,
        device: str='cuda' if torch.cuda.is_available() else 'cpu'
    ) -> dict[str, Any]:
        """
        Evaluate a PyTorch model's predictions using multiple image quality metrics.
        
        Args:
            model: Trained PyTorch model
            test_dataloader: PyTorch DataLoader containing test data
            device: Device to run the model on ('cuda' or 'cpu')
        
        Returns:
            Dictionary of metrics
        """
        model.eval()  # Set model to evaluation mode
        model = model.to(device)
        
        psnr_values = []
        mae_values = []
        mse_values = []
        ssim_values = []
        
        with torch.no_grad():  # No gradients needed for evaluation
            for x_batch, y_batch in test_dataloader:
                # Move tensors to the correct device
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                
                # Generate predictions
                y_pred = model(x_batch)
                
                # Process each sample in the batch
                for i in range(y_batch.size(0)):
                    y_true_sample = y_batch[i]
                    y_pred_sample = y_pred[i]
                    
                    # Fix shape mismatch if needed
                    if y_true_sample.shape != y_pred_sample.shape:
                        if len(y_true_sample.shape) == 3 and len(y_pred_sample.shape) == 4:
                            y_true_sample = y_true_sample.unsqueeze(-1)
                        elif len(y_true_sample.shape) == 3 and len(y_pred_sample.shape) == 3 and y_pred_sample.shape[-1] == 1:
                            y_pred_sample = y_pred_sample.squeeze(-1)
                    
                    # Calculate metrics (convert tensors to numpy)
                    psnr = self._calculate_psnr(y_true_sample, y_pred_sample)
                    mae = self._calculate_mae(y_true_sample, y_pred_sample)
                    mse = self._calculate_mse(y_true_sample, y_pred_sample)
                    ssim_val = self._calculate_ssim(y_true_sample, y_pred_sample)
                    
                    # Store values
                    psnr_values.append(psnr)
                    mae_values.append(mae)
                    mse_values.append(mse)
                    ssim_values.append(ssim_val)
        
        # Calculate average metrics
        metrics = {
            'PSNR': np.mean(psnr_values),
            'MAE': np.mean(mae_values),
            'MSE': np.mean(mse_values),
            'SSIM': np.mean(ssim_values)
        }
        
        return metrics

    def plot_model_training_evolution(self) -> None:
        """
        Displays 4 graphs with data obtained throughout the model training:
            - Evolution of training error and validation error
            - Evolution of training MAE and validation MAE --> MAE = Mean Absolute Error
            - Evolution of training accuracy and validation accuracy
            - Graph with all metrics
        """
        train_history = pd.read_csv(os.path.join('..', 'models', self.model_config['name'], 'logs', 'model_train.csv'))

        # Get metrics from the model training history
        train_loss = train_history['loss'].tolist()
        val_loss = train_history['val_loss'].tolist()
        train_mae = train_history['mae'].tolist()
        val_mae = train_history['val_mae'].tolist()
        if train_history.get('learning_rate') is not None:
            learning_rate = train_history['learning_rate'].tolist()
        else:
            learning_rate = None

        plt.figure(figsize=(16, 10))

        # Show loss
        plt.subplot(2, 2, 1)
        plt.plot(train_loss, label='Loss entrenamiento', color='blue')
        plt.plot(val_loss, label='Loss validación', color='orange')
        plt.title('Error (loss) de entrenamiento y validación')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()

        # Show MAE
        plt.subplot(2, 2, 2)
        plt.plot(train_mae, label='MAE entrenamiento', color='blue')
        plt.plot(val_mae, label='MAE validación', color='orange')
        plt.title('Mean Absolute Error (MAE) de entrenamiento y validación')
        plt.xlabel('Epochs')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid()

        # Show learning rate (if soted)
        if learning_rate is not None:
            plt.subplot(2, 2, 3)
            plt.plot(learning_rate, label='Learning rate', color='blue')
            plt.title('Learning rate durante el entrenamiento')
            plt.xlabel('Epochs')
            plt.ylabel('LR')
            plt.legend()
            plt.grid()

        # Show all the metrics
        plt.subplot(2, 2, 4)
        history_df = pd.DataFrame(train_history.drop(columns=['epoch']))
        history_df.plot(ax=plt.gca())
        plt.title('Historial de métricas del modelo')
        plt.xlabel('Epochs')
        plt.ylabel('Valor')
        plt.legend(loc='upper right')
        plt.grid()

        plt.tight_layout()
        plt.show()

    def _calculate_voxel_difference(self, true_image: np.ndarray, pred_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the difference between true and predicted images and create a color-mapped visualization.
        
        Args:
            true_image: Ground truth tensor or numpy array
            pred_image: Predicted tensor or numpy array
        
        Returns:
            diff_image: Difference image
            color_image: Color-mapped visualization of differences
        """
        # Convert to numpy if they're torch tensors
        if isinstance(true_image, torch.Tensor):
            true_image = true_image.detach().cpu().numpy()
        if isinstance(pred_image, torch.Tensor):
            pred_image = pred_image.detach().cpu().numpy()
        
        print("---" * 40)
        total_diff = 0
        diff_image = np.zeros(true_image.shape)
        color_image = np.zeros((true_image.shape[0], true_image.shape[1], 3), dtype=np.uint8)  # For RGB color image
        
        for row in range(true_image.shape[0]):
            for col in range(true_image.shape[1]):
                true_value = true_image[row, col]
                pred_value = pred_image[row, col]
                difference = true_value - pred_value
                diff_image[row, col] = difference
                total_diff += difference
                # Calculate the absolute difference for color mapping
                abs_difference = abs(difference)
                # Normalize the difference to the range [0, 255]
                normalized_diff = np.clip(abs_difference * 255, 0, 255).astype(np.uint8)  # Clip to avoid overflow
                # Map the normalized difference to a color (red for high difference)
                color_image[row, col] = [0, 0, normalized_diff]  # BGR format for OpenCV
        
        print(total_diff)
        return diff_image, color_image

    # Get one batch of test data and visualize results
    def visualize_predictions(
        self,
        model,
        test_dataloader: DataLoader,
        device: str='cuda' if torch.cuda.is_available() else 'cpu'
    ) -> None:
        """
        Visualize model predictions and differences for one batch of test data.
        
        Args:
            model: Trained PyTorch model
            test_dataloader: PyTorch DataLoader for test data
            device: Device to run model on
        """
        # Set model to evaluation mode
        model.eval()
        model = model.to(device)
        
        with torch.no_grad():  # No gradients needed for evaluation
            # Get first batch from test dataloader
            for x_batch, y_batch in test_dataloader:
                print(x_batch.shape)

                # Move tensors to the correct device
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                
                # Generate predictions
                y_pred = model(x_batch)
                
                print(f"Input shape: {x_batch.shape}")
                print(f"Target shape: {y_batch.shape}")
                print(f"Prediction shape: {y_pred.shape}")
                
                # Move tensors back to CPU for visualization
                x_batch = x_batch.cpu()
                y_batch = y_batch.cpu()
                y_pred = y_pred.cpu()
                
                # Create figure for visualization
                fig, axes = plt.subplots(5, 3, figsize=(18, 20))
                
                # Determine if we're dealing with 3D data (assume 3D if there are 5 dimensions)
                is_3d_data = (len(x_batch.shape) == 5)
                
                # For 3D data, we'll visualize the first slice from each volume
                slice_idx = 0
                
                for j in range(3):
                    # Extract the jth sample from the batch
                    if is_3d_data:
                        # For 3D data, get the first slice
                        input_image = x_batch[j, slice_idx].squeeze()
                        target_image = y_batch[j, slice_idx].squeeze()
                        pred_image = y_pred[j, slice_idx].squeeze()
                    else:                    
                        # For 2D data
                        input_image = x_batch[j].squeeze()                    
                        target_image = y_batch[j].squeeze()
                        pred_image = y_pred[j].squeeze()

                    # Plot input image
                    axes[0][j].imshow(input_image[j], cmap='gray')
                    axes[0][j].set_title(f'Input Image {j+1}')
                    axes[0][j].axis('off')
                    
                    # Plot ground truth
                    axes[1][j].imshow(target_image[j], cmap='gray')
                    axes[1][j].set_title(f'Ground Truth {j+1}')
                    axes[1][j].axis('off')
                    
                    # Calculate metrics
                    psnr = self._calculate_psnr(target_image[j], pred_image[j])
                    mae = self._calculate_mae(target_image[j], pred_image[j])
                    ssim_val = self._calculate_ssim(target_image[j], pred_image[j])

                    # Plot prediction
                    axes[2][j].imshow(pred_image[j], cmap='gray')
                    axes[2][j].set_title(f'Prediction\nPSNR: {psnr:.2f}dB, MAE: {mae:.4f}, SSIM: {ssim_val:.4f}')
                    axes[2][j].axis('off')
                    
                    # Calculate difference and color map
                    diff_image, color_map_image = self._calculate_voxel_difference(target_image[j], pred_image[j])
                    
                    # Plot difference image
                    im1 = axes[3][j].imshow(diff_image, cmap='bwr', vmin=-0.5, vmax=0.5)
                    axes[3][j].set_title(f'Difference')
                    axes[3][j].axis('off')
                    
                    # Plot color map
                    im2 = axes[4][j].imshow(diff_image/target_image[j], 'bwr', vmin=-1, vmax=1)
                    axes[4][j].set_title(f'Relative differences image {j+1}')
                    axes[4][j].axis('off')

                cbar = fig.colorbar(im1, ax=axes[3], shrink=0.6)
                cbar.set_label('Diferencia', rotation=270, labelpad=15)
                cbar2 = fig.colorbar(im2, ax=axes[4], shrink=0.6)
                cbar2.set_label('Diferencia Relativa', rotation=270, labelpad=15)
                plt.show()
                
                # Do it just for the first batch
                break

