# Builtins
import os
import json
from datetime import datetime
from pathlib import Path
# Installed
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    Dataset
)
# Local
from .utils.config_files import save_config
# Types
from typing import (
    Union,
    Any,
    Dict,
    Tuple,
    List
)


# Custom dataset for PyTorch
class PETDataset(Dataset):
    def __init__(self, inputs, targets):
        """
        Initialize the PETDataset class.
        
        Args:
            inputs (list or torch.Tensor): Input data for the dataset
            targets (list or torch.Tensor): Target labels corresponding to the input data
        """
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        """
        Return the number of samples in the dataset.
        
        Returns:
            int: The total number of samples in the dataset
        """
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        Retrieve a sample and its corresponding target from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
        
        Returns:
            tuple: A tuple containing the input tensor and the target tensor
        """
        # If inputs/targets are already torch tensors, return them directly
        if isinstance(self.inputs, torch.Tensor) and isinstance(self.targets, torch.Tensor):
            return self.inputs[idx], self.targets[idx]
        else:
            input_tensor = torch.tensor(self.inputs[idx], dtype=torch.float32)
            target_tensor = torch.tensor(self.targets[idx], dtype=torch.float32)
            return input_tensor, target_tensor

        

class TorchDataLoader:
    def __init__(
        self,
        data_dir_path: Union[Path | str],
        data_config: Dict[str, Any],
        model_config: Dict[str, Any],
        log_filename: str="data_loader.log"
    ):
        """
        Initialize the TorchDataLoader class.
        
        Args:
            data_dir_path (str): Path to the data directory
            data_config (dict): Configuration containing data parameters
            model_config (dict): Configuration containing model parameters like batch_size
            log_filename (str): Path to the log file
        """
        self.model_config = model_config
        self.data_dir_path = data_dir_path
        self.data_config = data_config
        self.log_path = os.path.join(os.path.abspath('..'), 'models', self.model_config['name'], 'logs', log_filename)
        self.data_config_path = os.path.join(os.path.abspath('..'), 'models', self.model_config['name'], 'config', 'data_config.json')
        self.model_config_path = os.path.join(os.path.abspath('..'), 'models', self.model_config['name'], 'config', 'model_config.json')
        
        if self.data_config['axis_cut'] not in ['axial', 'coronal', 'sagittal']:
            raise Exception("Axis cut is not recognized. Available options: [axial, coronal, sagittal]")
        
        if self.data_config['first_cut'] < 0 or self.data_config['first_cut'] > 126:
            raise Exception("The index of the first image layer must be between 0 and 126")
        
        if self.data_config['last_cut'] < 1 or self.data_config['last_cut'] > 127:
            raise Exception("The index of the last image layer must be between 1 and 127")

        if self.data_config['layer_qty'] > 128:
            raise Exception("The number of layers to load must be 128 o less")

        # Ensure directories exists
        os.makedirs(os.path.dirname(self.log_path) if os.path.dirname(self.log_path) else '.', exist_ok=True)
        os.makedirs(os.path.dirname(self.data_config_path) if os.path.dirname(self.data_config_path) else '.', exist_ok=True)
        os.makedirs(os.path.dirname(self.model_config_path) if os.path.dirname(self.model_config_path) else '.', exist_ok=True)

        # Save configs
        save_config(self.data_config_path, self.data_config, calling_class=self)
        save_config(self.model_config_path, self.model_config, calling_class=self)
        

    def _log(self, message: str) -> None:
        """
        Write a message to the log file with timestamp.
        
        Args:
            message (str): Message to log
        """
        print(message)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {message}\n")
 

    def _load_nifti_image(self, image_path: Union[Path | str]) -> np.array:
        """
        Load a NIfTI image and return it as a numpy array.
        
        Args:
            image_path (str): Path to the NIfTI file to load.
        
        Returns:
            tuple: A tuple containing:
                - img_data (numpy.ndarray): The image data array.
                - affine (numpy.ndarray): The affine transformation matrix.
                - header (nibabel.Nifti1Header): The NIfTI header information.
        
        Raises:
            FileNotFoundError: If the file_path does not exist.
            nibabel.filebasedimages.ImageFileError: If the file is not a valid NIfTI image.
            Exception: For any other errors during loading.
        """

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"The specified file does not exist: {image_path}")

        try:
            img = nib.load(image_path)
            return img.get_fdata(), img.affine, img.header
        except nib.filebasedimages.ImageFileError as e:
            raise nib.filebasedimages.ImageFileError(f"Error loading NIfTI image: {e}")
        except Exception as e:
            raise Exception(f"Unexpected error loading NIfTI image: {e}")


    def _all_batches_same_size(self, dataloader: DataLoader) -> None:
        """
            Check if all batches in the dataloader have the same size.

            Args:
                dataloader (DataLoader): PyTorch DataLoader to check.

            Returns:
                bool: True if all batches have the same size, False otherwise.
        """
        expected_batch_size = None
        inputs_shape = None
        targets_shape = None
        are_batches_equal = True

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            batch_size = inputs.size(0)
            if expected_batch_size is None:
                expected_batch_size = batch_size
                inputs_shape = inputs.shape
                targets_shape = targets.shape
            else:
                if batch_size != expected_batch_size:
                    self._log(f"Batch {batch_idx + 1} size {batch_size} != expected size {expected_batch_size}")
                    are_batches_equal = False

        self._log(f"Common batch shape (num_batches / batch_size, channels, depth / num_layers, height, width):")
        self._log(f" - Inputs: {inputs_shape}")
        self._log(f" - Targets: {targets_shape}")
            
        if are_batches_equal:
            self._log(f"All batches have the same size")


    def _get_layer_axis(
        self,
        image_data: np.array,
        cut_index: int
    ) -> np.array:
        """
        Extract a specific layer from a 3D image based on the specified axis and index.
        
        Args:
            image_data (np.array): A 3D numpy array representing the image data.
            cut_index (int): The index of the layer to extract along the specified axis.
        
        Returns:
            np.array: A 2D numpy array representing the extracted layer.
        
        Raises:
            Exception: If the cut_axis is not recognized. Possible values are 'axial', 'coronal', 'sagittal'.
        """
        if self.data_config['axis_cut'] == 'axial':
            return image_data[:, :, cut_index]  # Extract the axial layer
        elif self.data_config['axis_cut'] == 'coronal':
            return image_data[:, cut_index, :]  # Extract the coronal layer
        elif self.data_config['axis_cut'] == 'sagittal':
            return image_data[cut_index, :, :]  # Extract the sagittal layer
        else:
            raise Exception("Axis cut is not recognized. Possible values: [axial, coronal, sagittal]")


    def _load_data(
        self,
        is_test: bool=False
    ) -> Tuple[List[np.array], List[np.array]]:
        """
        Load images and labels from the specified directory structure.
        
        Args:
            is_test (bool): Whether the dataset to load is for trainig or test purposes
                Default: False

        Returns:
            Tuple[List[np.array], List[np.array]]: List with the input images and list with the label images 
        """
        self._log(f" ==  Loading {'test' if is_test else 'train'} images set  ==")
        
        images = []
        labels = []
        max_pixel_value = 0
        min_pixel_value = 100

        fold_path = os.path.join(self.data_dir_path, 'fold1')

        # Load training images and labels
        if is_test:
            images_path = os.path.join(fold_path, 'ImagesTs')
            labels_path = os.path.join(fold_path, 'LabelsTs')
        else:
            images_path = os.path.join(fold_path, 'ImagesTr')
            labels_path = os.path.join(fold_path, 'LabelsTr')

        self._log(f"\t- Loading {len(os.listdir(images_path))} images from '{fold_path}' ...")
        outlier_detected = 0

        for img_file in os.listdir(images_path):
            if img_file.endswith('.nii.gz') and not (img_file.startswith('05') or img_file.startswith('14')):
                img_path = os.path.join(images_path, img_file)
                label_path = os.path.join(labels_path, img_file.replace('_NAC', ''))

                # Load images and labels
                image_data, image_aff, image_hdr = self._load_nifti_image(img_path)
                label_data, label_aff, label_hdr = self._load_nifti_image(label_path)

                # Create empty np matrix
                image_stack = np.empty((0, 128, 128), dtype=np.float32)
                label_stack = np.empty((0, 128, 128), dtype=np.float32)
                layer_count = 0

                # Append to lists
                for idx in range(len(image_data)):
                    # Get only the selected range of cuts
                    if idx >= self.data_config['first_cut'] and idx <= self.data_config['last_cut']:
                        # Randomness on cut axis not implemented (should be decided here so the image and the label have the same cut)
                        image_stack = np.append(image_stack, [self._get_layer_axis(image_data=image_data, cut_index=idx)], axis=0)
                        label_stack = np.append(label_stack, [self._get_layer_axis(image_data=label_data, cut_index=idx)], axis=0)

                        # # Find the maximum value
                        max_value = np.max(image_data[idx])
                        if max_value > max_pixel_value:
                            max_pixel_value = max_value

                        # Find the minimum value
                        min_value = np.min(image_data[idx])
                        if min_value < min_pixel_value:
                            min_pixel_value = min_value

                        layer_count += 1

                        # Until the cut limit is reached, then create a new set
                        if layer_count == self.data_config['layer_qty']:
                            layer_count = 0
                            images.append(image_stack)
                            labels.append(label_stack)
                            image_stack = np.empty((0, 128, 128), dtype=np.float32)
                            label_stack = np.empty((0, 128, 128), dtype=np.float32)
            else:
                outlier_detected += 1

        self._log(f"\t- Could only load {len(os.listdir(images_path)) - outlier_detected} images. Detected {outlier_detected} outlier(s)")
            
        # Convert lists to numpy arrays
        images = np.array(images)
        labels = np.array(labels)

        self._log(f"\t- Pixels value range: [{round(min_pixel_value, 2)}, {round(max_pixel_value, 2)}]")

        # Normalize images (optional)
        if self.data_config['normalize']:
            self._log("\t- Normalizing the data to the range [0, 1]...")
            images = images / np.max(images)

        return images, labels


    def load_train_data(self) -> DataLoader:
        """
        Load and prepare training data.
        
        Returns:
            DataLoader: Training data loader
        """
        self._log("Loading training data...")
        train_inputs, train_targets = self._load_data()
        
        # Add channel dimension (at position 1)
        train_inputs = np.expand_dims(train_inputs, axis=1)
        train_targets = np.expand_dims(train_targets, axis=1)
        
        self._log(f"Train inputs shape: {train_inputs.shape}")
        self._log(f"Train targets shape: {train_targets.shape}")
        
        # Convert numpy arrays to torch tensors
        train_inputs = torch.from_numpy(train_inputs).float()
        train_targets = torch.from_numpy(train_targets).float()
        
        # Create PyTorch dataset and dataloader
        train_dataset = PETDataset(train_inputs, train_targets)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.model_config['batch_size'], 
            shuffle=True,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        self._log(f"Training loader generated. Batch size: {self.model_config['batch_size']}")
        self._log("== Train set loader ==")
        self._all_batches_same_size(train_loader)
        
        return train_loader
    

    def load_validation_data(self) -> DataLoader:
        """
        Load and prepare validation data (second half of test set).
        
        Returns:
            DataLoader: Validation data loader
        """
        self._log("Loading test data for validation...")
        images_test, labels_test = self._load_data(is_test=True)
        
        self._log("Splitting test set in half, the second half will be used for validation...")
        val_inputs = images_test[round(len(images_test)/2):]
        val_targets = labels_test[round(len(images_test)/2):]
        
        # Add channel dimension (at position 1)
        val_inputs = np.expand_dims(val_inputs, axis=1)
        val_targets = np.expand_dims(val_targets, axis=1)
        
        self._log(f"Validation inputs shape: {val_inputs.shape}")
        self._log(f"Validation targets shape: {val_targets.shape}")
        
        # Convert numpy arrays to torch tensors
        val_inputs = torch.from_numpy(val_inputs).float()
        val_targets = torch.from_numpy(val_targets).float()
        
        # Create PyTorch dataset and dataloader
        val_dataset = PETDataset(val_inputs, val_targets)
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.model_config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        self._log(f"Validation loader generated. Batch size: {self.model_config['batch_size']}")
        self._log("== Validation set loader ==")
        self._all_batches_same_size(val_loader)
        
        return val_loader
    
    
    def load_test_data(self) -> DataLoader:
        """
        Load and prepare test data (first half of test set).
        
        Returns:
            DataLoader: Test data loader
        """
        self._log("Loading test data...")
        images_test, labels_test = self._load_data(is_test=True)
        
        self._log("Using first half of test set for testing...")
        test_inputs = images_test[:round(len(images_test)/2)]
        test_targets = labels_test[:round(len(images_test)/2)]
        
        # Add channel dimension (at position 1)
        test_inputs = np.expand_dims(test_inputs, axis=1)
        test_targets = np.expand_dims(test_targets, axis=1)
        
        self._log(f"Test inputs shape: {test_inputs.shape}")
        self._log(f"Test targets shape: {test_targets.shape}")
        
        # Convert numpy arrays to torch tensors
        test_inputs = torch.from_numpy(test_inputs).float()
        test_targets = torch.from_numpy(test_targets).float()
        
        # Create PyTorch dataset and dataloader
        test_dataset = PETDataset(test_inputs, test_targets)
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.model_config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        self._log(f"Test loader generated. Batch size: {self.model_config['batch_size']}")
        self._log("== Test set loader ==")
        self._all_batches_same_size(test_loader)
        
        return test_loader
    
