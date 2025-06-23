# Builtins
import os
import math
from datetime import datetime
# Installed
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchview
from torchview import draw_graph
from torchinfo import summary
# Local
from .utils.config_files import save_config
# Types
from typing import (
    Tuple,
    Dict,
    Any
)


class UNet3D(nn.Module):
    def __init__(
        self,
        model_config: Dict[str, Any],
        log_filename: str='model_construction.log'
    ):
        # Inherit from nn.Module, which is the base class for all neural network modules in PyTorch and initialize the model
        super().__init__()
        # Set model name
        self.model_config = model_config
        self.log_path = os.path.join(os.path.abspath('..'), 'models', self.model_config['name'], 'logs', log_filename)
        self.model_config_path = os.path.join(os.path.abspath('..'), 'models', self.model_config['name'], 'config', 'model_config.json')
        self.model_graph_path = os.path.join(os.path.abspath('..'), 'models', self.model_config['name'], 'graph')

        # Ensure directories exists
        os.makedirs(os.path.dirname(self.log_path) if os.path.dirname(self.log_path) else '.', exist_ok=True)
        os.makedirs(os.path.dirname(self.model_config_path) if os.path.dirname(self.model_config_path) else '.', exist_ok=True)
        os.makedirs(os.path.dirname(self.model_graph_path) if os.path.dirname(self.model_graph_path) else '.', exist_ok=True)

        # Save configs
        save_config(self.model_config_path, self.model_config, calling_class=self)

        # Build NN
        # Extract input channels from shape (input_shape is expected to be (C, D, H, W))
        self.in_channels = self.model_config["input_shape"][0] if self.model_config["input_shape"] else 1

        self.n_blocks = self._check_min_feature_dim(self.model_config["input_shape"])

        if self.n_blocks > len(self.model_config['layer_filters'])-1:
            raise Exception("Need more filter values or a bigger dimensions for the feature map")

        if self.n_blocks > len(self.model_config['layer_dropouts'])-1:
            raise Exception("Need more dropout values or a bigger dimensions for the feature map")
        
        self.conv_encoder_list = nn.ModuleList()
        self.pool_encoder_list = nn.ModuleList()
        self.up_decoder_list = nn.ModuleList()
        self.conv_decoder_list = nn.ModuleList()
        
        """Encoder"""
        for i in range(self.n_blocks):
            if i == 0:
                enc_conv, enc_pool = self._encoder_block(
                    self.in_channels,
                    self.model_config['layer_filters'][i],
                    self.model_config['layer_dropouts'][i]
                )
            else:
                enc_conv, enc_pool = self._encoder_block(
                    self.model_config['layer_filters'][i-1],
                    self.model_config['layer_filters'][i],
                    self.model_config['layer_dropouts'][i]
                )
            self.conv_encoder_list.append(enc_conv)
            self.pool_encoder_list.append(enc_pool)

        """Bridge (bottleneck)"""
        self.bridge = self._conv_block(
            self.model_config['layer_filters'][self.n_blocks-1],
            self.model_config['layer_filters'][self.n_blocks],
            self.model_config['layer_dropouts'][self.n_blocks]
        )

        """Decoder"""
        for i in range(self.n_blocks-1, -1, -1):
            dec_up = self._decoder_block(
                self.model_config['layer_filters'][i+1],
                self.model_config['layer_filters'][i]
            )
            dec_conv = self._conv_block(
                self.model_config['layer_filters'][i+1],
                self.model_config['layer_filters'][i],
                self.model_config['layer_dropouts'][i]
            )
            self.up_decoder_list.append(dec_up)
            self.conv_decoder_list.append(dec_conv)
        
        """Output layer"""
        # The final layer is a 3D convolution that reduces the output to a single channel,
        #  followed by a sigmoid activation function to produce probabilities for binary segmentation tasks.
        self.outputs = nn.Conv3d(self.model_config['layer_filters'][0], self.in_channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def _log(self, message: str) -> None:
        """Write a message to the log file with timestamp."""
        print(message)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {message}\n")

    def _count_parameters(self, model) -> int:
        """Count the number of parameters of the model"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _check_min_feature_dim(self, input_shape: Tuple) -> int:
        """Get the number of necessary convolution layers to get to the desired feature map dimensions on the bridge layer"""
        # Get width and height
        image_dimensions = (input_shape[-1], input_shape[-2])
        # Get the reduction ratio to obtain a MODEL_CONFIG['min_feature_map_dim']xMODEL_CONFIG['min_feature_map_dim'] image on the bridge layer
        ratio = np.min(image_dimensions) / self.model_config['min_feature_map_dim']
        # Get the number of encoder blocks to get there (same ammount for decoder blocks)
        n_conv = int(math.log(ratio, 2))
        self._log(f"Using {n_conv} blocks based on input dimensions {image_dimensions} and the desired feature map dimensions ({self.model_config['min_feature_map_dim']}x{self.model_config['min_feature_map_dim']})\n")
        return n_conv
    
    def _conv_block(self, in_channels: int, out_channels: int, dropout_value: float) -> nn.Sequential:
        """Convolutional block with BatchNormalization, Dropout, and ReLU activations"""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.Dropout3d(dropout_value),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _encoder_block(self, in_channels: int, out_channels: int, dropout_value: float) -> Tuple[nn.Sequential, nn.MaxPool3d]:
        """Encoder block that returns both the pre-pooled feature maps and the pooled output"""
        conv = self._conv_block(in_channels, out_channels, dropout_value)
        pool = nn.MaxPool3d(kernel_size=(1, 2, 2))
        return conv, pool

    def _decoder_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Decoder block with upsampling, skip connection concatenation, and convolutions"""
        up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0)
        return nn.Sequential(
            up,
            # conv_block will be applied after concat in the forward method
        )
        
    def display_summary(self) -> None:
        """Display model summary (equivalent to model.summary() in Keras)"""
        input_size = (self.model_config["batch_size"], *self.model_config["input_shape"])
        self._log(summary(self, input_size=input_size, depth=4, verbose=2))
        self._log(f"Total trainable parameters: {self._count_parameters(self):,}")

    def view_model_graph(self, browser_view: bool=False) -> torchview.computation_graph.ComputationGraph:
        """View the network architechture as a graph. It can be displayed in a web browser or directly in the notebook"""
        # sudo apt install graphviz (linux systems)
        model_graph = draw_graph(
            self,
            input_size=self.model_config['input_shape_graph'],
            expand_nested=True,
            depth=1,
            directory=self.model_graph_path
        )
        if browser_view:
            model_graph.visual_graph.render(self.model_config['name'], view=True)
        # model_graph.visual_graph --> use this on returned object
        return model_graph

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines how the input tensor 'x' flows through the network"""
        """
            Encoder Path
                + The input is passed through the encoder blocks, and the outputs are stored in skip variables for later concatenation in the decoder.
        
            Bridge
                + The output from the encoder is processed through the bridge.
        
            Decoder Path
                + The output is upsampled and concatenated with the corresponding skip connections from the encoder, followed by convolutional layers.

            Output
                + Finally, the output is passed through the output layer and activated with a sigmoid function.
        
            The forward method processes the input tensor through the network layers and produces the output.
        """
        skips_list = []
        # Encoder
        for i in range(len(self.conv_encoder_list)):
            skip = self.conv_encoder_list[i](x)
            skips_list.append(skip)
            x = self.pool_encoder_list[i](skip)
        
        # Bridge
        x = self.bridge(x)
        
        # Decoder
        for i in range(len(self.conv_decoder_list)):
            x = self.up_decoder_list[i](x)
            x = torch.cat([x, skips_list[-(i + 1)]], dim=1)
            x = self.conv_decoder_list[i](x)
        
        # Output
        x = self.outputs(x)
        x = self.sigmoid(x)
        
        return x
    

class AttentionGate3D(nn.Module):
    """3D Attention Gate module for focusing on relevant features"""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate3D, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi
  

class AttentionUnet3D(nn.Module):
    """U-Net 3D with attention gates"""
    def __init__(
      self,
      model_config: Dict[str, Any],
      log_filename: str='model_construction.log'      
    ):
        super().__init__()
        self.model_config = model_config
        self.log_path = os.path.join(os.path.abspath('..'), 'models', self.model_config['name'], 'logs', log_filename)
        self.model_config_path = os.path.join(os.path.abspath('..'), 'models', self.model_config['name'], 'config', 'model_config.json')
        self.model_graph_path = os.path.join(os.path.abspath('..'), 'models', self.model_config['name'], 'graph')

        # Ensure directories exists
        os.makedirs(os.path.dirname(self.log_path) if os.path.dirname(self.log_path) else '.', exist_ok=True)
        os.makedirs(os.path.dirname(self.model_config_path) if os.path.dirname(self.model_config_path) else '.', exist_ok=True)
        os.makedirs(os.path.dirname(self.model_graph_path) if os.path.dirname(self.model_graph_path) else '.', exist_ok=True)

        # Save configs
        save_config(self.model_config_path, self.model_config, calling_class=self)
        
        self.in_channels = self.model_config["input_shape"][0] if self.model_config["input_shape"] else 1
        self.n_blocks = self._check_min_feature_dim(self.model_config["input_shape"])
        
        # Build encoder
        self.conv_encoder_list = nn.ModuleList()
        self.pool_encoder_list = nn.ModuleList()

        for i in range(self.n_blocks):
            if i == 0:
                enc_conv, enc_pool = self._encoder_block(
                    self.in_channels,
                    self.model_config['layer_filters'][i],
                    self.model_config['layer_dropouts'][i]
                )
            else:
                enc_conv, enc_pool = self._encoder_block(
                    self.model_config['layer_filters'][i-1],
                    self.model_config['layer_filters'][i],
                    self.model_config['layer_dropouts'][i]
                )
            self.conv_encoder_list.append(enc_conv)
            self.pool_encoder_list.append(enc_pool)

        # Bridge
        self.bridge = self._conv_block(
            self.model_config['layer_filters'][self.n_blocks-1],
            self.model_config['layer_filters'][self.n_blocks],
            self.model_config['layer_dropouts'][self.n_blocks]
        )

        # Build decoder with attention gates
        self.up_decoder_list = nn.ModuleList()
        self.conv_decoder_list = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        
        for i in range(self.n_blocks-1, -1, -1):
            # Upsampling
            dec_up = self._decoder_block(
                self.model_config['layer_filters'][i+1],
                self.model_config['layer_filters'][i]
            )
            
            # Attention gate
            attention_gate = AttentionGate3D(
                F_g=self.model_config['layer_filters'][i],  # gating signal
                F_l=self.model_config['layer_filters'][i],  # skip connection
                F_int=self.model_config['layer_filters'][i] // 2
            )
            
            # Decoder conv
            dec_conv = self._conv_block(
                self.model_config['layer_filters'][i+1],
                self.model_config['layer_filters'][i],
                self.model_config['layer_dropouts'][i]
            )
            
            self.up_decoder_list.append(dec_up)
            self.attention_gates.append(attention_gate)
            self.conv_decoder_list.append(dec_conv)
        
        # Output layer
        self.outputs = nn.Conv3d(self.model_config['layer_filters'][0], self.in_channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def _log(self, message: str) -> None:
        """Write a message to the log file with timestamp."""
        print(message)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {message}\n")

    def _count_parameters(self, model) -> int:
        """Count the number of parameters of the model"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _check_min_feature_dim(self, input_shape: Tuple) -> int:
        """Get the number of necessary convolution layers to get to the desired feature map dimensions on the bridge layer"""
        image_dimensions = (input_shape[-1], input_shape[-2])
        ratio = np.min(image_dimensions) / self.model_config['min_feature_map_dim']
        n_conv = int(math.log(ratio, 2))
        self._log(f"Using {n_conv} blocks for Attention U-Net based on input dimensions {image_dimensions}\n")
        return n_conv

    def _conv_block(self, in_channels: int, out_channels: int, dropout_value: float) -> nn.Sequential:
        """Convolutional block with BatchNormalization, Dropout, and ReLU activations"""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.Dropout3d(dropout_value),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _encoder_block(self, in_channels: int, out_channels: int, dropout_value: float) -> Tuple[nn.Sequential, nn.MaxPool3d]:
        """Encoder block that returns both the pre-pooled feature maps and the pooled output"""
        conv = self._conv_block(in_channels, out_channels, dropout_value)
        pool = nn.MaxPool3d(kernel_size=(1, 2, 2))
        return conv, pool

    def _decoder_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Decoder block with upsampling, skip connection concatenation, and convolutions"""
        up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0)
        return nn.Sequential(up)

    def display_summary(self) -> None:
        """Display model params"""
        input_size = (self.model_config["batch_size"], *self.model_config["input_shape"])
        self._log(f"Attention U-Net Model Summary:")
        self._log(f"Total trainable parameters: {self._count_parameters(self):,}")

    def view_model_graph(self, browser_view: bool=False) -> torchview.computation_graph.ComputationGraph:
        """View the network architecture as a graph. It can be displayed in a web browser or directly in the notebook."""
        # sudo apt install graphviz (linux systems)
        model_graph = draw_graph(
            self,
            input_size=self.model_config['input_shape_graph'],
            expand_nested=True,
            depth=1,
            directory=self.model_graph_path
        )
        if browser_view:
            model_graph.visual_graph.render(self.model_config['name'], view=True)
        # model_graph.visual_graph --> use this on returned object
        return model_graph

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines how the input tensor 'x' flows through the network"""
        skips_list = []
        
        # Encoder
        for i in range(len(self.conv_encoder_list)):
            skip = self.conv_encoder_list[i](x)
            skips_list.append(skip)
            x = self.pool_encoder_list[i](skip)
        
        # Bridge
        x = self.bridge(x)
        
        # Decoder with attention
        for i in range(len(self.conv_decoder_list)):
            x = self.up_decoder_list[i](x)
            
            # Apply attention gate
            skip_connection = skips_list[-(i + 1)]
            attended_skip = self.attention_gates[i](g=x, x=skip_connection)
            
            x = torch.cat([x, attended_skip], dim=1)
            x = self.conv_decoder_list[i](x)
        
        # Output
        x = self.outputs(x)
        x = self.sigmoid(x)
        
        return x
