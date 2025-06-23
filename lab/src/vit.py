# Builtins
import os
import math
from datetime import datetime
# Installed
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import torchview
from torchview import draw_graph
# Local
from .utils.config_files import save_config
# Types
from typing import (
    Tuple,
    Dict,
    Any
)


class VisionTransformer(nn.Module):
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

        # Save configs (comment out if save_config function is causing issues)
        try:
            save_config(self.model_config_path, self.model_config, calling_class=self)
        except Exception as e:
            print(f"Warning: Could not save config: {e}")

        # Extract parameters from config
        if len(self.model_config["input_shape"]) == 4:  # 3D data: (C, D, H, W)
            self.in_channels = self.model_config["input_shape"][0]
            self.img_size = self.model_config["input_shape"][-1]
            self._log("Warning: 3D input detected, ViT will reshape to process all slices: (B*D, C, H, W)")
        elif len(self.model_config["input_shape"]) == 3:  # 2D data: (C, H, W)
            self.in_channels = self.model_config["input_shape"][0]
            self.img_size = self.model_config["input_shape"][-1]
        else:
            raise ValueError(f"Invalid input shape: {self.model_config['input_shape']}. Expected 3D (C,H,W) or 4D (C,D,H,W)")
        
        self.patch_size = self.model_config["patch_size"]
        self.embed_dim = self.model_config["embed_dim"]
        self.num_heads = self.model_config["num_heads"]
        self.num_layers = self.model_config["num_layers"]
        self.dropout = self.model_config["dropout"]
        self.mlp_ratio = self.model_config["mlp_ratio"]
        
        # Add overlap configuration
        self.overlap_ratio = self.model_config.get("overlap_ratio", 0.5)
        self.use_spatial_attention = self.model_config.get("use_spatial_attention", True)
        self.use_smoothing = self.model_config.get("use_smoothing", True)

        self.first_batch = True

        # Calculate number of patches WITH overlap
        self.effective_stride = max(1, int(self.patch_size * (1 - self.overlap_ratio)))
        self.patches_per_side = (self.img_size - self.patch_size) // self.effective_stride + 1
        self.num_patches = self.patches_per_side ** 2
        self.patch_dim = self.in_channels * self.patch_size ** 2

        # Use standard Conv2d without padding issues
        self.patch_embed = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.effective_stride,
            padding=0
        )

        # Ensure positional embedding matches actual patch count
        # Recalculate actual number of patches after conv
        conv_output_size = (self.img_size - self.patch_size) // self.effective_stride + 1
        self.actual_patches_per_side = conv_output_size
        self.actual_num_patches = conv_output_size ** 2

        # Validate configuration
        self._validate_config()
        
        # Create positional encoding with correct dimensions
        self.pos_embed = nn.Parameter(torch.randn(1, self.actual_num_patches, self.embed_dim))
        
        # Class Token (kept for compatibility)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        # Dropout for embeddings
        self.pos_dropout = nn.Dropout(self.dropout)

        # Ensure layer_dropouts list is properly initialized
        if 'layer_dropouts' not in self.model_config or len(self.model_config['layer_dropouts']) < self.num_layers:
            self.model_config['layer_dropouts'] = [self.dropout] * self.num_layers

        # Transformer Blocks
        self.transformer_blocks = nn.ModuleList()
        for i in range(self.num_layers):
            if self.use_spatial_attention:
                block = SpatialTransformerBlock(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    dropout=self.model_config['layer_dropouts'][i],
                    patches_per_side=self.actual_patches_per_side  # Use actual patches
                )
            else:
                block = TransformerBlock(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    dropout=self.model_config['layer_dropouts'][i]
                )
            self.transformer_blocks.append(block)

        # Layer Normalization
        self.norm = nn.LayerNorm(self.embed_dim)

        # Reconstruction Head
        self.output_channels = self.model_config.get("output_channels", self.in_channels)
        self.reconstruction_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim * 2, self.patch_dim),
            nn.Tanh()
        )

        # Initialize weights
        self._init_weights()

    def _log(self, message: str) -> None:
        """Write a message to the log file with timestamp."""
        print(message)
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] {message}\n")
        except Exception as e:
            print(f"Warning: Could not write to log: {e}")

    def _count_parameters(self, model) -> int:
        """Count the number of parameters of the model"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _validate_config(self) -> None:
        """Validate model configuration parameters"""
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(f"Embedding dimension ({self.embed_dim}) must be divisible by number of heads ({self.num_heads})")
        
        self._log(f"Enhanced ViT Configuration: {self.actual_num_patches} patches ({self.actual_patches_per_side}x{self.actual_patches_per_side}), "
                 f"{self.num_layers} transformer blocks, {self.embed_dim} embedding dimension")
        self._log(f"Overlap ratio: {self.overlap_ratio}, Effective stride: {self.effective_stride}")

    def _smooth_reconstruction(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian smoothing to reduce patch boundary artifacts"""
        if not self.use_smoothing:
            return x
            
        # Create Gaussian kernel
        kernel_size = 3
        sigma = 0.8
        kernel_1d = torch.exp(-torch.pow(torch.arange(kernel_size, dtype=torch.float32, device=x.device) - kernel_size//2, 2) / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0).repeat(x.shape[1], 1, 1, 1)
        
        # Apply separable convolution for efficiency
        smoothed = F.conv2d(x, kernel_2d, padding=kernel_size//2, groups=x.shape[1])
        
        # Blend original and smoothed (70% original, 30% smoothed)
        return 0.7 * x + 0.3 * smoothed

    def _reconstruct_from_overlapping_patches(self, patch_outputs: torch.Tensor, batch_size: int, height: int, width: int) -> torch.Tensor:
        """Reconstruct image from overlapping patches with proper blending"""
        output = torch.zeros(batch_size, self.output_channels, height, width, device=patch_outputs.device)
        weight_mask = torch.zeros(batch_size, 1, height, width, device=patch_outputs.device)
        
        patch_outputs = patch_outputs.view(batch_size, self.actual_patches_per_side, self.actual_patches_per_side, 
                                         self.output_channels, self.patch_size, self.patch_size)
        
        for i in range(self.actual_patches_per_side):
            for j in range(self.actual_patches_per_side):
                # Calculate position in output image
                start_h = i * self.effective_stride
                start_w = j * self.effective_stride
                end_h = min(start_h + self.patch_size, height)
                end_w = min(start_w + self.patch_size, width)
                
                # Get patch dimensions that fit
                patch_h = end_h - start_h
                patch_w = end_w - start_w
                
                # Add patch to output with blending
                patch = patch_outputs[:, i, j, :, :patch_h, :patch_w]
                output[:, :, start_h:end_h, start_w:end_w] += patch
                weight_mask[:, :, start_h:end_h, start_w:end_w] += 1
        
        # Normalize by overlap count
        weight_mask = torch.clamp(weight_mask, min=1)
        output = output / weight_mask
        
        return output

    def _init_weights(self) -> None:
        """Initialize model weights"""
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def display_summary(self) -> None:
        """Display model summary with error handling"""
        try:
            if len(self.model_config["input_shape"]) == 4:  # 4D input
                input_size = (1, *self.model_config["input_shape"])  # Add batch dimension
            else:  # 3D input
                input_size = (1, *self.model_config["input_shape"])
                
            self._log(str(summary(self, input_size=input_size, depth=4, verbose=2)))
            self._log(f"Total trainable parameters: {self._count_parameters(self):,}")
        except Exception as e:
            self._log(f"Could not generate model summary: {e}")
            self._log(f"Model has {self._count_parameters(self):,} trainable parameters")

    def view_model_graph(self, browser_view: bool=False) -> torchview.computation_graph.ComputationGraph:
        """View the network architechture as a graph. It can be displayed in a web browser or directly in the notebook"""
        try:
            model_graph = draw_graph(
                self,
                input_size=self.model_config['input_shape_graph'],
                expand_nested=True,
                depth=1,
                directory=self.model_graph_path
            )
            if browser_view:
                model_graph.visual_graph.render(self.model_config['name'], view=True)
            return model_graph
        except Exception as e:
            self._log(f"Could not generate model graph: {e}")
            return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Enhanced forward pass with spatial continuity"""
        original_shape = x.shape
        is_3d_input = len(x.shape) == 5
        
        # Handle 3D input by flattening depth into batch dimension
        if is_3d_input:  # (B, C, D, H, W)
            batch_size, channels, depth, height, width = x.shape
            x = x.permute(0, 2, 1, 3, 4)  # (B, D, C, H, W)
            x = x.contiguous().view(batch_size * depth, channels, height, width)  # (B*D, C, H, W)
            if self.first_batch:
                self._log(f"Processing 3D input: flattening {depth} slices into batch dimension")
                self.first_batch = False
        
        current_batch_size, channels, height, width = x.shape
        
        # Patch Embedding
        x = self.patch_embed(x)  # (B*D, embed_dim, patches_h, patches_w)
        _, _, patches_h, patches_w = x.shape
        x = x.flatten(2)  # (B*D, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B*D, num_patches, embed_dim)
        
        # Add positional encoding
        if x.shape[1] != self.pos_embed.shape[1]:
            self._log(f"Warning: Patch count mismatch. Expected {self.pos_embed.shape[1]}, got {x.shape[1]}")
            # Resize pos_embed if needed
            pos_embed_resized = F.interpolate(
                self.pos_embed.transpose(1, 2).unsqueeze(0), 
                size=x.shape[1], 
                mode='linear', 
                align_corners=False
            ).squeeze(0).transpose(0, 1).unsqueeze(0)
            x = x + pos_embed_resized
        else:
            x = x + self.pos_embed
            
        x = self.pos_dropout(x)
        
        # Transformer Processing
        for block in self.transformer_blocks:
            x = block(x)
        
        # Layer Normalization
        x = self.norm(x)
        
        # Reconstruction
        x = self.reconstruction_head(x)  # (B*D, num_patches, patch_dim)
        
        # Reshape patches for reconstruction
        x = x.view(current_batch_size, self.actual_patches_per_side, self.actual_patches_per_side, 
                  self.output_channels, self.patch_size, self.patch_size)
        
        # Reconstruct with Overlap Handling
        x = self._reconstruct_from_overlapping_patches(x, current_batch_size, height, width)
        
        # Apply Smoothing
        x = self._smooth_reconstruction(x)
        
        # If original input was 3D, reshape back to 5D
        if is_3d_input:
            batch_size, depth = original_shape[0], original_shape[2]
            x = x.view(batch_size, depth, self.output_channels, height, width)  # (B, D, C, H, W)
            x = x.permute(0, 2, 1, 3, 4)  # (B, C, D, H, W)
        
        return x


class SpatialTransformerBlock(nn.Module):
    """Enhanced transformer block with spatial awareness for neighboring patches"""
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float, patches_per_side: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.patches_per_side = patches_per_side
        
        # Standard self-attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        # Spatial attention for neighboring patches
        self.spatial_norm = nn.LayerNorm(embed_dim)
        self.spatial_attn = nn.MultiheadAttention(embed_dim, max(1, num_heads//2), dropout=dropout, batch_first=True)
        
        # MLP block
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def get_spatial_neighbors(self, x: torch.Tensor) -> torch.Tensor:
        """Create spatially-aware features by incorporating neighboring patch information"""
        B, N, D = x.shape # Batch_size, number of patches, number of features for each patch
        
        # Reshape to spatial grid
        x_spatial = x.view(B, self.patches_per_side, self.patches_per_side, D)
        neighbors = torch.zeros_like(x_spatial)
        
        # Apply spatial convolution-like operation
        for i in range(self.patches_per_side):
            for j in range(self.patches_per_side):
                # Start with current patch
                neighbor_features = x_spatial[:, i, j, :].clone()
                count = 1
                
                # Add 8-connected neighbors (includes diagonal)
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.patches_per_side and 0 <= nj < self.patches_per_side:
                            # Weight closer neighbors more
                            weight = 0.5 if abs(di) + abs(dj) == 2 else 1.0  # Diagonal vs adjacent
                            neighbor_features += weight * x_spatial[:, ni, nj, :]
                            count += weight
                
                neighbors[:, i, j, :] = neighbor_features / count
        
        return neighbors.view(B, N, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard multi-head self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # Spatial attention with neighboring patches
        x_spatial = self.spatial_norm(x)
        neighbors = self.get_spatial_neighbors(x_spatial)
        spatial_attn_out, _ = self.spatial_attn(x_spatial, neighbors, neighbors)
        x = x + 0.5 * spatial_attn_out  # Reduced weight for spatial attention
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x


class TransformerBlock(nn.Module):
    """Standard transformer encoder block"""
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # MLP block
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-head attention with residual connection
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x
