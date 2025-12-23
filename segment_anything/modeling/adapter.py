"""
Iteration-Aware Dynamic Adapter Module

This module implements the dynamic adapter architecture that adapts based on 
the interaction step in interactive medical image segmentation.

Key Features:
- Global Adapter: Activated at step 0 for global semantic understanding
- Refine Adapter: Activated at step > 0 for local texture refinement
"""

import torch
import torch.nn as nn
from typing import Optional


class Adapter(nn.Module):
    """
    Basic Adapter module with down-projection and up-projection.
    
    Args:
        d_model: Input/output dimension
        bottleneck_dim: Bottleneck dimension (typically d_model // 4 or d_model // 8)
        dropout: Dropout rate
        init_scale: Initial scaling factor for the adapter output
    """
    def __init__(
        self,
        d_model: int,
        bottleneck_dim: int = 64,
        dropout: float = 0.0,
        init_scale: float = 0.1,
    ):
        super().__init__()
        
        self.down_proj = nn.Linear(d_model, bottleneck_dim)
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(bottleneck_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize with small weights for stable training
        nn.init.xavier_uniform_(self.down_proj.weight, gain=init_scale)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.xavier_uniform_(self.up_proj.weight, gain=init_scale)
        nn.init.zeros_(self.up_proj.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, N, D) or (B, D, H, W)
            
        Returns:
            Adapted features with residual connection
        """
        # Store original shape
        original_shape = x.shape
        
        # If input is 4D (B, D, H, W), reshape to (B, H*W, D)
        if len(x.shape) == 4:
            B, D, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)  # (B, H*W, D)
            
        # Adapter forward pass
        residual = x
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        
        # Add residual connection
        x = residual + x
        
        # Reshape back to original if needed
        if len(original_shape) == 4:
            x = x.transpose(1, 2).reshape(original_shape)
            
        return x


class DynamicAdapter(nn.Module):
    """
    Dynamic Adapter that switches between Global and Refine adapters based on interaction step.
    
    - Step 0: Uses Global Adapter for whole organ/region localization
    - Step > 0: Uses Refine Adapter for boundary refinement
    
    Args:
        d_model: Input/output dimension
        bottleneck_dim: Bottleneck dimension
        dropout: Dropout rate
        init_scale: Initial scaling factor
    """
    def __init__(
        self,
        d_model: int,
        bottleneck_dim: int = 64,
        dropout: float = 0.0,
        init_scale: float = 0.1,
    ):
        super().__init__()
        
        # Two separate adapters for different stages
        self.global_adapter = Adapter(d_model, bottleneck_dim, dropout, init_scale)
        self.refine_adapter = Adapter(d_model, bottleneck_dim, dropout, init_scale)
        
    def forward(self, x: torch.Tensor, step: int = 0) -> torch.Tensor:
        """
        Forward pass with step-aware adapter selection.
        
        Args:
            x: Input tensor of shape (B, N, D) or (B, D, H, W)
            step: Current interaction step (0 = first click, >0 = refinement)
            
        Returns:
            Adapted features
        """
        if step == 0:
            # First interaction: use global adapter for semantic understanding
            return self.global_adapter(x)
        else:
            # Subsequent interactions: use refine adapter for boundary refinement
            return self.refine_adapter(x)


class AdapterLayer(nn.Module):
    """
    Wrapper layer that integrates adapter into a transformer block.
    This can be inserted after the attention or MLP layer.
    
    Args:
        adapter: The adapter module (Adapter or DynamicAdapter)
    """
    def __init__(self, adapter: nn.Module):
        super().__init__()
        self.adapter = adapter
        
    def forward(self, x: torch.Tensor, step: int = 0) -> torch.Tensor:
        """
        Args:
            x: Input features
            step: Interaction step (passed to DynamicAdapter)
            
        Returns:
            Adapter output
        """
        if isinstance(self.adapter, DynamicAdapter):
            return self.adapter(x, step)
        else:
            return self.adapter(x)


if __name__ == '__main__':
    # Test the adapter modules
    print("Testing Adapter modules...")
    
    # Test basic Adapter
    adapter = Adapter(d_model=768, bottleneck_dim=64)
    x = torch.randn(2, 196, 768)  # (B, N, D)
    out = adapter(x)
    print(f"Basic Adapter - Input shape: {x.shape}, Output shape: {out.shape}")
    
    # Test DynamicAdapter
    dynamic_adapter = DynamicAdapter(d_model=768, bottleneck_dim=64)
    
    # Step 0 (global)
    out_global = dynamic_adapter(x, step=0)
    print(f"Dynamic Adapter (step=0) - Output shape: {out_global.shape}")
    
    # Step 1 (refine)
    out_refine = dynamic_adapter(x, step=1)
    print(f"Dynamic Adapter (step=1) - Output shape: {out_refine.shape}")
    
    # Test with 4D input (after conv layers)
    x_4d = torch.randn(2, 768, 14, 14)  # (B, D, H, W)
    out_4d = dynamic_adapter(x_4d, step=0)
    print(f"Dynamic Adapter 4D - Input shape: {x_4d.shape}, Output shape: {out_4d.shape}")
    
    print("\nAll tests passed!")
