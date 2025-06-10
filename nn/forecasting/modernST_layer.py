import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tsl.nn.utils import get_layer_activation
from tsl.nn.layers.graph_convs import DenseGraphConvOrderK


class DenseGraphConvCustomMLP(DenseGraphConvOrderK):
    """Custom graph convolution with MLP using grouped convolution for efficiency"""
    def __init__(self, input_size, hidden_size, order=2, include_self=True, channel_last=False):
        super().__init__(input_size, hidden_size, order=order, 
                        include_self=include_self, channel_last=channel_last)
        
        calculated_input_size = (order + (1 if include_self else 0)) * input_size
        self.mlp = nn.Conv2d(calculated_input_size, hidden_size, kernel_size=1, groups=input_size)


class DepthwiseLargeKernel(nn.Module):
    """Depthwise convolution with large kernel and causal padding"""
    def __init__(self, channels, kernel_size, groups=None):
        super().__init__()
        self.kernel_size = kernel_size
        groups = groups or channels
        
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels, 
            kernel_size=kernel_size,
            groups=groups
        )
    
    def forward(self, x):
        # Input: (batch, channels, time)
        # Causal padding on left side
        x = F.pad(x, pad=(self.kernel_size - 1, 0), mode='constant', value=0)
        return self.conv(x)


class BackboneBlock(nn.Module):
    """Single backbone block with temporal conv, spatial conv, and pointwise convs"""
    def __init__(self, num_nodes, kernel_size, spatial_step, num_features, hidden_size, activation, dropout=0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.hidden_size = hidden_size
        
        total_channels = num_features * hidden_size
        
        # Temporal convolution with large kernel
        self.temporal_conv = nn.Sequential(
            DepthwiseLargeKernel(total_channels, kernel_size, groups=total_channels),
            nn.Dropout(dropout),
            get_layer_activation(activation)()
        )
        
        # Spatial graph convolution
        self.spatial_conv = DenseGraphConvCustomMLP(
            input_size=total_channels,
            hidden_size=total_channels,
            order=spatial_step, 
            channel_last=True
        )
        
        # Pointwise convolutions
        self.pointwise_conv1 = nn.Sequential(
            nn.Conv1d(total_channels, total_channels, kernel_size=1, groups=num_features),
            nn.Dropout(dropout),
            get_layer_activation(activation)()
        )
        
        self.pointwise_conv2 = nn.Sequential(
            nn.Conv1d(total_channels, total_channels, kernel_size=1, groups=hidden_size),
            nn.Dropout(dropout),
            get_layer_activation(activation)()
        )
        
        self.layer_norm = nn.LayerNorm([num_nodes, hidden_size])

    def forward(self, x, adj_matrix=None, stem_features=None):
        # Input: (batch, nodes, features, hidden, time)
        batch_size, num_nodes, num_features, hidden_size, time_steps = x.shape
        
        # Temporal convolution
        # Reshape to (batch*nodes, features*hidden, time)
        x_temp = rearrange(x, 'b n f h t -> (b n) (f h) t')
        x_temp = self.temporal_conv(x_temp)
        
        # Spatial convolution (only if adjacency matrix provided)
        # Reshape to (batch, time, nodes, features*hidden)
        x_spatial = rearrange(x_temp, '(b n) (f h) t -> b t n (f h)',
                            b=batch_size, n=num_nodes, f=num_features, h=hidden_size)
        
        if (adj_matrix is not None) & (stem_features is not None):
            x_spatial = self.spatial_conv(x_spatial, adj_matrix)
            
            # Gate mechanism for stem features
            # First reshape stem_features to match current batch processing
            # stem_features: (batch*time*nodes*features, hidden) -> (batch, time, nodes, features*hidden)
            stem_reshaped = rearrange(stem_features, '(b t n f) h -> b t n (f h)',
                                    b=batch_size, t=time_steps, n=num_nodes, f=num_features)
            
            gate = torch.tanh(x_spatial + stem_reshaped)
            x_spatial = x_spatial * gate + stem_reshaped * (1 - gate)
        
        
        # Pointwise convolutions
        # Back to (batch*nodes, features*hidden, time)
        x_pw = rearrange(x_spatial, 'b t n (f h) -> (b n) (f h) t',
                        b=batch_size, n=num_nodes, f=num_features, h=hidden_size)
        x_pw = self.pointwise_conv1(x_pw)
        
        # Rearrange for second pointwise conv
        x_pw = rearrange(x_pw, '(b n) (f h) t -> (b n) (h f) t',
                        b=batch_size, n=num_nodes, f=num_features, h=hidden_size)
        x_pw = self.pointwise_conv2(x_pw)
        
        # Final reshape and residual connection
        x_out = rearrange(x_pw, '(b n) (h f) t -> b n f h t',
                         b=batch_size, n=num_nodes, f=num_features, h=hidden_size)
        x_out = x_out + x  # Residual connection
        
        # Layer normalization
        x_norm = rearrange(x_out, 'b n f h t -> (b t f) n h')
        x_norm = self.layer_norm(x_norm)
        x_norm = rearrange(x_norm, '(b t f) n h -> b t n h f',
                          b=batch_size, t=time_steps, f=num_features)
        
        return x_norm