import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
# from random_walk import uniform_random_walk, uniqueness
from .modernST_layer import BackboneBlock
from utils.random_walk import uniform_random_walk, uniqueness
from tsl.nn.utils import get_layer_activation, get_functional_activation
from tsl.nn.layers.base import NodeEmbedding


class ModernST(nn.Module):
    """Modern TCN with random walk and higher-order features"""
    def __init__(self,
                 input_size,
                 hidden_size, 
                 exog_size,
                 ff_size,
                 num_nodes,
                 kernel_sizes,  # List of kernel sizes for each block
                 spatial_step,
                 patch_size,
                 horizon,
                 rw_length,
                 rw_samples,
                 bias_walk=True,
                 dropout=0.1,
                 activation='relu',
                 use_learned_adj=True):  # Whether to use learned adjacency matrix
        super().__init__()
        
        # Random walk parameters
        self.rw_length = rw_length
        self.rw_samples = rw_samples
        self.patch_size = patch_size
        self.bias_walk = bias_walk
        self.total_features = input_size + exog_size
        self.activation = activation
        self.use_learned_adj = use_learned_adj
        
        # Learnable node embeddings for adjacency matrix (only if needed)
        if self.use_learned_adj:
            self.source_embeddings = NodeEmbedding(num_nodes, hidden_size)
            self.target_embeddings = NodeEmbedding(num_nodes, hidden_size)
        
        # Stem network for initial feature processing
        self.stem = nn.Sequential(
            nn.Conv2d(input_size, hidden_size, kernel_size=patch_size),
            get_layer_activation(activation)(),
            nn.Flatten(start_dim=-3),
            nn.Linear(hidden_size * ((rw_length + 1) - (patch_size[1] - 1)), hidden_size),
            get_layer_activation(activation)()
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm([num_nodes, hidden_size])
        
        # Backbone blocks
        self.blocks = nn.ModuleList([
            BackboneBlock(num_nodes, kernel_size, spatial_step,
                         self.total_features, hidden_size, activation, dropout)
            for kernel_size in kernel_sizes
        ])
        
        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(self.total_features * hidden_size, ff_size),
            get_layer_activation(activation)(),
            nn.Linear(ff_size, horizon)
        )
        
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize model parameters"""
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d)):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)

    def _gather_walk_features(self, features, walks):
        """Gather features along random walk paths"""
        # features: (batch, time, nodes, features)
        # walks: (batch, samples, time, nodes, length)
        
        features = rearrange(features, 'b t n f -> b 1 t n 1 f')
        walks = rearrange(walks, 'b s t n l -> b s t n l 1')
        
        # Gather features for each walk step
        gathered = torch.gather(
            features.expand(-1, walks.shape[1], -1, -1, walks.shape[4], -1),
            3,
            walks.expand(-1, -1, -1, -1, -1, features.shape[5])
        )
        
        # Output: (batch, samples, time, nodes, length, features)
        return gathered

    def get_learned_adjacency(self):
        """Compute learned adjacency matrix from node embeddings"""
        if not self.use_learned_adj:
            return None
        logits = get_functional_activation(self.activation)(self.source_embeddings() @ self.target_embeddings().T)
        return torch.softmax(logits, dim=1)

    def forward(self, x, edge_index, edge_weight, u=None,
                rw_edge_index=None, rw_edge_weight=None, 
                edge_features=None, triangle_features=None):
        """
        Forward pass
        
        Args:
            x: Node features (batch, time, nodes, features)
            edge_index: Edge connectivity (2, num_edges)
            edge_weight: Edge weights (num_edges,)
            u: Exogenous features (batch, time, nodes, features) or (batch, time, features)
            rw_edge_index: Random walk edge index (optional)
            rw_edge_weight: Random walk edge weights (optional)
            edge_features: Edge-level features (optional)
            triangle_features: Triangle-level features (optional)
        
        Returns:
            predictions: (batch, horizon, nodes, 1)
        """
        batch_size, time_steps, num_nodes, node_features = x.shape
        
        # Concatenate higher-order features if available
        features_list = [x]
    
        if edge_features is not None:
            # edge_features: (num_edges,) -> (batch, time, num_edges, node_features)
            edge_features_expanded = edge_features.unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(
                batch_size, time_steps, 1, node_features
            )
            features_list.append(edge_features_expanded)
            
        if triangle_features is not None:
            # triangle_features: (num_triangles,) -> (batch, time, num_triangles, node_features)
            triangle_features_expanded = triangle_features.unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(
                batch_size, time_steps, 1, node_features  
            )
            features_list.append(triangle_features_expanded)
        
        x_concat = torch.cat(features_list, dim=2)  # Concatenate along node dimension
        
        # Generate random walks
        rw_edge_idx = rw_edge_index if rw_edge_index is not None else edge_index
        rw_edge_wgt = rw_edge_weight if rw_edge_weight is not None else edge_weight
        
        walks = uniform_random_walk(
            edge_index=rw_edge_idx.to(x.device),
            edge_weight=rw_edge_wgt.to(x.device),
            nodes=torch.arange(num_nodes).to(x.device),
            batch_size=batch_size,
            num_samples=self.rw_samples,
            timesteps=time_steps,
            length=self.rw_length,
            bias_walk=self.bias_walk
        )  # (batch, samples, time, nodes, length)
        
        # Compute uniqueness features
        uniqueness_scores = uniqueness(walks).flip(-1)
        uniqueness_scores = uniqueness_scores / uniqueness_scores.shape[-1] * 2 * math.pi
        uniqueness_features = torch.cat([
            uniqueness_scores.sin().unsqueeze(-1),
            uniqueness_scores.cos().unsqueeze(-1)
        ], dim=-1)
        
        # Gather features along walks
        gathered_features = self._gather_walk_features(x_concat, walks.flip(-1))
        
        # Combine with uniqueness features
        x_walks = torch.cat([gathered_features, uniqueness_features], dim=-1)
        x_walks = x_walks.unsqueeze(-2)  # Add dimension for stem processing
        
        # Process exogenous features
        if u is not None:
            if u.dim() == 3:  # (batch, time, features)
                u = repeat(u, 'b t f -> b s t n l 1 f',
                          s=self.rw_samples, n=num_nodes, l=self.rw_length + 1)
            x_walks = torch.cat([x_walks, u], dim=-1)
        
        # Get learned adjacency matrix (only if enabled)
        adj_matrix = self.get_learned_adjacency() if self.use_learned_adj else None
        
        # Process through backbone blocks
        batch_size, num_samples, time_steps, num_nodes, length, _, total_features = x_walks.shape
        
        for i, block in enumerate(self.blocks):
            if i == 0:
                # Initial stem processing
                x_temp = rearrange(x_walks, 'b s t n l d f -> (b t n f) d s l')
                x_temp = self.stem(x_temp).squeeze()  # Remove spatial dimensions
                
                stem_features = x_temp.clone()
                residual = rearrange(x_temp, '(b t n f) d -> b t n d f',
                                   b=batch_size, n=num_nodes, f=total_features, t=time_steps)
            else:
                x_temp = rearrange(residual, 'b t n d f -> (b t n f) d')
            
            # Reshape for block processing
            x_block = rearrange(x_temp, '(b t n f) d -> b n f d t',
                               b=batch_size, n=num_nodes, f=total_features, t=time_steps)
            
            # Process through block (pass adj_matrix only if available)
            x_block = block(x_block, adj_matrix, stem_features)
            
            # Update residual
            residual = x_block + residual
        
        # Final processing
        x_final = rearrange(residual, 'b t n d f -> (b t f) n d')
        x_final = self.layer_norm(x_final)
        x_final = rearrange(x_final, '(b t f) n d -> b t n d f',
                           b=batch_size, t=time_steps, f=total_features)
        
        # Use last timestep for prediction
        x_pred = x_final[:, -1]  # (batch, nodes, hidden, features)
        x_pred = rearrange(x_pred, 'b n d f -> b n (f d)')
        
        # Generate predictions
        predictions = self.head(x_pred)  # (batch, nodes, horizon)
        predictions = rearrange(predictions, 'b n h -> b h n')
        
        return predictions.unsqueeze(-1)  # (batch, horizon, nodes, 1)