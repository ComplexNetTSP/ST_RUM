import torch
import math
from torch import nn
from custom_gpu import custom_GRU
import torch.nn.functional as F
from tsl.nn.layers.recurrent.base import GraphGRUCellBase
from tsl.nn.blocks.encoders.recurrent.base import RNNBase
from utils import uniform_random_walk, uniqueness



class Simplicial_RUMLayer(torch.nn.Module):
    def __init__(self,
                 input_size, 
                 output_size,
                 num_samples=5,
                 length=5,
                 *args,
                 **kwargs):
        super().__init__()
        self.num_samples = num_samples
        self.length = length
        self.output_size = output_size
        self.walkRNN = custom_GRU(2, output_size, bidirectional=True,*args,**kwargs)
        
        self.semanticRNN = custom_GRU(input_size + 2*output_size,
                                      output_size, *args,**kwargs)

        self.fc_edge = nn.Sequential(
            nn.Linear(1, input_size),
            nn.ReLU()
        )
        self.fc_triangle = nn.Sequential(
            nn.Linear(1, input_size),
            nn.ReLU()
        )

    def _gather_walk_features(self, features, walks):
        """Fully vectorized feature gathering without loops"""
        # Create batch indices tensor [batch_size, 1, 1, 1]
        batch_indices = torch.arange(walks.shape[0], device=walks.device).view(-1, 1, 1, 1)
        
        # Expand batch_indices to match walks shape
        batch_indices = batch_indices.expand_as(walks)
        
        # Gather features using advanced indexing
        # The result shape will be [batch_size, num_samples, num_nodes, length, feature_dim]
        return features[batch_indices, walks]

    def forward(self,x , rw_edge_index, edge_feature, triangle_feature):
        # x --> [batch_size, num_nodes, features]
        batch_size, num_nodes, features = x.shape

        edge_feature = edge_feature.unsqueeze(-1).repeat(batch_size,1,1)
        triangle_feature = triangle_feature.unsqueeze(-1).repeat(batch_size,1,1)

        mem_before = torch.cuda.memory_allocated()
        # Your logic here        
        x_new = torch.cat([x, self.fc_edge(edge_feature), self.fc_triangle(triangle_feature)], dim = 1) # --> [batch_size, num_(node + edge + triangle), feature]
        mem_after = torch.cuda.memory_allocated()
        # print(f"Memory delta in CustomModule: {(mem_after - mem_before) / 1e6:.2f} MB")

        
        walks, eids = uniform_random_walk(
            edge_index=rw_edge_index, 
            nodes=torch.arange(num_nodes).to('cuda'), 
            batch_size = batch_size,
            num_samples=self.num_samples,
            length=self.length
        ) # -- > [batch_size, num_samples, num_nodes, length]
        uniqueness_walk = uniqueness(walks)
        walks, uniqueness_walk = walks.flip(-1), uniqueness_walk.flip(-1)
        uniqueness_walk = uniqueness_walk / uniqueness_walk.shape[-1]
        uniqueness_walk = uniqueness_walk * math.pi * 2.0
        uniqueness_walk = torch.cat(
            [
                uniqueness_walk.sin().unsqueeze(-1),
                uniqueness_walk.cos().unsqueeze(-1),
            ],
            dim=-1,
        )

        # Gather features for each node in the walks
        # [batch_size, num_samples, num_nodes, length, feature_dim]
        gathered_features = self._gather_walk_features(x_new, walks)

        h0 = torch.zeros(self.walkRNN.num_layers*2, *gathered_features.shape[:-2],
                         self.output_size, device=gathered_features.device)

        y_walk, h_walk = self.walkRNN(uniqueness_walk, h0)
        
        h_walk = h_walk.mean(0, keepdim=True)

        if self.semanticRNN.num_layers > 1:
            h_walk = h_walk.repeat(self.semanticRNN.num_layers, 1, 1, 1, 1)
            gathered_features = torch.cat([gathered_features, y_walk], dim=-1)
        else:
            gathered_features = torch.cat([gathered_features, y_walk], dim=-1)
            
        y, h = self.semanticRNN(gathered_features, h_walk)
        y = y.mean(1)
        y = F.relu(y)
        # # First collect Python garbage
        # gc.collect()
        # # Then clear CUDA cache
        # torch.cuda.empty_cache()
        return y[:, :, -1, :]
    
    
class ST_RUM_Cell(GraphGRUCellBase):
    def __init__(self,
                 input_size,
                 hidden_size):
        # instantiate gates
        forget_gate = Simplicial_RUMLayer(input_size+hidden_size,hidden_size)
        
        update_gate = Simplicial_RUMLayer(input_size+hidden_size,hidden_size)
        
        candidate_gate = Simplicial_RUMLayer(input_size+hidden_size,hidden_size)
        
        super().__init__(hidden_size=hidden_size,
                        forget_gate=forget_gate,
                        update_gate=update_gate,
                        candidate_gate=candidate_gate)
        
class ST_RUM(RNNBase):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 n_layers: int = 1,
                 cat_states_layers: bool = False,
                 return_only_last_state: bool = False):
        self.input_size = input_size
        self.hidden_size = hidden_size
        rnn_cells = [
            ST_RUM_Cell(input_size if i == 0 else hidden_size,
                      hidden_size) for i in range(n_layers)
        ]
        super().__init__(rnn_cells,cat_states_layers,return_only_last_state)