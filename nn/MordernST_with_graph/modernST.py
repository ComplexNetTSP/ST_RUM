import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from random_walk import uniform_random_walk, uniqueness
from modernST import Backbone_blocks

class ModernTCN_rum(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_nodes,
                 large_kernel,
                 patch_size,
                 patch_stride,
                 windows, 
                 horizon,
                 rw_length, 
                 rw_sample,
                 dropout):
        super().__init__()

        self.rw_length = rw_length
        self.rw_sample = rw_sample
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.windows_patches = windows // patch_stride[0]
        self.rw_length_patches = (rw_length+1) // patch_stride[1]


        self.stem = nn.Sequential(
            nn.Conv3d(input_size, hidden_size, kernel_size=patch_size),
            nn.GELU()
        )
        self.layernorm = nn.Sequential(
            # Rearrange('... d t -> ... t d'),
            nn.LayerNorm([num_nodes,hidden_size])
            # Rearrange('... t d -> ... d t')
        )

        self.num_blocks = len(large_kernel)

        # backbones
        self.blocks = nn.ModuleList()
        for block_id in range(self.num_blocks):
            backbone = Backbone_blocks(num_nodes, large_kernel[block_id],
                                      input_size+4, hidden_size, dropout)
            self.blocks.append(backbone)


        self.head = nn.Sequential(
            nn.Linear((input_size+4)*hidden_size, 256),
            nn.GELU(),
            nn.Linear(256, horizon)
        )

        self.reset_parameters()


    def reset_parameters(self):
        """
        Simplified initialization using Xavier Gaussian for all weights.
        """
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                # Xavier Gaussian initialization for all convolutional layers
                # if 'dw' in name:
                #     # print(f"Initializing depthwise conv: {name}")
                #     nn.init.normal_(m.weight, 3)
                #     if m.bias is not None:
                #         nn.init.constant_(m.bias, 1)
                # else:
                nn.init.kaiming_normal_(m.weight)
                # Initialize bias if present
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.BatchNorm1d):
                # Standard initialization for batch norm
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.Linear):
                # Xavier Gaussian for all linear layers
                nn.init.kaiming_normal_(m.weight)
                print(f"Initializing Linear: {name}")
                # Initialize bias
                nn.init.constant_(m.bias, 0)

    def _gather_walk_features(self, features, walks):
        # Move tensors to CPU for processing
        cpu_features = features #--> batch_size, timestep, num_simplices, feature
        cpu_walks = walks #--> batch_size, num_random_walk_sample,timestep,num_nodes,random_walk_length

        # cpu_features torch.Size([64, 12, 1334, 1])
        # cpu_walks torch.Size([64, 5, 12, 207, 6])

        features = rearrange(cpu_features, 'b t n f -> b 1 t n 1 f').contiguous() 
        walks = rearrange(walks, 'b s t n l -> b s t n l 1').contiguous() 

        gathered_features = torch.gather(features.expand(-1, walks.shape[1], -1, -1, walks.shape[4], -1),
                                         3,
                                         walks.expand(-1, -1, -1, -1, -1, cpu_features.shape[3]))

        # Final shape: [batch_size, num_samples, timestep, num_nodes, length, feature_dim]
        # output = rearrange(gathered_features, 'b s t n l f -> b s n l f')
        
        return gathered_features.to(walks.device)

    def single_forward(self, x, u):
        batch_size, num_samples, timesteps, num_nodes, length, _, feature_dim = x.shape

        if u.dim() == 3:
            u = repeat(u, 'b t f -> b s t n l 1 f',
                       s = num_samples, n=num_nodes, l=length)
        x = torch.cat([x, u], -1)

        *_, feature_dim = x.shape

        # x = rearrange(x, 'b s t n l d f -> (b s n f) d t l')
        # _, _, timestep, rw_length = x.shape

        # residual = torch.zeros(1, 1, 1, 1, 1, device=x.device)
        for i in range(self.num_blocks):
            # [batch_size, num_samples, timesteps, num_nodes, length, D, feature_dim] 
            # -> [batch_size * num_samples * num_nodes * feature_dim, D, timesteps, length] 
            if i == 0:
                x = rearrange(x, 'b s t n l d f -> (b n f) d t s l').contiguous() 


                x = self.stem(x) # --> [batch_size  * num_nodes * feature_dim, D, timesteps_reduced, 1, 1]
                
                x = x.squeeze() # --> [batch_size * num_nodes * feature_dim, D, timesteps_reduced]
                
                # x = self.layernorm(x)
                residual = rearrange(x, '(b n f) d t -> b t n d f',
                          b=batch_size, n=num_nodes, f=feature_dim).contiguous()
            else:
                x = rearrange(residual, 'b t n d f -> (b n f) d t').contiguous()
            
            
            x = rearrange(x, '(b n f) d t -> b n f d t',
                          b=batch_size, n=num_nodes, f=feature_dim).contiguous() 
            
            x = self.blocks[i](x)

            residual = x + residual

        residual = rearrange(x, 'b t n d f -> (b f) t n d',
                      b=batch_size, n=num_nodes, f=feature_dim).contiguous()

        residual = self.layernorm(residual)

        residual = rearrange(residual, '(b f) t n d -> b t n d f',
                      b=batch_size, n=num_nodes, f=feature_dim).contiguous()
        return residual

    def forward(self, x, edge_index, u):
        # x --> [batch_size, time_steps, num_nodes, features]
        # u --> [batch_size, time_steps, num_nodes, features]

        # print('first')
        # check_tensor(x)
        batch_size, T, num_nodes, features = x.shape
        walks, eids = uniform_random_walk(
            edge_index=edge_index.to('cuda'), 
            nodes=torch.arange(num_nodes).to('cuda'), 
            batch_size = batch_size,
            num_samples=self.rw_sample,
            timesteps = T,
            length=self.rw_length
        ) # -- > [batch_size, num_samples, timesteps, num_nodes, length]
        uniqueness_walk = uniqueness(walks)
        walks, uniqueness_walk = walks.flip(-1), uniqueness_walk.flip(-1)
        # walks, uniqueness_walk = walks, uniqueness_walk
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
        # [batch_size, num_samples, timesteps, num_nodes, length, feature_dim]
        gathered_features = self._gather_walk_features(x, walks)  

        # [batch_size, num_samples, timesteps, num_nodes, length, 1, feature_dim]
        gathered_features = torch.concat([gathered_features, uniqueness_walk], dim = -1)
        x = gathered_features.unsqueeze(-2)
        
        # print(uniqueness_walk[0,0,0,0,:,0,:])
        # torch.Size([32, 1, 12, 207, 6, 1, 3])

        x = self.single_forward(x, u)

        # torch.Size([32, 1, 2, 207, 2, 32, 3])

        x = x[:, -1]
        x = rearrange(x, 'b n d f -> b n (f d)').contiguous() 
        # x = rearrange(x, 'b t n d f -> b n (f d t)').contiguous() 
        pred = self.head(x)

        pred = rearrange(pred, 'b n h -> b h n').contiguous() 

        return pred.unsqueeze(-1)