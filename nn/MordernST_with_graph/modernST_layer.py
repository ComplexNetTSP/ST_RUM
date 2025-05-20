import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


def conv_with_norm(input_size, hidden_size, kernel_size, stride, groups):
    modules = nn.Sequential(
        nn.Conv1d(in_channels = input_size,
                     out_channels = hidden_size,
                     kernel_size=kernel_size,
                     stride=stride,
                     groups=groups),
        nn.GELU()
    )
    return modules



class DW_large_kernel(nn.Module):
    def __init__(self,
                 input_size,hidden_size,
                 large_kernel,
                 stride, groups):
        super().__init__()

        self.hidden_size = hidden_size
        self.large_kernel = large_kernel
        self.stride = stride


        # only one large kernel
        self.large_conv = conv_with_norm(input_size = input_size, hidden_size = hidden_size,
                                       kernel_size=large_kernel, stride=stride, groups=groups)
    def forward(self, x_emb):
        # causual pad at left side
        x = F.pad(x_emb,
                  pad=((self.large_kernel - 1), 0),
                  mode='constant', value=0)
        out = self.large_conv(x)

        return out
    

class Backbone_blocks(nn.Module):
    def __init__(self, num_nodes, large_kernel, num_variables, hidden_size, drop=0.1):
        super().__init__()
        self.dw_conv = nn.Sequential(
            DW_large_kernel(num_variables*hidden_size,num_variables*hidden_size,
                              large_kernel,
                              stride=1,
                              groups = num_variables*hidden_size),
            nn.Dropout(p=drop),
            nn.GELU()
            
            
        )
        self.layernorm1 = nn.LayerNorm([num_nodes, hidden_size])
        
        self.pw_con1 = nn.Sequential(
            nn.Conv1d(
            in_channels=num_variables*hidden_size, 
            out_channels=num_variables*hidden_size, 
            kernel_size=1,
            groups=num_variables
            ),
            nn.Dropout(p=drop),
            nn.GELU()
    
        )
        
        self.pw_con2 = nn.Sequential(
            nn.Conv1d(
            in_channels=num_variables*hidden_size, 
            out_channels=num_variables*hidden_size, 
            kernel_size=1,
            groups=hidden_size
            ),
            nn.Dropout(p=drop),
            nn.GELU()       
        )
        self.layernorm2 = nn.LayerNorm([num_nodes, hidden_size])

    def forward(self, x_emb):
        # x_emb -> [batch_size, num_samples, num_nodes, feature_dim, D, timesteps_reduce]
        batch_size, num_nodes, feature_dim, D, timesteps_reduce = x_emb.shape

        x = rearrange(x_emb, 'b n f d t -> (b n) (f d) t').contiguous() 
        x = self.dw_conv(x)

        x = rearrange(x, '(b n) (f d) t -> (b f) t n d',
                      b=batch_size, n=num_nodes, f=feature_dim, d=D)

        x = self.layernorm1(x)
        
        x = rearrange(x, '(b f) t n d -> (b n) (f d) t ',
                      b=batch_size, n=num_nodes, f=feature_dim, d=D)

        x = self.pw_con1(x)
        x = rearrange(x, '(b n) (f d) t -> (b n) (d f) t',
                      b=batch_size, n=num_nodes, f=feature_dim, d=D).contiguous() 

        x = self.pw_con2(x)
        x = rearrange(x, '(b n) (d f) t -> b d n f t',
                      b=batch_size, d=D,
                      n=num_nodes, f=feature_dim).contiguous() 
        
        # [batch_size, num_samples, num_nodes, feature_dim, D, timesteps_reduce]
        x = rearrange(x, 'b d n f t -> b n f d t').contiguous() 
        
        # residual connection
        out = x + x_emb

        out = rearrange(out, 'b n f d t -> (b f) t n d').contiguous() 

        out = self.layernorm2(out)

        out = rearrange(out, '(b f) t n d -> b t n d f',
                        b=batch_size, n=num_nodes, f=feature_dim, d=D).contiguous() 

        return out