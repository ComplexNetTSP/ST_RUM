import torch

class custom_GRU(torch.nn.GRU):
    def __init__(self, *args, **kwargs):
        kwargs["batch_first"] = True
        super().__init__(*args, **kwargs)
    
    def forward(self, input, h_0):
        # input -- > [batch_size, num_samples, num_nodes, RW_length, features]
        # h0 -- > [num_layer*Direction(1 or 2), batch_size, num_samples, num_nodes, features]

        # input torch.Size([64, 10, 207, 4, 2])
        # h_0 torch.Size([1, 64, 10, 207, 64])
        
        num_direction = 2 if self.bidirectional else 1
        batch_shape = input.shape[:-2] #--> [batch_size, num_samples, num_nodes] 
        event_shape_input = input.shape[-2:] #--> [RW_length, features] 
        event_shape_h_0 = h_0.shape[-1]
        input = input.view(-1, *event_shape_input) #--> [batch_size * num_samples * num_nodes, RW_length, features]
        h_0 = h_0.view(num_direction * self.num_layers, -1, event_shape_h_0) #--> [num_layer*Direction(1 or 2), batch_size * num_samples * num_nodes, features]
        
        output, h_n = super().forward(input, h_0)
        output = output.view(*batch_shape, *output.shape[-2:])
        h_n = h_n.view(num_direction * self.num_layers, *batch_shape, *h_n.shape[-1:])
        return output, h_n