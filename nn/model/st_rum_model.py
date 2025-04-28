from torch import Tensor, nn
from einops import rearrange
from tsl.nn.blocks.decoders import MLPDecoder
from tsl.nn.models.base_model import BaseModel
from tsl.nn.blocks.encoders.conditional import ConditionalBlock
from layer import ST_RUM

class ST_RUM_Model(BaseModel):
    return_type = Tensor
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 horizon: int,
                 exog_size: int = 0,
                 hidden_size: int = 32,
                 ff_size: int = 256,
                 n_layers: int = 1,
                 dropout: float = 0.,
                 activation: str = 'relu'):
        super().__init__()
        if exog_size:
            self.input_encoder = ConditionalBlock(input_size=input_size,
                                                  exog_size=exog_size,
                                                  output_size=hidden_size,
                                                  activation=activation)
        else:
            self.input_encoder = nn.Linear(input_size, hidden_size)

        self.st_rum = ST_RUM(input_size=hidden_size,
                           hidden_size=hidden_size,
                           n_layers=n_layers,
                           return_only_last_state=True)

        

        self.readout = MLPDecoder(input_size=hidden_size,
                                  hidden_size=ff_size,
                                  output_size=output_size,
                                  horizon=horizon,
                                  activation=activation,
                                  dropout=dropout)

    def forward(self,
                x,
                rw_edge_index,
                edge_feature,
                triangle_feature,
                u):
        if u is not None:
            if u.dim() == 3:
                u = rearrange(u, 'b s c -> b s 1 c')
            x = self.input_encoder(x, u)
        else:
            x = self.input_encoder(x)
        out = self.st_rum(x,
                        rw_edge_index,
                        edge_feature,
                        triangle_feature)
        return self.readout(out)