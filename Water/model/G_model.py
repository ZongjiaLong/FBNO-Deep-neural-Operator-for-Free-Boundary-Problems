import torch
import torch.nn as nn

from .MLP import MLPLinear_res,FourierFeatureMap
from .phi_model import Model as Phi_Model

class Model(nn.Module):
    def __init__(self, embed_dim,device = 'cuda'):
        super(Model, self).__init__()

        self.ln1 = nn.LayerNorm(embed_dim)

        self.ffm_time = FourierFeatureMap(1,embed_dim,device)
        self.ffm_space = FourierFeatureMap(4,embed_dim)
        self.ffm_fx = FourierFeatureMap(2,embed_dim)
        # self.ffm_time = MLPLinear_res(1,embed_dim, num_layers=1)
        # self.ffm_space = MLPLinear_res(3,embed_dim, num_layers=1)
        # self.ffm_fx = MLPLinear_res(2,embed_dim, num_layers=1)
        self.corrdinate_mlp = MLPLinear_res(embed_dim, embed_dim, num_layers=3)
        self.fx_mlp = MLPLinear_res(embed_dim, embed_dim, num_layers=3)
        self.fc_deep = MLPLinear_res(embed_dim, embed_dim, num_layers=8)
        self.fc_invnorm = MLPLinear_res(embed_dim, 3, num_layers=2)


    def forward(self, x,y,z,t,fx):
        time = t.unsqueeze(1).expand(-1, x.shape[1], -1)
        coord = self.ffm_space(torch.cat([x,y,z,time],dim=-1))
        x = self.corrdinate_mlp(coord)
        fx = self.ffm_fx(fx)
        fx = self.fx_mlp(fx).unsqueeze(1).expand(-1, x.shape[1], -1)
        fx_in = self.ln1(x*fx)
        fx_out = self.fc_deep(fx_in)
        v = self.fc_invnorm(fx_out)
        return v



