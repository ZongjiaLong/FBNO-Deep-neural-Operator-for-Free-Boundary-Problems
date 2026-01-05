import torch
import torch.nn as nn

from .MLP import MLPLinear_res,FourierFeatureMap

class Model(nn.Module):
    def __init__(self, embed_dim,device = 'cuda'):
        super(Model, self).__init__()

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ffm_time = FourierFeatureMap(1,embed_dim,device)
        self.ffm_space = FourierFeatureMap(2,embed_dim)
        self.ffm_fx = FourierFeatureMap(2,embed_dim)
        self.corrdinate_mlp = MLPLinear_res(embed_dim,embed_dim,num_layers=5)
        self.fx_mlp = MLPLinear_res(embed_dim,embed_dim,num_layers=5)
        self.fc_deep = MLPLinear_res(embed_dim,embed_dim,num_layers=8)
        self.fc_invnorm = MLPLinear_res(embed_dim,2,num_layers=2)

    def fist_step(self, x,t,fx):
        time = self.ffm_time(t).unsqueeze(1).expand(-1, x.shape[1], -1)
        y = torch.ones_like(x,device=x.device,dtype=x.dtype)
        x = self.ffm_space(torch.cat([x,y],dim=-1))
        x = x*time
        x = self.corrdinate_mlp(x)
        fx = self.ffm_fx(fx)
        fx_pro = self.fx_mlp(fx)
        fx_expand = fx_pro.unsqueeze(1).expand(-1, x.shape[1], -1)
        fx_in = self.ln1(x*fx_expand)
        fx_out = self.fc_deep(fx_in)
        fx_out = self.fc_invnorm(fx_out)
        out_max, out_min = torch.unbind(fx_out, 2)
        return out_max, out_min,fx_pro
    def inv_first_step(self,x,y,t,fx):
        left,bound,fx_expand = self.fist_step(x,t,fx)
        y = y/bound.unsqueeze(-1)
        return y


