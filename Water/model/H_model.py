import torch
import torch.nn as nn

from .MLP import MLPLinear_res,FourierFeatureMap
from .phi_model import Model as Phi_Model

class Model(nn.Module):
    def __init__(self, embed_dim,device = 'cuda',phi_model_path = None):
        super(Model, self).__init__()

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ffm_time = FourierFeatureMap(1,embed_dim,device)
        self.ffm_space = FourierFeatureMap(3,embed_dim)
        self.ffm_fx = FourierFeatureMap(2,embed_dim)
        self.corrdinate_mlp = MLPLinear_res(embed_dim, embed_dim, num_layers=5)
        self.fx_mlp = MLPLinear_res(embed_dim, embed_dim, num_layers=5)
        self.fc_deep = MLPLinear_res(embed_dim, embed_dim, num_layers=8)
        self.fc_invnorm = MLPLinear_res(embed_dim, 1, num_layers=2)
        self.phi_model = Phi_Model(256,device)
        self.phi_model.load_state_dict(torch.load(phi_model_path))
        for param in self.phi_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def _get_processed_y(self, x, y, t, fx):
        return self.phi_model.inv_first_step(x, y, t, fx)

    def forward(self, x,y,t,fx):
        y = self._get_processed_y(x, y, t, fx)
        time = self.ffm_time(t).unsqueeze(1).expand(-1, x.shape[1], -1)
        z = torch.ones_like(x,device=x.device,dtype=x.dtype)
        x = self.ffm_space(torch.cat([x,y,z],dim=-1))
        x = x*time
        x = self.corrdinate_mlp(x)
        fx = self.ffm_fx(fx)
        fx = self.fx_mlp(fx).unsqueeze(1).expand(-1, x.shape[1], -1)
        fx_in = self.ln1(x*fx)
        fx_out = self.fc_deep(fx_in)
        z = self.fc_invnorm(fx_out)
        return z.squeeze(-1)



