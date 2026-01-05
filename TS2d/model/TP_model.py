from .MLP import MLPLinear_res
import torch
import torch.nn as nn
from .H_model import TS_H_model

class FourierFeatureMap(nn.Module):
    def __init__(self,in_feature= 3, num_features=64):
        super().__init__()
        self.B = torch.randn(in_feature, num_features).cuda()

    def forward(self, xy):
        proj = 2 * torch.pi * xy @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class TP_model(nn.Module):
    def __init__(self, input_dim=2, condition_dim=3, num_layers=6, embed_dim=128):
        super().__init__()
        self.ffm_x = FourierFeatureMap(in_feature=input_dim,num_features=int(embed_dim // 2))
        self.ffm_cond = FourierFeatureMap(in_feature=condition_dim,num_features=int(embed_dim // 2))
        self.mlp_deep = MLPLinear_res(embed_dim, 2, num_layers=num_layers,use_residual=True)

        self.H_model = TS_H_model(embed_dim=512 + 128)


    def forward(self, x, condition):
        condition = self.ffm_cond(condition).unsqueeze(1).expand(-1, x.size(1), -1)
        x = self.ffm_x(x)
        fx = x*condition
        x = self.mlp_deep(fx)
        T, P = torch.unbind(x , 2)
        return T, P