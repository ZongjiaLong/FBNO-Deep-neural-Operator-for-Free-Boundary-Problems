from .MLP import MLPLinear_res
import torch
import torch.nn as nn


class FourierFeatureMap(nn.Module):
    def __init__(self,in_feature= 3, num_features=64):
        super().__init__()
        self.B = torch.randn(in_feature, num_features).cuda()

    def forward(self, xy):
        proj = 2 * torch.pi * xy @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class TS_H_model(nn.Module):
    def __init__(self, input_dim=2, condition_dim=3, num_layers=6, embed_dim=128):
        super().__init__()
        self.ffm_x = FourierFeatureMap(in_feature=input_dim,num_features=int(embed_dim // 2))
        self.ffm_de = FourierFeatureMap(in_feature=input_dim,num_features=int(embed_dim // 2))

        self.ffm_cond = FourierFeatureMap(in_feature=condition_dim,num_features=int(embed_dim // 2))
        self.ffm_cond_de = FourierFeatureMap(in_feature=condition_dim,num_features=int(embed_dim // 2))
        self.mlp_deep = MLPLinear_res(embed_dim, 2, num_layers=num_layers,use_residual=True)
        self.mlp_deep_de = MLPLinear_res(embed_dim, 2, num_layers=num_layers,use_residual=True)

    def forward(self, condition,x = None):
        condition_en = self.ffm_cond(condition).unsqueeze(1).expand(-1, x.size(1), -1)
        if x is not None:
            x = self.ffm_x(x)
            fx = x*condition_en
            phi = self.mlp_deep(fx)

            phi_de = self.ffm_de(phi)
            fx_de = self.ffm_cond_de(condition).unsqueeze(1).expand(-1, x.size(1), -1)
            fx_in_de = phi_de*fx_de
            x_out = self.mlp_deep_de(fx_in_de)
            return phi,x_out

    def inv(self,phi,condition):
        phi_de = self.ffm_de(phi)
        fx_de = self.ffm_cond_de(condition).unsqueeze(1).expand(-1, phi.size(1), -1)
        fx_in_de = phi_de * fx_de
        x_out = self.mlp_deep_de(fx_in_de)
        return x_out