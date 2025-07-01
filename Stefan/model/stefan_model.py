import torch
import torch.nn as nn
import torch.nn.functional as F
from .MLP import MLPLinear_res,FourierFeatureMap
from .inital_weight import WeightInitializer


class Model(nn.Module):
    def __init__(self, embed_dim):
        super(Model, self).__init__()
        self.embed_dim = embed_dim
        self.FFM = FourierFeatureMap(2,int(self.embed_dim//2))
        self.phi_in = FourierFeatureMap(2,int(self.embed_dim//2))
        self.xt_in = MLPLinear_res(in_channels=embed_dim, out_channels=embed_dim,num_layers=4)
        self.fx = MLPLinear_res(in_channels=embed_dim, out_channels=embed_dim,num_layers=2)
        self.period_in = MLPLinear_res(in_channels=1, out_channels=embed_dim,num_layers=2)
        self.mag_in = MLPLinear_res(in_channels=1, out_channels=embed_dim,num_layers=2)
        self.f_deep = MLPLinear_res(in_channels=embed_dim, out_channels=1,num_layers=5)
        self.phi_deep = MLPLinear_res(in_channels=embed_dim, out_channels=1,num_layers=5)
        self.mag_ln = nn.LayerNorm(embed_dim)
        self.period_ln = nn.LayerNorm(embed_dim)
        self.ln1 = nn.LayerNorm(embed_dim)
    def forward(self, mag,period,t=None,x = None,phi_t = None):
        if t is not None:
            mag = self.mag_in(mag).unsqueeze(1).repeat(1, t.shape[1], 1)
            period = self.period_in(period).unsqueeze(1).repeat(1, t.shape[1], 1)
        else :
            mag = self.mag_in(mag).unsqueeze(1).repeat(1, phi_t.shape[1], 1)
            period = self.period_in(period).unsqueeze(1).repeat(1, phi_t.shape[1], 1)
        mag = self.mag_ln(mag)
        period = self.period_ln(period)
        fx_in = mag * period
        if x is not None:
            xt = torch.cat([x.unsqueeze(2),t.unsqueeze(2)],dim=-1)
            xt = self.FFM(xt)
            xt = self.xt_in(xt)
            emed = fx_in*xt
            phi = self.phi_deep(emed).squeeze(-1)
            tao = t.clone().detach().requires_grad_(True)
            phi_t = torch.cat([phi.unsqueeze(2),tao.unsqueeze(2)],dim=-1)
        else:
            phi_t = phi_t
            phi = None
            tao = None
        phi_t = self.phi_in(phi_t)
        fx_op_in = self.fx(fx_in)+phi_t
        fx_op_in = self.ln1(fx_op_in)
        op_in = fx_op_in*phi_t
        out = self.f_deep(op_in)
        return phi,out,tao
