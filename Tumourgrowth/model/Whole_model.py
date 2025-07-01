import torch
import torch.nn as nn
import torch.nn.functional as F
from .MLP import MLPLinear_res

class pretrain_Model(nn.Module):
    def __init__(self, embed_dim, num_points=500):
        super(pretrain_Model, self).__init__()
        self.num_points = num_points
        self.ffm = FourierFeatureMap(num_features=int(embed_dim // 2))
        self.ffm_de = FourierFeatureMap(num_features=int(embed_dim // 2))
        self.domain_embed = FourierFeatureMap(in_feature=80,num_features=int(embed_dim // 2))
        self.domain_embed_de = FourierFeatureMap(in_feature=80,num_features=int(embed_dim // 2))
        self.fc_1 = MLPLinear_res(in_channels=embed_dim, out_channels=embed_dim, num_layers=4, use_residual=True)
        self.fc_1_de = MLPLinear_res(in_channels=embed_dim, out_channels=embed_dim, num_layers=4, use_residual=True)
        self.fc_in = MLPLinear_res(in_channels=1, out_channels=embed_dim, num_layers=4, use_residual=True)
        self.fc_in_de = MLPLinear_res(in_channels=1, out_channels=embed_dim, num_layers=4, use_residual=True)
        self.fc_out = MLPLinear_res(in_channels=embed_dim, out_channels=2, num_layers=12, use_residual=True)
        self.fc_out_de = MLPLinear_res(in_channels=embed_dim, out_channels=2, num_layers=12, use_residual=True)

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln_de = nn.LayerNorm(embed_dim)
    def forward(self, t, fx,init_domain, x = None,y = None):
        de_input = torch.cat([x.unsqueeze(2), y.unsqueeze(2),t.unsqueeze(2)], dim=2)
        de_input = self.ffm_de(de_input)
        de_input = self.fc_1_de(de_input)
        fx_de = self.fc_in_de(fx).unsqueeze(1).repeat(1, de_input.shape[1], 1)
        domain_de = self.domain_embed_de(init_domain).repeat(1, de_input.shape[1], 1)
        fx = fx_de * domain_de
        fx_de = fx*de_input
        fx_de = self.ln_de(fx_de)
        out_de = self.fc_out_de(fx_de)
        de_x, de_y = torch.unbind(out_de, 2)
        return de_x, de_y,fx


class FourierFeatureMap(nn.Module):
    def __init__(self,in_feature= 3, num_features=64, scale=1.0):
        super().__init__()
        self.B = torch.randn(in_feature, num_features).cuda()

    def forward(self, xy):
        proj = 2 * torch.pi * xy @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_layers, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):

        x = x.permute(1, 0, 2)  # Swap batch and sequence dimensions
        for layer in self.layers:
            x = layer(x)
        x = x.permute(1, 0, 2)  # Swap back to original shape
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, hid_dim = 64):
        super().__init__()
        self.q_mid = MLPLinear_res(in_channels=embed_dim, out_channels=hid_dim, num_layers=1)
        self.e_mid = MLPLinear_res(in_channels=embed_dim, out_channels=hid_dim, num_layers=1)
        self.w_mid = MLPLinear_res(in_channels=embed_dim, out_channels=hid_dim, num_layers=2)
        self.q_proj = nn.Linear(hid_dim, hid_dim)
        self.k_proj = nn.Linear(hid_dim, hid_dim)
        self.v_proj = nn.Linear(hid_dim, hid_dim)
        self.fc_out = nn.Linear(hid_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, fx,x = None):
        # x shape: (seq_len, batch_size, embed_dim)
        if x is None:
            x = fx
        q_mid = self.q_mid(x)
        e_mid = self.e_mid(fx)
        w_mid = self.w_mid(fx)
        weight = torch.einsum('btlm,btln->btmn', q_mid, e_mid)
        attn = self.softmax(weight)
        w = torch.einsum('btpm,btms->btps', w_mid, attn)
        w = self.fc_out(w)
        return w
class Transolver_block(nn.Module):

    def __init__(
            self,
            embed_dim: int,
            hid_dim: int,

    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.Attn = TransformerBlock(embed_dim, hid_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = MLPLinear_res(embed_dim,embed_dim,num_layers=2)


    def forward(self, fx,x = None):
        if x is None:
            fx = self.Attn(self.ln_1(fx)) + fx
        else:
            fx = self.Attn(self.ln_1(fx),x) + x
        fx = self.mlp(self.ln_2(fx)) + fx
        return fx
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # Learnable gain parameter

    def _norm(self, x: torch.Tensor):
        # RMS = sqrt(mean(x^2) + eps)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        return self.weight * self._norm(x.float()).type_as(x)
class Embed_block(nn.Module):
    def __init__(self, embed_dim,hid_dim = 68,num_heads = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.mha = nn.MultiheadAttention(hid_dim, num_heads)
        self.in_linear_t= nn.Linear(self.embed_dim, hid_dim)
        self.in_linear_x = nn.Linear(self.embed_dim, hid_dim)
        self.rn1_t = RMSNorm(hid_dim)
        self.rn2_t = RMSNorm(hid_dim)
        self.rn1_f = RMSNorm(hid_dim)
        self.rn2_f = RMSNorm(hid_dim)
        self.ln1 = nn.LayerNorm(hid_dim)
        self.ln2 = nn.LayerNorm(hid_dim)
        self.ln_t = nn.LayerNorm(embed_dim)
        self.ln_f = nn.LayerNorm(embed_dim)
        self.t_out = MLPLinear_res(in_channels=hid_dim, out_channels=embed_dim, num_layers=2)
        self.fx_out = MLPLinear_res(in_channels=hid_dim, out_channels=embed_dim, num_layers=2)
    def forward(self, t,fx):
        t = self.ln_t(t)
        fx = self.ln_f(fx)
        t_in = self.in_linear_t(t)
        fx_in = self.in_linear_x(fx)
        t1 = self.rn1_t(t_in)
        t2 = self.rn2_t(t_in)
        fx1 = self.rn1_f(fx_in)
        fx2 = self.rn2_f(fx_in)
        q = t1*fx1
        k = t2*fx2
        v = t_in*fx_in
        att_out,_ = self.mha(q, k, v)
        t_out = self.ln1(att_out)
        fx_out = self.ln2(att_out)
        t_out = self.t_out(t_out)+t
        fx_out = self.fx_out(fx_out)+fx
        return t_out, fx_out

class Model(nn.Module):
    def __init__(self, embed_dim,pretrain_path=None):
        super(Model, self).__init__()
        self.pretrain_model = pretrain_Model(256, 500)
        self.pretrain_model.load_state_dict(torch.load(pretrain_path), strict=True)
        self.ffm = FourierFeatureMap(num_features=int(embed_dim // 2))
        self.fc_1 = MLPLinear_res(in_channels=embed_dim, out_channels=embed_dim, num_layers=6, use_residual=True)
        self.fc_out = MLPLinear_res(in_channels=embed_dim, out_channels=1, num_layers=10, use_residual=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        for param in self.pretrain_model.parameters():
            param.requires_grad = False
    def forward(self,x ,y , t, fx,init_domain):
        x_prime,y_prime,fx = self.pretrain_model(x = x ,y = y , t = t, fx = fx,init_domain = init_domain)
        input = torch.cat([x_prime.unsqueeze(2), y_prime.unsqueeze(2),t.unsqueeze(2)], dim=2)
        input = self.ffm(input)
        func = self.fc_1(fx)
        fx = func*input
        fx = self.ln1(fx)
        out = self.fc_out(fx)
        return out