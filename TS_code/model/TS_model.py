import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from .MLP import MLPLinear_res,MLP_conv
from .inital_weight import WeightInitializer

class TransBlock(nn.Module):
    def __init__(self, input_dim, embed_dim,  out_dim,num_heads = 2,
                 Trans_dropout=0.0,
                 ffn_nonlinearity=F.gelu,
                 is_last=False,
                 out_mlp_layers=1,
                 out_nonlinearity=nn.Identity(),
                 use_residual=True,
                 norm_type='layernorm',
                 init_type='xavier_normal',
                 ffn_layers=2,
                 **kwargs):
        super(TransBlock, self).__init__()
        self.weight_initializer = WeightInitializer(init_type=init_type)
        self.mha = MultiHeadAttention_operator(input_dim, embed_dim, num_heads, **kwargs)  # 传递参数
        self.ffn = MLPLinear_res(in_channels=embed_dim, num_layers=ffn_layers,
                                 out_channels=embed_dim, dropout=Trans_dropout,
                                 non_linearity=ffn_nonlinearity, use_residual=use_residual)
        self.weight_initializer.apply(self.ffn)
        if norm_type == 'layernorm':
            self.norm1 = nn.LayerNorm(input_dim)
            self.norm2 = nn.LayerNorm(embed_dim)
        elif norm_type == 'none':
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        else:
            raise ValueError("norm_type must be 'layernorm', or 'none'")
        self.is_last = is_last
        if is_last:
            self.out = MLPLinear_res(in_channels=embed_dim, out_channels=out_dim,
                                     dropout=Trans_dropout, non_linearity=out_nonlinearity,
                                     num_layers=out_mlp_layers, use_residual=use_residual)
            self.weight_initializer.apply(self.out)

    def forward(self, q,k = None,v = None):
        if k is None:
            k = q
        if v is None:
            v = q
        mha_output = self.mha( q,k,v)
        mha_output = self.norm1(mha_output + q)
        mlp_output = self.ffn(mha_output)
        output = mlp_output
        if self.is_last:
            output = self.out(output)
        return output


class MultiHeadAttention_operator(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, **kwargs):
        super(MultiHeadAttention_operator, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.query_mlp = MLPLinear_res(in_channels=self.input_dim, num_layers=kwargs.get('q_layers', 1),
                                       out_channels=embed_dim,
                                       dropout=kwargs.get('dropout', 0.0),
                                       non_linearity=kwargs.get('nonlinearity', F.gelu))
        self.key_mlp = MLPLinear_res(in_channels=self.input_dim, num_layers=kwargs.get('k_layers', 1),
                                     out_channels=embed_dim,
                                     dropout=kwargs.get('dropout', 0.0),
                                     non_linearity=kwargs.get('nonlinearity', F.gelu))
        self.value_mlp = MLPLinear_res(in_channels=self.input_dim, num_layers=kwargs.get('v_layers', 1),
                                       out_channels=embed_dim,
                                       dropout=kwargs.get('dropout', 0.0),
                                       non_linearity=kwargs.get('nonlinearity', F.gelu))
        self.merge_att = MLPLinear_res(in_channels=num_heads, num_layers=kwargs.get('merge_att_layer', 1),
                                       out_channels=num_heads,
                                       dropout=kwargs.get('merge_dropout', 0.0),
                                       non_linearity=kwargs.get('merge_nonlinearity', F.gelu))

        self.weight_initializer = WeightInitializer(init_type=kwargs.get('init_type', 'xavier_normal'))
        self.weight_initializer.apply(self.query_mlp)
        self.weight_initializer.apply(self.key_mlp)
        self.weight_initializer.apply(self.value_mlp)
        self.weight_initializer.apply(self.merge_att)

    def forward(self, q,k= None,v= None):
        if k is None:
            k = q
        if v is None:
            v = q
        batch_size, seq_len = v.size(0), v.size(1)
        seq_q_len = q.size(1)
        query = self.query_mlp(q)
        key = self.key_mlp(k)
        value = self.value_mlp(v)
        query = query.view(batch_size, seq_q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32))
        attention_weights = F.softmax(scores, dim=-1)
        concatenated_attention = attention_weights.permute(0, 2, 3, 1)
        transformed_attention = self.merge_att(concatenated_attention)
        attention_weights = transformed_attention.permute(0, 3, 1, 2)
        output = torch.matmul(attention_weights, value)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_q_len, self.embed_dim)
        return output


def func_expand(x,A,B,C):
    result = A / (1 + torch.exp(-B * x + B * C))
    result = result*15
    return result


def expand_tensor(tensor, n):
  batchsize, m = tensor.shape
  expanded_tensor = tensor.repeat_interleave(n, dim=0).reshape(batchsize, n, m)
  return expanded_tensor
class Model(nn.Module):
    def __init__(self, embed_dim,Heat_path= None,sensor_mesh = 128, lora0=False,lora1 = False):
        super(Model, self).__init__()
        self.embed_dim = embed_dim
        self.weight_initializer = WeightInitializer(init_type='xavier_normal')
        self.A = MLPLinear_res(in_channels=1, num_layers=2, out_channels=embed_dim)
        self.B = MLPLinear_res(in_channels=1, num_layers=2, out_channels=embed_dim)
        self.C = MLPLinear_res(in_channels=1, num_layers=2, out_channels=embed_dim)
        self.D = MLPLinear_res(in_channels=1, num_layers=2, out_channels=embed_dim)
        self.fc_layers2 = MLPLinear_res(in_channels=embed_dim,out_channels=embed_dim,num_layers=2)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.mlp_layers1 = MLPLinear_res(in_channels=2,out_channels=embed_dim,num_layers=2,use_residual=True)
        self.mlp_layers2 = MLPLinear_res(in_channels=embed_dim,out_channels=1,num_layers=6,use_residual=True)
        self.mlp_layers3 = MLPLinear_res(in_channels=2,out_channels=embed_dim,num_layers=2,use_residual=True)
        self.mlp_layers4 = MLPLinear_res(in_channels=embed_dim,out_channels=1,num_layers=6,use_residual=True)
        self.mlp_layers5 = MLPLinear_res(in_channels=embed_dim,out_channels=1,num_layers=6,use_residual=True)
        self.mlp_layers6 = MLPLinear_res(in_channels=embed_dim,out_channels=embed_dim,num_layers=4,use_residual=True)

    def forward(self, x, t, heat_para):
        A,B,C,D= torch.chunk(heat_para, chunks=4, dim=1)
        A = self.A(A)
        B = self.B(B)
        C = self.C(C)
        func1 = A*B*C
        func1 = expand_tensor(func1,x.size(1))

        xt = torch.cat([x.unsqueeze(-1), t.unsqueeze(-1)], dim=-1)
        xt_input = self.mlp_layers1(xt)
        emed = xt_input*func1

        phi = self.mlp_layers2(emed).squeeze(-1)

        tao = t.clone().detach().requires_grad_(True)
        # tao = t

        phi_in = torch.cat([phi.unsqueeze(-1), tao.unsqueeze(-1)], dim=-1)
        phi_in = self.mlp_layers3(phi_in)
        func2 = self.fc_layers2(func1)+phi_in
        func2 = self.ln1(func2)
        out1 = phi_in*func2
        out1 = self.mlp_layers6(out1)+func2
        T = self.mlp_layers4(out1)
        v = self.mlp_layers5(out1).squeeze()

        expan_A = (0.89 + 0.96)/2
        expan_B = (1.1 + 1.65)/2 * 4
        expan_C = -(0.35 + 0.5)/2
        rou = func_expand(-T, expan_A, expan_B, expan_C)
        rou = rou.squeeze(-1)
        urou = v * rou
        return phi, T.squeeze(), rou, v, urou,tao