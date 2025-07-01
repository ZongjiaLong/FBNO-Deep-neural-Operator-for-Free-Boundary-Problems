import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.nn.init as init


class MLP_conv(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels=None,
            hidden_channels=None,
            n_layers=2,
            non_linearity=F.gelu,
            dropout=0.0,
            use_residual=False):
        super().__init__()
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.hidden_channels = (
            max(self.in_channels, self.out_channels) if hidden_channels is None else hidden_channels
        )
        self.non_linearity = non_linearity
        self.use_residual = use_residual
        self.dropout = (
            nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])
            if dropout > 0.0
            else None
        )
        self.fcs = nn.ModuleList()
        for i in range(n_layers):
            if i == 0 and i == (n_layers - 1):
                self.fcs.append(nn.Conv1d(self.in_channels, self.out_channels, 1))
            elif i == 0 and i != (n_layers - 1):
                self.fcs.append(nn.Conv1d(self.in_channels, self.hidden_channels, 1))
            elif i == (n_layers - 1):
                self.fcs.append(nn.Conv1d(self.hidden_channels, self.out_channels, 1))
            else:
                self.fcs.append(nn.Conv1d(self.hidden_channels, self.hidden_channels, 1))

    def forward(self, x):
        residual = x if self.use_residual else None
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < self.n_layers - 1:
                x = self.non_linearity(x)
            if self.dropout is not None:
                x = self.dropout[i](x)

        if residual is not None and x.shape == residual.shape:
            x = x + residual

        return x

class MLPLinear(torch.nn.Module):
    def __init__(self, layers, non_linearity=F.gelu, dropout=0.0):
        super().__init__()
        self.n_layers = len(layers) - 1
        assert self.n_layers >= 1
        self.fcs = nn.ModuleList()
        self.non_linearity = non_linearity
        self.dropout = (
            nn.ModuleList([nn.Dropout(dropout) for _ in range(self.n_layers)])
            if dropout > 0.0
            else None
        )

        for j in range(self.n_layers):
            self.fcs.append(nn.Linear(layers[j], layers[j + 1]))
    def forward(self, x):
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < self.n_layers - 1:
                x = self.non_linearity(x)
            if self.dropout is not None:
                x = self.dropout[i](x)
        return x




class MLPLinear_res(nn.Module):
    def __init__(
            self, in_channels,
            out_channels=None,
            hidden_channels=None,
            num_layers=1,
            non_linearity=F.gelu,
            dropout=0.0,
            use_residual=False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.hidden_channels = (
            max(self.in_channels, self.out_channels) if hidden_channels is None else hidden_channels
        )
        self.num_layers = num_layers
        assert self.num_layers >= 1
        self.fcs = nn.ModuleList()
        self.non_linearity = non_linearity
        self.dropout = (
            nn.ModuleList([nn.Dropout(dropout) for _ in range(self.num_layers)])
            if dropout > 0.0
            else None
        )

        self.use_residual = use_residual
        self.fcs.append(nn.Linear(self.in_channels, self.hidden_channels if self.num_layers > 1 else self.out_channels))
        for _ in range(self.num_layers - 2):
            self.fcs.append(nn.Linear(self.hidden_channels, self.hidden_channels))
        if self.num_layers > 1:
            self.fcs.append(nn.Linear(self.hidden_channels, self.out_channels))

    def forward(self, x):
        residual = x
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < self.num_layers - 1:
                x = self.non_linearity(x);
                if self.use_residual and i > 0:
                    if x.shape == residual.shape:
                        x = x + residual
                    residual = x
            if self.dropout is not None:
                x = self.dropout[i](x)
        return x


class FourierFeatureMapping(nn.Module):
    def __init__(self, input_dim, mapping_size, scale=10.0):
        super(FourierFeatureMapping, self).__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.scale = scale

        # 随机生成傅里叶特征的权重矩阵
        self.B = torch.randn((input_dim, mapping_size)) * scale

    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class AdaptiveFourierFeatureMapping(nn.Module):
    def __init__(self, input_dim, mapping_size, scale=10.0):
        super(AdaptiveFourierFeatureMapping, self).__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.scale = scale

        # 将频率矩阵 B 设置为可学习参数
        self.B = nn.Parameter(torch.randn((input_dim, mapping_size)) * scale)

    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class FourierFeatureMap(nn.Module):
    def __init__(self,in_dim =2,  num_features=64, scale=1.0):
        super().__init__(),
        self.B = nn.Parameter(torch.randn(in_dim, num_features) * scale)

    def forward(self, xy):
        proj = 2 * torch.pi * xy @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
class MultiScaleFourierFeatureMapping(nn.Module):
    def __init__(self, input_dim, mapping_size, num_scales=3):
        super(MultiScaleFourierFeatureMapping, self).__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.num_scales = num_scales

        # 初始化多个频率矩阵
        self.B_list = nn.ParameterList([
            nn.Parameter(torch.randn((input_dim, mapping_size)) * (2 ** i))
            for i in range(num_scales)
        ])

    def forward(self, x):
        features = []
        for B in self.B_list:
            x_proj = 2 * np.pi * x @ B
            features.append(torch.sin(x_proj))
            features.append(torch.cos(x_proj))
        return torch.cat(features, dim=-1)