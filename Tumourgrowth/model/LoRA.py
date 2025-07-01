import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, linear: nn.Linear,  rank = 4, scale = 3, dropout = 0):
        super(LoRALinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.dropout = dropout
        self.scale = scale

        if linear.weight is not None:
            self.weight = nn.Parameter(linear.weight.clone().detach().requires_grad_(True))
        else:
            self.register_parameter('weight', None)

        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.clone().detach().requires_grad_(True))
        else:
            self.register_parameter('bias', None)

        if rank > 0:
            self.lora_b = nn.Parameter(torch.zeros(rank, in_features))
            self.lora_a = nn.Parameter(torch.zeros(out_features, rank))
        else:
            self.lora_a = None
            self.lora_b = None

        if self.dropout > 0:
            self.dropout_layer = nn.Dropout(self.dropout)
        else:
            self.dropout_layer = nn.Identity()
        self.initial_weights()
    def initial_weights(self):
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

        if self.weight.requires_grad is not None:
            self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def forward(self, x):
        if self.rank > 0 :
            output = F.linear(x, self.weight + self.lora_a @ self.lora_b * self.scale, self.bias)
            output = self.dropout_layer(output)
            return output
        else:
            output = F.linear(x, self.weight, self.bias)
            return self.dropout_layer(output)