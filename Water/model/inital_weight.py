import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightInitializer:
    def __init__(self, init_type='xavier_normal', mode='fan_in', nonlinearity='leaky_relu'):
        self.init_type = init_type
        self.mode = mode
        self.nonlinearity = nonlinearity

    def init_weights(self, module):
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            if self.init_type == 'none':
                # Do nothing, skip the initialization
                return
            elif self.init_type == 'kaiming_normal':
                nn.init.kaiming_normal_(module.weight, mode=self.mode, nonlinearity=self.nonlinearity)
            elif self.init_type == 'kaiming_uniform':
                nn.init.kaiming_uniform_(module.weight, mode=self.mode, nonlinearity=self.nonlinearity)
            elif self.init_type == 'xavier_normal':
                nn.init.xavier_normal_(module.weight)
            elif self.init_type == 'xavier_uniform':
                nn.init.xavier_uniform_(module.weight)
            elif self.init_type == 'orthogonal':
                nn.init.orthogonal_(module.weight)
            elif self.init_type == 'uniform':
                nn.init.uniform_(module.weight, a=-0.1, b=0.1)
            elif self.init_type == 'normal':
                nn.init.normal_(module.weight, mean=0.0, std=0.1)
            elif self.init_type == 'constant':
                nn.init.constant_(module.weight, val=0.9)
            elif self.init_type == 'zeros':
                nn.init.zeros_(module.weight)
            elif self.init_type == 'ones':
                nn.init.ones_(module.weight)
            else:
                raise ValueError(f"Unsupported initialization type: {self.init_type}")

            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def apply(self, model):
        model.apply(self.init_weights)