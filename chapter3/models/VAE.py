import torch.nn as nn
import torch
import numpy as np
from .AE import AutoEncoder
from typing import List


class VariationalAutoEncoder(AutoEncoder):
    
    def __init__(self,
                 input_dim: List[int],
                 conv_filters: List[int],
                 conv_t_filters: List[int],
                 kernel_sizes: List[int],
                 strides: List[int],
                 z_dim: int,
                 use_batch_norm: bool = False,
                 use_dropout: bool = False
                ):
        super().__init__(input_dim=input_dim,
                         conv_filters=conv_filters,
                         conv_t_filters=conv_t_filters,
                         kernel_sizes=kernel_sizes,
                         strides=strides,
                         z_dim=z_dim,
                         use_batch_norm=use_batch_norm,
                         use_dropout=use_dropout
                        )
    
    def _build(self):
        super()._build()
        enc_last_removed = list(self.encoder.children())[:-1]
        self.encoder = nn.Sequential(*enc_last_removed,
                                     VAELayer(in_features=np.prod(self.shape_before_flatten),
                                              out_features=self.z_dim
                                             )
                                    )


class VAELayer(nn.Module):
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.bilinear = VAELinear(in_features, out_features)
        self.sampler = Sampler()
    
    def forward(self, batch):
        self.mu, self.log_var = self.bilinear(batch)
        z = self.sampler(self.mu, self.log_var)
        
        return z

        
class VAELinear(nn.Module):
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.mu = nn.Linear(in_features, out_features)
        self.log_var = nn.Linear(in_features, out_features)
    
    def forward(self, batch):
        return self.mu(batch), self.log_var(batch)


class Sampler(nn.Module):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, mu, log_var):
        sig = torch.exp(0.5 * log_var).to(device='cuda')
        eps = torch.normal(mean=0.0, std=1.0, size=mu.shape).to(device='cuda')
        
        return mu + sig * eps
