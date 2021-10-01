import torch
import torch.nn as nn
import numpy as np
from typing import List


class GAN(nn.Module):

    def __init__(self,
                 input_shape: List[int],
                 disc_conv_filters: List[int],
                 disc_conv_kernels: List[int],
                 disc_conv_strides: List[int],
                 disc_batch_norm_mom: float,
                 disc_dropout_prob: float,
                 gen_unflattened_shape: List[int],
                 gen_upsample_scale: List[int],
                 gen_conv_filters: List[int],
                 gen_conv_kernels: List[int],
                 gen_conv_strides: List[int],
                 gen_batch_norm_mom: float,
                 gen_dropout_prob: float,
                 z_dim: int
                ):
        super().__init__()
        self.generator = Generator(z_dim=z_dim,
                                   unflattened_shape=gen_unflattened_shape,
                                   upsample_scale=gen_upsample_scale,
                                   filters=gen_conv_filters,
                                   kernels=gen_conv_kernels,
                                   strides=gen_conv_strides,
                                   batch_norm_mom=gen_batch_norm_mom,
                                   dropout_prob=gen_dropout_prob
                                  )
        self.discriminator = Discriminator(input_shape=input_shape,
                                           filters=disc_conv_filters,
                                           kernels=disc_conv_kernels,
                                           strides=disc_conv_strides,
                                           batch_norm_mom=disc_batch_norm_mom,
                                           dropout_prob=disc_dropout_prob
                                          )

        self._build()

    def _build(self):
        pass
    
    def forward(self, batch):
        pass


class Generator(nn.Module):

    def __init__(self,
                 z_dim: int,
                 unflattened_shape: List[int],
                 upsample_scale: List[int],
                 filters: List[int],
                 kernels: List[int],
                 strides: List[int],
                 batch_norm_mom: float,
                 dropout_prob: float
                ):
        super().__init__()

        self.z_dim = z_dim
        self.unflattened_shape = unflattened_shape
        self.upsample_scale = upsample_scale
        self.filters = filters
        self.kernels = kernels
        self.strides = strides
        self.batch_norm_mom = batch_norm_mom
        self.dropout_prob = dropout_prob
        
        self.n_conv_layers = len(filters)
        
        self._build()
    
    def _build(self):
        conv_layers = []
        
        for i in range(self.n_conv_layers):
            kernel_size = self.kernels[i]
            padding = kernel_size // 2
            
            conv_layers.extend([
                nn.Upsample(scale_factor=self.upsample_scale[i]),
                nn.Conv2d(in_channels=self.unflattened_shape[i] if i == 0 else self.filters[i - 1],
                          out_channels=self.filters[i],
                          kernel_size=kernel_size,
                          stride=self.strides[i],
                          padding=padding
                         )
            ])
            
            if i < self.n_conv_layers - 1:
                if self.batch_norm_mom:
                    conv_layers.append(
                        nn.BatchNorm2d(num_features=self.filters[i],
                                       momentum=self.batch_norm_mom
                                      )
                    )
                
                conv_layers.append(nn.ReLU())
            else:
                conv_layers.append(nn.Tanh())
        
        dense_layers = [
            nn.Linear(in_features=self.z_dim,
                      out_features=np.prod(self.unflattened_shape)
                     )
        ]
        
        if self.batch_norm_mom:
            dense_layers.append(
                nn.BatchNorm1d(num_features=np.prod(self.unflattened_shape),
                               momentum=self.batch_norm_mom
                              )
            )
        
        dense_layers.append(nn.ReLU())
        
        if self.dropout_prob:
            dense_layers.append(
                nn.Dropout(p=self.dropout_prob)
            )
        
        dense_layers.append(
            nn.Unflatten(dim=1,
                         unflattened_size=self.unflattened_shape
                        )
        )
        
        self.model = nn.Sequential(*dense_layers, *conv_layers)
    
    def forward(self, batch):
        return self.model(batch)


class Discriminator(nn.Module):

    def __init__(self,
                 input_shape: List[int],
                 filters: List[int],
                 kernels: List[int],
                 strides: List[int],
                 batch_norm_mom: float,
                 dropout_prob: float
                ):
        super().__init__()
        
        self.input_shape = input_shape
        self.filters = filters
        self.kernels = kernels
        self.strides = strides
        self.batch_norm_mom = batch_norm_mom
        self.dropout_prob = dropout_prob
        
        self.n_conv_layers = len(filters)
        
        self._build()
    
    def _build(self):
        conv_layers = []
        
        for i in range(self.n_conv_layers):
            kernel_size = self.kernels[i]
            padding = kernel_size // 2
            
            conv_layers.append(
                nn.Conv2d(in_channels=self.input_shape[i] if i == 0 else self.filters[i - 1],
                          out_channels=self.filters[i],
                          kernel_size=kernel_size,
                          stride=self.strides[i],
                          padding=padding
                         )
            )
            
            if self.batch_norm_mom:
                conv_layers.append(
                    nn.BatchNorm2d(num_features=self.filters[i],
                                   momentum=self.batch_norm_mom
                                  )
                )
            
            conv_layers.append(nn.ReLU())
            
            if self.dropout_prob:
                conv_layers.append(
                    nn.Dropout2d(p=self.dropout_prob)
                )
        
        dummy_input = torch.unsqueeze(torch.zeros(self.input_shape), 0)
        unflattened_shape = nn.Sequential(*conv_layers)(dummy_input).shape[1:]
        
        self.model = nn.Sequential(*conv_layers,
                                   nn.Flatten(),
                                   nn.Linear(in_features=np.prod(unflattened_shape),
                                             out_features=1
                                            ),
                                   nn.Sigmoid()
                                  )
    
    def forward(self, batch):
        return self.model(batch)