import torch
import torch.nn as nn
import numpy as np
from typing import List
from fastai.torch_core import default_device


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
                
                conv_layers.append(nn.LeakyReLU())
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
        
        dense_layers.append(nn.LeakyReLU())
        
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
            
            conv_layers.append(nn.LeakyReLU())
            
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


class GANModule(nn.Module):

    def __init__(self,
                 generator: Generator = None,
                 discriminator: Discriminator = None,
                 gen_mode: bool = False
                ):
        super().__init__()
        
        if generator is not None:
            self.generator = generator
        
        if discriminator is not None:
            self.discriminator = discriminator
    
    def switch(self, gen_mode=None):
        self.gen_mode = not self.gen_mode if gen_mode is None else gen_mode
    
    def forward(self, *args):
        if self.gen_mode:
            return self.generator(*args)
        
        return self.discriminator(*args)


class GANLoss(GANModule):
    
    def __init__(self,
                 gen_loss_func,
                 disc_loss_func,
                 gan: GANModule
                ):
        super().__init__()
        self.gen_loss_func = gen_loss_func
        self.disc_loss_func = disc_loss_func
        self.gan = gan
        
        def generator(self, gen_output, real_image):
            disc_pred = self.gan.discriminator(gen_output)
            
            return gen_loss_func(disc_pred)
        
        def discriminator(self, real_pred, noise):
            generated = self.gan.generator(noise)
            fake_pred = self.gan.discriminator(generated)
            
            return disc_loss_func(fake_pred, real_pred)