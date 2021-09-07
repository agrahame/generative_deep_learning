import torch.nn as nn
import torch
import numpy as np
from typing import List


class AutoEncoder(nn.Module):
    
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
        super().__init__()
        
        self.input_dim = input_dim
        self.conv_filters = conv_filters
        self.conv_t_filters = conv_t_filters
        self.conv_kernel_sizes = kernel_sizes
        self.conv_strides = strides
        self.conv_t_kernel_sizes = kernel_sizes[::-1]
        self.conv_t_strides = strides[::-1]
        self.z_dim = z_dim
        
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        
        self.num_conv_layers = len(self.conv_filters)
        self.num_conv_t_layers = len(self.conv_t_filters)
        
        self._build()
    
    def _build(self):
        # Encode
        conv_layers = []
        conv_paddings = []
        # assuming square inputs
        conv_input_sizes = [self.input_dim[1]]
        
        for i in range(self.num_conv_layers):
            # To ensure that the output size is the same as the input size
            # when stride is 1 use p = (k - 1) / 2. I assume k is odd. This is
            # equivalent to p = k // 2 when k is odd.
            kernel_size = self.conv_kernel_sizes[i]
            conv_paddings.append(int((kernel_size - 1) / 2))
            
            stride = self.conv_strides[i]
            padding = conv_paddings[i]
            
            if i < self.num_conv_layers - 1:
                # This is needed to calculate the output_padding param in the conv_t layers
                output_size = ((conv_input_sizes[i] + 2*padding - kernel_size) // stride) + 1
                conv_input_sizes.append(output_size)
            
            conv_layers.append(
                nn.Conv2d(in_channels=self.input_dim[0] if i == 0 else self.conv_filters[i - 1],
                          out_channels=self.conv_filters[i],
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=conv_paddings[i]
                         )
            )
            
            conv_layers.append(nn.LeakyReLU())
            
            if self.use_batch_norm:
                conv_layers.append(nn.BatchNorm2d(num_features=self.conv_filters[i]))
        
        conv = nn.Sequential(*conv_layers)
        
        # Using dummy batch size of 64
        zeros = torch.zeros([64, *self.input_dim])
        shape_before_flatten = conv(zeros).shape[1:]
        
        self.encoder = nn.Sequential(*conv_layers,
                                     nn.Flatten(),
                                     nn.Linear(in_features=np.prod(shape_before_flatten),
                                               out_features=self.z_dim
                                              )
                                    )
        
        # Decode
        conv_t_layers = []
        
        for i in range(self.num_conv_t_layers):
            # The padding needed in the conv transpose to get back the
            # shape of the input to the conv with padding p is given by
            # p' = k - p - 1. This is explained in the paper:
            # https://arxiv.org/pdf/1603.07285.pdf
            # Note the need to go in reverse when reading the conv params as
            # the convolution transpose layer at position i does not correspond
            # to the convolution layer at position i
            conv_padding = conv_paddings[-i-1]
            kernel_size = self.conv_t_kernel_sizes[i]
            conv_t_padding = kernel_size - conv_padding - 1
            
            # This handles the ambiguity of multiple input sizes being mapped to
            # multiple output sizes when stride > 1 as described in the paper above
            stride = self.conv_t_strides[i]
            output_padding = (conv_input_sizes[-i-1] + 2*conv_padding - kernel_size) % stride
            
            conv_t_layers.append(
                nn.ConvTranspose2d(in_channels=self.conv_filters[-1] if i == 0 else self.conv_t_filters[i - 1],
                                   out_channels=self.conv_t_filters[i],
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=conv_t_padding,
                                   output_padding=output_padding
                                  )
            )
            
            if i < self.num_conv_t_layers - 1:
                conv_t_layers.append(nn.LeakyReLU())
            
                if self.use_batch_norm:
                    conv_t_layers.append(nn.BatchNorm2d())
            else:
                conv_t_layers.append(nn.Sigmoid())
        
        self.decoder = nn.Sequential(nn.Linear(in_features=self.z_dim,
                                               out_features=np.prod(shape_before_flatten)
                                              ),
                                     nn.Unflatten(1, shape_before_flatten),
                                     *conv_t_layers
                                    )
        
        # self.model = nn.Sequential(self.encoder, self.decoder)
        
    
    def forward(self, batch):
        return nn.Sequential(self.encoder, self.decoder)(batch) # self.model(batch)
