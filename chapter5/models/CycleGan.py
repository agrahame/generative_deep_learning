import torch
import torch.nn as nn
from typing import List


class Sampler(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 dropout_rate: int = 0
                ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dropout_rate = dropout_rate
        
        self._build()
    
    def _build(self):
        layers = [
            nn.Conv2d(in_channels=self.in_channels,
                      out_channels=self.out_channels,
                      kernel_size=self.kernel_size,
                      stride=self.stride,
                      padding=self.kernel_size // 2
                     ),
            nn.InstanceNorm2d(num_features=self.out_channels),
            nn.ReLU()
        ]
        
        if self.dropout_rate:
            layers.append(
                nn.Dropout(p=self.dropout_rate)
            )
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, inputs):
        return self.model(inputs)


class Downsampler(Sampler):
    pass

class Upsampler(Sampler):
    
    def _build(self):
        super()._build()
        layers = list(self.model.children())
        self.model = nn.Sequential(nn.Upsample(scale_factor=2), *layers)
    
    def forward(self, inputs, skip_input):
        output = self.model(inputs)
        
        return torch.cat((output, skip_input), dim=1)


class GeneratorUNet(nn.Module):
    
    def __init__(self,
                 input_shape: List[int],
                 filters: List[int],
                 ds_kernel_size: int,
                 us_kernel_size: int,
                 ds_stride: int,
                 us_stride,
                 dropout_rate: int = 0):
        super().__init__()
        self.skip_inputs = {}
        self.input_shape = input_shape
        self.filters = filters
        self.ds_kernel_size = ds_kernel_size
        self.us_kernel_size = us_kernel_size
        self.ds_stride = ds_stride
        self.us_stride = us_stride
        self.dropout_rate = dropout_rate
        self.n_downsample_layers = len(self.filters)
        
        self._build()
    
    def _build(self):
        downsamplers = []
        upsamplers = []
        
        for i in range(self.n_downsample_layers):
            downsampler = Downsampler(in_channels=self.input_shape[i] if i == 0 else self.filters[i - 1],
                                      out_channels=self.filters[i],
                                      kernel_size=self.ds_kernel_size,
                                      stride=self.ds_stride
                                     )
            
            if i < self.n_downsample_layers - 1:
                downsampler.register_forward_hook(self._save_skip_input(f'ds{i}'))
            
            downsamplers.append(downsampler)
        
        for i in range(self.n_downsample_layers - 2, -1, -1):
            in_channels = self.filters[i + 1]
            upsampler = Upsampler(in_channels= in_channels if i == self.n_downsample_layers - 2 else in_channels * 2,
                                  out_channels=self.filters[i],
                                  kernel_size=self.us_kernel_size,
                                  stride=self.us_stride,
                                  dropout_rate=self.dropout_rate,
                                 )
            upsampler.register_forward_pre_hook(self._get_skip_input(f'ds{i}'))
            upsamplers.append(upsampler)
        
        self.model = nn.Sequential(*downsamplers,
                                   *upsamplers,
                                   nn.Upsample(scale_factor=2),
                                   nn.Conv2d(in_channels=self.filters[0] * 2,
                                             out_channels=self.input_shape[0],
                                             kernel_size=self.us_kernel_size,
                                             stride=self.us_stride,
                                             padding=self.us_kernel_size // 2
                                            ),
                                   nn.Tanh()
                                  )
    
    def _save_skip_input(self, layer_id):
        def hook(module, inputs, output):
            self.skip_inputs[layer_id] = output
        
        return hook
    
    def _get_skip_input(self, layer_id):
        def hook(module, inputs):
            # inputs is a tuple of the positional arguments
            return inputs[0], self.skip_inputs[layer_id]
        
        return hook
    
    def forward(self, inputs):
        return self.model(inputs)
            

class DiscriminatorUNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self._build()
    
    def _build(self):
        pass
