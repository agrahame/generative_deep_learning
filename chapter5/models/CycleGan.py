import torch
import torch.nn as nn
from typing import List
from fastai.vision.all import *
from fastai.vision.gan import FixedGANSwitcher


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
                 us_stride: int,
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


class CombndGens(nn.Module):
    
    def __init__(self,
                 input_shape: List[int],
                 filters: List[int],
                 ds_kernel_size: int,
                 us_kernel_size: int,
                 ds_stride: int,
                 us_stride: int,
                 dropout_rate: int = 0
                ):
        super().__init__()
        
        self.gen_ab = GeneratorUNet(input_shape=input_shape,
                                    filters=filters,
                                    ds_kernel_size=ds_kernel_size,
                                    us_kernel_size=us_kernel_size,
                                    ds_stride=ds_stride,
                                    us_stride=us_stride,
                                    dropout_rate=dropout_rate)
        self.gen_ba = GeneratorUNet(input_shape=input_shape,
                                    filters=filters,
                                    ds_kernel_size=ds_kernel_size,
                                    us_kernel_size=us_kernel_size,
                                    ds_stride=ds_stride,
                                    us_stride=us_stride,
                                    dropout_rate=dropout_rate)
    
    def forward(self, img_a, img_b):
        fake_b = self.gen_ab(img_a)
        fake_a = self.gen_ba(img_b)
        
        reconstr_a = self.gen_ba(fake_b)
        reconstr_b = self.gen_ab(fake_a)
        
        img_a_id = self.gen_ba(img_a)
        img_b_id = self.gen_ab(img_b)
        
        return fake_b, fake_a, reconstr_a, reconstr_b, img_a_id, img_b_id        


class DiscriminatorUNet(nn.Module):
    
    def __init__(self,
                 input_shape: List[int],
                 filters: List[int],
                 kernel_size: int
                ):
        super().__init__()
        self.input_shape = input_shape
        self.filters = filters
        self.kernel_size = kernel_size
        
        self._build()
    
    def _build(self):
        layers = []
        n_filters = len(self.filters)
        
        for i in range(n_filters):
            layers.append(
                nn.Conv2d(in_channels=self.input_shape[0] if i == 0 else self.filters[i - 1],
                          out_channels=self.filters[i],
                          kernel_size=self.kernel_size,
                          stride=2 if i < n_filters - 1 else 1,
                          padding=self.kernel_size // 2
                         )
            )
            
            if i > 0:
                layers.append(nn.InstanceNorm2d(num_features=self.filters[i]))
            
            layers.append(nn.LeakyReLU(negative_slope=0.2))
        
        self.model = nn.Sequential(*layers,
                                   nn.Conv2d(in_channels=self.filters[-1],
                                             out_channels=1,
                                             kernel_size=self.kernel_size,
                                             stride=1,
                                             padding=self.kernel_size // 2
                                            )
                                  )
    
    def forward(self, inputs):
        return self.model(inputs)


class CombndDiscs(nn.Module):
    
    def __init__(self,
                 input_shape: List[int],
                 filters: List[int],
                 kernel_size: int
                ):
        super().__init__()
        
        self.disc_a = DiscriminatorUNet(input_shape=input_shape,
                                        filters=filters,
                                        kernel_size=kernel_size)
        self.disc_b = DiscriminatorUNet(input_shape=input_shape,
                                        filters=filters,
                                        kernel_size=kernel_size)
    
    def forward(self, img_a, img_b):
        return self.disc_a(img_a), self.disc_b(img_b)


class CycleGANModule(nn.Module):
    
    def __init__(self,
                 generator: CombndGens = None,
                 discriminator: CombndDiscs = None,
                 gen_mode: bool = False
                ):
        super().__init__()
        
        if generator is not None: self.generator = generator
        if discriminator is not None: self.discriminator = discriminator
        
    def switch(self, gen_mode=None):
        self.gen_mode = not self.gen_mode if gen_mode is None else gen_mode
    
    def forward(self, *args):
        if self.gen_mode:
            return self.generator(*args)
        
        return self.discriminator(*args)


class CycleGANLoss(CycleGANModule):
    
    def __init__(self,
                 gan: CycleGANModule,
                 lambda_valid: int,
                 lambda_reconstr: int,
                 lambda_id: int
                ):
        super().__init__()
        
        self.gan = gan
        self.lambda_valid = lambda_valid
        self.lambda_reconstr = lambda_reconstr
        self.lambda_id = lambda_id
    
    def generator(self, gen_outputs, img_a, img_b):
        fake_b, fake_a, reconstr_a, reconstr_b, img_a_id, img_b_id = gen_outputs
        batch_size = img_a[0]
        img_size = img_a[-1]  # assuming square images
        # this reflects the output shape of disc output after the 3 stride-2 convs
        patch = img_size / 2**3
        valid_target = torch.ones(batch_size, 1, patch, patch)
        
        disc_a_pred, disc_b_pred = self.gan.discriminator(fake_a, fake_b)
        
        valid_loss = nn.MSELoss()
        reconstr_loss = nn.L1Loss()  # MAE
        id_loss = nn.L1Loss()  # MAE
        
        loss = self.lambda_valid * (valid_loss(disc_a_pred, valid_target) + valid_loss(disc_b_pred, valid_target))
        loss += self.lambda_reconstr * (reconstr_loss(reconstr_a, img_a) + reconstr_loss(reconstr_b, img_b))
        loss += self.lambda_id * (id_loss(img_a_id, img_a) + id_loss(img_b_id, img_b))
        
        return loss
    
    def discriminator(self, disc_outputs, img_a, img_b):
        disc_a_pred, disc_b_pred = disc_outputs
        batch_size = img_a[0]
        img_size = img_a[-1]  # assuming square images
        # this reflects the output shape of disc output after the 3 stride-2 convs
        patch = img_size / 2**3
        valid_target = torch.ones(batch_size, 1, patch, patch)
        fake_target = torch.zeros(batch_size, 1, patch, patch)
        
        fake_b, fake_a, _, _, _, _ = self.gan.generator(img_a, img_b)
        
        disc_loss = nn.MSELoss()
        
        disc_a_loss = 0.5 * (disc_loss(img_a, valid_target) + disc_loss(fake_a, fake_target))
        disc_b_loss = 0.5 * (disc_loss(img_b, valid_target) + disc_loss(fake_b, fake_target))
        
        return 0.5 * (disc_a_loss + disc_b_loss)
