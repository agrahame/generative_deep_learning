import torch
import torch.nn as nn
import numpy as np
from typing import List
from fastai.vision.all import *
from fastai.vision.gan import FixedGANSwitcher


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
        self.gen_loss = self.gen_loss_func(disc_pred)
        
        return self.gen_loss

    def discriminator(self, real_pred, noise):
        generated = self.gan.generator(noise)
        fake_pred = self.gan.discriminator(generated)
        self.disc_loss = self.disc_loss_func(fake_pred, real_pred)
        
        return self.disc_loss


def freeze_model(model, requires_grad):
    for param in model.parameters():
        param.requires_grad_(requires_grad)


class GANTrainer(Callback):
    
    def __init__(self, gen_first, switch_eval, beta):
        self.gen_first = gen_first
        self.switch_eval = switch_eval
        
        self.gen_loss = AvgSmoothLoss(beta=beta)
        self.disc_loss = AvgSmoothLoss(beta=beta)
    
    def _set_trainable(self):
        train_model = self.model.generator if self.model.gen_mode else self.model.discriminator
        eval_model = self.model.discriminator if self.model.gen_mode else self.model.generator
        freeze_model(train_model, requires_grad=True)
        freeze_model(eval_model, requires_grad=False)
        
        if self.switch_eval:
            train_model.train()
            eval_model.eval()
    
    def before_fit(self):
        self.switch(self.gen_first)
        self.gen_losses = []
        self.disc_losses = []
        self.gen_loss.reset()
        self.disc_loss.reset()
    
    def before_epoch(self):
        # Switch the gen or disc back to eval if necessary
        self.switch(self.model.gen_mode)
    
    def before_validate(self):
        self.switch(gen_mode=True)
    
    def before_batch(self):
        # Make sure the input is what we expect
        # The dataset items are (noise, real_image)
        if not self.model.gen_mode:
            self.learn.xb, self.learn.yb = self.yb, self.xb
    
    def after_batch(self):
        if not self.training:
            return
        
        if self.model.gen_mode:
            self.gen_loss.accumulate(self.learn)
            self.gen_losses.append(self.gen_loss.value)
        else:
            self.disc_loss.accumulate(self.learn)
            self.disc_losses.append(self.disc_loss.value)
    
    def switch(self, gen_mode=None):
        self.model.switch(gen_mode)
        self.loss_func.switch(gen_mode)
        self._set_trainable()


class GANLearner(Learner):
    
    def __init__(self,
                 dataloaders,
                 generator,
                 discriminator,
                 gen_loss_func,
                 disc_loss_func,
                 switcher=None,
                 gen_first=False,
                 beta=0.98,
                 switch_eval=True,
                 callbacks=None,
                 metrics=None,
                 **kwargs
                ):
        
        gan = GANModule(generator, discriminator, gen_mode=gen_first)
        loss_func = GANLoss(gen_loss_func, disc_loss_func, gan)
        switcher = FixedGANSwitcher() if switcher is None else switcher
        trainer = GANTrainer(gen_first, switch_eval, beta)
        callbacks = L(callbacks) + L(trainer, switcher)
        metrics = L(metrics) + L(*LossMetrics('gen_loss,disc_loss'))
        
        super().__init__(dataloaders,
                         gan,
                         loss_func=loss_func,
                         cbs=callbacks,
                         metrics=metrics,
                         **kwargs
                        )