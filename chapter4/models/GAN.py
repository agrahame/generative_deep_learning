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
                
                conv_layers.append(nn.LeakyReLU(negative_slope=0.2))
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


class Critic(nn.Module):

    def __init__(self,
                 input_shape: List[int],
                 filters: List[int],
                 kernels: List[int],
                 strides: List[int],
                 batch_norm_mom: float,
                 dropout_prob: float,
                 w_sigmoid: bool = False
                ):
        super().__init__()
        
        self.input_shape = input_shape
        self.filters = filters
        self.kernels = kernels
        self.strides = strides
        self.batch_norm_mom = batch_norm_mom
        self.dropout_prob = dropout_prob
        self.w_sigmoid = w_sigmoid
        
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
            
            conv_layers.append(nn.LeakyReLU(negative_slope=0.2))
            
            if self.dropout_prob:
                conv_layers.append(
                    nn.Dropout2d(p=self.dropout_prob)
                )
        
        dummy_input = torch.unsqueeze(torch.zeros(self.input_shape), 0)
        unflattened_shape = nn.Sequential(*conv_layers)(dummy_input).shape[1:]
        
        layers = [*conv_layers,
                  nn.Flatten(),
                  nn.Linear(in_features=np.prod(unflattened_shape),
                            out_features=1
                           )
                 ]
        
        if self.w_sigmoid:
            layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, batch):
        return self.model(batch)


class GANModule(nn.Module):

    def __init__(self,
                 generator: nn.Module = None,
                 critic: nn.Module = None,
                 gen_mode: bool = False
                ):
        super().__init__()
        
        if generator is not None:
            self.generator = generator
        
        if critic is not None:
            self.critic = critic
    
    def switch(self, gen_mode=None):
        self.gen_mode = not self.gen_mode if gen_mode is None else gen_mode
    
    def forward(self, *args):
        if self.gen_mode:
            return self.generator(*args)
        
        return self.critic(*args)


class GANLoss(GANModule):
    
    def __init__(self,
                 genr8r_loss_func,
                 critic_loss_func,
                 gan: GANModule
                ):
        super().__init__()
        self.genr8r_loss_func = genr8r_loss_func
        self.critic_loss_func = critic_loss_func
        self.gan = gan
        
    def generator(self, genr8r_output, real_image):
        critic_pred = self.gan.critic(genr8r_output)
        self.genr8r_loss = self.genr8r_loss_func(critic_pred)
        
        return self.genr8r_loss

    def critic(self, real_pred, noise):
        generated = self.gan.generator(noise)
        fake_pred = self.gan.critic(generated)
        self.critic_loss = self.critic_loss_func(fake_pred, real_pred)
        
        return self.critic_loss


def wgan_genr8r_loss(fake_pred):
    return fake_pred.mean()


def wgan_critic_loss(fake_pred, real_pred):
    return real_pred.mean() - fake_pred.mean()


def freeze_model(model, requires_grad):
    for param in model.parameters():
        param.requires_grad_(requires_grad)


class GANTrainer(Callback):
    
    def __init__(self, gen_first, switch_eval, beta, clip):
        self.gen_first = gen_first
        self.switch_eval = switch_eval
        self.clip = clip
        
        self.genr8r_loss = AvgSmoothLoss(beta=beta)
        self.critic_loss = AvgSmoothLoss(beta=beta)
    
    def _set_trainable(self):
        train_model = self.model.generator if self.model.gen_mode else self.model.critic
        eval_model = self.model.critic if self.model.gen_mode else self.model.generator
        freeze_model(train_model, requires_grad=True)
        freeze_model(eval_model, requires_grad=False)
        
        if self.switch_eval:
            train_model.train()
            eval_model.eval()
    
    def before_fit(self):
        self.switch(self.gen_first)
        self.genr8r_losses = []
        self.critic_losses = []
        self.genr8r_loss.reset()
        self.critic_loss.reset()
    
    def before_epoch(self):
        # Switch the gen or disc back to eval if necessary
        self.switch(self.model.gen_mode)
    
    def before_validate(self):
        self.switch(gen_mode=True)
    
    def before_batch(self):
        if self.training and self.clip is not None:
            for param in self.model.critic.parameters:
                param.data.clamp_(-self.clip, self.clip)
        
        # Make sure the input is what we expect
        # The dataset items are (noise, real_image)
        if not self.model.gen_mode:
            self.learn.xb, self.learn.yb = self.yb, self.xb
    
    def after_batch(self):
        if not self.training:
            return
        
        if self.model.gen_mode:
            self.genr8r_loss.accumulate(self.learn)
            self.genr8r_losses.append(self.genr8r_loss.value)
        else:
            self.critic_loss.accumulate(self.learn)
            self.critic_losses.append(self.critic_loss.value)
    
    def switch(self, gen_mode=None):
        self.model.switch(gen_mode)
        self.loss_func.switch(gen_mode)
        self._set_trainable()


class GANLearner(Learner):
    
    def __init__(self,
                 dataloaders,
                 generator,
                 critic,
                 genr8r_loss_func,
                 critic_loss_func,
                 switcher=None,
                 gen_first=False,
                 beta=0.98,
                 clip=None,
                 switch_eval=True,
                 callbacks=None,
                 metrics=None,
                 **kwargs
                ):
        
        gan = GANModule(generator, critic, gen_mode=gen_first)
        loss_func = GANLoss(genr8r_loss_func, critic_loss_func, gan)
        switcher = FixedGANSwitcher() if switcher is None else switcher
        trainer = GANTrainer(gen_first, switch_eval, beta, clip)
        callbacks = L(callbacks) + L(trainer, switcher)
        metrics = L(metrics) + L(*LossMetrics('genr8r_loss,critic_loss'))
        
        super().__init__(dataloaders,
                         gan,
                         loss_func=loss_func,
                         cbs=callbacks,
                         metrics=metrics,
                         **kwargs
                        )
    
    @classmethod
    def wgan(cls,
             dataloaders,
             generator,
             critic,
             switcher=None,
             clip=0.01,
             **kwargs):
        
        if switcher is None:
            switcher = FixedGANSwitcher(n_crit=5, n_gen=1)
        
        return cls(dataloaders=dataloaders,
                   generator=generator,
                   critic=critic,
                   genr8r_loss_func=wgan_genr8r_loss,
                   critic_loss_func=wgan_critic_loss,
                   switcher=switcher,
                   clip=clip,
                   **kwargs
                  )