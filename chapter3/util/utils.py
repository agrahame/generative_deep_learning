import torch.nn.functional as F
import torch
from models.VAE import VariationalAutoEncoder


def rmse_loss(inp, targ):
    return torch.sqrt(F.mse_loss(inp, targ))


def vae_rec_loss(inp, targ, rec_loss_factor=1000):
    rec_loss = rmse_loss(inp, targ)
    return rec_loss_factor * rec_loss


def vae_kl_loss(mu, log_var):
    return  -0.5 * torch.sum(1 + log_var - torch.square(mu) - torch.exp(log_var), dim=-1)


def vae_loss(inp, targ, model: VariationalAutoEncoder = None):
    if not model:
        raise ValueError("Model not provided to vae loss")
    
    bottleneck = list(model.encoder.children())[-1]    
    rec_loss = vae_rec_loss(inp, targ)
    kl_loss = vae_kl_loss(bottleneck.mu, bottleneck.log_var)

    return  rec_loss + kl_loss
