import torch.nn.functional as F
import torch
from models.VAE import VariationalAutoEncoder


def rmse_loss(pred, targ):
    return torch.sqrt(F.mse_loss(pred, targ))


def vae_rec_loss(pred, targ, rec_loss_factor=1000):
    rec_loss = rmse_loss(pred, targ)
    return rec_loss_factor * rec_loss


def vae_kl_loss(mu, log_var):
    return  -0.5 * torch.sum(1 + log_var - torch.square(mu) - torch.exp(log_var))

def vae_loss(pred, targ, model: VariationalAutoEncoder = None):
    if not model:
        raise ValueError("Model not provided to vae loss")
    
    bottleneck = list(model.encoder.children())[-1]    
    rec_loss = vae_rec_loss(pred, targ)
    kl_loss = vae_kl_loss(bottleneck.mu, bottleneck.log_var)

    return  rec_loss + kl_loss
