# engine/train_vae.py
import torch
import torch.nn.functional as F


def vae_loss_function(recon_x, x, mu, logvar):
    """
    ELBO Loss = Reconstruction Loss (BCE) + KL Divergence.
    """
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def train_one_epoch_vae(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for data, _ in loader:
        data = data.to(device)
        optimizer.zero_grad()

        recon, mu, logvar = model(data)
        loss = vae_loss_function(recon, data, mu, logvar)

        loss.backward()
        total_loss += loss.item()
        optimizer.step()

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_vae(model, loader, device):
    model.eval()
    total_loss = 0.0
    for data, _ in loader:
        data = data.to(device)
        recon, mu, logvar = model(data)
        loss = vae_loss_function(recon, data, mu, logvar)
        total_loss += loss.item()

    return total_loss / len(loader.dataset)