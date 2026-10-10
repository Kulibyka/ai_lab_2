import matplotlib.pyplot as plt
import os

import torch


def plot_training_history(train_losses, val_losses, val_accuracies, save_path):
    """Строит графики loss и accuracy, сохраняет на диск."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, marker='o', label='Train Loss')
    plt.plot(epochs, val_losses, marker='s', label='Val Loss')
    plt.title('Loss Dynamics')
    plt.xlabel('Epoch');
    plt.ylabel('Loss')
    plt.legend();
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, marker='o', color='green', label='Val Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch');
    plt.ylabel('Accuracy (%)')
    plt.legend();
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ Графики сохранены: {save_path}")
    plt.close()


def plot_vae_results(model, loader, device, save_path):
    """Визуализирует оригиналы, реконструкции и сгенерированные изображения."""
    model.eval()
    data, _ = next(iter(loader))
    data = data.to(device)

    with torch.no_grad():
        recon, _, _ = model(data)
        z = torch.randn(16, model.latent_dim).to(device)
        generated = model.decode(z)

    fig, axes = plt.subplots(3, 16, figsize=(16, 4))

    for i in range(16):
        axes[0, i].imshow(data[i].cpu().squeeze(), cmap='gray')
        axes[0, i].axis('off')
    axes[0, 0].set_title('Originals')

    for i in range(16):
        axes[1, i].imshow(recon[i].cpu().view(28, 28), cmap='gray')
        axes[1, i].axis('off')
    axes[1, 0].set_title('Reconstructions')

    for i in range(16):
        axes[2, i].imshow(generated[i].cpu().view(28, 28), cmap='gray')
        axes[2, i].axis('off')
    axes[2, 0].set_title('Generated from random z')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"VAE visualization saved to {save_path}")