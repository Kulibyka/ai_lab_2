# cnn_2_vae.py
import os
import torch
import torch.optim as optim

from config import BATCH_SIZE, DEVICE, MODEL_SAVE_DIR, RESULTS_SAVE_DIR
from data.mnist import get_mnist_loaders_vae
from models.vae import VAE
from engine.train_vae import train_one_epoch_vae, evaluate_vae
from visual.plots import plot_vae_results


def main():
    EPOCHS_VAE = 15  # Для VAE нужно чуть больше эпох
    LR_VAE = 1e-3

    print(f"Training VAE | Device: {DEVICE}")

    # Данные без нормализации
    train_loader, val_loader, test_loader = get_mnist_loaders_vae(BATCH_SIZE)

    # Модель
    model = VAE(latent_dim=20).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR_VAE)

    # Цикл обучения
    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    for epoch in range(EPOCHS_VAE):
        t_loss = train_one_epoch_vae(model, train_loader, optimizer, DEVICE)
        v_loss = evaluate_vae(model, val_loader, DEVICE)

        train_losses.append(t_loss)
        val_losses.append(v_loss)

        # Для VAE мы минимизируем Loss (чем меньше, тем лучше)
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_loss": v_loss,
            }, os.path.join(MODEL_SAVE_DIR, "best_vae.pth"))

        print(f"Epoch [{epoch + 1}/{EPOCHS_VAE}] | Train Loss: {t_loss:.2f} | Val Loss: {v_loss:.2f}")

    # Финальная визуализация
    print("\nFinal evaluation and visualization...")
    ckpt = torch.load(os.path.join(MODEL_SAVE_DIR, "best_vae.pth"), weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])

    plot_vae_results(model, test_loader, DEVICE, os.path.join(RESULTS_SAVE_DIR, "part3_vae_results.png"))


if __name__ == "__main__":
    main()