import os
import torch
import torch.nn as nn
import torch.optim as optim
from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, DEVICE, MODEL_SAVE_DIR, RESULTS_SAVE_DIR
from data.mnist import get_mnist_loaders
from models.resnet_transfer import ResNetTransfer
from engine.train import train_one_epoch
from engine.evaluate import evaluate
from visual.plots import plot_training_history

def main():
    print(f"Transfer Learning (ResNet18 Fine-Tuning) | Device: {DEVICE}")

    train_loader, val_loader, test_loader = get_mnist_loaders(BATCH_SIZE)

    model = ResNetTransfer().to(DEVICE)

    for param in model.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_losses, val_losses, val_accs = [], [], []
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        t_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        v_loss, v_acc = evaluate(model, val_loader, criterion, DEVICE)

        train_losses.append(t_loss)
        val_losses.append(v_loss)
        val_accs.append(v_acc)

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_accuracy": v_acc,
            }, os.path.join(MODEL_SAVE_DIR, "best_resnet_transfer.pth"))

        print(f"Epoch [{epoch + 1}/{EPOCHS}] | Train Loss: {t_loss:.4f} | Val Acc: {v_acc:.2f}%")

    print("\nФинальное тестирование на Test Set...")
    ckpt = torch.load(os.path.join(MODEL_SAVE_DIR, "best_resnet_transfer.pth"), weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    _, final_test_acc = evaluate(model, test_loader, criterion, DEVICE)
    print(f"Final Test Accuracy (Transfer): {final_test_acc:.2f}%")

    plot_training_history(
        train_losses, val_losses, val_accs,
        save_path=os.path.join(RESULTS_SAVE_DIR, "part2_transfer_learning.png")
    )

if __name__ == "__main__":
    main()