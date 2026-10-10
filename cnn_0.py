import os
import torch
import torch.nn as nn
import torch.optim as optim

from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, DEVICE, MODEL_SAVE_DIR, RESULTS_SAVE_DIR
from data.mnist import get_mnist_loaders
from models.cnn import SimpleCNN
from engine.train import train_one_epoch
from engine.evaluate import evaluate
from visual.plots import plot_training_history


def main():
    print(f"Запуск обучения | Device: {DEVICE}")

    # Данные
    train_loader, val_loader, test_loader = get_mnist_loaders(BATCH_SIZE)

    # Модель, лосс, оптимизатор
    model = SimpleCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Цикл обучения
    train_losses, val_losses, val_accs = [], [], []
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        t_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        v_loss, v_acc = evaluate(model, val_loader, criterion, DEVICE)

        train_losses.append(t_loss)
        val_losses.append(v_loss)
        val_accs.append(v_acc)

        # Сохранение лучшего чекпоинта
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": v_acc,
            }, os.path.join(MODEL_SAVE_DIR, "best_cnn_mnist.pth"))

        print(f"Epoch [{epoch + 1}/{EPOCHS}] | Train Loss: {t_loss:.4f} | Val Acc: {v_acc:.2f}%")

    # Финальный тест
    print("\nФинальное тестирование на Test Set...")
    ckpt = torch.load(os.path.join(MODEL_SAVE_DIR, "best_cnn_mnist.pth"), weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    _, final_test_acc = evaluate(model, test_loader, criterion, DEVICE)
    print(f"Final Test Accuracy: {final_test_acc:.2f}%")

    plot_training_history(
        train_losses, val_losses, val_accs,
        save_path=os.path.join(RESULTS_SAVE_DIR, "part1_cnn_scratch.png")
    )


if __name__ == "__main__":
    main()