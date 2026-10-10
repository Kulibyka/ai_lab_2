import torch
from config import BATCH_SIZE, DEVICE
from data.mnist import get_mnist_loaders_clip
from engine.eval_clip import get_clip_model, get_text_embeddings, evaluate_clip_zero_shot


def main():
    print(f"Testing CLIP (Zero-Shot) | Device: {DEVICE}")

    _, test_loader = get_mnist_loaders_clip(BATCH_SIZE)

    print("Загрузка модели CLIP и создание текстовых эмбеддингов...")
    model = get_clip_model(DEVICE)
    text_features = get_text_embeddings(model, DEVICE)

    print("Запуск Zero-Shot классификации...")
    accuracy = evaluate_clip_zero_shot(model, test_loader, text_features, DEVICE)

    print(f"\nCLIP Zero-Shot Accuracy on MNIST: {accuracy:.2f}%")

if __name__ == "__main__":
    main()