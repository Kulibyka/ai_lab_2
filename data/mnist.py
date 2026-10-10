from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from config import DATA_ROOT, VAL_RATIO
from torchvision import transforms


def get_mnist_loaders(batch_size: int):
    """Загружает MNIST и возвращает train, val, test лоадеры."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_train = datasets.MNIST(root=DATA_ROOT, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=transform)

    train_size = int((1 - VAL_RATIO) * len(full_train))
    val_size = len(full_train) - train_size
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    return train_loader, val_loader, test_loader

def get_mnist_loaders_vae(batch_size: int):
    """Загружает MNIST для VAE (пиксели строго в диапазоне [0, 1])."""
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    full_train = datasets.MNIST(root=DATA_ROOT, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=transform)

    train_size = int((1 - VAL_RATIO) * len(full_train))
    val_size = len(full_train) - train_size
    train_dataset, val_dataset = random_split(full_train, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    return train_loader, val_loader, test_loader

def get_mnist_loaders_clip(batch_size: int):
    """Загружает MNIST, адаптированный под CLIP (224x224, RGB, специфичная нормализация)."""
    clip_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Grayscale(num_output_channels=3), # Превращаем 1 канал в 3 (RGB)
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
    ])

    full_train = datasets.MNIST(root=DATA_ROOT, train=True, download=True, transform=clip_transform)
    test_dataset = datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=clip_transform)

    train_loader = DataLoader(full_train, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
