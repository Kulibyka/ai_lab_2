import torch
import torch.nn as nn


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion, device):
    """Оценка модели: возвращает (avg_loss, accuracy%)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model(data)

        total_loss += criterion(output, target).item()
        predicted = output.argmax(dim=1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy
