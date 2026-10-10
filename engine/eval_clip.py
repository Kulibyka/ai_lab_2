# engine/eval_clip.py
import torch
import open_clip


def get_clip_model(device):
    """Загружает предобученную модель CLIP (ViT-B/32)."""
    model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    model = model.to(device)
    model.eval()
    return model


def get_text_embeddings(model, device):
    """Создает текстовые эмбеддинги для 10 классов цифр."""
    templates = ["a photo of the number {}", "a handwritten digit {}", "the number {}"]
    prompts = []
    for i in range(10):
        for t in templates:
            prompts.append(t.format(i))

    text_tokens = open_clip.tokenize(prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    text_features = text_features.view(10, 3, -1).mean(dim=1)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    return text_features


@torch.no_grad()
def evaluate_clip_zero_shot(model, loader, text_features, device):
    """Zero-Shot классификация: сравниваем эмбеддинги картинок и текста."""
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        logits = (100.0 * image_features @ text_features.T)
        preds = logits.argmax(dim=-1)

        correct += (preds == targets).sum().item()
        total += targets.size(0)

    return 100.0 * correct / total