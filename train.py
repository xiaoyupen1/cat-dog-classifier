"""Train a cat-vs-dog image classifier with transfer learning."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = PROJECT_DIR / "data/raw/training_set/training_set"
VALID_DIR = PROJECT_DIR / "data/raw/test_set/test_set"
MODEL_PATH = PROJECT_DIR / "models/cat_dog_mobilenetv3.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-3

NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_loaders() -> tuple[DataLoader, DataLoader, list[str]]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            NORMALIZE,
        ]
    )
    valid_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            NORMALIZE,
        ]
    )

    train_data = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    valid_data = datasets.ImageFolder(VALID_DIR, transform=valid_transform)
    if train_data.classes != valid_data.classes:
        raise ValueError("训练集和测试集的类别目录不一致。")

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, valid_loader, train_data.classes


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main() -> None:
    if not TRAIN_DIR.exists() or not VALID_DIR.exists():
        raise FileNotFoundError("未找到数据集，请检查 data/raw 下的目录结构。")

    device = choose_device()
    train_loader, valid_loader, class_names = make_loaders()
    print(f"设备：{device}")
    print(f"类别：{class_names}")
    print(f"训练图片：{len(train_loader.dataset)}，验证图片：{len(valid_loader.dataset)}")

    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    best_accuracy = 0.0
    MODEL_PATH.parent.mkdir(exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        valid_loss, valid_accuracy = evaluate(model, valid_loader, device)
        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train loss: {running_loss / total:.4f}, train acc: {correct / total:.2%} | "
            f"valid loss: {valid_loss:.4f}, valid acc: {valid_accuracy:.2%}"
        )

        if valid_accuracy > best_accuracy:
            best_accuracy = valid_accuracy
            torch.save(
                {
                    "architecture": "mobilenet_v3_small",
                    "class_names": class_names,
                    "image_size": IMAGE_SIZE,
                    "model_state_dict": model.state_dict(),
                },
                MODEL_PATH,
            )
            print(f"已保存最佳模型：{MODEL_PATH}（验证准确率 {best_accuracy:.2%}）")


if __name__ == "__main__":
    main()
