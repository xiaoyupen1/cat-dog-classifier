"""Predict whether one image contains a cat or a dog."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models/cat_dog_mobilenetv3.pth"
NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)


def choose_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_model(device: torch.device) -> tuple[nn.Module, list[str], int]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"未找到模型：{MODEL_PATH}，请先运行 train.py。")

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    class_names = checkpoint["class_names"]
    image_size = checkpoint["image_size"]
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), class_names, image_size


def main() -> None:
    parser = argparse.ArgumentParser(description="识别一张图片是猫还是狗。")
    parser.add_argument("image", type=Path, help="待预测图片的路径")
    args = parser.parse_args()

    if not args.image.is_file():
        raise FileNotFoundError(f"未找到图片：{args.image}")

    device = choose_device()
    model, class_names, image_size = load_model(device)
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            NORMALIZE,
        ]
    )

    image = Image.open(args.image).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(image_tensor), dim=1)[0]

    index = int(probabilities.argmax())
    print(f"预测结果：{class_names[index]}")
    print(f"置信度：{probabilities[index].item():.2%}")


if __name__ == "__main__":
    main()
