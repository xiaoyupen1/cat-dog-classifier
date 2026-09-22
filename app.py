"""A small Streamlit interface for the cat-vs-dog classifier."""

from __future__ import annotations

import torch
import streamlit as st
from PIL import Image
from torchvision import transforms

from predict import NORMALIZE, choose_device, load_model


st.set_page_config(page_title="猫狗识别", page_icon="🐾", layout="centered")


@st.cache_resource
def get_model():
    device = choose_device()
    model, class_names, image_size = load_model(device)
    return model, class_names, image_size, device


def predict_image(image: Image.Image) -> tuple[list[str], list[float]]:
    model, class_names, image_size, device = get_model()
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            NORMALIZE,
        ]
    )
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().tolist()
    return class_names, probabilities


st.title("🐾 猫狗识别器")
st.write("上传一张图片，模型会判断它更像猫还是狗。")

uploaded_file = st.file_uploader("选择 JPG 或 PNG 图片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="待识别图片", use_container_width=True)

    try:
        class_names, probabilities = predict_image(image)
        best_index = max(range(len(probabilities)), key=probabilities.__getitem__)
        name_map = {"cats": "猫", "dogs": "狗"}
        best_name = name_map.get(class_names[best_index], class_names[best_index])

        st.success(f"识别结果：{best_name}")
        st.metric("置信度", f"{probabilities[best_index]:.2%}")
        st.subheader("各类别概率")
        for name, probability in zip(class_names, probabilities):
            st.write(f"{name_map.get(name, name)}：{probability:.2%}")
            st.progress(probability)
    except FileNotFoundError as error:
        st.error(str(error))
        st.info("请先在终端运行 `python train.py` 来生成模型文件。")
