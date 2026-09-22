# 猫狗识别器

一个使用 PyTorch 和 MobileNetV3 实现的猫狗图片分类网页。上传 JPG 或 PNG 图片后，页面会给出猫或狗的预测结果及置信度。

## 本地运行

```bash
source .venv/bin/activate
streamlit run app.py
```

## 部署到 Streamlit Community Cloud

将以下文件上传到 GitHub 仓库：

```text
app.py
predict.py
requirements.txt
models/cat_dog_mobilenetv3.pth
```

然后在 https://share.streamlit.io 用 GitHub 登录，创建应用并将入口文件设置为 `app.py`。

训练数据集和 `.venv` 已被 `.gitignore` 排除，不会上传到 GitHub。
