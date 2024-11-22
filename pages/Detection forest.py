import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import streamlit as st
from segmentation_models_pytorch import Unet  # Импорт архитектуры Unet
import requests
import cv2


# Функция для загрузки модели
@st.cache_resource
def load_model(path="models/unet-res3.pt"):
    model = Unet(
    encoder_name="resnet34",           # Более мощный энкодер для извлечения признаков
    encoder_weights="imagenet",       # Используем предобученные веса
    classes=1,                        # Бинарная сегментация
    activation="sigmoid",             # Активация для вероятностного выхода
    encoder_depth=4,                  # Уменьшение глубины энкодера для ускорения обучения
    decoder_channels=[256, 128, 64, 32]  # Настройка количества каналов в декодере
    )
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Функция для предсказания на одном изображении
def predict_image(model, image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)  # Добавляем batch dimension

    with torch.no_grad():
        output = model(input_tensor)  # Предсказание модели
        predicted_mask = (output > 0.5).float().squeeze().numpy()  # Бинаризуем маску

    return image, predicted_mask

# Функция для наложения маски на изображение
# Функция для наложения маски на изображение с красным цветом и яркостью
# Функция для наложения маски на изображение с красным цветом и яркостью
def overlay_mask_on_image(image, mask, color=(255, 0, 0), alpha=0.6):
    # Маска должна быть в формате изображения для наложения
    mask = np.array(mask)
    mask = np.expand_dims(mask, axis=-1)  # Преобразуем маску в 3D (цвет)
    mask = np.repeat(mask, 3, axis=-1)  # Дублируем для 3 каналов

    # Изменяем размер маски до размера изображения
    mask_colored = cv2.resize(mask, (image.width, image.height))

    # Наложение красного цвета на маску
    mask_colored = mask_colored * np.array(color)  # Маска с красным оттенком

    # Маска с альфа-каналом (прозрачность)
    mask_colored = mask_colored * alpha  # Прозрачность маски
    mask_colored = np.clip(mask_colored, 0, 255).astype(np.uint8)

    # Конвертируем исходное изображение в массив numpy
    image = np.array(image)

    # Совмещение маски с изображением
    overlay_image = cv2.addWeighted(image, 1.0, mask_colored, alpha, 0)

    return Image.fromarray(overlay_image)



# Загрузка модели
model_path = "models/unet-res3.pt"
st.sidebar.title("Model and Settings")
model = load_model(model_path)
st.sidebar.success("Model loaded successfully!")

# Главная страница приложения
st.title("UNet Image Segmentation Service")
st.write("Upload images or folders to perform segmentation.")

# Опции загрузки
upload_option = st.radio("Choose input method:", ["Single Image", "Multiple Images", "Image URL"])

# Загрузка и обработка одного изображения
if upload_option == "Single Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        with st.spinner("Processing image..."):
            image, mask = predict_image(model, uploaded_file)

            st.image(image, caption="Uploaded Image", use_container_width=True)
            st.image(mask, caption="Predicted Mask", use_container_width=True, clamp=True)

            # Наложение маски на изображение
            overlay_image = overlay_mask_on_image(image, mask)
            st.image(overlay_image, caption="Image with Mask Overlay", use_container_width=True)

# Загрузка и обработка нескольких изображений
elif upload_option == "Multiple Images":
    uploaded_files = st.file_uploader("Upload multiple images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                image, mask = predict_image(model, uploaded_file)

                st.image(image, caption=f"Uploaded Image: {uploaded_file.name}", use_container_width=True)
                st.image(mask, caption=f"Predicted Mask: {uploaded_file.name}", use_container_width=True, clamp=True)

                # Наложение маски на изображение
                overlay_image = overlay_mask_on_image(image, mask)
                st.image(overlay_image, caption=f"Image with Mask Overlay: {uploaded_file.name}", use_container_width=True)

# Загрузка изображения по ссылке
elif upload_option == "Image URL":
    image_url = st.text_input("Enter the image URL")

    if image_url:
        try:
            with st.spinner("Downloading and processing image..."):
                image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')
                temp_path = "temp_image.jpg"
                image.save(temp_path)

                _, mask = predict_image(model, temp_path)

                st.image(image, caption="Uploaded Image from URL", use_container_width=True)
                st.image(mask, caption="Predicted Mask", use_container_width=True, clamp=True)

                # Наложение маски на изображение
                overlay_image = overlay_mask_on_image(image, mask)
                st.image(overlay_image, caption="Image with Mask Overlay", use_container_width=True)

                os.remove(temp_path)
        except Exception as e:
            st.error(f"Failed to process the image from URL: {e}")

# Раздел с информацией о модели
st.sidebar.subheader("Model Information")
st.sidebar.write("**Architecture**: UNet with ResNet34 Encoder")
st.sidebar.write("**Trained on**: Custom dataset")
st.sidebar.write("**Epochs**: 23")
st.sidebar.write("**Sample size: 5108**")
st.sidebar.write("**Metrics**:")
st.sidebar.write("- IoU: 0.771571")
st.sidebar.write("- loss: 0.382533")
st.sidebar.write("- Accuracy: 0.83033")
