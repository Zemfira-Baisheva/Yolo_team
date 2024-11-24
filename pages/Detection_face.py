import streamlit as st
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from torchvision.io import read_image
import torchvision.transforms as T
from PIL import Image, ImageFilter, ImageDraw
import requests
from io import BytesIO



UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


st.title("YOLOv5 Object Detection")


@st.cache_resource
def load_model():
    model_path = 'models/best_face.pt'
    yolov5_path = 'yolov5/'

    if not os.path.exists(model_path):
        st.error("Файл 'best_face.pt' не найден. Пожалуйста, загрузите модель.")
        return None

    try:
        model = torch.hub.load(
            repo_or_dir=yolov5_path,
            model='custom',
            path=model_path,
            source='local'
        )
        st.success("Модель успешно загружена.")
        return model
    except Exception as e:
        st.error(f"Ошибка при загрузке модели: {e}")
        return None

model = load_model()


# Определение диапазона значений для уверенности
min_confidence = 0.0
max_confidence = 1.0
step = 0.01  # Шаг изменения

# Использование select_slider для выбора уровня уверенности
model_conf = st.sidebar.select_slider(
    "Model Confidence Selection:",
    options=[round(i, 2) for i in list(np.arange(min_confidence, max_confidence + step, step))],
    value=0.5  # Значение по умолчанию
)

def detect_face(image):
    model.eval()
    with torch.inference_mode():
        results = model(img)
  
    res_coord = [tensor.cpu().numpy() for tensor in results.xyxy]
    res_coord = np.array(res_coord)
    res_coord = res_coord.reshape(-1, 6)
    for coord in res_coord:

        x1, y1, x2, y2 = map(int, coord[:4])

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.width, x2), min(img.height, y2)


        roi = img.crop((x1, y1, x2, y2))


        blurred_roi = roi.filter(ImageFilter.GaussianBlur(radius=40))  # Вы можете настроить радиус размытия

        # Замена оригинальной области на размытую
        img.paste(blurred_roi, (x1, y1))


        # Отображение изображения с помощью Matplotlib
        plt.imshow(img)
        plt.axis('off')  # Скрыть оси
        plt.show()
    st.image(img, caption='Изображение с маскировкой', use_container_width=True)

st.sidebar.header("Загрузка одного файла")
uploaded_file = st.sidebar.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"])
if st.sidebar.button("Download from file"):

    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption='Загруженное изображение.', use_container_width=True)
    detect_face(img)


st.sidebar.header("Загрузка по прямой ссылке")
url = st.sidebar.text_input("Enter file URL")
if st.sidebar.button("Download from URL"):
    response = requests.get(url)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content)).convert("RGB")
    st.image(img, caption='Загруженное изображение.', use_container_width=True)
    detect_face(img)


st.sidebar.header("Загрузка нескольких файлов")
uploaded_files = st.sidebar.file_uploader("Choose files", accept_multiple_files=True)
if st.sidebar.button("Download from files"):
    for uploaded_file in uploaded_files:
        st.image(uploaded_file, caption='Загруженное изображение.', use_container_width=True)
        img = Image.open(uploaded_file).convert("RGB")
        detect_face(img)