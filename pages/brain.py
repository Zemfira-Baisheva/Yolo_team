import streamlit as st
import torch
import numpy as np
from PIL import Image

st.title("YOLOv5 Brain Tumor Detection")

# ---------------------------------------------------------------------------------------------#
### Mode for Axial detection ###
# File input #
st.sidebar.header("Axial")
uploaded_file_axial = st.sidebar.file_uploader("Select image from your folder...", type=["jpg", "jpeg", "png"], key="file_uploader_axial")

# Def model #
@st.cache_resource
def load_model_axial():
    return torch.hub.load(
        repo_or_dir='yolov5/',
        model='custom',
        path='models/brain_tumor_axial_best.pt',
        source='local'
    )

model_axial = load_model_axial()

# Confidence interval for Axial
model_conf_axial = st.sidebar.slider("Model Confidence Selection:", 0.0, 1.0, 0.5, 0.01)

# Predict function for Axial
def detect_axial(img):
    model_axial.conf = model_conf_axial
    with torch.inference_mode():
        results = model_axial(img)
    result_img = results.render()[0]
    result_pil = Image.fromarray(result_img)
    st.image(result_pil, caption='Detected Axial image', use_container_width=True)

if uploaded_file_axial is not None:
    img_axial = Image.open(uploaded_file_axial).convert("RGB")
    st.image(img_axial, caption='Non-detected Axial image.', use_container_width=True)
    if st.sidebar.button("Predict Axial"):
        detect_axial(img_axial)

st.sidebar.header("--------------------------------------------")
st.write('---')

# ---------------------------------------------------------------------------------------------#
### Mode for Coronal detection ###
st.sidebar.header("Coronal")
uploaded_file_coronal = st.sidebar.file_uploader("Select image from your folder...", type=["jpg", "jpeg", "png"], key='file_uploader_coronal')

# Def model #
@st.cache_resource
def load_model_coronal():
    return torch.hub.load(
        repo_or_dir='yolov5/',
        model='custom',
        path='models/brain_tumor_coronal_best.pt',
        source='local'
    )

model_coronal = load_model_coronal()

# Confidence interval for Coronal
model_conf_coronal = st.sidebar.slider("Model Confidence Selection:", 0.0, 1.0, 0.5, 0.01, key='coronal_conf')

def detect_coronal(img):
    model_coronal.conf = model_conf_coronal
    with torch.inference_mode():
        results = model_coronal(img)
    result_img = results.render()[0]
    result_pil = Image.fromarray(result_img)
    st.image(result_pil, caption='Detected Coronal image', use_container_width=True)

if uploaded_file_coronal is not None:
    img_coronal = Image.open(uploaded_file_coronal).convert("RGB")
    st.image(img_coronal, caption='Non-detected Coronal image.', use_container_width=True)
    if st.sidebar.button("Predict Coronal"):
        detect_coronal(img_coronal)

st.sidebar.header("--------------------------------------------")
st.write('---')

# ---------------------------------------------------------------------------------------------#
### Mode for Sagittal detection ###
st.sidebar.header("Sagittal")
uploaded_file_sagittal = st.sidebar.file_uploader("Select image from your folder...", type=["jpg", "jpeg", "png"], key='file_uploader_sagittal')

# Def model #
@st.cache_resource
def load_model_sagittal():
    return torch.hub.load(
        repo_or_dir='yolov5/',
        model='custom',
        path='models/brain_tumor_sagittal_best.pt',
        source='local'
    )

model_sagittal = load_model_sagittal()

# Confidence interval for Sagittal
model_conf_sagittal = st.sidebar.slider("Model Confidence Selection:", 0.0, 1.0, 0.5, 0.01, key='sagittal_conf')

def detect_sagittal(img):
    model_sagittal.conf = model_conf_sagittal
    with torch.inference_mode():
        results = model_sagittal(img)
    result_img = results.render()[0]
    result_pil = Image.fromarray(result_img)
    st.image(result_pil, caption='Detected Sagittal image', use_container_width=True)

if uploaded_file_sagittal is not None:
    img_sagittal = Image.open(uploaded_file_sagittal).convert("RGB")
    st.image(img_sagittal, caption='Non-detected Sagittal image.', use_container_width=True)
    if st.sidebar.button("Predict Sagittal"):
        detect_sagittal(img_sagittal)

st.sidebar.header("--------------------------------------------")
st.write('---')
