import streamlit as st
from PIL import Image



st.title("Информация по моделям")

st.write("### Детекция лиц с помощью любой версии YOLO c последующей маскировкой детектированной области")

st.markdown("#### Число эпох - 10")
st.markdown("#### Объем выборок:")
st.markdown("1. тренировочная выборка - 13 400 изображений, 2. валидационная - 3 347 изображений")

st.subheader("Метрики модели")

conf_mat_face = Image.open('images_metrics/confusion_matrix_face.png')
results_face = Image.open('images_metrics/results_face.png')
pr_curve_face = Image.open('images_metrics/PR_curve_face.png')
p_curve_face = Image.open('images_metrics/P_curve_face.png')
r_curve_face = Image.open('images_metrics/R_curve_face.png')
f1_curve_face = Image.open('images_metrics/F1_curve_face.png')

st.markdown("### Графики Loss-функции")
st.image(results_face, caption=' ', use_container_width=True)

st.markdown("### Precision-recall Кривая")
st.image(pr_curve_face, caption=' ', use_container_width=True)

st.markdown("### F1-Кривая")
st.image(f1_curve_face, caption=' ', use_container_width=True)

st.markdown("### Precision Кривая")
st.image(p_curve_face, caption=' ', use_container_width=True)

st.markdown("### Recall Кривая")
st.image(r_curve_face, caption=' ', use_container_width=True)

st.markdown("### Матрица ошибок")
st.image(conf_mat_face, caption=' ', use_container_width=True)


st.write("### Детекция опухулей мозга по фотографии")

st.write("### Axial")

st.markdown("#### Число эпох - 200")
st.markdown("#### Объем выборок:")
st.markdown("1. тренировочная выборка - 310 изображений, 2. валидационная - 75 изображений")

st.subheader("Метрики модели")

conf_mat_ax = Image.open('images_metrics/axial/confusion_matrix.png')
results_ax = Image.open('images_metrics/axial/results.png')
pr_curve_ax = Image.open('images_metrics/axial/PR_curve.png')
p_curve_ax = Image.open('images_metrics/axial/P_curve.png')
r_curve_ax = Image.open('images_metrics/axial/R_curve.png')
f1_curve_ax = Image.open('images_metrics/axial/F1_curve.png')

st.markdown("### Графики Loss-функции")
st.image(results_ax, caption=' ', use_container_width=True)

st.markdown("### Precision-recall Кривая")
st.image(pr_curve_ax, caption=' ', use_container_width=True)

st.markdown("### F1-Кривая")
st.image(f1_curve_ax, caption=' ', use_container_width=True)

st.markdown("### Precision Кривая")
st.image(p_curve_ax, caption=' ', use_container_width=True)

st.markdown("### Recall Кривая")
st.image(r_curve_ax, caption=' ', use_container_width=True)

st.markdown("### Матрица ошибок")
st.image(conf_mat_ax, caption=' ', use_container_width=True)



st.write("### Coronal")

st.markdown("#### Число эпох - 200")
st.markdown("#### Объем выборок:")
st.markdown("1. тренировочная выборка - 319 изображений, 2. валидационная - 78 изображений")

st.subheader("Метрики модели")

conf_mat_c = Image.open('images_metrics/coronal/confusion_matrix.png')
results_c = Image.open('images_metrics/coronal/results.png')
pr_curve_c = Image.open('images_metrics/coronal/PR_curve.png')
p_curve_c = Image.open('images_metrics/coronal/P_curve.png')
r_curve_c = Image.open('images_metrics/coronal/R_curve.png')
f1_curve_c = Image.open('images_metrics/coronal/F1_curve.png')

st.markdown("### Графики Loss-функции")
st.image(results_c, caption=' ', use_container_width =True)

st.markdown("### Precision-recall Кривая")
st.image(pr_curve_c, caption=' ', use_container_width =True)

st.markdown("### F1-Кривая")
st.image(f1_curve_c, caption=' ', use_container_width =True)

st.markdown("### Precision Кривая")
st.image(p_curve_c, caption=' ', use_container_width =True)

st.markdown("### Recall Кривая")
st.image(r_curve_c, caption=' ', use_container_width =True)

st.markdown("### Матрица ошибок")
st.image(conf_mat_c, caption=' ', use_container_width =True)


st.write("### Saggital")

st.markdown("#### Число эпох - 200")
st.markdown("#### Объем выборок:")
st.markdown("1. тренировочная выборка - 264 изображений, 2. валидационная - 70 изображений")

st.subheader("Метрики модели")

conf_mat_s = Image.open('images_metrics/saggital/confusion_matrix.png')
results_s = Image.open('images_metrics/saggital/results.png')
pr_curve_s = Image.open('images_metrics/saggital/PR_curve.png')
p_curve_s = Image.open('images_metrics/saggital/P_curve.png')
r_curve_s = Image.open('images_metrics/saggital/R_curve.png')
f1_curve_s = Image.open('images_metrics/saggital/F1_curve.png')

st.markdown("### Графики Loss-функции")
st.image(results_s, caption=' ', use_container_width=True)

st.markdown("### Precision-recall Кривая")
st.image(pr_curve_s, caption=' ', use_container_width=True)

st.markdown("### F1-Кривая")
st.image(f1_curve_s, caption=' ', use_container_width=True)

st.markdown("### Precision Кривая")
st.image(p_curve_s, caption=' ', use_container_width=True)

st.markdown("### Recall Кривая")
st.image(r_curve_s, caption=' ', use_container_width=True)

st.markdown("### Матрица ошибок")
st.image(conf_mat_s, caption=' ', use_container_width=True)




st.write("### Семантическая сегментация аэрокосмических снимков")

st.markdown("#### Число эпох - 23")
st.markdown("#### Объем выборок:")
st.markdown("1. тренировочная выборка - 4 086 изображений, 2. валидационная - 1 022 изображений")

st.subheader("Метрики модели")

acc = Image.open('images_metrics/unet/acc.jpg')
iou = Image.open('images_metrics/unet/iou.jpg')
loss = Image.open('images_metrics/unet/loss.jpg')


st.markdown("### Loss-функция")
st.image(loss, caption=' ', use_container_width =True)

st.markdown("### Accuracy")
st.image(acc, caption=' ', use_container_width =True)

st.markdown("### IOU ")
st.image(iou, caption=' ', use_container_width =True)