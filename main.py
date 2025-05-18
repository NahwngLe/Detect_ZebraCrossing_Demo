import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import tempfile


model_option = st.selectbox(
    "Chọn mô hình sử dụng",
    ("Model 1 - IOU 0.60 - Datav1",
     "Model 2 - IOU 0.55 - Datav1",
     "Model 3 - IOU 0.53 - Datav2",
     "Model 4 - IOU 0.50 - Datav2")
)

model_paths = {
    "Model 1 - IOU 0.60 - Datav1": "model_final_0.6ious.h5",
    "Model 2 - IOU 0.55 - Datav1": "model_datav1_100epochs_0.55iou_expr10.h5",
    "Model 3 - IOU 0.53 - Datav2": "model_datav2_100epochs_0.53iou.h5",
    "Model 4 - IOU 0.50 - Datav2": "model_datav2_100epochs_0.50iou_expr11.h5",
}

@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path, compile=False)

model = load_model(model_paths[model_option])
IMG_SIZE = (128, 128)

st.title("🚦 Nhận diện vạch kẻ đường cho người đi bộ")
st.markdown("""
Ứng dụng sử dụng mô hình **Convolutional Neural Network** với backbone **ResNet50** để phát hiện vạch kẻ đường cho người đi bộ.  
Tải lên ảnh để mô hình phân tích và hiển thị kết quả.
""")

uploaded_file = st.file_uploader("Tải lên ảnh", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    img = cv2.imread(tmp_path)
    if img is None:
        st.error("Không thể đọc ảnh.")
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR → RGB
        image_np = img.copy()

        st.image(image_np, caption="Ảnh gốc", use_container_width=True)

        img_resized = cv2.resize(img, IMG_SIZE)  # resize về (224, 224)
        img_array = img_resized / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)


        pred_class, pred_bbox = model.predict(img_array)
        label = "Có vạch" if pred_class[0][0] > 0.5 else "Không có vạch"
        print(pred_class[0][0])
        if label == "Có vạch":
            h, w = image_np.shape[:2]
            x_center, y_center, box_w, box_h = pred_bbox[0]
            x1 = int((x_center - box_w / 2) * w)
            y1 = int((y_center - box_h / 2) * h)
            x2 = int((x_center + box_w / 2) * w)
            y2 = int((y_center + box_h / 2) * h)

            image_with_box = image_np.copy()
            cv2.rectangle(image_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)

            st.image(image_with_box, caption=f"Kết quả: {label}", use_container_width=True)
        else:
            image_with_box = image_np.copy()
            st.image(image_with_box, caption=f"Kết quả: {label}", use_container_width=True)

