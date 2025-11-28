import streamlit as st
import numpy as np
from PIL import Image

# Import your modules
from app.detection import detect_image
from app.classification import classify_image

# -----------------------
# CLASS NAMES FOR RESNET
# -----------------------
CLASS_NAMES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltrate", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation",
    "Edema", "Emphysema", "Fibrosis", "Hernia",
    "Pleural Thickening", "Normal"
]

# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------
st.set_page_config(page_title="Lung Disease Detector", layout="wide")

st.title("🩻 Lung Disease Detection System")

uploaded_file = st.file_uploader(
    "Upload Chest X-ray", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.image(image_np, caption="Uploaded X-ray", width=400)

    # -----------------------------------------
    # OBJECT DETECTION (FASTER R-CNN)
    # -----------------------------------------
    st.subheader("📦 Object Detection (Faster R-CNN)")

    det_img, det_results = detect_image(image_np)

    st.image(det_img, caption="Detection Result", width=600)

    if len(det_results) == 0:
        st.warning("⚠ No objects detected.")
    else:
        st.write("### 🩺 Detected Findings:")
        for cls, sc in det_results:
            st.write(f"**{cls}** → {sc:.3f}")

    # -----------------------------------------
    # MULTI-LABEL CLASSIFICATION (RESNET50)
    # -----------------------------------------
    st.subheader("🔍 Multi-Label Classification (ResNet-50)")

    cls_out = classify_image(image, CLASS_NAMES)

    for cls_name, score in cls_out:
        st.write(f"**{cls_name}** → {score:.3f}")
