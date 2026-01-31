import os
import torch
import streamlit as st
from PIL import Image
import torchvision.transforms as transforms

from xray.ml.model.arch import Net

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="X-ray Pneumonia Detection",
    page_icon="🫁",
    layout="centered"
)

st.title("🫁 X-ray Pneumonia Detection")
st.write("Chest X-ray upload karo aur Pneumonia / Normal prediction dekho.")

# ---------------- DEVICE ----------------
device = torch.device("cpu")

# ---------------- LOAD MODEL ----------------
MODEL_PATH = os.path.join(os.getcwd(), "xray_model.pth")

@st.cache_resource
def load_model():
    model = Net().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model

model = load_model()

# ---------------- TRANSFORMS ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

label_map = {0: "Normal", 1: "Pneumonia"}

# ---------------- UI ----------------
uploaded_file = st.file_uploader(
    "Chest X-ray Image Upload karo",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    if st.button("🔍 Predict"):
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).item()

        st.success(f"🧠 Prediction: **{label_map[pred]}**")
