
import os
import torch
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from io import BytesIO
import torchvision.transforms as transforms

from xray.ml.model.arch import Net 

app = FastAPI(title="X-ray Pneumonia Detection API")


device = torch.device("cpu")

# Load model
MODEL_PATH = os.path.join(os.getcwd(), "xray_model.pth")

model = Net().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Label mapping
label_map = {
    0: "Normal",
    1: "Pneumonia"
}

@app.get("/")
def root():
    return {
        "message": "X-ray Diagnosis API is running",
        "docs": "/docs"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = torch.argmax(output, dim=1).item()

    return {
        "prediction_index": pred_idx,
        "prediction_label": label_map[pred_idx]
    }
