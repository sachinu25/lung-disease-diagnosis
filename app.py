
import os
import sys
from io import BytesIO

import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, HTTPException, UploadFile, File
from PIL import Image

from xray.constant.training_pipeline import NORMALIZE_LIST_1, NORMALIZE_LIST_2, RESIZE
from xray.exception import XRayException
from xray.logger import logging
from xray.ml.model.arch import get_model

app = FastAPI(title="X-ray Pneumonia Detection API")

device = torch.device("cpu")

# ------------------------------------------------------------------
# Preprocessing – must match the training test_transform pipeline
# ------------------------------------------------------------------
transform = transforms.Compose(
    [
        transforms.Resize((RESIZE, RESIZE)),
        transforms.CenterCrop(RESIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_LIST_1, std=NORMALIZE_LIST_2),
    ]
)

# Label mapping (class indices as assigned by ImageFolder: NORMAL=0, PNEUMONIA=1)
label_map = {0: "Normal", 1: "Pneumonia"}


def _load_model():
    """Load the best available model checkpoint from disk.

    Supports both the new checkpoint dict format and the legacy full-model
    format produced by older training runs.
    """
    # Prefer the new checkpoint produced by the upgraded training pipeline
    candidate_paths = [
        os.path.join(os.getcwd(), "artifacts", "best_model.pt"),
        os.path.join(os.getcwd(), "xray_model.pth"),
    ]

    # Also search inside timestamped artifact directories
    artifact_root = os.path.join(os.getcwd(), "artifacts")
    if os.path.isdir(artifact_root):
        for entry in sorted(os.listdir(artifact_root), reverse=True):
            training_dir = os.path.join(
                artifact_root, entry, "model_training", "best_model.pt"
            )
            if os.path.isfile(training_dir):
                candidate_paths.insert(0, training_dir)
                break

    for ckpt_path in candidate_paths:
        if not os.path.isfile(ckpt_path):
            continue

        try:
            payload = torch.load(ckpt_path, map_location=device)

            if isinstance(payload, dict) and "model_state_dict" in payload:
                model_type = payload.get("model_type", "efficientnet_b0")
                threshold = float(payload.get("best_threshold", 0.5))

                model = get_model(model_type=model_type, pretrained=False)
                model.load_state_dict(payload["model_state_dict"])
                model.eval()

                logging.info(
                    f"Loaded checkpoint from '{ckpt_path}' "
                    f"(model_type={model_type}, threshold={threshold:.2f})"
                )
                return model, threshold

            else:
                # Legacy full-model object
                model = payload
                model.eval()
                logging.info(f"Loaded legacy full-model from '{ckpt_path}'")
                return model, 0.5

        except Exception as err:
            logging.warning(f"Could not load checkpoint '{ckpt_path}': {err}")

    raise RuntimeError(
        "No valid model checkpoint found. "
        "Run train.py to produce a checkpoint before starting the API."
    )


try:
    _model, _threshold = _load_model()
except Exception as startup_err:
    logging.warning(f"Model not loaded at startup: {startup_err}")
    _model, _threshold = None, 0.5


@app.get("/")
def root():
    return {"message": "X-ray Diagnosis API is running", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first by running train.py.",
        )

    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    try:
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = _model(input_tensor)
            prob_pneumonia = torch.softmax(output, dim=1)[0, 1].item()
            pred_idx = 1 if prob_pneumonia >= _threshold else 0

        return {
            "prediction_index": pred_idx,
            "prediction_label": label_map[pred_idx],
            "confidence": round(
                prob_pneumonia if pred_idx == 1 else 1.0 - prob_pneumonia, 4
            ),
            "threshold_used": _threshold,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
