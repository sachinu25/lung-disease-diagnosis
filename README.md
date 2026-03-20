#  Xray Lung Classifier

## Problem statement
Pneumonia is an inflammatory condition of the lungs that primarily affects the alveoli (small air sacs). It is commonly caused by bacterial or viral infections and presents symptoms such as cough, chest pain, fever, and difficulty breathing. The severity varies depending on the patient's health condition and immune response.

Early and accurate diagnosis is critical for effective treatment. Chest X-ray imaging is one of the most widely used diagnostic tools.
The objective of this project is to build an end-to-end deep learning–based API that classifies chest X-ray images as Pneumonia or Normal.

## Solution Proposed
This project uses Computer Vision and Deep Learning to automatically detect pneumonia from chest X-ray images.

Key Highlights:
- **Transfer learning** with EfficientNet-B0 (default) or ResNet-18, both with ImageNet initialisation
- Two-phase training: frozen backbone → classifier head, then fine-tuning of deeper layers
- Custom CNN (`--model-type custom`) retained for backward compatibility
- The trained model is exposed via a FastAPI service
- The application is Dockerized for portability
- Deployed on AWS Cloud using modern MLOps practices
- CI/CD enabled using GitHub Actions

---

## 🚀 Quick-Start Training

### 1. Set up environment
```bash
conda create -p env python=3.10 -y
conda activate ./env
pip install -r requirements.txt
```

### 2. Export AWS credentials (for S3 data download)
```bash
export AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID>
export AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>
export AWS_DEFAULT_REGION=<AWS_DEFAULT_REGION>
```

### 3. Train the model

**Recommended – EfficientNet-B0 with all improvements (targets ~98% accuracy):**
```bash
python train.py \
  --model-type efficientnet_b0 \
  --head-epochs 5 \
  --finetune-epochs 20 \
  --lr 1e-3 \
  --batch-size 32 \
  --patience 7 \
  --seed 42
```

**ResNet-18 alternative:**
```bash
python train.py --model-type resnet18 --head-epochs 5 --finetune-epochs 15
```

**Legacy custom CNN (backward-compatible):**
```bash
python train.py --model-type custom --epochs 30 --lr 5e-3
```

**Disable optional features if needed:**
```bash
python train.py --no-class-weights --no-mixed-precision --no-threshold-tuning
```

### All CLI flags
| Flag | Default | Description |
|---|---|---|
| `--model-type` | `efficientnet_b0` | `efficientnet_b0`, `resnet18`, or `custom` |
| `--head-epochs` | `5` | Phase-1 (frozen backbone) epochs |
| `--finetune-epochs` | `15` | Phase-2 (unfrozen layers) epochs |
| `--epochs` | `20` | Total epochs for custom CNN |
| `--lr` | `1e-3` | AdamW base learning rate |
| `--weight-decay` | `1e-4` | AdamW weight decay |
| `--batch-size` | `32` | DataLoader batch size |
| `--patience` | `7` | Early-stopping patience |
| `--seed` | `42` | Random seed |
| `--no-pretrained` | off | Skip ImageNet weights |
| `--no-class-weights` | off | Disable WeightedRandomSampler |
| `--no-mixed-precision` | off | Disable AMP |
| `--no-threshold-tuning` | off | Use fixed 0.5 threshold |

---

## 🧠 Model Architecture

### Transfer Learning Models (recommended)
| Model | Params | Notes |
|---|---|---|
| **EfficientNet-B0** | 5.3M | Default; best accuracy/speed trade-off |
| **ResNet-18** | 11.2M | Strong baseline; well understood |

Both models use two-phase training:
1. **Phase 1** — backbone frozen, only the new classification head is trained
2. **Phase 2** — last 3 blocks unfrozen, full end-to-end fine-tuning at a lower LR

### Custom CNN (legacy)
Original lightweight CNN in `xray/ml/model/arch.py`, selectable via `--model-type custom`.

---

## 🔧 Training Improvements

| Feature | Implementation |
|---|---|
| **Optimizer** | AdamW (replaces SGD) |
| **LR Scheduler** | ReduceLROnPlateau (patience=2, factor=0.5) |
| **Mixed Precision** | `torch.amp.autocast` + `GradScaler` (auto-enabled on CUDA) |
| **Class Imbalance** | `WeightedRandomSampler` (inverse frequency) + optional weighted loss |
| **Early Stopping** | Val-accuracy–based with configurable patience |
| **Best Checkpoint** | Saved to `artifacts/<timestamp>/model_training/best_model.pt` |
| **Threshold Tuning** | Val-set F1-optimal threshold (replaces fixed 0.5) |
| **Reproducibility** | Fixed seed for Python, NumPy, PyTorch, and CUDA |

### Data Augmentation (training only)
- `RandomHorizontalFlip`
- `RandomRotation(±10°)`
- `RandomAffine` (translation ±5%, scale 0.95–1.05)
- `ColorJitter` (brightness/contrast/saturation/hue)
- ImageNet normalisation (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)

Test/inference transform: **Resize → CenterCrop → ToTensor → Normalize** (no augmentation).

---

## 📊 Metrics & Evaluation

After training, the pipeline automatically evaluates on the test set and reports:
```
============================
MODEL EVALUATION RESULTS
============================
  Accuracy  : xx.xx%
  Precision : x.xxxx
  Recall    : x.xxxx
  F1-Score  : x.xxxx
  ROC-AUC   : x.xxxx
  Threshold : x.xx

Classification Report:
              precision    recall  f1-score   support
      NORMAL       ...
   PNEUMONIA       ...
Confusion Matrix:
[[ TN  FP ]
 [ FN  TP ]]
```

**Target:** ~98% validation/test accuracy with EfficientNet-B0 + 20 fine-tune epochs on the Kaggle Chest X-Ray dataset. Actual results will vary with dataset size and quality.

---

## 🌐 FastAPI Inference

### Run the API
```bash
uvicorn app:app --host 0.0.0.0 --port 8001 --reload
```

### Endpoints
| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check |
| `/health` | GET | Model loaded status |
| `/predict` | POST | Upload a JPEG/PNG chest X-ray |

### Example request
```bash
curl -X POST "http://localhost:8001/predict" \
  -H "accept: application/json" \
  -F "file=@chest_xray.jpg"
```

### Example response
```json
{
  "prediction_index": 1,
  "prediction_label": "Pneumonia",
  "confidence": 0.9821,
  "threshold_used": 0.45
}
```

### Checkpoint loading (automatic)
The API searches for the best checkpoint in this priority order:
1. `artifacts/<latest_timestamp>/model_training/best_model.pt`
2. `artifacts/best_model.pt`
3. `xray_model.pth` (legacy format)

Both the new dict-based checkpoint format and the legacy full-model format are supported.

---

## 🐳 Docker

```bash
# Build
docker build -t xray_classification .

# Run
docker run -d -p 8001:8001 \
  -e AWS_ACCESS_KEY_ID=<key> \
  -e AWS_SECRET_ACCESS_KEY=<secret> \
  xray_classification
```

---

## 📁 Project Structure

```
xray/
├── components/
│   ├── data_ingestion.py       # S3 download
│   ├── data_transformation.py  # Augmentation + WeightedRandomSampler
│   ├── model_training.py       # Two-phase training, AMP, early stopping
│   ├── model_evaluation.py     # Accuracy/Precision/Recall/F1/ROC-AUC
│   └── model_pusher.py         # BentoML packaging
├── constant/training_pipeline/ # All hyperparameter defaults
├── entity/
│   ├── artifacts_entity.py     # Typed artifact dataclasses
│   └── config_entity.py        # Configuration dataclasses
├── exception.py
├── logger.py
└── ml/model/
    ├── arch.py                 # Net (custom CNN), EfficientNetB0, ResNet18, get_model()
    └── model_service.py        # BentoML service definition
app.py                          # FastAPI app
train.py                        # CLI training entry point
requirements.txt
```

---

## ⚠️ Notes on Expected Accuracy

- **~98% accuracy** is achievable on the Kaggle Chest X-Ray Images (Pneumonia) dataset with EfficientNet-B0 and ≥20 fine-tune epochs on a GPU.
- Results depend heavily on the train/test split and class distribution. The dataset is **imbalanced** (~3:1 Pneumonia:Normal), which is handled by `WeightedRandomSampler`.
- On CPU, training will be slow. Use `--finetune-epochs 5` for a quick test run.
- Reported accuracy on external or proprietary datasets may differ. Always interpret F1 and ROC-AUC alongside accuracy for medical tasks.

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API%20Framework-green?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue?logo=docker&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-Cloud%20Deployment-orange?logo=amazonaws&logoColor=white)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-black?logo=githubactions&logoColor=white)

![Model Accuracy](https://img.shields.io/badge/Target%20Accuracy-~98%25-success)
![Medical AI](https://img.shields.io/badge/Domain-Medical%20AI%20%7C%20Healthcare-red?logo=databricks)
