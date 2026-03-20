import os
import random
import sys

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from xray.constant.training_pipeline import DEVICE, SEED
from xray.entity.artifacts_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from xray.entity.config_entity import ModelTrainerConfig
from xray.exception import XRayException
from xray.logger import logging
from xray.ml.model.arch import get_model


def set_seed(seed: int = SEED) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """Stop training when validation metric stops improving."""

    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: float = None
        self.should_stop: bool = False

    def __call__(self, val_score: float) -> bool:
        if self.best_score is None:
            self.best_score = val_score
            return False
        if val_score > self.best_score + self.min_delta:
            self.best_score = val_score
            self.counter = 0
        else:
            self.counter += 1
            logging.info(
                f"EarlyStopping counter: {self.counter}/{self.patience}"
            )
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        return False


class ModelTrainer:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.model_trainer_config: ModelTrainerConfig = model_trainer_config

        self.data_transformation_artifact: DataTransformationArtifact = (
            data_transformation_artifact
        )

        set_seed(model_trainer_config.seed)

        self.model: Module = get_model(
            model_type=model_trainer_config.model_type,
            pretrained=model_trainer_config.pretrained,
        )

        self.use_amp: bool = (
            model_trainer_config.mixed_precision
            and torch.cuda.is_available()
        )

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def _compute_class_weights(self, loader: DataLoader) -> torch.Tensor:
        """Compute inverse-frequency class weights from a DataLoader."""
        counts = torch.zeros(2)
        for _, labels in loader:
            for label in labels:
                counts[label.item()] += 1
        weights = counts.sum() / (2.0 * counts.clamp(min=1))
        return weights.to(self.model_trainer_config.device)

    def _train_one_epoch(
        self,
        optimizer: Optimizer,
        criterion: nn.Module,
        scaler: GradScaler,
        loader: DataLoader,
    ) -> float:
        self.model.train()
        correct = 0
        total = 0
        pbar = tqdm(loader, desc="Training", leave=False)
        for data, target in pbar:
            data, target = (
                data.to(self.model_trainer_config.device),
                target.to(self.model_trainer_config.device),
            )
            optimizer.zero_grad()

            with autocast("cuda" if torch.cuda.is_available() else "cpu", enabled=self.use_amp):
                output = self.model(data)
                loss = criterion(output, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += len(target)

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100.0 * correct / total:.2f}%",
            )

        return 100.0 * correct / max(total, 1)

    def _validate(
        self, criterion: nn.Module, loader: DataLoader
    ):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for data, target in loader:
                data, target = (
                    data.to(self.model_trainer_config.device),
                    target.to(self.model_trainer_config.device),
                )
                with autocast("cuda" if torch.cuda.is_available() else "cpu", enabled=self.use_amp):
                    output = self.model(data)
                    loss = criterion(output, target)

                val_loss += loss.item() * len(target)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += len(target)

                probs = torch.softmax(output, dim=1)[:, 1]
                all_probs.extend(probs.cpu().tolist())
                all_targets.extend(target.cpu().tolist())

        avg_loss = val_loss / max(total, 1)
        accuracy = 100.0 * correct / max(total, 1)
        return avg_loss, accuracy, all_probs, all_targets

    def _tune_threshold(self, probs, targets) -> float:
        """Find the decision threshold that maximises F1 on the validation set."""
        from sklearn.metrics import f1_score

        best_thresh = 0.5
        best_f1 = 0.0
        for thresh in [i / 100 for i in range(10, 91)]:
            preds = [1 if p >= thresh else 0 for p in probs]
            f1 = f1_score(targets, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        logging.info(f"Best threshold: {best_thresh:.2f}  (val F1={best_f1:.4f})")
        return best_thresh

    # ------------------------------------------------------------------
    # Two-phase training for transfer learning models
    # ------------------------------------------------------------------

    def _phase_train(
        self,
        num_epochs: int,
        phase: str,
        criterion: nn.Module,
        base_lr: float,
    ):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=base_lr,
            weight_decay=self.model_trainer_config.optimizer_params.get(
                "weight_decay", 1e-4
            ),
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2
        )
        scaler = GradScaler("cuda" if torch.cuda.is_available() else "cpu", enabled=self.use_amp)
        early_stopping = EarlyStopping(
            patience=self.model_trainer_config.early_stopping_patience
        )

        best_val_acc = 0.0
        best_state = None
        best_threshold = 0.5

        train_loader = self.data_transformation_artifact.transformed_train_object
        val_loader = self.data_transformation_artifact.transformed_test_object

        for epoch in range(1, num_epochs + 1):
            logging.info(f"[{phase}] Epoch {epoch}/{num_epochs}")

            train_acc = self._train_one_epoch(
                optimizer, criterion, scaler, train_loader
            )

            val_loss, val_acc, val_probs, val_targets = self._validate(
                criterion, val_loader
            )

            scheduler.step(val_acc)

            logging.info(
                f"[{phase}] Epoch {epoch}: "
                f"train_acc={train_acc:.2f}%  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.2f}%"
            )
            print(
                f"[{phase}] Epoch {epoch}/{num_epochs}  "
                f"train_acc={train_acc:.2f}%  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.2f}%"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {
                    k: v.cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
                if self.model_trainer_config.tune_threshold:
                    best_threshold = self._tune_threshold(val_probs, val_targets)

            if early_stopping(val_acc):
                logging.info(f"[{phase}] Early stopping triggered at epoch {epoch}")
                break

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return best_val_acc, best_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info(
                "Entered the initiate_model_trainer method of Model trainer class"
            )

            device = self.model_trainer_config.device
            model_type = self.model_trainer_config.model_type

            self.model = self.model.to(device)

            # Class-weighted loss
            train_loader = self.data_transformation_artifact.transformed_train_object
            class_weights = self._compute_class_weights(train_loader)
            criterion = nn.CrossEntropyLoss(weight=class_weights)

            best_val_acc = 0.0
            best_threshold = 0.5

            is_transfer = model_type in ("efficientnet_b0", "resnet18")

            if is_transfer:
                # Phase 1 – head-only training
                logging.info("Phase 1: Freezing backbone, training classifier head")
                self.model.freeze_backbone()

                phase1_acc, phase1_thresh = self._phase_train(
                    num_epochs=self.model_trainer_config.head_epochs,
                    phase="head",
                    criterion=criterion,
                    base_lr=1e-3,
                )

                # Phase 2 – fine-tune deeper layers
                logging.info("Phase 2: Unfreezing deeper layers for fine-tuning")
                self.model.unfreeze_layers(num_blocks=3)

                phase2_acc, phase2_thresh = self._phase_train(
                    num_epochs=self.model_trainer_config.finetune_epochs,
                    phase="finetune",
                    criterion=criterion,
                    base_lr=1e-4,
                )

                best_val_acc = max(phase1_acc, phase2_acc)
                best_threshold = (
                    phase2_thresh if phase2_acc >= phase1_acc else phase1_thresh
                )

            else:
                # Single-phase training for the custom CNN
                best_val_acc, best_threshold = self._phase_train(
                    num_epochs=self.model_trainer_config.epochs,
                    phase="custom_cnn",
                    criterion=criterion,
                    base_lr=self.model_trainer_config.optimizer_params.get("lr", 1e-3),
                )

            # Save checkpoint
            os.makedirs(self.model_trainer_config.artifact_dir, exist_ok=True)

            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "model_type": model_type,
                "best_val_accuracy": best_val_acc,
                "best_threshold": best_threshold,
            }

            torch.save(checkpoint, self.model_trainer_config.best_model_path)
            # Also save full model at the legacy path for BentoML compatibility
            torch.save(self.model, self.model_trainer_config.trained_model_path)

            train_transforms_obj = joblib.load(
                self.data_transformation_artifact.train_transform_file_path
            )

            try:
                import bentoml

                bentoml.pytorch.save_model(
                    name="xray_model",
                    model=self.model,
                    custom_objects={
                        "transform": train_transforms_obj,
                        "model_type": model_type,
                        "best_threshold": best_threshold,
                    },
                )
            except Exception as bento_err:
                logging.warning(f"BentoML save skipped: {bento_err}")

            model_trainer_artifact: ModelTrainerArtifact = ModelTrainerArtifact(
                trained_model_path=self.model_trainer_config.best_model_path,
                best_val_accuracy=best_val_acc,
                best_threshold=best_threshold,
            )

            logging.info(
                f"Training complete. Best val accuracy: {best_val_acc:.2f}%  "
                f"Threshold: {best_threshold:.2f}"
            )
            logging.info(
                "Exited the initiate_model_trainer method of Model trainer class"
            )

            return model_trainer_artifact

        except Exception as e:
            raise XRayException(e, sys)
