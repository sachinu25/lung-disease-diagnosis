import sys
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from xray.entity.artifacts_entity import (
    DataTransformationArtifact,
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
)
from xray.entity.config_entity import ModelEvaluationConfig
from xray.exception import XRayException
from xray.logger import logging
from xray.ml.model.arch import get_model


class ModelEvaluation:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_evaluation_config: ModelEvaluationConfig,
        model_trainer_artifact: ModelTrainerArtifact,
    ):
        self.data_transformation_artifact = data_transformation_artifact

        self.model_evaluation_config = model_evaluation_config

        self.model_trainer_artifact = model_trainer_artifact

    def _load_model(self) -> Tuple[nn.Module, float]:
        """Load model from the best-checkpoint file; return model and threshold."""
        checkpoint_path = self.model_trainer_artifact.trained_model_path
        device = self.model_evaluation_config.device

        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)

            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model_type = checkpoint.get("model_type", "efficientnet_b0")
                threshold = float(checkpoint.get("best_threshold", 0.5))

                model = get_model(model_type=model_type, pretrained=False)
                model.load_state_dict(checkpoint["model_state_dict"])

                logging.info(
                    f"Loaded checkpoint: model_type={model_type}, "
                    f"threshold={threshold:.2f}"
                )
            else:
                # Legacy: full model object saved with torch.save(model, path)
                model = checkpoint
                threshold = float(
                    getattr(self.model_trainer_artifact, "best_threshold", 0.5)
                )
                logging.info("Loaded legacy full-model checkpoint")

        except Exception as load_err:
            raise XRayException(
                RuntimeError(
                    f"Failed to load model from '{checkpoint_path}': {load_err}"
                ),
                sys,
            )

        model.to(device)
        model.eval()
        return model, threshold

    def _collect_predictions(
        self, model: nn.Module, loader: DataLoader, threshold: float
    ) -> Tuple[List[int], List[int], List[float]]:
        """Run inference and collect labels, predictions, and class-1 probabilities."""
        all_labels: List[int] = []
        all_preds: List[int] = []
        all_probs: List[float] = []
        device = self.model_evaluation_config.device

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)

                output = model(images)
                probs = torch.softmax(output, dim=1)[:, 1]

                preds = (probs >= threshold).long()

                all_labels.extend(labels.cpu().tolist())
                all_preds.extend(preds.cpu().tolist())
                all_probs.extend(probs.cpu().tolist())

        return all_labels, all_preds, all_probs

    def test_net(self) -> Tuple[float, float, float, float, float, float]:
        """Evaluate the model and return (accuracy, precision, recall, f1, roc_auc, threshold)."""
        logging.info("Entered the test_net method of Model evaluation class")

        try:
            from sklearn.metrics import (
                accuracy_score,
                classification_report,
                confusion_matrix,
                f1_score,
                precision_score,
                recall_score,
                roc_auc_score,
            )

            model, threshold = self._load_model()

            test_loader = (
                self.data_transformation_artifact.transformed_test_object
            )

            all_labels, all_preds, all_probs = self._collect_predictions(
                model, test_loader, threshold
            )

            accuracy = accuracy_score(all_labels, all_preds) * 100
            precision = precision_score(all_labels, all_preds, zero_division=0)
            recall = recall_score(all_labels, all_preds, zero_division=0)
            f1 = f1_score(all_labels, all_preds, zero_division=0)

            try:
                roc_auc = roc_auc_score(all_labels, all_probs)
            except ValueError:
                roc_auc = 0.0

            cm = confusion_matrix(all_labels, all_preds)
            report = classification_report(
                all_labels, all_preds, target_names=["NORMAL", "PNEUMONIA"]
            )

            logging.info(
                f"Evaluation Results — "
                f"Accuracy: {accuracy:.2f}%  "
                f"Precision: {precision:.4f}  "
                f"Recall: {recall:.4f}  "
                f"F1: {f1:.4f}  "
                f"ROC-AUC: {roc_auc:.4f}  "
                f"Threshold: {threshold:.2f}"
            )

            print("\n" + "=" * 60)
            print("MODEL EVALUATION RESULTS")
            print("=" * 60)
            print(f"  Accuracy  : {accuracy:.2f}%")
            print(f"  Precision : {precision:.4f}")
            print(f"  Recall    : {recall:.4f}")
            print(f"  F1-Score  : {f1:.4f}")
            print(f"  ROC-AUC   : {roc_auc:.4f}")
            print(f"  Threshold : {threshold:.2f}")
            print("\nClassification Report:")
            print(report)
            print("Confusion Matrix:")
            print(cm)
            print("=" * 60 + "\n")

            logging.info("Exited the test_net method of Model evaluation class")

            return accuracy, precision, recall, f1, roc_auc, threshold

        except Exception as e:
            raise XRayException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        logging.info(
            "Entered the initiate_model_evaluation method of Model evaluation class"
        )

        try:
            accuracy, precision, recall, f1, roc_auc, threshold = self.test_net()

            model_evaluation_artifact: ModelEvaluationArtifact = (
                ModelEvaluationArtifact(
                    model_accuracy=accuracy,
                    model_precision=precision,
                    model_recall=recall,
                    model_f1=f1,
                    model_roc_auc=roc_auc,
                    best_threshold=threshold,
                )
            )

            logging.info(
                "Exited the initiate_model_evaluation method of Model evaluation class"
            )

            return model_evaluation_artifact

        except Exception as e:
            raise XRayException(e, sys)
