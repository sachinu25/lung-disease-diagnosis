from dataclasses import dataclass, field
from torch.utils.data.dataloader import DataLoader


@dataclass
class DataIngestionArtifact:
    train_file_path: str

    test_file_path: str


@dataclass
class DataTransformationArtifact:
    transformed_train_object: DataLoader

    transformed_test_object: DataLoader

    train_transform_file_path: str

    test_transform_file_path: str


@dataclass
class ModelTrainerArtifact:
    trained_model_path: str

    best_val_accuracy: float = 0.0

    best_threshold: float = 0.5


@dataclass
class ModelEvaluationArtifact:
    model_accuracy: float

    model_precision: float = 0.0

    model_recall: float = 0.0

    model_f1: float = 0.0

    model_roc_auc: float = 0.0

    best_threshold: float = 0.5


@dataclass
class ModelPusherArtifact:
    bentoml_model_name: str

    bentoml_service_name: str
