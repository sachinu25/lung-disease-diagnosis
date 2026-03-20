import os
import sys
from typing import Tuple

import joblib
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

from xray.entity.artifacts_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
)
from xray.entity.config_entity import DataTransformationConfig
from xray.exception import XRayException
from xray.logger import logging


class DataTransformation:
    def __init__(
        self,
        data_transformation_config: DataTransformationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
    ):
        self.data_transformation_config = data_transformation_config

        self.data_ingestion_artifact = data_ingestion_artifact

    def transforming_training_data(self) -> transforms.Compose:
        try:
            logging.info(
                "Entered the transforming_training_data method of Data transformation class"
            )

            train_transform: transforms.Compose = transforms.Compose(
                [
                    transforms.Resize(self.data_transformation_config.RESIZE),
                    transforms.CenterCrop(self.data_transformation_config.CENTERCROP),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(
                        self.data_transformation_config.RANDOMROTATION
                    ),
                    transforms.RandomAffine(
                        degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)
                    ),
                    transforms.ColorJitter(
                        **self.data_transformation_config.color_jitter_transforms
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        **self.data_transformation_config.normalize_transforms
                    ),
                ]
            )

            logging.info(
                "Exited the transforming_training_data method of Data transformation class"
            )

            return train_transform

        except Exception as e:
            raise XRayException(e, sys)

    def transforming_testing_data(self) -> transforms.Compose:
        logging.info(
            "Entered the transforming_testing_data method of Data transformation class"
        )

        try:
            test_transform: transforms.Compose = transforms.Compose(
                [
                    transforms.Resize(self.data_transformation_config.RESIZE),
                    transforms.CenterCrop(self.data_transformation_config.CENTERCROP),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        **self.data_transformation_config.normalize_transforms
                    ),
                ]
            )

            logging.info(
                "Exited the transforming_testing_data method of Data transformation class"
            )

            return test_transform

        except Exception as e:
            raise XRayException(e, sys)

    def _build_weighted_sampler(self, dataset: ImageFolder) -> WeightedRandomSampler:
        """Build a WeightedRandomSampler to handle class imbalance."""
        targets = dataset.targets
        class_counts = torch.zeros(len(dataset.classes))
        for label in targets:
            class_counts[label] += 1

        class_weights = 1.0 / class_counts.clamp(min=1)
        sample_weights = torch.tensor([class_weights[t] for t in targets])

        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    def data_loader(
        self, train_transform: transforms.Compose, test_transform: transforms.Compose
    ) -> Tuple[DataLoader, DataLoader]:
        try:
            logging.info("Entered the data_loader method of Data transformation class")

            train_data: Dataset = ImageFolder(
                os.path.join(self.data_ingestion_artifact.train_file_path),
                transform=train_transform,
            )

            test_data: Dataset = ImageFolder(
                os.path.join(self.data_ingestion_artifact.test_file_path),
                transform=test_transform,
            )

            logging.info("Created train data and test data paths")

            loader_params = dict(self.data_transformation_config.data_loader_params)

            if self.data_transformation_config.use_class_weights:
                sampler = self._build_weighted_sampler(train_data)
                # WeightedRandomSampler is mutually exclusive with shuffle=True
                loader_params["shuffle"] = False
                train_loader: DataLoader = DataLoader(
                    train_data,
                    sampler=sampler,
                    **{k: v for k, v in loader_params.items() if k != "shuffle"},
                )
            else:
                loader_params["shuffle"] = True
                train_loader: DataLoader = DataLoader(train_data, **loader_params)

            test_loader: DataLoader = DataLoader(
                test_data,
                batch_size=loader_params["batch_size"],
                shuffle=False,
                pin_memory=loader_params.get("pin_memory", True),
            )

            logging.info("Exited the data_loader method of Data transformation class")

            return train_loader, test_loader

        except Exception as e:
            raise XRayException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info(
                "Entered the initiate_data_transformation method of Data transformation class"
            )

            train_transform: transforms.Compose = self.transforming_training_data()

            test_transform: transforms.Compose = self.transforming_testing_data()

            os.makedirs(self.data_transformation_config.artifact_dir, exist_ok=True)

            joblib.dump(
                train_transform, self.data_transformation_config.train_transforms_file
            )

            joblib.dump(
                test_transform, self.data_transformation_config.test_transforms_file
            )

            train_loader, test_loader = self.data_loader(
                train_transform=train_transform, test_transform=test_transform
            )

            data_transformation_artifact: DataTransformationArtifact = DataTransformationArtifact(
                transformed_train_object=train_loader,
                transformed_test_object=test_loader,
                train_transform_file_path=self.data_transformation_config.train_transforms_file,
                test_transform_file_path=self.data_transformation_config.test_transforms_file,
            )

            logging.info(
                "Exited the initiate_data_transformation method of Data transformation class"
            )

            return data_transformation_artifact

        except Exception as e:
            raise XRayException(e, sys)
