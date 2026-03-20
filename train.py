import argparse
import sys

from xray.constant.training_pipeline import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    EPOCH,
    FINETUNE_EPOCHS,
    HEAD_EPOCHS,
    LEARNING_RATE,
    MIXED_PRECISION,
    MODEL_TYPE,
    PRETRAINED,
    SEED,
    TUNE_THRESHOLD,
    USE_CLASS_WEIGHTS,
    WEIGHT_DECAY,
)
from xray.exception import XRayException
from xray.pipeline.train_pipeline import TrainPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lung X-ray Pneumonia Classifier — Training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=MODEL_TYPE,
        choices=["efficientnet_b0", "resnet18", "custom"],
        help="Model architecture to use.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        default=not PRETRAINED,
        help="Disable ImageNet pre-trained weights (for transfer learning models).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCH,
        help="Total training epochs (used for custom CNN).",
    )
    parser.add_argument(
        "--head-epochs",
        type=int,
        default=HEAD_EPOCHS,
        help="Epochs for Phase-1 head-only training (transfer learning).",
    )
    parser.add_argument(
        "--finetune-epochs",
        type=int,
        default=FINETUNE_EPOCHS,
        help="Epochs for Phase-2 fine-tuning (transfer learning).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LEARNING_RATE,
        help="Initial learning rate for AdamW optimizer.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=WEIGHT_DECAY,
        help="L2 weight decay for AdamW optimizer.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="DataLoader batch size.",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        default=not USE_CLASS_WEIGHTS,
        help="Disable WeightedRandomSampler class balancing.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=EARLY_STOPPING_PATIENCE,
        help="Early stopping patience (epochs without improvement).",
    )
    parser.add_argument(
        "--no-mixed-precision",
        action="store_true",
        default=not MIXED_PRECISION,
        help="Disable automatic mixed precision (AMP).",
    )
    parser.add_argument(
        "--no-threshold-tuning",
        action="store_true",
        default=not TUNE_THRESHOLD,
        help="Disable validation-set threshold tuning.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def start_training(args: argparse.Namespace) -> None:
    try:
        # Propagate CLI overrides into the constant namespace so that
        # all config dataclasses pick them up automatically.
        import xray.constant.training_pipeline as C

        C.MODEL_TYPE = args.model_type
        C.PRETRAINED = not args.no_pretrained
        C.EPOCH = args.epochs
        C.HEAD_EPOCHS = args.head_epochs
        C.FINETUNE_EPOCHS = args.finetune_epochs
        C.LEARNING_RATE = args.lr
        C.WEIGHT_DECAY = args.weight_decay
        C.BATCH_SIZE = args.batch_size
        C.USE_CLASS_WEIGHTS = not args.no_class_weights
        C.EARLY_STOPPING_PATIENCE = args.patience
        C.MIXED_PRECISION = not args.no_mixed_precision
        C.TUNE_THRESHOLD = not args.no_threshold_tuning
        C.SEED = args.seed

        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()

    except Exception as e:
        raise XRayException(e, sys)


if __name__ == "__main__":
    args = parse_args()
    print("Training configuration:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print()
    start_training(args)
