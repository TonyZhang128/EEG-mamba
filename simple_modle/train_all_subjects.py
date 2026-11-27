"""Compare simple Mamba variants on binary valence labels from all subjects."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


EEGMAMBA_ROOT = Path(__file__).resolve().parents[1]
if str(EEGMAMBA_ROOT) not in sys.path:
    sys.path.insert(0, str(EEGMAMBA_ROOT))

from dataset import load_subject_dataset, parse_subject_ids
from simple_modle.model import SimpleMambaClassifier, SimpleMambaConfig


LOGGER = logging.getLogger(__name__)
PROCESSED_DIR = EEGMAMBA_ROOT / "data" / "processed"
DEFAULT_SUBJECT_DIR = PROCESSED_DIR / "overlap50"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "runs"
VARIANTS = ("plain", "channel", "two_level")
METRIC_NAMES = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "roc_auc",
    "log_loss",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare plain, channel-conditioned, and two-level-conditioned "
            "Mamba on binary valence classification across subjects."
        )
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default=None,
        help="optional subject IDs such as s01,s02,5-7; default uses all",
    )
    parser.add_argument(
        "--subject-dir",
        type=Path,
        default=DEFAULT_SUBJECT_DIR,
        help=f"directory containing sXX.npz files (default: {DEFAULT_SUBJECT_DIR})",
    )
    parser.add_argument(
        "--test-subject-ratio",
        type=float,
        default=0.2,
        help="fraction of complete subjects held out for testing",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--d-state", type=int, default=8)
    parser.add_argument("--dt-rank", type=int, default=4)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--condition-alpha", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    args = parser.parse_args()
    if not 0.0 < args.test_subject_ratio < 1.0:
        parser.error("--test-subject-ratio must be in (0, 1)")
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("--epochs and --batch-size must be positive")
    return args


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def standardize(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit feature mean/std on training subjects only."""
    train_flat = x_train.reshape(-1, x_train.shape[-1]).astype(np.float64)
    mean = train_flat.mean(axis=0)
    std = train_flat.std(axis=0)
    std = np.where(std > 1e-8, std, 1.0)

    def transform(values: np.ndarray) -> np.ndarray:
        normalized = (
            values - mean.reshape(1, 1, 1, -1)
        ) / std.reshape(1, 1, 1, -1)
        return normalized.astype(np.float32)

    return transform(x_train), transform(x_test), mean, std


def balanced_class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    weights = labels.size / (2.0 * counts)
    return torch.from_numpy(weights.astype(np.float32))


def train_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.to(device)
    train_dataset = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(x_train)),
        torch.from_numpy(np.ascontiguousarray(y_train)),
    )
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=learning_rate * 0.05,
    )
    criterion = nn.CrossEntropyLoss(
        weight=balanced_class_weights(y_train).to(device)
    )

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            LOGGER.info(
                "epoch %d/%d train_loss=%.4f",
                epoch + 1,
                epochs,
                total_loss / len(train_dataset),
            )

    model.eval()
    probability_parts = []
    with torch.inference_mode():
        for start in range(0, len(x_test), batch_size):
            test_batch = torch.from_numpy(
                np.ascontiguousarray(x_test[start : start + batch_size])
            ).to(device)
            probability_parts.append(
                torch.softmax(model(test_batch), dim=-1).cpu()
            )
    probabilities = torch.cat(probability_parts, dim=0).numpy()
    return probabilities.argmax(axis=1), probabilities


def binary_metrics(
    y_true: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "macro_f1": float(f1_score(y_true, predictions, average="macro")),
        "log_loss": float(log_loss(y_true, probabilities, labels=np.arange(2))),
    }
    metrics["roc_auc"] = (
        float(roc_auc_score(y_true, probabilities[:, 1]))
        if np.unique(y_true).size == 2
        else float("nan")
    )
    return metrics


def select_subjects(
    subject_values: np.ndarray,
    *,
    test_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    subjects = np.unique(subject_values).astype(np.int64)
    if subjects.size < 2:
        raise ValueError("at least two subjects are required for a group split")
    n_test = max(1, round(subjects.size * test_ratio))
    n_test = min(n_test, subjects.size - 1)
    generator = np.random.default_rng(seed)
    shuffled = generator.permutation(subjects)
    return sorted(shuffled[n_test:].tolist()), sorted(shuffled[:n_test].tolist())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    requested_subjects = parse_subject_ids(args.subjects)
    x, y_four_class, subject_values, metadata = load_subject_dataset(
        args.subject_dir,
        subject_ids=requested_subjects,
    )
    if x.ndim != 4 or x.shape[-1] != 5:
        raise ValueError(f"unexpected input shape: {x.shape}")
    if not np.all((y_four_class >= 0) & (y_four_class <= 3)):
        raise ValueError("four-class labels must be in [0, 3]")
    y = (y_four_class // 2).astype(np.int64)

    train_subjects, test_subjects = select_subjects(
        subject_values,
        test_ratio=args.test_subject_ratio,
        seed=args.seed,
    )
    train_mask = np.isin(subject_values, train_subjects)
    test_mask = np.isin(subject_values, test_subjects)
    x_train_raw = x[train_mask]
    x_test_raw = x[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    test_subject_values = subject_values[test_mask]
    x_train, x_test, feature_mean, feature_std = standardize(
        x_train_raw,
        x_test_raw,
    )

    device = resolve_device(args.device)
    LOGGER.info(
        "loaded subjects=%d samples=%d shape=%s",
        len(np.unique(subject_values)),
        len(x),
        tuple(x.shape[1:]),
    )
    LOGGER.info(
        "train_subjects=%s test_subjects=%s train_samples=%d test_samples=%d",
        train_subjects,
        test_subjects,
        len(x_train),
        len(x_test),
    )

    results: dict[str, dict] = {}
    for variant in VARIANTS:
        seed = args.seed
        set_seed(seed)
        model = SimpleMambaClassifier(
            SimpleMambaConfig(
                input_dim=x.shape[-1],
                num_channels=x.shape[1],
                num_classes=2,
                hidden_dim=args.hidden_dim,
                d_state=args.d_state,
                dt_rank=args.dt_rank,
                d_conv=args.d_conv,
                dropout=args.dropout,
                condition_alpha=args.condition_alpha,
                condition_mode=variant,
            )
        )
        LOGGER.info("training variant=%s", variant)
        predictions, probabilities = train_model(
            model,
            x_train,
            y_train,
            x_test,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            device=device,
            seed=seed,
        )
        metrics = binary_metrics(y_test, predictions, probabilities)
        per_subject = {}
        for subject in test_subjects:
            mask = test_subject_values == subject
            per_subject[str(subject)] = binary_metrics(
                y_test[mask],
                predictions[mask],
                probabilities[mask],
            )
        results[variant] = {
            "metrics": metrics,
            "per_subject": per_subject,
            "y_true": y_test.tolist(),
            "y_pred": predictions.tolist(),
            "probabilities": probabilities.tolist(),
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        partial_path = args.output_dir / "all_subjects_valence_binary.partial.json"
        partial_path.write_text(
            json.dumps(
                {
                    "task": "valence_binary",
                    "train_subjects": train_subjects,
                    "test_subjects": test_subjects,
                    "results": results,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        LOGGER.info(
            "%s accuracy=%.4f balanced_accuracy=%.4f macro_f1=%.4f "
            "roc_auc=%.4f log_loss=%.4f",
            variant,
            metrics["accuracy"],
            metrics["balanced_accuracy"],
            metrics["macro_f1"],
            metrics["roc_auc"],
            metrics["log_loss"],
        )

    output = {
        "task": "valence_binary",
        "label_definition": "y_four_class // 2",
        "class_names": ["low_valence", "high_valence"],
        "subject_dir": str(args.subject_dir.resolve()),
        "input_shape": list(x.shape[1:]),
        "metadata": metadata,
        "protocol": {
            "split": "subject_disjoint_holdout",
            "test_subject_ratio": args.test_subject_ratio,
            "seed": args.seed,
            "train_subjects": train_subjects,
            "test_subjects": test_subjects,
            "train_samples": int(train_mask.sum()),
            "test_samples": int(test_mask.sum()),
            "subject_disjoint": True,
        },
        "feature_normalization": {
            "mean": feature_mean.tolist(),
            "std": feature_std.tolist(),
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "d_state": args.d_state,
            "dt_rank": args.dt_rank,
            "d_conv": args.d_conv,
            "dropout": args.dropout,
            "condition_alpha": args.condition_alpha,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "class_weighting": "balanced",
            "device": str(device),
        },
        "results": results,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "all_subjects_valence_binary.json"
    csv_path = args.output_dir / "all_subjects_valence_binary.csv"
    json_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    partial_path = args.output_dir / "all_subjects_valence_binary.partial.json"
    if partial_path.exists():
        partial_path.unlink()
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("variant", *METRIC_NAMES),
        )
        writer.writeheader()
        for variant, result in results.items():
            writer.writerow({"variant": variant, **result["metrics"]})

    print(
        f"subjects={len(np.unique(subject_values))} "
        f"train_subjects={train_subjects} test_subjects={test_subjects}"
    )
    print(
        f"train_samples={len(x_train)} test_samples={len(x_test)} "
        f"shape={tuple(x.shape[1:])}"
    )
    print(
        f"{'variant':<12} {'accuracy':>10} {'balanced':>10} "
        f"{'macro_f1':>10} {'roc_auc':>10} {'log_loss':>10}"
    )
    for variant, result in results.items():
        metrics = result["metrics"]
        print(
            f"{variant:<12} {metrics['accuracy']:>10.4f} "
            f"{metrics['balanced_accuracy']:>10.4f} "
            f"{metrics['macro_f1']:>10.4f} "
            f"{metrics['roc_auc']:>10.4f} "
            f"{metrics['log_loss']:>10.4f}"
        )
    print(f"results_json={json_path.resolve()}")
    print(f"results_csv={csv_path.resolve()}")


if __name__ == "__main__":
    main()
