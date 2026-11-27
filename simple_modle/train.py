"""Train and compare simple single-direction Mamba variants on one subject."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler


EEGMAMBA_ROOT = Path(__file__).resolve().parents[1]
if str(EEGMAMBA_ROOT) not in sys.path:
    sys.path.insert(0, str(EEGMAMBA_ROOT))

from simple_modle.model import ModelVariant, SimpleMambaClassifier, SimpleMambaConfig


LOGGER = logging.getLogger(__name__)
PROCESSED_DIR = EEGMAMBA_ROOT / "data" / "processed"
DEFAULT_SUBJECT_DIR = PROCESSED_DIR / "overlap50"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "runs"
CLASS_NAMES = ("LVLA", "LVHA", "HVLA", "HVHA")
METRIC_NAMES = ("accuracy", "balanced_accuracy", "macro_f1", "log_loss")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a plain single-direction Mamba with channel-conditioned "
            "and two-level-conditioned variants on one subject."
        )
    )
    parser.add_argument("--subject", type=int, default=1, help="subject number")
    parser.add_argument(
        "--data-file",
        type=Path,
        default=None,
        help="single-subject .npz path; overrides --subject file discovery",
    )
    parser.add_argument(
        "--subject-dir",
        type=Path,
        default=DEFAULT_SUBJECT_DIR,
        help=f"directory containing sXX.npz files (default: {DEFAULT_SUBJECT_DIR})",
    )
    parser.add_argument(
        "--combined-cache",
        type=Path,
        default=None,
        help="optional combined cache fallback if the subject file is absent",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("plain", "channel", "two_level"),
        default=("plain", "channel", "two_level"),
    )
    parser.add_argument(
        "--split-mode",
        choices=("holdout", "cv"),
        default="holdout",
        help="single stratified train/test split or stratified cross-validation",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="test fraction for holdout mode (default: 0.2)",
    )
    parser.add_argument("--folds", type=int, default=5, help="stratified CV folds")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--d-state", type=int, default=8)
    parser.add_argument("--dt-rank", type=int, default=4)
    parser.add_argument("--d-conv", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--condition-alpha", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or a specific torch device such as cuda:0",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"result directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    if args.subject <= 0:
        parser.error("--subject must be positive")
    if args.split_mode == "cv" and not 2 <= args.folds:
        parser.error("--folds must be at least 2")
    if not 0.0 < args.test_size < 1.0:
        parser.error("--test-size must be in (0, 1)")
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("--epochs and --batch-size must be positive")
    return args


def _subject_filename(subject: int) -> str:
    width = max(2, len(str(subject)))
    return f"s{subject:0{width}d}.npz"


def load_subject_data(
    *,
    subject: int,
    data_file: Path | None,
    subject_dir: Path,
    combined_cache: Path | None,
) -> tuple[np.ndarray, np.ndarray, Path]:
    """Load one subject from an sXX.npz file or the combined cache."""
    source_path = (
        data_file
        if data_file is not None
        else subject_dir / _subject_filename(subject)
    )
    if source_path.exists():
        with np.load(source_path, allow_pickle=False) as archive:
            if "subjects" in archive.files and not np.all(
                archive["subjects"] == subject
            ):
                raise ValueError(
                    f"{source_path} contains subjects other than {subject}"
                )
            x = np.asarray(archive["x"], dtype=np.float32)
            y = np.asarray(archive["y"], dtype=np.int64)
        return x, y, source_path.resolve()

    if data_file is not None:
        raise FileNotFoundError(f"data file does not exist: {data_file}")
    if combined_cache is None:
        raise FileNotFoundError(f"subject data file does not exist: {source_path}")
    if not combined_cache.exists():
        raise FileNotFoundError(
            f"neither {source_path} nor fallback cache {combined_cache} exists"
        )
    with np.load(combined_cache, allow_pickle=False) as archive:
        subjects = np.asarray(archive["subjects"])
        mask = subjects == subject
        if not np.any(mask):
            raise ValueError(f"subject {subject} is absent from {combined_cache}")
        x = np.asarray(archive["x"][mask], dtype=np.float32)
        y = np.asarray(archive["y"][mask], dtype=np.int64)
    return x, y, combined_cache.resolve()


def validate_data(x: np.ndarray, y: np.ndarray) -> None:
    if x.ndim != 4:
        raise ValueError(f"x must have shape [samples, channels, time, features], got {x.shape}")
    if y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError(f"x/y sample counts differ: {x.shape} and {y.shape}")
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError("labels must be integer values")
    if np.any(y < 0) or np.any(y >= len(CLASS_NAMES)):
        raise ValueError(f"labels must be in [0, {len(CLASS_NAMES) - 1}]")
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains NaN or infinite values")


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


def normalize_fold(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit feature normalization on training samples only."""
    scaler = StandardScaler()
    train_flat = x_train.reshape(-1, x_train.shape[-1])
    scaler.fit(train_flat)

    def transform(values: np.ndarray) -> np.ndarray:
        transformed = scaler.transform(values.reshape(-1, values.shape[-1]))
        return transformed.reshape(values.shape).astype(np.float32, copy=False)

    return transform(x_train), transform(x_test)


def balanced_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    present = counts > 0
    weights = np.zeros(num_classes, dtype=np.float32)
    weights[present] = labels.size / (present.sum() * counts[present])
    return torch.from_numpy(weights)


def train_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    seed: int,
    num_classes: int,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.to(device)
    x_train_t = torch.from_numpy(np.ascontiguousarray(x_train))
    y_train_t = torch.from_numpy(np.ascontiguousarray(y_train))
    x_test_t = torch.from_numpy(np.ascontiguousarray(x_test)).to(device)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(x_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
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
    class_weights = balanced_class_weights(y_train, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

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
        if (epoch + 1) % 20 == 0:
            LOGGER.info(
                "epoch %d/%d train_loss=%.4f",
                epoch + 1,
                epochs,
                total_loss / len(loader.dataset),
            )

    model.eval()
    with torch.no_grad():
        logits = model(x_test_t)
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
    predictions = probabilities.argmax(axis=1)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, predictions)
        ),
        "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        "log_loss": float(
            log_loss(y_test, probabilities, labels=np.arange(num_classes))
        ),
    }
    return metrics, predictions, probabilities


def aggregate_metrics(records: Iterable[dict[str, float]]) -> dict[str, dict[str, float]]:
    rows = list(records)
    aggregate: dict[str, dict[str, float]] = {}
    for metric in METRIC_NAMES:
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        aggregate[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        }
    return aggregate


def run_experiment(args: argparse.Namespace) -> dict:
    x, y, source_path = load_subject_data(
        subject=args.subject,
        data_file=args.data_file,
        subject_dir=args.subject_dir,
        combined_cache=args.combined_cache,
    )
    validate_data(x, y)
    class_counts = np.bincount(y, minlength=len(CLASS_NAMES))
    if args.split_mode == "cv" and class_counts.min() < args.folds:
        raise ValueError(
            f"{args.folds}-fold stratified CV needs at least {args.folds} "
            f"samples per class, got {class_counts.tolist()}"
        )

    device = resolve_device(args.device)
    if args.split_mode == "holdout":
        train_indices, test_indices = train_test_split(
            np.arange(y.shape[0]),
            test_size=args.test_size,
            stratify=y,
            random_state=args.seed,
        )
        splits = [(train_indices, test_indices)]
    else:
        splitter = StratifiedKFold(
            n_splits=args.folds,
            shuffle=True,
            random_state=args.seed,
        )
        splits = list(splitter.split(x, y))
    records: dict[str, list[dict]] = {variant: [] for variant in args.variants}

    for fold_index, (train_indices, test_indices) in enumerate(splits, start=1):
        x_train, x_test = normalize_fold(x[train_indices], x[test_indices])
        y_train, y_test = y[train_indices], y[test_indices]
        LOGGER.info(
            "split %d/%d train=%d test=%d",
            fold_index,
            len(splits),
            len(train_indices),
            len(test_indices),
        )

        for variant in args.variants:
            seed = args.seed + fold_index * 1000
            set_seed(seed)
            config = SimpleMambaConfig(
                input_dim=x.shape[-1],
                num_channels=x.shape[1],
                num_classes=len(CLASS_NAMES),
                hidden_dim=args.hidden_dim,
                d_state=args.d_state,
                dt_rank=args.dt_rank,
                d_conv=args.d_conv,
                dropout=args.dropout,
                condition_alpha=args.condition_alpha,
                condition_mode=variant,
            )
            model = SimpleMambaClassifier(config)
            LOGGER.info("training variant=%s fold=%d", variant, fold_index)
            metrics, predictions, probabilities = train_model(
                model,
                x_train,
                y_train,
                x_test,
                y_test,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                weight_decay=args.weight_decay,
                device=device,
                seed=seed,
                num_classes=len(CLASS_NAMES),
            )
            records[variant].append(
                {
                    "fold": fold_index,
                    "train_indices": train_indices.tolist(),
                    "test_indices": test_indices.tolist(),
                    "y_true": y_test.tolist(),
                    "y_pred": predictions.tolist(),
                    "probabilities": probabilities.tolist(),
                    **metrics,
                }
            )
            LOGGER.info(
                "%s fold=%d accuracy=%.4f balanced_accuracy=%.4f macro_f1=%.4f",
                variant,
                fold_index,
                metrics["accuracy"],
                metrics["balanced_accuracy"],
                metrics["macro_f1"],
            )

    aggregates = {
        variant: aggregate_metrics(records[variant]) for variant in args.variants
    }
    comparisons: dict[str, dict[str, dict[str, float]]] = {}
    if "plain" in args.variants:
        plain_records = records["plain"]
        for variant in args.variants:
            if variant == "plain":
                continue
            deltas = []
            for plain, conditioned in zip(plain_records, records[variant]):
                deltas.append(
                    {
                        metric: conditioned[metric] - plain[metric]
                        for metric in METRIC_NAMES
                    }
                )
            comparisons[variant] = aggregate_metrics(deltas)

    result = {
        "subject": args.subject,
        "source_file": str(source_path),
        "shape": list(x.shape),
        "class_counts": class_counts.tolist(),
        "class_names": list(CLASS_NAMES),
        "protocol": {
            "split": (
                "stratified_holdout"
                if args.split_mode == "holdout"
                else "stratified_k_fold"
            ),
            "test_size": args.test_size if args.split_mode == "holdout" else None,
            "folds": args.folds if args.split_mode == "cv" else None,
            "seed": args.seed,
            "normalization": "StandardScaler fitted on training data only",
            "model_selection": "none; final epoch is evaluated",
            "device": str(device),
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "class_weighting": "balanced",
            "hidden_dim": args.hidden_dim,
            "d_state": args.d_state,
            "dt_rank": args.dt_rank,
            "d_conv": args.d_conv,
            "dropout": args.dropout,
            "condition_alpha": args.condition_alpha,
        },
        "aggregate": aggregates,
        "paired_difference_vs_plain": comparisons,
        "folds": records,
    }
    return result


def print_summary(result: dict) -> None:
    print(
        f"subject=s{result['subject']:02d} "
        f"shape={tuple(result['shape'])} "
        f"class_counts={result['class_counts']}"
    )
    print(
        f"{'variant':<12} {'accuracy':>16} {'balanced_acc':>16} "
        f"{'macro_f1':>16} {'log_loss':>16}"
    )
    holdout = result["protocol"]["split"] == "stratified_holdout"
    for variant, metrics in result["aggregate"].items():
        if holdout:
            values = [f"{metrics[name]['mean']:.4f}" for name in METRIC_NAMES]
        else:
            values = [
                f"{metrics[name]['mean']:.4f}±{metrics[name]['std']:.4f}"
                for name in METRIC_NAMES
            ]
        print(f"{variant:<12} {values[0]:>16} {values[1]:>16} {values[2]:>16} {values[3]:>16}")

    for variant, metrics in result["paired_difference_vs_plain"].items():
        accuracy = metrics["accuracy"]
        macro_f1 = metrics["macro_f1"]
        if holdout:
            print(
                f"{variant} - plain: "
                f"accuracy={accuracy['mean']:+.4f}, "
                f"macro_f1={macro_f1['mean']:+.4f}"
            )
        else:
            print(
                f"{variant} - plain: "
                f"accuracy={accuracy['mean']:+.4f}±{accuracy['std']:.4f}, "
                f"macro_f1={macro_f1['mean']:+.4f}±{macro_f1['std']:.4f}"
            )


def save_results(result: dict, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"s{result['subject']:02d}_simple_mamba"
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("variant", "metric", "mean", "std"),
        )
        writer.writeheader()
        for variant, metrics in result["aggregate"].items():
            for metric in METRIC_NAMES:
                writer.writerow(
                    {
                        "variant": variant,
                        "metric": metric,
                        "mean": metrics[metric]["mean"],
                        "std": metrics[metric]["std"],
                    }
                )
    return json_path, csv_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    result = run_experiment(args)
    print_summary(result)
    json_path, csv_path = save_results(result, args.output_dir)
    print(f"results_json={json_path.resolve()}")
    print(f"results_csv={csv_path.resolve()}")


if __name__ == "__main__":
    main()
