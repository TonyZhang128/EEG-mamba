"""Per-subject DEAP preprocessing and PyTorch dataset utilities.

Each ``sXX.mat`` file is processed independently and written directly to
``data/processed/overlap50/sXX.npz``. No combined cache is required.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
import torch
from torch.utils.data import Dataset


LOGGER = logging.getLogger(__name__)

EEGMAMBA_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = Path(r"D:\PythonProject\元路径\情绪data\extracted")
DEFAULT_OUTPUT_DIR = EEGMAMBA_ROOT / "data" / "processed" / "overlap50"
DEFAULT_FS = 128
DEFAULT_N_CHANNELS = 32
DEFAULT_BASELINE_SECONDS = 3.0
DEFAULT_TRIAL_SECONDS = 60.0
DEFAULT_WINDOW_SECONDS = 1.0
DEFAULT_OVERLAP = 0.5
DEFAULT_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 14.0),
    "beta": (14.0, 31.0),
    "gamma": (31.0, 50.0),
}
CLASS_NAMES = ("LVLA", "LVHA", "HVLA", "HVHA")
LABEL_SCHEME = "valence_arousal_median_4class"


@dataclass(frozen=True)
class SubjectResult:
    """Summary for one processed or reused subject file."""

    subject_id: int
    path: Path
    samples: int
    time_steps: int
    processed: bool


def parse_subject_ids(value: Optional[str]) -> Optional[list[int]]:
    """Parse values such as ``s01,s02,5-7`` into sorted subject IDs."""
    if value is None or not value.strip():
        return None

    subjects: set[int] = set()
    for item in value.replace(";", ",").split(","):
        token = item.strip().lower()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text.lstrip("s"))
            end = int(end_text.lstrip("s"))
            if end < start:
                raise ValueError(f"invalid subject range: {token}")
            subjects.update(range(start, end + 1))
        else:
            subjects.add(int(token.lstrip("s")))
    return sorted(subjects)


def _subject_number(path: Path) -> int:
    match = re.fullmatch(r"s(\d+)\.(?:mat|npz)", path.name, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"unexpected subject filename: {path.name}")
    return int(match.group(1))


def _subject_filename(subject_id: int) -> str:
    width = max(2, len(str(subject_id)))
    return f"s{subject_id:0{width}d}.npz"


def _window_layout(
    *,
    signal_samples: int,
    window_size: int,
    overlap: float,
) -> tuple[int, int]:
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    if signal_samples < window_size:
        raise ValueError("signal is shorter than one window")
    hop_size = round(window_size * (1.0 - overlap))
    if hop_size <= 0:
        raise ValueError("overlap leaves no positive hop size")
    num_windows = 1 + (signal_samples - window_size) // hop_size
    return hop_size, num_windows


def _build_metadata(
    *,
    subject_id: int,
    fs: int,
    n_channels: int,
    baseline_samples: int,
    signal_samples: int,
    window_size: int,
    hop_size: int,
    overlap: float,
    num_windows: int,
    bands: dict[str, tuple[float, float]],
) -> dict:
    return {
        "subject_id": subject_id,
        "fs": fs,
        "n_channels": n_channels,
        "baseline_samples": baseline_samples,
        "signal_samples": signal_samples,
        "window_size": window_size,
        "window_seconds": window_size / fs,
        "overlap": overlap,
        "hop_size": hop_size,
        "num_windows": num_windows,
        "bands": {name: list(limits) for name, limits in bands.items()},
        "class_names": list(CLASS_NAMES),
        "label_scheme": LABEL_SCHEME,
        "feature_shape": [n_channels, num_windows, len(bands)],
    }


def _labels_from_ratings(labels: np.ndarray) -> np.ndarray:
    if labels.ndim != 2 or labels.shape[1] < 2:
        raise ValueError("labels must have shape [trials, at least 2]")
    # Each subject is binarized independently by its own V/A medians.
    valence = labels[:, 0]
    arousal = labels[:, 1]
    valence_binary = (valence >= np.median(valence)).astype(np.int64)
    arousal_binary = (arousal >= np.median(arousal)).astype(np.int64)
    return valence_binary * 2 + arousal_binary


def process_subject(
    path: Path,
    *,
    fs: int = DEFAULT_FS,
    n_channels: int = DEFAULT_N_CHANNELS,
    baseline_seconds: float = DEFAULT_BASELINE_SECONDS,
    trial_seconds: float = DEFAULT_TRIAL_SECONDS,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    overlap: float = DEFAULT_OVERLAP,
    bands: Optional[dict[str, tuple[float, float]]] = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Convert one DEAP ``sXX.mat`` file to overlapping DE features."""
    bands = DEFAULT_BANDS if bands is None else bands
    path = Path(path)
    subject_id = _subject_number(path)
    baseline_samples = round(fs * baseline_seconds)
    signal_samples = round(fs * trial_seconds)
    window_size = round(fs * window_seconds)
    required_samples = baseline_samples + signal_samples
    hop_size, num_windows = _window_layout(
        signal_samples=signal_samples,
        window_size=window_size,
        overlap=overlap,
    )

    mat = loadmat(path)
    if "data" not in mat or "labels" not in mat:
        raise ValueError(f"{path.name} must contain 'data' and 'labels'")
    raw_eeg = mat["data"]
    ratings = np.asarray(mat["labels"], dtype=np.float64)
    del mat

    eeg = np.asarray(raw_eeg, dtype=np.float64)
    del raw_eeg
    if eeg.ndim != 3:
        raise ValueError(f"{path.name}: data must have shape [trials, channels, time]")
    if ratings.ndim != 2:
        raise ValueError(f"{path.name}: labels must have shape [trials, ratings]")
    if eeg.shape[0] != ratings.shape[0]:
        raise ValueError(f"{path.name}: trial counts do not match")
    if eeg.shape[1] < n_channels:
        raise ValueError(f"{path.name}: expected at least {n_channels} channels")
    if eeg.shape[2] < required_samples:
        raise ValueError(
            f"{path.name}: expected at least {required_samples} time samples, "
            f"got {eeg.shape[2]}"
        )

    n_trials = eeg.shape[0]
    eeg = eeg[:, :n_channels, :required_samples]
    baseline = eeg[:, :, :baseline_samples].mean(axis=2, keepdims=True)
    corrected = eeg[:, :, baseline_samples:] - baseline

    features = np.empty(
        (n_trials, n_channels, num_windows, len(bands)),
        dtype=np.float32,
    )
    nyquist = fs / 2.0
    for band_index, (name, (low, high)) in enumerate(bands.items()):
        if not 0.0 < low < high < nyquist:
            raise ValueError(
                f"invalid {name} band ({low}, {high}) for sampling rate {fs}"
            )
        coefficients = butter(
            4,
            (low / nyquist, high / nyquist),
            btype="band",
        )

        # Filter the full 60 s signal before extracting overlapping windows.
        filtered = filtfilt(*coefficients, corrected, axis=2)
        all_windows = sliding_window_view(filtered, window_size, axis=2)
        windows = all_windows[:, :, ::hop_size, :][:, :, :num_windows, :]
        if windows.shape[2] != num_windows:
            raise RuntimeError(
                f"expected {num_windows} windows, got {windows.shape[2]}"
            )

        power = np.mean(np.square(windows), axis=3)
        differential_entropy = 0.5 * np.log(
            2.0 * np.pi * np.e * power + 1e-12
        )
        features[..., band_index] = differential_entropy.astype(np.float32)

    metadata = _build_metadata(
        subject_id=subject_id,
        fs=fs,
        n_channels=n_channels,
        baseline_samples=baseline_samples,
        signal_samples=signal_samples,
        window_size=window_size,
        hop_size=hop_size,
        overlap=overlap,
        num_windows=num_windows,
        bands=bands,
    )
    return features, _labels_from_ratings(ratings), metadata


def _read_metadata(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        if "metadata" not in archive.files:
            raise ValueError(f"{path}: missing metadata")
        return json.loads(str(archive["metadata"].item()))


def _is_reusable_subject_file(path: Path, expected_metadata: dict) -> bool:
    if not path.exists():
        return False
    if _read_metadata(path) != expected_metadata:
        raise ValueError(
            f"metadata mismatch at {path}; rerun with --force to regenerate it"
        )
    with np.load(path, allow_pickle=False) as archive:
        x = archive["x"]
        y = archive["y"]
        subjects = archive["subjects"]
    expected_shape = (
        y.shape[0],
        expected_metadata["n_channels"],
        expected_metadata["num_windows"],
        len(expected_metadata["bands"]),
    )
    if x.shape != expected_shape or y.shape != (x.shape[0],):
        raise ValueError(f"invalid array shapes in {path}")
    if subjects.shape != (x.shape[0],) or not np.all(
        subjects == expected_metadata["subject_id"]
    ):
        raise ValueError(f"invalid subject IDs in {path}")
    return True


def _write_subject_npz(
    path: Path,
    x: np.ndarray,
    y: np.ndarray,
    subject_id: int,
    metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp.npz")
    try:
        np.savez_compressed(
            temporary_path,
            x=x,
            y=y,
            subjects=np.full(y.shape[0], subject_id, dtype=np.int64),
            metadata=np.array(json.dumps(metadata, ensure_ascii=False)),
        )
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _discover_mat_files(
    data_dir: Path,
    subject_ids: Optional[Sequence[int]],
) -> list[Path]:
    files = sorted(data_dir.glob("s*.mat"), key=_subject_number)
    if subject_ids is not None:
        selected = set(int(subject) for subject in subject_ids)
        files = [path for path in files if _subject_number(path) in selected]
    if not files:
        raise FileNotFoundError(
            f"no DEAP subject files found in {data_dir}"
            + (f" for subjects {sorted(subject_ids)}" if subject_ids else "")
        )
    return files


def prepare_subject_files(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    *,
    subject_ids: Optional[Sequence[int]] = None,
    force: bool = False,
    fs: int = DEFAULT_FS,
    n_channels: int = DEFAULT_N_CHANNELS,
    baseline_seconds: float = DEFAULT_BASELINE_SECONDS,
    trial_seconds: float = DEFAULT_TRIAL_SECONDS,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    overlap: float = DEFAULT_OVERLAP,
    bands: Optional[dict[str, tuple[float, float]]] = None,
) -> list[SubjectResult]:
    """Process every selected subject and write one reusable NPZ per subject."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    bands = DEFAULT_BANDS if bands is None else bands
    files = _discover_mat_files(data_dir, subject_ids)
    results: list[SubjectResult] = []

    for path in files:
        subject_id = _subject_number(path)
        output_path = output_dir / _subject_filename(subject_id)
        baseline_samples = round(fs * baseline_seconds)
        signal_samples = round(fs * trial_seconds)
        window_size = round(fs * window_seconds)
        hop_size, num_windows = _window_layout(
            signal_samples=signal_samples,
            window_size=window_size,
            overlap=overlap,
        )
        expected_metadata = _build_metadata(
            subject_id=subject_id,
            fs=fs,
            n_channels=n_channels,
            baseline_samples=baseline_samples,
            signal_samples=signal_samples,
            window_size=window_size,
            hop_size=hop_size,
            overlap=overlap,
            num_windows=num_windows,
            bands=bands,
        )

        if not force and _is_reusable_subject_file(
            output_path,
            expected_metadata,
        ):
            with np.load(output_path, allow_pickle=False) as archive:
                samples = int(archive["x"].shape[0])
            LOGGER.info(
                "reused s%02d samples=%d shape=(%d,%d,%d)",
                subject_id,
                samples,
                n_channels,
                num_windows,
                len(bands),
            )
            results.append(
                SubjectResult(
                    subject_id=subject_id,
                    path=output_path,
                    samples=samples,
                    time_steps=num_windows,
                    processed=False,
                )
            )
            continue

        LOGGER.info("processing %s", path.name)
        x, y, metadata = process_subject(
            path,
            fs=fs,
            n_channels=n_channels,
            baseline_seconds=baseline_seconds,
            trial_seconds=trial_seconds,
            window_seconds=window_seconds,
            overlap=overlap,
            bands=bands,
        )
        _write_subject_npz(output_path, x, y, subject_id, metadata)
        LOGGER.info(
            "wrote %s samples=%d shape=(%d,%d,%d)",
            output_path,
            x.shape[0],
            n_channels,
            num_windows,
            len(bands),
        )
        results.append(
            SubjectResult(
                subject_id=subject_id,
                path=output_path,
                samples=x.shape[0],
                time_steps=num_windows,
                processed=True,
            )
        )
    return results


def load_subject_dataset(
    subject_dir: Path | str = DEFAULT_OUTPUT_DIR,
    *,
    subject_ids: Optional[Sequence[int]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load and concatenate per-subject NPZ files in memory."""
    subject_dir = Path(subject_dir)
    files = sorted(subject_dir.glob("s*.npz"), key=_subject_number)
    if subject_ids is not None:
        selected = set(int(subject) for subject in subject_ids)
        files = [path for path in files if _subject_number(path) in selected]
    if not files:
        raise FileNotFoundError(f"no subject NPZ files found in {subject_dir}")

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    subject_parts: list[np.ndarray] = []
    common_metadata: Optional[dict] = None
    loaded_subjects: list[int] = []
    for path in files:
        with np.load(path, allow_pickle=False) as archive:
            x = np.asarray(archive["x"], dtype=np.float32)
            y = np.asarray(archive["y"], dtype=np.int64)
            subjects = np.asarray(archive["subjects"], dtype=np.int64)
            metadata = json.loads(str(archive["metadata"].item()))

        subject_id = _subject_number(path)
        if metadata.get("subject_id") != subject_id:
            raise ValueError(f"{path}: subject ID does not match its filename")
        common = dict(metadata)
        common.pop("subject_id", None)
        if common_metadata is None:
            common_metadata = common
        elif common != common_metadata:
            raise ValueError(f"{path}: preprocessing metadata differs from other files")
        if x.shape[0] != y.shape[0] or subjects.shape != (x.shape[0],):
            raise ValueError(f"{path}: inconsistent sample counts")
        if not np.all(subjects == subject_id):
            raise ValueError(f"{path}: contains unexpected subject IDs")

        x_parts.append(x)
        y_parts.append(y)
        subject_parts.append(subjects)
        loaded_subjects.append(subject_id)

    assert common_metadata is not None
    common_metadata["subject_ids"] = loaded_subjects
    return (
        np.concatenate(x_parts, axis=0),
        np.concatenate(y_parts, axis=0),
        np.concatenate(subject_parts, axis=0),
        common_metadata,
    )


def prepare_deap_dataset(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    subject_dir: Path | str = DEFAULT_OUTPUT_DIR,
    *,
    subject_ids: Optional[Sequence[int]] = None,
    force: bool = False,
    fs: int = DEFAULT_FS,
    n_channels: int = DEFAULT_N_CHANNELS,
    baseline_seconds: float = DEFAULT_BASELINE_SECONDS,
    trial_seconds: float = DEFAULT_TRIAL_SECONDS,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    overlap: float = DEFAULT_OVERLAP,
    bands: Optional[dict[str, tuple[float, float]]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Prepare missing per-subject files, then load them in memory."""
    prepare_subject_files(
        data_dir=data_dir,
        output_dir=subject_dir,
        subject_ids=subject_ids,
        force=force,
        fs=fs,
        n_channels=n_channels,
        baseline_seconds=baseline_seconds,
        trial_seconds=trial_seconds,
        window_seconds=window_seconds,
        overlap=overlap,
        bands=bands,
    )
    return load_subject_dataset(subject_dir, subject_ids=subject_ids)


class DEAPDataset(Dataset):
    """Trial-level DEAP dataset with tensors shaped ``[C, T, d]``."""

    def __init__(
        self,
        data_dir: Path | str = DEFAULT_DATA_DIR,
        subject_dir: Path | str = DEFAULT_OUTPUT_DIR,
        *,
        subject_ids: Optional[Sequence[int]] = None,
        force_reprocess: bool = False,
        overlap: float = DEFAULT_OVERLAP,
    ) -> None:
        x, y, subjects, metadata = prepare_deap_dataset(
            data_dir=data_dir,
            subject_dir=subject_dir,
            subject_ids=subject_ids,
            force=force_reprocess,
            overlap=overlap,
        )
        if x.size == 0:
            raise ValueError("dataset is empty")

        self.x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        self.y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64))
        self.subjects = torch.from_numpy(
            np.ascontiguousarray(subjects, dtype=np.int64)
        )
        self.metadata = metadata
        self.class_names = tuple(metadata["class_names"])

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]

    @property
    def num_channels(self) -> int:
        return self.x.shape[1]

    @property
    def sequence_length(self) -> int:
        return self.x.shape[2]

    @property
    def input_dim(self) -> int:
        return self.x.shape[3]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def unique_subject_ids(self) -> list[int]:
        return sorted(torch.unique(self.subjects).tolist())

    def indices_for_subjects(self, subject_ids: Iterable[int]) -> list[int]:
        selected = torch.tensor(
            sorted(set(int(subject) for subject in subject_ids)),
            dtype=self.subjects.dtype,
        )
        if selected.numel() == 0:
            return []
        return torch.nonzero(
            torch.isin(self.subjects, selected),
            as_tuple=False,
        ).flatten().tolist()


def split_subjects(
    subject_ids: Sequence[int],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Split subjects into disjoint training and validation groups."""
    unique = sorted(set(int(subject) for subject in subject_ids))
    if len(unique) < 2:
        raise ValueError("at least two subjects are required for a subject split")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be in (0, 1)")

    generator = np.random.default_rng(seed)
    shuffled = np.asarray(unique, dtype=np.int64)
    generator.shuffle(shuffled)
    n_val = max(1, round(len(unique) * val_ratio))
    n_val = min(n_val, len(unique) - 1)
    val_subjects = sorted(int(subject) for subject in shuffled[:n_val])
    train_subjects = sorted(int(subject) for subject in shuffled[n_val:])
    return train_subjects, val_subjects


def class_counts(dataset: Dataset) -> list[int]:
    if isinstance(dataset, DEAPDataset):
        targets = dataset.y
        num_classes = dataset.num_classes
    else:
        targets = torch.tensor([int(dataset[index][1]) for index in range(len(dataset))])
        num_classes = 4
    return torch.bincount(
        targets.to(torch.int64),
        minlength=num_classes,
    ).tolist()


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess DEAP .mat files directly into one overlapping-window "
            "NPZ file per subject."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--subjects", type=str, default=None)
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP)
    parser.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_argument_parser().parse_args()
    subject_ids = parse_subject_ids(args.subjects)
    results = prepare_subject_files(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        subject_ids=subject_ids,
        force=args.force,
        overlap=args.overlap,
        window_seconds=args.window_seconds,
    )
    processed = sum(result.processed for result in results)
    time_steps = sorted({result.time_steps for result in results})
    print(f"subjects={[result.subject_id for result in results]}")
    print(f"processed={processed} reused={len(results) - processed}")
    print(f"time_steps={time_steps}")
    print(f"output_dir={Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
