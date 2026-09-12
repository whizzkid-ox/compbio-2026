#!/usr/bin/env python3
"""Reproducible TRAIN-to-TEST decoding of the completed SHD SNN sweeps.

This script replaces the stateful analysis in ``Copia_di_decoding.ipynb``.  It
uses only the fixed TRAIN/TEST split, decodes each physical simulation once,
keeps network parameter sweeps descriptive, and performs paired inference on
the shared TEST trials.  Random-split and speaker-held-out analyses are outside
its scope.

The main representation is a 25 ms time-by-neuron spike-count matrix.  A true
count-only representation sums over time and keeps one total per neuron.  The
co-activation analysis uses a logical AND for pairs connected in the recurrent
baseline graph and compares them with an equally sized deterministic non-edge
set; all representations use the same 25 ms bins.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BIN_MS = 25.0
T_MAX_MS = 800.0
N_BINS = 32
DEFAULT_ALPHA = 1.0
DEFAULT_BOOTSTRAPS = 10_000
MODEL_NAMES = {"adex": "AdEx", "hh": "HH"}
REPRESENTATIONS = (
    "temporal_25ms",
    "per_neuron_total_count",
    "population_temporal_25ms",
)
BASELINE_EXTRA_REPRESENTATIONS = (
    "baseline_edge_coactivation_25ms",
    "matched_nonedge_coactivation_25ms",
    "temporal_plus_edge_coactivation_25ms",
)


AUDIT_FINDINGS = [
    ("Wrong count-only axis", "X.sum(axis=2) removes neuron identity but retains time. The corrected feature is X.sum(axis=1), one total count per neuron."),
    ("Protocol mismatch", "Several later cells use within-TRAIN cross-validation and the default logistic readout, while the headline analysis uses Ridge fitted on TRAIN and evaluated on TEST."),
    ("Invalid trials re-enter analyses", "Later notebook cells decode X and y without consistently applying the saved valid mask."),
    ("Pseudoreplication", "Network configurations and repeated baseline references are not independent samples; t-tests, Mann-Whitney tests, and Cohen's d across those accuracies are invalid."),
    ("Duplicated baselines", "The index exposes the same recurrent baseline in three sweep families. Treating all references as new datasets repeats identical predictions and inflates apparent n."),
    ("Unmatched raw-SHD comparison", "The notebook changes trial count, smoothing, split protocol, feature count, and sometimes classifier before comparing raw SHD with SNN output."),
    ("Ill-conditioned 5 ms calculation", "The 5 ms raw design is extremely wide and produced numerical warnings; it cannot support the reported conclusion without a stable, matched protocol."),
    ("Co-activation called synchrony", "Summing binary activity within arbitrary groups measures pooled activity. It is not spike-time synchrony, phase locking, cross-correlation, or coincidence probability."),
    ("Pair feature is not an interaction", "The notebook's pair value 0/1/2 is the number of active neurons and retains marginal activity. True co-activation is the logical AND of the two activity indicators."),
    ("Arbitrary neuron pairs", "Pairs (0,1), (2,3), ... are based only on adjacent IDs. They are not recurrently connected, spatially local, or selected independently of the analysis."),
    ("Mixed time resolutions", "The count-plus-group analysis concatenates 25 ms count features with 5 ms pooled-activity features, changing timing resolution and representation simultaneously."),
    ("Mislabelled global feature", "Groups 0-49 and 50-99 are coarse population activity; the second group mixes excitatory and inhibitory neurons and is not a global synchrony measure."),
    ("TEST-set selection", "Calling p=0.20 or p=0.30 'best' after inspecting TEST accuracy leaks TEST information into model selection. Sweeps should remain sensitivity analyses unless selection uses TRAIN only."),
    ("Missing identity assertions", "Rows are assumed to align across configurations without checking trial IDs, labels, speakers, valid masks, offsets, or archive/CSV agreement."),
    ("Hard-coded and stateful outputs", "Plots retype accuracy values, and at least one cell depends on stale dataset_dir/condition variables. Results can silently diverge from the calculations."),
    ("155 ms bin truncation", "int(800/155) creates five bins covering only 775 ms, so the purported count control silently drops the final 25 ms."),
]


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    model: str
    condition: str
    config_id: str
    parameter_name: str
    parameter_value: float | None
    rec_probability: float
    rec_gain: float
    ei_ratio: float
    input_g: float
    train_dir: str
    test_dir: str


@dataclass
class BinnedDataset:
    X: np.ndarray
    y: np.ndarray
    trial_ids: np.ndarray
    speaker_ids: np.ndarray
    recurrent_connectivity: np.ndarray


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def win_long_path(path: Path) -> Path:
    """Return an absolute path that also works beyond Win32 MAX_PATH."""
    resolved = path.expanduser().resolve()
    text = str(resolved)
    if os.name == "nt" and not text.startswith("\\\\?\\"):
        return Path("\\\\?\\" + text)
    return resolved


def display_path(path: Path) -> str:
    text = str(path)
    return text[4:] if text.startswith("\\\\?\\") else text


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def atomic_json(path: Path, value) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    os.replace(temporary, path)


def one_match(root: Path, pattern: str) -> Path:
    matches = list((root / "sweeps").glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {pattern!r} below {root}, found {len(matches)}")
    return matches[0]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def discover_datasets(train_root: Path, test_root: Path) -> list[DatasetSpec]:
    specs: list[DatasetSpec] = []
    for model_slug, model_name in MODEL_NAMES.items():
        pattern = f"*_{model_name}_parameter_sweep_*"
        train_manifest_dir = one_match(train_root, pattern)
        test_manifest_dir = one_match(test_root, pattern)
        train_manifest = load_json(train_manifest_dir / "sweep_manifest.json")
        test_manifest = load_json(test_manifest_dir / "sweep_manifest.json")
        if train_manifest["model"] != model_name or test_manifest["model"] != model_name:
            raise ValueError(f"Model mismatch in {model_name} manifests")
        train_configs = {row["config_id"]: row for row in train_manifest["configurations"]}
        test_configs = {row["config_id"]: row for row in test_manifest["configurations"]}
        if train_configs.keys() != test_configs.keys() or len(train_configs) != 11:
            raise ValueError(f"TRAIN/TEST configuration mismatch for {model_name}")

        base = train_configs["baseline"]
        specs.append(DatasetSpec(
            dataset_id=f"{model_slug}_ff", model=model_name, condition="FF",
            config_id="baseline_FF", parameter_name="baseline_ff",
            parameter_value=None, rec_probability=float(base["rec_probability"]),
            rec_gain=float(base["rec_gain"]), ei_ratio=float(base["EI_ratio"]),
            input_g=float(base["input_g"]),
            train_dir=str(train_manifest_dir / "baseline_FF"),
            test_dir=str(test_manifest_dir / "baseline_FF"),
        ))
        for config_id, train_row in train_configs.items():
            test_row = test_configs[config_id]
            keys = ("parameter_name", "parameter_value", "rec_probability", "rec_gain", "EI_ratio", "input_g")
            if any(train_row[key] != test_row[key] for key in keys):
                raise ValueError(f"TRAIN/TEST parameters differ for {model_name} {config_id}")
            dataset_id = f"{model_slug}_rec" if config_id == "baseline" else f"{model_slug}_{config_id.lower()}"
            specs.append(DatasetSpec(
                dataset_id=dataset_id, model=model_name, condition="REC",
                config_id=config_id, parameter_name=str(train_row["parameter_name"]),
                parameter_value=train_row["parameter_value"],
                rec_probability=float(train_row["rec_probability"]),
                rec_gain=float(train_row["rec_gain"]), ei_ratio=float(train_row["EI_ratio"]),
                input_g=float(train_row["input_g"]),
                train_dir=str(train_manifest_dir / train_row["output_directory"]),
                test_dir=str(test_manifest_dir / test_row["output_directory"]),
            ))
    if len(specs) != 24 or len({spec.dataset_id for spec in specs}) != 24:
        raise AssertionError("Expected 24 unique physical SNN datasets")
    return specs


def bool_column(values: pd.Series) -> np.ndarray:
    if values.dtype == bool:
        return values.to_numpy(dtype=bool)
    normalised = values.astype(str).str.strip().str.lower()
    if not normalised.isin(("true", "false", "1", "0")).all():
        raise ValueError("valid column contains non-Boolean values")
    return normalised.isin(("true", "1")).to_numpy(dtype=bool)


def archive_identity(dataset_dir: Path, condition: str) -> tuple[np.ndarray, ...]:
    if not (dataset_dir / "COMPLETE.txt").is_file():
        raise ValueError(f"Dataset is not marked complete: {dataset_dir}")
    metadata = load_json(dataset_dir / "metadata.json")
    if metadata.get("status") != "complete":
        raise ValueError(f"Dataset metadata is not complete: {dataset_dir}")
    trials = pd.read_csv(dataset_dir / "trials.csv")
    required = {"trial_id", "label", "speaker_id", "valid"}
    if not required.issubset(trials.columns):
        raise ValueError(f"Missing trial columns in {dataset_dir}: {required - set(trials.columns)}")
    with np.load(dataset_dir / "events.npz", allow_pickle=False) as events:
        trial_ids = np.asarray(events["trial_ids"], dtype=np.int64)
        labels = np.asarray(events["labels"], dtype=np.int64)
        speakers = np.asarray(events["speaker_ids"], dtype=np.int64)
        valid = np.asarray(events[f"{condition}_valid"], dtype=bool)
        offsets = np.asarray(events[f"{condition}_offsets"], dtype=np.int64)
        n_events = len(events[f"{condition}_times_ms"])
    csv_values = (
        trials["trial_id"].to_numpy(dtype=np.int64),
        trials["label"].to_numpy(dtype=np.int64),
        trials["speaker_id"].to_numpy(dtype=np.int64),
        bool_column(trials["valid"]),
    )
    for archive, table, name in zip((trial_ids, labels, speakers, valid), csv_values,
                                    ("trial_id", "label", "speaker_id", "valid")):
        if not np.array_equal(archive, table):
            raise ValueError(f"Archive/CSV {name} mismatch in {dataset_dir}")
    if offsets.shape != (len(trials) + 1,) or offsets[0] != 0 or offsets[-1] != n_events:
        raise ValueError(f"Invalid event offsets in {dataset_dir}")
    if np.any(np.diff(offsets) < 0):
        raise ValueError(f"Non-monotonic event offsets in {dataset_dir}")
    return trial_ids, labels, speakers, valid


def validate_catalogue(specs: list[DatasetSpec]) -> dict:
    reference: dict[str, tuple[np.ndarray, ...]] = {}
    summary: dict[str, dict] = {}
    for spec in specs:
        for split, directory in (("train", Path(spec.train_dir)), ("test", Path(spec.test_dir))):
            identity = archive_identity(directory, spec.condition)
            if split not in reference:
                reference[split] = identity
            else:
                for left, right, name in zip(reference[split], identity,
                                             ("trial_ids", "labels", "speaker_ids", "valid")):
                    if not np.array_equal(left, right):
                        raise ValueError(f"{split} {name} is not aligned for {spec.dataset_id}")
            summary[f"{spec.dataset_id}_{split}"] = {
                "directory": display_path(directory),
                "trials": int(len(identity[0])),
                "valid_trials": int(identity[3].sum()),
            }
    return summary


def balanced_prefix(y: np.ndarray, per_class: int) -> np.ndarray:
    if per_class <= 0:
        return np.arange(len(y))
    selected = []
    for label in np.unique(y):
        members = np.flatnonzero(y == label)
        if len(members) < per_class:
            raise ValueError(f"Only {len(members)} trials available for label {label}")
        selected.extend(members[:per_class])
    return np.asarray(sorted(selected), dtype=np.int64)


def load_snn(dataset_dir: Path, condition: str, smoke_per_class: int = 0) -> BinnedDataset:
    trials = pd.read_csv(dataset_dir / "trials.csv")
    valid = bool_column(trials["valid"])
    with np.load(dataset_dir / "events.npz", allow_pickle=False) as events:
        times = np.asarray(events[f"{condition}_times_ms"], dtype=np.float64)
        neuron_ids = np.asarray(events[f"{condition}_neuron_ids"], dtype=np.int64)
        offsets = np.asarray(events[f"{condition}_offsets"], dtype=np.int64)
        archive_valid = np.asarray(events[f"{condition}_valid"], dtype=bool)
        event_trial_ids = np.asarray(events["trial_ids"], dtype=np.int64)
        event_labels = np.asarray(events["labels"], dtype=np.int64)
        event_speakers = np.asarray(events["speaker_ids"], dtype=np.int64)
        roster = np.asarray(events["neuron_ids"], dtype=np.int64)
        connectivity = np.asarray(events["recurrent_connectivity"], dtype=bool)
    if not np.array_equal(valid, archive_valid):
        raise ValueError(f"valid mask mismatch in {dataset_dir}")
    if not np.array_equal(roster, np.arange(len(roster))):
        raise ValueError(f"Neuron roster is not contiguous in {dataset_dir}")
    if connectivity.shape != (len(roster), len(roster)):
        raise ValueError(f"Invalid recurrent connectivity shape in {dataset_dir}")
    if len(times) != len(neuron_ids) or offsets[-1] != len(times):
        raise ValueError(f"Event-array length mismatch in {dataset_dir}")
    if not np.isfinite(times).all():
        raise ValueError(f"Non-finite event time in {dataset_dir}")
    if np.any((neuron_ids < 0) | (neuron_ids >= len(roster))):
        raise ValueError(f"Out-of-range neuron ID in {dataset_dir}")

    valid_rows = np.flatnonzero(valid)
    choose = balanced_prefix(event_labels[valid_rows], smoke_per_class)
    source_rows = valid_rows[choose]
    X = np.zeros((len(source_rows), N_BINS, len(roster)), dtype=np.float32)
    for output_row, source_row in enumerate(source_rows):
        start, end = int(offsets[source_row]), int(offsets[source_row + 1])
        trial_times = times[start:end]
        trial_neurons = neuron_ids[start:end]
        inside = (trial_times >= 0.0) & (trial_times < T_MAX_MS)
        bins = np.floor(trial_times[inside] / BIN_MS).astype(np.int64)
        np.add.at(X[output_row], (bins, trial_neurons[inside]), 1.0)
    return BinnedDataset(
        X=X,
        y=event_labels[source_rows],
        trial_ids=event_trial_ids[source_rows],
        speaker_ids=event_speakers[source_rows],
        recurrent_connectivity=connectivity,
    )


def load_raw_shd(h5_path: Path, smoke_per_class: int = 0) -> BinnedDataset:
    import tables

    with tables.open_file(str(h5_path), mode="r") as handle:
        all_labels = np.asarray(handle.root.labels[:], dtype=np.int64)
        try:
            all_speakers = np.asarray(handle.root.extra.speaker[:], dtype=np.int64)
        except tables.NoSuchNodeError:
            all_speakers = np.full(len(all_labels), -1, dtype=np.int64)
        english_ids = np.flatnonzero(all_labels < 10)
        choose = balanced_prefix(all_labels[english_ids], smoke_per_class)
        trial_ids = english_ids[choose]
        X = np.zeros((len(trial_ids), N_BINS, 700), dtype=np.float32)
        for output_row, trial_id in enumerate(trial_ids):
            seconds = np.asarray(handle.root.spikes.times[int(trial_id)], dtype=np.float64)
            units = np.asarray(handle.root.spikes.units[int(trial_id)], dtype=np.int64)
            if len(seconds) != len(units) or not np.isfinite(seconds).all():
                raise ValueError(f"Invalid raw SHD trial {trial_id}")
            times_ms = seconds * 1000.0
            inside = (times_ms >= 0.0) & (times_ms < T_MAX_MS) & (units >= 0) & (units < 700)
            bins = np.floor(times_ms[inside] / BIN_MS).astype(np.int64)
            np.add.at(X[output_row], (bins, units[inside]), 1.0)
    return BinnedDataset(
        X=X, y=all_labels[trial_ids], trial_ids=trial_ids,
        speaker_ids=all_speakers[trial_ids],
        recurrent_connectivity=np.zeros((700, 700), dtype=bool),
    )


def make_estimator(alpha: float):
    # LSQR avoids the ill-conditioned normal-equation solve seen in the notebook.
    return make_pipeline(
        StandardScaler(with_mean=True),
        RidgeClassifier(alpha=alpha, solver="lsqr", tol=1e-6),
    )


def score_representation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
) -> tuple[dict, np.ndarray]:
    if X_train.ndim != 2 or X_test.ndim != 2 or X_train.shape[1] != X_test.shape[1]:
        raise ValueError("TRAIN/TEST feature matrices are incompatible")
    estimator = make_estimator(alpha)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        estimator.fit(X_train, y_train)
    prediction = estimator.predict(X_test).astype(np.int64, copy=False)
    result = {
        "accuracy": float(accuracy_score(y_test, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
        "macro_f1": float(f1_score(y_test, prediction, average="macro")),
        "chance": float(np.bincount(y_test).max() / len(y_test)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_features": int(X_train.shape[1]),
    }
    return result, prediction


def pair_sets(connectivity: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    graph = np.logical_or(connectivity, connectivity.T)
    np.fill_diagonal(graph, False)
    upper = np.triu_indices_from(graph, k=1)
    edges = np.column_stack((upper[0][graph[upper]], upper[1][graph[upper]])).astype(np.int64)
    nonedges = np.column_stack((upper[0][~graph[upper]], upper[1][~graph[upper]])).astype(np.int64)
    if len(edges) == 0 or len(nonedges) < len(edges):
        raise ValueError("Cannot construct connected and matched non-edge pair sets")
    rng = np.random.default_rng(seed)
    matched = nonedges[np.sort(rng.choice(len(nonedges), size=len(edges), replace=False))]
    return edges, matched


def coactivation(X: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    active = X > 0
    values = np.logical_and(active[:, :, pairs[:, 0]], active[:, :, pairs[:, 1]])
    return values.reshape(len(X), -1).astype(np.float32, copy=False)


def base_representations(X: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "temporal_25ms": X.reshape(len(X), -1),
        "per_neuron_total_count": X.sum(axis=1, dtype=np.float32),
        "population_temporal_25ms": X.sum(axis=2, dtype=np.float32),
    }


def partial_paths(output_dir: Path, dataset_id: str) -> tuple[Path, Path]:
    partial = output_dir / "partials"
    return partial / f"{dataset_id}.json", partial / f"{dataset_id}.npz"


def analyse_snn_dataset(
    spec: DatasetSpec,
    output_dir: str,
    alpha: float,
    pair_seed: int,
    smoke_per_class: int,
) -> str:
    output = Path(output_dir)
    metadata_path, predictions_path = partial_paths(output, spec.dataset_id)
    if metadata_path.is_file() and predictions_path.is_file():
        saved = load_json(metadata_path)
        if saved.get("fingerprint") == dataset_fingerprint(spec, alpha, pair_seed, smoke_per_class):
            return spec.dataset_id

    train = load_snn(Path(spec.train_dir), spec.condition, smoke_per_class)
    test = load_snn(Path(spec.test_dir), spec.condition, smoke_per_class)
    if train.X.shape[1:] != test.X.shape[1:]:
        raise ValueError(f"TRAIN/TEST geometry mismatch for {spec.dataset_id}")
    representations_train = base_representations(train.X)
    representations_test = base_representations(test.X)
    pair_counts = {"connected": None, "matched_nonedge": None}
    if spec.config_id in ("baseline_FF", "baseline"):
        if not np.array_equal(train.recurrent_connectivity, test.recurrent_connectivity):
            raise ValueError(f"TRAIN/TEST connectivity differs for {spec.dataset_id}")
        edges, nonedges = pair_sets(train.recurrent_connectivity, pair_seed)
        edge_train = coactivation(train.X, edges)
        edge_test = coactivation(test.X, edges)
        representations_train["baseline_edge_coactivation_25ms"] = edge_train
        representations_test["baseline_edge_coactivation_25ms"] = edge_test
        representations_train["matched_nonedge_coactivation_25ms"] = coactivation(train.X, nonedges)
        representations_test["matched_nonedge_coactivation_25ms"] = coactivation(test.X, nonedges)
        representations_train["temporal_plus_edge_coactivation_25ms"] = np.concatenate(
            (representations_train["temporal_25ms"], edge_train), axis=1
        )
        representations_test["temporal_plus_edge_coactivation_25ms"] = np.concatenate(
            (representations_test["temporal_25ms"], edge_test), axis=1
        )
        pair_counts = {"connected": int(len(edges)), "matched_nonedge": int(len(nonedges))}

    summaries = []
    arrays = {
        "trial_ids": test.trial_ids,
        "speaker_ids": test.speaker_ids,
        "y_true": test.y,
    }
    for representation, X_train in representations_train.items():
        result, prediction = score_representation(
            X_train, train.y, representations_test[representation], test.y, alpha
        )
        summaries.append({
            **{key: value for key, value in asdict(spec).items() if key not in ("train_dir", "test_dir")},
            "representation": representation,
            **result,
        })
        arrays[f"pred__{representation}"] = prediction

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = predictions_path.with_name(predictions_path.name + f".tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, predictions_path)
    atomic_json(metadata_path, {
        "fingerprint": dataset_fingerprint(spec, alpha, pair_seed, smoke_per_class),
        "dataset": asdict(spec),
        "summaries": summaries,
        "pair_counts": pair_counts,
        "created_utc": utc_now(),
    })
    return spec.dataset_id


def dataset_fingerprint(spec: DatasetSpec, alpha: float, pair_seed: int, smoke_per_class: int) -> str:
    payload = {
        "script_sha256": sha256(Path(__file__)), "dataset": asdict(spec),
        "alpha": alpha, "pair_seed": pair_seed, "smoke_per_class": smoke_per_class,
        "bin_ms": BIN_MS, "t_max_ms": T_MAX_MS, "schema": 1,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def analyse_raw(
    train_h5: Path,
    test_h5: Path,
    output_dir: Path,
    alpha: float,
    smoke_per_class: int,
    expected_train: tuple[np.ndarray, np.ndarray],
    expected_test: tuple[np.ndarray, np.ndarray],
) -> None:
    metadata_path, predictions_path = partial_paths(output_dir, "raw_shd")
    fingerprint = hashlib.sha256(json.dumps({
        "script": sha256(Path(__file__)), "train": str(train_h5), "test": str(test_h5),
        "alpha": alpha, "smoke": smoke_per_class, "bin": BIN_MS, "tmax": T_MAX_MS,
    }, sort_keys=True).encode()).hexdigest()
    if metadata_path.is_file() and predictions_path.is_file() and load_json(metadata_path).get("fingerprint") == fingerprint:
        return
    train = load_raw_shd(train_h5, smoke_per_class)
    test = load_raw_shd(test_h5, smoke_per_class)
    for actual, expected, name in (
        (train.trial_ids, expected_train[0], "TRAIN trial IDs"),
        (train.y, expected_train[1], "TRAIN labels"),
        (test.trial_ids, expected_test[0], "TEST trial IDs"),
        (test.y, expected_test[1], "TEST labels"),
    ):
        if not np.array_equal(actual, expected):
            raise ValueError(f"Raw SHD and SNN {name} do not align")
    summaries = []
    arrays = {"trial_ids": test.trial_ids, "speaker_ids": test.speaker_ids, "y_true": test.y}
    train_reps, test_reps = base_representations(train.X), base_representations(test.X)
    for representation in REPRESENTATIONS:
        result, prediction = score_representation(train_reps[representation], train.y,
                                                  test_reps[representation], test.y, alpha)
        summaries.append({
            "dataset_id": "raw_shd", "model": "Raw SHD", "condition": "INPUT",
            "config_id": "raw_shd", "parameter_name": "none", "parameter_value": None,
            "rec_probability": None, "rec_gain": None, "ei_ratio": None, "input_g": None,
            "representation": representation, **result,
        })
        arrays[f"pred__{representation}"] = prediction
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = predictions_path.with_name(predictions_path.name + f".tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, predictions_path)
    atomic_json(metadata_path, {
        "fingerprint": fingerprint, "summaries": summaries, "created_utc": utc_now(),
        "train_h5": display_path(train_h5), "test_h5": display_path(test_h5),
    })


def load_partial_results(output_dir: Path, ids: list[str]) -> tuple[list[dict], list[dict]]:
    summaries: list[dict] = []
    predictions: list[dict] = []
    for dataset_id in ids:
        metadata_path, predictions_path = partial_paths(output_dir, dataset_id)
        metadata = load_json(metadata_path)
        summaries.extend(metadata["summaries"])
        with np.load(predictions_path, allow_pickle=False) as arrays:
            base = {
                "trial_id": np.asarray(arrays["trial_ids"], dtype=np.int64),
                "speaker_id": np.asarray(arrays["speaker_ids"], dtype=np.int64),
                "y_true": np.asarray(arrays["y_true"], dtype=np.int64),
            }
            for key in arrays.files:
                if not key.startswith("pred__"):
                    continue
                representation = key.removeprefix("pred__")
                prediction = np.asarray(arrays[key], dtype=np.int64)
                for i in range(len(prediction)):
                    predictions.append({
                        "dataset_id": dataset_id, "representation": representation,
                        "trial_id": int(base["trial_id"][i]),
                        "speaker_id": int(base["speaker_id"][i]),
                        "y_true": int(base["y_true"][i]),
                        "y_pred": int(prediction[i]),
                        "correct": bool(prediction[i] == base["y_true"][i]),
                    })
    return summaries, predictions


def paired_bootstrap_ci(
    left_correct: np.ndarray,
    right_correct: np.ndarray,
    labels: np.ndarray,
    repeats: int,
    seed: int,
) -> tuple[float, float]:
    difference = right_correct.astype(np.int8) - left_correct.astype(np.int8)
    rng = np.random.default_rng(seed)
    sums = np.zeros(repeats, dtype=np.float64)
    for label in np.unique(labels):
        values = difference[labels == label]
        draws = rng.integers(0, len(values), size=(repeats, len(values)))
        sums += values[draws].sum(axis=1)
    distribution = sums / len(labels)
    low, high = np.quantile(distribution, (0.025, 0.975))
    return float(low), float(high)


def paired_comparison(
    predictions: pd.DataFrame,
    left_id: str,
    right_id: str,
    comparison_id: str,
    family: str,
    bootstraps: int,
    seed: int,
) -> dict:
    columns = ["trial_id", "y_true", "correct"]
    left = predictions[(predictions.dataset_id == left_id) &
                       (predictions.representation == "temporal_25ms")][columns]
    right = predictions[(predictions.dataset_id == right_id) &
                        (predictions.representation == "temporal_25ms")][columns]
    paired = left.merge(right, on="trial_id", suffixes=("_left", "_right"), validate="one_to_one")
    if len(paired) != len(left) or len(paired) != len(right):
        raise ValueError(f"Unmatched TEST trials for {comparison_id}")
    if not np.array_equal(paired.y_true_left, paired.y_true_right):
        raise ValueError(f"Label mismatch for {comparison_id}")
    left_correct = paired.correct_left.to_numpy(dtype=bool)
    right_correct = paired.correct_right.to_numpy(dtype=bool)
    right_wins = int(np.sum(~left_correct & right_correct))
    left_wins = int(np.sum(left_correct & ~right_correct))
    discordant = right_wins + left_wins
    pvalue = 1.0 if discordant == 0 else float(
        binomtest(min(right_wins, left_wins), discordant, 0.5, alternative="two-sided").pvalue
    )
    derived_seed = int(hashlib.sha256(comparison_id.encode()).hexdigest()[:8], 16) ^ seed
    ci_low, ci_high = paired_bootstrap_ci(
        left_correct, right_correct, paired.y_true_left.to_numpy(), bootstraps, derived_seed
    )
    return {
        "comparison_id": comparison_id, "family": family,
        "left_dataset": left_id, "right_dataset": right_id,
        "n_test": int(len(paired)),
        "left_accuracy": float(left_correct.mean()),
        "right_accuracy": float(right_correct.mean()),
        "delta_accuracy_right_minus_left": float(right_correct.mean() - left_correct.mean()),
        "bootstrap_95_ci_low": ci_low, "bootstrap_95_ci_high": ci_high,
        "right_only_correct": right_wins, "left_only_correct": left_wins,
        "mcnemar_exact_p": pvalue,
    }


def holm_adjust(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=np.float64)
    running = 0.0
    n = len(pvalues)
    for rank, index in enumerate(order):
        running = max(running, (n - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def make_comparisons(specs: list[DatasetSpec], predictions: pd.DataFrame,
                     bootstraps: int, seed: int) -> list[dict]:
    rows = [
        paired_comparison(predictions, "adex_ff", "adex_rec", "AdEx_REC_minus_FF",
                          "baseline_architecture", bootstraps, seed),
        paired_comparison(predictions, "hh_ff", "hh_rec", "HH_REC_minus_FF",
                          "baseline_architecture", bootstraps, seed),
        paired_comparison(predictions, "hh_ff", "adex_ff", "AdEx_minus_HH_FF",
                          "baseline_model", bootstraps, seed),
        paired_comparison(predictions, "hh_rec", "adex_rec", "AdEx_minus_HH_REC",
                          "baseline_model", bootstraps, seed),
    ]
    sweep_rows = []
    for spec in specs:
        if spec.condition != "REC" or spec.config_id == "baseline":
            continue
        baseline = "adex_rec" if spec.model == "AdEx" else "hh_rec"
        sweep_rows.append(paired_comparison(
            predictions, baseline, spec.dataset_id,
            f"{spec.dataset_id}_minus_{baseline}", "sweep_vs_recurrent_baseline",
            bootstraps, seed,
        ))
    adjusted = holm_adjust([row["mcnemar_exact_p"] for row in sweep_rows])
    for row, value in zip(sweep_rows, adjusted):
        row["mcnemar_holm_p_across_20_sweeps"] = value
    for row in rows:
        row["mcnemar_holm_p_across_20_sweeps"] = None
    return rows + sweep_rows


def make_sweep_table(specs: list[DatasetSpec], results: pd.DataFrame) -> list[dict]:
    temporal = results[(results.representation == "temporal_25ms") &
                       (results.dataset_id != "raw_shd")].set_index("dataset_id")
    output = []
    baselines = {spec.model: spec for spec in specs if spec.config_id == "baseline"}
    baseline_values = {"rec_probability": 0.1, "rec_gain": 1.0, "EI_ratio": 4.0}
    for model in MODEL_NAMES.values():
        baseline = baselines[model]
        for family, baseline_value in baseline_values.items():
            result = temporal.loc[baseline.dataset_id]
            output.append({
                "model": model, "parameter_family": family,
                "parameter_value": baseline_value, "dataset_id": baseline.dataset_id,
                "is_shared_baseline_reference": True,
                "accuracy": result.accuracy, "balanced_accuracy": result.balanced_accuracy,
                "macro_f1": result.macro_f1,
            })
            for spec in specs:
                if spec.model != model or spec.parameter_name != family or spec.config_id == "baseline":
                    continue
                result = temporal.loc[spec.dataset_id]
                output.append({
                    "model": model, "parameter_family": family,
                    "parameter_value": spec.parameter_value, "dataset_id": spec.dataset_id,
                    "is_shared_baseline_reference": False,
                    "accuracy": result.accuracy, "balanced_accuracy": result.balanced_accuracy,
                    "macro_f1": result.macro_f1,
                })
    return sorted(output, key=lambda row: (row["model"], row["parameter_family"], row["parameter_value"]))


def make_figures(output_dir: Path, results: pd.DataFrame, sweeps: pd.DataFrame,
                 comparisons: pd.DataFrame) -> None:
    figures = output_dir / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"figure.dpi": 140, "axes.spines.top": False, "axes.spines.right": False})

    ids = ["adex_ff", "adex_rec", "hh_ff", "hh_rec"]
    labels = ["AdEx FF", "AdEx REC", "HH FF", "HH REC"]
    baseline = results[(results.dataset_id.isin(ids)) & (results.representation == "temporal_25ms")].set_index("dataset_id")
    fig, ax = plt.subplots(figsize=(7, 4))
    values = [baseline.loc[item, "accuracy"] for item in ids]
    ax.bar(labels, values, color=["#4C78A8", "#72B7B2", "#F58518", "#ECA82C"])
    ax.axhline(0.1, color="0.35", linestyle="--", linewidth=1, label="Nominal 10-class chance")
    ax.set(ylabel="TEST accuracy", ylim=(0, 1), title="25 ms temporal Ridge decoding")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(frameon=False)
    for i, value in enumerate(values):
        ax.text(i, value + 0.02, f"{value:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(figures / "baseline_accuracy.png")
    plt.close(fig)

    families = [("rec_probability", "Recurrent probability"), ("rec_gain", "Recurrent gain"), ("EI_ratio", "E/I ratio")]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
    for row_index, model in enumerate(("AdEx", "HH")):
        for column_index, (family, title) in enumerate(families):
            ax = axes[row_index, column_index]
            frame = sweeps[(sweeps.model == model) & (sweeps.parameter_family == family)].sort_values("parameter_value")
            ax.plot(frame.parameter_value, frame.accuracy, marker="o")
            ax.set(title=f"{model}: {title}", xlabel=title)
            if column_index == 0:
                ax.set_ylabel("TEST accuracy")
            ax.grid(alpha=0.25)
    fig.suptitle("Network sweeps are reported as sensitivity curves; no TEST-selected optimum")
    fig.tight_layout()
    fig.savefig(figures / "parameter_sweep_accuracy.png")
    plt.close(fig)

    representations = list(REPRESENTATIONS + BASELINE_EXTRA_REPRESENTATIONS)
    rep_labels = ["Temporal", "Neuron totals", "Population temporal", "Edge co-activation",
                  "Matched non-edge", "Temporal + edge"]
    frame = results[results.dataset_id.isin(ids) & results.representation.isin(representations)]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(ids))
    width = 0.13
    for i, (representation, label) in enumerate(zip(representations, rep_labels)):
        indexed = frame[frame.representation == representation].set_index("dataset_id")
        ax.bar(x + (i - 2.5) * width, [indexed.loc[item, "accuracy"] for item in ids], width, label=label)
    ax.set_xticks(x, labels)
    ax.set(ylabel="TEST accuracy", ylim=(0, 1), title="Matched 25 ms baseline representations")
    ax.legend(ncol=3, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "baseline_representations.png")
    plt.close(fig)

    if "raw_shd" in set(results.dataset_id):
        temporal = results[results.representation == "temporal_25ms"].set_index("dataset_id")
        compare_ids = ["raw_shd", *ids]
        fig, ax = plt.subplots(figsize=(8, 4))
        values = [temporal.loc[item, "accuracy"] for item in compare_ids]
        ax.bar(["Raw SHD", *labels], values, color=["0.45", "#4C78A8", "#72B7B2", "#F58518", "#ECA82C"])
        ax.set(ylabel="TEST accuracy", ylim=(0, 1), title="Matched 25 ms/no-smoothing/Ridge protocol")
        ax.tick_params(axis="x", rotation=20)
        for i, value in enumerate(values):
            ax.text(i, value + 0.02, f"{value:.3f}", ha="center", fontsize=9)
        fig.tight_layout()
        fig.savefig(figures / "raw_shd_vs_snn.png")
        plt.close(fig)


def make_report(output_dir: Path, results: pd.DataFrame, comparisons: pd.DataFrame,
                raw_included: bool, pair_counts: dict, alpha: float) -> None:
    temporal = results[results.representation == "temporal_25ms"].set_index("dataset_id")
    lines = [
        "# Corrected SHD SNN decoding analysis", "",
        "## Protocol", "",
        f"- Fixed TRAIN fit and TEST evaluation; {BIN_MS:g} ms bins over [0, {T_MAX_MS:g}) ms.",
        f"- `StandardScaler` plus `RidgeClassifier(alpha={alpha:g}, solver='lsqr')` for every representation.",
        "- All valid English trials are used. Random-split and speaker-held-out analyses are omitted.",
        "- Each of the 24 physical SNN datasets is decoded once; repeated sweep-baseline references reuse its result.",
        "- Parameter sweeps are sensitivity curves. No configuration is selected by TEST accuracy.",
        "- Accuracy differences use the same TEST trials, paired stratified bootstrap intervals, and exact McNemar tests.",
    ]
    if raw_included:
        lines.append("- Raw SHD uses the same trials, window, bin width, no smoothing, scaler, classifier, and TRAIN/TEST protocol; its 700 input channels still differ from the 100-neuron SNN output.")
    lines += ["", "## Baseline temporal decoding", "",
              "| Dataset | Accuracy | Balanced accuracy | Macro F1 |", "|---|---:|---:|---:|"]
    for dataset_id, label in (("adex_ff", "AdEx FF"), ("adex_rec", "AdEx REC"),
                              ("hh_ff", "HH FF"), ("hh_rec", "HH REC")):
        row = temporal.loc[dataset_id]
        lines.append(f"| {label} | {row.accuracy:.4f} | {row.balanced_accuracy:.4f} | {row.macro_f1:.4f} |")
    if raw_included:
        row = temporal.loc["raw_shd"]
        lines.append(f"| Raw SHD input | {row.accuracy:.4f} | {row.balanced_accuracy:.4f} | {row.macro_f1:.4f} |")

    lines += ["", "## Paired baseline comparisons", "",
              "Positive delta favours the right-hand dataset.", "",
              "| Comparison | Delta accuracy | 95% paired bootstrap CI | Exact McNemar p |", "|---|---:|---:|---:|"]
    for _, row in comparisons[comparisons.family != "sweep_vs_recurrent_baseline"].iterrows():
        lines.append(f"| {row.comparison_id} | {row.delta_accuracy_right_minus_left:+.4f} | "
                     f"[{row.bootstrap_95_ci_low:+.4f}, {row.bootstrap_95_ci_high:+.4f}] | {row.mcnemar_exact_p:.4g} |")

    lines += ["", "## Representation controls", "",
              "`per_neuron_total_count` is the corrected timing-removed feature. "
              "`population_temporal_25ms` is what the notebook's wrong axis actually computed. "
              "Co-activation is a logical AND in the same 25 ms bin, evaluated on baseline recurrent-graph pairs "
              f"({pair_counts.get('connected', 'unknown')} pairs) and an equally sized deterministic non-edge control.",
              "", "## Notebook issues corrected or removed", ""]
    for title, detail in AUDIT_FINDINGS:
        lines.append(f"- **{title}:** {detail}")
    lines += ["", "## Interpretation limits", "",
              "- These are decoder results for one fixed simulated network seed. Trial-level intervals do not represent variation across independently simulated networks.",
              "- A decoder can reveal information present in a representation; it does not establish a causal circuit mechanism.",
              "- Raw SHD and SNN output have different channel counts and transformations, so their matched accuracies remain descriptive.",
              "- The parameter-sweep table reports all tested settings. Choosing a production setting requires a TRAIN-only selection rule or a new untouched evaluation set.", ""]
    atomic_text(output_dir / "analysis_report.md", "\n".join(lines))


def package_versions() -> dict:
    names = ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "tables")
    return {name: importlib.metadata.version(name) for name in names}


def configure_output(output_dir: Path, args, specs: list[DatasetSpec]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": 1, "script_sha256": sha256(Path(__file__)),
        "train_root": display_path(args.train_root), "test_root": display_path(args.test_root),
        "raw_train_h5": display_path(args.raw_train_h5) if args.raw_train_h5 else None,
        "raw_test_h5": display_path(args.raw_test_h5) if args.raw_test_h5 else None,
        "alpha": args.alpha, "bin_ms": BIN_MS, "t_max_ms": T_MAX_MS,
        "bootstraps": args.bootstraps, "seed": args.seed,
        "smoke_per_class": args.smoke_per_class,
        "datasets": [asdict(spec) for spec in specs],
    }
    config_path = output_dir / "run_config.json"
    if config_path.exists():
        old = load_json(config_path)
        old.pop("created_utc", None)
        if old != config:
            raise ValueError(f"Existing output was created with a different configuration: {output_dir}")
    else:
        config["created_utc"] = utc_now()
        atomic_json(config_path, config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, required=True,
                        help="Completed TRAIN run root containing sweeps/")
    parser.add_argument("--test-root", type=Path, required=True,
                        help="Completed TEST run root containing sweeps/")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw-train-h5", type=Path)
    parser.add_argument("--raw-test-h5", type=Path)
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel dataset processes; combine with BLAS thread limits to stay within allocation")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--bootstraps", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--smoke-per-class", type=int, default=0,
                        help="Validation only: use the first N valid trials per digit in each split")
    args = parser.parse_args()
    if (args.raw_train_h5 is None) != (args.raw_test_h5 is None):
        parser.error("Provide both raw HDF5 paths or neither")
    if args.workers < 1 or args.workers > 40:
        parser.error("--workers must be between 1 and 40")
    if args.alpha <= 0 or args.bootstraps < 100:
        parser.error("alpha must be positive and bootstraps must be at least 100")

    args.train_root = win_long_path(args.train_root)
    args.test_root = win_long_path(args.test_root)
    args.output_dir = win_long_path(args.output_dir)
    if args.raw_train_h5:
        args.raw_train_h5 = args.raw_train_h5.expanduser().resolve()
        args.raw_test_h5 = args.raw_test_h5.expanduser().resolve()

    start = time.perf_counter()
    specs = discover_datasets(args.train_root, args.test_root)
    catalogue = validate_catalogue(specs)
    configure_output(args.output_dir, args, specs)
    atomic_json(args.output_dir / "catalogue_validation.json", {
        "status": "passed", "datasets": catalogue, "created_utc": utc_now(),
    })
    print(f"Validated {len(specs)} physical SNN datasets", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(analyse_snn_dataset, spec, str(args.output_dir), args.alpha,
                               args.seed, args.smoke_per_class) for spec in specs]
        for future in as_completed(futures):
            print(f"Completed {future.result()}", flush=True)

    ids = [spec.dataset_id for spec in specs]
    first = load_snn(Path(specs[0].train_dir), specs[0].condition, args.smoke_per_class)
    first_test = load_snn(Path(specs[0].test_dir), specs[0].condition, args.smoke_per_class)
    if args.raw_train_h5:
        analyse_raw(args.raw_train_h5, args.raw_test_h5, args.output_dir, args.alpha,
                    args.smoke_per_class, (first.trial_ids, first.y), (first_test.trial_ids, first_test.y))
        ids.append("raw_shd")
        print("Completed raw_shd", flush=True)

    summaries, prediction_rows = load_partial_results(args.output_dir, ids)
    results = pd.DataFrame(summaries).sort_values(["model", "dataset_id", "representation"])
    predictions = pd.DataFrame(prediction_rows)
    comparisons = pd.DataFrame(make_comparisons(specs, predictions, args.bootstraps, args.seed))
    sweep_rows = make_sweep_table(specs, results)
    sweeps = pd.DataFrame(sweep_rows)

    results.to_csv(args.output_dir / "decoding_results.csv", index=False)
    sweeps.to_csv(args.output_dir / "parameter_sweep_results.csv", index=False)
    comparisons.to_csv(args.output_dir / "paired_comparisons.csv", index=False)
    with gzip.open(args.output_dir / "test_predictions.csv.gz", "wt", encoding="utf-8", newline="") as stream:
        predictions.to_csv(stream, index=False)

    baseline_metadata = load_json(partial_paths(args.output_dir, "adex_rec")[0])
    make_figures(args.output_dir, results, sweeps, comparisons)
    make_report(args.output_dir, results, comparisons, bool(args.raw_train_h5),
                baseline_metadata["pair_counts"], args.alpha)
    elapsed = time.perf_counter() - start
    completion = {
        "status": "complete", "completed_utc": utc_now(), "wall_seconds": elapsed,
        "physical_snn_datasets": len(specs), "result_rows": int(len(results)),
        "test_prediction_rows": int(len(predictions)), "paired_comparisons": int(len(comparisons)),
        "audit_findings": len(AUDIT_FINDINGS), "packages": package_versions(),
        "python": sys.version, "platform": platform.platform(),
    }
    atomic_json(args.output_dir / "COMPLETE.json", completion)
    print(json.dumps(completion, indent=2), flush=True)


if __name__ == "__main__":
    main()
