"""Dataset checks and training implementation for the adaptive path predictor."""

from __future__ import annotations

import copy
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DATASET_SCHEMA_VERSION = "adaptive_router_dataset_v1"
PREDICTOR_SCHEMA_VERSION = "adaptive_router_predictor_v1"
LABEL_NAMES = {0: "rtl_direct", 1: "c_first"}
ARCHITECTURE_NAMES = ("tiny", "legacy_large", "overfit_control")
TINY_HIDDEN_DIM = 8
TINY_BOTTLENECK_DIM = 2
LEGACY_HIDDEN_DIM = 128
LEGACY_BOTTLENECK_DIM = 32
DEFAULT_SPLIT_SEEDS = (7, 17, 27, 37, 47)


@dataclass(frozen=True)
class PredictorConfig:
    # The default is intentionally tiny: the available labeled data are small
    # and the router is a binary decision aid, not a large representation model.
    architecture: str = "tiny"
    hidden_dim: int = TINY_HIDDEN_DIM
    bottleneck_dim: int = TINY_BOTTLENECK_DIM
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 200
    patience: int = 20
    seed: int = 7
    min_training_samples: int = 50
    min_training_families: int = 5
    validation_fraction: float = 0.2
    test_fraction: float = 0.2
    split_seeds: tuple[int, ...] = DEFAULT_SPLIT_SEEDS
    require_objective_label_coverage: bool = True
    min_train_combo_count: int = 2
    min_eval_combo_count_warning: int = 3

    def __post_init__(self) -> None:
        architecture = str(self.architecture).strip().lower()
        if architecture not in ARCHITECTURE_NAMES:
            raise ValueError(
                f"Unknown MLP architecture {self.architecture!r}; choose from "
                f"{list(ARCHITECTURE_NAMES)}"
            )
        object.__setattr__(self, "architecture", architecture)

        dimensions = (int(self.hidden_dim), int(self.bottleneck_dim))
        if architecture == "tiny":
            if dimensions != (TINY_HIDDEN_DIM, TINY_BOTTLENECK_DIM):
                raise ValueError(
                    "128->32 is legacy-only; select architecture='legacy_large' "
                    "or architecture='overfit_control' explicitly"
                )
        else:
            # Let an explicit legacy architecture select the historical width
            # without requiring callers to duplicate the two dimensions.
            if dimensions == (TINY_HIDDEN_DIM, TINY_BOTTLENECK_DIM):
                object.__setattr__(self, "hidden_dim", LEGACY_HIDDEN_DIM)
                object.__setattr__(self, "bottleneck_dim", LEGACY_BOTTLENECK_DIM)
            elif dimensions != (LEGACY_HIDDEN_DIM, LEGACY_BOTTLENECK_DIM):
                raise ValueError(
                    "legacy_large/overfit_control must use hidden_dim=128 and "
                    "bottleneck_dim=32"
                )

        seeds = tuple(int(seed) for seed in self.split_seeds)
        if not seeds:
            raise ValueError("split_seeds must contain at least one seed")
        object.__setattr__(self, "split_seeds", seeds)
        if self.min_train_combo_count < 1:
            raise ValueError("min_train_combo_count must be at least one")
        if self.min_eval_combo_count_warning < 0:
            raise ValueError("min_eval_combo_count_warning must be non-negative")


@dataclass
class RouterDataset:
    explicit: np.ndarray
    latent: np.ndarray
    labels: np.ndarray
    spec_ids: np.ndarray
    family_ids: np.ndarray
    objectives: np.ndarray
    manifest: dict[str, Any]
    npz_path: Path
    manifest_path: Path

    @property
    def combined(self) -> np.ndarray:
        return np.concatenate([self.explicit, self.latent], axis=1).astype(
            np.float32, copy=False
        )


class RouterMLP(nn.Module):
    """Shallow binary classifier that emits a c_first logit."""

    def __init__(self, input_dim: int, config: PredictorConfig | None = None) -> None:
        super().__init__()
        cfg = config or PredictorConfig()
        self.input_dim = int(input_dim)
        layers: list[nn.Module] = []
        if cfg.architecture in {"legacy_large", "overfit_control"}:
            # Keep the historical state-dict layout for old checkpoints.  The
            # tiny model already receives train-only StandardScaler output, so
            # an additional per-row LayerNorm would erase useful scale signal.
            layers.append(nn.LayerNorm(self.input_dim))
        layers.extend(
            [
                nn.Linear(self.input_dim, cfg.hidden_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_dim, cfg.bottleneck_dim),
                nn.GELU(),
                nn.Linear(cfg.bottleneck_dim, 1),
            ]
        )
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported dataset schema {payload.get('schema_version')!r} in {path}"
        )
    return payload


def load_router_dataset(
    npz_path: str | Path, manifest_path: str | Path | None = None
) -> RouterDataset:
    npz = Path(npz_path).resolve()
    manifest = Path(manifest_path).resolve() if manifest_path else npz.with_suffix(".json")
    if not npz.is_file() or not manifest.is_file():
        raise FileNotFoundError(f"Dataset pair is incomplete: {npz}, {manifest}")
    metadata = _load_manifest(manifest)
    with np.load(npz, allow_pickle=False) as payload:
        dataset = RouterDataset(
            explicit=np.asarray(payload["explicit"], dtype=np.float32),
            latent=np.asarray(payload["latent"], dtype=np.float32),
            labels=np.asarray(payload["labels"], dtype=np.int64),
            spec_ids=np.asarray(payload["spec_ids"], dtype=str),
            family_ids=np.asarray(payload["family_ids"], dtype=str),
            objectives=np.asarray(payload["objectives"], dtype=str),
            manifest=metadata,
            npz_path=npz,
            manifest_path=manifest,
        )
    validate_dataset_shapes(dataset)
    return dataset


def validate_dataset_shapes(dataset: RouterDataset) -> None:
    if dataset.explicit.ndim != 2 or dataset.latent.ndim != 2:
        raise ValueError("explicit and latent arrays must both be rank 2")
    sample_count = dataset.explicit.shape[0]
    arrays = {
        "latent": dataset.latent,
        "labels": dataset.labels,
        "spec_ids": dataset.spec_ids,
        "family_ids": dataset.family_ids,
        "objectives": dataset.objectives,
    }
    for name, value in arrays.items():
        if value.shape[0] != sample_count:
            raise ValueError(f"{name} sample count does not match explicit features")
    if len(set(dataset.spec_ids.tolist())) != sample_count:
        raise ValueError("spec_ids must be unique within a router dataset")
    if not np.isfinite(dataset.explicit).all() or not np.isfinite(dataset.latent).all():
        raise ValueError("Dataset contains NaN or infinity")
    if dataset.manifest.get("sample_count") != sample_count:
        raise ValueError("Dataset manifest sample_count does not match arrays")
    expected_explicit = dataset.manifest.get("explicit_dim")
    expected_latent = dataset.manifest.get("latent_dim")
    if expected_explicit != dataset.explicit.shape[1]:
        raise ValueError("Dataset manifest explicit_dim does not match array")
    if expected_latent != dataset.latent.shape[1]:
        raise ValueError("Dataset manifest latent_dim does not match array")


def _objective_values(dataset: RouterDataset) -> set[str]:
    return {
        str(value).strip().upper()
        for value in np.asarray(dataset.objectives, dtype=str).tolist()
        if str(value).strip()
    }


def _objective_label_counts(
    dataset: RouterDataset, indices: np.ndarray
) -> Counter[tuple[str, int]]:
    return Counter(
        (
            str(objective).strip().upper(),
            int(label),
        )
        for objective, label in zip(
            dataset.objectives[indices], dataset.labels[indices]
        )
    )


def required_objective_label_combinations(
    dataset: RouterDataset,
) -> tuple[tuple[str, int], ...]:
    """Return every objective/label combination a valid split must contain.

    AREA-only datasets retain the original two-class requirement.  A combined
    AREA+TIMING dataset must contain all four combinations in every partition;
    otherwise a model comparison could hide that one objective is absent from a
    validation or test result.
    """

    objectives = sorted(_objective_values(dataset))
    return tuple((objective, label) for objective in objectives for label in (0, 1))


def _missing_objective_label_combinations(
    dataset: RouterDataset, indices: np.ndarray
) -> list[tuple[str, int]]:
    required = set(required_objective_label_combinations(dataset))
    observed = set(_objective_label_counts(dataset, indices))
    return sorted(required - observed)


def _counter_to_json(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def summarize_split(
    dataset: RouterDataset, split: dict[str, np.ndarray]
) -> dict[str, dict[str, Any]]:
    """Describe each partition without exposing a train accuracy metric."""

    summary: dict[str, dict[str, Any]] = {}
    for name, indices in split.items():
        objective_counts = Counter(
            str(value).strip().upper() for value in dataset.objectives[indices]
        )
        label_counts = Counter(int(value) for value in dataset.labels[indices])
        family_counts = Counter(str(value) for value in dataset.family_ids[indices])
        combo_counts = _objective_label_counts(dataset, indices)
        summary[name] = {
            "sample_count": int(indices.size),
            "objective_counts": _counter_to_json(objective_counts),
            "label_counts": _counter_to_json(label_counts),
            "family_count": int(len(family_counts)),
            "family_counts": _counter_to_json(family_counts),
            "objective_label_counts": {
                f"{objective}_y{label}": int(count)
                for (objective, label), count in sorted(combo_counts.items())
            },
        }
    return summary


def split_coverage_warnings(
    dataset: RouterDataset,
    split: dict[str, np.ndarray],
    config: PredictorConfig | None = None,
) -> list[str]:
    """Warn when an evaluation partition has only a tiny objective/label sample."""

    cfg = config or PredictorConfig()
    threshold = int(cfg.min_eval_combo_count_warning)
    if threshold <= 0:
        return []
    warnings: list[str] = []
    required = required_objective_label_combinations(dataset)
    for partition in ("validation", "test"):
        counts = _objective_label_counts(dataset, split[partition])
        for objective, label in required:
            count = int(counts.get((objective, label), 0))
            if count < threshold:
                warnings.append(
                    f"{partition} has only {count} samples for "
                    f"{objective}_y{label}; recommended minimum is {threshold}"
                )
    return warnings


def validate_training_eligibility(
    dataset: RouterDataset, config: PredictorConfig | None = None
) -> None:
    cfg = config or PredictorConfig()
    if dataset.manifest.get("training_allowed") is not True:
        raise ValueError(
            "Dataset manifest does not authorize training; pilot/smoke data are test-only"
        )
    if dataset.explicit.shape[0] < cfg.min_training_samples:
        raise ValueError(
            f"Need at least {cfg.min_training_samples} labeled samples, got "
            f"{dataset.explicit.shape[0]}"
        )
    unique_families = np.unique(dataset.family_ids)
    if unique_families.size < cfg.min_training_families:
        raise ValueError(
            f"Need at least {cfg.min_training_families} design families, got "
            f"{unique_families.size}"
        )
    labels = set(dataset.labels.tolist())
    if not labels.issubset({0, 1}) or labels != {0, 1}:
        raise ValueError("Training labels must contain both 0=rtl_direct and 1=c_first")
    if cfg.require_objective_label_coverage:
        missing = _missing_objective_label_combinations(
            dataset, np.arange(dataset.explicit.shape[0])
        )
        if missing:
            formatted = ", ".join(f"{objective}_y{label}" for objective, label in missing)
            raise ValueError(
                "Training data must contain every objective/label combination; "
                f"missing: {formatted}"
            )
    for label, label_name in LABEL_NAMES.items():
        label_families = np.unique(dataset.family_ids[dataset.labels == label])
        if label_families.size < 3:
            raise ValueError(
                f"Label {label_name} must occur in at least three design families "
                "for family-disjoint train/validation/test evaluation"
            )


def family_disjoint_split(
    dataset: RouterDataset,
    config: PredictorConfig | None = None,
    *,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    cfg = config or PredictorConfig()
    split_seed = cfg.seed if seed is None else int(seed)
    indices = np.arange(dataset.explicit.shape[0])
    relative_validation = cfg.validation_fraction / (1.0 - cfg.test_fraction)
    split: dict[str, np.ndarray] | None = None
    best_score: tuple[float, float, float] | None = None
    if cfg.require_objective_label_coverage:
        missing = _missing_objective_label_combinations(dataset, indices)
        if missing:
            formatted = ", ".join(f"{objective}_y{label}" for objective, label in missing)
            raise ValueError(
                "Cannot split: dataset lacks required objective/label combinations: "
                f"{formatted}"
            )
    outer = GroupShuffleSplit(
        n_splits=100, test_size=cfg.test_fraction, random_state=split_seed
    )
    for outer_index, (train_val_idx, test_idx) in enumerate(
        outer.split(indices, dataset.labels, groups=dataset.family_ids)
    ):
        inner = GroupShuffleSplit(
            n_splits=30,
            test_size=relative_validation,
            random_state=split_seed + 1 + outer_index,
        )
        for inner_train, inner_val in inner.split(
            train_val_idx,
            dataset.labels[train_val_idx],
            groups=dataset.family_ids[train_val_idx],
        ):
            candidate = {
                "train": train_val_idx[inner_train],
                "validation": train_val_idx[inner_val],
                "test": test_idx,
            }
            if all(
                set(dataset.labels[index].tolist()) == {0, 1}
                for index in candidate.values()
            ) and (
                not cfg.require_objective_label_coverage
                or all(
                    not _missing_objective_label_combinations(dataset, index)
                    for index in candidate.values()
                )
            ):
                train_combo_counts = _objective_label_counts(
                    dataset, candidate["train"]
                )
                required_combinations = required_objective_label_combinations(dataset)
                if any(
                    train_combo_counts.get(combination, 0)
                    < cfg.min_train_combo_count
                    for combination in required_combinations
                ):
                    continue

                # GroupShuffleSplit controls the number of families rather than
                # the number of samples.  A large family can therefore make the
                # first legal candidate very far from 60/20/20.  Search all
                # candidates and retain the closest, most representative one.
                target_fractions = {
                    "train": 1.0 - cfg.validation_fraction - cfg.test_fraction,
                    "validation": cfg.validation_fraction,
                    "test": cfg.test_fraction,
                }
                sample_count = float(indices.size)
                size_error = sum(
                    abs(candidate[name].size / sample_count - target_fraction)
                    for name, target_fraction in target_fractions.items()
                )
                global_combo_counts = _objective_label_counts(dataset, indices)
                distribution_error = 0.0
                for name, index in candidate.items():
                    candidate_counts = _objective_label_counts(dataset, index)
                    for combination, global_count in global_combo_counts.items():
                        distribution_error += abs(
                            candidate_counts.get(combination, 0) / index.size
                            - global_count / sample_count
                        )
                score = (
                    size_error + 0.1 * distribution_error,
                    size_error,
                    distribution_error,
                )
                if best_score is None or score < best_score:
                    best_score = score
                    split = candidate
    if split is None:
        raise ValueError(
            "Could not construct family-disjoint splits containing both labels; "
            "collect more design families for each route label"
        )
    family_sets = {
        name: set(dataset.family_ids[index].tolist()) for name, index in split.items()
    }
    if (
        family_sets["train"] & family_sets["validation"]
        or family_sets["train"] & family_sets["test"]
        or family_sets["validation"] & family_sets["test"]
    ):
        raise RuntimeError("Family-disjoint split leaked a design family")
    for name, index in split.items():
        if index.size == 0:
            raise ValueError(f"Family split produced an empty {name} partition")
    return split


def family_disjoint_splits(
    dataset: RouterDataset,
    config: PredictorConfig | None = None,
    seeds: tuple[int, ...] | list[int] | None = None,
) -> dict[int, dict[str, np.ndarray]]:
    """Construct independent, family-disjoint splits for each requested seed."""

    cfg = config or PredictorConfig()
    split_seeds = tuple(cfg.split_seeds if seeds is None else (int(seed) for seed in seeds))
    if not split_seeds:
        raise ValueError("At least one split seed is required")
    result: dict[int, dict[str, np.ndarray]] = {}
    for split_seed in split_seeds:
        if split_seed in result:
            raise ValueError(f"Duplicate split seed: {split_seed}")
        result[split_seed] = family_disjoint_split(
            dataset, cfg, seed=split_seed
        )
    return result


def dry_run(dataset: RouterDataset, config: PredictorConfig | None = None) -> dict[str, Any]:
    """Validate shape/model wiring without labels, optimization, or saved weights."""

    cfg = config or PredictorConfig()
    combined = dataset.combined
    torch.manual_seed(cfg.seed)
    model = RouterMLP(combined.shape[1], cfg)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(combined))
        probabilities = torch.sigmoid(logits)
    if logits.shape != (combined.shape[0],):
        raise RuntimeError("MLP dry-run output shape is invalid")
    if not torch.isfinite(probabilities).all():
        raise RuntimeError("MLP dry-run produced NaN or infinity")
    return {
        "mode": "dry_run_no_training",
        "sample_count": int(combined.shape[0]),
        "explicit_dim": int(dataset.explicit.shape[1]),
        "latent_dim": int(dataset.latent.shape[1]),
        "input_dim": int(combined.shape[1]),
        "architecture": cfg.architecture,
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "output_shape": list(logits.shape),
        "optimizer_steps": 0,
        "weights_saved": False,
    }


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | None]:
    predictions = (probabilities >= 0.5).astype(np.int64)
    result: dict[str, float | None] = {
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "auroc": None,
    }
    if np.unique(labels).size == 2:
        result["auroc"] = float(roc_auc_score(labels, probabilities))
    return result


def _partition_metrics(
    dataset: RouterDataset,
    indices: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float | None]:
    """Report overall and per-objective scores for an unseen partition."""

    result = dict(_metrics(dataset.labels[indices], probabilities))
    partition_objectives = np.asarray(dataset.objectives[indices], dtype=str)
    for objective in sorted(set(partition_objectives.tolist())):
        mask = partition_objectives == objective
        objective_metrics = _metrics(dataset.labels[indices][mask], probabilities[mask])
        prefix = objective.strip().lower()
        result.update(
            {f"{prefix}_{name}": value for name, value in objective_metrics.items()}
        )
    return result


def train_logistic_baseline(
    dataset: RouterDataset,
    split: dict[str, np.ndarray],
    output_dir: str | Path,
    config: PredictorConfig | None = None,
) -> dict[str, Any]:
    cfg = config or PredictorConfig()
    features = dataset.combined
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=cfg.seed,
                ),
            ),
        ]
    )
    model.fit(features[split["train"]], dataset.labels[split["train"]])
    metrics = {}
    for name in ("validation", "test"):
        indices = split[name]
        probabilities = model.predict_proba(features[indices])[:, 1]
        metrics[name] = _partition_metrics(dataset, indices, probabilities)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "logistic_baseline.joblib"
    joblib.dump(model, model_path)
    return {"model_path": str(model_path), "metrics": metrics}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_mlp(
    dataset: RouterDataset,
    split: dict[str, np.ndarray],
    output_dir: str | Path,
    config: PredictorConfig | None = None,
    *,
    device: str = "cuda",
) -> dict[str, Any]:
    cfg = config or PredictorConfig()
    _seed_everything(cfg.seed)
    features = dataset.combined
    scaler = StandardScaler().fit(features[split["train"]])
    scaled = scaler.transform(features).astype(np.float32)

    train_labels = dataset.labels[split["train"]]
    positives = max(1, int((train_labels == 1).sum()))
    negatives = max(1, int((train_labels == 0).sum()))
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model = RouterMLP(scaled.shape[1], cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    generator = torch.Generator().manual_seed(cfg.seed)
    train_dataset = TensorDataset(
        torch.from_numpy(scaled[split["train"]]),
        torch.from_numpy(train_labels.astype(np.float32)),
    )
    loader = DataLoader(
        train_dataset,
        batch_size=min(cfg.batch_size, len(train_dataset)),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    val_x = torch.from_numpy(scaled[split["validation"]]).to(device)
    val_y = torch.from_numpy(
        dataset.labels[split["validation"]].astype(np.float32)
    ).to(device)
    best_state: dict[str, Any] | None = None
    best_val_loss = math.inf
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(cfg.max_epochs):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach()) * batch_x.shape[0]
            train_count += batch_x.shape[0]

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_x), val_y).detach())
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss_sum / max(1, train_count),
                "validation_loss": val_loss,
            }
        )
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.patience:
                break
    if best_state is None:
        raise RuntimeError("MLP training did not produce a finite validation state")
    model.load_state_dict(best_state)

    metrics = {}
    model.eval()
    with torch.no_grad():
        for name in ("validation", "test"):
            indices = split[name]
            tensor = torch.from_numpy(scaled[indices]).to(device)
            probabilities = torch.sigmoid(model(tensor)).cpu().numpy()
            metrics[name] = _partition_metrics(dataset, indices, probabilities)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "router_mlp.pt"
    torch.save(
        {
            "schema_version": PREDICTOR_SCHEMA_VERSION,
            "state_dict": model.state_dict(),
            "input_dim": int(scaled.shape[1]),
            "explicit_dim": int(dataset.explicit.shape[1]),
            "latent_dim": int(dataset.latent.shape[1]),
            "architecture": cfg.architecture,
            "config": asdict(cfg),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "dataset_manifest": dataset.manifest,
        },
        checkpoint_path,
    )
    return {
        "checkpoint_path": str(checkpoint_path),
        "architecture": cfg.architecture,
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "best_validation_loss": best_val_loss,
        "epochs_completed": len(history),
        "metrics": metrics,
        "history": history,
    }


# ── Inference ─────────────────────────────────────────────────────────────
#
# Everything above trains and writes ``router_mlp.pt``. Nothing read it back
# until this block: the checkpoint was a write-only artifact, so the plan's
# second decision stage ("latent predictor") had no way to produce a p at all.


@dataclass(frozen=True)
class RouterProbability:
    """One spec's c_first probability, with the numbers behind it."""

    p_c_first: float
    logit: float
    explicit_dim: int
    latent_dim: int
    checkpoint_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RouterPredictor:
    """A trained ``router_mlp.pt`` restored for inference.

    The scaler is the part that is easy to get wrong: training standardised the
    combined feature matrix, so inference MUST reuse the stored
    ``scaler_mean`` / ``scaler_scale``. Re-fitting, or skipping standardisation,
    silently produces a p that looks plausible and means nothing.
    """

    def __init__(
        self,
        model: RouterMLP,
        *,
        scaler_mean: np.ndarray,
        scaler_scale: np.ndarray,
        explicit_dim: int,
        latent_dim: int,
        config: PredictorConfig,
        dataset_manifest: dict[str, Any],
        checkpoint_path: Path,
        device: str,
    ) -> None:
        self.model = model
        self.scaler_mean = np.asarray(scaler_mean, dtype=np.float64).reshape(-1)
        self.scaler_scale = np.asarray(scaler_scale, dtype=np.float64).reshape(-1)
        # StandardScaler already maps zero-variance columns to scale 1.0, but the
        # explicit block genuinely contains constant dimensions, so a corrupted
        # or hand-edited checkpoint must not turn into a division by zero.
        if not np.all(np.isfinite(self.scaler_scale)) or np.any(
            self.scaler_scale == 0.0
        ):
            raise ValueError(
                f"{checkpoint_path}: scaler_scale contains zero or non-finite "
                "entries; the checkpoint cannot standardise features"
            )
        self.explicit_dim = int(explicit_dim)
        self.latent_dim = int(latent_dim)
        self.input_dim = self.explicit_dim + self.latent_dim
        if self.scaler_mean.shape != (self.input_dim,):
            raise ValueError(
                f"{checkpoint_path}: scaler covers {self.scaler_mean.shape[0]} "
                f"features but explicit+latent is {self.input_dim}"
            )
        self.config = config
        self.dataset_manifest = dict(dataset_manifest or {})
        self.checkpoint_path = Path(checkpoint_path)
        self.device = str(device)

    @property
    def trained_on_allowed_dataset(self) -> bool:
        """Whether the dataset this was trained on cleared the §5 label gate.

        A predictor trained on an unverified or too-small dataset still loads --
        that is deliberate, so the wiring can be tested -- but the caller is
        expected to report this instead of presenting p as a real measurement.
        """
        return bool(self.dataset_manifest.get("training_allowed", False))

    def _standardize(self, features: np.ndarray) -> np.ndarray:
        return ((features - self.scaler_mean) / self.scaler_scale).astype(np.float32)

    def _combine(self, explicit: Any, latent: Any) -> np.ndarray:
        explicit_array = np.asarray(explicit, dtype=np.float64).reshape(1, -1)
        latent_array = np.asarray(latent, dtype=np.float64).reshape(1, -1)
        if explicit_array.shape[1] != self.explicit_dim:
            raise ValueError(
                f"explicit vector has {explicit_array.shape[1]} dims, checkpoint "
                f"expects {self.explicit_dim}"
            )
        if latent_array.shape[1] != self.latent_dim:
            raise ValueError(
                f"latent vector has {latent_array.shape[1]} dims, checkpoint "
                f"expects {self.latent_dim}"
            )
        combined = np.concatenate([explicit_array, latent_array], axis=1)
        if not np.all(np.isfinite(combined)):
            raise ValueError("feature vector contains NaN or infinity")
        return combined

    def predict_proba(self, explicit: Any, latent: Any) -> RouterProbability:
        """Return P(c_first | spec) for one spec."""
        scaled = self._standardize(self._combine(explicit, latent))
        self.model.eval()
        with torch.no_grad():
            logit = float(
                self.model(torch.from_numpy(scaled).to(self.device)).cpu().item()
            )
        return RouterProbability(
            p_c_first=float(1.0 / (1.0 + math.exp(-logit))),
            logit=logit,
            explicit_dim=self.explicit_dim,
            latent_dim=self.latent_dim,
            checkpoint_path=str(self.checkpoint_path),
        )


def load_predictor(
    checkpoint_path: str | Path, *, device: str = "cpu"
) -> RouterPredictor:
    """Restore a trained router MLP for inference.

    ``device`` defaults to cpu: even the legacy overfitting-control head is tiny
    compared with the frozen embedding model, and inference may run next to a DC
    or JasperGold job that needs the accelerator resources.
    """
    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"No router checkpoint at {path}")
    # weights_only defaults to True from torch 2.6 on, which rejects the numpy
    # scaler arrays this checkpoint legitimately carries.
    payload = torch.load(path, map_location=device, weights_only=False)
    schema = payload.get("schema_version")
    if schema != PREDICTOR_SCHEMA_VERSION:
        raise ValueError(
            f"{path}: unsupported predictor schema {schema!r}, expected "
            f"{PREDICTOR_SCHEMA_VERSION!r}"
        )
    for key in ("state_dict", "input_dim", "explicit_dim", "latent_dim",
                "scaler_mean", "scaler_scale"):
        if key not in payload:
            raise ValueError(f"{path}: checkpoint is missing {key!r}")

    raw_config = dict(payload.get("config") or {})
    # v1 checkpoints created before the tiny default existed do not carry an
    # architecture field.  Their 128->32 dimensions identify the historical
    # model, so load them explicitly as legacy_large instead of silently
    # rebuilding the new 8->2 network and failing with an opaque state mismatch.
    if "architecture" not in raw_config:
        if (
            int(raw_config.get("hidden_dim", TINY_HIDDEN_DIM)) == LEGACY_HIDDEN_DIM
            and int(raw_config.get("bottleneck_dim", TINY_BOTTLENECK_DIM))
            == LEGACY_BOTTLENECK_DIM
        ):
            raw_config["architecture"] = "legacy_large"
    config = PredictorConfig(**raw_config)
    model = RouterMLP(int(payload["input_dim"]), config)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return RouterPredictor(
        model,
        scaler_mean=payload["scaler_mean"],
        scaler_scale=payload["scaler_scale"],
        explicit_dim=int(payload["explicit_dim"]),
        latent_dim=int(payload["latent_dim"]),
        config=config,
        dataset_manifest=payload.get("dataset_manifest") or {},
        checkpoint_path=path,
        device=device,
    )
