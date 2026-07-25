"""Stage-locked optimization for CADENCE models."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from cadence.model import HierarchicalControlledSSM, LossBreakdown, SequenceBatch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _trainable_parameters(model: torch.nn.Module) -> list[Tensor]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("stage has no trainable parameters")
    return parameters


def move_batch(batch: SequenceBatch, device: torch.device | str) -> SequenceBatch:
    return SequenceBatch(
        animal_id=batch.animal_id,
        neural=batch.neural.to(device),
        behavior=batch.behavior.to(device),
        inputs=batch.inputs.to(device),
        intervention=batch.intervention.to(device),
        onset=batch.onset,
        neural_mask=None if batch.neural_mask is None else batch.neural_mask.to(device),
        behavior_mask=None if batch.behavior_mask is None else batch.behavior_mask.to(device),
    )


@dataclass(slots=True)
class FitConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 500
    patience: int = 40
    gradient_clip: float = 5.0
    seed: int = 0
    device: str = "cuda"
    mixed_precision: bool = True


@dataclass(slots=True)
class EpochRecord:
    epoch: int
    train_loss: float
    validation_loss: float
    neural_loss: float
    behavior_loss: float


@dataclass(slots=True)
class FitResult:
    stage: str
    best_epoch: int
    best_validation_loss: float
    history: list[EpochRecord]
    config: FitConfig

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "best_epoch": self.best_epoch,
            "best_validation_loss": self.best_validation_loss,
            "history": [asdict(record) for record in self.history],
            "config": asdict(self.config),
        }


def _mean_breakdown(losses: list[LossBreakdown]) -> dict[str, float]:
    if not losses:
        raise ValueError("empty batch collection")
    names = ("total", "neural", "behavior")
    return {
        name: float(np.mean([float(getattr(loss, name).detach().cpu()) for loss in losses]))
        for name in names
    }


def _compute_loss(
    model: HierarchicalControlledSSM,
    batch: SequenceBatch,
    stage: Literal["normal", "intervention", "target_adaptation"],
) -> LossBreakdown:
    if stage in {"normal", "target_adaptation"}:
        return model.normal_loss(batch)
    return model.intervention_loss(batch, include_donor_delta=True)


def fit_stage(
    model: HierarchicalControlledSSM,
    train_batches: Iterable[SequenceBatch],
    validation_batches: Iterable[SequenceBatch],
    *,
    stage: Literal["normal", "intervention", "target_adaptation"],
    config: FitConfig,
    target_animal: str | None = None,
) -> FitResult:
    """Fit one stage with early stopping on protocol-eligible data only."""
    seed_everything(config.seed)
    train = list(train_batches)
    validation = list(validation_batches)
    if not train or not validation:
        raise ValueError("train and validation batches must be nonempty")
    if stage == "target_adaptation":
        if target_animal is None:
            raise ValueError("target_adaptation requires target_animal")
        allowed = {batch.animal_id for batch in train + validation}
        if allowed != {target_animal}:
            raise ValueError("target adaptation batches must belong only to the target animal")
    model.configure_stage(stage, target_animal=target_animal)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    train = [move_batch(batch, device) for batch in train]
    validation = [move_batch(batch, device) for batch in validation]
    optimizer = torch.optim.AdamW(
        _trainable_parameters(model),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    use_amp = config.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, Tensor] | None = None
    stale = 0
    history: list[EpochRecord] = []
    generator = random.Random(config.seed)

    for epoch in range(config.max_epochs):
        model.train()
        order = list(range(len(train)))
        generator.shuffle(order)
        train_losses: list[LossBreakdown] = []
        for index in order:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                loss = _compute_loss(model, train[index], stage)
            scaler.scale(loss.total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(_trainable_parameters(model), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss)

        model.eval()
        validation_losses: list[LossBreakdown] = []
        with torch.no_grad():
            for batch in validation:
                validation_losses.append(_compute_loss(model, batch, stage))
        train_mean = _mean_breakdown(train_losses)
        validation_mean = _mean_breakdown(validation_losses)
        history.append(
            EpochRecord(
                epoch,
                train_mean["total"],
                validation_mean["total"],
                validation_mean["neural"],
                validation_mean["behavior"],
            )
        )
        if validation_mean["total"] < best_loss - 1e-8:
            best_loss = validation_mean["total"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("optimization produced no finite validation checkpoint")
    model.load_state_dict(best_state)
    return FitResult(stage, best_epoch, best_loss, history, config)


def save_frozen_prediction_bundle(
    destination: Path,
    *,
    predictions: dict[str, np.ndarray],
    config: dict[str, object],
    split_manifest: dict[str, object],
    git_commit: str,
) -> str:
    """Save and hash predictions before held-out outcomes are opened."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "config": config,
        "split_manifest": split_manifest,
        "git_commit": git_commit,
        "arrays": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in predictions.items()
        },
    }
    metadata_json = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    np.savez_compressed(destination, metadata=np.asarray(metadata_json), **predictions)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix(destination.suffix + ".sha256").write_text(
        f"{digest}  {destination.name}\n",
        encoding="utf-8",
    )
    return digest
