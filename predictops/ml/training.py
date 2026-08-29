"""Training and inference for the sequence models.

Conventions that keep the comparison honest:

* the training set is class-balanced by *subsampling negatives*; validation and
  test are always the untouched distribution,
* early stopping watches validation PR-AUC (F1 at a fixed cut is unstable when
  positives are 1% of rows),
* the decision threshold is chosen on validation and then frozen before the
  test split is touched even once,
* every run is seeded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset

from ..config import MODEL_DIR, TrainingConfig
from ..data.preprocessing import Scaler, WindowIndex, balanced_subsample
from .torch_models import (
    EnsembleModel,
    LSTMClassifier,
    TemporalFusionTransformer,
)

MACHINE_TYPES = ["COMPRESSOR", "CONVEYOR", "MOTOR", "PUMP"]
TYPE_INDEX = {t: i for i, t in enumerate(MACHINE_TYPES)}


def static_index(machine_id: str) -> int:
    return TYPE_INDEX.get(machine_id.split("-")[0], 0)


class WindowDataset(Dataset):
    def __init__(self, wi: WindowIndex, idx: np.ndarray | None = None):
        self.wi = wi
        self.idx = np.arange(len(wi)) if idx is None else np.asarray(idx)
        self.static = np.array(
            [static_index(wi.machine_of(int(i))) for i in self.idx],
            dtype=np.int64)

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        j = int(self.idx[i])
        return (torch.from_numpy(np.ascontiguousarray(self.wi.window(j))),
                torch.tensor(self.static[i]),
                torch.tensor(float(self.wi.labels[j])))


@dataclass
class TrainedModel:
    model: nn.Module
    name: str
    channels: list[str]
    scaler: Scaler
    threshold: float
    val_pr_auc: float
    epochs_run: int
    history: list[dict]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.model.state_dict(),
            "name": self.name,
            "channels": self.channels,
            "scaler": self.scaler.to_dict(),
            "threshold": self.threshold,
            "val_pr_auc": self.val_pr_auc,
            "history": self.history,
        }, path)


def build_model(kind: str, n_channels: int, seed: int = 42) -> nn.Module:
    torch.manual_seed(seed)
    if kind == "lstm":
        return LSTMClassifier(n_channels, n_static=len(MACHINE_TYPES))
    if kind == "tft":
        return TemporalFusionTransformer(n_channels, n_static=len(MACHINE_TYPES))
    raise ValueError(f"unknown sequence model: {kind}")


def materialise(wi: WindowIndex, idx: np.ndarray | None = None):
    """Stack a window subset into dense tensors.

    Slicing windows one at a time inside a DataLoader dominated the epoch time
    (~190 ms per batch of 256).  The subsets used here are ~100 MB, so paying
    the copy once and then batching tensor slices is far cheaper.
    """
    idx = np.arange(len(wi)) if idx is None else np.asarray(idx)
    x = np.empty((len(idx), wi.lookback, len(wi.columns)), dtype=np.float32)
    for k, j in enumerate(idx):
        x[k] = wi.window(int(j))
    static = np.array([static_index(wi.machine_of(int(j))) for j in idx],
                      dtype=np.int64)
    y = wi.labels[idx].astype(np.float32)
    return (torch.from_numpy(x), torch.from_numpy(static),
            torch.from_numpy(y))


@torch.no_grad()
def predict_tensors(model: nn.Module, x: torch.Tensor, static: torch.Tensor,
                    batch_size: int = 2048, device: str = "cpu") -> np.ndarray:
    model.eval().to(device)
    out = []
    for b in range(0, len(x), batch_size):
        out.append(torch.sigmoid(
            model(x[b:b + batch_size].to(device),
                  static[b:b + batch_size].to(device))).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


@torch.no_grad()
def predict_windows(model: nn.Module, wi: WindowIndex, batch_size: int = 2048,
                    device: str = "cpu", chunk: int = 16384) -> np.ndarray:
    """Chunked inference so a large split never materialises all at once."""
    model.eval().to(device)
    out = []
    for start in range(0, len(wi), chunk):
        idx = np.arange(start, min(start + chunk, len(wi)))
        x, st, _ = materialise(wi, idx)
        for b in range(0, len(idx), batch_size):
            xb = x[b:b + batch_size].to(device)
            sb = st[b:b + batch_size].to(device)
            out.append(torch.sigmoid(model(xb, sb)).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def train_sequence_model(kind: str, train_wi: WindowIndex, val_wi: WindowIndex,
                         cfg: TrainingConfig = TrainingConfig(),
                         neg_per_pos: float = 6.0,
                         progress: bool = True) -> TrainedModel:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    idx = balanced_subsample(train_wi.labels, neg_per_pos, cfg.seed)
    x_tr, s_tr, y_tr = materialise(train_wi, idx)
    n_train = len(idx)
    # validation is scored after every epoch -- stack it once, not 30 times
    x_va, s_va, _ = materialise(val_wi)

    n_channels = len(train_wi.columns)
    model = build_model(kind, n_channels, cfg.seed).to(cfg.device)

    y_sub = train_wi.labels[idx]
    pos_weight = torch.tensor(
        max((y_sub == 0).sum() / max((y_sub == 1).sum(), 1), 1.0),
        dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max",
                                                       factor=0.5, patience=2)

    best_state, best_score, best_epoch = None, -1.0, 0
    history: list[dict] = []

    gen = torch.Generator().manual_seed(cfg.seed)
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        total, n_batches = 0.0, 0
        perm = torch.randperm(n_train, generator=gen)
        for b in range(0, n_train, cfg.batch_size):
            sel = perm[b:b + cfg.batch_size]
            opt.zero_grad()
            logits = model(x_tr[sel].to(cfg.device), s_tr[sel].to(cfg.device))
            loss = loss_fn(logits, y_tr[sel].to(cfg.device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item())
            n_batches += 1

        val_prob = predict_tensors(model, x_va, s_va, device=cfg.device)
        score = (float(average_precision_score(val_wi.labels, val_prob))
                 if val_wi.labels.sum() > 0 else 0.0)
        sched.step(score)
        history.append({"epoch": epoch,
                        "train_loss": round(total / max(n_batches, 1), 5),
                        "val_pr_auc": round(score, 5)})
        if progress:
            print(f"  [{kind}] epoch {epoch:>2}  loss {total / max(n_batches, 1):.4f}"
                  f"  val PR-AUC {score:.4f}")

        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= cfg.patience:
            if progress:
                print(f"  [{kind}] early stop at epoch {epoch} "
                      f"(best {best_epoch}, PR-AUC {best_score:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    from .evaluation import pick_threshold
    val_prob = predict_tensors(model, x_va, s_va, device=cfg.device)
    threshold = pick_threshold(val_wi.labels, val_prob)

    return TrainedModel(model=model, name=kind, channels=list(train_wi.columns),
                        scaler=Scaler(np.zeros(1), np.ones(1), []),
                        threshold=threshold, val_pr_auc=best_score,
                        epochs_run=len(history), history=history)


def fit_ensemble_weight(models: list[nn.Module], val_wi: WindowIndex,
                        device: str = "cpu") -> tuple[list[float], float]:
    """Grid-search the blend weight on validation PR-AUC.

    If the blend is not actually better than its best member, the caller should
    say so rather than shipping an ensemble for its own sake.
    """
    probs = [predict_windows(m, val_wi, device=device) for m in models]
    y = val_wi.labels
    best_w, best_score = [1.0] + [0.0] * (len(models) - 1), -1.0
    for a in np.linspace(0.0, 1.0, 21):
        w = [a, 1.0 - a] if len(models) == 2 else None
        if w is None:
            break
        p = w[0] * probs[0] + w[1] * probs[1]
        s = float(average_precision_score(y, p))
        if s > best_score:
            best_score, best_w = s, list(w)
    return best_w, best_score


def load_model(path: Path, kind: str, n_channels: int) -> nn.Module:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(kind, n_channels)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model
