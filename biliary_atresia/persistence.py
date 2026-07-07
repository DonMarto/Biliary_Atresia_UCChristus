"""
persistence.py
--------------
Save / load models and training data, update the model with new samples,
and run inference on a single image or feature vector.
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from .model import RiskMLP, train_model, predict_probability
from .imaging import load_image_rgb, segment_image
from .features import extract_lab_feature


# ---------------------------------------------------------------------------
# Paths / conventions
# ---------------------------------------------------------------------------

DEFAULT_DIR = Path("checkpoints")

def _model_path(directory: Path, name: str) -> Path:
    return directory / f"{name}.pt"

def _data_path(directory: Path, name: str) -> Path:
    return directory / f"{name}_data.npz"

def _meta_path(directory: Path, name: str) -> Path:
    return directory / f"{name}_meta.json"


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(
    model: RiskMLP,
    name: str = "model",
    directory: str | Path = DEFAULT_DIR,
    metadata: dict = None,
) -> Path:
    """
    Save *model* weights to ``<directory>/<name>.pt``.

    A JSON sidecar (``<name>_meta.json``) is written alongside with
    the input dimension, timestamp, and any extra *metadata* you pass in.

    Parameters
    ----------
    model : RiskMLP
    name : str
        Base filename (no extension).
    directory : str or Path
    metadata : dict or None
        Arbitrary key/value pairs merged into the sidecar.

    Returns
    -------
    path : Path  – where the weights were written
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    path = _model_path(directory, name)
    torch.save(model.state_dict(), path)

    # Infer input_dim from the first Linear layer
    input_dim = next(
        m.in_features for m in model.modules() if isinstance(m, torch.nn.Linear)
    )

    meta = {
        "input_dim": input_dim,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **(metadata or {}),
    }
    _meta_path(directory, name).write_text(json.dumps(meta, indent=2))

    print(f"Model saved → {path}")
    return path


def load_model(
    name: str = "model",
    directory: str | Path = DEFAULT_DIR,
) -> RiskMLP:
    """
    Load a :class:`RiskMLP` from ``<directory>/<name>.pt``.

    The companion ``<name>_meta.json`` is used to restore ``input_dim``
    so you never have to pass it manually.

    Parameters
    ----------
    name : str
    directory : str or Path

    Returns
    -------
    model : RiskMLP  (eval mode, on the appropriate device)
    """
    directory = Path(directory)
    meta = json.loads(_meta_path(directory, name).read_text())
    input_dim = meta["input_dim"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RiskMLP(input_dim=input_dim).to(device)
    model.load_state_dict(
        torch.load(_model_path(directory, name), map_location=device)
    )
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Data persistence
# ---------------------------------------------------------------------------

def save_data(
    X_syn,
    y_syn,
    X_pillar,
    y_pillar,
    name: str = "model",
    directory: str | Path = DEFAULT_DIR,
) -> Path:
    """
    Save training arrays to ``<directory>/<name>_data.npz``.

    Returns
    -------
    path : Path
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    path = _data_path(directory, name)
    np.savez(
        path,
        X_syn=np.asarray(X_syn),
        y_syn=np.asarray(y_syn),
        X_pillar=np.asarray(X_pillar),
        y_pillar=np.asarray(y_pillar),
    )
    print(f"Data saved  → {path}")
    return path


def load_data(
    name: str = "model",
    directory: str | Path = DEFAULT_DIR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load training arrays from ``<directory>/<name>_data.npz``.

    Returns
    -------
    X_syn, y_syn, X_pillar, y_pillar : np.ndarray
    """
    path = _data_path(Path(directory), name)
    data = np.load(path)
    return data["X_syn"], data["y_syn"], data["X_pillar"], data["y_pillar"]


def save_checkpoint(
    model: RiskMLP,
    X_syn,
    y_syn,
    X_pillar,
    y_pillar,
    name: str = "model",
    directory: str | Path = DEFAULT_DIR,
    metadata: dict = None,
) -> None:
    """Atomically save model weights and training data together."""
    save_model(model, name=name, directory=directory, metadata=metadata)
    save_data(X_syn, y_syn, X_pillar, y_pillar, name=name, directory=directory)


# ---------------------------------------------------------------------------
# Model update
# ---------------------------------------------------------------------------

def update_model(
    new_X_syn=None,
    new_y_syn=None,
    new_X_pillar=None,
    new_y_pillar=None,
    name: str = "model",
    directory: str | Path = DEFAULT_DIR,
    train_kwargs: dict = None,
) -> RiskMLP:
    """
    Merge new samples into the stored dataset and retrain from scratch.

    Pass only the arrays that have new data; the rest are left unchanged.

    Parameters
    ----------
    new_X_syn : array-like or None
        New synthetic feature rows to append.
    new_y_syn : array-like or None
        Corresponding labels.
    new_X_pillar : array-like or None
        New anchor (pillar) feature rows to append.
    new_y_pillar : array-like or None
        Corresponding labels.
    name : str
        Checkpoint name to load from and overwrite.
    directory : str or Path
    train_kwargs : dict or None
        Keyword arguments forwarded to :func:`train_model`
        (e.g. ``{'hidden_epochs': 1000, 'lam': 3.0}``).

    Returns
    -------
    model : RiskMLP  – freshly trained on the merged dataset
    """
    # Load existing data
    X_syn, y_syn, X_pillar, y_pillar = load_data(name=name, directory=directory)

    # Merge new synthetic samples
    if new_X_syn is not None and new_y_syn is not None:
        X_syn    = np.vstack([X_syn,    np.asarray(new_X_syn)])
        y_syn    = np.concatenate([y_syn, np.asarray(new_y_syn)])
        print(f"Synthetic data: {len(y_syn) - len(new_y_syn)} → {len(y_syn)} samples")

    # Merge new pillar samples
    if new_X_pillar is not None and new_y_pillar is not None:
        X_pillar = np.vstack([X_pillar, np.asarray(new_X_pillar)])
        y_pillar = np.concatenate([y_pillar, np.asarray(new_y_pillar)])
        print(f"Pillar data:    {len(y_pillar) - len(new_y_pillar)} → {len(y_pillar)} samples")

    # Retrain
    model = train_model(
        X_syn, y_syn, X_pillar, y_pillar,
        **(train_kwargs or {}),
    )

    # Overwrite checkpoint atomically
    save_checkpoint(
        model, X_syn, y_syn, X_pillar, y_pillar,
        name=name, directory=directory,
        metadata={"n_syn": int(len(y_syn)), "n_pillar": int(len(y_pillar))},
    )

    return model


# ---------------------------------------------------------------------------
# Single-point inference
# ---------------------------------------------------------------------------

def predict_single(
    source,
    feature_method: str = "avg",
    name: str = "model",
    directory: str | Path = DEFAULT_DIR,
    model: RiskMLP = None,
    p_safe: float = 0.25,
    p_risk: float = 0.75,
) -> dict:
    """
    Run inference on a single input and return a detailed result dict.

    *source* can be:

    * a **file path** (str / Path) → the image is loaded, segmented, and
      features are extracted with *feature_method*;
    * a **numpy array of shape (H, W, 3)** → treated as an already-loaded
      RGB image and processed the same way;
    * a **1-D numpy array of shape (d,)** → used directly as the feature
      vector (segmentation is skipped).

    Parameters
    ----------
    source : str, Path, or ndarray
    feature_method : {'avg', 'gauss', 'cluster', 'conv'}
        Used only when *source* is an image.
    name : str
        Checkpoint name to load when *model* is None.
    directory : str or Path
        Checkpoint directory to load from when *model* is None.
    model : RiskMLP or None
        Pass an already-loaded model to skip disk I/O.
    p_safe : float
        Probability threshold below which the sample is called "safe".
    p_risk : float
        Probability threshold above which the sample is called "at risk".

    Returns
    -------
    result : dict with keys
        ``'probability'``  – float in [0, 1]
        ``'label'``        – int in {0, 1, 2}  (0=safe, 1=unsure, 2=at_risk)
        ``'label_str'``    – human-readable string
        ``'feature'``      – the feature vector actually fed to the model
    """
    # --- load model if not provided ---
    if model is None:
        model = load_model(name=name, directory=directory)

    # --- resolve feature vector ---
    source = np.asarray(source) if not isinstance(source, (str, Path)) else source

    if isinstance(source, (str, Path)):
        img_rgb = load_image_rgb(str(source))
        img_lab, big_mask, _ = segment_image(img_rgb)
        raw = extract_lab_feature(img_lab, big_mask, method=feature_method)
        feature = raw["features"].mean(axis=0) if isinstance(raw, dict) else raw

    elif isinstance(source, np.ndarray) and source.ndim == 3:
        # Already an RGB image array
        img_lab, big_mask, _ = segment_image(source)
        raw = extract_lab_feature(img_lab, big_mask, method=feature_method)
        feature = raw["features"].mean(axis=0) if isinstance(raw, dict) else raw

    elif isinstance(source, np.ndarray) and source.ndim == 1:
        # Raw feature vector — use directly
        feature = source

    else:
        raise TypeError(
            "source must be a file path, an (H, W, 3) RGB array, "
            "or a 1-D feature vector."
        )

    # --- inference ---
    prob = float(predict_probability(model, feature[None, :])[0])

    if prob < p_safe:
        label, label_str = 0, "safe"
    elif prob > p_risk:
        label, label_str = 2, "at_risk"
    else:
        label, label_str = 1, "unsure"

    return {
        "probability": prob,
        "label": label,
        "label_str": label_str,
        "feature": feature,
    }
