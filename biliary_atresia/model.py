"""
model.py
--------
RiskMLP: a small fully-connected network that classifies samples as
safe / at-risk, trained with a combined BCE + pillar-margin loss.
"""

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class RiskMLP(nn.Module):
    """Three-layer MLP for binary risk classification."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

_bce_loss = nn.BCEWithLogitsLoss()


def pillar_margin_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 2.0,
) -> torch.Tensor:
    """
    Squared hinge loss that pushes *anchor* (pillar) samples at least
    *margin* units away from the decision boundary.

    Parameters
    ----------
    logits : (N,) tensor
    labels : (N,) tensor with values in {0, 1}
    margin : float
    """
    labels_pm = 2.0 * labels.float() - 1.0          # map {0,1} → {-1,+1}
    signed_margin = labels_pm * logits
    return (torch.relu(margin - signed_margin) ** 2).mean()


def total_loss(
    model: RiskMLP,
    X_syn: torch.Tensor,
    y_syn: torch.Tensor,
    X_pillar: torch.Tensor,
    y_pillar: torch.Tensor,
    lam: float = 1.0,
    margin: float = 2.0,
) -> torch.Tensor:
    """
    Combined loss: BCE on synthetic data + weighted pillar margin loss.

    Parameters
    ----------
    model : RiskMLP
    X_syn, y_syn : synthetic training batch
    X_pillar, y_pillar : anchor (pillar) samples
    lam : float
        Weight of the pillar loss term.
    margin : float
        Minimum desired distance from the decision boundary for pillar samples.
    """
    loss_syn = _bce_loss(model(X_syn), y_syn.float())

    if len(X_pillar) > 0:
        loss_pillar = pillar_margin_loss(model(X_pillar), y_pillar, margin=margin)
    else:
        loss_pillar = torch.tensor(0.0, device=X_syn.device)

    return loss_syn + lam * loss_pillar


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    X_syn,
    y_syn,
    X_pillar,
    y_pillar,
    hidden_epochs: int = 500,
    lr: float = 1e-3,
    lam: float = 2.0,
    margin: float = 2.0,
    verbose: bool = True,
) -> RiskMLP:
    """
    Train a :class:`RiskMLP` on synthetic data anchored by pillar samples.

    Parameters
    ----------
    X_syn : array-like, shape (M, d)
    y_syn : array-like, shape (M,)  – labels in {0, 1}
    X_pillar : array-like, shape (N, d)
    y_pillar : array-like, shape (N,) – labels in {0, 1}
    hidden_epochs : int
    lr : float
        Adam learning rate.
    lam : float
        Pillar loss weight.
    margin : float
        Pillar margin (see :func:`pillar_margin_loss`).
    verbose : bool
        Print loss every 50 epochs.

    Returns
    -------
    model : RiskMLP  (in eval mode, on the training device)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def _to_tensor(arr, dtype=torch.float32):
        return torch.tensor(np.asarray(arr), dtype=dtype, device=device)

    X_syn_t    = _to_tensor(X_syn)
    y_syn_t    = _to_tensor(y_syn)
    X_pillar_t = _to_tensor(X_pillar)
    y_pillar_t = _to_tensor(y_pillar)

    model = RiskMLP(input_dim=X_syn_t.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(hidden_epochs):
        optimizer.zero_grad()
        loss = total_loss(
            model, X_syn_t, y_syn_t, X_pillar_t, y_pillar_t,
            lam=lam, margin=margin,
        )
        loss.backward()
        optimizer.step()

        if verbose and epoch % 50 == 0:
            print(f"Epoch {epoch:4d}  Loss = {loss.item():.5f}")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_probability(model: RiskMLP, X) -> np.ndarray:
    """
    Return sigmoid probabilities for each sample in *X*.

    Parameters
    ----------
    model : RiskMLP
    X : array-like, shape (N, d) or (d,) for a single sample

    Returns
    -------
    probs : (N,) float32 ndarray
    """
    device = next(model.parameters()).device
    X_arr = np.asarray(X)
    if X_arr.ndim == 1:
        X_arr = X_arr[None, :]      # single sample → batch of 1
    X_t = torch.tensor(X_arr, dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_t))
    return probs.cpu().numpy()


def predict_binary(model: RiskMLP, X, threshold: float = 0.5) -> np.ndarray:
    """
    Return binary predictions {0, 1} using *threshold*.

    Parameters
    ----------
    model : RiskMLP
    X : array-like, shape (N, d)
    threshold : float

    Returns
    -------
    predictions : (N,) int ndarray
    """
    return (predict_probability(model, X) >= threshold).astype(int)


def predict_three_state(
    model: RiskMLP,
    X,
    p_safe: float = 0.25,
    p_risk: float = 0.75,
) -> np.ndarray:
    """
    Return a three-class prediction.

    Classes
    -------
    0 – safe        (probability < *p_safe*)
    1 – unsure      (*p_safe* ≤ probability ≤ *p_risk*)
    2 – at risk     (probability > *p_risk*)

    Parameters
    ----------
    model : RiskMLP
    X : array-like, shape (N, d)
    p_safe, p_risk : float

    Returns
    -------
    predictions : (N,) int ndarray
    """
    probs = predict_probability(model, X)
    pred = np.ones(len(probs), dtype=int)   # default: unsure
    pred[probs < p_safe] = 0
    pred[probs > p_risk] = 2
    return pred
