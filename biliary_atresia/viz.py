"""
viz.py
------
Plotting utilities for decision boundaries and data exploration.
"""

import numpy as np
import matplotlib.pyplot as plt

from .model import RiskMLP, predict_probability


def plot_decision_boundary_2d(
    model: RiskMLP,
    X,
    y,
    z_fixed: float = None,
    title: str = None,
    ax=None,
) -> None:
    """
    Plot the 2-D decision boundary of *model* by slicing the third feature
    dimension at *z_fixed* (defaults to the column mean).

    Parameters
    ----------
    model : RiskMLP
    X : array-like, shape (N, 3)
    y : array-like, shape (N,)
    z_fixed : float or None
    title : str or None
    ax : matplotlib Axes or None
    """
    X = np.asarray(X)
    y = np.asarray(y)
    X_2d = X[:, :2]

    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200),
    )

    if z_fixed is None:
        z_fixed = float(X[:, 2].mean())

    grid_3d = np.c_[xx.ravel(), yy.ravel(), np.full(xx.size, z_fixed)]
    probs = predict_probability(model, grid_3d).reshape(xx.shape)

    own_fig = ax is None
    if own_fig:
        _, ax = plt.subplots(figsize=(7, 6))

    ax.contourf(xx, yy, probs, levels=50, cmap="RdBu", alpha=0.6)
    ax.contour(xx, yy, probs, levels=[0.5], colors="black")
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="RdBu", edgecolors="k")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title(title or f"Decision boundary (z = {z_fixed:.2f})")

    if own_fig:
        plt.show()


def plot_decision_boundary_and_3d(model: RiskMLP, X, y) -> None:
    """
    Side-by-side figure: 2-D decision boundary (left) and 3-D scatter (right).

    Parameters
    ----------
    model : RiskMLP
    X : array-like, shape (N, 3)
    y : array-like, shape (N,)
    """
    try:
        X = np.vstack(X)
    except Exception:
        X = np.array(X)
    y = np.array(y)

    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(1, 2, 1)
    plot_decision_boundary_2d(model, X, y, ax=ax1)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    probs_points = predict_probability(model, X)
    ax2.scatter(X[:, 0], X[:, 1], X[:, 2], c=probs_points, cmap="RdBu", edgecolors="k")
    ax2.set_xlabel("Feature 1")
    ax2.set_ylabel("Feature 2")
    ax2.set_zlabel("Feature 3")
    ax2.set_title("3D data")

    plt.tight_layout()
    plt.show()


def plot_interactive_3d(model: RiskMLP, X, y, X_pillar=None, y_pillar=None):
    """
    Interactive Plotly 3-D scatter.  Pillar (anchor) points are shown as
    gold/black diamonds if *X_pillar* is provided.

    Parameters
    ----------
    model : RiskMLP
    X : array-like, shape (N, 3)
    y : array-like, shape (N,)
    X_pillar : array-like or None, shape (P, 3)
    y_pillar : array-like or None, shape (P,)
    """
    import plotly.graph_objects as go

    X = np.vstack(X) if not isinstance(X, np.ndarray) else X
    probs = predict_probability(model, X)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=X[:, 0], y=X[:, 1], z=X[:, 2],
            mode="markers",
            marker=dict(size=4, color=probs, colorscale="RdBu", opacity=0.7),
            name="Data",
        )
    )

    if X_pillar is not None:
        X_p = np.vstack(X_pillar)
        y_p = np.asarray(y_pillar) if y_pillar is not None else None
        pillar_colors = (
            ["gold" if label == 0 else "black" for label in y_p]
            if y_p is not None
            else "purple"
        )
        fig.add_trace(
            go.Scatter3d(
                x=X_p[:, 0], y=X_p[:, 1], z=X_p[:, 2],
                mode="markers",
                marker=dict(
                    size=7,
                    color=pillar_colors,
                    symbol="diamond",
                    line=dict(width=2, color="black"),
                ),
                name="Pillar points",
            )
        )

    fig.update_layout(
        title="3D interactive plot",
        scene=dict(
            xaxis_title="Feature 1",
            yaxis_title="Feature 2",
            zaxis_title="Feature 3",
        ),
    )
    fig.show()
