"""
features.py
-----------
LAB colour feature extraction from segmented image regions.
"""

import numpy as np
from sklearn.cluster import KMeans


def extract_lab_feature(
    lab_image: np.ndarray,
    mask: np.ndarray,
    method: str = "avg",
    max_k: int = 10,
    min_prop: float = 0.05,
    random_state: int = 0,
):
    """
    Extract a colour feature vector (or structure) from *lab_image* at the
    pixels selected by *mask*.

    Parameters
    ----------
    lab_image : (H, W, 3) ndarray
        LAB image.
    mask : (H, W) ndarray of bool or uint8
        Region-of-interest mask (non-zero pixels are used).
    method : {'avg', 'gauss', 'cluster', 'conv'}
        Feature extraction strategy (see *Returns* for details).
    max_k : int
        Maximum number of K-Means clusters to try (``method='cluster'`` only).
    min_prop : float
        Minimum proportion a cluster must represent to be kept
        (``method='cluster'`` only).
    random_state : int
        Random seed passed to K-Means.

    Returns
    -------
    feature : ndarray or dict

        ``avg``
            shape ``(3,)`` – per-pixel mean in LAB space.

        ``gauss``
            shape ``(3,)`` – Mahalanobis-weighted mean; pixels closer to the
            spatial centre of mass receive higher weight.

        ``cluster``
            dict with keys:

            * ``'K'``           – number of clusters selected
            * ``'centroids'``   – ``(K, 3)`` cluster centres, sorted by
                                  descending proportion
            * ``'proportions'`` – ``(K,)`` fractional sizes

        ``conv``
            dict with keys:

            * ``'features'``  – ``(P, 3)`` patch-level mean colours
            * ``'centers'``   – ``(P, 2)`` (x, y) centre of each patch
            * ``'window'``    – patch side length in pixels
            * ``'stride'``    – stride used between patches
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("Mask contains no pixels.")

    lab_pixels = lab_image[ys, xs].astype(float)

    # ------------------------------------------------------------------
    # Simple average
    # ------------------------------------------------------------------
    if method == "avg":
        return lab_pixels.mean(axis=0)

    # ------------------------------------------------------------------
    # Gaussian (Mahalanobis) weighted average
    # ------------------------------------------------------------------
    elif method == "gauss":
        coords = np.column_stack([xs, ys]).astype(float)
        mu = coords.mean(axis=0)
        cov = np.cov(coords.T) + 1e-6 * np.eye(2)
        d = coords - mu
        inv_cov = np.linalg.inv(cov)
        mahal = np.sum((d @ inv_cov) * d, axis=1)
        weights = np.exp(-0.5 * mahal)
        weights /= weights.sum()
        return np.sum(weights[:, None] * lab_pixels, axis=0)

    # ------------------------------------------------------------------
    # K-Means colour clusters
    # ------------------------------------------------------------------
    elif method == "cluster":
        N = len(lab_pixels)
        selected_k = 1
        selected_labels = np.zeros(N, dtype=int)
        selected_centers = lab_pixels.mean(axis=0)[None, :]

        for K in range(max_k, 0, -1):
            km = KMeans(n_clusters=K, n_init=20, random_state=random_state)
            labels = km.fit_predict(lab_pixels)
            counts = np.bincount(labels)
            props = counts / N
            if np.all(props >= min_prop):
                selected_k = K
                selected_labels = labels
                selected_centers = km.cluster_centers_
                break

        counts = np.bincount(selected_labels, minlength=selected_k)
        props = counts / N
        order = np.argsort(props)[::-1]
        return {
            "K": selected_k,
            "centroids": selected_centers[order],
            "proportions": props[order],
        }

    # ------------------------------------------------------------------
    # Sliding-window (convolutional) patch features
    # ------------------------------------------------------------------
    elif method == "conv":
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()

        D = max(xmax - xmin + 1, ymax - ymin + 1)
        window = max(3, int(0.1 * D))
        stride = max(1, int(0.95 * window))

        features, centers = [], []

        for y0 in range(ymin, ymax - window + 2, stride):
            for x0 in range(xmin, xmax - window + 2, stride):
                patch_mask = mask[y0 : y0 + window, x0 : x0 + window].astype(bool)
                if patch_mask.sum() < 20:
                    continue
                patch_lab = lab_image[y0 : y0 + window, x0 : x0 + window]
                features.append(patch_lab[patch_mask].mean(axis=0))
                centers.append([x0 + window / 2, y0 + window / 2])

        return {
            "features": np.asarray(features),
            "centers": np.asarray(centers),
            "window": window,
            "stride": stride,
        }

    else:
        raise ValueError("method must be 'avg', 'gauss', 'cluster', or 'conv'")
