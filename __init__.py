"""
fecal_risk
==========
Image-based risk classification pipeline.

Submodules
----------
imaging   – image loading, LAB masking, segmentation
features  – LAB colour feature extraction
model     – RiskMLP architecture, training, prediction
viz       – matplotlib / plotly visualisation helpers
"""

from .imaging import (
    load_image_rgb,
    mask_lab,
    fill_mask_poly,
    fill_mask_bin,
    fill_mask_hull,
    contour_largest,
    segment_image,
    pixelate,
)
from .features import extract_lab_feature
from .model import (
    RiskMLP,
    pillar_margin_loss,
    total_loss,
    train_model,
    predict_probability,
    predict_binary,
    predict_three_state,
)
from .persistence import (
    save_model,
    load_model,
    save_data,
    load_data,
    save_checkpoint,
    update_model,
    predict_single,
)
from .viz import (
    plot_decision_boundary_2d,
    plot_decision_boundary_and_3d,
    plot_interactive_3d,
)

__all__ = [
    # imaging
    "load_image_rgb",
    "mask_lab",
    "fill_mask_poly",
    "fill_mask_bin",
    "fill_mask_hull",
    "contour_largest",
    "segment_image",
    "pixelate",
    # features
    "extract_lab_feature",
    # model
    "RiskMLP",
    "pillar_margin_loss",
    "total_loss",
    "train_model",
    "predict_probability",
    "predict_binary",
    "predict_three_state",
    # persistence
    "save_model",
    "load_model",
    "save_data",
    "load_data",
    "save_checkpoint",
    "update_model",
    "predict_single",
    # viz
    "plot_decision_boundary_2d",
    "plot_decision_boundary_and_3d",
    "plot_interactive_3d",
]
