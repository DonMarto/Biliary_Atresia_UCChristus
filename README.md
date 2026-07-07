# Biliary Atresia

Pending Translation

Image-based risk classification pipeline: take an RGB image, isolate the region of
interest in LAB colour space, turn that region into a small colour feature vector, and
feed it to a lightweight neural classifier that outputs a risk probability (and, if
you want it, a safe / unsure / at‑risk category).

The package is deliberately small and linear — each module does one stage of the
pipeline, and `predict_single()` in `persistence.py` chains all of them together for
you. That function is the one entry point most integrations (including a mobile app
backend) should call.

```
image  →  imaging.py  →  features.py  →  model.py  →  probability / category
                                ↑
                          persistence.py (save/load/update, wraps it all together)
                                ↑
                             viz.py (inspect/debug what the model learned)
```

## Modules

### `imaging.py` — loading & segmentation
Loads images and finds the region of interest.

- `load_image_rgb(path)` — reads a file into an RGB array (OpenCV loads BGR by default,
  this converts it).
- `mask_lab(img_rgb, ...)` — manual LAB-channel thresholding, for when you want to hand-tune
  per-channel cutoffs instead of using automatic (Otsu) thresholding.
- `fill_mask_poly` / `fill_mask_bin` / `fill_mask_hull` — three different strategies for
  closing holes in a raw mask (polygon fill, morphological close + flood fill, convex hull).
- `contour_largest(mask)` — keeps only the largest connected blob in a mask and also returns
  its edge/border map.
- `segment_image(img_rgb)` — the standard pipeline: Otsu-threshold the *a* and *b* LAB
  channels, AND them together, fill holes, keep the largest component. Returns the LAB image,
  the final mask, and the border map. **This is the function everything else builds on.**
- `pixelate(img, factor)` — a downscale/upscale blur, used for visualization or
  privacy-masking, not part of the inference path.

### `features.py` — turning a masked region into numbers
`extract_lab_feature(lab_image, mask, method=...)` converts the pixels selected by a mask
into a feature representation. Four interchangeable strategies:

| method    | returns                              | when to use it |
|-----------|---------------------------------------|----------------|
| `avg`     | `(3,)` mean LAB colour                | fast, simple, what single-image inference uses |
| `gauss`   | `(3,)` centre-weighted mean LAB       | de-emphasizes noisy edge pixels |
| `cluster` | dict of `K` dominant colour clusters  | exploratory analysis of colour composition |
| `conv`    | dict of per-patch mean LAB features   | generating many training samples from one image |

`avg` and `gauss` always return a plain `(3,)` vector, so they can be fed straight into
the model. `cluster` and `conv` return richer dict structures meant for training-time
data augmentation or exploratory analysis — see the integration report for exactly how
`conv` output gets reduced back down to a `(3,)` vector before hitting the model.

### `model.py` — the classifier
- `RiskMLP` — a plain 3‑layer fully-connected network (`3 → 64 → 32 → 1` for LAB features),
  outputting a single logit.
- `pillar_margin_loss` — a squared-hinge loss that pushes a small set of trusted "pillar"
  (anchor) samples at least `margin` units from the decision boundary, so the model doesn't
  drift away from known-correct reference points as it trains on bulk/synthetic data.
- `total_loss` — BCE on the main dataset + weighted pillar-margin loss on the anchors.
- `train_model(...)` — full training loop (Adam optimizer, prints loss periodically), returns
  a model in `eval()` mode.
- `predict_probability` / `predict_binary` / `predict_three_state` — inference at three
  levels of granularity: raw sigmoid probability, thresholded 0/1, or a 3-way
  safe / unsure / at-risk label.

### `persistence.py` — save, load, update, and single-shot inference
This is the module a service or app integration will interact with most.

- `save_model` / `load_model` — weights + a small JSON sidecar (input dimension, timestamp,
  any metadata you attach).
- `save_data` / `load_data` — the exact training arrays (`X_syn`, `y_syn`, `X_pillar`,
  `y_pillar`) that produced a given checkpoint.
- `save_checkpoint` — does both of the above together, atomically, under one `name`.
- `update_model(...)` — appends newly-labeled samples to a stored checkpoint's dataset and
  retrains from scratch, then overwrites the checkpoint. This is the mechanism for
  incorporating new field data over time.
- `predict_single(source, ...)` — the all-in-one entry point: give it a file path, an
  already-loaded RGB array, or a raw feature vector, and it runs (as needed) segmentation,
  feature extraction, and inference, returning a dict with `probability`, `label`,
  `label_str`, and the `feature` vector actually used. **This is the function a mobile-app
  backend should call.**

### `viz.py` — inspecting what the model learned
- `plot_decision_boundary_2d` — 2-D slice of the 3-D decision boundary at a fixed value of
  the third feature.
- `plot_decision_boundary_and_3d` — that 2-D plot next to a 3-D scatter of the raw data.
- `plot_interactive_3d` — a rotatable Plotly 3-D scatter, with pillar/anchor points shown
  as gold/black diamonds.

These are for debugging and presentations, not part of the inference path.

## Typical end-to-end flow

1. Extract training features image-by-image with `extract_lab_feature(..., method="conv")`
   to get many samples per image, plus a small hand-picked pillar set.
2. `train_model(...)` on that data.
3. `save_checkpoint(...)` to persist it.
4. At inference time, call `predict_single(image_or_path, feature_method="avg")` and use
   `result["probability"]` or `result["label_str"]`.
5. As new labeled data comes in, call `update_model(...)` to retrain and overwrite the
   checkpoint.

See `tutorial.ipynb` for a runnable walkthrough of all of the above, and the integration
report for details on wiring `predict_single` into a mobile application, plus a list of
edge cases worth guarding against before shipping.

## A few things worth knowing before you rely on this in production

- `predict_single` only correctly supports the `avg`, `gauss`, and `conv` feature methods.
  Calling it with `feature_method="cluster"` will raise a `KeyError`, because `cluster`'s
  return dict doesn't have a `"features"` key (see `features.py`). Stick to `avg` for
  single-image inference.
- If the segmented region is very small, the `conv` method can return zero patches, which
  silently produces a `NaN` feature vector (not an exception) if nothing guards against it.
  See the integration report for a recommended fallback.
- If segmentation finds *no* foreground pixels at all, `contour_largest` can end up
  selecting the background as the "largest component," rather than failing loudly.
  Worth an explicit sanity check (e.g. mask non-empty, mask not the whole frame) before
  trusting a segmentation result.

Full detail on all three, plus suggested fixes, is in the integration report.
