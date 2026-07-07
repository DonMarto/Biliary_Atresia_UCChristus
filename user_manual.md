# Integration Report: Wiring `biliary_atresia` into a Mobile App

**Goal:** a mobile app captures/selects an image, the image is run through this
pipeline, and the app receives back a number in `[0, 1]` (or a category) that it can
interpret and display to the user.

This report covers, file by file, what to actually call, what to send/expect at each
boundary, and the edge cases you need to guard against before this goes into a
production app.

---

## 1. Recommended architecture: server-side inference, thin mobile client

The pipeline depends on OpenCV, SciPy, scikit-learn, and PyTorch. None of these port
cleanly to iOS/Android without a large native-porting effort (and PyTorch Mobile
doesn't help with the OpenCV/SciPy segmentation code, only the model itself). The
practical path is:

```
[Mobile App] --(image, JPEG/PNG)--> [Backend service running biliary_atresia] --(JSON: probability, label)--> [Mobile App]
```

The mobile app's job is: capture/select an image, upload it, display the result.
The backend's job is everything in this package. This is the pattern the rest of this
report assumes. (An on-device alternative is discussed in §7 for completeness, but
it's substantially more work and is not recommended as a first version.)

---

## 2. The one function your backend endpoint should call

Everything funnels through `persistence.predict_single`:

```python
from biliary_atresia import load_model
from biliary_atresia.persistence import predict_single

# Load once at process startup, not per-request
model = load_model(name="risk_model_v1", directory="checkpoints")

def handle_upload(image_path: str) -> dict:
    result = predict_single(
        image_path,
        feature_method="avg",   # see §4 — do not use "cluster" here
        model=model,            # reuse the loaded model; skips disk I/O per request
        p_safe=0.25,
        p_risk=0.75,
    )
    return {
        "probability": result["probability"],   # float in [0, 1] — send this to the app
        "label": result["label"],                 # 0 / 1 / 2
        "label_str": result["label_str"],          # "safe" / "unsure" / "at_risk"
    }
```

A minimal REST wrapper (FastAPI, Flask, whatever you're already using) around
`handle_upload` is the entire backend surface the mobile app needs to talk to. Suggested
response contract:

```json
{
  "probability": 0.83,
  "label": 2,
  "label_str": "at_risk"
}
```

The mobile app can then either:
- show `probability` directly as a 0–1 (or 0–100%) risk score, or
- show `label_str` as a friendlier three-state result ("looks fine" / "not sure,
  recheck" / "flagged"), which is usually the better UX for a non-technical user than
  a raw probability.

---

## 3. What each file means for the mobile integration

### `imaging.py` — do not expose this to the client; keep it server-side
`segment_image` is invoked automatically inside `predict_single`, you don't call it
directly. What matters for the mobile app is **capture quality**, since segmentation
quality depends entirely on it:

- Consistent, diffuse lighting (avoid hard shadows / glare — Otsu thresholding on LAB
  channels is sensitive to lighting shifts).
- A plain, consistent background color that contrasts with the subject (segmentation
  assumes the subject is separable via `a`/`b` channel thresholds).
- The app should require photos to be reasonably in-focus and fill a minimum fraction
  of the frame — see §5 for why this matters for a specific failure mode.

If you control the capture flow, adding an in-app camera overlay (guide frame + a
lighting/blur check before upload) will do more for real-world accuracy than any
change to the model itself.

### `features.py` — you don't call this directly either, but know the constraint
`predict_single` always ends up needing a plain `(3,)` LAB vector to feed the model.
Of the four methods, only `avg`, `gauss`, and `conv` reduce down to that; `cluster`
does not (see §4). **Use `feature_method="avg"` for all live single-image inference
calls** — it's the fastest, simplest, and the one this pipeline is implicitly designed
around for single images. `conv` and `cluster` are training-time tools, not
inference-time tools.

### `model.py` — this is what actually ships / gets versioned
`RiskMLP` is a tiny 3→64→32→1 network — inference is sub-millisecond on a CPU, so
compute is not the bottleneck anywhere in this system; image capture and network
transfer will dominate latency. This also means you don't need a GPU on the backend.

### `persistence.py` — your checkpoint is your deployable artifact
A checkpoint is three files: `<name>.pt` (weights), `<name>_meta.json` (input dim +
metadata), `<name>_data.npz` (the training data that produced it). Treat a checkpoint
directory like you'd treat a mobile app build:

- **Version it.** Use `name` like `risk_model_v3` rather than overwriting `model`
  every time, so you can roll back a bad release without retraining.
- **Record what went into it.** Pass a meaningful `metadata` dict to `save_checkpoint`
  / `update_model` (e.g. `{"trained_on": "2026-07-07", "n_syn": ..., "n_pillar": ...,
  "app_version_min": "1.4.0"}`) so a specific checkpoint's provenance is always
  recoverable from the sidecar JSON.
- **`update_model` is your retraining pipeline**, not something to call from a live
  request handler. Run it offline/on a schedule as new labeled data accumulates, then
  promote the new checkpoint the same way you'd promote any new build — ideally
  behind a staging/canary step, since it retrains from scratch each time and could in
  principle produce a worse model than the one it's replacing.

### `viz.py` — internal tooling only
Useful for you when validating a new checkpoint before you ship it (plot the decision
boundary, sanity-check the pillar points are on the correct side of it), but has no
place in the request path or the app itself.

---

## 4. Known bug: `predict_single` breaks on `feature_method="cluster"`

`extract_lab_feature(..., method="cluster")` returns a dict with keys `K`,
`centroids`, `proportions`. `predict_single` reduces dict outputs to a vector via
`raw["features"].mean(axis=0)` — but `"features"` is a key that only the `conv`
output has. Calling `predict_single(img, feature_method="cluster")` will raise a
`KeyError`, not fail gracefully.

**Fix before shipping:** either don't expose `"cluster"` as a valid option for
`feature_method` in your API layer at all (simplest — just hardcode `"avg"` server-side
and don't take this as a client parameter), or patch `predict_single` to branch on the
method name explicitly rather than assuming all dict outputs share the same shape.

## 5. Known bug: tiny segmented regions can silently produce `NaN` results

In `extract_lab_feature(..., method="conv")`, a patch is only kept if it has at least
20 unmasked pixels — a hardcoded threshold, regardless of the computed `window` size.
For a small segmented region, `window` itself can come out to 3×3 = 9 pixels, which
can never clear the 20-pixel bar. Result: zero patches, and
`raw["features"].mean(axis=0)` on an empty array returns `[nan, nan, nan]` **without
raising an exception**.

Because `predict_probability`/`predict_single`'s three-state logic is
`< p_safe` / `> p_risk` / else, a `NaN` probability satisfies neither comparison and
silently lands in `"unsure"` — i.e. a failed feature extraction can present to the app
as an ordinary "unsure" result instead of an error the app can react to.

This specific failure mode only affects `conv` at training time in this codebase, but
if you ever use `conv` server-side for anything image-derived at request time, add an
explicit check:

```python
feat = raw["features"]
if len(feat) == 0 or np.isnan(feat).any():
    raise ValueError("Feature extraction failed — region too small or empty mask")
```

More generally: **have your endpoint check `math.isnan(probability)` before returning
a response**, and return an explicit error/"couldn't analyze this photo, please
retake it" state to the app rather than a silently-generated `"unsure"`.

## 6. Known bug: segmentation can silently pick the background

`contour_largest` zeroes out the background label's pixel count before taking an
`argmax` over connected-component sizes — but if segmentation found *no* foreground at
all (a single, all-background component), the zeroed-out background label is still
the only entry, so `argmax` selects it anyway. The result is a "big_mask" that's
actually most of the frame, with no error raised.

**Recommendation:** before extracting features, sanity-check the mask coming back
from `segment_image`:

```python
img_lab, big_mask, _ = segment_image(img_rgb)
foreground_fraction = (big_mask > 0).mean()
if foreground_fraction < 0.01 or foreground_fraction > 0.9:
    # Segmentation almost certainly failed — reject rather than run inference on garbage
    raise ValueError("Could not isolate a subject in this photo")
```

Surface this back to the app as a "please retake the photo — couldn't clearly see the
subject" message rather than a risk score. This is the single highest-value guard to
add before shipping, since a bad photo is the most likely real-world failure mode a
mobile user will trigger.

---

## 7. If you do need on-device inference later

If network dependency becomes a problem (offline use, privacy requirements, latency),
the model itself — `RiskMLP` — is small enough to export to TorchScript or ONNX and
run with PyTorch Mobile / ONNX Runtime Mobile / Core ML (via `onnx-coreml` or
`coremltools`). That gets you on-device *inference*, but you would still need to
reimplement `imaging.segment_image` and `features.extract_lab_feature("avg")` in
Swift/Kotlin (or ship a compiled OpenCV for mobile) since those aren't part of the
exported model graph. This is a meaningfully larger effort than the server-side
approach and is only worth it once the server-side version is validated and the
product actually needs offline capability.

Minimal export example, for when that day comes:

```python
import torch
from biliary_atresia import load_model

model = load_model(name="risk_model_v1", directory="checkpoints")
example_input = torch.zeros(1, 3)  # (batch, LAB features)
traced = torch.jit.trace(model, example_input)
traced.save("risk_model_v1.pt")  # load this with PyTorch Mobile / LibTorch
```

Remember the exported graph only takes a `(N, 3)` LAB feature tensor — segmentation
and feature extraction happen *before* this point and are not included.

---

## 8. Summary checklist before shipping the mobile integration

- [ ] Backend endpoint calls `predict_single(..., feature_method="avg")` only —
      never `"cluster"`, and don't take feature method as a client-controlled parameter.
- [ ] Endpoint validates the segmented mask isn't empty/near-total before running
      inference (§6), and rejects with a clear "retake photo" error if it is.
- [ ] Endpoint checks the returned probability isn't `NaN` before responding (§5).
- [ ] Checkpoints are named/versioned per release, with `metadata` recording
      provenance; `update_model` runs offline as part of a retraining pipeline, not
      inline with user requests.
- [ ] Mobile app enforces basic capture quality (lighting, framing, focus) before
      upload — this reduces the rate at which the above edge cases get hit at all.
- [ ] Response contract to the app is the small JSON shown in §2 — probability +
      label + label_str, nothing else needs to cross the wire.
