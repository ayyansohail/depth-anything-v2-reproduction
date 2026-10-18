"""Image validation and serialized prediction service; importing loads no model."""
from __future__ import annotations

import gc
import hashlib
import io
import json
import logging
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MODEL_REVISIONS = {
    "v2-small": "5426e4f0f36572d16453bbda7a8389317b1bef99",
    "v2-base": "b1958afc87fb45a9e3746cb387596094de553ed8",
    "v2-large": "7581137eff8d4e94f6e796d3baea0e9fa79b22d2",
}
LABELS = {"v2-small": "Small (24.8M)", "v2-base": "Base (97.5M)", "v2-large": "Large (335.3M)"}
DEFAULT_LIMITS = {"max_upload_bytes": 10 * 1024 * 1024, "max_pixels": 16_000_000,
                  "max_edge": 1024, "max_aspect_ratio": 2.0}
LOGGER = logging.getLogger("dav2.deployment")


class InputError(ValueError):
    """A readable upload-validation failure."""


class PredictionError(RuntimeError):
    """A readable inference failure; detailed cause is in server logs."""


@dataclass
class PreparedImage:
    image: Image.Image
    original_size: tuple[int, int]
    input_hash: str
    resized: bool


def load_config(path=None):
    path = Path(path) if path else Path(__file__).with_name("deployment.json")
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    enabled = config.get("enabled_models", [])
    if (not enabled or enabled[0] != "v2-small" or len(enabled) != len(set(enabled))
            or any(key not in MODEL_REVISIONS for key in enabled)):
        raise ValueError("Enabled models must start with Small and contain only unique V2 model keys.")
    limits = DEFAULT_LIMITS | config.get("limits", {})
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0
           for value in limits.values()):
        raise ValueError("Image limits must be positive numbers.")
    for key in ("max_upload_bytes", "max_pixels", "max_edge"):
        if not isinstance(limits[key], int):
            raise ValueError(f"{key} must be an integer.")
    if limits["max_aspect_ratio"] < 1:
        raise ValueError("Aspect ratio limit cannot be below one.")
    return enabled, limits


def prepare_image(content: bytes, filename: str, limits=None) -> PreparedImage:
    limits = DEFAULT_LIMITS | (limits or {})
    if not content:
        raise InputError("Choose a JPG or PNG image first.")
    if len(content) > limits["max_upload_bytes"]:
        raise InputError("The image exceeds the 10 MB upload limit. Please upload a smaller file.")
    if Path(filename).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise InputError("Upload a JPG, JPEG, or PNG image.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as probe:
                if probe.format not in {"JPEG", "PNG"}:
                    raise InputError("The file contents must be a JPEG or PNG image.")
                width, height = probe.size
                if width <= 0 or height <= 0 or width * height > limits["max_pixels"]:
                    raise InputError("The image exceeds 16 megapixels. Please resize it first.")
                if max(width, height) / min(width, height) > limits["max_aspect_ratio"]:
                    raise InputError("Use an image with an aspect ratio no wider or taller than 2:1. Crop panoramic images first.")
                if getattr(probe, "n_frames", 1) != 1:
                    raise InputError("Animated images are not supported. Upload one still image.")
                probe.verify()
            with Image.open(io.BytesIO(content)) as source:
                image = ImageOps.exif_transpose(source)
                original_size = image.size
                # Composite transparency explicitly; hidden RGB values must not become black artifacts.
                if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
                    rgba = image.convert("RGBA")
                    background = Image.new("RGBA", rgba.size, "white")
                    image = Image.alpha_composite(background, rgba).convert("RGB")
                else:
                    image = image.convert("RGB")
                image.thumbnail((limits["max_edge"], limits["max_edge"]), Image.Resampling.LANCZOS)
                image = image.copy()
    except InputError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError,
            Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise InputError("This image could not be read. Try a different JPG or PNG file.") from error
    return PreparedImage(image, original_size, hashlib.sha256(content).hexdigest(), image.size != original_size)


def png_bytes(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def encode_prediction(disparity, size):
    values = np.asarray(disparity, dtype=np.float32)
    if values.shape != (size[1], size[0]) or not np.isfinite(values).all():
        raise PredictionError("The model returned invalid output. Please try again or choose another image.")
    low, high = float(values.min()), float(values.max())
    span = high - low
    if not np.isfinite(span):
        raise PredictionError("The model returned an invalid depth range.")
    normalized = np.zeros_like(values) if span <= 1e-8 else (values - low) / span
    normalized = np.clip(normalized, 0, 1)
    from matplotlib import colormaps
    colour = (colormaps["magma"](normalized)[..., :3] * 255).astype(np.uint8)
    depth16 = np.rint(normalized * 65535).astype(np.uint16)
    return png_bytes(colour), png_bytes(depth16), span <= 1e-8


def default_loader(key, revision):
    import torch
    from dav2.models import load_model
    torch.set_num_threads(1)
    return load_model(key, device="cpu", revision=revision)


class ModelManager:
    """One model per process; lock covers loading, prediction and output conversion."""

    def __init__(self, loader: Callable = default_loader):
        self.loader = loader
        self._lock = threading.Lock()
        self._model = None
        self._key = None
        self._revision = None

    def clear(self):
        with self._lock:
            self._release()

    def _release(self):
        self._model = None
        self._key = None
        self._revision = None
        gc.collect()

    def predict(self, key, prepared: PreparedImage, on_status=None):
        if key not in MODEL_REVISIONS:
            raise PredictionError("This model is not supported.")
        notify = on_status or (lambda message: None)
        notify("Waiting for the prediction service…")
        with self._lock:
            try:
                revision = MODEL_REVISIONS[key]
                loaded = self._key != key or self._revision != revision or self._model is None
                load_ms = 0.0
                if loaded:
                    self._release()
                    notify(f"Loading {LABELS[key]}; the first request may download weights…")
                    started = time.perf_counter()
                    model = self.loader(key, revision)
                    self._model, self._key, self._revision = model, key, revision
                    del model
                    load_ms = (time.perf_counter() - started) * 1000
                notify("Generating relative depth…")
                started = time.perf_counter()
                disparity = self._model.predict(prepared.image)
                prediction_ms = (time.perf_counter() - started) * 1000
                colour, depth16, constant = encode_prediction(disparity, prepared.image.size)
                result = {"key": key, "revision": revision, "input_hash": prepared.input_hash,
                          "size": list(prepared.image.size), "original_size": list(prepared.original_size),
                          "resized": prepared.resized, "prediction_ms": prediction_ms,
                          "model_load_ms": load_ms, "loaded_this_request": loaded,
                          "device": self._model.device, "params_m": self._model.params_m,
                          "input_png": png_bytes(np.asarray(prepared.image)),
                          "colour_png": colour, "depth16_png": depth16,
                          "constant": constant}
                LOGGER.info("prediction model=%s revision=%s size=%s prediction_ms=%.1f load_ms=%.1f",
                            key, revision, result["size"], prediction_ms, load_ms)
                return result
            except Exception as error:
                self._release()
                LOGGER.exception("Prediction failed for %s", key)
                if isinstance(error, PredictionError):
                    raise
                raise PredictionError("Prediction could not finish. Try again with Small or a smaller image. If it persists, contact the app owner.") from error


def update_result(state, content, filename, key, manager, *, submitted=False, limits=None, on_status=None):
    """Reruns/downloads only read results; a new input or model invalidates stale output."""
    token = (hashlib.sha256(content).hexdigest(), key) if content else None
    if state.get("request_token") != token:
        state.pop("result", None)
        state["request_token"] = token
    if not submitted:
        return state.get("result")
    state.pop("result", None)
    prepared = prepare_image(content, filename, limits)
    result = manager.predict(key, prepared, on_status=on_status)
    state["result"] = result
    return result
