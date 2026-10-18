"""Model loading + inference for Depth Anything V1/V2 (S/B/L).

Design choice: we use the Hugging Face `transformers` port of the official
checkpoints. Same weights, no repo-cloning or sys.path hacks, and it works
identically in Colab, Kaggle, local CPU and the Streamlit app. The official
repo (DepthAnything/Depth-Anything-V2) is our reference for architecture and
evaluation protocol; see PROVENANCE.md.

Output convention: `predict()` returns RELATIVE INVERSE DEPTH (disparity-like):
larger value = closer to the camera. It is NOT in metres.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

REGISTRY = {
    "v2-small": "depth-anything/Depth-Anything-V2-Small-hf",
    "v2-base": "depth-anything/Depth-Anything-V2-Base-hf",
    "v2-large": "depth-anything/Depth-Anything-V2-Large-hf",
    "v1-small": "LiheYoung/depth-anything-small-hf",
    "v1-base": "LiheYoung/depth-anything-base-hf",
    "v1-large": "LiheYoung/depth-anything-large-hf",
}
V2_KEYS = ["v2-small", "v2-base", "v2-large"]
V1_KEYS = ["v1-small", "v1-base", "v1-large"]


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device: str) -> None:
    """Wait for queued GPU work so wall-clock timing is honest."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


@dataclass
class DepthModel:
    key: str
    device: str
    processor: object
    model: torch.nn.Module

    @property
    def params_m(self) -> float:
        return sum(p.numel() for p in self.model.parameters()) / 1e6

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> np.ndarray:
        """PIL image -> (H, W) float32 relative inverse depth at the ORIGINAL resolution."""
        image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs).predicted_depth  # (1, h, w)
        out = F.interpolate(
            out.unsqueeze(1), size=image.size[::-1], mode="bilinear", align_corners=False
        )
        return out[0, 0].float().cpu().numpy()

    @torch.inference_mode()
    def forward_raw(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Bare forward pass on a preprocessed tensor (used for latency benchmarking)."""
        return self.model(pixel_values=pixel_values).predicted_depth


def load_model(key: str, device: str | None = None, *, revision: str | None = None) -> DepthModel:
    if key not in REGISTRY:
        raise KeyError(f"unknown model '{key}'. choose from {list(REGISTRY)}")
    device = device or pick_device()
    repo = REGISTRY[key]
    # Preserve legacy callers; deployments can pin both weights and processor.
    options = {"revision": revision} if revision is not None else {}
    processor_options = options | ({"use_fast": False} if revision is not None else {})
    processor = AutoImageProcessor.from_pretrained(repo, **processor_options)
    model_options = options | ({"torch_dtype": torch.float32} if revision is not None else {})
    model = AutoModelForDepthEstimation.from_pretrained(repo, **model_options).to(device).eval()
    return DepthModel(key=key, device=device, processor=processor, model=model)
