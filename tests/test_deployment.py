"""Stage 5 tests. Execute in Colab; importing helpers downloads no weights."""
import gc
import io
import threading
import time
import weakref

import numpy as np
from PIL import Image
import pytest

from app.depth_service import (DEFAULT_LIMITS, InputError, MODEL_REVISIONS,
                               ModelManager, PredictionError, encode_prediction,
                               load_config, prepare_image, update_result)


def image_bytes(mode="RGB", size=(64, 48), colour=None, fmt="PNG", exif=None):
    image = Image.new(mode, size, colour)
    buffer = io.BytesIO()
    kwargs = {"exif": exif} if exif is not None else {}
    image.save(buffer, format=fmt, **kwargs)
    return buffer.getvalue()


class FakeModel:
    device = "cpu"
    params_m = 24.8

    def __init__(self):
        self.calls = 0

    def predict(self, image):
        self.calls += 1
        return np.tile(np.linspace(0, 1, image.width, dtype=np.float32), (image.height, 1))


@pytest.mark.parametrize("mode,colour", [("RGB", "red"), ("L", 50), ("RGBA", (255, 0, 0, 0))])
def test_valid_image_modes_and_transparency(mode, colour):
    prepared = prepare_image(image_bytes(mode, colour=colour), "photo.png")
    assert prepared.image.mode == "RGB" and prepared.image.size == (64, 48)
    if mode == "RGBA":
        assert prepared.image.getpixel((0, 0)) == (255, 255, 255)


def test_jpeg_rotation_and_resizing():
    exif = Image.Exif()
    exif[274] = 6
    prepared = prepare_image(image_bytes(size=(80, 40), colour="red", fmt="JPEG", exif=exif), "a.jpg")
    assert prepared.original_size == (40, 80) and prepared.image.size == (40, 80)
    large = prepare_image(image_bytes(size=(2048, 1024), colour="red"), "a.png")
    assert large.resized and large.image.size == (1024, 512)


@pytest.mark.parametrize("content,name", [(b"broken", "a.png"), (b"", "a.png"),
                                           (image_bytes(), "a.gif"),
                                           (image_bytes(fmt="GIF"), "a.png")])
def test_invalid_images_are_rejected(content, name):
    with pytest.raises(InputError):
        prepare_image(content, name)


def test_size_limits_and_aspect_boundaries():
    content = image_bytes()
    prepare_image(content, "a.png", {"max_upload_bytes": len(content)})
    with pytest.raises(InputError, match="upload limit"):
        prepare_image(content, "a.png", {"max_upload_bytes": len(content) - 1})
    prepare_image(content, "a.png", {"max_pixels": 64 * 48})
    with pytest.raises(InputError, match="megapixels"):
        prepare_image(content, "a.png", {"max_pixels": 64 * 48 - 1})
    prepare_image(image_bytes(size=(100, 50)), "a.png")
    with pytest.raises(InputError, match="aspect ratio"):
        prepare_image(image_bytes(size=(101, 50)), "a.png")


def test_constant_and_invalid_prediction_outputs():
    colour, raw, constant = encode_prediction(np.ones((48, 64)), (64, 48))
    assert constant and Image.open(io.BytesIO(colour)).size == (64, 48)
    assert np.asarray(Image.open(io.BytesIO(raw))).max() == 0
    for values in [np.full((48, 64), np.nan), np.full((48, 64), np.inf), np.zeros((1, 2))]:
        with pytest.raises(PredictionError):
            encode_prediction(values, (64, 48))


def test_downloads_are_rgb_and_16_bit_with_monotonic_near_values():
    values = np.tile(np.linspace(0, 10, 64), (48, 1))
    colour, raw, constant = encode_prediction(values, (64, 48))
    assert not constant and Image.open(io.BytesIO(colour)).mode == "RGB"
    decoded = np.asarray(Image.open(io.BytesIO(raw)))
    assert decoded.shape == (48, 64) and decoded.min() == 0 and decoded.max() == 65535
    assert (np.diff(decoded[0].astype(np.int64)) >= 0).all()


def test_result_retention_and_invalidation_do_not_repeat_inference():
    model = FakeModel()
    manager = ModelManager(lambda key, revision: model)
    state, content = {}, image_bytes(colour="red")
    assert update_result(state, content, "a.png", "v2-small", manager) is None
    first = update_result(state, content, "a.png", "v2-small", manager, submitted=True)
    for _ in range(3):
        assert update_result(state, content, "a.png", "v2-small", manager) is first
    assert model.calls == 1
    assert update_result(state, content, "a.png", "v2-base", manager) is None
    assert "result" not in state
    update_result(state, content, "a.png", "v2-small", manager, submitted=True)
    assert update_result(state, image_bytes(colour="blue"), "a.png", "v2-small", manager) is None
    assert "result" not in state


def test_switch_releases_old_model_and_passes_pinned_revision():
    references, calls = [], []
    def loader(key, revision):
        gc.collect()
        if references:
            assert references[-1]() is None, "Previous model must be released before loading the next"
        model = FakeModel()
        references.append(weakref.ref(model))
        calls.append((key, revision))
        return model
    manager = ModelManager(loader)
    prepared = prepare_image(image_bytes(), "a.png")
    manager.predict("v2-small", prepared)
    manager.predict("v2-small", prepared)
    manager.predict("v2-base", prepared)
    assert calls == [("v2-small", MODEL_REVISIONS["v2-small"]), ("v2-base", MODEL_REVISIONS["v2-base"])]
    manager.clear()
    assert references[-1]() is None


def test_failed_load_or_prediction_can_be_retried_without_stale_result():
    attempts = []
    def loader(key, revision):
        attempts.append(key)
        if len(attempts) == 1:
            raise OSError("simulated model download failure")
        return FakeModel()
    manager, state = ModelManager(loader), {}
    content = image_bytes()
    with pytest.raises(PredictionError):
        update_result(state, content, "a.png", "v2-small", manager, submitted=True)
    assert "result" not in state
    assert update_result(state, content, "a.png", "v2-small", manager, submitted=True)
    manager._model.predict = lambda image: np.full((48, 64), np.nan)
    with pytest.raises(PredictionError):
        update_result(state, content, "a.png", "v2-small", manager, submitted=True)
    assert "result" not in state and manager._model is None


def test_predictions_are_serialized_across_sessions():
    activity = {"active": 0, "peak": 0}
    class ConcurrentFake(FakeModel):
        def predict(self, image):
            activity["active"] += 1
            activity["peak"] = max(activity["peak"], activity["active"])
            time.sleep(0.01)
            output = super().predict(image)
            activity["active"] -= 1
            return output
    manager = ModelManager(lambda key, revision: ConcurrentFake())
    prepared = prepare_image(image_bytes(), "a.png")
    failures = []
    def request():
        try:
            manager.predict("v2-small", prepared)
        except Exception as error:
            failures.append(error)
    threads = [threading.Thread(target=request) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert not failures and all(not t.is_alive() for t in threads)
    assert activity["peak"] == 1


def test_public_config_has_known_models_with_small_default():
    enabled, limits = load_config()
    assert enabled[0] == "v2-small" and set(enabled) <= set(MODEL_REVISIONS) and limits == DEFAULT_LIMITS


def test_optional_revision_preserves_existing_loader_calls(monkeypatch):
    from dav2 import models
    events = []
    class FakeBackend:
        def to(self, device):
            return self
        def eval(self):
            return self
    def processor(repo, **kwargs):
        events.append(("processor", kwargs))
        return object()
    def backend(repo, **kwargs):
        events.append(("model", kwargs))
        return FakeBackend()
    monkeypatch.setattr(models.AutoImageProcessor, "from_pretrained", processor)
    monkeypatch.setattr(models.AutoModelForDepthEstimation, "from_pretrained", backend)
    models.load_model("v2-small", device="cpu")
    assert events == [("processor", {}), ("model", {})]
    events.clear()
    models.load_model("v2-small", device="cpu", revision="pinned")
    assert events[0] == ("processor", {"revision": "pinned", "use_fast": False})
    assert events[1][1]["revision"] == "pinned" and "torch_dtype" in events[1][1]


def test_streamlit_rerun_and_download_controls_use_retained_result(monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    from pathlib import Path
    from app import depth_service
    class Upload(io.BytesIO):
        name = "example.png"
    upload = Upload(image_bytes(colour="red"))
    calls = {"predictions": 0}
    class UiFake(FakeModel):
        def predict(self, image):
            calls["predictions"] += 1
            return super().predict(image)
    real_manager = depth_service.ModelManager
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: upload)
    monkeypatch.setattr(depth_service, "ModelManager", lambda: real_manager(lambda key, revision: UiFake()))
    monkeypatch.setattr(depth_service, "load_config", lambda: (["v2-small", "v2-base"], dict(DEFAULT_LIMITS)))
    st.cache_resource.clear()
    path = Path(__file__).resolve().parents[1] / "app/streamlit_app.py"
    app = AppTest.from_file(str(path), default_timeout=30).run()
    assert not app.exception and calls["predictions"] == 0
    app.button[0].click().run()
    assert not app.exception and calls["predictions"] == 1
    assert len(app.get("download_button")) == 2
    assert app.session_state["result"]["key"] == "v2-small"
    app.run()
    assert not app.exception and calls["predictions"] == 1
    app.selectbox[0].select("v2-base").run()
    assert not app.exception and calls["predictions"] == 1
    assert "result" not in app.session_state.filtered_state
    app.button[0].click().run()
    assert not app.exception and calls["predictions"] == 2
    assert app.session_state["result"]["key"] == "v2-base"
    st.cache_resource.clear()
