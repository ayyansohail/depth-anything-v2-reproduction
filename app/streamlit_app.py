"""Streamlit entrypoint. Run from the repository: streamlit run app/streamlit_app.py."""
from pathlib import Path
import logging
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
# Community Cloud executes this file without installing the repository package.
for folder in (ROOT, ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import streamlit as st
from app.depth_service import (InputError, LABELS, ModelManager, PredictionError,
                               load_config, prepare_image, update_result)

st.set_page_config(page_title="Depth Anything V2", layout="wide")
logging.basicConfig(level=logging.INFO)


@st.cache_resource(show_spinner=False)
def prediction_service():
    return ModelManager()


def deployment_revision():
    """Read the host checkout for deployment troubleshooting; no model work."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL, timeout=2,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"

def main():
    st.title("Depth Anything V2")
    st.write("Turn one photo into a relative-depth map.")
    st.caption("Brighter colours mean nearer objects. These predictions describe relative inverse depth, not distance in metres.")
    try:
        enabled, limits = load_config()
    except (ValueError, OSError) as error:
        logging.getLogger("dav2.deployment").exception("Invalid app configuration")
        st.error("The app configuration needs attention. Please contact the app owner.")
        return
    with st.sidebar:
        key = st.selectbox("Model size", enabled, format_func=LABELS.get, index=0)
        st.caption("Small is the lowest-cost option. This CPU demo offers only models enabled for this host.")
        st.caption("Models load on demand. Initial downloads can take longer than later predictions.")
        with st.expander("About the models"):
            st.write("Depth Anything V2 — Yang et al., NeurIPS 2024. Released pretrained checkpoints; no training in this app.")
            st.markdown("[Paper and official code](https://github.com/DepthAnything/Depth-Anything-V2)")
            st.caption("Small: Apache-2.0. Base and Large: CC-BY-NC-4.0, for non-commercial use.")
            st.caption(f"Deployment build: {deployment_revision()}")
            st.caption(f"Python: {platform.python_version()}")
    upload = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"],
                              help="JPG/PNG, up to 10 MB and 16 megapixels; aspect ratio up to 2:1.")
    content = upload.getvalue() if upload is not None else b""
    filename = upload.name if upload is not None else ""
    manager = prediction_service()
    # This call only invalidates stale results or reads a completed result.
    update_result(st.session_state, content, filename, key, manager)
    if upload is None:
        st.info("Upload a JPG or PNG image to begin.")
        return
    try:
        prepared = prepare_image(content, filename, limits)
    except InputError as error:
        st.error(str(error))
        return
    if prepared.resized:
        st.caption(f"This image is resized from {prepared.original_size[0]} × {prepared.original_size[1]} to {prepared.image.width} × {prepared.image.height} for this demo.")
    if st.button("Generate depth", type="primary"):
        progress = st.empty()
        try:
            with st.spinner("Preparing your depth map…"):
                update_result(st.session_state, content, filename, key, manager,
                              submitted=True, limits=limits, on_status=progress.info)
        except (InputError, PredictionError) as error:
            st.error(str(error))
        finally:
            progress.empty()
    result = st.session_state.get("result")
    if result is None:
        st.image(prepared.image, caption="Input photo", use_container_width=True)
        return
    left, right = st.columns(2)
    left.image(result["input_png"], caption="Input photo", use_container_width=True)
    right.image(result["colour_png"], caption=f"Relative depth · {LABELS[result['key']]}", use_container_width=True)
    st.caption(f"Prediction: {result['prediction_ms'] / 1000:.2f} s · CPU · {result['params_m']:.1f}M parameters. Timing excludes model loading, upload and display.")
    if result["loaded_this_request"]:
        st.caption(f"Model preparation for this request: {result['model_load_ms'] / 1000:.2f} s.")
    if result["constant"]:
        st.warning("The model returned a nearly constant map. Try a different photo.")
    a, b = st.columns(2)
    a.download_button("Download colour PNG", result["colour_png"],
                      file_name=f"depth_{result['key']}.png", mime="image/png", on_click="ignore")
    b.download_button("Download normalized 16-bit PNG", result["depth16_png"],
                      file_name=f"relative_inverse_depth16_{result['key']}.png", mime="image/png", on_click="ignore")
    st.caption("The 16-bit PNG is normalized relative inverse depth: 0 is farther, 65535 is nearer within this image. Values are not metres and cannot be compared as distances between photos.")


if __name__ == "__main__":
    main()
