# Stage 5 Streamlit deployment

Branch: **feat/streamlit-deployment**, based on the completed Stage 4 branch at
5f4c44c4ca4b0ecaaa480ace3f9788f0ba0f6b38. Main and the Stage 4 branch stay unchanged.

Status: Colab verification accepted (24 tests; Small, Base, Large, then Small real CPU predictions). See [the evidence review](../deployment/reviewed/stage5_verification_20261018T202743076758Z/REVIEW.md). Initial Small hosted smoke passed at https://depth-anything-v2-repro.streamlit.app/ (prediction, repeat prediction, both downloads and saved screenshot). Extended hosted acceptance remains pending; see deployment/hosted/small_initial/verification.json.
The public app opens without owner login and Small produces a result; Base/Large, extended acceptance and the exact runtime commit still need verification. The formal report is deferred.

## 1 Verify the application in Colab

Open [the Stage 5 verification notebook](https://colab.research.google.com/github/ayyansohail/depth-anything-v2-reproduction/blob/feat/streamlit-deployment/notebooks/stage5_deployment_verification.ipynb).
Use a fresh CPU runtime with Python 3.13. Execute its cells individually:

1. Clone this branch and create the isolated CPU environment. If standard venv fails while bootstrapping pip, the notebook preserves that failure and falls back to virtualenv 20.35.4 in a separate workspace directory. The bootstrap tool is installed into a temporary target directory, not the notebook environment.
2. Parse source and run metric, helper, loader and mocked Streamlit UI tests.
3. Upload one JPG/PNG suitable for inclusion in coursework evidence.
4. Verify real predictions using pinned Small, Base, Large, then Small again.
5. Download the evidence ZIP and return it for review before deployment.

No datasets, training, Stage 4 benchmarks or GPU are needed. The real-model check
may download several GB of weights to Colab. They are excluded from evidence ZIPs.
If any step fails, stop and return the error or use the final cell to download
available logs. An isolated environment avoids the notebook's preinstalled GPU
torch/torchvision affecting the CPU stack. Verification subprocesses explicitly use the Agg plotting backend so Colab's inline backend does not leak into the isolated environment. The owner should return Colab images
and logs; successful Colab checks do not prove hosting memory feasibility.

## 2 Deploy Small first on Community Cloud

After verification review:

1. Sign into [Streamlit Community Cloud](https://share.streamlit.io/) and connect the GitHub account with access to this public repository.
2. Choose Create app, then the existing-app flow.
3. Repository: `ayyansohail/depth-anything-v2-reproduction`.
4. Branch: `feat/streamlit-deployment`.
5. Main file path: `app/streamlit_app.py`.
6. Advanced settings: Python **3.13**. No token or other secret is required for these public checkpoints.
7. Deploy, retain build/runtime logs, and return the public URL or any error.

The app-specific `app/requirements.txt` takes precedence over root research
dependencies. It pins matching CPU torch/torchvision, Transformers, Hugging Face Hub, NumPy, Pillow,
Matplotlib and Streamlit. The entrypoint adds the checked-out src package path;
it does not require an editable install. `.streamlit/config.toml` enables headless
mode and a 10 MB uploader limit. Do not use an unpinned root requirements file
as a workaround for a deployment error; diagnose and update the app stack.

References: [dependency selection](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies),
[deployment and Python selection](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

## 3 User workflow and output meaning

Upload JPG, JPEG or PNG, select one enabled model, then choose **Generate depth**.
The input and coloured depth map appear together. Brighter values indicate nearer
objects. This is relative inverse depth, not metres. A normalized 16-bit PNG is
also available; 0 represents farther and 65535 nearer within that image. Such
values cannot be compared as absolute distances between different images.

Limits: 10 MiB upload, 16 million decoded pixels, aspect ratio up to 2:1 in either
orientation. EXIF orientation is corrected; transparent pixels are composited on
white. Images are downscaled without upscaling to a longest edge of 1,024 pixels.
The resulting image determines both output dimensions and inference input. This
app-specific preprocessing may differ from benchmark inputs; benchmark accuracy
is not asserted for resized uploads. Constant outputs produce a warning; invalid
or non-finite outputs are rejected.

Prediction time includes the model wrapper's preprocessing, forward pass and
upsampling. It excludes weight loading, upload, PNG encoding and display. Loading
has a separate message and timing. Downloading outputs or changing widgets does
not request another prediction. A changed image or model clears old outputs so
they are not mislabeled as predictions for a new request.

## 4 Memory management and enabled models

The cached server resource is one ModelManager. Its lock covers model loading,
prediction and output encoding across sessions. It releases its previous model
before loading another. Only image/output bytes and scalar metadata are stored
in per-user session state. Model weights can remain in the host's disk cache;
multiple model instances are not deliberately retained in RAM.

`app/deployment.json` currently enables verified Small/Base plus candidate Large for actual-host testing. Base-to-Small switching has passed. Small remains
the default. After a successful Small hosted check, the assistant enables Base
in a reviewed commit, the owner repeats hosted tests, then Large is tried in
another reviewed commit. A failing candidate is disabled again. CPU out-of-memory
may kill the host process rather than raise a recoverable Python exception; use
Cloud logs/restarts for diagnosis and do not repeatedly expose an unstable model.
No paid hardware upgrade is automatic. A reliable Small-only deployment can
satisfy Stage 5, with larger-model limits explicitly recorded.

Models use the exact V2 revisions recorded in Stage 4; the pinned loader also
uses slow/Pillow image processing; its required CPU torchvision dependency is pinned alongside torch.
Legacy callers without a revision retain their original loader behaviour.
Small is Apache-2.0; Base/Large are CC-BY-NC-4.0 (non-commercial).

## 5 Hosted acceptance and evidence

Use the same worked-example photo to check:

- Public URL opens in a logged-out/incognito browser.
- Fresh session loads and predicts with Small; initial model download is visible.
- Input/depth images have matching processed dimensions and the selected model is labeled correctly.
- Both PNG downloads open; colour is RGB and the second is a normalized 16-bit image.
- Repeated Generate requests work; downloads do not start another prediction.
- Changing an image/model clears old results and switching back works.
- Every enabled model completes repeated requests; invalid uploads show useful errors.
- Another session can use the service without concurrent model loads; long requests wait.
- Capture cold and warm behaviour on this host without promising Colab-equivalent latency.

Return the public URL, hosted logs if there are errors, and a screenshot or short
recording of a successful input-to-result workflow. The assistant records the
actual deployment commit, verified models and evidence in the repository. Do not
mark acceptance complete solely because the landing page or build succeeds.
If the owner cannot inspect a download's bit depth in a viewer, its PNG can be
returned for file inspection without running inference locally.

## Reproduction commands (Colab or hosted Linux environment)

```bash
python -m pip install -r app/requirements.txt
python -m streamlit run app/streamlit_app.py --server.headless=true
```

In the isolated Colab environment, use its Python executable for both commands.
Use the supplied notebook for tests and real-image verification. No local Python,
installs, tests or inference were executed during source preparation.

## Completion

Stage 5 requires a working public URL, hosted verification of every enabled
model, pinned dependency/startup documentation, a worked example, and a screenshot
or recording. The implementation alone is not a completed deployment. Current
acceptance status is recorded in `app/hosted_acceptance.json`; the exact runtime commit stays null
until confirmed from host records. Keep Streamlit account credentials out of Git.
