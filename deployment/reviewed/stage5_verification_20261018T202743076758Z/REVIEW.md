# Stage 5 Colab verification review

Accepted for proceeding to Small-first hosting. Stage 5 hosted acceptance remains pending.

Execution source: `dedcdd309d1282c40f30d7e0a2ed759632a4a9bf` on `feat/streamlit-deployment`.
Returned archive: `stage5_verification_20261018T202743076758Z.zip`, preserved as `original.zip`.
SHA256: `c5830dff1893db4cdc9a00f71578a6cea5669b0f559e5de800841a32140fafde`.
Extracted evidence is preserved unchanged alongside this review. The later import commit records evidence and documentation; it is not a new application execution.

## Evidence accepted

- Final `tests.log`: 24 tests passed in 10.10 seconds. The earlier 7-failure log remains as historical evidence; it is superseded by the final run after the recorded setup fixes.
- Python 3.13.16, torch 2.11.0+cpu, torchvision 0.26.0+cpu and the other pinned direct dependencies match the application requirements. Exact installed versions are in `real_predictions/requirements.snapshot.txt`.
- Checkout and real-prediction manifest agree on the execution commit and branch. All three model revisions match the Stage 4 pins. The deployment configuration hash matches after normalizing this Windows checkout's CRLF line endings to Git/Colab LF; this is a byte-format difference.
- Small, Base, Large, then Small all completed on CPU with finite positive recorded times, nonconstant outputs and the expected pinned revisions. Small reloaded successfully after Large. Colab retained-result checks passed; mocked tests cover serialization, releasing old models, errors, upload validation and Streamlit reruns/download controls.
- The 4032 x 3024 supplied living-room photograph was resized to 1024 x 768. All 12 saved PNGs have those processed dimensions. Input/colour PNG headers are 8-bit RGB, and normalized depth PNG headers are 16-bit grayscale. The Colab script checked successful decoding, matching dimensions, and normalized depth range 0-65535.
- Input and all three colour maps were visually inspected during review. Nearby furniture/floor is brighter and the background wall is darker, consistent with the declared inverse-depth convention. This is a qualitative inspection, not a ground-truth accuracy measurement. The two Small colour outputs and two Small 16-bit outputs have identical respective SHA256 hashes.

## Recorded CPU timings for this example

| Sequence | Model | Prediction seconds | Model loading seconds | Cumulative process peak RSS MiB |
|---|---|---:|---:|---:|
| 1 | Small | 5.454 | 5.196 | 719.6 |
| 2 | Base | 12.656 | 3.982 | 1216.5 |
| 3 | Large | 39.738 | 11.457 | 2605.7 |
| 4 | Small reload | 3.721 | 0.348 | 2605.7 |

These are individual smoke-check observations, not a benchmark. Prediction excludes loading and output encoding. Peak RSS is cumulative over this process; the final Small value does not describe its current memory use. Colab speed and memory observations do not establish Streamlit Community Cloud feasibility.

## Remaining hosted checks

Deploy Small only on Community Cloud using this branch, entrypoint `app/streamlit_app.py`, and Python 3.13. Keep Base/Large disabled until individually verified on the actual host. Check public access without the owner's login, a complete upload/predict/both-downloads flow, repeated requests, invalid uploads, and fresh/cold sessions. Capture a screenshot or short recording and retain the public URL and deployed commit. The included worked example and prediction images are Colab evidence, not a screenshot of a deployed app.

The existing unauthenticated Hub notice and deprecated argument warnings did not prevent any prediction. No training, local execution, hosted deployment, or formal report editing was performed during this review.
