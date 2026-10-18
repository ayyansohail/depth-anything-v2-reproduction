# Repository migration

Current repository: https://github.com/ayyansohail/depth-anything-v2-reproduction

The team project was copied from https://github.com/maaz-zubairi/depth-anything-v2-repro with its existing commit history, author attribution, licenses and saved evidence. Copying the repository does not change authorship. The original repository is retained unchanged.

The source branch tips before migration were:

| Branch | Preserved source commit |
|---|---|
| main | 564b81ff1f66ef8bde15370414cf773e545bde07 |
| feat/computational-analysis-experiment | 5f4c44c4ca4b0ecaaa480ace3f9788f0ba0f6b38 |
| feat/streamlit-deployment | e9678940b2e5c5044744092ade9d74cf897b7fa5 |

The destination's initial README commit, 8bbfe1c0e2cf2e578983c5f36122f235dde94502, is preserved in main's history through a merge. The complete project README supersedes that single-line repository title. No force push or source deletion is needed.

Each branch receives repository-link and notebook-clone updates. Archived results, original ZIPs and their recorded execution commits remain unchanged. Existing Colab evidence still describes the commits that actually ran; migration does not require new model inference, training or benchmarks.

For Streamlit deployment select repository **ayyansohail/depth-anything-v2-reproduction**, branch **feat/streamlit-deployment**, entrypoint **app/streamlit_app.py**, Python **3.13**. The deployment branch contains the reviewed Stage 5 evidence; main retains the earlier project state. Small alone is enabled until actual-host checks pass. No public deployment is claimed by this migration.
