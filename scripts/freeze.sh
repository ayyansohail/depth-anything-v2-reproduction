#!/usr/bin/env bash
# Capture the CURRENT environment. This establishes historical versions only
# if it is still the unchanged results environment. See docs/COLAB_STAGE3.md.
# A package snapshot is not an installation-tested portable lockfile.
pip freeze > requirements.colab.snapshot.txt
echo "wrote requirements.colab.snapshot.txt for the current environment"
