#!/bin/bash
set -euo pipefail

# Shared apt build-toolchain/utility layer for Dockerfile (published, multi-arch release image) and
# Dockerfile.ci-env (CI-only test-dependency cache, see checks.yml's build-ci-env-ml/build-ci-env-default
# jobs) - single source of truth so a toolchain fix needed by one image doesn't silently stay missing from
# the other. This is exactly what happened before this file existed: Dockerfile.ci-env's base stage grew
# gfortran/libopenblas-dev/pkg-config to fix a numpy==1.26.4 from-source build failure, but Dockerfile's own,
# separately-maintained apt-get line never got the same fix.
#
# - build-essential/gfortran/libopenblas-dev/pkg-config: numpy==1.26.4 (pinned via pyproject.toml's
#   "numpy<2" in requirements.txt/requirements-dev.txt/requirements-ml.txt alike) has no wheels for newer
#   Pythons and builds from source - gfortran + libopenblas-dev + pkg-config are its meson build requirements
#   (Fortran + BLAS/LAPACK backend + config lookup); meson/ninja/pybind11 themselves come from numpy's own
#   PEP 517 build-requires via pip's build isolation, so they don't need to be installed here.
# - ffmpeg: used by flatland's rendering code path (both at runtime for Dockerfile's users, and by
#   Dockerfile.ci-env's rendering tests).
# - git: Dockerfile needs it for its final `pip install git+https://...@ref` step; Dockerfile.ci-env needs
#   it for setuptools_scm's `git describe --tags` during `pip install .` (checks.yml's test job) - without
#   it, the failure is swallowed and surfaces as the misleading "setuptools-scm was unable to detect
#   version" instead of a "git: command not found" (see pypa/setuptools-scm#278 for others hitting the same
#   git-less-container symptom).
# - curl/wget: Dockerfile fetches requirements*.txt via curl; Dockerfile.ci-env's episodes-download step
#   (test/testml) uses wget.
# - zip/unzip: episodes/scenario archives are unzipped inside Dockerfile.ci-env's steps; zip isn't currently
#   exercised by either image's own RUN steps but is cheap to keep alongside unzip for symmetry.
#
# The Ubuntu CI runner (checks.yml's non-containerized jobs) has all of these preinstalled by default,
# masking these requirements; neither Debian slim base here has any of them.
apt-get update
apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    libopenblas-dev \
    pkg-config \
    ffmpeg \
    git \
    curl \
    wget \
    zip \
    unzip
apt-get clean
rm -rf /var/lib/apt/lists/*
