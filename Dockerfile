# This is the published, user-facing runtime image (built/pushed by docker.yml/publish.yml) - not to be
# confused with the sibling Dockerfile.ci-env (checks.yml's throwaway test-dependency cache). Deliberately
# does NOT `FROM` or otherwise inherit from a Dockerfile.ci-env-built image, even though they share the
# apt toolchain layer (docker/install-build-toolchain.sh):
# - Content: Dockerfile.ci-env bakes in requirements-dev.txt (pytest, deptry, flake8, tox, ...) - dev/test
#   tooling with no business in a published runtime image.
# - Architecture: Dockerfile.ci-env images are linux/amd64-only (checks.yml's build-ci-env.yml never builds
#   arm64, since none of its consumers need it); this image is genuinely multi-arch
#   (linux/amd64,linux/arm64), so inheriting would either break arm64 outright or force checks.yml to build
#   arm64 on every PR/push - far more often than this image is actually published - for no benefit to
#   checks.yml itself.
# - Version binding: this image installs flatland-rl from an arbitrary FLATLAND_RL_REF (a git ref, often a
#   past release tag, decoupled from whatever's currently checked out); Dockerfile.ci-env's image is tagged
#   by a hash of the *currently checked out* requirements-dev.txt/requirements-ml.txt, i.e. tied to HEAD, not
#   to an arbitrary historical ref. Reconciling the two would mean re-fetching/rebuilding per-ref anyway,
#   which is exactly the "no cross-run cache" situation this image's own no-cache build already accepts (see
#   docker.yml) - little caching benefit to gain from sharing.
ARG PYTHON_VERSION=3.12
ARG WITH_ML=false
ARG FLATLAND_RL_REF=main

FROM python:${PYTHON_VERSION}-slim AS base

ARG FLATLAND_RL_REF

WORKDIR /app

# See docker/install-build-toolchain.sh for what's installed and why - shared with Dockerfile.ci-env so a
# toolchain fix needed by one doesn't silently stay missing from the other. Notably this is what supplies
# gfortran/libopenblas-dev/pkg-config for numpy==1.26.4's from-source build on Pythons without a wheel -
# previously missing here entirely (masked so far by docker.yml only building python-version 3.12, which
# does have a numpy wheel).
COPY docker/install-build-toolchain.sh /tmp/install-build-toolchain.sh
RUN bash /tmp/install-build-toolchain.sh && rm /tmp/install-build-toolchain.sh && ffmpeg --help

RUN curl -fsSL "https://raw.githubusercontent.com/flatland-association/flatland-rl/${FLATLAND_RL_REF}/requirements.txt" -o requirements.txt \
    && pip install --no-cache-dir -r requirements.txt

FROM base AS ml-true

ARG FLATLAND_RL_REF

RUN curl -fsSL "https://raw.githubusercontent.com/flatland-association/flatland-rl/${FLATLAND_RL_REF}/requirements-ml.txt" -o requirements-ml.txt \
    && pip install --no-cache-dir -r requirements-ml.txt

FROM base AS ml-false

FROM ml-${WITH_ML} AS final

ARG FLATLAND_RL_REF
# Verified: when FLATLAND_RL_REF is a version tag (e.g. "v1.2.3" `pip show flatland-rl` reports "1.2.3", not "v1.2.3" and not a ".devN")
RUN python -m pip install "git+https://github.com/flatland-association/flatland-rl.git@${FLATLAND_RL_REF}"

