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

