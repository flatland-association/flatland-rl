ARG PYTHON_VERSION=3.12
ARG WITH_ML=false
ARG FLATLAND_RL_REF=main

FROM python:${PYTHON_VERSION}-slim AS base

ARG FLATLAND_RL_REF

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl gcc build-essential wget zip ffmpeg git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    ffmpeg --help

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

