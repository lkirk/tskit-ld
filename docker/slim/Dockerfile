FROM ghcr.io/lkirk/containers/tools/slim:4.2.2 AS slim
FROM python:3.12-slim-bookworm as build

RUN \
    set -ex; \
    apt-get update; \
    apt-get install -y git; \
    rm -rf /var/lib/apt/lists/*

RUN pip wheel --no-cache-dir --wheel-dir /wheels \
    tskit tszip \
    'htcluster@git+https://github.com/lkirk/htcluster.git'

FROM python:3.12-slim-bookworm as slim-spatial

COPY --from=slim /usr/local/bin /usr/local/bin
COPY --from=build /wheels /wheels

RUN pip install --no-cache --no-deps /wheels/* && rm -r /wheels
ADD main-spatial.slim /opt/main.slim

# workdir blank so that mounts don't clobber
WORKDIR /analysis

ENTRYPOINT ["htcluster-wrapper"]
