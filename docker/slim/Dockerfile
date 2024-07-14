FROM ghcr.io/lkirk/containers/tools/slim:4.2.2 AS slim
FROM python:3.11-slim-bullseye as build

RUN \
	set -ex; \
	apt-get update; \
	apt-get install -y git; \
	rm -rf /var/lib/apt/lists/*

RUN pip wheel --no-cache-dir --wheel-dir /wheels \
	tszip \
	'htcluster@git+https://github.com/lkirk/htcluster.git'
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheels \
	'tskit-ld@git+https://github.com/lkirk/tskit-ld.git'

FROM python:3.11-slim-bullseye as slim-spatial

COPY --from=slim /usr/local/bin /usr/local/bin
COPY --from=build /wheels /wheels

RUN pip install --no-cache --no-deps /wheels/* && rm -r /wheels
ADD main-spatial.slim /opt/main.slim

# workdir blank so that mounts don't clobber
WORKDIR /analysis

ENTRYPOINT ["htcluster-wrapper"]
