FROM python:3.11-slim-bullseye as build

RUN \
	set -ex; \
	apt-get update; \
	apt-get install -y git gcc make; \
	rm -rf /var/lib/apt/lists/*

RUN \
	set -ex; \
	pip install -U 'pip==24.1.1' 'setuptools==70.1.1'; \
	git clone https://github.com/lkirk/tskit -b two-locus-staging; \
	cd tskit/python; \
	make; \
	pip wheel --no-cache-dir --no-deps --wheel-dir /wheels /tskit/python

FROM python:3.11-slim-bullseye as base

COPY --from=build /wheels /wheels

RUN pip install --no-cache /wheels/* && rm -r /wheels
