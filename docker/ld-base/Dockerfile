FROM python:3.12-slim-bookworm as build

RUN \
	set -ex; \
	apt-get update; \
	apt-get install -y git gcc make; \
	rm -rf /var/lib/apt/lists/*

RUN \
	set -ex; \
	pip install -U 'pip==24.1.1' 'setuptools==70.1.1'; \
	# Setup requires numpy, make sure we're using the same version as downstream
	pip install 'numpy==1.26.4'; \
	pip wheel --no-cache-dir --no-deps --wheel-dir /wheels \
	'tskit@git+https://github.com/lkirk/tskit.git@two-locus-staging&subdirectory=python'


FROM python:3.12-slim-bookworm as base

COPY --from=build /wheels /wheels

RUN pip install --no-cache /wheels/* && rm -r /wheels
