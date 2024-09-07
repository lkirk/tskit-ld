FROM python:3.12-slim-bookworm as build-wheels

RUN \
    set -ex; \
    apt-get update; \
    apt-get install -y git gcc make; \
    rm -rf /var/lib/apt/lists/*

ADD requirements-frozen.txt .
RUN pip install -U 'pip==24.1.1' 'setuptools==70.1.1'
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheels -r requirements-frozen.txt
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheels \
    'tskit-ld@git+https://github.com/lkirk/tskit-ld.git'

FROM python:3.12-slim-bookworm as analysis

COPY --from=build-wheels /wheels /wheels
RUN pip install --no-cache-dir --no-deps /wheels/* && rm -r /wheels
WORKDIR /analysis

ENTRYPOINT ["htcluster-wrapper"]
