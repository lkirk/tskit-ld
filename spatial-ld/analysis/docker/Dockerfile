FROM ghcr.io/lkirk/tskit-ld/ld-base as build

ADD requirements-frozen.txt .

RUN pip wheel --no-cache-dir --no-deps --wheel-dir /wheels -r requirements-frozen.txt


FROM python:3.11-slim-bullseye as base

COPY --from=build /wheels /wheels

RUN pip install --no-cache --no-deps /wheels/* && rm -r /wheels

WORKDIR /opt

ADD spatial.py .
ADD entrypoint .

ENTRYPOINT ["/opt/entrypoint"]
