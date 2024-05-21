FROM ghcr.io/lkirk/containers/tools/slim:4.2.2 AS slim
FROM debian:trixie-slim AS base

COPY --from=slim /usr/local/bin /usr/local/bin

WORKDIR /opt

ADD entrypoint /opt
ADD main.slim /opt

ENTRYPOINT ["/opt/entrypoint"]
