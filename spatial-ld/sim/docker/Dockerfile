FROM ghcr.io/lkirk/containers/tools/slim:4.2.2 AS slim
FROM debian:trixie-slim AS base

COPY --from=slim /usr/local/bin /usr/local/bin

ADD entrypoint /opt
ADD main.slim /opt

# workdir blank so that mounts don't clobber
WORKDIR /analysis

ENTRYPOINT ["/opt/entrypoint"]
