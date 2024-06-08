FROM ghcr.io/lkirk/containers/tools/slim:4.2.2 AS slim
FROM debian:trixie-slim AS base

# for now, not using the base python image because I've compiled slim on trixie
RUN \
	set -ex; \
	apt-get update; \
	apt-get install -y --no-install-recommends python3-pip jq; \
	rm -rf /var/lib/apt/lists/*

RUN set -ex; python3 -m pip install tszip --break-system-packages

COPY --from=slim /usr/local/bin /usr/local/bin

ADD entrypoint /opt
ADD main-spatial.slim /opt/main.slim

# workdir blank so that mounts don't clobber
WORKDIR /analysis

ENTRYPOINT ["/opt/entrypoint"]
