FROM python:3.11-slim-bullseye

ADD requirements.txt .

RUN \
	set -ex; \
	apt-get update; \
	apt-get install -y git gcc make; \
	rm -rf /var/lib/apt/lists/*

RUN \
	set -ex; \
	git clone https://github.com/lkirk/tskit -b two-locus-branch-stats-positions; \
	cd tskit/python; \
	make; \
	# install local build of tskit
	pip install /tskit/python

RUN \
	set -ex; \
	# update pip, setuptools
	pip install -U pip setuptools; \
	# install requirements
	pip install -r requirements.txt

WORKDIR /opt

ADD entrypoint .

ENTRYPOINT ["/opt/entrypoint"]
