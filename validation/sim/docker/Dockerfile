FROM python:3.11-slim-bullseye

ADD requirements.txt .

RUN \
	set -ex; \
	apt-get update; \
	apt-get install -y git gcc; \
	rm -rf /var/lib/apt/lists/*

RUN \
	set -ex; \
	git clone https://github.com/lkirk/tskit -b two-locus-branch-stats-positions

RUN \
	set -ex; \
	# create virtual environment
	python -m venv ve; \
	# update pip, setuptools
	ve/bin/pip install -U pip setuptools; \
	# install requirements
	ve/bin/pip install -r requirements.txt

WORKDIR opt

ADD analysis.py .

ENTRYPOINT ["/ve/bin/python", "/opt/analysis.py"]
