.PHONY: install etl build-graph train-baseline train-gnn train-gnn-rte train-hybrid check-timestamps check-graph check-density debug all

install:
	pip install -r requirements.txt

env:
	python3 -m venv .venv
	@echo "Virtual environment created. To activate:"
	@echo "  source .venv/bin/activate"
	.venv/bin/pip install -r requirements.txt

etl:
	python src/cli.py extract-data

build-graph:
	python src/cli.py build-graph

train-baseline:
	python src/cli.py train-baseline

# --- Experiments ---

exp-gat:
	python src/cli.py train-embeddings --model gat
	python src/cli.py train-hybrid --model gat

exp-sage:
	python src/cli.py train-embeddings --model sage
	python src/cli.py train-hybrid --model sage

exp-hgt:
	python src/cli.py train-embeddings --model hgt
	python src/cli.py train-hybrid --model hgt

exp-hgt-rte:
	python src/cli.py train-embeddings --model hgt_rte
	python src/cli.py train-hybrid --model hgt_rte

# -------------------

check-timestamps:
	python src/cli.py check-timestamps

check-graph:
	python src/cli.py check-graph-timestamps

check-density:
	python src/cli.py check-density

debug:
	python src/cli.py debug-polars

all: etl build-graph train-baseline train-gnn-rte train-hybrid
