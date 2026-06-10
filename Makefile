.PHONY: help install test lint format mlflow dagster lineage

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Create venv and install adapter-sdk (editable, with dev deps)
	python3 -m venv .venv
	. .venv/bin/activate && pip install -e "adapter-sdk[dev]"

test:  ## Run pytest across packages
	. .venv/bin/activate && pytest adapter-sdk

lint:  ## ruff check + format check
	. .venv/bin/activate && ruff check adapter-sdk && ruff format --check adapter-sdk

format:  ## ruff auto-format
	. .venv/bin/activate && ruff format adapter-sdk && ruff check --fix adapter-sdk

mlflow:  ## start local MLflow UI (sqlite backend)
	@echo "MLflow arrives in M2 — target stubbed."

dagster:  ## dagster dev
	@echo "Dagster arrives in M4 — target stubbed."

lineage:  ## docker compose up marquez
	@echo "Marquez/OpenLineage arrives in M4 — target stubbed."
