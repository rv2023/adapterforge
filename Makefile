.PHONY: help install test lint format mlflow dagster lineage lineage-down \
        control-plane serving drift loop register

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Create venv, install adapter-sdk (editable) + pipeline/serving deps
	python3 -m venv .venv
	. .venv/bin/activate && pip install -e "adapter-sdk[dev]" && pip install -r requirements.txt

test:  ## Run pytest across packages
	. .venv/bin/activate && pytest adapter-sdk

lint:  ## ruff check + format check
	. .venv/bin/activate && ruff check adapter-sdk pipelines control-plane serving && ruff format --check adapter-sdk

format:  ## ruff auto-format + fix
	. .venv/bin/activate && ruff format adapter-sdk pipelines control-plane serving && ruff check --fix adapter-sdk pipelines control-plane serving

# --- M2: tracking ---
mlflow:  ## start the local MLflow UI (sqlite backend) at :5000
	. .venv/bin/activate && mlflow ui --backend-store-uri sqlite:///mlflow.db

# --- M3: control plane + serving ---
register:  ## register the baseline as fpb-sentiment v1 with its dossier
	. .venv/bin/activate && python pipelines/register_baseline.py

control-plane:  ## run the governance API at :8000
	. .venv/bin/activate && uvicorn app:app --app-dir control-plane --port 8000

serving:  ## run the inference API at :8001 (needs control-plane up)
	. .venv/bin/activate && uvicorn app:app --app-dir serving --port 8001

# --- M4: orchestration / lineage / drift / the loop ---
dagster:  ## dagster dev — the ingest->train->register DAG (:3000)
	. .venv/bin/activate && dagster dev -f pipelines/dag.py

lineage:  ## start Marquez (OpenLineage backend) in Docker — UI :3001, API :5000
	docker compose -f docker-compose.marquez.yml up -d

lineage-down:  ## stop Marquez
	docker compose -f docker-compose.marquez.yml down

drift:  ## run drift detection (PSI/KS + Evidently report)
	. .venv/bin/activate && python pipelines/drift.py

loop:  ## run the self-healing loop (drift -> retrain -> gate -> auto-promote)
	. .venv/bin/activate && python pipelines/loop.py
