.DEFAULT_GOAL := help
SHELL := /bin/bash

POETRY ?= poetry
COMPOSE ?= docker compose

.PHONY: help install train serve up down logs load bench report test lint fmt airflow clean

help: ## Lista os alvos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Instala as dependências do projeto
	$(POETRY) install --with dev

train: ## Roda o pipeline completo: ingestão → treino → ONNX → benchmark → promoção
	$(POETRY) run python -m src.pipeline

bench: ## Refaz o benchmark de latência sobre o modelo promovido
	$(POETRY) run python -m src.evaluation.benchmark

report: ## Imprime a tabela comparativa de latência em Markdown
	$(POETRY) run python scripts/report_latency.py

serve: ## Sobe a API localmente com reload
	$(POETRY) run uvicorn src.app.api:app --reload --port 8000

up: ## Sobe API + Prometheus + Grafana
	$(COMPOSE) up -d --build api prometheus grafana

airflow: ## Sobe o Airflow (profile orchestration)
	$(COMPOSE) --profile orchestration up -d --build airflow

load: ## Gera tráfego contra a API para popular os painéis
	$(COMPOSE) --profile load run --rm loadgen

logs: ## Acompanha os logs da stack
	$(COMPOSE) logs -f

down: ## Derruba a stack preservando os volumes
	$(COMPOSE) --profile orchestration --profile load down

test: ## Roda a suíte de testes com cobertura
	$(POETRY) run pytest --cov=src --cov-report=term-missing

lint: ## Verifica lint e formatação
	$(POETRY) run ruff check src/ tests/ scripts/ airflow/
	$(POETRY) run ruff format --check src/ tests/ scripts/ airflow/

fmt: ## Aplica a formatação
	$(POETRY) run ruff format src/ tests/ scripts/ airflow/
	$(POETRY) run ruff check --fix src/ tests/ scripts/ airflow/

clean: ## Remove artefatos gerados
	rm -rf data/raw/*.csv data/processed/*.csv models/*/ models/current.json \
		metrics/*.json reports/*.json .pytest_cache .ruff_cache
