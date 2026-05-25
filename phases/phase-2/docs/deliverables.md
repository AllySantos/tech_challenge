# Entregáveis — Fase 2

## Entrega obrigatória

### 1. Repositório GitHub

- [ ] Branch `main` (ou `release/phase-2`) com todo o código da Fase 2 sob `phases/phase-2/`.
- [ ] Histórico de commits semântico (`feat:`, `fix:`, `chore:`, `docs:`, `test:`...).
- [ ] `pyproject.toml` + lock file commitado.
- [ ] `.dockerignore`, `.gitignore`, `.env.example` configurados.
- [ ] README com instruções completas de setup, treino e deploy.
- [ ] Model Card em [model_card.md](model_card.md) (criar na Etapa 4).
- [ ] CI verde no GitHub Actions (lint + testes).

### 2. Vídeo STAR — 5 minutos

- [ ] Roteiro escrito antes da gravação (sugestão: 1 min Situation + 1 min Task + 2 min Action + 1 min Result).
- [ ] Cobre os 4 elementos: **Situation, Task, Action, Result**.
- [ ] Duração efetiva ≤ 5 minutos.
- [ ] Link incluído no README da Fase 2.

## Entrega opcional (bônus +5%)

### 3. Deploy em nuvem

- [ ] Container do modelo acessível via URL pública (AWS, Azure ou GCP).
- [ ] Endpoint testado com `curl` ou `httpx` documentado no README.
- [ ] Trabalho de infra commitado (Terraform, Pulumi ou similar) — ou ao menos o script `deploy.sh`.

## Checklist por etapa

### Etapa 1 — Clean Code e Estrutura
- [ ] Estrutura `src/`, `tests/`, `data/`, `models/`, `configs/` criada
- [ ] Naming conventions consistentes e SOLID aplicado desde a primeira linha
- [ ] ≥ 1 design pattern implementado (Factory de modelos OU Strategy de preprocessors)
- [ ] Type hints em todas as funções públicas + docstrings Google style
- [ ] `ruff check` e `ruff format --check` sem erros
- [ ] Pre-commit hooks configurados (`pre-commit install`)
- [ ] **Entregável:** Repositório base com estrutura limpa e linting passando

### Etapa 2 — Ambiente e Dependências
- [ ] `pyproject.toml` com Poetry, deps prod (`torch`, `scikit-learn`, `mlflow`, `dvc`) e dev (`pytest`, `ruff`, `pre-commit`) separadas
- [ ] `poetry.lock` gerado e commitado
- [ ] Configurações externalizadas via `.env` + Pydantic Settings
- [ ] `scripts/validate_env.py` que valida Python, dependências e variáveis críticas
- [ ] Verificado em ambiente novo (VM ou container) que `poetry install` funciona do zero
- [ ] **Entregável:** Projeto instalável do zero com `poetry install`

### Etapa 3 — Containerização e Versionamento
- [ ] Dockerfile multi-stage (`builder` para deps + `runtime` para app)
- [ ] `docker-compose.yml` com serviço de treino + MLflow server
- [ ] `dvc init` rodado, dataset versionado, remote configurado (local ou S3)
- [ ] Pipeline `dvc.yaml` com pelo menos 3 stages: `preprocess → feature_eng → train → evaluate`
- [ ] MLflow tracking integrado: params, métricas e artefatos em cada run
- [ ] **Entregável:** Pipeline reprodutível via `dvc repro` + Docker funcional

### Etapa 4 — Rede Neural, Registry e Entrega
- [ ] MLP / embedding-based em PyTorch treinada
- [ ] Early stopping implementado
- [ ] Comparação com ≥ 1 baseline sklearn usando ≥ 4 métricas
- [ ] Modelo registrado no MLflow Model Registry → promovido `Staging` → `Production`
- [ ] Model Card escrito com performance, limitações e vieses
- [ ] README final com instruções completas
- [ ] Vídeo STAR gravado e link no README
- [ ] (Opcional) Deploy em nuvem rodando
- [ ] **Entregável:** Repositório final + modelo no Registry + vídeo STAR
