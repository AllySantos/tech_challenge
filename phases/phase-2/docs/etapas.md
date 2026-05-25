# Etapas de Desenvolvimento — Fase 2

Quatro etapas, alinhadas com as quatro disciplinas da fase. Cada uma tem entregável claro.

---

## Etapa 1

**Foco:** projeto limpo com padrões de engenharia desde o início.

| # | Tarefa | Referência |
|---|--------|------------|
| 1.1 | Definir estrutura de projeto com `src/`, `tests/`, `data/`, `models/`, `configs/` | Clean Code, Aula 01 |
| 1.2 | Aplicar naming conventions e SOLID desde a primeira linha | Clean Code, Aulas 01–02 |
| 1.3 | Implementar ≥ 1 design pattern (Factory para criar modelos, Strategy para preprocessors) | Clean Code, Aula 03 |
| 1.4 | Type hints em todas as funções públicas + docstrings Google style | Clean Code, Aula 03 |
| 1.5 | Configurar `ruff` sem erros + pre-commit hooks | Clean Code, Aula 03 |

**Entregável:** repositório base com estrutura limpa e linting passando.

---

## Etapa 2

**Foco:** reprodutibilidade garantida com gerenciamento moderno de dependências.

| # | Tarefa | Referência |
|---|--------|------------|
| 2.1 | Configurar `pyproject.toml` com Poetry: deps de prod (torch, sklearn, mlflow) e dev (pytest, ruff) | Dependências, Aula 02 |
| 2.2 | Gerar e commitar `poetry.lock` | Dependências, Aula 03 |
| 2.3 | Externalizar configurações para `.env` + Pydantic Settings | Dependências, Aula 03 |
| 2.4 | Script de validação de ambiente (`scripts/validate_env.py`) | Dependências, Aula 01 |
| 2.5 | Verificar instalação limpa em ambiente novo | Dependências, Aula 02 |

**Entregável:** projeto instalável do zero com `poetry install`.

---

## Etapa 3

**Foco:** Docker + DVC + MLflow integrados em pipeline reprodutível.

| # | Tarefa | Referência |
|---|--------|------------|
| 3.1 | Dockerfile multi-stage: builder (deps) + runtime (app) | Docker, Aulas 01–03 |
| 3.2 | `docker-compose.yml` com serviço de treino + MLflow server | Docker, Aula 04 |
| 3.3 | `dvc init`, versionar dataset, configurar remote (local ou S3) | DVC+MLflow, Aulas 01–03 |
| 3.4 | Pipeline DVC (`dvc.yaml`): `preprocess → feature_eng → train → evaluate` | DVC+MLflow, Aula 04 |
| 3.5 | MLflow tracking: logar params, métricas e artefatos em cada run | DVC+MLflow, Aula 04 |

**Entregável:** pipeline reprodutível via `dvc repro` + Docker funcional.

---

## Etapa 4

**Foco:** modelo neural treinado, registrado e documentado.

| # | Tarefa | Referência |
|---|--------|------------|
| 4.1 | Treinar MLP / embedding-model com PyTorch para recomendação | — (PyTorch) |
| 4.2 | Comparar com baselines (Scikit-Learn) usando ≥ 4 métricas | — (Scikit-Learn) |
| 4.3 | Registrar modelo no MLflow Model Registry → Staging → Production | DVC+MLflow, Aula 05 |
| 4.4 | Escrever Model Card com performance, limitações e vieses | — (boas práticas) |
| 4.5 | Finalizar README com instruções completas | — |
| 4.6 | Gravar vídeo STAR de 5 minutos | — |
| 4.7 | (Opcional) Deploy em nuvem via Docker | — |

**Entregável:** repositório final + modelo no Registry + vídeo STAR.

---

## Definição de pronto (DoR e DoD do grupo)

**Definition of Ready** — antes de pegar uma issue:
- A issue tem critérios de aceitação claros.
- Você sabe em qual etapa ela se encaixa.
- As dependências (issues que precisam ser fechadas antes) estão resolvidas.

**Definition of Done** — antes de marcar como completa:
- Código com testes (unit ≥ 70% cov no que adicionou; e2e/integration onde fizer sentido).
- `ruff check` e `ruff format --check` passando.
- PR aberto com descrição linkando para a issue.
- Pelo menos 1 review aprovada do grupo.
- CI verde.
