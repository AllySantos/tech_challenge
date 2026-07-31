# Tech Challenge — FIAP Pós Tech ML · Grupo 102

Mono-repositório do grupo de estudo para os Tech Challenges da Pós-Graduação em **Machine Learning Engineering** da FIAP.

Cada fase vive em sua própria pasta sob [`phases/`](phases/) com `README`, dependências e Dockerfiles próprios — você pode rodar **qualquer fase isoladamente** sem interferir nas outras.

---

## Fases

| Fase | Tema                                | Status                                      | README                                                | Milestone                                                                                    |
| ---- | ------------------------------------ | -------------------------------------------- | ------------------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| 1    | Churn Prediction (Telco)             | ✅ Entregue                                  | [phases/phase-1/README.md](phases/phase-1/README.md) | —                                                                                              |
| 2    | Sistema de Recomendação E-commerce   | 🔧 Pipeline funcional · finalizando entrega | [phases/phase-2/README.md](phases/phase-2/README.md) | [Fase 2 — Sistema de Recomendação](https://github.com/AllySantos/tech_challenge/milestones)   |

> Novas fases entram como `phases/phase-N/` seguindo o mesmo padrão.

---

## Começando

```bash
git clone https://github.com/AllySantos/tech_challenge.git
cd tech_challenge

# Trabalhar numa fase específica:
cd phases/phase-2
# ... siga as instruções do README da fase
```

**Pré-requisitos gerais:** Python 3.11+, Git, Docker (a partir da Fase 2).

> **Windows:** use WSL (Ubuntu) ou Git Bash. Comandos `make` e shell scripts não funcionam no CMD/PowerShell nativo.

---

## Como o grupo trabalha

1. **Issues e milestones** — cada fase tem uma milestone no GitHub. Cada tarefa é uma issue com label `phase-N` e label da etapa (`etapa-1`...`etapa-4`).
2. **Auto-atribuição** — issues nascem sem assignee. Pegue uma que esteja `pending`, atribua a si mesmo e mova para `in progress`.
3. **Branch por issue** — `<tipo>/<nome-curto>` (ex.: `feat/dvc-pipeline`, `fix/early-stopping`). Cada branch parte do `main`.
4. **Commits semânticos** — `feat:`, `fix:`, `docs:`, `chore:`, `test:`, `refactor:`. Mensagens em português ou inglês, mas consistentes na fase.
5. **PR review obrigatório** — pelo menos 1 aprovação antes de mergear. CI verde é pré-requisito.

Detalhes em [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Estrutura do repositório

```
tech_challenge/
├── README.md                       # este hub
├── CONTRIBUTING.md                 # como contribuir (compartilhado)
├── CODE_OF_CONDUCT.md
├── SECURITY.md
├── LICENSE
├── .github/
│   ├── workflows/
│   │   └── phase-1-ci.yml          # CI específica da Fase 1 (path-filter em phases/phase-1/**)
│   ├── ISSUE_TEMPLATE/
│   └── PULL_REQUEST_TEMPLATE.md
└── phases/
    ├── phase-1/                    # Churn Prediction — código completo, MLflow, AWS infra
    │   ├── README.md
    │   ├── Makefile
    │   ├── pyproject.toml
    │   ├── Dockerfile.app
    │   ├── Dockerfile.training
    │   ├── src/  tests/  notebooks/  data/  scripts/  models/  docs/
    │
    └── phase-2/                    # Recomendação E-commerce — pipeline Docker + DVC + MLflow funcional
        ├── README.md
        ├── docker/                 # Dockerfile.app (multi-stage)
        ├── docker-compose.yml      # serviços: mlflow, train, api
        ├── dvc.yaml / dvc.lock     # pipeline: preprocess → feature_eng → train → evaluate
        ├── docs/                   # challenge, objetivos, etapas, avaliação, model_card
        ├── src/                    # configs, data, features, models, evaluation, app (API)
        ├── scripts/                # validate_env, compare_models, register_model
        ├── tests/  notebooks/  data/  models/  metrics/
        └── .env.example
```

---

## Comunidade e padrões

- [CONTRIBUTING.md](CONTRIBUTING.md) — convenções de branch, commit e PR.
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) — comportamento esperado entre membros do grupo.
- [SECURITY.md](SECURITY.md) — como reportar problemas de segurança.

---

## Licença

Distribuído sob a licença em [LICENSE](LICENSE).
