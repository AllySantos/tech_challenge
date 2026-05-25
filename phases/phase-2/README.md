# Fase 2 — Sistema de Recomendação para E-commerce

> **Status:** 🚧 Em desenvolvimento &nbsp;·&nbsp; **Disciplinas:** 01 (Clean Code) · 02 (Dependências) · 03 (Docker) · 04 (DVC + MLflow + PyTorch)

Rede neural (MLP / embedding-based) treinada com PyTorch para recomendar produtos a partir do comportamento de navegação de usuários, com pipeline reprodutível (DVC), experimentos rastreados (MLflow) e tudo containerizado (Docker).

---

## Documentação da fase

| Documento                                | O que tem dentro                                                     |
| ---------------------------------------- | -------------------------------------------------------------------- |
| [docs/challenge.md](docs/challenge.md)   | Enunciado completo (transcrição do PDF do Tech Challenge)            |
| [docs/objectives.md](docs/objectives.md) | Objetivos técnicos e de negócio + critérios de sucesso               |
| [docs/deliverables.md](docs/deliverables.md) | Lista exata do que precisa ser entregue + checklist por etapa     |
| [docs/etapas.md](docs/etapas.md)         | As 4 etapas com tarefas, referências de aula e definição de pronto   |
| [docs/evaluation.md](docs/evaluation.md) | Critérios de avaliação com pesos (15% / 15% / 15% / 15% / 15% / 10% / 10% / +5%) |
| [docs/architecture.md](docs/architecture.md) | Diagrama lógico planejado (será preenchido pelo grupo na Etapa 3) |

---

## Estrutura planejada da fase

```
phases/phase-2/
├── README.md               # este arquivo
├── pyproject.toml          # Poetry (Etapa 2)
├── poetry.lock             # commitado (Etapa 2)
├── .env.example            # Pydantic Settings (Etapa 2)
├── Dockerfile              # multi-stage (Etapa 3)
├── docker-compose.yml      # treino + MLflow server (Etapa 3)
├── dvc.yaml                # pipeline ≥ 3 stages (Etapa 3)
├── configs/                # YAML / TOML de hiperparâmetros e features
├── src/
│   ├── data/               # loaders e splits
│   ├── features/           # pré-processadores (Strategy pattern)
│   ├── models/             # MLP / embedding (Factory pattern)
│   ├── training/           # loop de treino, early stopping
│   └── evaluation/         # métricas (Recall@K, NDCG@K, MAP, HR)
├── tests/
│   ├── unit/
│   └── e2e/
├── notebooks/              # EDA inicial — código de produção vive em src/
├── scripts/                # CLIs e validação de ambiente
├── data/                   # versionado por DVC, não commitado
└── docs/                   # documentação da fase
```

---

## Como rodar (vai ser preenchido conforme as Etapas forem entregues)

```bash
cd phases/phase-2

# Etapa 2 — instalação
poetry install
cp .env.example .env

# Etapa 3 — pipeline reprodutível
dvc pull
dvc repro

# Etapa 3 — Docker
docker compose up --build

# Etapa 4 — promoção do modelo
mlflow models serve -m "models:/recsys/Production"
```

---

## Quadro de andamento

- **Milestone:** [Fase 2 — Sistema de Recomendação](https://github.com/AllySantos/tech_challenge/milestones)
- **Issues abertas:** [label:phase-2](https://github.com/AllySantos/tech_challenge/issues?q=is%3Aissue+label%3Aphase-2)
- **Quadro Kanban:** a definir na primeira reunião do grupo

Para contribuir, comece por [docs/etapas.md](docs/etapas.md) e auto-atribua uma issue que ainda não tem assignee.
