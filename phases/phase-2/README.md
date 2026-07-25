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

## Como executar localmente

Este projeto foi pensado para ser reprodutível do zero, com gerenciamento moderno de dependências via Poetry, configuração externa em .env e validação explícita do ambiente.

### 1. Pré-requisitos

- Python 3.11
- Poetry

### 2. Criar o ambiente e instalar dependências

```bash
cd phases/phase-2
poetry env use 3.11
poetry install
```

No Windows PowerShell, o fluxo é o mesmo:

```powershell
cd phases/phase-2
poetry env use 3.11
poetry install
```

### 3. Configurar variáveis de ambiente

```bash
cp .env.example .env
```

No Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Edite o arquivo .env com os valores apropriados para o seu ambiente.

### 4. Validar se o ambiente está pronto

```bash
poetry run python scripts/validate_env.py
```

Esse script verifica a versão do Python, as dependências principais, a leitura das configurações e a existência dos diretórios esperados.

### 5. Rodar testes

```bash
poetry run pytest tests/unit -q
poetry run pytest tests/e2e -q
```

### 6. Reprodutibilidade garantida

- Sempre que alterar dependências, gere e commite o lock file:

```bash
poetry lock
```

- Mantenha o projeto instalável do zero com:

```bash
poetry install
```

- Para checar uma instalação limpa em ambiente novo, basta repetir os passos acima a partir de um ambiente vazio.

---

## Quadro de andamento

- **Milestone:** [Fase 2 — Sistema de Recomendação](https://github.com/AllySantos/tech_challenge/milestones)
- **Issues abertas:** [label:phase-2](https://github.com/AllySantos/tech_challenge/issues?q=is%3Aissue+label%3Aphase-2)
- **Quadro Kanban:** a definir na primeira reunião do grupo

Para contribuir, comece por [docs/etapas.md](docs/etapas.md) e auto-atribua uma issue que ainda não tem assignee.
