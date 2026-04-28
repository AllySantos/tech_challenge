# Churn Prediction — FIAP Pós Tech ML · Grupo 102

Rede neural (MLP com PyTorch) para prever cancelamento de clientes de uma operadora de telecom, com pipeline profissional end-to-end.

**Dataset:** [Telco Customer Churn — IBM](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) · 7.043 clientes · 20 features

---

## Status do Projeto

| Etapa | Descrição | Status |
|-------|-----------|--------|
| **Etapa 1** | EDA + ML Canvas + Baselines | ✅ Concluída |
| **Etapa 2** | MLP PyTorch + comparação de modelos | ✅ Concluída |
| **Etapa 3** | Refatoração + API FastAPI + Testes | 🔄 Em andamento |
| **Etapa 4** | Model Card + Documentação + Vídeo STAR | ⏳ Pendente |

---

## O que estamos construindo

Uma operadora de telecom está perdendo clientes. Construímos um modelo preditivo que identifica clientes com risco de cancelamento, servido via API REST.

```
Dataset → EDA → Baselines → MLP (PyTorch) → MLflow → API (FastAPI) → Deploy
```

**Métricas-alvo:** AUC-ROC > 0.85 · PR-AUC > 0.65 · F1 > 0.65

---

## Como rodar

### 1. Pré-requisitos
- Python 3.11+
- Git

### 2. Clone e instale

```bash
git clone https://github.com/AllySantos/tech_challenge.git
cd tech_challenge
make install
source .venv/bin/activate
```

### 3. Coloque o dataset

Baixe o arquivo `WA_Fn-UseC_-Telco-Customer-Churn.csv` e coloque em:
```
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

> Você pode baixar direto do [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) ou pedir no grupo do Discord.

### 4. Execute os notebooks em ordem

```bash
jupyter notebook notebooks/
```

| Arquivo | O que faz |
|---------|-----------|
| `01_eda_baseline.ipynb` | Análise exploratória + modelos baseline |
| `02_mlp.ipynb` | Treinamento da rede neural MLP |

### 5. Veja os experimentos no MLflow

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Acesse: http://localhost:5000
```

### 6. Suba a API (Etapa 3 — em construção)

```bash
# Em breve
uvicorn src.api.main:app --reload
```

---

## Estrutura do projeto

```
tech_challenge/
├── data/
│   ├── raw/                  # Dataset original (não commitado)
│   └── processed/            # Dados pré-processados
├── docs/                     # Model Card, arquitetura, monitoramento
├── models/                   # Artefatos do modelo treinado (.pth, .pkl)
├── notebooks/
│   ├── 01_eda_baseline.ipynb # Etapa 1
│   └── 02_mlp.ipynb          # Etapa 2
├── src/
│   ├── model/                # Arquitetura MLP (PyTorch)
│   ├── pipeline/             # Pipeline sklearn (preprocessamento)
│   ├── services/             # MLflow, DataFrame, Preprocessing
│   ├── training/             # Train loop, evaluate
│   └── utils/                # Encoders, loaders, feature_identifier
├── tests/                    # Testes automatizados (pytest)
├── Makefile                  # install, lint, test, run
├── pyproject.toml            # Dependências + configuração do projeto
└── requirements.txt
```

---

## Decisões técnicas

| Decisão | Escolha | Por quê |
|---------|---------|---------|
| Modelo principal | MLP PyTorch (256→128→64→1) | Requisito do challenge |
| Baselines | DummyClassifier + Regressão Logística | Referência de comparação |
| Tracking | MLflow + SQLite | Leve, sem servidor externo |
| API | FastAPI + Pydantic | Performance + validação automática |
| Preprocessamento | sklearn Pipeline | Reprodutibilidade garantida |
| Ativação oculta | ReLU + BatchNorm + Dropout(0.3) | Regularização + estabilidade |
| Saída | Sigmoid | Probabilidade de churn (0–1) |

---

## Time — Grupo 102

| Nome | RM | GitHub | Contribuição |
|------|----|--------|--------------|
| Gabriel Furtado | 371440 | — | EDA, MLP, ML Canvas |
| Alícia Santos | 374128 | [@AllySantos](https://github.com/AllySantos) | MLP, MLflow, repo |
| Rogerio Junior | — | [@nimesko](https://github.com/nimesko) | Estrutura src/, pip |
| Junior Silva | 374224 | — | — |
| Diego Ribeiro | 370996 | [@diegowribeiro](https://github.com/diegowribeiro) | Organização, infra |

---

## Critérios de avaliação

| Critério | Peso |
|----------|------|
| Qualidade do código e estrutura | 20% |
| Rede neural PyTorch | 25% |
| Pipeline e reprodutibilidade | 15% |
| API de inferência | 15% |
| Documentação e Model Card | 10% |
| Vídeo STAR | 10% |
| Bônus: deploy em nuvem | 5% |
