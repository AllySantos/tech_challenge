# Churn Prediction — FIAP Pós Tech ML · Grupo 102

Rede neural (MLP com PyTorch) para prever cancelamento de clientes de uma operadora de telecom.

**Dataset:** [Telco Customer Churn — IBM](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) · 7.043 clientes · 20 features

---

## Status do Projeto

| Etapa | Descrição | Status |
|-------|-----------|--------|
| **Etapa 1** | EDA + ML Canvas + Baselines | ✅ Concluída |
| **Etapa 2** | MLP PyTorch + MLflow + comparação de modelos | ✅ Concluída |
| **Etapa 3** | API FastAPI + testes automatizados | 🔄 Em andamento |
| **Etapa 4** | Model Card + documentação + vídeo STAR | ⏳ Pendente |

---

## Pré-requisitos

- Python 3.11 ou superior
- Git

> **Windows:** use WSL (Ubuntu) ou Git Bash. O comando `make` não funciona no CMD/PowerShell nativo.

---

## Setup inicial

```bash
git clone https://github.com/AllySantos/tech_challenge.git
cd tech_challenge
git checkout setup/project-organization
```

**Baixe o dataset diretamente pelo terminal:**

```bash
curl -L -o data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv \
  https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
```

> Sem terminal ou no Windows: [download direto aqui](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv) — salve como `WA_Fn-UseC_-Telco-Customer-Churn.csv` dentro de `data/raw/`.

---

## Opção A — Via Makefile (Mac / Linux)

```bash
make install      # cria .venv e instala todas as dependências
make train        # treina a MLP e salva o modelo
make evaluate     # compara MLP vs baselines (LogReg, RF, GBM, DT)
make lint         # verifica qualidade do código
make mlflow-ui    # abre os experimentos em http://localhost:5001
```

---

## Opção B — Via Jupyter (qualquer SO)

**1. Instale as dependências:**

```bash
python3 -m venv .venv

# Mac / Linux
source .venv/bin/activate

# Windows (Git Bash)
source .venv/Scripts/activate

pip install -r requirements.txt
nbstripout --install
```

**2. Abra o Jupyter:**

```bash
jupyter notebook
```

**3. Execute os notebooks em ordem com Kernel → Restart & Run All:**

| Notebook | Etapa | O que faz |
|----------|-------|-----------|
| `notebooks/01_eda_baseline.ipynb` | Etapa 1 | Análise exploratória + baselines sklearn |
| `notebooks/02_mlp_pytorch.ipynb` | Etapa 2 | Treinamento da MLP + métricas + MLflow |

**4. Veja os experimentos no MLflow:**

```bash
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Acesse: http://localhost:5001

> **Mac:** a porta 5000 é reservada pelo sistema (AirPlay) — use 5001.

---

## O que o `make` faz por baixo

| Comando | Equivalente manual |
|---|---|
| `make install` | `python3 -m venv .venv && pip install -r requirements.txt && nbstripout --install` |
| `make train` | `cd src && python training/train.py` |
| `make evaluate` | `cd src && python training/evaluate.py` |
| `make lint` | `ruff check src/ tests/` |
| `make mlflow-ui` | `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001` |

---

## Estrutura do projeto

```
tech_challenge/
├── data/
│   ├── raw/                        # Dataset original (não commitado — coloque aqui)
│   └── processed/                  # Dados pré-processados
├── notebooks/
│   ├── 01_eda_baseline.ipynb       # Etapa 1 — EDA e baselines
│   └── 02_mlp_pytorch.ipynb        # Etapa 2 — MLP PyTorch
├── src/
│   ├── model/
│   │   ├── architecture.py         # ChurnMLP: Input→256→128→64→1
│   │   └── artifacts/              # model.pth + pipeline.pkl (gerados pelo treino)
│   ├── pipeline/builder.py         # Pipeline sklearn (Label + OneHot + Scaler)
│   ├── services/                   # MLflowService, DataFrameService, PreprocessingService
│   ├── training/
│   │   ├── train.py                # Loop de treino + early stopping + MLflow
│   │   └── evaluate.py             # Baselines + comparação + análise de threshold
│   └── utils/                      # Loaders, encoders, feature_identifier
├── tests/                          # Testes automatizados (Etapa 3)
├── docs/                           # Model Card, arquitetura (Etapa 4)
├── .gitattributes                  # Remove outputs de notebooks no commit (nbstripout)
├── Makefile
├── pyproject.toml
└── requirements.txt
```

---

## Arquitetura do modelo

```
Input (45 features)
    ↓
Linear(256) → BatchNorm → ReLU → Dropout(0.3)
    ↓
Linear(128) → BatchNorm → ReLU → Dropout(0.3)
    ↓
Linear(64)  → BatchNorm → ReLU → Dropout(0.3)
    ↓
Linear(1)   → logit  [BCEWithLogitsLoss aplica sigmoid internamente]
```

**Treino:** BCEWithLogitsLoss com peso de classe · Adam · EarlyStopping(patience=10) · Gradient clipping

---

## Decisões técnicas

| Decisão | Escolha | Por quê |
|---------|---------|---------|
| Loss | BCEWithLogitsLoss sem Sigmoid na saída | Mais estável numericamente; evita dupla aplicação de sigmoid |
| Tracking | MLflow + SQLite (`mlflow.db` na raiz) | MLflow 3.x descontinuou file-based tracking |
| Preprocessamento | sklearn Pipeline (fit só no treino) | Evita data leakage; pipeline salvo para uso na API |
| Notebooks | nbstripout via `.gitattributes` | Remove outputs automaticamente no commit |

---

## Time — Grupo 102

| Nome | RM | GitHub |
|------|----|--------|
| Gabriel Furtado | 371440 | — |
| Alícia Santos | 374128 | [@AllySantos](https://github.com/AllySantos) |
| Rogerio Junior | — | [@nimesko](https://github.com/nimesko) |
| Junior Silva | 374224 | — |
| Diego Ribeiro | 370996 | [@diegowribeiro](https://github.com/diegowribeiro) |

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
