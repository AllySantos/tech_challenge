# Churn Prediction — FIAP Pós Tech ML · Grupo 102

Rede neural (MLP com PyTorch) para prever cancelamento de clientes de uma operadora de telecom.

**Dataset:** [Telco Customer Churn — IBM](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) · 7.043 clientes · 20 features

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
```

---

## Comandos

```bash
| Comando         | O que faz |
|-----------------|-----------|
| make install    | cria .venv e instala todas as dependências |
| make train      | treina a MLP e salva o modelo |
| make evaluate   | compara MLP vs baselines (LogReg, RF, GBM, DT) |
| make lint       | verifica qualidade do código |
| make format     | formata o código |
| make test       | executa os testes |
| make e2e        | executa testes de ponta a ponta |
| make run          | inicia o servidor FastAPI |
| make mlflow-ui    | abre a interface do MLflow |
| make clean        | limpa arquivos gerados |
```

## Jupyter

**1. Instale as dependências:**

```bash
python3 -m venv .venv

# Mac / Linux
source .venv/bin/activate

# Windows (Git Bash)
source .venv/Scripts/activate

pip install -e .
nbstripout --install
```

**2. Abra o Jupyter:**

```bash
jupyter notebook
```

**3. Execute os notebooks em ordem com Kernel → Restart & Run All:**

| Notebook                          | Etapa   | O que faz                                |
| --------------------------------- | ------- | ---------------------------------------- |
| `notebooks/01_eda_baseline.ipynb` | Etapa 1 | Análise exploratória + baselines sklearn |
| `notebooks/02_mlp_pytorch.ipynb`  | Etapa 2 | Treinamento da MLP + métricas + MLflow   |

**4. Veja os experimentos no MLflow:**

```bash
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Acesse: http://localhost:5001

> **Mac:** a porta 5000 é reservada pelo sistema (AirPlay) — use 5001.

## Estrutura do projeto

```
tech_challenge/
├── data/
│   ├── raw/                        # Dataset original (não commitado — coloque aqui)
│   └── processed/                  # Dados pré-processados
├── notebooks/
│   ├── 01_eda_baseline.ipynb       # EDA e baselines
│   └── 02_mlp_pytorch.ipynb        # MLP PyTorch
├── src/
│   ├── app/                        # API em FastAPI
│   ├── ml/                         # Recursos de ML (modelos, pipeline, treino)
├── docs/                           # Model Card, arquitetura
├── .gitattributes                  # Remove outputs de notebooks no commit (nbstripout)
├── Makefile
└── pyproject.toml
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

| Decisão          | Escolha                                | Por quê                                                      |
| ---------------- | -------------------------------------- | ------------------------------------------------------------ |
| Loss             | BCEWithLogitsLoss sem Sigmoid na saída | Mais estável numericamente; evita dupla aplicação de sigmoid |
| Tracking         | MLflow + SQLite (`mlflow.db` na raiz)  | MLflow 3.x descontinuou file-based tracking                  |
| Preprocessamento | sklearn Pipeline (fit só no treino)    | Evita data leakage; pipeline salvo para uso na API           |
| Notebooks        | nbstripout via `.gitattributes`        | Remove outputs automaticamente no commit                     |

---

## Time — Grupo 102

| Nome            | RM     | GitHub                                             |
| --------------- | ------ | -------------------------------------------------- |
| Gabriel Furtado | 371440 | —                                                  |
| Alícia Santos   | 374128 | [@AllySantos](https://github.com/AllySantos)       |
| Rogerio Junior  | 370501 | [@nimesko](https://github.com/nimesko)             |
| Diego Ribeiro   | 370996 | [@diegowribeiro](https://github.com/diegowribeiro) |

---

## Docker

Instruções para rodar o projeto via docker:

1. docker build -f Dockerfile.app . -t churn-prediction .
2. docker run -p 8000:8000 churn-prediction:latest

## AWS Setup

Instruções para criar a infraestrutura na AWS usando Terraform e configurar as permissões necessárias para o deploy:

1. Crie uma conta AWS
2. Crie o arquivo .env na raiz do projeto com as seguintes variáveis:

```
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
```

3. Rode make `plan-aws` para ver os artefatos para criar
4. Rode make `build-aws` para criar a infraestrutura na AWS

## URL do projeto

O projeto está disponível em http://churn-prediction-alb-654855468.us-east-1.elb.amazonaws.com
Temos o endpoint the health check em http://churn-prediction-alb-654855468.us-east-1.elb.amazonaws.com/health

## Critérios de avaliação

| Critério                        | Peso |
| ------------------------------- | ---- |
| Qualidade do código e estrutura | 20%  |
| Rede neural PyTorch             | 25%  |
| Pipeline e reprodutibilidade    | 15%  |
| API de inferência               | 15%  |
| Documentação e Model Card       | 10%  |
| Vídeo STAR                      | 10%  |
| Bônus: deploy em nuvem          | 5%   |
