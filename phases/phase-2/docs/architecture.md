# Arquitetura — Fase 2

> Documento vivo. Será preenchido na **Etapa 3** quando o pipeline DVC e a topologia Docker estiverem definidos.

## Diagrama lógico (rascunho)

```
                   ┌──────────────────────┐
                   │  Dataset (DVC)       │
                   │  s3://... ou local   │
                   └──────────┬───────────┘
                              │ dvc pull
                              ▼
   ┌─────────────────────────────────────────────────────┐
   │  Pipeline DVC (dvc.yaml)                            │
   │                                                     │
   │   preprocess  →  feature_eng  →  train  →  evaluate │
   │       │              │             │         │      │
   │       └──────────────┴─────────────┴─────────┘      │
   │                       │                              │
   │                       ▼ params, metrics, artifacts   │
   └───────────────────┬────────────────────┬────────────┘
                       │                    │
                       ▼                    ▼
              ┌─────────────────┐   ┌────────────────────┐
              │ MLflow Tracking │   │ MLflow Registry    │
              │  (runs / metrs) │   │ Staging→Production │
              └─────────────────┘   └────────────────────┘
```

## Componentes

| Componente            | Responsabilidade                                                          |
| --------------------- | ------------------------------------------------------------------------- |
| `src/data/`           | Carregamento, split temporal/aleatório e amostragem negativa              |
| `src/features/`       | Encoders, normalizadores, montagem de embeddings (Strategy pattern)       |
| `src/models/`         | MLP e variantes embedding-based (Factory pattern)                         |
| `src/training/`       | Loop de treino, early stopping, callbacks, log no MLflow                  |
| `src/evaluation/`     | Recall@K, NDCG@K, MAP, HR e métricas secundárias                          |
| `configs/`            | Hiperparâmetros e seeds (1 YAML por experimento)                          |
| `dvc.yaml`            | Pipeline reprodutível: preprocess → feature_eng → train → evaluate        |
| `docker-compose.yml`  | Treino + MLflow server + (opcional) MinIO como backend de artefatos       |

## Decisões em aberto

Cada item vira uma issue. Decisões aqui são tomadas pelo grupo e documentadas neste doc + no PR que as implementa.

- [ ] Dataset definitivo (issue **Definir dataset da Fase 2**)
- [ ] Estratégia de split: temporal vs leave-one-out vs aleatório
- [ ] Negative sampling: 1:N ratio?
- [ ] Embedding dim e arquitetura do MLP
- [ ] Backend do MLflow: SQLite local OU PostgreSQL no compose
- [ ] Storage de artefatos: filesystem OU MinIO no compose OU S3
