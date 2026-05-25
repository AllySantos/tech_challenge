# Critérios de Avaliação — Fase 2

Pesos oficiais do PDF do Tech Challenge.

| Critério | Peso | O que conta |
|----------|------|-------------|
| **Clean code e estrutura**       | 15% | SOLID, naming, type hints, design patterns, linting |
| **Reprodutibilidade**            | 15% | Poetry, lock file, `.env`, instalação limpa |
| **Docker**                       | 15% | Multi-stage, imagem otimizada, compose funcional |
| **DVC + Pipeline**               | 15% | Dataset versionado, pipeline ≥ 3 stages, `dvc repro` funcional |
| **Rede neural (PyTorch)**        | 15% | MLP funcional, early stopping, comparação com baselines |
| **MLflow + Registry**            | 10% | ≥ 3 runs rastreados, modelo promovido a `Production` |
| **Vídeo STAR**                   | 10% | Clareza, cobertura dos 4 elementos, ≤ 5 min |
| **Bônus: deploy em nuvem**       | +5% | Container acessível via URL pública |

**Total:** 100% obrigatórios + 5% bônus opcional.

## Como o grupo se auto-avalia antes da entrega

Use este formulário rápido na última reunião antes de gravar o vídeo:

- [ ] Clonei o repo numa máquina nova, rodei `poetry install` e `dvc repro` — funcionou sem ajustes?
- [ ] `docker compose up` sobe MLflow + treino sem erro?
- [ ] MLflow UI mostra ≥ 3 runs e tem um modelo em `Production`?
- [ ] Model Card está completo (performance + vieses + limitações)?
- [ ] Vídeo cobre Situation, Task, Action, Result e dura ≤ 5 min?
- [ ] README tem instruções suficientes para alguém de fora rodar o projeto?
