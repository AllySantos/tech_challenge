# Objetivos — Fase 2

## Objetivo de negócio

Construir um **sistema de recomendação de produtos** para uma empresa de e-commerce, capaz de sugerir itens relevantes a partir do histórico de navegação do usuário, aumentando engajamento e conversão.

## Objetivos técnicos

1. **Modelo neural funcional** — MLP ou embedding-based em PyTorch, com early stopping, batching e seeds fixos.
2. **Pipeline reprodutível** — `dvc repro` regenera todos os artefatos do zero a partir do dataset versionado.
3. **Experimentos rastreáveis** — ≥ 3 runs no MLflow comparando hiperparâmetros e arquiteturas, com promoção para `Production` no Registry.
4. **Ambiente isolado** — `poetry install` num diretório limpo recria o ambiente sem ajustes manuais; lock file commitado.
5. **Imagem Docker enxuta** — Dockerfile multi-stage, imagem final sem dependências de build.
6. **Código profissional** — SOLID, funções ≤ 20 linhas, type hints, docstrings Google style, ruff sem erros, pre-commit hooks.
7. **Comparação com baselines** — pelo menos um baseline sklearn (popularidade, KNN ou item-based CF) com **≥ 4 métricas** lado a lado.

## Critérios de sucesso (definition of done da fase)

| ✅ Critério | Como validar |
|------------|--------------|
| Repositório clonado, `poetry install` e `dvc repro` rodam sem ajustes manuais | Validação em VM/container limpo |
| MLflow tem ≥ 3 runs e modelo promovido a `Production` | UI do MLflow ou `mlflow models list` |
| Dockerfile multi-stage com imagem final < 2 GB (alvo) | `docker images` |
| `ruff check` e `ruff format --check` passando | CI verde |
| Cobertura de testes ≥ 70% nas funções utilitárias e de preprocessamento | `pytest --cov` |
| Model Card preenchido com métricas, vieses e limitações | [model_card.md](model_card.md) (a criar na Etapa 4) |
| Vídeo STAR de 5 min publicado no link do README | Link no README raiz da Fase 2 |

## Métricas que vamos comparar

Para o domínio de recomendação, padrão de mercado e adequadas para o trabalho:

- **Recall@K** (K=5, 10)
- **NDCG@K** (K=5, 10)
- **MAP (Mean Average Precision)**
- **Hit Rate@K**
- **Coverage** e **Diversity** como métricas secundárias (qualidade da recomendação além de acurácia)

A definição final do K e da estratégia de split (temporal vs aleatório) sai na issue de modelagem.

## Restrições

- Reprodutibilidade > performance absoluta. Um modelo mediano que roda em qualquer máquina vale mais que um SOTA não-reproduzível.
- Seeds fixos em numpy, torch, sklearn e DataLoader (`generator=torch.Generator().manual_seed(seed)`).
- Não usar serviços pagos como obrigatórios — o "deploy em nuvem" é opcional/bônus (5%).
