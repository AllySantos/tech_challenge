# Enunciado — Tech Challenge Fase 2

> Transcrição literal do PDF do desafio, organizada em seções para referência rápida durante o desenvolvimento.

## Apresentação

Tech Challenge é o projeto da fase que engloba os conhecimentos obtidos em todas as disciplinas. **Atividade em grupo, obrigatória e avaliada.** Vale **90% da nota** de todas as disciplinas da fase.

- **Entrega obrigatória:** Repositório GitHub + Vídeo de 5 minutos (método STAR).
- **Entrega opcional:** Deploy em ambiente de produção em nuvem (AWS, Azure ou GCP).

## O problema

Uma empresa de **e-commerce** precisa de um sistema de **recomendação de produtos** baseado no comportamento de navegação dos usuários. O modelo central é uma **rede neural (MLP ou embedding-based) treinada com PyTorch**, com:

- Pipeline completo containerizado em **Docker**
- Dados versionados com **DVC**
- Experimentos rastreados no **MLflow**
- Código seguindo padrões profissionais de **clean code**

## Vídeo (5 minutos — método STAR)

- **Situation:** Problema de negócio e contexto do dataset.
- **Task:** Objetivos técnicos e restrições.
- **Action:** Decisões de arquitetura, modelo, versionamento e containerização.
- **Result:** Resultados obtidos, trade-offs e lições aprendidas.

## Bibliotecas obrigatórias

| Biblioteca       | Uso                                                  |
| ---------------- | ---------------------------------------------------- |
| **PyTorch**      | Rede neural para o modelo de recomendação            |
| **Scikit-Learn** | Pré-processamento e baselines                        |
| **MLflow**       | Tracking de experimentos e Model Registry            |
| **DVC**          | Versionamento de dados e pipeline reprodutível       |

## Requisitos obrigatórios

### Repositório GitHub

- Estrutura clean code: módulos curtos, nomes descritivos, **SOLID**, type hints.
- `pyproject.toml` com Poetry/uv, dependências prod/dev separadas, **lock file commitado**.
- `.dockerignore`, `.gitignore`, `.env.example` configurados.
- Histórico de **commits semântico**.

### Boas práticas obrigatórias

- Clean code: **funções ≤ 20 linhas**, naming conventions, type hints.
- Design patterns aplicados: **Factory, Strategy ou Template Method** (no mínimo um).
- **Dockerfile multi-stage** com imagem otimizada.
- **Pipeline DVC com ≥ 3 stages.**
- Seeds fixados, lock file, `.env`.

## Dataset sugerido

Dataset de interações de e-commerce, com no mínimo **10.000 interações user-item**. Sugestões:

- [Instacart Market Basket](https://www.kaggle.com/c/instacart-market-basket-analysis/data)
- [RetailRocket E-commerce dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)
- [MovieLens (small ou 1M)](https://grouplens.org/datasets/movielens/) — alternativa mais leve

A escolha será feita na issue [\[Etapa 0\] Definir dataset da Fase 2](#) — veja a milestone.

## Passo a passo resumido

| Etapa | Foco | Detalhes |
|-------|------|----------|
| **1** | Clean code + design patterns + linting | [docs/etapas.md#etapa-1](etapas.md#etapa-1) |
| **2** | Poetry + lock file + `.env` + validação de ambiente | [docs/etapas.md#etapa-2](etapas.md#etapa-2) |
| **3** | Docker multi-stage + DVC pipeline + MLflow tracking | [docs/etapas.md#etapa-3](etapas.md#etapa-3) |
| **4** | MLP PyTorch + Model Registry + Model Card + vídeo STAR | [docs/etapas.md#etapa-4](etapas.md#etapa-4) |
