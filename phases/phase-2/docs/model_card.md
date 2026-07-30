# Model Card — recsys-mlp

## Descrição do Modelo
Rede neural MLP (Multilayer Perceptron) com embeddings para recomendação 
de produtos em e-commerce, treinada sobre o dataset RetailRocket.

## Uso Pretendido
- **Caso de uso primário:** Recomendar produtos para usuários baseado em 
  histórico de navegação
- **Usuários pretendidos:** Times de produto e engenharia de e-commerce
- **Casos de uso fora do escopo:** Recomendação para usuários novos 
  (cold start)

## Dataset
- **Fonte:** RetailRocket E-commerce Dataset (Kaggle)
- **Período:** Dados de comportamento de navegação
- **Volume:** 2.7M eventos brutos → 833K após filtragem
- **Usuários:** 80.112 únicos
- **Itens:** 38.977 únicos
- **Eventos:** view (peso 1), addtocart (peso 3), transaction (peso 5)

## Arquitetura
- **Tipo:** MLP com embeddings de usuário e item
- **Embedding dim:** 32
- **Camadas:** Linear(64) → ReLU → Dropout(0.2) → Linear(32) → ReLU → Linear(1) → Sigmoid
- **Otimizador:** Adam (lr=0.001)
- **Loss:** Binary Cross Entropy
- **Early stopping:** paciência de 5 epochs

## Treinamento
- **Epochs:** 50
- **Batch size:** 256
- **Negative sampling:** 1:1 (positivos:negativos)
- **Seed:** 42
- **Hardware:** NVIDIA RTX 4060 Laptop GPU

## Métricas de Avaliação (@K=10)

| Métrica | Baseline (Popularity) | MLP |
|---|---|---|
| Precision@10 | 0.0046 | 0.0014 |
| Recall@10 | 0.0053 | 0.0044 |
| NDCG@10 | 0.0063 | 0.0028 |
| Hit Rate@10 | 0.0254 | 0.0138 |

## Limitações e Vieses
- **Cold start:** Não recomenda para usuários ou itens sem histórico
- **Avaliação:** Métricas calculadas nos dados de treino — sem split 
  treino/teste adequado
- **Baseline:** O modelo de popularidade superou o MLP nas métricas 
  atuais — indica necessidade de mais epochs e tuning
- **Dados:** IDs de usuários e itens são anonimizados (hashed)

## Próximos Passos
- Implementar split treino/validação/teste
- Aumentar embedding_dim para 64 ou 128
- Testar Matrix Factorization como alternativa
- Adicionar features de conteúdo dos itens

## Versionamento
- **Versão:** 1.0.0
- **MLflow Model Registry:** recsys-mlp @ production
- **Rastreabilidade:** Todos os experimentos logados no MLflow