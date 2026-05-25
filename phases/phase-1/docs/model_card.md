# Churn Model Card

## Model Details

O Chrun Predction Model é um modelo de rede neural classificatório cujo objetivo é, a partir de features prédeterminadas, prever a probabilidade de um cliente cancelar o contrato com a empresa de telecomunicações Telco, trazendo o resultado de forma binária (Churn = Yes ou Churn = No) 

### Model Date

Maio de 2026

### Model Type

Rede Neural de Classificação Binária (MLP — PyTorch)         

### Model Version

churn-prediction-model-v1.0

## Model Architecture

### Design

O modelo foi construído utilizando redes neurais via PyTorch,. A arquitetura possui 3 camadas que, após a entrada, vão se reduzindo pela metade até a saída ser um único neurônio com a predição final.

| Camada      | Dimensão                  | Ativação        |
|-------------|---------------------------|-----------------|
| 1ª          | Entrada (45) → 64 neurônios | ReLU          |
| 2ª          | 64 → 32 neurônios         | ReLU            |
| 3ª (saída)  | 32 → 1 neurônio           | — (logit bruto) |

A função de ativação ReLU (Rectified Linear Unit) foi adotada nas camadas ocultas de forma a manter os gradientes que atualizam os pesos durante a retropropagação, garantindo que a rede neural continue se aprimorando a cada época. A camada de saída não utiliza função sigmoide explícita, pois essa operação é incorporada na função de perda `BCEWithLogitsLoss` durante o treinamento.

### Training Strategy

O modelo foi treinado em até 100 épocas, limite que garante um aprendizado estável sem que o modelo estagne.

**Função de perda:** Foi adotada a `BCEWithLogitsLoss`, que combina internamente uma função sigmoide com entropia cruzada binária. Essa escolha é especialmente adequada ao contexto de desbalanceamento de classes (73% No Churn / 27% Churn), pois elimina instabilidades numéricas ao operar diretamente sobre os logits brutos.

**Otimizador:** O otimizador utilizado foi o Adam (Adaptive Moment Estimation), com learning rate de `0.001`. O valor baixo foi adotado como forma de estabilizar o treino e garantir atualizações de pesos consistentes.

**Early Stopping:** Para evitar overfitting, foi gerado um dataset de validação representando 16% do dataset total. Esse dataset é monitorado a cada época: caso o modelo não apresente melhora na loss de validação por 10 épocas consecutivas, o treinamento é interrompido automaticamente. Além disso, o estado do modelo é salvo a cada melhora na loss de validação, garantindo que a versão final utilizada seja a de melhor desempenho e não necessariamente a da última época executada.


### Inputs

Os dados aceitos de entrada do modelo são:

```json
{
  "customerID": "string",
  "gender": "string",
  "SeniorCitizen": "integer",
  "Partner": "string",
  "Dependents": "string",
  "tenure": "integer",
  "PhoneService": "string",
  "MultipleLines": "string",
  "InternetService": "string",
  "OnlineSecurity": "string",
  "OnlineBackup": "string",
  "DeviceProtection": "string",
  "TechSupport": "string",
  "StreamingTV": "string",
  "StreamingMovies": "string",
  "Contract": "string",
  "PaperlessBilling": "string",
  "PaymentMethod": "string",
  "MonthlyCharges": "float",
  "TotalCharges": "float",
  "Churn": "string"
}
```
Para processar os dados e garantir que eles estejam normalizados para serem utilizados pelo modelo, eles passam por dois processos de pré-processamento:
* **Standard Scaler:** Normaliza dados numéricos para que o modelo tenha a dimensão correta das grandezas de cada feature
* **One-Hot Encoder:** Tranforma dados categóricos (textuais) em valores numéricos (1 e 0)


### Outputs

Como saída o modelo traz o valor se o cliente irá ou não cancelar o contrato
```json
{
    "churn" : int // 1 = Churn (Yes)  |  0 = Sem churn (No)
}
```

## Model Data

### Training Dataset

Os dados utilizados foram extraídos do dataset público da empresa Telco, contendo 7.043 registros de clientes.

| Split     | Proporção | Linhas | Uso              |
|-----------|-----------|--------|------------------|
| Treino    | 70%       | 4.506  | Ajuste de pesos  |
| Validação | 16%       | 1.127  | Early stopping   |
| Teste     | 14%       | 1.409  | Avaliação final  |


Do total de dados obtidos, a distribuição, considerando a variável target, é a seguinte:

Churn = No -> 73% do Dataset
Churn = Yes -> 27% do Dataset

Apesar do desbalanceamento de classes, esse fator foi equilibrado durante a criação e treinamento da rede neural, priorizando funções e recursos (como visto na seção Model Architecture) que ajudem a lidar com esse fator, garantindo que o modelo não priorize indevidamente a classe majoritária.

## Model Evaluation

### Metrics

Abaixo seguem as métricas da versão disponível do modelo:

**Target Accuracy:** `60.0%`

| Metric | Score |
|--------|-------|
| Accuracy | 0.82 |
| F1 Score | 0.60 |
| Cost | 500.00 |
| Precision | 0.70 |
| Recall | 0.53 |

### Quantitative Analysis

O modelo presente, com os dados de treino obtidos do dataset da empresa Telco apresenta os seguintes dados quantitativos:

| Métrica | Valor | Significado |
|---------|-------|-------------|
| **Verdadeiros Negativos (TN)** | 936 | Clientes que não churnam e foram corretamente identificados ✓ |
| **Falsos Positivos (FP)** | 99 | Clientes que não churnam, mas o modelo previu churn (custo: campanha desnecessária) |
| **Falsos Negativos (FN)** | 185 | Clientes que churnam mas o modelo não previu (maior custo: perda de receita) |
| **Verdadeiros Positivos (TP)** | 189 | Clientes que churnam e foram corretamente identificados ✓ |


### Business Analysis

Premissas de custo: campanha desnecessária (FP) = R$ 50 · receita perdida por churn não detectado (FN) = R$ 500

| Tipo de Erro            | Cálculo        | Custo      |
|-------------------------|----------------|------------|
| Falsos Positivos (99)   | 99 × R$ 50     | R$ 4.950   |
| Falsos Negativos (185)  | 185 × R$ 500   | R$ 92.500  |
| **Total**               | —              | **R$ 97.450** |

**Contexto de negócio:** como referência comparativa, um modelo baseline que classifica todos os clientes como "Sem Churn" geraria um custo de 374 × R$ 500 = **R$ 187.000** em receita perdida — o equivalente a todo o volume de churns no conjunto de teste. O modelo atual reduz esse custo em aproximadamente **48%**, demonstrando valor operacional mesmo com espaço para melhorias no recall.

## Known Limitations

Devido a falta de amplitude temporal do dataset, o modelo não é sensivel a mudanças sazonais, o que pode acarretar em um comportamento inesperado conforme o lançamento de novos produtos/serviços ou a época do ano.

Com recall de 53%, o modelo não identifica ~47% dos churns reais. Aplicações que priorizem cobertura devem considerar ajuste do threshold ou retreinamento com dados adicionais.

O modelo foi treinado exclusivamente com dados da **Telco**. Aplicação em outras empresas ou segmentos exige revalidação e, possivelmente, retreinamento

## Intended Use

O modelo é destinado ao uso operacional dos analistas das equipes de Relacionamento com o Cliente da empresa Telco. Deve ser utilizado exclusivamente para clientes consumidores dos produtos e serviços da Telco, como ferramenta de apoio à decisão em campanhas de retenção.

O modelo **não** deve ser utilizado nas seguintes situações:

- Como único critério para decisões de rescisão de contratos ou ações legais contra clientes.
- Para segmentos fora da base de clientes da Telco, sem revalidação prévia do desempenho.
- Como substituto de análise humana em casos de alto impacto financeiro ou reputacional.