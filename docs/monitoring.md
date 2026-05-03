# Monitoring Plan | Churn Model Prediction

Este documento define as diretrizes, métricas, frequências e critérios de ação para o monitoramento contínuo do Churn Prediction Model em produção. Seu objetivo é garantir que o modelo mantenha desempenho aceitável ao longo do tempo e que degradações sejam detectadas e tratadas antes de gerar impacto de negócio relevante.

## Evaluation Metrics - Alerts

### Model Metrics

As métricas do modelo são calculadas periodicamente comparando as predições do modelo com os rótulos reais (churn confirmado ou não), coletados com defasagem natural do negócio.
 
| Métrica do Alerta   | Valor de Referência (baseline) | Limiar de Alerta | Limiar Crítico | Prioridade |
|-----------|-------------------------------|------------------|----------------|------------|
| Recall    | 0.53                          | < 0.48           | < 0.43         | 🔴 Alta    |
| Accuracy  | 0.82                          | < 0.77           | < 0.72         | 🟡 Média   |
| Precision | 0.70                          | < 0.63           | < 0.56         | 🟡 Média   |

### API Metrics

Os dados da API serão capturados pelas ferramentas de observabilidade. Os indicadores estarão em disponíveis para a equipe de Infraestrutura monitorar junto ao time de Engenharia de Dados

| Métrica do Alerta          | Limiar de Alerta | Limiar Crítico | Notas                              |
|----------------------|------------------|----------------|------------------------------------|
| Latência p95         | > 800ms          | > 2000ms       | Medir no endpoint de predição      |
| Taxa de erros HTTP   | > 1%             | > 5%           | Erros 4xx e 5xx                    |
| Volume de requisições | < 50% da média  | < 20% da média | Quedas bruscas indicam falha upstream |
| Taxa de predição positiva (Churn = 1) | Desvio > 10pp da baseline | Desvio > 20pp | Indicador precoce de data drift |

## Data Drift

Os dados relacionados aos dados (entrada e scores) serão monitorados através dos insumos que vem do CRM, com apoio do time de engenharia de dados para acompanhamento. 
 
**Drift de features de entrada:**
 
| Feature              | Tipo        | Método de Detecção (Alerta)       | Frequência  |
|----------------------|-------------|----------------------------|-------------|
| MonthlyCharges       | Numérica    | KS Test (p-value < 0.05)   | Semanal     |
| TotalCharges         | Numérica    | KS Test (p-value < 0.05)   | Semanal     |
| tenure               | Numérica    | KS Test (p-value < 0.05)   | Semanal     |
| Contract             | Categórica  | Chi-quadrado               | Semanal     |
| InternetService      | Categórica  | Chi-quadrado               | Semanal     |
| PaymentMethod        | Categórica  | Chi-quadrado               | Semanal     |
 
**Drift de distribuição de scores (output drift):**
 
Monitora a distribuição das probabilidades brutas retornadas pelo modelo antes da aplicação do threshold. 
 
| Métrica do Alerta                      | Baseline (referência)         | Limiar de Alerta                        | Limiar Crítico                          | Frequência |
|----------------------------------|-------------------------------|-----------------------------------------|-----------------------------------------|------------|
| Score médio (Churn = 1)          | Calcular na primeira semana   | Desvio > 0.05 da média baseline         | Desvio > 0.10 da média baseline         | Semanal    |
| % scores > 0.8 (alta confiança)  | Calcular na primeira semana   | Variação > 30% em relação ao baseline   | Variação > 50% em relação ao baseline   | Semanal    |
| % scores entre 0.4–0.6 (zona de incerteza) | Calcular na primeira semana | Aumento > 10pp em relação ao baseline | Aumento > 20pp em relação ao baseline | Semanal    |
| KS Test vs. distribuição baseline | —                            | p-value < 0.05                          | p-value < 0.01                          | Mensal     |

## Frequency

 
| Nível         | O que é verificado                                          | Frequência   | Responsável         |
|---------------|-------------------------------------------------------------|----------------|---------------------|
| Operacional   | Latência, taxa de erros, volume de requisições              | Contínuo       | Time de Dados/ML    |
| Drift de dados | Distribuição das features de entrada                       | Semanal        | Time de Dados/ML    |
| Performance   | Recall, Accuracy, F1, Precision vs. rótulos reais          | Mensal         | Time de Dados/ML    |
| Revisão geral | Análise completa + decisão sobre retreinamento              | Trimestral     | Time de Dados/ML    |
 

## Playbook
 
### Incident Decision Flow
 
```
Métrica em Alerta?
├── NÃO → Registrar no relatório mensal. Nenhuma ação necessária.
└── SIM
    ├── É Limiar de Alerta (não crítico)?
    │   └── Investigar causa (drift? sazonalidade? bug de dados?)
    │       ├── Causa identificada e reversível → Monitorar por 2 semanas
    │       └── Causa não identificada → Escalar para Limiar Crítico
    └── É Limiar Crítico?
        ├── Degradação confirmada → Iniciar processo de retreinamento
        └── Falha de dados/pipeline → Corrigir pipeline antes de avaliar modelo
```
 
### Incident Decision
 
| Alerta                                   | Ação Imediata                                    | Prazo     | Notificar                                      |
|----------------------------------------------|--------------------------------------------------|-----------|------------------------------------------------|
| Recall < 0.48 (alerta)                       | Investigar distribuição de FN; checar drift      | 5 dias    | Time de Dados/ML                               |
| Recall < 0.43 (crítico)                      | Iniciar retreinamento                            | Imediato  | Time de Dados/ML + Liderança de CRM/Negócio   |
| Accuracy < 0.77 (alerta)                     | Verificar qualidade dos dados de entrada         | 7 dias    | Time de Dados/ML                               |
| Accuracy < 0.72 (crítico)                    | Avaliar rollback ou retreinamento                | 2 dias    | Time de Dados/ML + Liderança de CRM/Negócio   |
| Drift em ≥ 2 features principais (alerta)    | Antecipar ciclo de avaliação de performance      | 3 dias    | Time de Dados/ML                               |
| Drift de score (KS p-value < 0.01)           | Verificar pipeline de pré-processamento (scaler) | 2 dias    | Time de Dados/ML                               |
| Taxa de predição positiva com desvio > 20pp  | Verificar pipeline; checar distribuição de score | Imediato  | Time de Dados/ML                               |
| Latência p95 > 2000ms                        | Verificar infraestrutura do endpoint             | Imediato  | Time de Dados/ML + Infraestrutura        |
| Volume de requisições < 20% da média         | Verificar integração upstream com sistema CRM    | Imediato  | Time de Dados/ML + Infraestrutura        |

## Retraining
 
O retreinamento deve ser considerado em qualquer um dos seguintes gatilhos:
 
- Recall ou Accuracy atingem limiar crítico
- Drift confirmado em 2 ou mais features de alta importância por 2 semanas consecutivas
- Revisão trimestral indica degradação consistente mesmo sem limiar crítico atingido
- Mudança relevante no portfólio de produtos/serviços da Telco
 
## Responsability
 

> **R** = Responsible (executa) · **A** = Accountable (responde pelo resultado) · **C** = Consulted · **I** = Informed · **—** = Não envolvido
 
| Atividade                        | Machine Learning | CRM | Infraestrutura |
|-----------|:--:|:---:|:-----:|
| Monitoramento operacional | R | — | C |
| Análise de drift semanal | R | — | — |
| Cálculo mensal de métricas | R | I | — |
| Validação de rótulos | C | R | — |
| Comunicação de degradação | R | I | — |
| Decisão de retreinamento | RA | I | — |
| Execução do retreinamento | RA | I | C |
| Incidente de infraestrutura | I | I | R |
 

## Records
 
Cada ciclo de monitoramento deve gerar um registro contendo:
 
- Data de avaliação
- Período dos dados avaliados
- Valores calculados para cada métrica
- Comparação com baseline e limiares
- Ocorrência ou não de alertas
- Ações tomadas (se houver)
- Responsável pela análise

Esses registros devem ser armazenados e versionados junto ao repositório do modelo para garantir rastreabilidade histórica da performance em produção.
 