# Deploy Architecture | Churn Model Prediction

**Versão:** churn-prediction-model-v1.0 · **Data:** Maio de 2026

## Details

Para permitir a integração com os dados dos clientes foi adotada uma arquitetura em **batch**, que processa os dados em lote de forma assíncrona e desacoplada do CRM. Essa abordagem evita sobrecarga na plataforma, respeita o ritmo natural de mudança comportamental dos clientes e viabiliza a criação de iniciativas de retenção alinhadas ao ciclo tático da equipe de Relacionamento.

A API do Churn Prediction Model está hospedada em **cloud (AWS/GCP/Azure)** e é acionada semanalmente por um job de integração. Os resultados são exportados em arquivo (CSV/JSON) e ingeridos de volta ao CRM para uso operacional pela equipe.

## Justify

O comportamento de um cliente dentro da plataforma é gradativo — mudanças relevantes de perfil não ocorrem em intervalos de horas ou dias. Além disso, a criação de iniciativas de retenção requer tempo de alinhamento tático e operacional com as equipes de Relacionamento e Comercial.

O processamento **semanal** foi escolhido porque:

- Acompanha o ciclo de trabalho da equipe de Relacionamento (revisão WoW — Week over Week)
- Permite monitorar tendências de risco antes que o churn se concretize
- Mantém a carga de integração com o CRM previsível e controlada

---

## Architecture Components

| Componente            | Descrição                                                                 | Responsável             |
|-----------------------|---------------------------------------------------------------------------|-------------------------|
| CRM da plataforma     | Fonte dos dados de clientes; destino final das predições                  | Time CRM/Sistemas       |
| Relatório de clientes | Seleção dos clientes a serem analisados, gerenciado pelo time de Relacionamento | Gerente de Relacionamento |
| Job de integração     | Orquestra a extração, formatação e chamada à API      | Engenharia de Dados     |
| API de predição       | Endpoint hospedado em cloud que recebe os dados e retorna a predição de churn | Time de Dados/ML        |
| Arquivo de saída      | CSV/JSON com os resultados, gerado a cada execução do job                 | Engenharia de Dados     |
| Ingestão no CRM       | Leitura do arquivo de saída e atualização dos registros no CRM            | Time CRM/Sistemas       |


---

## Data Flow

```
[CRM da Plataforma]
       │
       │  Relatório semanal de clientes ativos
       │  (seleção pela equipe de Relacionamento)
       ▼
[Job de Integração]
       │
       ├─ Extração dos dados do relatório
       ├─ Validação e formatação dos campos
       │
       ▼
[API /predict — Cloud]
       │
       ├─ Predição: churn (0/1)
       │
       ▼
[Arquivo de Saída — CSV/JSON]
       │  customerID · churn · timestamp
       │
       ├─────────────────────────────────────┐
       ▼                                     ▼
[Ingestão no CRM]                   [Log de Inferências]
Atualiza flag de risco               Armazenado para
por cliente                          monitoramento e auditoria
```


---

## Technologies

A API foi construida utilizando **FastAPI**, que suporta requisições assíncronas com baixa latência, além da validação de payload (Pydantic) e integração com as demais bibliotecas de Machine Learning existentes no ecossistema Python (PyTorch, scikit-learn). Ela está disponível dentro da hospedagem da AWS

Considerando que o modelo opera em batch semanal, o perfil de carga é previsível e concentrado — o que simplifica o dimensionamento. Para escalabilidade, é recomendado uma **escala vertical pontual** nos casos que o dimensionamento da instância não suporte a volumetria. Se a quantidade de clientes à serem analisados aumentar de forma significativa e constante ao longo das semanas, será necessário um escalamento horizontal. 

**Stack**
```
FastAPI + Uvicorn          → servidor ASGI
PyTorch                    → inferência do modelo
scikit-learn               → Standard Scaler + One-Hot Encoder (pré-processamento)
Pydantic                   → validação de inputs
```

