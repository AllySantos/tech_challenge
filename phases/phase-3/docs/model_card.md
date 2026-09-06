# Model Card — Classificador de Urgência de Laudos

## Identificação

| | |
| --- | --- |
| **Nome** | `medical-triage` |
| **Versão de referência** | `20260906T015610Z` |
| **Tipo** | Classificador de texto multiclasse (3 classes) |
| **Arquitetura** | TF-IDF (uni + bigramas) + Regressão Logística |
| **Artefato servido** | `model.pruned.onnx` — 0,42 MB |
| **Licença do corpus** | CC-BY-4.0 |

## Uso pretendido

**Caso de uso primário.** Atribuir uma prioridade preliminar a laudos médicos
recém-liberados, para ordenar a fila de triagem hospitalar.

**Usuários pretendidos.** Equipe de triagem clínica, como apoio à ordenação —
nunca como decisor.

**Fora de escopo, explicitamente:**

- Diagnóstico, ou qualquer inferência sobre a condição do paciente.
- Decisão autônoma de prioridade sem revisão humana.
- Laudos em português — o modelo foi treinado em inglês (ver Limitações).
- Textos que não sejam laudos ou abstracts clínicos.

## Dados

**Fonte.** [Medical Abstracts TC Corpus](https://github.com/sebischair/Medical-Abstracts-TC-Corpus)
— 14.438 abstracts médicos em inglês.

**Preparação.** Os splits oficiais de treino e teste são concatenados e
redivididos de forma estratificada por urgência, porque a divisão original é
orientada à taxonomia anatômica. Laudos com menos de 50 caracteres e duplicatas
exatas são descartados: 14.438 → 11.227 registros, divididos em 9.542 de treino
e 1.685 de validação.

### O rótulo de urgência é derivado

O corpus rotula cada texto pelo **sistema do corpo acometido**, não por
prioridade de atendimento. Não há corpus público de triagem, com rótulo de
urgência e volume adequado, disponível livremente. A prioridade é portanto
derivada por um mapeamento determinístico:

| Classe original | Urgência | Justificativa clínica |
| --- | --- | --- |
| Cardiovascular · Sistema nervoso | `urgente` | Protocolos tempo-dependentes: infarto e AVC têm janela terapêutica medida em minutos |
| Neoplasias · Sistema digestivo | `atencao` | Investigação prioritária — fast-track oncológico, abdome agudo, hemorragia digestiva |
| Condições patológicas gerais | `normal` | Achados inespecíficos, compatíveis com fluxo eletivo |

Distribuição resultante no treino: 35,0% urgente, 33,3% atenção, 31,7% normal.

**Esta é a limitação central do modelo.** O mapeamento é uma aproximação
razoável e clinicamente argumentável, mas continua sendo uma aproximação: um
abstract sobre cardiopatia congênita estável recebe `urgente`, e um sobre
sepse classificado como condição geral recebe `normal`. O modelo aprende a
distinguir sistemas do corpo, e a urgência vem por herança do mapeamento.

Um sistema em uso real exige rótulos atribuídos por triagem clínica sobre
laudos reais. O que este projeto demonstra é o **ciclo de vida** — treino,
otimização, portão de qualidade, deploy, observabilidade —, não a validade
clínica do classificador.

## Configuração de treino

| Hiperparâmetro | Valor |
| --- | --- |
| Vocabulário TF-IDF | 30.000 termos, uni + bigramas, `min_df=3`, `sublinear_tf` |
| Stop words | Lista em inglês do scikit-learn |
| Regularização (`C`) | 4,0 |
| Balanceamento | `class_weight="balanced"` |
| Iterações máximas | 1.000 |
| Seed | 42 |
| Pruning | 10.000 termos mantidos, por maior peso absoluto, com reajuste |

`strip_accents` foi deliberadamente omitido: o conversor ONNX não o suporta, e
mantê-lo faria o grafo exportado divergir do pipeline de origem. A normalização
de texto acontece antes, na etapa de preparação.

## Desempenho

Conjunto de validação: 1.685 laudos. Modelo servido (com pruning).

| Métrica | Valor |
| --- | --- |
| Acurácia | 0,7288 |
| **F1 macro** | **0,7267** |
| F1 do baseline sem pruning | 0,7353 |
| Custo do pruning | −0,86 ponto percentual |

### Por classe

| Urgência | Precisão | Recall | F1 | Suporte |
| --- | ---: | ---: | ---: | ---: |
| `atencao` | 0,800 | 0,800 | 0,800 | 561 |
| `normal` | 0,607 | 0,618 | 0,612 | 534 |
| `urgente` | 0,774 | 0,761 | 0,768 | 590 |

### Matriz de confusão

Linhas são o valor real, colunas o previsto.

| real \ previsto | atencao | normal | urgente |
| --- | ---: | ---: | ---: |
| **atencao** | 449 | 92 | 20 |
| **normal** | 93 | 330 | 111 |
| **urgente** | 19 | 122 | 449 |

**O erro que mais importa é o canto inferior direito da primeira coluna.**
Apenas 19 laudos `urgente` foram classificados como `atencao`, e nenhum salta
direto para o outro extremo — mas 122 `urgente` foram para `normal`, que é o
subdiagnóstico de gravidade, o erro mais caro em triagem. Em um sistema real,
o limiar de decisão deveria ser deslocado para favorecer o falso positivo de
urgência sobre o falso negativo, aceitando piorar a precisão de `urgente` para
elevar o recall.

A classe `normal` é a mais fraca (F1 0,612), o que é coerente com sua origem:
ela agrega "condições patológicas gerais", a categoria mais heterogênea e
menos linguisticamente distinta do corpus.

## Latência

Medida com 500 inferências unitárias por backend, após 50 de aquecimento, com
uma thread por sessão do ONNX Runtime. Detalhamento em
[`optimization.md`](optimization.md).

| Backend | p50 | p95 | p99 | Artefato |
| --- | ---: | ---: | ---: | ---: |
| scikit-learn | 0,34 ms | 0,39 ms | 0,42 ms | 1,02 MB |
| ONNX Runtime | 0,17 ms | 0,20 ms | 0,23 ms | 1,32 MB |
| ONNX Runtime INT8 | 0,17 ms | 0,19 ms | 0,20 ms | 1,32 MB |
| **ONNX Runtime + pruning** | **0,09 ms** | **0,10 ms** | **0,11 ms** | **0,42 MB** |

## Portão de promoção

Nenhuma versão chega ao serving sem atender simultaneamente a:

| Critério | Limite |
| --- | ---: |
| F1 macro | ≥ 0,60 |
| p95 de inferência | ≤ 25 ms |

Falha em qualquer um mantém o ponteiro `current.json` na versão anterior.

## Considerações éticas e riscos

- **Subdiagnóstico de gravidade.** 20,7% dos laudos `urgente` foram
  classificados como `normal`. Sem revisão humana, isso se traduz em atraso de
  atendimento. O modelo pressupõe triagem assistida, não automatizada.
- **Viés herdado do corpus.** Abstracts de literatura médica não representam a
  distribuição de laudos de um pronto-socorro, nem a de uma população
  específica. Desempenho medido aqui não transfere para produção.
- **Ausência de auditoria demográfica.** O corpus não traz atributos de
  paciente, o que impossibilita avaliar disparidade de desempenho entre grupos
  — uma verificação obrigatória antes de qualquer uso clínico.
- **Rastreabilidade.** Toda resposta da API carrega `model_version` e
  `backend`, para que uma classificação contestada possa ser reconstruída a
  partir do artefato exato que a produziu.

## Manutenção

Retreino semanal pela DAG `triage_training_pipeline`. As métricas de
`triage_predictions_total` e `triage_prediction_confidence`, expostas no
dashboard, são os indicadores de desvio: mudança na distribuição de classes ou
queda sustentada de confiança sinalizam alteração na população de entrada,
normalmente antes de a qualidade cair de forma mensurável.
