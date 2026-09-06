# Otimização de latência

Este documento registra as três técnicas aplicadas ao modelo de triagem, o que
cada uma rendeu e — no caso da que não funcionou — por quê.

Todas as medições foram feitas com `python -m src.evaluation.benchmark`:
500 inferências **unitárias** por backend, após 50 de aquecimento, em um
MacBook Pro (Apple Silicon, 1 thread por sessão do ONNX Runtime). Latência
média é omitida de propósito; o que interessa em produção é a cauda.

## Ponto de partida

O modelo é um `Pipeline` do scikit-learn com dois estágios: `TfidfVectorizer`
(uni + bigramas, 30 mil termos) seguido de `LogisticRegression` multiclasse.

Antes de otimizar, vale saber onde o tempo é gasto. Inspecionando o grafo
exportado:

```
Reshape → StringNormalizer → Tokenizer → Flatten → TfIdfVectorizer
       → Add → Log → Mul → Normalizer → LinearClassifier → Normalizer
```

O produto matricial do classificador é minúsculo — uma matriz 30.000 × 3. O
custo real está no bloco de texto: normalizar, tokenizar e casar cada token
contra um vocabulário de dezenas de milhares de termos. Essa observação é o
que orienta as escolhas abaixo.

## Técnica 1 — Conversão para ONNX Runtime

O pipeline é convertido com `skl2onnx` e servido pelo ONNX Runtime, trocando o
caminho de inferência em Python (com scipy sparse) por um grafo compilado
executado em C++.

**Resultado: p95 de 0,39 ms → 0,20 ms (1,95× mais rápido).**

Duas restrições apareceram na conversão e moldaram o código:

- `strip_accents="unicode"` não é suportado pelo conversor. O parâmetro foi
  removido do vetorizador — o corpus é em inglês e a normalização de texto já
  acontece antes, em `src/data/preprocess.py`, então os dois caminhos
  permanecem equivalentes.
- O nó `StringNormalizer` abre a locale `en_US.UTF-8` ao inicializar a sessão.
  A imagem `python:3.11-slim` não traz nenhuma locale gerada, e a ausência
  derrubava a API no boot. O `Dockerfile.api` gera a locale explicitamente.

## Técnica 2 — Pruning do vocabulário

Como o gargalo é o vetorizador, a otimização que ataca o gargalo é encolher o
vocabulário. Cada termo é pontuado pelo maior peso absoluto que recebeu entre
as três classes, os 10 mil mais influentes são mantidos e o classificador é
reajustado sobre o espaço reduzido — a mesma ideia do pruning por magnitude
aplicado a redes neurais.

O reajuste importa: herdar os pesos treinados no espaço de 30 mil features
deixaria o classificador mal calibrado para a entrada que passou a receber.

**Resultado: p95 de 0,20 ms → 0,10 ms (1,95× adicional), artefato 3,1× menor,
ao custo de 0,9 ponto percentual de F1 macro.**

A seleção por magnitude também bate o corte ingênuo por frequência: treinar
direto com `max_features=10000` dá F1 macro de 0,7249, contra 0,7267 do
pruning por peso — mesmo tamanho de vocabulário, modelo um pouco melhor.

### Escolha do ponto de corte

| Termos mantidos | F1 macro | p95 | Artefato |
| ---: | ---: | ---: | ---: |
| 30.000 (sem pruning) | 0,7353 | 0,19 ms | 1,39 MB |
| 10.000 | 0,7249 | 0,11 ms | 0,44 MB |
| 5.000 | 0,7045 | 0,08 ms | 0,21 MB |
| 2.000 | 0,6987 | 0,07 ms | 0,08 MB |

10 mil é o joelho da curva: abaixo disso o F1 cai três vezes mais rápido por
termo removido, enquanto o ganho de latência já se esgotou em grande parte.

## Técnica 3 — Quantização dinâmica INT8 (sem efeito)

`onnxruntime.quantization.quantize_dynamic` foi aplicado ao grafo. **Não
produziu nenhum efeito**: nem em latência, nem em tamanho.

A verificação no grafo explica o motivo:

```python
>>> [n.op_type for n in onnx.load("model.onnx").graph.node]
['Reshape', 'StringNormalizer', 'Tokenizer', 'Flatten', 'TfIdfVectorizer',
 'Add', 'Log', 'Mul', 'Normalizer', 'LinearClassifier', 'Normalizer']
```

Não há nenhum `MatMul`, `Gemm` ou `Conv` — os operadores que a quantização
dinâmica sabe converter. O classificador é um `LinearClassifier` do domínio
`ai.onnx.ml`, e seus coeficientes vivem em **atributos do nó**, não em
initializers do grafo. O quantizador não tem o que tocar e devolve o modelo
intacto: os dois arquivos têm exatamente os mesmos nós e os mesmos
initializers, diferindo apenas nos metadados de produtor.

O backend foi mantido no comparativo em vez de removido. Uma técnica testada
que não se aplica ao modelo em questão é um resultado, e omiti-la deixaria a
tabela mais bonita e menos verdadeira. Ela também documenta a condição em que
a quantização passaria a valer a pena: se o classificador linear fosse
substituído por um MLP denso, os `MatMul` apareceriam e o INT8 teria o que
comprimir.

## Consolidado

| Backend | p50 | p95 | p99 | Throughput | Artefato | Ganho no p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| scikit-learn (baseline) | 0,34 ms | 0,39 ms | 0,42 ms | 2.902 req/s | 1,02 MB | 1,00× |
| ONNX Runtime | 0,17 ms | 0,20 ms | 0,23 ms | 5.730 req/s | 1,32 MB | 1,95× |
| ONNX Runtime INT8 | 0,17 ms | 0,19 ms | 0,20 ms | 5.828 req/s | 1,32 MB | 2,04× |
| **ONNX Runtime + pruning** | **0,09 ms** | **0,10 ms** | **0,11 ms** | **11.650 req/s** | **0,42 MB** | **3,85×** |

O backend servido em produção é o **ONNX Runtime + pruning**, configurado em
`INFERENCE_BACKEND`.

## O que essa latência significa ponta a ponta

O benchmark mede só a inferência. Medido dentro do container, sob carga do
gerador de tráfego, o p95 **HTTP completo** fica em torno de 4,8 ms — a
diferença é overhead de rede local, parsing de JSON e validação Pydantic, não
do modelo. Em outras palavras: com o pruning, o modelo deixou de ser o termo
dominante do tempo de resposta, e otimizá-lo mais renderia pouco. O próximo
ganho relevante viria do transporte, não do classificador.

Essa é também a razão de o portão de qualidade usar 25 ms como teto de p95 de
inferência: é uma margem ampla sobre os 0,10 ms atuais, dimensionada para
barrar uma regressão real de arquitetura, não para reagir a ruído de medição.
