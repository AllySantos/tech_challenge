# Deploy na AWS

O Terraform em [`src/infra/`](../src/infra) provisiona a stack completa de
produção. **Nada aqui foi aplicado** — a infraestrutura está escrita, formatada
e validada (`terraform validate` e `terraform plan`: 58 recursos a criar), mas
não existe nenhuma conta com esses recursos ligados. Quem quiser conferir
provisiona na própria conta e destrói depois.

O enunciado da fase pede a decisão arquitetural **de forma textual** no README.
Este Terraform vai além do pedido, por coerência com a Fase 1, que também
entregou infraestrutura como código.

## O que é provisionado

| Arquivo | Recursos |
| --- | --- |
| `vpc.tf` | VPC, 2 subnets públicas e 2 privadas, IGW, NAT, gateway endpoint do S3, 3 security groups |
| `ecr.tf` | Repositório da imagem, com scan on push e expiração das imagens antigas |
| `s3.tf` | Bucket do registry de modelos: versionado, criptografado, sem acesso público |
| `alb.tf` | Load balancer, target group com health check em `/health`, regra que bloqueia `/metrics` |
| `ecs.tf` | Cluster, task definition da API (3 containers), serviço com circuit breaker e autoscaling |
| `retraining.tf` | Task de retreino e agendamento semanal no EventBridge Scheduler |
| `observability.tf` | Workspaces gerenciados de Prometheus e Grafana, tópico SNS e 3 alarmes |
| `iam.tf` | Roles de execução, da API, do retreino, do agendador e do deploy via OIDC |

## Como o modelo chega ao container

Localmente, a API lê `models/` por um bind mount somente-leitura. Na AWS o
contrato é o mesmo, com o S3 no lugar do disco — e sem nenhuma alteração no
código da aplicação.

A task da API tem três containers:

```
model-sync (aws-cli)  ──►  volume "models"  ──►  api (somente leitura)
      │                                            ▲
      └── aws s3 sync s3://.../models /models      │
          precisa terminar com exit 0 ─────────────┘

otel-collector ──► raspa localhost:8000/metrics ──► Amazon Managed Prometheus
```

O `dependsOn` com condição `SUCCESS` garante que a API só sobe depois que o
registry foi sincronizado. Se o sync falhar, a task não fica no ar servindo um
modelo desatualizado — ela simplesmente não sobe, e o circuit breaker do
serviço reverte o deploy.

## Retreino: por que não MWAA

O Amazon MWAA seria a tradução literal do Airflow local, mas cobra por ambiente
ligado 24 horas por dia — na ordem de 350 USD/mês — para executar um pipeline
que roda um minuto por semana.

A task agendada no Fargate executa exatamente o mesmo `python -m src.pipeline`,
com o mesmo portão de qualidade, e só é cobrada enquanto roda. A mesma
estrutura de dois containers é usada, agora ao contrário:

```
train (imagem da API)  ──►  volume "workspace"  ──►  publish (aws-cli)
  python -m src.pipeline                              aws s3 sync → S3
  exit 0 somente se passar no portão ─────────────────────┘
```

Como o portão de qualidade levanta exceção quando o F1 macro ou o p95 regridem,
o container `train` sai com código diferente de zero e o `publish` nunca roda.
**Uma versão reprovada não chega ao S3**, e o serving continua na anterior.

A contrapartida honesta: perde-se a UI do Airflow, o retry por task e a
visualização do grafo. Para um pipeline linear de oito passos que roda semanalmente,
é uma troca que compensa. Se o pipeline crescer em ramificações ou passar a ter
dependências entre DAGs, o MWAA volta a fazer sentido.

## Observabilidade

O coletor ADOT roda como sidecar na própria task, raspa o `/metrics` da API via
`localhost` e faz remote write para o Amazon Managed Prometheus. As queries e o
dashboard JSON de `monitoring/grafana/provisioning/` continuam válidos — o
Grafana gerenciado aponta para o AMP como datasource.

Três alarmes cobrem o que o dashboard mostra mas ninguém acompanha de
madrugada: taxa de 5xx, p95 acima do orçamento e tasks fora do target group.
Todos publicam no mesmo tópico SNS.

## Sequência de provisionamento

Pré-requisitos: Terraform ≥ 1.10, AWS CLI configurado e Docker.

```bash
# 1. Validar sem tocar em nenhuma conta
make validate-aws

# 2. Revisar o plano
make plan-aws

# 3. Provisionar
make build-aws

# 4. Publicar a imagem no ECR
make push-image

# 5. Enviar o registry local de modelos para o S3
make seed-models

# 6. Reciclar o serviço para que ele carregue o modelo
aws ecs update-service \
  --cluster medical-triage-cluster \
  --service medical-triage-api \
  --force-new-deployment

# URL da API
terraform -chdir=src/infra output api_url
```

Para derrubar tudo:

```bash
make destroy-aws
```

> O passo 4 precisa vir antes do 6: o serviço aponta para a tag `app-latest`, e
> subir sem imagem publicada deixa as tasks em laço de falha.

## Deploy contínuo

O job `deploy` do workflow de CI publica a imagem e atualiza o serviço a cada
merge na `main`, autenticando por **OIDC** — sem chave de acesso guardada nos
secrets.

Ele vem **desligado**. Para ativar, depois de provisionar:

1. Defina a variável de repositório `AWS_DEPLOY_ENABLED` como `true`.
2. Defina o secret `AWS_ACCOUNT_ID` com o ID da conta.

Sem isso o job é pulado, não falha — o CI continua verde em quem clonar o
repositório sem infraestrutura nenhuma.

## Custo estimado

Ordem de grandeza para `us-east-1`, com a stack ligada o mês inteiro:

| Recurso | Estimativa mensal |
| --- | ---: |
| ALB | ~US$ 17 |
| NAT Gateway | ~US$ 33 |
| Fargate — API (2 tasks, 0,5 vCPU / 1 GB) | ~US$ 30 |
| Fargate — retreino (4 execuções de ~2 min) | < US$ 1 |
| Amazon Managed Prometheus | ~US$ 5 |
| Amazon Managed Grafana | US$ 9 por usuário |
| S3, ECR, CloudWatch | < US$ 5 |
| **Total** | **~US$ 100/mês** |

Dois cortes óbvios se o objetivo for só demonstrar: remover o par AMP + AMG
(−US$ 14 e um usuário) e colocar as tasks em subnet pública para dispensar o
NAT (−US$ 33), ao custo de expor as tasks à internet. Nenhum dos dois está
aplicado — a configuração padrão privilegia o desenho correto sobre o mais
barato.

Uma stack de produção custaria mais: o segundo NAT, o WAF, a retenção maior de
logs e as contas separadas por ambiente somam antes de qualquer aumento de
tráfego. A seção seguinte detalha o que muda.

## Este projeto não é uma stack de produção — e isso é deliberado

O que está aqui é a versão **enxuta**: o desenho arquitetural correto, com os
cortes de custo e de escopo que fazem sentido para um trabalho acadêmico que
precisa ser provisionado e destruído por quem for avaliar. Nenhum desses cortes
é acidental, e nenhum deles muda a arquitetura — mudam o nível de redundância,
de segurança de borda e de operação.

A tabela abaixo é o mapa entre as duas versões.

| Aspecto | Aqui | Em produção | Impacto |
| --- | --- | --- | --- |
| **NAT Gateway** | Um só, na primeira AZ | Um por AZ | Se a AZ do NAT cair, as tasks da outra AZ perdem saída para a internet. Economiza ~US$ 33/mês |
| **TLS** | Listener HTTP na porta 80 | Certificado no ACM, listener 443, redirect 80→443, domínio no Route 53 | Tráfego em claro. Só não está aqui porque depende de um domínio que o projeto não tem |
| **Estado do Terraform** | Arquivo local na máquina de quem aplica | Bucket S3 versionado com trava | Duas pessoas aplicando ao mesmo tempo corrompem o estado. Inviável em equipe |
| **Borda** | ALB aberto | WAF com rate limiting e regras gerenciadas | Sem proteção contra abuso ou varredura automatizada |
| **Alarmes** | Tópico SNS criado, sem inscritos | Inscrição em e-mail, PagerDuty ou Slack | Os alarmes disparam e ninguém fica sabendo |
| **Orquestração do retreino** | Task Fargate agendada | MWAA ou Airflow em EKS, se o pipeline ramificar | Perde-se UI, retry por task e visualização do grafo |
| **Recarga de modelo** | Exige `--force-new-deployment` | Reciclagem encadeada ao fim do retreino, ou recarga a quente | Uma versão promovida não entra em serving sozinha |
| **Retenção de logs** | 14 dias | Retenção maior, com arquivamento em S3 Glacier | Insuficiente para auditoria clínica |
| **Provider OIDC** | Flag `create_github_oidc_provider` | Gerenciado uma vez, em um módulo de plataforma separado | Só existe um por URL em cada conta; a Fase 1 já cria o dela |
| **Conta AWS** | Uma só | Contas separadas por ambiente, com organização e SCP | Sem isolamento entre o que é experimento e o que é produção |

O que **não** muda entre as duas versões, e é onde está o valor do exercício:
o desenho de rede com as tasks em subnet privada, o contrato de serving via
registry somente-leitura, o portão de qualidade que barra a promoção, a
federação OIDC no lugar de chave de acesso e as métricas do modelo indo para o
Prometheus. A ponte entre "trabalho de faculdade" e "produção" aqui é uma lista
de itens operacionais, não uma reescrita.

### Um caso à parte: a natureza clínica do sistema

Os itens acima são de infraestrutura. Um sistema de triagem que toque paciente
real tem uma lista própria, e mais pesada: rótulos de urgência atribuídos por
triagem clínica em vez de derivados de taxonomia (ver
[`model_card.md`](model_card.md)), corpus em português, auditoria de disparidade
de desempenho entre grupos de pacientes, retenção e rastreabilidade compatíveis
com a LGPD, e validação clínica formal antes de qualquer uso assistencial.

Nenhum ajuste de Terraform resolve isso. É a limitação que separa o projeto de
um produto, e ela é de dados e de processo, não de nuvem.
