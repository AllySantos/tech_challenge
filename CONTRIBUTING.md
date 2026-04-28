# Como contribuir

## Fluxo de trabalho

1. **Pegue uma issue** no [GitHub Issues](https://github.com/AllySantos/tech_challenge/issues)
2. **Crie uma branch** a partir da `main`:
   ```bash
   git checkout main && git pull
   git checkout -b seu-nome/descricao-curta
   ```
3. **Faça as mudanças** e commit com mensagem clara:
   ```bash
   git commit -m "feat: descrição do que foi feito"
   ```
4. **Abra um Pull Request** para a `main` e marque alguém para revisar
5. **Feche a issue** quando o PR for mergeado

## Convenção de branches

```
nome/descricao      # ex: gabriel/fastapi-predict
fix/descricao       # ex: fix/mlflow-tracking
```

## Convenção de commits

```
feat: nova funcionalidade
fix: correção de bug
docs: documentação
refactor: refatoração sem mudança de comportamento
test: adição ou correção de testes
```

## Rodando localmente

```bash
make install
source .venv/bin/activate
```

## Antes de abrir PR

- [ ] O notebook roda do zero sem erros?
- [ ] Os testes passam? (`make test`)
- [ ] O linting passa? (`make lint`)
- [ ] A issue correspondente está linkada no PR?
