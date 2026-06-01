# BIRD Mini-Dev Bootstrap Summary

## Implementacao

Foi criado `src/bootstrap_bird_mini_dev.py`, um script reutilizavel para baixar ou consumir o BIRD Mini-Dev, descobrir todos os bancos SQLite, percorrer todas as tabelas e gerar artefatos compativeis com os contratos existentes do projeto.

O benchmark bruto fica separado em `benchmark/bird_mini_dev/`. A pasta `data/bird_mini_dev/` contem apenas os artefatos gerados para teste do pipeline: amostras, amostras de profiling, perfis, dicionarios e manifest.

Artefatos gerados:

- `data/bird_mini_dev/dictionaries/*.json`: dicionarios-base por tabela no formato aceito por `src/compare_results_dictionary.py`.
- `data/bird_mini_dev/samples/*.parquet`: amostras pequenas para prompts de `src/dictionary_generation.py`.
- `data/bird_mini_dev/profile_samples/*.parquet`: amostras maiores para profiling.
- `data/bird_mini_dev/profiles/*.json`: perfis compactos por tabela.
- `data/bird_mini_dev/manifest.json`: manifest consolidado com banco, tabela, SQLite, descricoes e paths gerados.

## Contratos e Compatibilidade

Matriz de compatibilidade:

| Origem BIRD/SQLite | Campo esperado | Obrigatorio | Transformacao |
| --- | --- | --- | --- |
| Nome da tabela SQLite | `table_name` | Sim | Preservado no JSON; nome de artefato recebe prefixo `bird__db__table` para evitar colisoes |
| CSV `database_description` + `dev_tables.json` + fallback SQLite | `table_description` | Sim | Descricao padrao por banco/tabela, indicando origem BIRD e introspeccao |
| `PRAGMA table_info` | `fields` | Sim | Uma entrada por coluna SQLite |
| Coluna SQLite | `field_name` | Sim | Preservado exatamente |
| CSV `data_format` ou tipo SQLite | `data_type` | Sim | Preferencia para metadado BIRD, fallback para SQLite |
| CSV `column_description`, `column_name`, `value_description` + PK/FK/not-null | `field_description` | Sim | Concatenacao limpa de descricao semantica e papel relacional |
| Primeiro valor nao nulo | `example_value` | Opcional | Extraido por coluna quando existir |
| Baixa cardinalidade | `domain_values` | Opcional | Inferido para colunas com ate 30 valores distintos e baixa razao de cardinalidade |
| Descricao + dominio ou exemplo | `full_description` | Sim | Gerado para embedding/comparacao |

Riscos de incompatibilidade:

- Alguns CSVs BIRD usam encodings diferentes de UTF-8; o script aplica fallback `utf-8-sig`, `utf-8`, `cp1252`, `latin-1` e, quando disponivel, `charset-normalizer`.
- `src/compare_results_dictionary.py` compara campos por `field_name`; se um LLM renomear campos no output, a comparacao nao encontra correspondencia.
- Perfis gerados pelo bootstrap sao compactos e reproduziveis, nao relatorios completos do `dataprofiler`. Eles preservam o contrato de entrada do prompt, que exige JSON de perfil, mas nao substituem todas as estatisticas do `dataprofiler`.

## Execucao

Com os dados ja baixados:

```bash
uv run python src/bootstrap_bird_mini_dev.py --update-config
```

Opcoes relevantes:

- `--sample-size`: tamanho da amostra pequena, padrao 100.
- `--profile-sample-size`: tamanho da amostra de profiling, padrao 10000.
- `--dataset-root`: diretorio onde `minidev/MINIDEV` sera procurado ou extraido; por padrao, `benchmark/bird_mini_dev`.
- `--dictionary-dir`: diretorio de saida dos dicionarios-base; por padrao, `data/bird_mini_dev/dictionaries`.
- `--update-config`: atualiza `config.yaml` para apontar o pipeline existente para os artefatos BIRD.

## Validacao Executada

Comandos executados:

```bash
uv run python -m compileall src/bootstrap_bird_mini_dev.py tests/test_bootstrap_bird_mini_dev.py
uv run pytest tests/test_bootstrap_bird_mini_dev.py
uv run python src/bootstrap_bird_mini_dev.py --update-config
```

Resultados:

- Testes unitarios: 4 passed.
- Bootstrap completo: 75 tabelas processadas em 11 bancos SQLite.
- Arquivos gerados: 75 dicionarios, 75 samples, 75 profile samples, 75 perfis.
- Validacao de schema dos dicionarios: 0 erros para chaves obrigatorias.
- `config.yaml` atualizado com 75 entradas em cada lista usada pelo pipeline.

## Arquivos Alterados

- `src/bootstrap_bird_mini_dev.py`
- `tests/test_bootstrap_bird_mini_dev.py`
- `config.yaml`
- `pyproject.toml`
- `uv.lock`
- `README.md`
- `benchmark/bird_mini_dev/bird_mini_dev_bootstrap_summary.md`
- `benchmark/bird_mini_dev/`
- `data/bird_mini_dev/`

## Limitacoes Remanescentes

- O ZIP oficial baixado em `benchmark/bird_mini_dev/minidev.zip` tem aproximadamente 764 MB; os artefatos extraidos e gerados podem ser grandes.
- A inferencia de `domain_values` e heuristica; para campos textuais de alta cardinalidade ela e omitida.
- O bootstrap nao chama LLMs nem executa embeddings do comparador, pois isso depende de credenciais/modelos e pode ser custoso. Ele valida os contratos de arquivos que esses scripts consomem.
