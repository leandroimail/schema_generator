# Schema Generator: Geração e Avaliação Automática de Dicionários de Dados e *Schema Matching* com LLMs

> **Artigo técnico-científico do projeto `schema_generator`.**
> Documenta arquitetura, processos, sequências, APIs/interfaces, modo de uso e os
> resultados do benchmark executado sobre o corpus BIRD Mini-Dev.
> Gerado em junho/2026. Diagramas em Mermaid (exportados para PNG) e gráficos em
> matplotlib estão embutidos ao longo do texto.
> English version: [`SCHEMA_GENERATOR_ARTICLE.en.md`](SCHEMA_GENERATOR_ARTICLE.en.md).

---

## Resumo (Abstract)

A integração de dados entre modelos relacionais normalizados (3FN) e modelos analíticos
multidimensionais (*star schema*) exige o alinhamento de esquemas que frequentemente usam
nomes, tipos e estruturas distintos para conceitos equivalentes. Esse alinhamento — o
*schema matching* — é tradicionalmente manual, custoso e propenso a erro. Este trabalho
apresenta o **`schema_generator`**, um *framework* modular e dirigido por *prompts* que
automatiza duas tarefas correlatas: **(i)** a geração de **dicionários de dados** semânticos
a partir de perfis estatísticos e amostras de tabelas, usando múltiplos *Large Language
Models* (LLMs); e **(ii)** o ***schema matching*** entre modelos relacionais e *star schemas*.
A qualidade dos dicionários gerados é avaliada quantitativamente por **similaridade de
cosseno** entre *embeddings* de descrições geradas e descrições de referência.

A avaliação empírica foi conduzida sobre o **BIRD Mini-Dev** (11 bancos SQLite, 75 tabelas,
798 campos de referência), comparando **6 LLMs** (DeepSeek, Google Gemini e OpenAI) em
**4.701 comparações campo-a-campo**. O resultado central é o **ranking de qualidade dos
LLMs**: `deepseek-v4-flash` é o vencedor mais consistente no confronto direto por tabela —
produz o melhor dicionário em **30 das 75 tabelas** sob BGE (e 25/75 sob o instrumento
alternativo MiniLM), enquanto `gpt-5.4-nano` (o menor/mais barato) é o pior por média global
sob ambas as réguas. Os quatro modelos intermediários ficam em quase-empate técnico, e o
*ranking de média* deles **depende** do modelo de *embedding* usado como instrumento — por
isso a recomendação se ancora no confronto *head-to-head* (Figura 9), que é estável às duas
réguas, e não na média global.

---

## 1. Introdução e Motivação

A migração de dados transacionais (normalizados) para ambientes analíticos (denormalizados,
em *star schema*) é um problema recorrente em engenharia de dados. O desafio central é o
*schema matching*: mapear campos de origem para campos de destino quando:

- múltiplas tabelas normalizadas precisam ser mapeadas para tabelas-fato e dimensões
  denormalizadas;
- transformações complexas (agregações, conversões de tipo, concatenações, campos derivados)
  precisam ser detectadas e documentadas;
- a correspondência é **semântica**, não meramente lexical — nomes e significados de negócio
  divergem entre modelos operacionais e analíticos.

O `schema_generator` ataca esse problema com uma abordagem **dirigida por *prompts***: LLMs
são guiados por instruções detalhadas para **(a)** descrever semanticamente cada tabela e
campo (dicionário de dados) e **(b)** produzir mapeamentos auditáveis origem → destino,
incluindo as transformações necessárias e cobertura total dos campos de destino (suportando
mapeamentos 1→N, N→1 e compostos).

O insumo para os LLMs não é o dado bruto, mas sim **perfis estatísticos** (tipos,
contagens, nulos, distribuições) e **amostras** representativas — uma escolha que reduz o
volume de *tokens* e foca o modelo no significado dos campos.

---

## 2. Arquitetura do Sistema

O sistema é organizado em camadas desacopladas, cada uma materializada por um script em
`src/` e orquestrada por `run.py`. A Figura 1 apresenta a visão de componentes.

![Figura 1 — Arquitetura de componentes do schema_generator](images/diag_architecture.png)
**Figura 1.** Arquitetura de componentes. As fontes BIRD (SQLite + CSV) são transformadas
pela camada de *bootstrap* em artefatos (dicionários-base, *samples*, perfis, *manifest*);
a camada de geração usa LLMs para enriquecer dicionários; a camada de avaliação compara o
resultado contra o *ground truth* via *embeddings*. O `run.py` orquestra todas as etapas.

### 2.1 Componentes principais (`src/`)

| Módulo | Responsabilidade | Entrada → Saída |
| --- | --- | --- |
| `bootstrap_bird_mini_dev.py` | Baixa/extrai o BIRD Mini-Dev, introspecta SQLite, lê CSVs de descrição e gera dicionários-base, *samples*, perfis e `manifest.json`. | `minidev.zip` / SQLite → artefatos em `data/bird_mini_dev/` |
| `samples.py` | Gera amostras aleatórias (100/1.000/10.000 linhas) de arquivos Parquet, com estratégias adaptadas ao tamanho do arquivo. | Parquet completo → `data/samples/*.parquet` |
| `profilling.py` | Usa a biblioteca `dataprofiler` para gerar perfis estatísticos JSON. | *samples* → `data/profiles/*.json` |
| `dictionary_generation.py` | Monta o *prompt* (perfil + amostra) e dispara os LLMs para gerar dicionários, validando contra um *schema* Pydantic. | perfis + amostras + *prompt* → `data/llm_results/.../*_parsed.json` |
| `schema_matching_generation.py` | Gera o mapeamento origem→destino (relacional → *star schema*) via *prompt* dedicado. | dicionário origem + modelo destino → `data/schema_matching/` |
| `compare_results_dictionary.py` | Calcula *embeddings* e similaridade de cosseno entre dicionários LLM e de referência; consolida métricas de distribuição. | dicionários LLM + referência → `data/distance_calculation/` |
| `llm.py` | Cliente LLM unificado (OpenAI, Gemini, DeepSeek): autenticação, *structured output*, concorrência assíncrona, persistência. | usado pelos módulos de geração |
| `load_data.py` | Carrega *samples* Parquet em DuckDB e gera *scripts* DDL/verificação. | Parquet → DuckDB + `.sql` |

### 2.2 Configuração e interfaces

O sistema é **dirigido por configuração**, sem *hard-coding* de caminhos ou modelos:

- **`config.yaml`** — caminhos de perfis, dicionários, *samples*, modelo-alvo, diretórios de
  saída, seleção do LLM de *schema matching* e o bloco **`embedding:`** (modelo de
  *embedding*, *device*, *cache*, normalização, *batch size*).
- **`.env`** — chaves de API e definição dos *targets* de LLM via o padrão
  `LLM_CLIENTS=<id1>,<id2>,...` e variáveis `LLM_<ID>_*` (provedor, modelo, *base URL*,
  *flags* de *structured output*). Múltiplos *targets* — combinando provedores, modelos e
  chaves — podem ser executados numa única passada.

---

## 3. Os Processos do Pipeline

O `run.py` é o **ponto de entrada único**. Cada subcomando encapsula um script `src/*.py`
como subprocesso, de modo que falhas se propagam como códigos de saída e compõem
naturalmente com `set -e` e CI. A Figura 2 mostra o fluxo de controle.

![Figura 2 — Fluxo do pipeline orquestrado por run.py](images/diag_pipeline.png)
**Figura 2.** Fluxo do pipeline. `validate` é um *gate*: se houver erros, `retry-llm`
reexecuta apenas as combinações provedor/perfil que falharam antes de `compare`.

| Subcomando | Encapsula | Propósito |
| --- | --- | --- |
| `bootstrap` | `bootstrap_bird_mini_dev.py` | Baixa/processa o BIRD Mini-Dev e gera artefatos. |
| `generate` | `dictionary_generation.py` | Gera os dicionários LLM para cada perfil. |
| `validate` | auditoria *in-process* | Inspeciona todos os `*_parsed.json` em busca de erros. |
| `retry-llm` | `dictionary_generation.py` | Reexecuta só as combinações cujo JSON está ausente/ inválido. |
| `compare` | `compare_results_dictionary.py` | Calcula distâncias de cosseno e gera os *summaries*. |
| `metrics` | leitura *in-process* | Imprime o resumo consolidado `metrics_by_model`. |
| `all` | validate → (retry) → compare | Conveniência pós-geração. |
| `pipeline` | bootstrap → generate → all | *Pipeline* completo, ponta a ponta. |

```bash
# Pipeline completo: bootstrap + generate + validate + retry-llm + compare
uv run python run.py pipeline --with-bootstrap --with-generate --with-retry
```

### 3.1 Processo 1 — *Bootstrap* do benchmark

O `bootstrap_bird_mini_dev.py` torna o pipeline reprodutível a partir de um *dataset*
público. Ele: localiza/baixa o `minidev.zip`; varre `dev_databases/*/*.sqlite`; usa
`PRAGMA table_info` para enumerar colunas (com PK/FK/*not-null*); lê os CSVs de
`database_description` (com *fallback* de *encoding* `utf-8-sig`→`utf-8`→`cp1252`→`latin-1`);
e infere **`domain_values`** para colunas de baixa cardinalidade (≤ 30 distintos e razão
≤ 0,2). O resultado é um **dicionário-base** por tabela (que serve de *ground truth*),
*samples* de *prompt* (100 linhas), *samples* de *profiling* (10.000 linhas), perfis
compactos e um `manifest.json` consolidado. Com `--update-config`, as listas do
`config.yaml` passam a apontar para esses artefatos automaticamente.

### 3.2 Processo 2 — Amostragem e *profiling*

`samples.py` adota três estratégias conforme o tamanho do arquivo (amostragem direta para
arquivos pequenos; amostragem esparsa por índices para arquivos > 10M linhas; amostragem
fracionária com *trim* exato para os intermediários), garantindo reprodutibilidade via
`random_state`. `profilling.py` usa `dataprofiler` (`Profiler(...).report(output_format=
"compact")`) para produzir os perfis JSON que alimentam o *prompt*.

### 3.3 Processo 3 — Geração de dicionários via LLM

O coração da geração está em `dictionary_generation.py` + `llm.py`. O *prompt*
(`prompts/dictionary_generation.md`) instrui o LLM a agir como especialista em modelagem e
documentação, recebendo dois blocos — **`## Data Profile`** (estatísticas) e
**`## Data Sample`** (amostra em *markdown table*) — e a retornar **apenas** um JSON válido
com `table_name`, `table_description` e, por campo, `field_name`, `data_type`,
`field_description`, `example_value`, `domain_values` (opcional) e `full_description`
(concatenação descrição + domínio/exemplo, usada depois na comparação).

A Figura 3 detalha a sequência. A saída é validada contra o modelo Pydantic
`DataDictionary` (`extra="forbid"`), e três artefatos são persistidos por modelo
(`*_raw.txt`, `*_parsed.json`, `*_prompt.txt`) para auditabilidade e reprodutibilidade.

![Figura 3 — Sequência da geração de dicionário via LLM](images/diag_seq_generation.png)
**Figura 3.** Sequência de geração. Os clientes são inicializados e invocados de forma
**concorrente** (`asyncio.gather`); cada resposta é validada por *schema* antes de ser salva.

### 3.4 Processo 4 — *Schema matching*

`schema_matching_generation.py` recebe o esquema de origem (dicionário JSON, possivelmente
enriquecido) e o *star schema* de destino, e instrui o LLM a: comparar nomes/tipos/descrições;
explorar sinônimos, abreviações, contexto de negócio e metadados de PK/FK; **detectar e
documentar transformações** (conversões, agregações, concatenações); e **garantir cobertura
total** dos campos de destino (mesmo sem correspondência, suportando 1→N, N→1 e compostos).
A saída é um JSON estrito por tabela/campo de destino.

### 3.5 Processo 5 — Avaliação por similaridade semântica

`compare_results_dictionary.py` quantifica a qualidade. Para cada chave comum
`(tabela, modelo, campo)`, ele computa *embeddings* de `full_description` do dicionário LLM e
do dicionário de referência (`SentenceTransformer.encode`) e a **similaridade de cosseno**
(`sklearn.metrics.pairwise.cosine_similarity`). Arquivos com chave `error` (falhas de
*rate-limit*/parse) são ignorados. O resultado consolidado em
`all_similarities_results.json` contém três blocos:

- `results` — *score* por `(tabela, campo, modelo)` (a distribuição bruta — 4.701 valores);
- `metrics_by_table_and_model` — métricas de distribuição por `(tabela, modelo)`;
- `metrics_by_model` — métricas por modelo, agregando todas as tabelas.

As métricas reportadas por `compute_similarity_metrics` são: `mean`, `std` (amostral,
ddof=1), `q25`, `median`, `q75`, `d90`, `d99`, `min`, `max`, `count` (valores não-finitos
são descartados). A Figura 4 mostra a sequência.

![Figura 4 — Sequência da avaliação por similaridade](images/diag_seq_compare.png)
**Figura 4.** Sequência de comparação. O modelo de *embedding* é construído por
`_load_embedding_model(config)` a partir do bloco `embedding:` do `config.yaml`.

---

## 4. APIs, Interfaces e o Cliente LLM Unificado

O `llm.py` abstrai três provedores sob uma interface comum. A Figura 5 resume a hierarquia.

![Figura 5 — Hierarquia de clientes LLM](images/diag_llm_clients.png)
**Figura 5.** `BaseLLMClient` define o contrato; `OpenAIClient` implementa *structured
output* via JSON Schema estrito; `DeepSeekClient` herda de `OpenAIClient` (modo JSON object,
sem *schema* estrito); `GeminiClient` usa o *endpoint* compatível com OpenAI do Gemini e
extrai JSON de blocos *markdown* quando necessário.

**Pontos-chave de projeto:**

- **Configuração por ambiente:** `load_llm_client_configs()` lê `LLM_CLIENTS` e as variáveis
  `LLM_<ID>_*`, com *fallback* para o modo legado (`OPENAI_API_KEY`, `GEMINI_API_KEY`,
  `DEEPSEEK_API_KEY`). Cada `LLMClientConfig` é imutável e carrega `provider`, `model_name`,
  `structured_output`, `use_json_schema` etc.
- ***Structured output* por provedor:** OpenAI e Gemini podem exigir JSON Schema estrito
  (`_strict_json_schema` torna todas as propriedades obrigatórias e proíbe propriedades
  adicionais); DeepSeek opera em modo JSON *object* com *hint* de *schema* no *prompt*.
- **Concorrência assíncrona:** inicialização, processamento e encerramento dos clientes são
  paralelizados via `asyncio.gather`, permitindo executar os 6 modelos lado a lado.
- **Persistência e resiliência:** `save_response_data` grava `raw/`, `json/` e `prompts/`;
  há *fallback* de *backup* de emergência em `logs/emergency_backups/`.

As funções públicas são `run_llm_clients(prompt, base_output_dir, response_model)`,
`run_llm_clients_one(...)` (cliente único, usado no *retry*) e
`load_llm_client_configs(selected)`.

### 4.1 Fluxo de artefatos

A Figura 6 mostra como os artefatos fluem entre diretórios. Note que os dicionários de
referência podem vir do BIRD (`data/bird_mini_dev/dictionaries/`) **ou** de
`docs/model_sources/` — o pipeline é agnóstico à origem do *ground truth*.

![Figura 6 — Fluxo de artefatos entre diretórios](images/diag_artifacts.png)
**Figura 6.** Fluxo de dados/artefatos. Cada seta representa uma dependência de
produção/consumo entre etapas.

---

## 5. Como Usar

```bash
# 1. Dependências (uv recomendado)
uv sync

# 2. Configurar segredos e caminhos
cp .env.example .env   # preencher chaves de API e LLM_CLIENTS
#   ajustar config.yaml (caminhos, bloco embedding:, schema_matching_llm)

# 3a. Caminho rápido: benchmark reprodutível BIRD Mini-Dev
uv run python run.py bootstrap --update-config

# 3b. Ou caminho manual com dados próprios
python src/samples.py        # amostras 100/1k/10k
python src/profilling.py     # perfis JSON

# 4. Geração de dicionários (6 LLMs em paralelo)
uv run python run.py generate

# 5. Auditoria + retry seletivo + comparação
uv run python run.py all --with-retry

# 6. Resumo consolidado
uv run python run.py metrics
```

Para *schema matching*: `python src/schema_matching_generation.py` (após configurar o
modelo-alvo em `docs/model/model.json` e selecionar `schema_matching_llm` no `config.yaml`).

---

## 6. Configuração Experimental do Benchmark

| Dimensão | Valor |
| --- | --- |
| Corpus | **BIRD Mini-Dev** |
| Bancos de dados | **11** (`california_schools`, `card_games`, `codebase_community`, `debit_card_specializing`, `european_football_2`, `financial`, `formula_1`, `student_club`, `superhero`, `thrombosis_prediction`, `toxicology`) |
| Tabelas (dicionários-base) | **75** |
| Campos de referência | **798** |
| LLMs avaliados | **6**: `deepseek-v4-flash`, `deepseek-v4-pro`, `gemini-3.1-flash-lite`, `gemini-3.5-flash`, `gpt-5.4-mini`, `gpt-5.4-nano` |
| Comparações campo-a-campo (BGE) | **4.701** chaves `(tabela, modelo, campo)` |
| Instrumento de medição (*default*) | `BAAI/bge-small-en-v1.5` (384 dim, *device* `cpu`) |
| Instrumento *baseline* | `all-MiniLM-L6-v2` (384 dim) |
| Métrica | Similaridade de cosseno entre `full_description` LLM × referência |

> **Nota sobre identificadores de modelo.** Os *ids* (`gpt-5.4-*`, `gemini-3.x`,
> `deepseek-v4-*`) e a data (2026) são os registrados pelo próprio projeto e reproduzidos
> aqui como estão; este artigo descreve **o que o projeto mediu**, não uma taxonomia
> externa de modelos.

### 6.1 Por que 75 tabelas?

O número 75 **não é uma amostragem nem um corte arbitrário**: é a **contagem exaustiva de
todas as tabelas dos 11 bancos** do BIRD Mini-Dev. Na etapa de *bootstrap* (§3.1), a função
`discover_tables()` percorre `dev_databases/*/*.sqlite`, executa `PRAGMA table_info` em cada
banco e enumera **todas** as tabelas (excluindo as de sistema `sqlite_*`). A soma dá 75:

| Banco SQLite | Tabelas |
| --- | ---: |
| formula_1 | 13 |
| superhero | 10 |
| student_club | 8 |
| codebase_community | 8 |
| financial | 8 |
| european_football_2 | 7 |
| card_games | 6 |
| debit_card_specializing | 5 |
| toxicology | 4 |
| thrombosis_prediction | 3 |
| california_schools | 3 |
| **Total (11 bancos)** | **75** |

A cadeia do corpus é, portanto: **11 bancos → 75 tabelas → 798 campos de referência**. O
`--sample-size` e os filtros do *bootstrap* controlam apenas o **tamanho das amostras** (linhas
por tabela), **não** o número de tabelas; reduzir as 75 exigiria restringir quais
bancos/tabelas o `discover_tables()` processa.

### 6.2 Cobertura e robustez

Das **450** gerações tentadas (75 tabelas × 6 modelos), **442 foram válidas** e **8 falharam**
(relatório `reports/dictionary_generation_error_report.md`). As falhas concentram-se em **3
tabelas grandes** (`european_football_2.match`, `codebase_community.posts` e `.posthistory`)
e em **categorias de limite operacional**, não de qualidade do modelo:

| Categoria de erro | Ocorrências |
| --- | --- |
| `openai_tpm_or_prompt_too_large` | 6 |
| `context_window_exceeded` | 1 |
| `provider_quota_exceeded` | 1 |

Isso explica por que o `count` por modelo varia (717–873): tabelas com mais campos e/ou
*prompts* maiores às vezes excedem o limite de *tokens-por-minuto* (TPM) ou a janela de
contexto. A presença do *gate* `validate` + `retry-llm` no pipeline é precisamente o
mecanismo de mitigação para esse tipo de falha transitória.

---

## 7. Resultados — Comparação entre os LLMs

O objeto central da avaliação são os **6 modelos de linguagem**. A pergunta prática é:
*qual LLM gera os dicionários de dados semanticamente mais próximos do `ground truth`?*
Todos os resultados desta seção usam o mesmo instrumento de medição (BGE), fixado para que
a comparação seja entre LLMs e não entre réguas (ver a nota metodológica em §7.5).

### 7.1 Ranking de qualidade por LLM

A Figura 7 ordena os modelos pela similaridade média de cosseno (com intervalo de confiança
de 95%), calculada sobre os 4.701 valores brutos do bloco `results`.

![Figura 7 — Ranking dos LLMs por similaridade média](images/fig_llm_ranking.png)
**Figura 7.** Ranking dos LLMs. `deepseek-v4-flash` lidera (0,798); `gpt-5.4-nano` é o
último por margem clara (0,758). Os quatro modelos do meio são estatisticamente próximos.

`metrics_by_model` (instrumento BGE):

| Posição | Modelo | mean | std | median | d90 | d99 | min | max | n |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1º | **deepseek-v4-flash** | **0,7978** | 0,0905 | 0,8079 | 0,9102 | 0,9801 | 0,5165 | 0,9930 | 873 |
| 2º | deepseek-v4-pro | 0,7907 | 0,0832 | 0,7996 | 0,8882 | 0,9728 | 0,5295 | 0,9931 | 873 |
| 3º | gpt-5.4-mini | 0,7862 | 0,0787 | 0,7940 | 0,8833 | 0,9430 | 0,5387 | 0,9632 | 717 |
| 4º | gemini-3.5-flash | 0,7849 | 0,0861 | 0,7932 | 0,8916 | 0,9501 | 0,5350 | 0,9939 | 756 |
| 5º | gemini-3.1-flash-lite | 0,7824 | 0,0907 | 0,7865 | 0,8990 | 0,9698 | 0,5163 | 0,9930 | 757 |
| 6º | gpt-5.4-nano | 0,7581 | 0,0745 | 0,7661 | 0,8489 | 0,8956 | 0,5062 | 0,9298 | 725 |

**Leitura:** o topo (`deepseek-v4-flash`) e o fundo (`gpt-5.4-nano`) são separações robustas;
os quatro modelos intermediários (0,782–0,791) estão dentro de ~1 IC entre si — um
quase-empate técnico. O *nano*, além da menor média, tem a menor cauda superior
(d99 = 0,896, contra >0,94 de todos os outros): mesmo nos campos mais fáceis ele raramente
chega às descrições quase perfeitas que os demais alcançam.

### 7.2 Distribuição campo-a-campo

A Figura 8 mostra a distribuição completa por modelo. As medianas são próximas; o que
diferencia os LLMs é a **cauda inferior** — campos de nomenclatura opaca que os modelos
menores descrevem mal.

![Figura 8 — Box plot da similaridade por LLM](images/fig_box_bge.png)
**Figura 8.** Distribuição campo-a-campo por LLM (n = 4.701). `gpt-5.4-nano` tem a caixa mais
baixa e a menor dispersão para cima; `deepseek-v4-flash` tem a mediana e a cauda superior
mais altas.

### 7.3 Confronto direto por tabela (quem vence onde)

Médias agregadas escondem a variância por tabela. A Figura 9 conta, para cada uma das 75
tabelas, **qual LLM produz a maior similaridade média** — uma comparação *head-to-head* mais
robusta a *outliers* do que a média global.

![Figura 9 — Vitórias por tabela entre os LLMs](images/fig_llm_winrate.png)
**Figura 9.** Nº de tabelas (de 75) em que cada LLM é o melhor. A linha tracejada conta os
quase-empates (a ≤ 0,005 do topo). `deepseek-v4-flash` vence **30/75** tabelas (e fica no
quase-empate em 39); `gpt-5.4-nano` vence apenas **2**.

| Modelo | Vitórias (de 75) | Quase-empates (≤0,005 do topo) |
| --- | ---: | ---: |
| **deepseek-v4-flash** | **30** | 39 |
| deepseek-v4-pro | 15 | 24 |
| gemini-3.5-flash | 13 | 19 |
| gemini-3.1-flash-lite | 8 | 11 |
| gpt-5.4-mini | 7 | 12 |
| gpt-5.4-nano | 2 | 3 |

Esse confronto por tabela **confirma e reforça** o sinal principal: `deepseek-v4-flash` é o
vencedor dominante (vence 40% das tabelas, mais que o dobro do segundo colocado). Aqui a
separação entre o líder e o pelotão do meio é **mais nítida** do que na média global — sinal
de que o `deepseek-v4-flash` é a escolha mais consistente para esta tarefa. O `nano` é o pior
por média global, embora pelo confronto vença algumas tabelas (2 sob BGE); o seu ponto fraco
é a média, não cada tabela individualmente.

### 7.4 Visão por banco de dados

A Figura 10 detalha a média de similaridade por banco × LLM, revelando que a dificuldade é
mais função do **domínio/tabela** do que do modelo: bancos com nomenclatura mais opaca
produzem médias menores transversalmente a todos os LLMs.

![Figura 10 — Heatmap similaridade média por banco × LLM](images/fig_heatmap_db.png)
**Figura 10.** Média de similaridade por banco BIRD × LLM. A variação **vertical** (entre
bancos) rivaliza com a **horizontal** (entre modelos): o domínio dos dados é um fator de
dificuldade tão importante quanto a escolha do LLM. Ainda assim, dentro de cada banco a
coluna do `deepseek-v4-flash` tende a ser a mais clara (maior similaridade).

### 7.5 Análise de custo e custo-benefício

Qualidade isolada não decide a escolha em produção: o **custo de execução** é o outro eixo.
A Tabela abaixo cruza o custo (USD) para gerar os dicionários das 75 tabelas, por modelo, com
a qualidade (média BGE) e as vitórias no confronto direto (§7.3). A métrica de custo-benefício
adotada é **vitórias por dólar** (`vitórias ÷ custo`), que combina qualidade *head-to-head* e
preço numa só leitura.

| Modelo | Custo (US$) | Média BGE | Vitórias (de 75) | **Vitórias / US$** |
| --- | ---: | ---: | ---: | ---: |
| **deepseek-v4-flash** | 0,270 | **0,7978** | **30** | **111,1** |
| deepseek-v4-pro | 0,860 | 0,7907 | 15 | 17,4 |
| gemini-3.1-flash-lite | 0,504 | 0,7824 | 8 | 15,9 |
| gpt-5.4-nano | **0,226** | 0,7581 | 2 | 8,8 |
| gpt-5.4-mini | 0,896 | 0,7862 | 7 | 7,8 |
| gemini-3.5-flash | **4,020** | 0,7849 | 13 | 3,2 |

A Figura 11 posiciona cada modelo no plano **custo × qualidade** (eixo de custo em escala
logarítmica).

![Figura 11 — Custo vs. qualidade por LLM](images/fig_cost_quality.png)
**Figura 11.** Custo (US$, log) vs. similaridade média (BGE). A linha tracejada é a
**fronteira de Pareto**. `deepseek-v4-flash` está no canto ideal (maior qualidade, segundo
menor custo); `gemini-3.5-flash` está no canto ruim (caro e não-líder).

**Achados de custo-benefício:**

1. **`deepseek-v4-flash` domina por Pareto todos os modelos exceto o mais barato.** Ele tem a
   **maior qualidade** (0,798) **e** o **segundo menor custo** (US$ 0,27) — portanto domina
   `deepseek-v4-pro`, `gpt-5.4-mini`, `gemini-3.5-flash` e `gemini-3.1-flash-lite`, que são
   simultaneamente **mais caros e piores**. A fronteira de Pareto tem apenas dois pontos:
   `gpt-5.4-nano` (o mais barato) e `deepseek-v4-flash` (o melhor).
2. **`gemini-3.5-flash` é o pior custo-benefício por larga margem.** Custa **US$ 4,02 — ~15×**
   o `deepseek-v4-flash` — e ainda assim fica em 4º na qualidade média e em 3º nas vitórias.
   Seu índice de **3,2 vitórias/US$** é o menor de todos (a Figura 12 deixa isso explícito).
3. **`gpt-5.4-nano` é o mais barato, mas o desconto não compensa.** Economizar US$ 0,044
   (~16%) em relação ao `flash` custa **−0,040 de qualidade** e a queda de **30 → 2 vitórias**.
   Só faz sentido se o custo for absolutamente crítico e a qualidade semântica, secundária.
4. **A família DeepSeek é a melhor relação custo-qualidade.** O `flash` é a escolha racional
   default; o `pro` (17,4 vitórias/US$) é um distante segundo lugar em eficiência, à frente de
   todos os modelos OpenAI e Google.

![Figura 12 — Custo-efetividade (vitórias por dólar)](images/fig_cost_effectiveness.png)
**Figura 12.** Custo-efetividade = vitórias por tabela ÷ custo (US$). `deepseek-v4-flash`
(111,1) é **~6×** mais custo-efetivo que o segundo colocado e **~35×** mais que o
`gemini-3.5-flash`.

> **Nota sobre a unidade de custo.** Os valores em US$ são o custo total informado para a
> passada de geração das 75 tabelas por modelo. Como o pipeline é idêntico entre modelos
> (mesmo *prompt*, mesmos perfis/amostras), as **razões de custo entre modelos** são o que
> importa para a decisão e são robustas à unidade exata.

### 7.6 Nota metodológica — o instrumento de medição

Por completude, registra-se que a comparação acima usa `BAAI/bge-small-en-v1.5` como
*embedding*. Um estudo de robustez (detalhado em `reports/EMBEDDING_MODEL.md`) repetiu a
medição com `all-MiniLM-L6-v2`. Duas coisas mudam e duas se mantêm:

**O que muda com a régua:** os valores **absolutos** (BGE reporta +0,16–0,19 de média e ~50%
menos desvio, pois é uma régua diferente) **e o ranking de média do pelotão intermediário**.
Por média global, sob MiniLM o líder é na verdade `gemini-3.5-flash` (0,6242), com
`deepseek-v4-flash` em 2º (0,6196) e `deepseek-v4-pro` caindo para 5º — ou seja, o ranking
*por média* dos quatro modelos do meio **não** é estável e **não** se deve dizer que "a
família DeepSeek lidera sob ambas as réguas".

**O que se mantém (conclusões robustas):**

1. **`gpt-5.4-nano` é o pior por média global sob ambas as réguas** (BGE 0,758; MiniLM 0,570).
2. **`deepseek-v4-flash` vence o maior número de tabelas no confronto direto sob ambas as
   réguas** — **30/75 sob BGE** e **25/75 sob MiniLM**, em ambos os casos com folga clara
   sobre o segundo colocado (15). É essa métrica *head-to-head* (Figura 9), e não a média
   global, que sustenta a recomendação do `deepseek-v4-flash`.

| Modelo | Vitórias sob BGE | Vitórias sob MiniLM |
| --- | ---: | ---: |
| **deepseek-v4-flash** | **30** | **25** |
| deepseek-v4-pro | 15 | 15 |
| gemini-3.5-flash | 13 | 11 |
| gemini-3.1-flash-lite | 8 | 13 |
| gpt-5.4-mini | 7 | 5 |
| gpt-5.4-nano | 2 | 6 |

> **Cautela estatística:** a alta correlação *por campo* entre as duas réguas
> (Pearson *r* = 0,85 sobre 4.701 pontos) mede concordância **ponto-a-ponto**, dominada pela
> variância de dificuldade dos campos compartilhada pelas duas réguas — ela **não** é, por si
> só, prova de estabilidade de *ranking*. A estabilidade do ranking deve ser afirmada apenas
> onde os dados a sustentam: o confronto por tabela (itens 1–2), não a média global do meio.
> Não compare *scores* absolutos entre réguas nem reuse *thresholds* sem recalibrar.

---

## 8. Discussão

1. **Qual LLM escolher.** A evidência mais acionável é o confronto por tabela (Figura 9):
   `deepseek-v4-flash` vence em 40% das tabelas — mais que o dobro do segundo colocado — e
   integra o pelotão de topo de forma consistente. Para esta tarefa (gerar descrições de
   dicionário a partir de perfil + amostra), a família DeepSeek é a recomendação; o
   `gpt-5.4-nano` deve ser evitado quando a qualidade semântica importa.

2. **Custo vs. qualidade (§7.5).** A análise de custo torna a recomendação inequívoca:
   `deepseek-v4-flash` **domina por Pareto** todos os modelos exceto o mais barato — entrega a
   maior qualidade pelo segundo menor preço (US$ 0,27) e é **~6× mais custo-efetivo** que o
   segundo colocado. No extremo oposto, `gemini-3.5-flash` é o pior negócio: paga ~15× mais
   caro por qualidade não-líder. O `nano`, apesar de ser o mais barato, não compensa — o
   desconto de ~16% custa uma queda grande de qualidade e de vitórias.

3. **Domínio domina dificuldade.** A Figura 10 sugere que ganhos futuros virão mais de
   enriquecer o contexto do *prompt* para domínios opacos (ex.: glossários, *value
   descriptions* dos CSVs BIRD) do que de trocar de LLM — a variação entre bancos rivaliza
   com a variação entre modelos.

4. **Falhas são operacionais, não semânticas.** As 8 falhas (1,8%) são todas de limite de
   *tokens*/quota em tabelas grandes — mitigáveis por *chunking* por colunas ou compactação
   de perfil/amostra, e já isoladas pelo mecanismo `validate`/`retry-llm`.

5. **O que é estável na medição.** As conclusões *robustas* à troca de instrumento são duas
   (§7.5): `gpt-5.4-nano` é o pior por média global sob ambas as réguas, e `deepseek-v4-flash`
   vence mais tabelas no confronto direto sob ambas (30/75 e 25/75). Já o *ranking de média*
   do pelotão intermediário **não** é estável (sob MiniLM, `gemini-3.5-flash` lidera a média).
   Por isso a recomendação prática se apoia no confronto *head-to-head*, não na média global.

---

## 9. Limitações e Trabalhos Futuros

- **Comparação por `field_name`.** A avaliação casa campos por nome exato; se um LLM
  renomeia um campo, a correspondência é perdida (subestimando o modelo). Um *matching*
  semântico de campos (e não apenas de descrições) é um próximo passo.
- **Perfis compactos do *bootstrap*.** Os perfis gerados pelo *bootstrap* são reproduzíveis
  mas não substituem todas as estatísticas do `dataprofiler`.
- **STS ≠ tarefa real.** Como alertam Muennighoff et al. (2023), STS correlaciona
  imperfeitamente com casos de uso reais; por isso a validação empírica no corpus é essencial.
- **Roadmap do projeto:** métricas/visualizações de *schema matching* mais ricas; *prompts*
  alternativos (*few-shot*, *chain-of-thought*); novos domínios (saúde, finanças, varejo);
  e um **módulo multiagente** para gerar código de integração (SQL/Python/PySpark) a partir
  dos mapeamentos.

---

## 10. Reprodutibilidade

```bash
# Baseline (MiniLM) — lê o summary pré-existente
python -c "import json; d=json.load(open('data/distance_calculation_old_alll_MiniLm-L6-v2/all_similarities_results.json')); print(d['metrics_by_model'])"

# BGE (novo default) — gera novo summary
python run.py compare && python run.py metrics
# Artefatos:
#   data/distance_calculation/all_similarities_results.json     (BGE, ativo)
#   data/distance_calculation/all_similarities_results.BGE.json (cópia BGE)
#   data/distance_calculation_old_alll_MiniLm-L6-v2/...         (MiniLM, intocado)
```

Os gráficos das Figuras 7–12 foram gerados a partir do bloco `results` (distribuições
brutas) dos *summaries* acima; os diagramas das Figuras 1–6 são fontes Mermaid renderizadas
para PNG em `reports/images/`.

---

## 11. Conclusão

O `schema_generator` demonstra que LLMs guiados por *prompts* e alimentados por *perfis +
amostras* produzem dicionários de dados semânticos de alta qualidade (similaridade mediana
≈ 0,79–0,81 contra um *ground truth* curado), com um pipeline reprodutível, auditável e
dirigido por configuração. A principal conclusão prática é o **ranking dos modelos de
linguagem**: entre os 6 LLMs avaliados, `deepseek-v4-flash` é o mais consistente — vencedor
do confronto direto em 30 das 75 tabelas (e 25/75 sob o instrumento alternativo) —, o pelotão
intermediário (`deepseek-v4-pro`, `gpt-5.4-mini`, os dois Gemini) é tecnicamente empatado, e
`gpt-5.4-nano` é o pior por média global. As conclusões robustas à troca do modelo de
*embedding* são o *nano* como pior por média e o *flash* como vencedor *head-to-head*; o
ranking de média do meio, esse depende do instrumento (§7.5). O *framework* está pronto para
estender-se a *schema matching* operacional e à geração automática de código de integração.

---

## Referências

Reaproveitadas de `reports/EMBEDDING_MODEL.md` (somente *venues peer-reviewed* e benchmarks
oficiais HuggingFace):

1. Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese
   BERT-Networks*. EMNLP-IJCNLP 2019, pp. 3982–3992. [ACL D19-1410](https://aclanthology.org/D19-1410/)
2. Wang, W. et al. (2020). *MiniLM: Deep Self-Attention Distillation for Task-Agnostic
   Compression of Pre-Trained Transformers*. NeurIPS 2020.
3. Gao, T., Yao, X. & Chen, D. (2021). *SimCSE: Simple Contrastive Learning of Sentence
   Embeddings*. EMNLP 2021, pp. 6894–6910. [ACL 2021.emnlp-main.552](https://aclanthology.org/2021.emnlp-main.552/)
4. Muennighoff, N., Tazi, N., Magne, L. & Reimers, N. (2023). *MTEB: Massive Text Embedding
   Benchmark*. EACL 2023, pp. 2014–2037. [ACL 2023.eacl-main.148](https://aclanthology.org/2023.eacl-main.148/)
5. BAAI (2024). *Model card: BAAI/bge-small-en-v1.5*. HuggingFace.
6. BIRD-bench. *BIRD Mini-Dev*. <https://bird-bench.github.io/>

---

*Documento gerado por análise direta do código-fonte, dos *summaries* de similaridade e dos
relatórios do projeto. Figuras 1–6: Mermaid → PNG. Figuras 7–12: matplotlib sobre as
distribuições brutas em `all_similarities_results(.BGE).json`.*
