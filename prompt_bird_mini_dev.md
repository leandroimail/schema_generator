# Prompt: BIRD Mini-Dev Metadata Extraction

Quero que você atue como um engenheiro de dados sênior e agente de implementação Python, trabalhando diretamente neste repositório.

## Objetivo

Implementar uma solução reutilizável para processar o benchmark BIRD Mini-Dev (`https://github.com/bird-bench/mini_dev`) e gerar, para todas as bases de dados e todas as tabelas do benchmark, os artefatos necessários para usar este projeto como fonte da verdade de dicionários de dados e comparação com dicionários gerados por IA.

## Contexto do repositório

Antes de codar, leia e siga os contratos já existentes no projeto, principalmente:

- `src/dictionary_generation.py`
- `src/compare_results_dictionary.py`
- `src/samples.py`
- `src/profilling.py`
- `config.yaml`
- `prompts/dictionary_generation.md`

## Requisitos funcionais

1. Baixar ou consumir a estrutura do benchmark BIRD Mini-Dev.
2. Identificar todas as bases de dados do benchmark.
3. Para cada base:
   - localizar o SQLite correspondente;
   - extrair todas as tabelas;
   - extrair metadados completos de cada tabela.
4. Para cada tabela, gerar um dicionário-base ("ground truth") compatível com `src/compare_results_dictionary.py`, contendo no mínimo:
   - `table_name`
   - `table_description`
   - `fields` (lista)
   - para cada campo:
     - `field_name`
     - `data_type`
     - `field_description`
     - `example_value` quando fizer sentido
     - `domain_values` quando fizer sentido
     - `full_description`
5. Sempre que possível, enriquecer descrições usando os artefatos de descrição do próprio benchmark, e complementar com introspecção do SQLite.
6. Gerar amostragens de dados para profiling e geração de prompt, respeitando a lógica e configuração já existentes em `config.yaml`.
7. Produzir artefatos que possam ser usados pelos scripts existentes, especialmente:
   - `src/profilling.py`
   - `src/dictionary_generation.py`
   - `src/compare_results_dictionary.py`
8. A abordagem deve ser criar um script reutilizável, robusto e configurável, e não um notebook descartável.
9. O script deve ser implementado e testado por você no repositório.

## Restrições de implementação

- Não quebre os contratos já existentes sem necessidade.
- Reaproveite padrões do projeto antes de criar novas abstrações.
- Se precisar ajustar `config.yaml` ou adaptar paths/nomes para suportar múltiplas bases/tabelas do benchmark, faça isso de forma consistente e explícita.
- Preserve compatibilidade com o formato esperado por `src/compare_results_dictionary.py`.
- A solução deve funcionar para todas as tabelas de todas as bases do benchmark, não apenas um caso isolado.
- Evite hardcode de nomes de bases/tabelas.
- Se houver ambiguidades no benchmark, documente as suposições no código e no relatório final.

## Entregáveis

1. Um script Python reutilizável, por exemplo algo como:
   - `src/bootstrap_bird_mini_dev.py`
   - ou nome equivalente mais adequado ao projeto.
2. Estruturas de saída para:
   - dicionários-base por tabela;
   - samples por tabela;
   - perfis por tabela, se fizer parte do pipeline automatizado.
3. Ajustes mínimos necessários em `config.yaml` e/ou no `README`.
4. Testes ou validações executáveis.
5. Um resumo final com:
   - o que foi implementado;
   - quais arquivos foram alterados;
   - como executar;
   - limitações remanescentes.
   - salve como .md o resumo e indique onde foi salvo

## Plano de execução esperado

1. Inspecione o código atual e explique brevemente os contratos de entrada/saída relevantes.
2. Inspecione a estrutura do benchmark Mini-Dev antes de implementar.
3. Proponha uma estratégia curta e objetiva.
4. Implemente o script.
5. Ajuste configurações e integração com o pipeline existente.
6. Execute testes/validações reais localmente.
7. Corrija eventuais falhas encontradas.
8. Entregue o resumo final.

## Critérios de aceitação

Considere a tarefa concluída somente se:

- o benchmark Mini-Dev tiver sido percorrido programaticamente;
- os SQLite tiverem sido inspecionados;
- os dicionários-base gerados estiverem no formato aceito por `src/compare_results_dictionary.py`;
- as amostragens tiverem sido geradas conforme a configuração;
- o pipeline estiver reproduzível;
- o código tiver sido testado de verdade, não apenas descrito.

## Formato da resposta durante a execução

- Seja direto e técnico.
- Informe decisões e suposições de forma explícita.
- Não pare em proposta; implemente.
- Se algo bloquear a execução, tente contornar primeiro e só então reporte o bloqueio.
- No final, traga evidências objetivas de teste.

## Reforço recomendado

Antes de implementar, produza também uma matriz de compatibilidade entre:

- o formato de metadados extraído do BIRD Mini-Dev;
- e o formato esperado por `src/compare_results_dictionary.py`.

Essa matriz deve apontar:

- campos obrigatórios;
- campos opcionais;
- transformações necessárias;
- riscos de incompatibilidade.
