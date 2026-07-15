# Pipeline Eleitoral JSON

Pacote principal do projeto. Ele trabalha somente com JSON/JSONL/NDJSON ja preparados e organiza o processamento em tres blocos:

1. `stage_individual.py`: analise individual por arquivo e por ano.
2. `stage_global.py`: consolidacao das saidas individuais em tabelas globais CSV/Parquet.
3. `stage_simulation.py`: simulacao de 2026 alimentada pela base global.

Para bases grandes, o caminho recomendado agora e criar primeiro o banco limpo em Parquet:

- `bronze`: JSONs normalizados e unidos por schema/campos iguais;
- `prata`: tabelas limpas por dominio, particionadas por UF;
- `ouro`: tabelas prontas para graficos, analises, dashboard e simulacao.

## Execucao

Pela raiz do projeto:

```bash
python scripts/run_pipeline_eleitoral_json.py dados --out completo --modo completo --predict-2026
```

Criar somente o banco limpo base (`bronze` + `prata`):

```bash
python scripts/run_pipeline_eleitoral_json.py dados/json --modo banco --banco-out dados/banco_eleitoral --banco-overwrite
```

Rodar camada `ouro`, analises e simulacao usando somente o banco limpo:

```bash
python scripts/run_pipeline_eleitoral_json.py dados/json --modo analise_banco --banco-out dados/banco_eleitoral --predict-2026
```

Usar a camada `ouro` ja existente sem reconstruir o banco:

```bash
python scripts/run_pipeline_eleitoral_json.py dados/banco_eleitoral --modo dashboard_banco --banco-out dados/banco_eleitoral
```

Dashboard Streamlit do mesmo run, junto com os HTMLs gerados:

```bash
streamlit run scripts/dashboard_streamlit_eleitoral.py -- --run resultados/completo
```

Dashboard alternativo em Dash/DuckDB, lendo os Parquets tratados diretamente:

```bash
python scripts/dashboard_dash_eleitoral.py --run resultados/completo --host 127.0.0.1 --port 8050
```

Dashboard Dash lendo o banco limpo diretamente:

```bash
python scripts/dashboard_dash_eleitoral.py --run dados/banco_eleitoral --host 127.0.0.1 --port 8050
```

Tambem pode ser executado como modulo se `scripts` estiver no `PYTHONPATH`:

```bash
python -m pipeline_eleitoral_json dados --out completo --modo completo --predict-2026
```

## Estrutura

```text
pipeline_eleitoral_json/
  config.py
  main.py
  stage_individual.py
  stage_global.py
  stage_simulation.py
  clean_database.py
  json_reader.py
  profiler.py
  aggregation.py
  global_correlation.py
  global_cluster_analysis.py
  electoral_analysis.py
  comportamento_eleitoral.py
  simulation.py
  explainability.py
  stats.py
  plots.py
  utils.py
```

## Saidas

O pipeline gera:

- relatorios individuais por arquivo;
- `dados/banco_eleitoral/bronze`, `prata` e `ouro`, quando `--modo banco` for usado;
- `dados/banco_eleitoral/ouro/base_gold_global/`, dataset Parquet particionado por ano/UF para analise, dashboard e simulacao no modo `analise_banco`;
- subpastas individuais por ano;
- `global/tabelas/base_gold_global.csv`;
- `global/parquet/base_gold_global.parquet`, quando `--parquet` estiver ativo;
- `global/correlacao_codigos/parquet/por_ano/ano_<ano>/`, com Parquets anuais correlacionados por codigo de municipio/zona/secao, sem UF na chave;
- `global/correlacao_codigos/tabelas/manifesto_parquets_correlacionados_por_ano.csv`;
- `global/correlacao_codigos/clusters/`, com clusters globais discriminados, graficos e interpretacao;
- analises eleitorais globais;
- clusters comportamentais e graficos;
- simulacao 2026 por secao/municipio e resumo nacional;
- simulacao 2026 por partido em `preditivo_2026/parquet/partidos_2026_brasil.parquet`, `partidos_2026_estados.parquet` e `partidos_2026_municipios.parquet`;
- perfil de eleitor associado a cada partido e justificativa de correlacao historica em `partidos_2026_correlacao_historica.parquet`.

## Processamento grande

- `--full-aggregations --parquet` escreve partes Parquet por arquivo e evita manter o JSON inteiro na memoria.
- `--aggregate-chunk-rows` controla o tamanho dos blocos lidos do JSON.
- `--workers-individual` paraleliza arquivos pequenos; `--workers-large-files 1` mantem arquivos gigantes em fila para nao estourar RAM.
- `--workers-parquet` paraleliza escrita/particionamento Parquet.
- No modo banco, `0` nas flags `--banco-workers`, `--banco-workers-large-files`, `--banco-chunk-rows`, `--banco-ouro-workers` e `--banco-duckdb-threads` ativa auto-tuning por CPU/RAM.
- O modo banco usa uma fila unica do menor arquivo para o maior, processando um arquivo por vez.
- `--banco-workers` e `--banco-workers-large-files` controlam quantos workers internos podem ser usados no arquivo atual. JSONL/NDJSON e dividido por faixas; JSON array/outros formatos usam leitor streaming e lotes paralelos.
- `--banco-apagar-json-apos-processar` apaga o JSON original somente depois que aquele arquivo foi gravado com sucesso em Parquet.
- `--banco-ouro-workers` executa queries independentes da camada ouro em paralelo.
- `--banco-duckdb-threads` controla o total aproximado de threads usadas pelo DuckDB nas agregacoes da camada ouro.
- `--engine pyspark` ou `--engine auto` usa PySpark como alternativa quando instalado; sem PySpark, o pipeline continua com pandas + Parquet.

## Limite metodologico

As analises de perfil e voto usam dados agregados. Quando o projeto cruza perfil de eleitorado e voto, isso e uma proxy ecologica por territorio/secao, nao uma prova de voto individual.
