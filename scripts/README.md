# Catalogo de Scripts

Esta pasta concentra todo o codigo operacional do projeto.

```text
scripts/
  run_pipeline_eleitoral_json.py
  dashboard_streamlit_eleitoral.py
  dashboard_dash_eleitoral.py
  extracao/
    extrair_organizar_dados.py
    extrair_organizar_dados_json.py
  analise_individual/
    analisar_json_eleicoes_amostra_gpu_v5.py
  pipeline_eleitoral_json/
    stage_individual.py
    stage_global.py
    stage_simulation.py
    clean_database.py
    main.py
    ...
```

## Blocos

### `extracao/`

Scripts para extrair dados brutos do TSE, abrir ZIPs, separar PDFs e converter CSVs para CSV agrupado ou JSONL.

### `analise_individual/`

Script legado/auxiliar para analise amostral segura de JSONs grandes.

### `pipeline_eleitoral_json/`

Pacote principal do pipeline eleitoral:

- `stage_individual.py`: analise individual por arquivo e por ano;
- `stage_global.py`: consolidacao das saidas individuais em tabelas globais CSV/Parquet;
- `stage_simulation.py`: simulacao baseada na base global;
- `main.py`: orquestracao do fluxo;
- `global_correlation.py`: correlacao global por ano/codigo e geracao de Parquets anuais;
- `global_cluster_analysis.py`: clusters globais focados em valores discriminados/categoricos;
- `electoral_analysis.py`: perguntas eleitorais, vencedores por secao, perfil do eleitor e proxies;
- `comportamento_eleitoral.py`: clusters comportamentais;
- `simulation.py`: cenarios e Monte Carlo.
- `clean_database.py`: cria a base bronze/prata/ouro em Parquet, particionada por UF.

## Entrada Principal

Execute o pipeline pela raiz do projeto:

```bash
python scripts/run_pipeline_eleitoral_json.py dados --out completo --modo completo --predict-2026
```

Crie somente o banco limpo base (`bronze` + `prata`):

```bash
python scripts/run_pipeline_eleitoral_json.py dados/json --modo banco --banco-out dados/banco_eleitoral --banco-overwrite
```

Rode camada `ouro`, analises e simulacao usando o banco limpo:

```bash
python scripts/run_pipeline_eleitoral_json.py dados/json --modo analise_banco --banco-out dados/banco_eleitoral --predict-2026
```

Abra o dashboard Streamlit do mesmo run:

```bash
streamlit run scripts/dashboard_streamlit_eleitoral.py -- --run resultados/completo
```

Abra o dashboard Dash/DuckDB, que consulta os Parquets tratados sob demanda:

```bash
python scripts/dashboard_dash_eleitoral.py --run resultados/completo --host 127.0.0.1 --port 8050
```

Para o banco limpo, aponte o Dash direto para `dados/banco_eleitoral`:

```bash
python scripts/dashboard_dash_eleitoral.py --run dados/banco_eleitoral --host 127.0.0.1 --port 8050
```
