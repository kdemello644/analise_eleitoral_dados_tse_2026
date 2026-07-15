# Analise Eleitoral Brasileira com Dados Abertos do TSE

Este projeto cria um banco eleitoral em Parquet a partir de JSON/JSONL derivados dos Dados Abertos do TSE, organiza os dados em camadas `bronze`, `prata` e `ouro`, gera analises eleitorais, dashboards, graficos, relatorios em PDF e simulacoes de cenarios para 2026.

A fonte declarada dos dados e o Portal de Dados Abertos do TSE:

```text
https://dadosabertos.tse.jus.br/
```

O projeto foi desenhado para arquivos grandes. A regra principal e: os JSONs brutos sao lidos uma vez para criar Parquet; depois disso, as analises, dashboards e PDFs devem consultar os Parquets tratados.

## O Que Este Projeto Responde

O projeto busca responder, em diferentes niveis territoriais:

- quem sao os eleitores por ano, Brasil, UF, municipio, zona e secao;
- qual o perfil predominante do eleitorado: sexo/genero, faixa etaria, escolaridade, estado civil e raca/cor quando existir;
- como votaram os territorios por partido e candidato;
- qual perfil de eleitorado aparece associado a cada partido ou candidato;
- como o perfil eleitoral evolui entre anos;
- quais perfis aparecem em clusters eleitorais;
- qual cenario possivel de voto por partido pode ser simulado para 2026;
- como consultar tudo em dashboard, cards, mapas, graficos, HTML e PDF.

Importante: os dados do TSE usados aqui sao agregados. O projeto nao identifica pessoas individualmente e nao prova voto individual. Toda relacao entre perfil de eleitorado e voto e uma aproximacao territorial, tambem chamada de proxy ecologica.

## Visao Rapida Para Quem So Quer Rodar

Use estes comandos no WSL, dentro da raiz do projeto:

```bash
cd /mnt/c/Users/kevin/OneDrive/Escritorio/Analise_Eleitoral
```

Instale as dependencias:

```bash
pip install -r scripts/pipeline_eleitoral_json/requirements.txt
```

Crie o banco `bronze` + `prata` a partir dos JSONs:

```bash
python3 scripts/run_pipeline_eleitoral_json.py dados/json \
  --modo banco \
  --banco-out dados/banco_eleitoral \
  --resume \
  --banco-workers 8 \
  --banco-workers-large-files 8 \
  --banco-chunk-rows 10000
```

Teste rapido de criacao `bronze` + `prata`, processando poucos arquivos:

```bash
python3 scripts/run_pipeline_eleitoral_json.py dados/json \
  --modo banco \
  --banco-out dados/banco_eleitoral \
  --resume \
  --banco-max-files 5 \
  --banco-workers 2 \
  --banco-workers-large-files 1 \
  --banco-chunk-rows 5000
```

Gere uma analise rapida para testar o front, com todos os estados e Brasil, sem municipal detalhado:

```bash
python3 scripts/run_pipeline_eleitoral_json.py dados/banco_eleitoral \
  --modo analise_banco \
  --banco-out dados/banco_eleitoral \
  --resume \
  --banco-modalidade-analise estados_brasil \
  --banco-somente-estados-brasil \
  --banco-skip-heavy-analyses \
  --banco-skip-clusters \
  --sem-clustering \
  --banco-ouro-workers 1 \
  --banco-duckdb-threads 1 \
  --cenarios 100
```

Rode a API:

```bash
python3 scripts/dashboard_api_eleitoral.py \
  --run dados/banco_eleitoral \
  --host 0.0.0.0 \
  --port 8055 \
  --engine polars
```

Em outro terminal, rode o dashboard Streamlit:

```bash
streamlit run scripts/dashboard_streamlit_api_eleitoral.py -- \
  --api http://localhost:8055
```

Abra no navegador:

```text
http://localhost:8501
```

## Fluxo Geral Dos Dados

O fluxo completo e:

```text
Dados Abertos do TSE
  -> arquivos ZIP oficiais
  -> CSVs extraidos
  -> JSON/JSONL preparados
  -> banco Parquet bronze/prata
  -> camada ouro analitica
  -> dashboard/API/PDF/simulacao
```

Este repositorio trabalha principalmente a partir de `dados/json`, onde devem estar os JSON/JSONL/NDJSON ja preparados. A conversao ZIP -> CSV -> JSON pode ser feita antes, fora do pipeline principal.

## Pastas Principais

```text
Analise_Eleitoral/
  dados/
    json/
      arquivos JSON/JSONL/NDJSON de entrada
    banco_eleitoral/
      bronze/
      prata/
      ouro/
      preditivo_2026/
      logs/
      metadados/
  resultados/
    dashboards, HTMLs, PDFs e runs antigos
  scripts/
    pipeline e ferramentas de dashboard/PDF
  README.md
```

## Arquitetura Do Banco

### Camada Bronze

A `bronze` guarda dados em Parquet ainda proximos do arquivo original, mas ja normalizados o bastante para consulta. Ela serve para auditoria e preservacao.

Exemplo:

```text
dados/banco_eleitoral/bronze/
  eleitorado/schema_id=<hash>/uf=SP/shard=<arquivo>/part-000000.parquet
```

### Camada Prata

A `prata` e a camada limpa por dominio. Ela mantem somente o que interessa para analise, correlacao e agregacao.

Exemplo:

```text
dados/banco_eleitoral/prata/
  eleitorado/uf=SP/shard=<arquivo>/part-000000.parquet
  candidatos/uf=SP/shard=<arquivo>/part-000000.parquet
  resultados_votos/uf=SP/shard=<arquivo>/part-000000.parquet
```

Dominios principais:

- `eleitorado`: perfil do eleitor, eleitorado por local/secao, sexo/genero, faixa etaria, escolaridade, estado civil, raca/cor quando houver;
- `resultados_votos`: votos, partido, candidato, cargo, turno, municipio, zona e secao;
- `candidatos`: dados de candidatos, partido, cargo, municipio, sexo/genero, escolaridade, idade, estado civil quando disponivel;
- `outros`: arquivos que nao entram diretamente nas analises principais.

### Camada Ouro

A `ouro` contem os dados prontos para dashboard, PDF e simulacao.

Estrutura atual:

```text
dados/banco_eleitoral/ouro/
  municipal/
    resumo/
    perfil_eleitor/
    contagem_colunas_perfil_eleitor/
    resultado_partido/
    perfil_partido/
    contagem_colunas_perfil_partido/
    resultado_candidato/
    perfil_candidato/
    contagem_colunas_perfil_candidato/
    clusters_eleitores/
    contagem_colunas_clusters_eleitores/
    clusters_eleitores_resultado/
    contagem_colunas_clusters_eleitores_resultado/
  estadual/
    resumo/
    perfil_eleitor/
    contagem_colunas_perfil_eleitor/
    resultado_partido/
    perfil_partido/
    contagem_colunas_perfil_partido/
    resultado_candidato/
    perfil_candidato/
    contagem_colunas_perfil_candidato/
    clusters_eleitores/
    contagem_colunas_clusters_eleitores/
    clusters_eleitores_resultado/
    contagem_colunas_clusters_eleitores_resultado/
  brasil/
    resumo/
    perfil_eleitor/
    contagem_colunas_perfil_eleitor/
    resultado_partido/
    perfil_partido/
    contagem_colunas_perfil_partido/
    resultado_candidato/
    perfil_candidato/
    contagem_colunas_perfil_candidato/
    clusters_eleitores/
    contagem_colunas_clusters_eleitores/
    clusters_eleitores_resultado/
    contagem_colunas_clusters_eleitores_resultado/
  timeline_uf/
  timeline_municipal/
  timeline_nacional.parquet
  perfil_eleitor_por_ano/
  perfil_eleitor_por_partido/
  perfil_eleitor_por_candidato/
  top10_perfis_federacao_estado_municipio/
```

A ideia correta da `ouro` e:

1. gerar resultados menores;
2. reaproveitar os resultados menores;
3. subir de municipio para estado;
4. subir de estado para Brasil;
5. evitar varrer os dados brutos repetidamente.

### Tabelas Auxiliares De Contabilidade E Histogramas

A camada `ouro` tambem gera tabelas auxiliares chamadas `contagem_colunas_*`. Elas existem para que graficos, dashboard e PDF nao precisem recalcular histogramas em cima de tabelas grandes.

Essas tabelas sao a base dos histogramas. Cada linha representa uma contagem de uma dimensao discreta:

```text
nivel
ano
uf
cd_municipio
nm_municipio
dimensao_perfil
valor_perfil
qtd_pessoas ou qtd_votos
share_histograma
rank_histograma
grafico_sugerido
descricao
```

As dimensoes discretas contabilizadas sao:

```text
perfil_combinado
faixa_etaria
sexo_genero
escolaridade
estado_civil
raca_cor
```

Valores nulos, vazios, `sem valor`, `nao informado`, `#NULO#` e equivalentes sao descartados antes de gravar essas tabelas. Portanto, o front e o PDF nao devem precisar filtrar lixo depois.

Arquivos gerados:

```text
ouro/municipal/contagem_colunas_perfil_eleitor/
ouro/municipal/contagem_colunas_perfil_partido/
ouro/municipal/contagem_colunas_perfil_candidato/
ouro/municipal/contagem_colunas_clusters_eleitores/
ouro/municipal/contagem_colunas_clusters_eleitores_resultado/

ouro/estadual/contagem_colunas_perfil_eleitor/
ouro/estadual/contagem_colunas_perfil_partido/
ouro/estadual/contagem_colunas_perfil_candidato/
ouro/estadual/contagem_colunas_clusters_eleitores/
ouro/estadual/contagem_colunas_clusters_eleitores_resultado/

ouro/brasil/contagem_colunas_perfil_eleitor/
ouro/brasil/contagem_colunas_perfil_partido/
ouro/brasil/contagem_colunas_perfil_candidato/
ouro/brasil/contagem_colunas_clusters_eleitores/
ouro/brasil/contagem_colunas_clusters_eleitores_resultado/
```

Como interpretar:

- `contagem_colunas_perfil_eleitor`: quantas pessoas existem em cada perfil/dimensao do eleitorado;
- `contagem_colunas_perfil_partido`: dentro de cada partido, qual perfil de eleitor aparece associado aos votos daquele partido;
- `contagem_colunas_perfil_candidato`: dentro de cada candidato, qual perfil aparece associado aos votos daquele candidato;
- `contagem_colunas_clusters_eleitores`: histograma dos campos discretos dentro dos clusters de eleitores;
- `contagem_colunas_clusters_eleitores_resultado`: histograma dos clusters que combinam perfil de eleitor e resultado/partido.

Para eleitor, as tabelas trazem tambem comparecimento:

```text
qtd_pessoas
comparecimento_estimado
abstencao_estimado
taxa_comparecimento
taxa_abstencao
```

Para partido/candidato, a metrica central e:

```text
qtd_votos
share_histograma
resultado_eleitoral
```

`resultado_eleitoral` separa vencedores e nao vencedores quando a camada de resultado ja conseguiu rankear a disputa.

## Campos Mais Importantes

Chaves eleitorais:

```text
ano
uf
cd_municipio
nm_municipio
zona
secao
cargo
turno
partido
candidato
nr_votavel
sq_candidato
```

Metricas numericas mantidas:

```text
eleitorado
votos
comparecimento_estimado
abstencao_estimado
brancos
nulos
validos_estimados
```

Variaveis discretas prioritarias:

```text
sexo/genero
faixa etaria
escolaridade/grau de instrucao
estado civil
raca/cor
partido
candidato
cargo
turno
UF
municipio
zona
secao
```

Biometria nao e prioridade analitica e deve ser ignorada nos graficos e clusters, salvo necessidade especifica.

## Modos Do Pipeline Principal

O script principal e:

```bash
python3 scripts/run_pipeline_eleitoral_json.py <entrada> [opcoes]
```

### `--modo banco`

Cria ou continua a base `bronze` e `prata` a partir dos JSONs.

Use quando:

- voce ainda nao criou o banco Parquet;
- voce tem novos JSONs;
- voce quer transformar entrada bruta em base consultavel.

Comando:

```bash
python3 scripts/run_pipeline_eleitoral_json.py dados/json \
  --modo banco \
  --banco-out dados/banco_eleitoral \
  --resume \
  --banco-workers 8 \
  --banco-workers-large-files 8 \
  --banco-chunk-rows 10000
```

### `--modo analise_banco`

Usa a `prata` para criar/atualizar a `ouro`, gerar analises e simulacao.

Use quando:

- o banco `dados/banco_eleitoral/prata` ja existe;
- voce quer gerar dados para dashboard, PDF e predicao;
- voce quer rodar uma modalidade especifica.

### `--modo dashboard_banco`

Nao recria dados. Apenas aponta para o banco existente e informa o comando do dashboard.

Comando:

```bash
python3 scripts/run_pipeline_eleitoral_json.py dados/banco_eleitoral \
  --modo dashboard_banco \
  --banco-out dados/banco_eleitoral \
  --resume
```

### Modos legados por JSON direto

Estes modos existem, mas para bases grandes o recomendado e usar `banco` e `analise_banco`.

- `inventario`: inventaria JSONs e campos;
- `individual`: analise individual dos arquivos;
- `global`: analise global a partir das individuais;
- `preditivo`: simulacao a partir de uma global ja criada;
- `completo`: individual + global + preditivo.

Exemplo legado:

```bash
python3 scripts/run_pipeline_eleitoral_json.py dados \
  --out teste_legado \
  --modo completo \
  --parquet \
  --predict-2026
```

## Modalidades Da Camada Ouro

A flag principal e:

```bash
--banco-modalidade-analise <modalidade>
```

Modalidades disponiveis:

```text
completa
estados_brasil
eleitor
candidato
eleitor_partido
eleitor_candidato_partido
```

### `completa`

Roda tudo:

- perfil do eleitor;
- partido;
- candidato;
- clusters;
- municipal;
- estadual;
- Brasil;
- compatibilidade para dashboard/PDF;
- simulacao.

Comando pesado:

```bash
python3 scripts/run_pipeline_eleitoral_json.py dados/banco_eleitoral \
  --modo analise_banco \
  --banco-out dados/banco_eleitoral \
  --resume \
  --banco-modalidade-analise completa \
  --cluster-min-k 2 \
  --cluster-max-k 10 \
  --banco-ouro-workers 1 \
  --banco-duckdb-threads 1 \
  --cenarios 3000
```

### `estados_brasil`

Roda so UF e Brasil. Nao faz municipio por municipio.

Use para:

- testar o front rapidamente;
- gerar uma visao nacional e estadual;
- evitar a parte municipal longa.

Comando para todos os estados:

```bash
python3 scripts/run_pipeline_eleitoral_json.py dados/banco_eleitoral \
  --modo analise_banco \
  --banco-out dados/banco_eleitoral \
  --resume \
  --banco-modalidade-analise estados_brasil \
  --banco-somente-estados-brasil \
  --banco-skip-heavy-analyses \
  --banco-skip-clusters \
  --sem-clustering \
  --banco-ouro-workers 1 \
  --banco-duckdb-threads 1 \
  --cenarios 100
```

Comando para uma amostra de estados:

```bash
python3 scripts/run_pipeline_eleitoral_json.py dados/banco_eleitoral \
  --modo analise_banco \
  --banco-out dados/banco_eleitoral \
  --resume \
  --banco-modalidade-analise estados_brasil \
  --banco-somente-estados-brasil \
  --banco-ufs SP,RJ,MG,BA,PE,CE,RS,PR,PA,AM,GO,DF \
  --banco-skip-heavy-analyses \
  --banco-skip-clusters \
  --sem-clustering \
  --banco-ouro-workers 1 \
  --banco-duckdb-threads 1 \
  --cenarios 100
```

### `eleitor`

Roda analise focada no perfil geral do eleitor.

Gera principalmente:

- perfil do eleitor por territorio;
- top perfis;
- resumo eleitoral basico.

Comando:

```bash
python3 scripts/run_pipeline_eleitoral_json.py dados/banco_eleitoral \
  --modo analise_banco \
  --banco-out dados/banco_eleitoral \
  --resume \
  --banco-modalidade-analise eleitor \
  --banco-max-municipios-por-uf 20 \
  --banco-skip-clusters \
  --sem-clustering \
  --banco-ouro-workers 1 \
  --banco-duckdb-threads 1 \
  --cenarios 50
```

### `candidato`

Roda analise focada em candidato, sem clusters.

Gera:

- resultado por candidato;
- perfil associado ao candidato;
- fechamento por estado e Brasil quando houver dados suficientes.

Comando:

```bash
python3 scripts/run_pipeline_eleitoral_json.py dados/banco_eleitoral \
  --modo analise_banco \
  --banco-out dados/banco_eleitoral \
  --resume \
  --banco-modalidade-analise candidato \
  --banco-max-municipios-por-uf 20 \
  --banco-skip-clusters \
  --sem-clustering \
  --banco-ouro-workers 1 \
  --banco-duckdb-threads 1 \
  --cenarios 50
```

### `eleitor_partido`

Roda a modalidade rapida municipal mais util para dashboard:

- perfil geral do eleitor;
- resultado por partido;
- relacao eleitorado x partido;
- sem cluster;
- sem candidato.

Use quando voce quer saber: "quem e o eleitor associado a cada partido?".

Comando:

```bash
python3 scripts/run_pipeline_eleitoral_json.py dados/banco_eleitoral \
  --modo analise_banco \
  --banco-out dados/banco_eleitoral \
  --resume \
  --banco-modalidade-analise eleitor_partido \
  --banco-max-municipios-por-uf 20 \
  --banco-skip-clusters \
  --sem-clustering \
  --banco-ouro-workers 1 \
  --banco-duckdb-threads 1 \
  --cenarios 50
```

### `eleitor_candidato_partido`

Roda eleitor + partido + candidato, mas sem clusters.

Use quando voce quer comparar:

- perfil do eleitor;
- perfil associado a partido;
- perfil associado a candidato.

Comando:

```bash
python3 scripts/run_pipeline_eleitoral_json.py dados/banco_eleitoral \
  --modo analise_banco \
  --banco-out dados/banco_eleitoral \
  --resume \
  --banco-modalidade-analise eleitor_candidato_partido \
  --banco-max-municipios-por-uf 20 \
  --banco-skip-clusters \
  --sem-clustering \
  --banco-ouro-workers 1 \
  --banco-duckdb-threads 1 \
  --cenarios 100
```

## Atualizar Apenas Histogramas E Contagens Pendentes

As tabelas `contagem_colunas_*` sao tarefas da camada `ouro`. Quando voce usa `--resume`, o pipeline consulta os marcadores de progresso em `dados/banco_eleitoral/logs/ouro/` e pula as tarefas antigas que ja terminaram. Portanto, se o banco ja tem os dados principais processados, ele tende a executar apenas as tabelas novas ou pendentes.

Use este comando para gerar/atualizar contagens de Brasil + estados:

```bash
python3 scripts/run_pipeline_eleitoral_json.py dados/banco_eleitoral \
  --modo analise_banco \
  --banco-out dados/banco_eleitoral \
  --resume \
  --banco-modalidade-analise estados_brasil \
  --banco-somente-estados-brasil \
  --banco-skip-heavy-analyses \
  --banco-skip-clusters \
  --sem-clustering \
  --banco-ouro-workers 1 \
  --banco-duckdb-threads 1 \
  --cenarios 100
```

Use este comando para gerar/atualizar contagens por partido:

```bash
python3 scripts/run_pipeline_eleitoral_json.py dados/banco_eleitoral \
  --modo analise_banco \
  --banco-out dados/banco_eleitoral \
  --resume \
  --banco-modalidade-analise eleitor_partido \
  --banco-skip-clusters \
  --sem-clustering \
  --banco-ouro-workers 1 \
  --banco-duckdb-threads 1 \
  --cenarios 50
```

Use este comando para gerar/atualizar contagens por partido e candidato:

```bash
python3 scripts/run_pipeline_eleitoral_json.py dados/banco_eleitoral \
  --modo analise_banco \
  --banco-out dados/banco_eleitoral \
  --resume \
  --banco-modalidade-analise eleitor_candidato_partido \
  --banco-skip-clusters \
  --sem-clustering \
  --banco-ouro-workers 1 \
  --banco-duckdb-threads 1 \
  --cenarios 100
```

Use este comando quando quiser incluir histogramas de clusters:

```bash
python3 scripts/run_pipeline_eleitoral_json.py dados/banco_eleitoral \
  --modo analise_banco \
  --banco-out dados/banco_eleitoral \
  --resume \
  --banco-modalidade-analise completa \
  --cluster-min-k 2 \
  --cluster-max-k 10 \
  --banco-ouro-workers 1 \
  --banco-duckdb-threads 1 \
  --cenarios 3000
```

Se uma tabela antiga aparecer como `Pulando tarefa ouro ja concluida`, isso e esperado. O ponto importante e observar no log tarefas com nomes como:

```text
contagem_colunas_perfil_eleitor
contagem_colunas_perfil_partido
contagem_colunas_perfil_candidato
contagem_colunas_clusters_eleitores
```

## Dashboard Com API + Streamlit

O dashboard novo e separado em duas partes:

1. API FastAPI, que consulta Parquet com Polars;
2. Front Streamlit, que chama a API.

Isso evita embutir tabelas gigantes no HTML.

### Rodar API

```bash
python3 scripts/dashboard_api_eleitoral.py \
  --run dados/banco_eleitoral \
  --host 0.0.0.0 \
  --port 8055 \
  --engine polars
```

Documentacao interativa da API:

```text
http://localhost:8055/docs
```

Endpoints principais:

```text
GET  /api/health
GET  /api/modalidades
GET  /api/tabelas
GET  /api/progresso
GET  /api/logs
GET  /api/processamento
GET  /api/municipios?uf=SP
GET  /api/mapa/estados
GET  /api/brasil
GET  /api/perfis
GET  /api/partidos
GET  /api/candidatos
GET  /api/metricas
GET  /api/clusters
GET  /api/tabela
POST /api/analises/jobs
GET  /api/analises/jobs
GET  /api/analises/jobs/{job_id}
GET  /api/analises/jobs/{job_id}/logs
POST /api/pdf/jobs
GET  /api/pdf/jobs
GET  /api/pdf/jobs/{job_id}
GET  /api/pdf/jobs/{job_id}/logs
```

Os endpoints analiticos aceitam o parametro `modalidade`, por exemplo:

```text
/api/brasil?modalidade=estados_brasil
/api/perfis?nivel=estado&uf=SP&modalidade=eleitor
/api/partidos?escopo=municipio&uf=SP&modalidade=eleitor_partido
/api/candidatos?escopo=estado&uf=SP&tipo=perfil&modalidade=candidato
/api/clusters?nivel=brasil&modalidade=completa
```

Quando uma modalidade nao gera aquela parte, a API responde com `habilitado=false`, dados vazios e um aviso em vez de quebrar.

Exemplos de consulta da API por modalidade:

```bash
# completa: Brasil com perfil, partido e bases para cluster quando existirem
curl "http://localhost:8055/api/brasil?modalidade=completa&limit=20"

# estados_brasil: visao estadual/nacional, sem municipio detalhado
curl "http://localhost:8055/api/mapa/estados?modalidade=estados_brasil&limit=80"

# eleitor: perfil do eleitor em nivel de estado
curl "http://localhost:8055/api/perfis?nivel=estado&uf=SP&modalidade=eleitor&limit=20"

# candidato: resultado/perfil de candidato
curl "http://localhost:8055/api/candidatos?escopo=estado&uf=SP&tipo=perfil&modalidade=candidato&limit=20"

# eleitor_partido: ranking de partido e perfil associado
curl "http://localhost:8055/api/partidos?escopo=municipio&uf=SP&municipio=71072%7CSAO%20PAULO&modalidade=eleitor_partido&limit=20"

# eleitor_candidato_partido: partido + candidato + perfil
curl "http://localhost:8055/api/candidatos?escopo=municipio&uf=SP&municipio=71072%7CSAO%20PAULO&tipo=resultado&modalidade=eleitor_candidato_partido&limit=20"
```

### Rodar analises pela API

A API tambem consegue disparar o proprio `analise_banco` em background. Isso serve para rodar a camada ouro pelo front ou pelo Swagger sem abrir um terminal novo.

Endpoint:

```text
POST /api/analises/jobs
```

Exemplo pelo `curl`, rodando `estados_brasil`:

```bash
curl -X POST http://localhost:8055/api/analises/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "modalidade_analise": "estados_brasil",
    "somente_estados_brasil": true,
    "ufs": "",
    "max_municipios_por_uf": 0,
    "cenarios": 100,
    "banco_ouro_workers": 1,
    "banco_duckdb_threads": 1,
    "skip_heavy_analyses": true,
    "skip_clusters": true,
    "predict_2026": true
  }'
```

Exemplo `eleitor_partido` limitado a alguns estados:

```bash
curl -X POST http://localhost:8055/api/analises/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "modalidade_analise": "eleitor_partido",
    "ufs": "SP,RJ,MG,BA",
    "somente_estados_brasil": false,
    "max_municipios_por_uf": 20,
    "cenarios": 50,
    "banco_ouro_workers": 1,
    "banco_duckdb_threads": 1,
    "skip_heavy_analyses": true,
    "skip_clusters": true,
    "predict_2026": true
  }'
```

Ver jobs de analise:

```bash
curl http://localhost:8055/api/analises/jobs
```

Ver um job especifico:

```bash
curl http://localhost:8055/api/analises/jobs/SEU_JOB_ID
```

Ver logs de um job:

```bash
curl "http://localhost:8055/api/analises/jobs/SEU_JOB_ID/logs?max_lines=220"
```

Os logs ficam em:

```text
dados/banco_eleitoral/logs/api_jobs/
```

Disparar cada modalidade pela API:

```bash
# completa
curl -X POST http://localhost:8055/api/analises/jobs -H "Content-Type: application/json" -d '{"modalidade_analise":"completa","cenarios":3000,"banco_ouro_workers":1,"banco_duckdb_threads":1,"skip_heavy_analyses":false,"skip_clusters":false,"predict_2026":true}'

# estados_brasil
curl -X POST http://localhost:8055/api/analises/jobs -H "Content-Type: application/json" -d '{"modalidade_analise":"estados_brasil","somente_estados_brasil":true,"cenarios":100,"banco_ouro_workers":1,"banco_duckdb_threads":1,"skip_heavy_analyses":true,"skip_clusters":true,"predict_2026":true}'

# eleitor
curl -X POST http://localhost:8055/api/analises/jobs -H "Content-Type: application/json" -d '{"modalidade_analise":"eleitor","max_municipios_por_uf":20,"cenarios":50,"banco_ouro_workers":1,"banco_duckdb_threads":1,"skip_heavy_analyses":true,"skip_clusters":true,"predict_2026":false}'

# candidato
curl -X POST http://localhost:8055/api/analises/jobs -H "Content-Type: application/json" -d '{"modalidade_analise":"candidato","max_municipios_por_uf":20,"cenarios":50,"banco_ouro_workers":1,"banco_duckdb_threads":1,"skip_heavy_analyses":true,"skip_clusters":true,"predict_2026":false}'

# eleitor_partido
curl -X POST http://localhost:8055/api/analises/jobs -H "Content-Type: application/json" -d '{"modalidade_analise":"eleitor_partido","max_municipios_por_uf":20,"cenarios":50,"banco_ouro_workers":1,"banco_duckdb_threads":1,"skip_heavy_analyses":true,"skip_clusters":true,"predict_2026":true}'

# eleitor_candidato_partido
curl -X POST http://localhost:8055/api/analises/jobs -H "Content-Type: application/json" -d '{"modalidade_analise":"eleitor_candidato_partido","max_municipios_por_uf":20,"cenarios":100,"banco_ouro_workers":1,"banco_duckdb_threads":1,"skip_heavy_analyses":true,"skip_clusters":true,"predict_2026":true}'
```

### Rodar Streamlit

Em outro terminal:

```bash
streamlit run scripts/dashboard_streamlit_api_eleitoral.py -- \
  --api http://localhost:8055
```

Abra:

```text
http://localhost:8501
```

No Streamlit existem abas para:

- consultar Brasil, estado e municipio somente quando o usuario clica em buscar;
- carregar mapa e graficos a partir da API;
- acompanhar progresso e logs;
- consultar tabelas tratadas;
- gerar PDF pela API;
- rodar analise do banco ouro pela API.

Na aba `Rodar analise`, use os presets:

```text
Estados + Brasil rapido
Eleitor + partido rapido
Candidato rapido
Eleitor + candidato + partido
Completa segura
Personalizado
```

O front envia a modalidade para `POST /api/analises/jobs`, mostra o Job ID e permite consultar o log do processamento sem sair da tela.

## Dashboard HTML E PDF Estaticos

Para gerar varios HTMLs em `resultados/`, usando a camada ouro ja existente:

```bash
python3 scripts/gerar_dashboards_ouro_html_pdf.py dados/banco_eleitoral \
  --out resultados/dashboards_ouro \
  --top-n 20 \
  --max-municipios-por-estado 350 \
  --ano 2024 \
  --cenario base
```

Somente HTML, sem PDF:

```bash
python3 scripts/gerar_dashboards_ouro_html_pdf.py dados/banco_eleitoral \
  --out resultados/dashboards_ouro \
  --top-n 20 \
  --sem-pdf
```

Gerar apenas algumas UFs:

```bash
python3 scripts/gerar_dashboards_ouro_html_pdf.py dados/banco_eleitoral \
  --out resultados/dashboards_sp_rj_mg \
  --ufs SP,RJ,MG \
  --top-n 20 \
  --sem-pdf
```

## Relatorio PDF

Gerar PDF completo a partir dos Parquets:

```bash
python3 scripts/gerar_relatorio_pdf_eleitoral.py \
  --run dados/banco_eleitoral \
  --out dados/banco_eleitoral/relatorios/relatorio_completo_eleitoral.pdf \
  --modalidade-analise completa \
  --max-pages 1000 \
  --top-n 15 \
  --municipios-por-uf 20 \
  --query-engine polars \
  --duckdb-threads 1
```

Gerar PDF menor para teste:

```bash
python3 scripts/gerar_relatorio_pdf_eleitoral.py \
  --run dados/banco_eleitoral \
  --out resultados/relatorio_teste.pdf \
  --modalidade-analise estados_brasil \
  --max-pages 80 \
  --top-n 10 \
  --ufs SP,RJ,MG \
  --municipios-por-uf 5 \
  --query-engine polars \
  --duckdb-threads 1
```

Com logs detalhados:

```bash
python3 scripts/gerar_relatorio_pdf_eleitoral.py \
  --run dados/banco_eleitoral \
  --out resultados/relatorio_teste.pdf \
  --modalidade-analise eleitor_partido \
  --log-dir resultados/logs_pdf_teste \
  --max-pages 80 \
  --top-n 10 \
  --ufs SP,RJ,MG \
  --municipios-por-uf 5 \
  --query-engine polars
```

### PDF por modalidade

Troque apenas `--modalidade-analise` para gerar relatorios com foco diferente.

PDF `completa`:

```bash
python3 scripts/gerar_relatorio_pdf_eleitoral.py \
  --run dados/banco_eleitoral \
  --out resultados/pdf_completa.pdf \
  --modalidade-analise completa \
  --max-pages 1000 \
  --top-n 15 \
  --municipios-por-uf 20 \
  --query-engine polars \
  --duckdb-threads 1
```

PDF `estados_brasil`:

```bash
python3 scripts/gerar_relatorio_pdf_eleitoral.py \
  --run dados/banco_eleitoral \
  --out resultados/pdf_estados_brasil.pdf \
  --modalidade-analise estados_brasil \
  --max-pages 120 \
  --top-n 15 \
  --municipios-por-uf 0 \
  --query-engine polars \
  --duckdb-threads 1
```

PDFs separados por nivel, sempre na ordem Brasil -> estados -> municipios:

```bash
python3 scripts/gerar_relatorio_pdf_eleitoral.py \
  --run dados/banco_eleitoral \
  --out resultados/pdfs_por_nivel/relatorio.pdf \
  --modalidade-analise estados_brasil \
  --pdf-separado-por-nivel \
  --max-pages 120 \
  --top-n 20 \
  --municipios-por-uf 0 \
  --query-engine polars \
  --duckdb-threads 1
```

Quando a modalidade incluir municipio e `--municipios-por-uf` for maior que zero, o mesmo comando gera:

```text
00_brasil.pdf
01_estado_ac.pdf
02_estado_al.pdf
...
01_001_municipio_ac_<codigo>.pdf
manifesto_pdfs.json
```

PDF `eleitor`:

```bash
python3 scripts/gerar_relatorio_pdf_eleitoral.py \
  --run dados/banco_eleitoral \
  --out resultados/pdf_eleitor.pdf \
  --modalidade-analise eleitor \
  --max-pages 200 \
  --top-n 15 \
  --municipios-por-uf 10 \
  --query-engine polars \
  --duckdb-threads 1
```

PDF `candidato`:

```bash
python3 scripts/gerar_relatorio_pdf_eleitoral.py \
  --run dados/banco_eleitoral \
  --out resultados/pdf_candidato.pdf \
  --modalidade-analise candidato \
  --max-pages 200 \
  --top-n 15 \
  --municipios-por-uf 10 \
  --query-engine polars \
  --duckdb-threads 1
```

PDF `eleitor_partido`:

```bash
python3 scripts/gerar_relatorio_pdf_eleitoral.py \
  --run dados/banco_eleitoral \
  --out resultados/pdf_eleitor_partido.pdf \
  --modalidade-analise eleitor_partido \
  --max-pages 250 \
  --top-n 15 \
  --municipios-por-uf 10 \
  --query-engine polars \
  --duckdb-threads 1
```

PDF `eleitor_candidato_partido`:

```bash
python3 scripts/gerar_relatorio_pdf_eleitoral.py \
  --run dados/banco_eleitoral \
  --out resultados/pdf_eleitor_candidato_partido.pdf \
  --modalidade-analise eleitor_candidato_partido \
  --max-pages 300 \
  --top-n 15 \
  --municipios-por-uf 10 \
  --query-engine polars \
  --duckdb-threads 1
```

## Simulacao 2026

A simulacao e gerada no modo `analise_banco` quando o pipeline fecha os dados necessarios. Ela grava em:

```text
dados/banco_eleitoral/preditivo_2026/
```

Arquivos esperados:

```text
partidos_2026_brasil
partidos_2026_estados
partidos_2026_municipios
correlacao_historica
cenarios
```

Para uma simulacao leve:

```bash
--cenarios 50
```

Para uma simulacao mais robusta:

```bash
--cenarios 3000
```

Quanto maior `--cenarios`, mais demorado fica.

## Clustering

O clustering usa KMeans quando ativo.

Flags:

```bash
--clustering
--sem-clustering
--cluster-min-k 2
--cluster-max-k 10
```

Por padrao, as modalidades curtas pulam clusters. A modalidade `completa` pode gerar:

- `clusters_eleitores`: perfil do eleitor;
- `clusters_eleitores_resultado`: perfil do eleitor + resultado.

O foco dos clusters deve ser em variaveis discretas:

```text
faixa etaria
sexo/genero
escolaridade
estado civil
raca/cor
partido, quando a analise inclui resultado
```

Nao use biometria como variavel central de cluster.

## Retomar Depois De Crash

Use sempre:

```bash
--resume
```

O pipeline grava progresso em:

```text
dados/banco_eleitoral/logs/
dados/banco_eleitoral/logs/ouro/
```

Se cair, rode o mesmo comando novamente com `--resume`.

Exemplo:

```bash
python3 scripts/run_pipeline_eleitoral_json.py dados/banco_eleitoral \
  --modo analise_banco \
  --banco-out dados/banco_eleitoral \
  --resume \
  --banco-modalidade-analise eleitor_partido \
  --banco-max-municipios-por-uf 20 \
  --banco-skip-clusters \
  --sem-clustering \
  --banco-ouro-workers 1 \
  --banco-duckdb-threads 1 \
  --cenarios 50
```

## Logs

Logs principais:

```text
dados/banco_eleitoral/logs/
dados/banco_eleitoral/logs/ouro/
dados/banco_eleitoral/logs/eventos_pipeline.jsonl
```

O que procurar:

- `Banco eleitoral limpo iniciado`: criacao bronze/prata;
- `Parquet prata gravado`: escrita de uma parte Parquet;
- `Modo analise_banco`: inicio da camada ouro;
- `Ouro municipal`: analise por municipio;
- `Ouro estados+Brasil`: modo curto sem municipal detalhado;
- `DuckDB COPY iniciado`: inicio de uma query Parquet;
- `DuckDB COPY finalizado`: query terminou;
- `Out of Memory`: falta de memoria;
- `Pulando tarefa ouro ja concluida`: `--resume` reaproveitou output.

## Flags Mais Importantes

### Entrada e saida

```text
dados
```

Primeiro argumento. Pode ser `dados/json` no modo banco ou `dados/banco_eleitoral` no modo analise.

```bash
--out nome_do_run
```

Usado pelos modos legados para gravar dentro de `resultados/`.

```bash
--banco-out dados/banco_eleitoral
```

Pasta do banco Parquet.

### Controle de retomada

```bash
--resume
```

Reaproveita o que ja foi processado.

```bash
--banco-overwrite
```

Recria o banco. Use com cuidado, pois pode sobrescrever saidas.

### Criacao do banco

```bash
--banco-workers 8
```

Workers para arquivos pequenos/medios.

```bash
--banco-workers-large-files 8
```

Workers para arquivos grandes.

```bash
--banco-chunk-rows 10000
```

Tamanho dos blocos de escrita Parquet.

```bash
--banco-max-files 10
```

Processa somente alguns arquivos. Bom para teste.

```bash
--banco-apagar-json-apos-processar
```

Apaga o JSON original depois que ele foi gravado em Parquet com sucesso. Use somente se voce tiver certeza de que pode apagar a entrada.

### Camada ouro

```bash
--banco-modalidade-analise completa
--banco-modalidade-analise estados_brasil
--banco-modalidade-analise eleitor
--banco-modalidade-analise candidato
--banco-modalidade-analise eleitor_partido
--banco-modalidade-analise eleitor_candidato_partido
```

Escolhe o tipo de analise.

```bash
--banco-somente-estados-brasil
```

Pula municipio detalhado e gera UF + Brasil.

```bash
--banco-ufs SP,RJ,MG
```

Limita a analise a UFs especificas.

```bash
--banco-max-municipios-por-uf 20
```

Limita quantidade de municipios por UF. Bom para teste do front.

```bash
--banco-skip-heavy-analyses
```

Pula partes mais pesadas, principalmente candidato por perfil em alguns fluxos.

```bash
--banco-skip-clusters
--sem-clustering
```

Pula clusters.

```bash
--banco-ouro-workers 1
--banco-duckdb-threads 1
```

Configuracao segura de memoria. Mais lenta, mas reduz risco de OOM.

```bash
--banco-ouro-paralelo-agressivo
```

Forca mais paralelismo na ouro. Pode ser mais rapido, mas aumenta risco de estourar memoria.

### Simulacao

```bash
--predict-2026
--cenarios 3000
--monte-carlo-sigma 0.035
--prediction-entity auto
--prediction-cargo-filter ""
```

`--cenarios` controla quantas simulacoes serao feitas.

## Como Escolher O Comando Certo

### Quero criar o banco do zero

```bash
python3 scripts/run_pipeline_eleitoral_json.py dados/json \
  --modo banco \
  --banco-out dados/banco_eleitoral \
  --resume \
  --banco-workers 8 \
  --banco-workers-large-files 8 \
  --banco-chunk-rows 10000
```

### Quero testar o dashboard rapidamente

```bash
python3 scripts/run_pipeline_eleitoral_json.py dados/banco_eleitoral \
  --modo analise_banco \
  --banco-out dados/banco_eleitoral \
  --resume \
  --banco-modalidade-analise estados_brasil \
  --banco-somente-estados-brasil \
  --banco-skip-heavy-analyses \
  --banco-skip-clusters \
  --sem-clustering \
  --banco-ouro-workers 1 \
  --banco-duckdb-threads 1 \
  --cenarios 100
```

### Quero analise rapida municipal de eleitor + partido

```bash
python3 scripts/run_pipeline_eleitoral_json.py dados/banco_eleitoral \
  --modo analise_banco \
  --banco-out dados/banco_eleitoral \
  --resume \
  --banco-modalidade-analise eleitor_partido \
  --banco-max-municipios-por-uf 20 \
  --banco-skip-clusters \
  --sem-clustering \
  --banco-ouro-workers 1 \
  --banco-duckdb-threads 1 \
  --cenarios 50
```

### Quero analise rapida de candidato

```bash
python3 scripts/run_pipeline_eleitoral_json.py dados/banco_eleitoral \
  --modo analise_banco \
  --banco-out dados/banco_eleitoral \
  --resume \
  --banco-modalidade-analise candidato \
  --banco-max-municipios-por-uf 20 \
  --banco-skip-clusters \
  --sem-clustering \
  --banco-ouro-workers 1 \
  --banco-duckdb-threads 1 \
  --cenarios 50
```

### Quero tudo, aceitando que vai demorar

```bash
python3 scripts/run_pipeline_eleitoral_json.py dados/banco_eleitoral \
  --modo analise_banco \
  --banco-out dados/banco_eleitoral \
  --resume \
  --banco-modalidade-analise completa \
  --cluster-min-k 2 \
  --cluster-max-k 10 \
  --banco-ouro-workers 1 \
  --banco-duckdb-threads 1 \
  --cenarios 3000
```

## Problemas Comuns

### O dashboard abre, mas aparece vazio

Possiveis causas:

- a camada `ouro/brasil` ainda nao existe;
- a camada `ouro/estadual` ainda nao existe;
- voce rodou so municipio parcial e nao fechou Brasil;
- a API esta apontando para a pasta errada.

Solucao rapida:

```bash
python3 scripts/run_pipeline_eleitoral_json.py dados/banco_eleitoral \
  --modo analise_banco \
  --banco-out dados/banco_eleitoral \
  --resume \
  --banco-modalidade-analise estados_brasil \
  --banco-somente-estados-brasil \
  --banco-skip-heavy-analyses \
  --banco-skip-clusters \
  --sem-clustering \
  --banco-ouro-workers 1 \
  --banco-duckdb-threads 1 \
  --cenarios 100
```

### Deu `Out of Memory`

Use configuracao segura:

```bash
--banco-ouro-workers 1
--banco-duckdb-threads 1
--banco-skip-clusters
--sem-clustering
```

Depois rode de novo com:

```bash
--resume
```

### Esta muito lento

Para testar front, nao rode completo. Use:

```bash
--banco-modalidade-analise estados_brasil
--banco-somente-estados-brasil
```

Ou limite municipios:

```bash
--banco-max-municipios-por-uf 20
```

### Quero processar todos os estados sem municipal

Nao passe `--banco-ufs`. Assim ele pega todas as UFs detectadas.

### Quero processar so alguns estados

Use:

```bash
--banco-ufs SP,RJ,MG
```

### Posso apagar os JSONs depois do banco?

Depois que `bronze` e `prata` estiverem criadas e conferidas, tecnicamente as analises podem usar o banco Parquet sem os JSONs. Mas apague os JSONs somente se voce tiver certeza de que nao precisara reprocessar a origem.

Se quiser automatizar no modo banco:

```bash
--banco-apagar-json-apos-processar
```

## Limites Metodologicos

- O projeto trabalha com dados agregados.
- A relacao eleitorado x voto nao prova voto individual.
- Perfil por partido/candidato e uma aproximacao territorial.
- Clusters agrupam perfis predominantes, nao pessoas reais.
- Simulacao 2026 e cenario estatistico, nao previsao garantida.

## Resumo Mental Da Arquitetura

Pense assim:

```text
JSON bruto
  -> bronze: preserva e audita
  -> prata: limpa e padroniza
  -> ouro municipal: analises por municipio
  -> ouro estadual: fecha cada UF
  -> ouro Brasil: fecha a federacao
  -> API/Dashboard/PDF: consulta so Parquet tratado
```

Para teste rapido, pule o municipal:

```text
prata
  -> ouro estadual
  -> ouro Brasil
  -> dashboard
```

Para analise completa, rode:

```text
prata
  -> municipal completo
  -> estadual
  -> Brasil
  -> clusters
  -> simulacao
  -> dashboard/PDF
```
