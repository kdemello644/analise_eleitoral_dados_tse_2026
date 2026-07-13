# Analise Eleitoral Brasileira com Dados Abertos do TSE

Este projeto organiza, limpa, correlaciona, analisa e simula dados eleitorais brasileiros a partir dos arquivos publicos disponibilizados pelo Tribunal Superior Eleitoral (TSE). O objetivo e criar uma base persistente em Parquet, estruturada em camadas `bronze`, `prata` e `ouro`, para permitir analises eleitorais por Brasil, UF, municipio, zona, secao eleitoral, partido, candidato e perfil do eleitorado.

A fonte declarada dos dados e o Portal de Dados Abertos do TSE:

```text
https://dadosabertos.tse.jus.br/
```

Segundo o proprio portal, ele disponibiliza dados gerados ou custodiados pelo TSE para acesso, tratamento e compartilhamento pela sociedade. O portal substituiu o antigo Repositorio de Dados Eleitorais, descontinuado em janeiro de 2022.

## Objetivo do Projeto

O projeto foi pensado para responder perguntas eleitorais usando dados agregados oficiais:

- quem sao os eleitores por ano, UF, municipio e secao;
- quais perfis eleitorais predominam em cada territorio;
- qual partido ou candidato ganhou em cada secao eleitoral;
- como a vitoria aconteceu, usando votos, share, comparecimento, abstencao e distribuicao territorial;
- qual perfil de eleitorado esta associado a partidos e candidatos, sempre por proxy territorial;
- como o perfil do eleitorado evolui entre anos eleitorais;
- quais clusters de eleitores aparecem por estado, municipio e Brasil;
- qual cenario de eleitores e votos por partido pode ser simulado para 2026;
- como consultar os resultados em dashboard, cards, graficos e mapas.

Importante: os arquivos do TSE usados aqui sao dados agregados. O projeto nao identifica pessoas individualmente, nao prova voto individual e nao afirma motivacao psicologica do eleitor. Quando correlaciona perfil de eleitorado com votos, ele usa aproximacao ecologica por territorio, secao, municipio e ano.

## Visao Geral da Esteira de Dados

A esteira completa do projeto segue este fluxo:

```text
Dados Abertos do TSE
  -> downloads oficiais em ZIP
  -> extracao dos CSVs oficiais
  -> conversao dos CSVs para JSON/JSONL
  -> leitura streaming dos JSON/JSONL
  -> banco Parquet em camadas bronze/prata/ouro
  -> analises, clusters, dashboards e simulacao 2026
```

O pipeline principal deste repositorio trabalha a partir dos JSON/JSONL ja preparados. A etapa anterior, de preparacao, consiste em baixar os ZIPs do TSE, extrair os CSVs e converter esses CSVs para JSON ou JSONL. Depois disso, o projeto transforma os JSONs em Parquet e constroi o banco eleitoral.

## Origem dos Dados

Os dados sao provenientes dos conjuntos publicos do TSE, principalmente relacionados a:

- eleitorado;
- perfil do eleitorado;
- locais de votacao;
- resultados de votacao por secao, municipio, zona e cargo;
- votacao por candidato;
- votacao por partido;
- candidatos;
- vagas;
- coligacoes;
- situacao ou motivo de cassacao quando disponivel;
- transferencias temporarias e outras tabelas eleitorais presentes na base.

Os nomes dos arquivos costumam conter o ano eleitoral, por exemplo `2014`, `2018`, `2022` ou `2024`. O pipeline extrai esse ano do nome do arquivo quando o campo de ano nao esta explicitamente padronizado nos dados.

## Metodologia de Extracao

A metodologia de extracao foi desenhada para lidar com arquivos muito grandes e formatos heterogeneos.

1. Os arquivos originais sao obtidos no Portal de Dados Abertos do TSE, geralmente em arquivos ZIP.
2. Cada ZIP e extraido para recuperar os CSVs oficiais.
3. Os CSVs sao convertidos para JSON/JSONL, preservando os campos originais.
4. O pipeline le os JSON/JSONL sem carregar o arquivo inteiro na memoria sempre que possivel.
5. Os registros sao normalizados para colunas canonicas.
6. Os dados sao gravados em Parquet, particionados por dominio, UF, ano e codigos eleitorais.
7. Depois da base limpa criada, as analises passam a usar os Parquets tratados, nao os JSONs brutos.

Essa estrategia reduz o risco de estourar memoria em arquivos de dezenas de gigabytes.

## Metodologia de Organizacao dos Dados

O projeto separa os dados por dominio analitico:

```text
eleitorado
candidatos
resultados_votos
outros/metadados
```

Cada dominio recebe colunas canonicas para facilitar correlacoes.

### Chaves principais

As principais chaves de correlacao sao:

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

Para correlacionar eleitorado e resultados, as chaves mais importantes sao:

```text
ano + uf + cd_municipio + zona + secao + cargo + turno
```

Quando a analise precisa ser mais leve, ela usa niveis agregados:

```text
Brasil
UF
municipio
zona
secao
```

## Arquitetura Bronze, Prata e Ouro

O caminho recomendado do projeto e o modo banco:

```text
dados/
  json/
    arquivos JSON/JSONL preparados a partir dos CSVs do TSE
  banco_eleitoral/
    bronze/
    prata/
    ouro/
    preditivo_2026/
    logs/
    metadados/
```

### Camada Bronze

A camada `bronze` guarda os dados normalizados em Parquet, ainda proximos ao formato original.

Ela organiza arquivos por:

```text
dominio
schema_id
uf
shard
```

Exemplo:

```text
dados/banco_eleitoral/bronze/
  eleitorado/schema_id=<hash>/uf=SP/shard=<arquivo>/part-000000.parquet
```

Objetivo da bronze:

- preservar o maximo possivel do dado original;
- agrupar documentos com campos parecidos;
- extrair `ano` do nome do arquivo quando necessario;
- permitir auditoria do que foi lido;
- criar uma base intermediaria em Parquet.

### Camada Prata

A camada `prata` limpa e padroniza os dados por dominio.

Exemplo:

```text
dados/banco_eleitoral/prata/
  eleitorado/uf=SP/shard=<arquivo>/part-000000.parquet
  candidatos/uf=SP/shard=<arquivo>/part-000000.parquet
  resultados_votos/uf=SP/shard=<arquivo>/part-000000.parquet
```

Objetivo da prata:

- reduzir colunas irrelevantes;
- manter codigos de correlacao;
- manter variaveis discretas importantes;
- padronizar nomes de municipio, UF, zona e secao;
- manter metricas eleitorais essenciais;
- servir como base para a camada ouro.

Campos numericos continuos nao essenciais sao evitados na analise principal. As metricas numericas mantidas com prioridade sao:

```text
eleitorado
votos
comparecimento_estimado
abstencao_estimado
brancos
nulos
validos_estimados
```

### Camada Ouro

A camada `ouro` contem dados prontos para analise, dashboard, graficos e simulacao.

Ela e gerada a partir da prata e evita arquivos monoliticos enormes. As tabelas grandes sao datasets Parquet particionados.

Exemplo:

```text
dados/banco_eleitoral/ouro/
  base_gold_global/
  resultados_vencedores_secao/
  resultado_eleitorado_por_secao/
  perfil_eleitor_por_ano/
  perfil_eleitor_por_partido/
  perfil_eleitor_por_candidato/
  perfil_candidatos/
  timeline_nacional.parquet
  timeline_uf/
  timeline_municipal/
  top10_perfis_federacao_estado_municipio/
```

Para poupar memoria, a camada ouro trabalha em fatias pequenas:

```text
UF + ano
```

E, quando necessario, particiona tambem por codigos:

```text
ano / uf / cd_municipio / zona / secao
```

Isso evita varrer todos os dados de uma vez e permite retomar o processamento com `--resume`.

## Metodologia de Limpeza

A limpeza segue estes principios:

- preservar codigos eleitorais usados em correlacoes;
- remover ou ignorar campos sem utilidade analitica imediata;
- padronizar campos de UF, municipio, zona, secao, cargo e turno;
- extrair ano do nome do arquivo quando necessario;
- converter valores numericos de votos e eleitorado;
- tratar valores nulos, vazios, `#NULO#`, `nan`, `None` e equivalentes;
- priorizar variaveis discretas com mais de uma categoria;
- evitar plotar variaveis com categoria unica;
- evitar graficos vazios, nulos ou repetidos.

Variaveis discretas priorizadas:

```text
sexo/genero
faixa etaria
escolaridade/grau de instrucao
estado civil
raca/cor quando disponivel
partido
candidato
cargo
turno
UF
municipio
zona
secao
```

A biometria nao e prioridade para clustering nem para graficos principais, porque nao ajuda a descrever comportamento eleitoral no objetivo atual do projeto.

## Metodologia de Correlacao

A correlacao e feita por codigos, nao por texto livre.

O projeto cruza:

```text
eleitorado + resultados + candidatos
```

As correlacoes principais usam:

```text
ano
uf
cd_municipio
zona
secao
cargo
turno
```

O objetivo e responder:

- qual era o perfil predominante do eleitorado em uma secao;
- qual partido ou candidato recebeu votos naquela secao;
- quem venceu em cada secao;
- como o perfil do eleitorado varia por ano;
- como votos e perfis se associam por territorio;
- quais perfis aparecem associados a partidos;
- quais perfis aparecem associados a candidatos;
- quais padroes se repetem entre anos.

Essa correlacao nao afirma que uma pessoa especifica votou em um partido. Ela estima associacoes por agregacao territorial.

## Metodologia de Analise Individual

O caminho legado do pipeline ainda permite analisar documento por documento.

Etapas:

1. Ler cada JSON/JSONL.
2. Identificar campos, tipos e quantidade de registros.
3. Extrair ano a partir do nome ou conteudo.
4. Gerar amostra e resumo.
5. Criar gold individual.
6. Produzir analises por ano.
7. Gerar saidas por arquivo, preservando nome original.

Saidas tipicas:

```text
resultados/<run>/individual/<arquivo_original>/
  tabelas/
  parquet/
  anos/<ano>/
```

Essa etapa e util para entender cada documento isoladamente. Para bases muito grandes, o modo banco e preferido.

## Metodologia de Analise Global

A analise global usa as tabelas consolidadas, preferindo Parquet.

Ela produz:

- timeline nacional;
- timeline por UF;
- timeline por municipio;
- retrato municipal;
- vencedor por secao;
- perfil do eleitor por ano;
- top 10 perfis por Brasil, UF e municipio;
- perfil do eleitor associado a partidos;
- perfil do eleitor associado a candidatos;
- perfil dos candidatos;
- correlacao entre candidato e eleitorado;
- dados para graficos e dashboards.

Niveis de analise:

```text
Brasil
UF
municipio
zona
secao eleitoral
```

Na camada ouro, os resultados ficam em:

```text
dados/banco_eleitoral/ouro/
```

## Metodologia de Analise de Resultados

A analise de resultados considera:

- votos por partido;
- votos por candidato;
- votos brancos;
- votos nulos;
- votos validos estimados;
- comparecimento;
- abstencao;
- cargo;
- turno;
- vencedor por secao.

Para cada secao, o pipeline calcula:

```text
partido_vencedor
candidato_vencedor
votos_vencedor
votos_total_secao
share_vencedor
```

Esses resultados sao correlacionados com os perfis predominantes do eleitorado no mesmo recorte territorial.

## Metodologia de Perfil do Eleitor

O perfil do eleitor e descrito com variaveis discretas:

```text
faixa etaria
sexo/genero
escolaridade
estado civil
raca/cor quando disponivel
UF
municipio
secao
```

O projeto busca responder:

- quem e o eleitor medio no Brasil;
- quem e o eleitor medio por estado;
- quem e o eleitor medio por municipio;
- quais perfis predominam por ano;
- quais perfis crescem ou diminuem entre anos;
- quais perfis aparecem associados a partidos e candidatos.

## Metodologia de Clustering

Os clusters sao usados para agrupar perfis eleitorais parecidos.

Principios:

- foco em dados discretos;
- evitar dados continuos desnecessarios;
- nao usar biometria como variavel principal;
- manter sexo/genero, escolaridade, estado civil e faixa etaria;
- usar valores categorizados e codificados;
- descrever o cluster como uma pessoa/perfil, nao apenas como numero.

Algoritmo principal:

```text
KMeans
```

Selecao de quantidade de clusters:

```text
tecnica do cotovelo
```

O projeto pode gerar dois tipos de cluster:

1. Cluster focado somente no eleitorado.
2. Cluster combinando eleitorado com resultados eleitorais.

Cada cluster deve ser descrito em linguagem analitica, por exemplo:

```text
Cluster 2: eleitorado predominante feminino, ensino medio completo,
faixa de 35 a 44 anos, solteiro, concentrado no Sudeste,
com maior associacao historica a determinados partidos.
```

## Metodologia de Simulacao 2026

A simulacao de 2026 usa a camada ouro como base de evidencia.

Ela estima cenarios por:

```text
Brasil
UF
municipio
secao quando disponivel
partido
cargo
turno
```

Para a simulacao partidaria de 2026, o foco e partido, nao candidato.

O pipeline estima:

- percentual possivel de votos por partido no Brasil;
- percentual possivel de votos por partido em cada UF;
- percentual possivel de votos por partido em cada municipio;
- perfil de eleitor associado a cada partido;
- tendencia historica por anos analisados;
- justificativa de correlacao historica quando existe serie comparavel.

Metodologias usadas:

- historico de share de votos;
- variacao temporal entre anos;
- swing historico;
- cenarios deterministios;
- Monte Carlo;
- intervalos de incerteza;
- comparacao entre anos eleitorais.

Saidas principais:

```text
dados/banco_eleitoral/preditivo_2026/
  parquet/
  tabelas/
  plots/
  explicabilidade/
```

## Metodologia de Dashboard

O projeto possui dashboards para consulta dos resultados.

Dashboard Dash:

```bash
python3 scripts/dashboard_dash_eleitoral.py \
  --run dados/banco_eleitoral \
  --host 0.0.0.0 \
  --port 8050
```

Acesso:

```text
http://localhost:8050
```

Dashboard Streamlit:

```bash
streamlit run scripts/dashboard_streamlit_eleitoral.py -- --run dados/banco_eleitoral
```

Os dashboards consultam os Parquets tratados diretamente, especialmente as tabelas da camada ouro. A ideia e evitar tabelas gigantes na tela e priorizar:

- cards;
- graficos;
- filtros por municipio;
- filtros por UF;
- clusters descritos como perfis;
- mapas interativos;
- consulta resumida de tabelas quando necessario.

## Como Executar

O comando principal sempre tem esta forma:

```bash
python3 scripts/run_pipeline_eleitoral_json.py <entrada> [opcoes]
```

O argumento `<entrada>` muda conforme o modo:

- para criar banco a partir dos JSONs: use `dados/json`;
- para analisar banco ja criado: use `dados/banco_eleitoral`;
- para o pipeline legado por documentos: use a pasta que contem os JSONs originais.

### Modos de execucao

O parametro `--modo` define qual parte do pipeline roda.

| Modo | O que faz | Quando usar |
|---|---|---|
| `inventario` | Lista arquivos, campos e estrutura basica dos JSONs. | Diagnostico inicial. |
| `individual` | Analisa cada JSON/JSONL separadamente. | Entender documento por documento. |
| `global` | Consolida saidas individuais ja geradas. | Caminho legado, depois de `individual`. |
| `preditivo` | Roda somente a simulacao usando base global existente. | Quando a base global ja existe. |
| `completo` | Roda inventario, individual, global e simulacao no caminho legado. | Bases menores ou testes. |
| `banco` | Cria banco `bronze` e `prata` em Parquet. | Caminho recomendado para arquivos grandes. |
| `analise_banco` | Gera camada `ouro`, analises, simulacao e artefatos a partir do banco. | Depois de criar `dados/banco_eleitoral`. |

### Principais campos do comando

| Campo | Significado |
|---|---|
| `dados/json` | Pasta de entrada com JSON/JSONL/NDJSON preparados a partir dos CSVs do TSE. |
| `dados/banco_eleitoral` | Pasta do banco persistente com bronze, prata, ouro, logs e simulacao. |
| `--out <nome>` | Nome da pasta dentro de `resultados/` no pipeline legado. Ex.: `--out teste` cria `resultados/teste`. |
| `--banco-out <pasta>` | Pasta onde o banco bronze/prata/ouro sera criado ou consultado. |
| `--resume` | Continua de onde parou, pulando tarefas ja concluidas quando possivel. |
| `--banco-overwrite` | Recria o banco do zero. Use com cuidado. |
| `--predict-2026` | Ativa a simulacao de cenarios para 2026. |
| `--cenarios` | Quantidade de simulacoes/cenarios Monte Carlo. |
| `--cluster-min-k` | Menor quantidade de clusters testada. |
| `--cluster-max-k` | Maior quantidade de clusters testada. |
| `--log-level` | Nivel dos logs. Ex.: `INFO`, `DEBUG`, `WARNING`. |

### Campos de memoria e desempenho

| Campo | Significado |
|---|---|
| `--banco-workers` | Workers internos para arquivos pequenos/medios no modo banco. `0` usa auto-tuning. |
| `--banco-workers-large-files` | Workers internos para arquivos grandes no modo banco. `0` usa auto-tuning. |
| `--banco-chunk-rows` | Linhas por lote/parte Parquet no banco. Menor usa menos memoria. |
| `--banco-large-file-threshold-gb` | Tamanho a partir do qual um arquivo e tratado como grande. |
| `--banco-ouro-workers` | Quantas queries independentes da camada ouro rodam em paralelo. Para evitar OOM, use `1`. |
| `--banco-duckdb-threads` | Threads usadas pelo DuckDB dentro da tarefa atual. |
| `--banco-usar-todos-workers` | Usa todos os CPUs logicos por arquivo. Mais rapido, mas mais agressivo. |
| `--banco-ouro-paralelo-agressivo` | Permite paralelismo pesado na ouro. Pode estourar RAM. |
| `--banco-skip-heavy-analyses` | Pula analises ouro mais pesadas, como perfil por candidato. |
| `--banco-apagar-json-apos-processar` | Apaga JSON original somente depois de gravar Parquet com sucesso. Use com cuidado. |

### Campos do caminho legado

| Campo | Significado |
|---|---|
| `--sample-mode head` | Usa as primeiras linhas como amostra. Mais rapido. |
| `--sample-mode reservoir` | Amostra distribuida ao longo do arquivo. Mais representativo. |
| `--sample-frac` | Fracao aproximada de amostragem. |
| `--max-sample-rows` | Maximo de linhas na amostra. |
| `--min-sample-rows` | Minimo de linhas na amostra quando houver dados. |
| `--full-aggregations` | Forca agregacoes completas em vez de apenas amostras. |
| `--aggregate-chunk-rows` | Tamanho dos blocos de agregacao completa. |
| `--analysis-max-rows` | Limite de linhas para analises em memoria/HTML. |
| `--global-max-gold-rows` | Limite opcional para carregar gold global em memoria. `0` tenta usar tudo. |
| `--gold-csv-max-rows` | Acima deste limite, o CSV vira preview e o completo fica em Parquet. |
| `--workers-individual` | Paralelismo na analise individual. |
| `--workers-large-files` | Paralelismo para arquivos grandes no caminho legado. |
| `--workers-parquet` | Paralelismo de escrita/particionamento Parquet. |
| `--large-file-threshold-gb` | Limiar para classificar arquivo grande no caminho legado. |
| `--parquet` | Mantem saidas Parquet ativas. Padrao ligado. |
| `--sem-parquet` | Desativa saidas Parquet no caminho legado. |
| `--top-n-html` | Quantidade de itens exibidos em tabelas/cards HTML. |
| `--top-n-plots` | Quantidade de itens nos graficos. |

### Campos de engine

| Campo | Significado |
|---|---|
| `--engine pandas` | Usa pandas/Parquet. Caminho padrao. |
| `--engine pyspark` | Tenta usar PySpark quando instalado. |
| `--engine auto` | Tenta PySpark quando disponivel e cai para pandas se nao estiver. |
| `--spark-master` | Master do Spark. Ex.: `local[*]`. |

### Campos de simulacao

| Campo | Significado |
|---|---|
| `--monte-carlo-sigma` | Desvio usado como ruido base na simulacao Monte Carlo. |
| `--prediction-entity` | Entidade prevista: `auto`, `partido`, `candidato` ou coluna compativel. |
| `--prediction-cargo-filter` | Filtro textual de cargo para a simulacao. |

### Campos do PDF

O PDF e gerado por um script separado, porque pode ter centenas ou milhares de paginas.

| Campo | Significado |
|---|---|
| `--run` | Pasta do banco/run que sera consultada. Padrao: `dados/banco_eleitoral`. |
| `--out` | Caminho do PDF final. Padrao: `<run>/relatorios/relatorio_completo_eleitoral.pdf`. |
| `--max-pages` | Limite maximo de paginas. Ex.: `1000`. |
| `--top-n` | Quantidade de itens por ranking/grafico. |
| `--ufs` | Lista de UFs separadas por virgula. Vazio usa todas detectadas. |
| `--municipios-por-uf` | Quantos municipios detalhar por UF. |
| `--incluir-secoes` | Inclui paginas com amostra de secoes eleitorais. |
| `--secoes-por-uf` | Quantas secoes listar por UF quando `--incluir-secoes`. |
| `--duckdb-threads` | Threads DuckDB usadas nas consultas do PDF. |

### Criar banco bronze e prata

Entrada esperada:

```text
dados/json/
```

Comando:

```bash
python3 scripts/run_pipeline_eleitoral_json.py dados/json \
  --modo banco \
  --banco-out dados/banco_eleitoral \
  --resume \
  --banco-workers 4 \
  --banco-workers-large-files 4 \
  --banco-chunk-rows 10000
```

Para apagar JSONs ja processados com sucesso:

```bash
python3 scripts/run_pipeline_eleitoral_json.py dados/json \
  --modo banco \
  --banco-out dados/banco_eleitoral \
  --resume \
  --banco-workers 4 \
  --banco-workers-large-files 4 \
  --banco-chunk-rows 10000 \
  --banco-apagar-json-apos-processar
```

### Rodar camada ouro, analise e simulacao

```bash
python3 scripts/run_pipeline_eleitoral_json.py dados/banco_eleitoral \
  --modo analise_banco \
  --banco-out dados/banco_eleitoral \
  --resume \
  --predict-2026 \
  --cenarios 3000 \
  --cluster-min-k 2 \
  --cluster-max-k 10 \
  --banco-ouro-workers 1 \
  --banco-duckdb-threads 8
```

Use `--resume` para continuar de onde parou.

## Onde Ver os Resultados

Camada ouro:

```text
dados/banco_eleitoral/ouro/
```

Simulacao:

```text
dados/banco_eleitoral/preditivo_2026/
```

Logs:

```text
dados/banco_eleitoral/logs/
```

Metadados:

```text
dados/banco_eleitoral/metadados/
```

Parquets corrompidos, quando encontrados:

```text
dados/banco_eleitoral/metadados/parquets_corrompidos/
```

Relatorios PDF:

```text
dados/banco_eleitoral/relatorios/
```

## Gerar Relatorio PDF Completo

Depois que a camada ouro existir, gere o PDF completo assim:

```bash
python3 scripts/gerar_relatorio_pdf_eleitoral.py \
  --run dados/banco_eleitoral \
  --out dados/banco_eleitoral/relatorios/relatorio_completo_eleitoral.pdf \
  --max-pages 1000 \
  --top-n 20 \
  --municipios-por-uf 30 \
  --incluir-secoes \
  --secoes-por-uf 80 \
  --duckdb-threads 4
```

Para um PDF menor de verificacao:

```bash
python3 scripts/gerar_relatorio_pdf_eleitoral.py \
  --run dados/banco_eleitoral \
  --out dados/banco_eleitoral/relatorios/relatorio_teste.pdf \
  --max-pages 30 \
  --top-n 10 \
  --municipios-por-uf 2 \
  --duckdb-threads 2
```

## Monitoramento

Durante a execucao:

```bash
du -sh dados/banco_eleitoral/ouro
```

```bash
find dados/banco_eleitoral/ouro -name "*.parquet" | wc -l
```

Ver logs da ouro:

```bash
ls dados/banco_eleitoral/logs/ouro
```

## Robustez e Memoria

O projeto foi ajustado para bases grandes:

- evita carregar JSON inteiro em memoria;
- usa Parquet como armazenamento intermediario;
- processa arquivos grandes por partes;
- processa camada ouro em fatias `UF + ano`;
- particiona dados por codigos eleitorais;
- grava Parquets antes de seguir para a proxima fatia;
- permite `--resume`;
- move Parquets corrompidos para quarentena quando detectados;
- evita multiplas queries pesadas simultaneas na camada ouro.

Recomendacao para maquinas comuns:

```bash
--banco-ouro-workers 1
--banco-duckdb-threads 8
```

Isso roda uma tarefa pesada por vez, usando varias threads dentro da tarefa atual.

## Limitacoes Metodologicas

Este projeto nao mede voto individual. Ele trabalha com dados agregados por territorio e secao.

Limitacoes:

- associacao entre perfil e voto e ecologica;
- resultados dependem da qualidade dos arquivos oficiais baixados;
- arquivos incompletos ou corrompidos precisam ser reprocessados;
- mudancas de layout do TSE podem exigir novos mapeamentos;
- categorias ausentes ou muito agregadas reduzem poder explicativo;
- simulacao 2026 nao e previsao deterministica, e sim cenario estatistico baseado em historico.

## Estrutura Principal do Codigo

```text
scripts/
  run_pipeline_eleitoral_json.py
  pipeline_eleitoral_json/
    main.py
    config.py
    clean_database.py
    json_reader.py
    stage_individual.py
    stage_global.py
    stage_simulation.py
    aggregation.py
    global_correlation.py
    global_cluster_analysis.py
    electoral_analysis.py
    comportamento_eleitoral.py
    simulation.py
    explainability.py
    plots.py
    utils.py
  dashboard_dash_eleitoral.py
  dashboard_streamlit_eleitoral.py
```

## Dependencias

Instalacao recomendada:

```bash
python3 -m pip install -r scripts/pipeline_eleitoral_json/requirements.txt
```

Principais bibliotecas:

- pandas;
- pyarrow;
- duckdb;
- numpy;
- scikit-learn;
- plotly;
- dash;
- streamlit.

## Licenca e Uso dos Dados

O codigo organiza e analisa dados publicos. Os dados originais pertencem ao ecossistema de dados abertos do TSE e devem ser citados conforme a fonte oficial:

```text
Tribunal Superior Eleitoral - Portal de Dados Abertos do TSE
https://dadosabertos.tse.jus.br/
```

Ao divulgar resultados, informe que as conclusoes sao derivadas de dados agregados e que correlacoes territoriais nao representam prova de comportamento individual.
