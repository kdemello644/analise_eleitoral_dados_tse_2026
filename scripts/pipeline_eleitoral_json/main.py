from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
import html
import json
import logging
import sys

from .config import parse_args
from .clean_database import CleanDatabaseConfig, build_clean_database, run_clean_database_analyses
from .json_reader import find_json_files
from .stage_global import run_global_stage
from .stage_individual import run_individual_stage
from .stage_simulation import run_simulation_stage
from .reporting import generate_executive_report, update_global_dashboard_with_simulation
from .utils import save_html, save_json, setup_logging


def load_previous_results(out: Path) -> list[dict[str, Any]]:
    p = out / "logs" / "resultados_individuais.json"
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []


def load_previous_global_info(out: Path) -> dict[str, Any]:
    p = out / "logs" / "global_info.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass

    return {
        "global_gold_csv": str(out / "global" / "tabelas" / "base_gold_global.csv"),
        "global_gold_parquet": str(out / "global" / "parquet" / "base_gold_global.parquet"),
        "municipal_csv": str(out / "global" / "tabelas" / "retrato_municipal_global.csv"),
        "municipal_parquet": str(out / "global" / "parquet" / "retrato_municipal_global.parquet"),
        "inventario_temporal_csv": str(out / "global" / "timeline" / "inventario_temporal_arquivos.csv"),
        "matriz_arquivo_ano_csv": str(out / "global" / "timeline" / "matriz_arquivo_ano.csv"),
        "timeline_nacional_csv": str(out / "global" / "timeline" / "timeline_nacional.csv"),
        "timeline_uf_csv": str(out / "global" / "timeline" / "timeline_uf.csv"),
        "timeline_municipal_csv": str(out / "global" / "timeline" / "timeline_municipal.csv"),
        "timeline_entidades_csv": str(out / "global" / "timeline" / "timeline_entidades.csv"),
        "evolucao_municipal_csv": str(out / "global" / "timeline" / "evolucao_municipal.csv"),
        "correlacoes_temporais_csv": str(out / "global" / "correlacoes" / "correlacoes_temporais_municipais.csv"),
        "correlacoes_entidades_csv": str(out / "global" / "correlacoes" / "correlacoes_share_entidades_entre_anos.csv"),
        "similaridade_tabelas_csv": str(out / "global" / "schema" / "similaridade_entre_tabelas.csv"),
        "similaridade_campos_csv": str(out / "global" / "schema" / "similaridade_entre_campos.csv"),
        "mapa_canonico_csv": str(out / "global" / "schema" / "mapa_canonico_campos_aprendido.csv"),
        "correlacao_codigos_outputs": {
            "correlacao_codigos_dir": str(out / "global" / "correlacao_codigos"),
            "manifesto_parquets_correlacionados_csv": str(out / "global" / "correlacao_codigos" / "tabelas" / "manifesto_parquets_correlacionados_por_ano.csv"),
            "estatisticas_correlacionadas_por_ano_csv": str(out / "global" / "correlacao_codigos" / "tabelas" / "estatisticas_correlacionadas_por_ano.csv"),
            "dicionario_correlacao_codigos_csv": str(out / "global" / "correlacao_codigos" / "tabelas" / "dicionario_correlacao_codigos.csv"),
            "parquet_por_ano_dir": str(out / "global" / "correlacao_codigos" / "parquet" / "por_ano"),
        },
        "html": str(out / "global" / "relatorio_global.html"),
    }


def main() -> None:
    cfg = parse_args()

    if cfg.modo in {"banco", "analise_banco"}:
        run_clean_database_mode(cfg)
        return

    cfg.out.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging(cfg.out, cfg.log_level)
    save_json(asdict(cfg), cfg.out / "logs" / "config.json")

    if not cfg.dados.exists() or not cfg.dados.is_dir():
        print(f"[ERRO] Pasta de dados nao existe: {cfg.dados}")
        sys.exit(1)

    logging.info("Pipeline eleitoral JSON iniciado.")
    logging.info("Entrada: %s", cfg.dados)
    logging.info("Saida: %s", cfg.out)
    logging.info("Modo: %s", cfg.modo)

    json_files = find_json_files(cfg.dados, include_metadata=cfg.incluir_metadados_json)
    logging.info("Arquivos JSON/JSONL/NDJSON de dados encontrados: %s", len(json_files))

    if not json_files:
        print(f"[ERRO] Nenhum JSON/JSONL/NDJSON de dados encontrado em: {cfg.dados}")
        sys.exit(1)

    results: list[dict[str, Any]] = []
    if cfg.modo in {"individual", "global", "preditivo"}:
        results = load_previous_results(cfg.out)

    if cfg.modo in {"inventario", "individual", "completo"}:
        results = run_individual_stage(json_files, cfg)

    if cfg.modo == "inventario":
        global_info = run_global_stage(results, cfg)
        pred_info: dict[str, Any] = {}
    else:
        global_info: dict[str, Any] = {}
        if cfg.modo in {"global", "completo"}:
            if not results:
                results = load_previous_results(cfg.out)
            global_info = run_global_stage(results, cfg)
        elif cfg.modo == "preditivo":
            global_info = load_previous_global_info(cfg.out)

        pred_info = {}
        if cfg.predict_2026 or cfg.modo in {"preditivo", "completo"}:
            if not global_info:
                global_info = load_previous_global_info(cfg.out)
            pred_info = run_simulation_stage(global_info, cfg)
            if pred_info.get("status") == "ok":
                pred_info["global_dashboard_atualizado"] = update_global_dashboard_with_simulation(cfg, pred_info)
                pred_info["relatorio_executivo"] = generate_executive_report(cfg, global_info, pred_info)

    ok = sum(1 for r in results if r.get("status") == "ok")
    error = sum(1 for r in results if r.get("status") != "ok")
    streamlit_cmd = f'streamlit run scripts/dashboard_streamlit_eleitoral.py -- --run "{cfg.out}"'
    dash_cmd = f'python scripts/dashboard_dash_eleitoral.py --run "{cfg.out}" --host 127.0.0.1 --port 8050'

    body = f"""
<h2>Pipeline finalizado</h2>
<p><strong>Entrada:</strong> {html.escape(str(cfg.dados))}</p>
<p><strong>Saida:</strong> {html.escape(str(cfg.out))}</p>
<p><strong>Log:</strong> {html.escape(str(log_path))}</p>

<h2>Blocos executados</h2>
<ol>
<li><strong>Analise individual:</strong> perfil, gold, respostas eleitorais e saidas por ano para cada JSON original.</li>
<li><strong>Analise global:</strong> consolida as saidas individuais em tabelas CSV/Parquet, correlaciona e gera clusters/graficos.</li>
<li><strong>Simulacao:</strong> usa somente a base global consolidada para cenarios de 2026 por municipio/secao/entidade.</li>
</ol>

<ul>
<li><a href="global/relatorio_global.html">Relatorio global</a></li>
<li><a href="preditivo_2026/relatorio_simulacao.html">Relatorio de simulacao</a></li>
<li><a href="preditivo_2026/explicabilidade/explicacao_detalhada_simulacao.md">Explicacao detalhada da simulacao</a></li>
<li><a href="relatorio_executivo/relatorio_eleicoes_simulacao.html">Relatorio executivo enxuto</a></li>
<li><a href="relatorio_executivo/relatorio_eleicoes_simulacao.pdf">PDF do relatorio executivo</a></li>
</ul>

<h2>Dashboard Streamlit</h2>
<p>Use este front junto com os HTMLs quando quiser consultar por Brasil, estado, municipio, secao, clusters e simulacao com cards e graficos interativos.</p>
<pre>{html.escape(streamlit_cmd)}</pre>

<h2>Dashboard Dash/DuckDB</h2>
<p>Use este front quando quiser consultar os Parquets tratados diretamente, sem embutir tabelas grandes no HTML estatico.</p>
<pre>{html.escape(dash_cmd)}</pre>

<pre>{html.escape(json.dumps({
    "arquivos_json_encontrados": len(json_files),
    "arquivos_ok": ok,
    "arquivos_erro": error,
    "global": global_info,
    "predicao": pred_info,
}, ensure_ascii=False, indent=2, default=str))}</pre>
"""
    save_html(cfg.out / "index.html", "Indice pipeline eleitoral JSON", body)

    print("\n" + "=" * 100)
    print("PIPELINE ELEITORAL JSON FINALIZADO")
    print(f"JSONs encontrados:  {len(json_files)}")
    print(f"Arquivos OK:        {ok}")
    print(f"Arquivos com erro:  {error}")
    print(f"Resultados:         {cfg.out}")
    print(f"Indice:             {cfg.out / 'index.html'}")
    print(f"Relatorio global:   {cfg.out / 'global' / 'relatorio_global.html'}")
    print(f"Simulacao:          {cfg.out / 'preditivo_2026' / 'relatorio_simulacao.html'}")
    print(f"Log:                {log_path}")
    print("=" * 100)


def run_clean_database_mode(cfg) -> None:
    if not cfg.dados.exists() or not cfg.dados.is_dir():
        print(f"[ERRO] Pasta de dados nao existe: {cfg.dados}")
        sys.exit(1)

    clean_cfg = CleanDatabaseConfig(
        dados=cfg.dados,
        out=cfg.banco_out,
        chunk_rows=cfg.banco_chunk_rows,
        max_files=cfg.banco_max_files,
        workers=cfg.banco_workers,
        workers_large_files=cfg.banco_workers_large_files,
        large_file_threshold_gb=cfg.banco_large_file_threshold_gb,
        ouro_workers=cfg.banco_ouro_workers,
        duckdb_threads=cfg.banco_duckdb_threads,
        overwrite=cfg.banco_overwrite,
        resume=cfg.resume,
        include_metadata=cfg.incluir_metadados_json,
        skip_heavy_analyses=cfg.banco_skip_heavy_analyses,
        delete_source_after_success=cfg.banco_delete_source_after_success,
        ouro_parallel_aggressive=cfg.banco_ouro_parallel_aggressive,
        auto_tune_info=cfg.banco_auto_tune_info,
        log_level=cfg.log_level,
    )

    if cfg.modo == "banco":
        summary = build_clean_database(clean_cfg)
        print("\n" + "=" * 100)
        print("BANCO ELEITORAL LIMPO CRIADO")
        print(f"Bronze: {cfg.banco_out / 'bronze'}")
        print(f"Prata:  {cfg.banco_out / 'prata'}")
        print("Ouro:   pendente; gere com --modo analise_banco")
        print(f"Auto:   workers_por_arquivo={max(cfg.banco_workers, cfg.banco_workers_large_files)}, chunk={cfg.banco_chunk_rows}, duckdb_threads={cfg.banco_duckdb_threads}, todos_workers={cfg.banco_use_all_workers}")
        print(f"Resumo: {cfg.banco_out / '_banco_eleitoral_limpo.json'}")
        print("=" * 100)
        return

    if not cfg.banco_out.exists() or not (cfg.banco_out / "prata").exists():
        print(f"[ERRO] Banco limpo nao existe: {cfg.banco_out}")
        print("Crie primeiro com --modo banco.")
        sys.exit(1)

    cfg.banco_out.mkdir(parents=True, exist_ok=True)
    setup_logging(cfg.banco_out, cfg.log_level)
    logging.info("Modo analise_banco: lendo banco limpo em %s", cfg.banco_out)
    analysis_info = run_clean_database_analyses(clean_cfg)

    cfg.out = cfg.banco_out
    global_info = {
        "global_gold_parquet": str(cfg.banco_out / "ouro" / "base_gold_global"),
        "global_gold_csv": "",
        "municipal_parquet": str(cfg.banco_out / "ouro" / "retrato_municipal"),
        "timeline_nacional_parquet": str(cfg.banco_out / "ouro" / "timeline_nacional.parquet"),
        "analise_eleitoral_outputs": {
            "perfil_eleitor_por_ano_parquet": str(cfg.banco_out / "ouro" / "perfil_eleitor_por_ano"),
            "perfil_eleitor_por_partido_parquet": str(cfg.banco_out / "ouro" / "perfil_eleitor_por_partido"),
            "perfil_eleitor_por_candidato_parquet": str(cfg.banco_out / "ouro" / "perfil_eleitor_por_candidato"),
            "top10_perfis_federacao_estado_municipio_parquet": str(cfg.banco_out / "ouro" / "top10_perfis_federacao_estado_municipio"),
        },
    }
    pred_info = {}
    if cfg.predict_2026 or cfg.modo == "analise_banco":
        pred_info = run_simulation_stage(global_info, cfg)

    save_json({"status": "ok", "analises": analysis_info, "predicao": pred_info}, cfg.banco_out / "logs" / "analise_banco_info.json")
    print("\n" + "=" * 100)
    print("ANALISE/SIMULACAO SOBRE BANCO LIMPO FINALIZADA")
    print(f"Banco:     {cfg.banco_out}")
    print(f"Ouro:      {cfg.banco_out / 'ouro'}")
    print(f"Simulacao: {cfg.banco_out / 'preditivo_2026'}")
    print(f"Dashboard: python3 scripts/dashboard_dash_eleitoral.py --run {cfg.banco_out} --host 0.0.0.0 --port 8050")
    print("=" * 100)


if __name__ == "__main__":
    main()
