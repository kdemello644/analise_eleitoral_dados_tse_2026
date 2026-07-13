from __future__ import annotations

from pathlib import Path
from typing import Any
import html
import json
import logging

import pandas as pd

from .aggregation import (
    GOLD_KEYS,
    GOLD_METRICS,
    add_derived_metrics,
    build_file_temporal_inventory,
    build_file_year_matrix,
    build_timelines,
    consolidate_municipal,
    entity_share_correlations,
    temporal_correlations,
)
from .plots import plot_behavior_clusters, plot_global
from .electoral_analysis import run_electoral_analysis
from .comportamento_eleitoral import run_behavioral_cluster_analysis
from .global_correlation import build_correlated_year_parquets
from .global_cluster_analysis import run_global_discriminated_cluster_analysis
from .discrete import discrete_summary
from .profiler import (
    collect_profiles,
    field_similarity,
    learned_canonical_map,
    table_similarity,
)
from .utils import (
    clean_memory,
    df_to_html,
    img_tag,
    save_csv,
    save_html,
    save_parquet,
    safe_text,
)


def _read_csv_if_exists(path_value: str) -> pd.DataFrame:
    if not path_value:
        return pd.DataFrame()
    path = Path(path_value)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep=";", dtype=str)
    except Exception as exc:
        logging.warning("Falha lendo tabela auxiliar global %s: %s", path, exc)
        return pd.DataFrame()


def _read_table_if_exists(path_value: str, max_rows: int | None = None) -> pd.DataFrame:
    if not path_value:
        return pd.DataFrame()
    path = Path(path_value)
    if not path.exists():
        return pd.DataFrame()
    try:
        if path.is_dir() or path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
            return df.head(max_rows) if max_rows else df
        return pd.read_csv(path, sep=";", dtype=str, nrows=max_rows)
    except Exception as exc:
        logging.warning("Falha lendo tabela %s: %s", path, exc)
        return pd.DataFrame()


def _read_csv_preview(path_value: str, max_rows: int = 80) -> pd.DataFrame:
    if not path_value:
        return pd.DataFrame()
    path = Path(path_value)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep=";", dtype=str, nrows=max_rows)
    except Exception as exc:
        logging.warning("Falha lendo preview %s: %s", path, exc)
        return pd.DataFrame()


def _read_output_table(outputs: dict[str, Any], parquet_key: str, csv_key: str, max_rows: int | None = None) -> pd.DataFrame:
    df = _read_table_if_exists(safe_text(outputs.get(parquet_key, "")), max_rows=max_rows)
    if not df.empty:
        return df
    return _read_table_if_exists(safe_text(outputs.get(csv_key, "")), max_rows=max_rows)


def materialize_global_gold(results: list[dict[str, Any]], cfg, global_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    parts_dir = global_dir / "parquet" / "base_gold_global_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    compact_parts: list[Path] = []
    pending: list[pd.DataFrame] = []
    source_rows = []
    total_sources = 0
    total_input_rows = 0
    part_idx = 0
    chunk_rows = max(50000, int(getattr(cfg, "aggregate_chunk_rows", 75000) or 75000))

    for r in results:
        if r.get("status") != "ok":
            continue
        for source in _gold_sources_for_result(r):
            total_sources += 1
            df = _read_gold_source(source)
            if df.empty:
                continue
            if "arquivo_origem" not in df.columns:
                df["arquivo_origem"] = r.get("relativo", "")
            if "aggregation_mode" not in df.columns:
                df["aggregation_mode"] = r.get("gold_storage", {}).get("modo_gold", "")
            total_input_rows += len(df)
            pending.append(_compact_gold_frame(df))
            source_rows.append({
                "arquivo_origem": r.get("relativo", ""),
                "source": source,
                "linhas_lidas": int(len(df)),
            })
            if sum(len(p) for p in pending if p is not None) >= chunk_rows:
                part_idx += 1
                path = _write_global_gold_compact_part(pending, parts_dir, part_idx)
                if path:
                    compact_parts.append(path)
                pending = []
                clean_memory()

    if pending:
        part_idx += 1
        path = _write_global_gold_compact_part(pending, parts_dir, part_idx)
        if path:
            compact_parts.append(path)
        pending = []
        clean_memory()

    final = _merge_global_gold_parts(compact_parts)
    max_rows = int(getattr(cfg, "global_max_gold_rows", 0) or 0)
    capped = False
    if max_rows > 0 and len(final) > max_rows:
        logging.warning(
            "Aplicando limite explicito global_max_gold_rows=%s sobre %s linhas compactadas.",
            max_rows,
            len(final),
        )
        final = final.head(max_rows).copy()
        capped = True

    manifest = pd.DataFrame(source_rows)
    manifest_path = global_dir / "tabelas" / "manifesto_fontes_gold_global.csv"
    save_csv(manifest, manifest_path)
    info = {
        "fontes_gold_lidas": int(total_sources),
        "linhas_gold_entrada": int(total_input_rows),
        "linhas_gold_compactadas": int(len(final)),
        "partes_gold_global": len(compact_parts),
        "base_gold_global_parts_dir": str(parts_dir),
        "manifesto_fontes_gold_global_csv": str(manifest_path),
        "global_max_gold_rows_aplicado": capped,
    }
    return final, info


def _gold_sources_for_result(result: dict[str, Any]) -> list[str]:
    sources: list[str] = []
    manifest_value = safe_text(result.get("gold_parts_manifest_parquet", "")) or safe_text(result.get("gold_parts_manifest", ""))
    if manifest_value:
        manifest = _read_table_if_exists(manifest_value)
        for _, row in manifest.iterrows():
            parquet = safe_text(row.get("parquet", ""))
            csv = safe_text(row.get("csv", ""))
            if parquet:
                sources.append(parquet)
            elif csv:
                sources.append(csv)
    parquet = safe_text(result.get("gold_parquet", ""))
    csv = safe_text(result.get("gold_csv", ""))
    if parquet and parquet not in sources:
        sources.append(parquet)
    if not sources and csv:
        sources.append(csv)
    return sources


def _read_gold_source(path_value: str) -> pd.DataFrame:
    path = Path(path_value)
    if not path.exists():
        return pd.DataFrame()
    try:
        if path.is_dir() or path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        if path.stat().st_size <= 5:
            return pd.DataFrame()
        return pd.read_csv(path, sep=";", dtype=str, compression="infer")
    except Exception as exc:
        logging.warning("Falha lendo fonte gold %s: %s", path, exc)
        return pd.DataFrame()


def _compact_gold_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for col in GOLD_KEYS:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].map(lambda x: safe_text(x, ""))
    if "arquivo_origem" not in out.columns:
        out["arquivo_origem"] = ""
    if "aggregation_mode" not in out.columns:
        out["aggregation_mode"] = ""
    out["arquivo_origem"] = out["arquivo_origem"].map(lambda x: safe_text(x, ""))
    out["aggregation_mode"] = out["aggregation_mode"].map(lambda x: safe_text(x, ""))
    for col in GOLD_METRICS + ["linhas_origem"]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    group = GOLD_KEYS + ["arquivo_origem", "aggregation_mode"]
    compact = out.groupby(group, dropna=False)[GOLD_METRICS + ["linhas_origem"]].sum().reset_index()
    return add_derived_metrics(compact)


def _write_global_gold_compact_part(frames: list[pd.DataFrame], parts_dir: Path, part_idx: int) -> Path | None:
    merged = _merge_gold_frames(frames)
    if merged.empty:
        return None
    path = parts_dir / f"base_gold_global_compact_{part_idx:05d}.parquet"
    if save_parquet(merged, path):
        return path
    csv_path = parts_dir / f"base_gold_global_compact_{part_idx:05d}.csv.gz"
    merged.to_csv(csv_path, sep=";", index=False, encoding="utf-8-sig", compression="gzip")
    return csv_path


def _merge_global_gold_parts(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = _read_gold_source(str(path))
        if not df.empty:
            frames.append(df)
        if sum(len(f) for f in frames) >= 500000:
            frames = [_merge_gold_frames(frames)]
            clean_memory()
    return _merge_gold_frames(frames)


def _merge_gold_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    valid = [f for f in frames if f is not None and not f.empty]
    if not valid:
        return pd.DataFrame()
    merged = pd.concat(valid, ignore_index=True, sort=False)
    return _compact_gold_frame(merged)


def build_global(results: list[dict[str, Any]], cfg) -> dict[str, Any]:
    global_dir = Path(cfg.out) / "global"
    tables_dir = global_dir / "tabelas"
    schema_dir = global_dir / "schema"
    timeline_dir = global_dir / "timeline"
    corr_dir = global_dir / "correlacoes"
    plots_dir = global_dir / "plots"
    parquet_dir = global_dir / "parquet"

    for d in [tables_dir, schema_dir, timeline_dir, corr_dir, plots_dir, parquet_dir]:
        d.mkdir(parents=True, exist_ok=True)

    catalog = pd.DataFrame(results)
    save_csv(catalog, tables_dir / "catalogo_processamento.csv")

    profiles = collect_profiles(results)
    sim_tables = table_similarity(profiles)
    sim_fields = field_similarity(profiles)
    canonical_map = learned_canonical_map(profiles, sim_fields)

    save_csv(profiles, schema_dir / "catalogo_campos_todos_arquivos.csv")
    save_csv(sim_tables, schema_dir / "similaridade_entre_tabelas.csv")
    save_csv(sim_fields, schema_dir / "similaridade_entre_campos.csv")
    save_csv(canonical_map, schema_dir / "mapa_canonico_campos_aprendido.csv")

    global_gold, global_gold_materialization = materialize_global_gold(results, cfg, global_dir)
    for c in [
        "votos", "eleitorado", "comparecimento", "abstencao", "brancos", "nulos", "validos",
        "validos_estimados", "comparecimento_estimado", "abstencao_estimado",
        "pct_comparecimento", "pct_abstencao", "share_votos_grupo",
    ]:
        if c in global_gold.columns:
            global_gold[c] = pd.to_numeric(global_gold[c], errors="coerce")

    global_gold, correlation_outputs = build_correlated_year_parquets(global_gold, results, global_dir, cfg)

    gold_csv = tables_dir / "base_gold_global.csv"
    gold_parquet = parquet_dir / "base_gold_global.parquet"
    csv_limit = int(getattr(cfg, "gold_csv_max_rows", 150000) or 150000)
    save_csv(global_gold.head(csv_limit), gold_csv)
    if cfg.parquet:
        save_parquet(global_gold, gold_parquet)

    inventory = build_file_temporal_inventory(results, global_gold)
    year_matrix = build_file_year_matrix(inventory)
    save_csv(inventory, timeline_dir / "inventario_temporal_arquivos.csv")
    save_csv(year_matrix, timeline_dir / "matriz_arquivo_ano.csv")

    municipal = consolidate_municipal(global_gold)
    municipal_csv = tables_dir / "retrato_municipal_global.csv"
    municipal_parquet = parquet_dir / "retrato_municipal_global.parquet"
    save_csv(municipal, municipal_csv)
    if cfg.parquet:
        save_parquet(municipal, municipal_parquet)

    timelines = build_timelines(global_gold, municipal)
    for name, df in timelines.items():
        save_csv(df, timeline_dir / f"{name}.csv")
        if cfg.parquet:
            save_parquet(df, parquet_dir / f"{name}.parquet")

    corr_temporal = temporal_correlations(timelines.get("timeline_municipal", pd.DataFrame()))
    corr_entities = entity_share_correlations(timelines.get("timeline_entidades", pd.DataFrame()))
    save_csv(corr_temporal, corr_dir / "correlacoes_temporais_municipais.csv")
    save_csv(corr_entities, corr_dir / "correlacoes_share_entidades_entre_anos.csv")

    correlation_manifest = _read_table_if_exists(correlation_outputs.get("manifesto_parquets_correlacionados_parquet", "")) 
    if correlation_manifest.empty:
        correlation_manifest = _read_csv_if_exists(correlation_outputs.get("manifesto_parquets_correlacionados_csv", ""))
    correlation_stats = _read_table_if_exists(correlation_outputs.get("estatisticas_correlacionadas_por_ano_parquet", ""))
    if correlation_stats.empty:
        correlation_stats = _read_csv_if_exists(correlation_outputs.get("estatisticas_correlacionadas_por_ano_csv", ""))
    correlation_dictionary = _read_csv_if_exists(correlation_outputs.get("dicionario_correlacao_codigos_csv", ""))

    global_discriminated_cluster_outputs = run_global_discriminated_cluster_analysis(
        correlation_outputs,
        Path(correlation_outputs.get("correlacao_codigos_dir", global_dir / "correlacao_codigos")) / "clusters",
        cfg,
    )
    cluster_global_summary = _read_output_table(global_discriminated_cluster_outputs, "clusters_globais_discriminados_resumo_parquet", "clusters_globais_discriminados_resumo_csv")
    cluster_global_personas = _read_output_table(global_discriminated_cluster_outputs, "clusters_personas_parquet", "clusters_personas_csv")
    cluster_global_discriminants = _read_output_table(global_discriminated_cluster_outputs, "clusters_valores_discriminantes_parquet", "clusters_valores_discriminantes_csv")
    cluster_global_year_region = _read_output_table(global_discriminated_cluster_outputs, "clusters_ano_regiao_parquet", "clusters_ano_regiao_csv")
    cluster_global_municipalities = _read_output_table(global_discriminated_cluster_outputs, "clusters_municipios_parquet", "clusters_municipios_csv")
    cluster_global_entities = _read_output_table(global_discriminated_cluster_outputs, "clusters_entidades_parquet", "clusters_entidades_csv")
    cluster_global_prediction = _read_output_table(global_discriminated_cluster_outputs, "clusters_predicao_2026_parquet", "clusters_predicao_2026_csv")
    cluster_global_elbow = _read_output_table(global_discriminated_cluster_outputs, "clusters_cotovelo_k_parquet", "clusters_cotovelo_k_csv")
    cluster_voter_summary = _read_output_table(global_discriminated_cluster_outputs, "clusters_globais_eleitores_resumo_parquet", "clusters_globais_eleitores_resumo_csv")
    cluster_voter_personas = _read_output_table(global_discriminated_cluster_outputs, "clusters_eleitores_personas_parquet", "clusters_eleitores_personas_csv")
    cluster_voter_discriminants = _read_output_table(global_discriminated_cluster_outputs, "clusters_eleitores_valores_discriminantes_parquet", "clusters_eleitores_valores_discriminantes_csv")
    cluster_voter_year_region = _read_output_table(global_discriminated_cluster_outputs, "clusters_eleitores_ano_regiao_parquet", "clusters_eleitores_ano_regiao_csv")
    cluster_voter_municipalities = _read_output_table(global_discriminated_cluster_outputs, "clusters_eleitores_municipios_parquet", "clusters_eleitores_municipios_csv")
    cluster_voter_elbow = _read_output_table(global_discriminated_cluster_outputs, "clusters_eleitores_cotovelo_k_parquet", "clusters_eleitores_cotovelo_k_csv")

    eleitoral_dir = global_dir / "analise_eleitoral"
    electoral_outputs = run_electoral_analysis(global_gold, eleitoral_dir, profiles)

    comportamento_dir = global_dir / "comportamento_eleitoral"
    comportamento_outputs = run_behavioral_cluster_analysis(global_gold, comportamento_dir)

    images = plot_global(municipal, timelines, corr_temporal, plots_dir, cfg)
    global_cluster_images = [
        Path(p) for p in global_discriminated_cluster_outputs.get("plots", [])
        if p and Path(p).exists()
    ]
    images.extend(global_cluster_images)
    cluster_images = []
    try:
        clustered_path = comportamento_outputs.get("clusters_comportamento_eleitoral", "")
        interp_path = comportamento_outputs.get("clusters_comportamento_interpretacao", "")
        clustered_df = pd.read_csv(clustered_path, sep=";", dtype=str) if clustered_path else pd.DataFrame()
        interp_df = pd.read_csv(interp_path, sep=";", dtype=str) if interp_path else pd.DataFrame()
        for c in ["votos", "eleitorado", "taxa_abstencao", "taxa_comparecimento"]:
            if c in clustered_df.columns:
                clustered_df[c] = pd.to_numeric(clustered_df[c], errors="coerce")
        for c in ["total_votos_cluster", "share_entidade_no_cluster", "taxa_abstencao_media"]:
            if c in interp_df.columns:
                interp_df[c] = pd.to_numeric(interp_df[c], errors="coerce")
        cluster_images = plot_behavior_clusters(clustered_df, interp_df, plots_dir / "clusters", cfg)
        images.extend(cluster_images)
    except Exception as exc:
        logging.warning("Falha gerando graficos de clusters comportamentais: %s", exc)

    story = global_story(
        profiles=profiles,
        inventory=inventory,
        sim_tables=sim_tables,
        sim_fields=sim_fields,
        corr_temporal=corr_temporal,
        corr_entities=corr_entities,
        municipal=municipal,
        correlation_outputs=correlation_outputs,
        correlation_stats=correlation_stats,
    )

    body = f"""
<h2>Storytelling global</h2>
<pre>{html.escape(story)}</pre>

<h2>Inventário temporal dos arquivos</h2>
{df_to_html(inventory, cfg.top_n_html)}

<h2>Matriz arquivo × ano</h2>
{df_to_html(year_matrix, cfg.top_n_html)}

<h2>Parquets correlacionados por ano/codigo</h2>
{df_to_html(correlation_manifest, cfg.top_n_html)}

<h2>Estatisticas correlacionadas por ano/codigo</h2>
{df_to_html(correlation_stats, cfg.top_n_html)}

<h2>Dicionario da correlacao por codigo</h2>
{df_to_html(correlation_dictionary, cfg.top_n_html)}

<h2>Catálogo de campos de todos os arquivos</h2>
{df_to_html(profiles, cfg.top_n_html)}

<h2>Mapa canônico de campos aprendido dos dados</h2>
{df_to_html(canonical_map, cfg.top_n_html)}

<h2>Similaridade entre tabelas</h2>
{df_to_html(sim_tables, cfg.top_n_html)}

<h2>Similaridade entre campos</h2>
{df_to_html(sim_fields, cfg.top_n_html)}

<h2>Timeline nacional</h2>
{df_to_html(timelines.get("timeline_nacional"), cfg.top_n_html)}

<h2>Timeline por UF</h2>
{df_to_html(timelines.get("timeline_uf"), cfg.top_n_html)}

<h2>Evolução municipal</h2>
{df_to_html(timelines.get("evolucao_municipal"), cfg.top_n_html)}

<h2>Correlação temporal municipal</h2>
{df_to_html(corr_temporal, cfg.top_n_html)}

<h2>Correlação de entidades entre anos</h2>
{df_to_html(corr_entities, cfg.top_n_html)}

<h2>Retrato municipal global</h2>
{df_to_html(municipal, cfg.top_n_html)}

<h2>Análises eleitorais globais geradas</h2>
<pre>{html.escape(__import__("json").dumps(electoral_outputs, ensure_ascii=False, indent=2, default=str))}</pre>

<h2>Clusters comportamentais: quem vota em quem, onde e com que perfil</h2>
<pre>{html.escape(__import__("json").dumps(comportamento_outputs, ensure_ascii=False, indent=2, default=str))}</pre>

<h2>Relatório interpretável dos clusters comportamentais</h2>
<p><strong>Arquivo:</strong> {html.escape(comportamento_outputs.get("clusters_comportamento_relatorio_md", ""))}</p>
{df_to_html(_read_csv_preview(comportamento_outputs.get("clusters_comportamento_interpretacao", ""), cfg.top_n_html), cfg.top_n_html)}

<h2>Quem vota em quem</h2>
{df_to_html(_read_csv_preview(comportamento_outputs.get("quem_vota_em_quem", ""), cfg.top_n_html), cfg.top_n_html)}

<h2>Afinidade perfil-candidato/partido</h2>
{df_to_html(_read_csv_preview(comportamento_outputs.get("afinidade_perfil_candidato", ""), cfg.top_n_html), cfg.top_n_html)}

<h2>Perfil do eleitorado</h2>
{df_to_html(_read_csv_preview(electoral_outputs.get("perfis_eleitorado", ""), cfg.top_n_html), cfg.top_n_html)}

<h2>Clusters de perfil do eleitorado</h2>
{df_to_html(_read_csv_preview(electoral_outputs.get("clusters_perfil_eleitorado_resumo", ""), cfg.top_n_html), cfg.top_n_html)}

<h2>Como votou cada seção</h2>
{df_to_html(_read_csv_preview(electoral_outputs.get("como_votou_cada_secao", ""), cfg.top_n_html), cfg.top_n_html)}

<h2>Vencedor por seção</h2>
{df_to_html(_read_csv_preview(electoral_outputs.get("vencedor_por_secao", ""), cfg.top_n_html), cfg.top_n_html)}

<h2>Vencedor por seção explicado</h2>
{df_to_html(_read_csv_preview(electoral_outputs.get("vencedor_secao_explicado", ""), cfg.top_n_html), cfg.top_n_html)}

<h2>Perfil do eleitor por ano</h2>
{df_to_html(_read_csv_preview(electoral_outputs.get("perfil_eleitor_por_ano", ""), cfg.top_n_html), cfg.top_n_html)}

<h2>Perfil voto proxy</h2>
{df_to_html(_read_csv_preview(electoral_outputs.get("perfil_voto_proxy", ""), cfg.top_n_html), cfg.top_n_html)}

<h2>Respostas eleitorais</h2>
{df_to_html(_read_csv_preview(electoral_outputs.get("respostas_perguntas_eleitorais", ""), cfg.top_n_html), cfg.top_n_html)}

<h2>Tendências de voto partido/candidato</h2>
{df_to_html(_read_csv_preview(electoral_outputs.get("tendencias_voto_partido_candidato", ""), cfg.top_n_html), cfg.top_n_html)}

<h2>Matriz de transferência de votos proxy</h2>
{df_to_html(_read_csv_preview(electoral_outputs.get("matriz_transferencia_votos_proxy", ""), cfg.top_n_html), cfg.top_n_html)}

<h2>Voto útil proxy</h2>
{df_to_html(_read_csv_preview(electoral_outputs.get("voto_util_proxy", ""), cfg.top_n_html), cfg.top_n_html)}

<h2>Gráficos</h2>
{''.join(img_tag(img, global_dir) for img in images)}
"""
    body = build_global_dashboard_body(
        cfg=cfg,
        global_dir=global_dir,
        story=story,
        results=results,
        profiles=profiles,
        inventory=inventory,
        year_matrix=year_matrix,
        canonical_map=canonical_map,
        sim_tables=sim_tables,
        sim_fields=sim_fields,
        correlation_manifest=correlation_manifest,
        correlation_stats=correlation_stats,
        correlation_dictionary=correlation_dictionary,
        timelines=timelines,
        corr_temporal=corr_temporal,
        corr_entities=corr_entities,
        municipal=municipal,
        electoral_outputs=electoral_outputs,
        comportamento_outputs=comportamento_outputs,
        global_discriminated_cluster_outputs=global_discriminated_cluster_outputs,
        cluster_global_summary=cluster_global_summary,
        cluster_global_personas=cluster_global_personas,
        cluster_global_discriminants=cluster_global_discriminants,
        cluster_global_year_region=cluster_global_year_region,
        cluster_global_municipalities=cluster_global_municipalities,
        cluster_global_entities=cluster_global_entities,
        cluster_global_prediction=cluster_global_prediction,
        cluster_global_elbow=cluster_global_elbow,
        cluster_voter_summary=cluster_voter_summary,
        cluster_voter_personas=cluster_voter_personas,
        cluster_voter_discriminants=cluster_voter_discriminants,
        cluster_voter_year_region=cluster_voter_year_region,
        cluster_voter_municipalities=cluster_voter_municipalities,
        cluster_voter_elbow=cluster_voter_elbow,
        images=images,
    )
    html_path = global_dir / "relatorio_global.html"
    save_html(html_path, "Relatório global - análise data-driven dos JSON", body)

    return {
        "global_gold_csv": str(gold_csv),
        "global_gold_parquet": str(gold_parquet) if cfg.parquet else "",
        "global_gold_csv_observacao": f"Preview limitado a {getattr(cfg, 'gold_csv_max_rows', 150000)} linhas; base completa fica no Parquet.",
        "global_gold_materialization": global_gold_materialization,
        "municipal_csv": str(municipal_csv),
        "municipal_parquet": str(municipal_parquet) if cfg.parquet else "",
        "inventario_temporal_csv": str(timeline_dir / "inventario_temporal_arquivos.csv"),
        "matriz_arquivo_ano_csv": str(timeline_dir / "matriz_arquivo_ano.csv"),
        "timeline_nacional_csv": str(timeline_dir / "timeline_nacional.csv"),
        "timeline_uf_csv": str(timeline_dir / "timeline_uf.csv"),
        "timeline_municipal_csv": str(timeline_dir / "timeline_municipal.csv"),
        "timeline_entidades_csv": str(timeline_dir / "timeline_entidades.csv"),
        "evolucao_municipal_csv": str(timeline_dir / "evolucao_municipal.csv"),
        "correlacoes_temporais_csv": str(corr_dir / "correlacoes_temporais_municipais.csv"),
        "correlacoes_entidades_csv": str(corr_dir / "correlacoes_share_entidades_entre_anos.csv"),
        "similaridade_tabelas_csv": str(schema_dir / "similaridade_entre_tabelas.csv"),
        "similaridade_campos_csv": str(schema_dir / "similaridade_entre_campos.csv"),
        "mapa_canonico_csv": str(schema_dir / "mapa_canonico_campos_aprendido.csv"),
        "correlacao_codigos_outputs": correlation_outputs,
        "analise_eleitoral_dir": str(eleitoral_dir),
        "analise_eleitoral_outputs": electoral_outputs,
        "comportamento_eleitoral_dir": str(comportamento_dir),
        "comportamento_eleitoral_outputs": comportamento_outputs,
        "clusters_globais_discriminados_outputs": global_discriminated_cluster_outputs,
        "clusters_globais_discriminados_plots": [str(p) for p in global_cluster_images],
        "cluster_plots": [str(p) for p in cluster_images],
        "html": str(html_path),
    }


def build_global_dashboard_body(
    cfg,
    global_dir: Path,
    story: str,
    results: list[dict[str, Any]],
    profiles: pd.DataFrame,
    inventory: pd.DataFrame,
    year_matrix: pd.DataFrame,
    canonical_map: pd.DataFrame,
    sim_tables: pd.DataFrame,
    sim_fields: pd.DataFrame,
    correlation_manifest: pd.DataFrame,
    correlation_stats: pd.DataFrame,
    correlation_dictionary: pd.DataFrame,
    timelines: dict[str, pd.DataFrame],
    corr_temporal: pd.DataFrame,
    corr_entities: pd.DataFrame,
    municipal: pd.DataFrame,
    electoral_outputs: dict[str, str],
    comportamento_outputs: dict[str, str],
    global_discriminated_cluster_outputs: dict[str, Any],
    cluster_global_summary: pd.DataFrame,
    cluster_global_personas: pd.DataFrame,
    cluster_global_discriminants: pd.DataFrame,
    cluster_global_year_region: pd.DataFrame,
    cluster_global_municipalities: pd.DataFrame,
    cluster_global_entities: pd.DataFrame,
    cluster_global_prediction: pd.DataFrame,
    cluster_global_elbow: pd.DataFrame,
    cluster_voter_summary: pd.DataFrame,
    cluster_voter_personas: pd.DataFrame,
    cluster_voter_discriminants: pd.DataFrame,
    cluster_voter_year_region: pd.DataFrame,
    cluster_voter_municipalities: pd.DataFrame,
    cluster_voter_elbow: pd.DataFrame,
    images: list[Path],
) -> str:
    preview_rows = min(int(getattr(cfg, "top_n_html", 120) or 120), 120)

    timeline_nacional = timelines.get("timeline_nacional", pd.DataFrame())
    timeline_uf = timelines.get("timeline_uf", pd.DataFrame())
    evolucao_municipal = timelines.get("evolucao_municipal", pd.DataFrame())

    vencedor_secao = _read_csv_preview(electoral_outputs.get("vencedor_por_secao", ""), preview_rows)
    vencedor_secao_exp = _read_csv_preview(electoral_outputs.get("vencedor_secao_explicado", ""), preview_rows)
    perfil_ano = _read_csv_preview(electoral_outputs.get("perfil_eleitor_por_ano", ""), preview_rows)
    respostas = _read_csv_preview(electoral_outputs.get("respostas_perguntas_eleitorais", ""), preview_rows)
    perfil_partido = _read_csv_preview(electoral_outputs.get("perfil_eleitor_por_partido", ""), preview_rows)
    perfil_candidato = _read_csv_preview(electoral_outputs.get("perfil_eleitor_por_candidato", ""), preview_rows)
    perfil_do_candidato = _read_csv_preview(electoral_outputs.get("perfil_do_candidato_correlacionado_eleitorado", ""), preview_rows)
    resultado_eleitorado = _read_csv_preview(electoral_outputs.get("resultado_eleitorado_correlacionado", ""), preview_rows)
    comparativo_perfil = _read_csv_preview(electoral_outputs.get("comparativo_anual_perfil_eleitor", ""), preview_rows)
    comparativo_partido = _read_csv_preview(electoral_outputs.get("comparativo_anual_perfil_partido", ""), preview_rows)
    comparativo_candidato = _read_csv_preview(electoral_outputs.get("comparativo_anual_perfil_candidato", ""), preview_rows)
    top10_perfis = _read_csv_preview(electoral_outputs.get("top10_perfis_federacao_estado_municipio", ""), preview_rows)
    quem_vota = _read_csv_preview(comportamento_outputs.get("quem_vota_em_quem", ""), preview_rows)
    afinidade = _read_csv_preview(comportamento_outputs.get("afinidade_perfil_candidato", ""), preview_rows)
    comportamento_interp = _read_csv_preview(comportamento_outputs.get("clusters_comportamento_interpretacao", ""), preview_rows)

    kpis = [
        ("Arquivos OK", str(sum(1 for r in results if r.get("status") == "ok")), "JSONs processados na etapa individual"),
        ("Anos", str(_nunique(correlation_stats, "ano") or _years_from_inventory(inventory)), "anos correlacionados"),
        ("Municipios", str(_nunique(municipal, "cd_municipio")), "municipios na base global"),
        ("Setores", _sum_as_int(correlation_stats, "setores_eleitorais"), "setores eleitorais por ano"),
        ("Clusters", f"{_nunique(cluster_voter_summary, 'cluster_global_discriminado')}/{_nunique(cluster_global_summary, 'cluster_global_discriminado')}", "eleitores / eleitores+resultado"),
        ("Graficos", str(len(images)), "figuras geradas no run"),
    ]

    image_gallery = _image_gallery(images, global_dir)
    individual_html = _individual_files_html(results, global_dir)
    cluster_report = html.escape(global_discriminated_cluster_outputs.get("clusters_relatorio_md", ""))
    cluster_voter_report = html.escape(global_discriminated_cluster_outputs.get("clusters_eleitores_relatorio_md", ""))
    behavior_report = html.escape(comportamento_outputs.get("clusters_comportamento_relatorio_md", ""))
    dashboard_payload = _build_dashboard_payload(
        municipal=municipal,
        evolucao_municipal=evolucao_municipal,
        cluster_personas=cluster_global_personas,
        cluster_municipalities=cluster_global_municipalities,
        cluster_prediction=cluster_global_prediction,
        cluster_elbow=cluster_global_elbow,
        cluster_summary=cluster_global_summary,
        cluster_discriminants=cluster_global_discriminants,
        cluster_year_region=cluster_global_year_region,
        cluster_entities=cluster_global_entities,
        cluster_voter_personas=cluster_voter_personas,
        cluster_voter_discriminants=cluster_voter_discriminants,
        cluster_voter_year_region=cluster_voter_year_region,
        cluster_voter_municipalities=cluster_voter_municipalities,
        cluster_voter_elbow=cluster_voter_elbow,
        timeline_uf=timeline_uf,
        timeline_nacional=timeline_nacional,
        respostas=respostas,
        perfil_ano=perfil_ano,
        perfil_partido=perfil_partido,
        perfil_candidato=perfil_candidato,
        perfil_do_candidato=perfil_do_candidato,
        resultado_eleitorado=resultado_eleitorado,
        comparativo_perfil=comparativo_perfil,
        comparativo_partido=comparativo_partido,
        comparativo_candidato=comparativo_candidato,
        top10_perfis=top10_perfis,
        vencedor_secao=vencedor_secao,
        vencedor_secao_exp=vencedor_secao_exp,
        quem_vota=quem_vota,
        afinidade=afinidade,
        comportamento_interp=comportamento_interp,
    )
    dashboard_script = _dashboard_script(dashboard_payload)

    return f"""
<style>
.dash {{ --bg:#f6f7f9; --ink:#18212f; --muted:#667085; --line:#d8dde6; --panel:#ffffff; --accent:#1f6feb; --accent2:#0f766e; background:var(--bg); margin:-32px; padding:0 0 36px; color:var(--ink); }}
.dash * {{ box-sizing:border-box; }}
.hero {{ padding:28px 34px 20px; background:#111827; color:white; }}
.hero h2 {{ margin:0; color:white; font-size:28px; }}
.hero p {{ max-width:980px; color:#d1d5db; }}
.wrap {{ max-width:1440px; margin:0 auto; padding:22px 28px; }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin-top:18px; }}
.kpi {{ background:#172033; border:1px solid #2d3748; border-radius:8px; padding:14px; }}
.kpi strong {{ display:block; font-size:25px; color:white; }}
.kpi span {{ color:#cbd5e1; font-size:12px; }}
.tabs {{ display:flex; gap:8px; flex-wrap:wrap; margin:18px 0; position:sticky; top:0; background:var(--bg); padding:10px 0; z-index:2; }}
.tabs button {{ border:1px solid var(--line); background:white; border-radius:7px; padding:9px 12px; cursor:pointer; color:var(--ink); }}
.tabs button.active {{ background:var(--accent); color:white; border-color:var(--accent); }}
.panel {{ display:none; }}
.panel.active {{ display:block; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:14px; align-items:start; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:16px; box-shadow:0 1px 2px rgba(16,24,40,.04); overflow:auto; }}
.card h3 {{ margin:0 0 10px; font-size:17px; color:#111827; }}
.card p, .smallnote {{ color:var(--muted); font-size:13px; }}
.wide {{ grid-column:1/-1; }}
.image-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(360px,1fr)); gap:14px; }}
.fig {{ background:white; border:1px solid var(--line); border-radius:8px; padding:10px; }}
.fig img {{ width:100%; height:auto; display:block; border:0; }}
.fig div {{ font-size:12px; color:var(--muted); padding-top:6px; }}
.fig,.card,.persona,.metric {{ transition:transform .16s ease, box-shadow .16s ease, border-color .16s ease; }}
.fig:hover,.card:hover,.persona:hover,.metric:hover {{ transform:scale(1.025); box-shadow:0 14px 34px rgba(15,23,42,.16); border-color:#9db7e8; z-index:3; }}
.dash table {{ font-size:12px; width:100%; border-collapse:collapse; }}
.dash th {{ background:#eef2f7; position:sticky; top:0; }}
.dash td,.dash th {{ border:1px solid #e5e7eb; padding:6px; vertical-align:top; }}
details {{ background:white; border:1px solid var(--line); border-radius:8px; margin:12px 0; padding:10px 12px; }}
summary {{ cursor:pointer; font-weight:700; }}
pre.story {{ max-height:280px; overflow:auto; background:#0b1220; color:#dbeafe; border:0; border-radius:8px; }}
.pill {{ display:inline-block; padding:3px 8px; border-radius:999px; background:#e0f2fe; color:#075985; font-size:12px; margin:2px; }}
.controlbar {{ display:grid; grid-template-columns:minmax(280px,1.2fr) minmax(180px,.8fr) auto; gap:10px; align-items:end; }}
.controlbar label {{ display:block; font-size:12px; color:var(--muted); margin-bottom:4px; }}
.controlbar select,.controlbar input {{ width:100%; border:1px solid var(--line); border-radius:7px; padding:10px; background:white; color:var(--ink); }}
.metric-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:10px; }}
.metric {{ border:1px solid #e5e7eb; border-radius:8px; padding:12px; background:#fbfcfe; }}
.metric span {{ display:block; color:var(--muted); font-size:12px; }}
.metric strong {{ display:block; font-size:22px; color:#111827; margin-top:4px; }}
.br-map-wrap {{ display:grid; grid-template-columns:minmax(320px,1.2fr) minmax(260px,.8fr); gap:14px; align-items:stretch; }}
.br-map {{ position:relative; min-height:430px; border:1px solid #e5e7eb; border-radius:8px; background:linear-gradient(180deg,#f8fbff,#eef7f6); overflow:hidden; }}
.br-map:before {{ content:""; position:absolute; inset:28px 46px; border-radius:46% 54% 52% 48%; border:2px dashed rgba(15,118,110,.22); transform:rotate(-12deg); }}
.uf-dot {{ position:absolute; transform:translate(-50%,-50%); min-width:42px; min-height:34px; border:1px solid #93c5fd; border-radius:8px; background:#fff; color:#0f172a; font-weight:700; cursor:pointer; box-shadow:0 8px 22px rgba(15,23,42,.08); animation:pulseState 2.6s ease-in-out infinite; }}
.uf-dot:hover,.uf-dot.active {{ background:#1f6feb; color:white; border-color:#1f6feb; transform:translate(-50%,-50%) scale(1.12); }}
.state-detail {{ border:1px solid #e5e7eb; border-radius:8px; padding:14px; background:#fbfcfe; min-height:160px; }}
@keyframes pulseState {{ 0%,100% {{ box-shadow:0 0 0 0 rgba(31,111,235,.18); }} 50% {{ box-shadow:0 0 0 8px rgba(31,111,235,0); }} }}
.bar-list {{ display:grid; gap:8px; }}
.bar-row {{ display:grid; grid-template-columns:minmax(120px,220px) 1fr minmax(48px,70px); gap:8px; align-items:center; font-size:12px; }}
.bar-label {{ overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:#344054; }}
.bar-track {{ height:12px; background:#e5e7eb; border-radius:999px; overflow:hidden; }}
.bar-fill {{ height:100%; background:var(--accent2); border-radius:999px; }}
.persona-list {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:12px; }}
.persona {{ border:1px solid #dbe3ef; border-radius:8px; padding:13px; background:#fbfdff; }}
.persona h4 {{ margin:0 0 8px; font-size:15px; }}
.persona p {{ margin:6px 0; }}
.table-tools {{ display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin-bottom:10px; }}
.table-tools select,.table-tools input {{ border:1px solid var(--line); border-radius:7px; padding:10px; min-width:220px; }}
.mini-table {{ max-height:520px; overflow:auto; border:1px solid #e5e7eb; border-radius:8px; }}
@media (max-width:760px) {{ .controlbar,.br-map-wrap {{ grid-template-columns:1fr; }} .bar-row {{ grid-template-columns:1fr; }} }}
</style>
<div class="dash">
  <div class="hero">
    <h2>Dashboard Global Eleitoral</h2>
    <p>Visao por Brasil, estado, municipio e secao eleitoral. Os graficos e cards ficam na frente; tabelas grandes e artefatos completos ficam em notas recolhidas.</p>
    <div class="kpis">{''.join(_kpi_card(*item) for item in kpis)}</div>
  </div>
  <div class="wrap">
    <div class="tabs">
      <button class="active" data-panel="brasil" onclick="showPanel('brasil', this)">Brasil</button>
      <button data-panel="estados" onclick="showPanel('estados', this)">Estados</button>
      <button data-panel="municipios" onclick="showPanel('municipios', this)">Municipios</button>
      <button data-panel="secoes" onclick="showPanel('secoes', this)">Secoes</button>
      <button data-panel="clusters" onclick="showPanel('clusters', this)">Clusters</button>
      <button data-panel="consulta" onclick="showPanel('consulta', this)">Tabelas</button>
      <button data-panel="arquivos" onclick="showPanel('arquivos', this)">Arquivos</button>
      <button data-panel="notas" onclick="showPanel('notas', this)">Notas</button>
    </div>

    <section id="brasil" class="panel active">
      <div class="grid">
        {_card("Eleitor medio no Brasil", "<div id='brasilEleitorMedio' class='persona-list'></div>", "wide")}
        {_card("Mapa interativo do Brasil", "<div id='brasilMap'></div>", "wide")}
        {_card("Perfil nacional dominante", "<div id='brasilPerfilChart' class='bar-list'></div>")}
        {_card("Votos por UF", "<div id='brasilVotosUf' class='bar-list'></div>")}
        {_card("Quem vota por partido", "<div id='brasilPartidos' class='bar-list'></div>", "wide")}
        {_card("Quem vota por candidato", "<div id='brasilCandidatos' class='persona-list'></div>", "wide")}
        {_card("Top 10 perfis - federacao", "<div id='top10FederacaoCards' class='persona-list'></div>", "wide")}
        {_card("Resumo nacional da eleicao", "<div id='brasilTimeline' class='bar-list'></div>")}
        {_card("Respostas eleitorais em cards", "<div id='brasilRespostas' class='persona-list'></div>", "wide")}
      </div>
    </section>

    <section id="estados" class="panel">
      <div class="grid">
        {_card("Analise por estado / UF", "<div id='estadoUfCards' class='persona-list'></div>", "wide")}
        {_card("Top 10 perfis por estado", "<div id='top10EstadoCards' class='persona-list'></div>", "wide")}
        {_card("Padrao anual do perfil por estado", "<div id='estadoPerfilCards' class='persona-list'></div>", "wide")}
        {_card("Clusters por ano, regiao e UF", "<div id='clusterAnoRegiaoCards' class='persona-list'></div>", "wide")}
      </div>
    </section>

    <section id="municipios" class="panel">
      <div class="grid">
        {_card("Escolha um municipio", _municipality_selector_html(), "wide")}
        {_card("Resumo do municipio", "<div id='municipioCards' class='metric-grid'></div>", "wide")}
        {_card("Evolucao por ano", "<div id='municipioTimeline' class='bar-list'></div>")}
        {_card("Clusters presentes no municipio", "<div id='municipioClusters' class='bar-list'></div>")}
        {_card("Top 10 perfis por municipio", "<div id='top10MunicipioCards' class='persona-list'></div>", "wide")}
        {_card("Recortes do municipio", "<div id='municipioRows' class='persona-list'></div>", "wide")}
      </div>
    </section>

    <section id="secoes" class="panel">
      <div class="grid">
        {_card("Vencedor por secao", "<div id='secaoVencedorCards' class='persona-list'></div>", "wide")}
        {_card("Vencedor por secao explicado", "<div id='secaoExpCards' class='persona-list'></div>", "wide")}
        {_card("Quem vota em quem", "<div id='quemVotaCards' class='persona-list'></div>", "wide")}
        {_card("Afinidade perfil-candidato/partido", "<div id='afinidadeCards' class='persona-list'></div>", "wide")}
        {_card("Perfil do candidato correlacionado ao eleitorado", "<div id='perfilCandidatoCards' class='persona-list'></div>", "wide")}
        {_card("Resultado + eleitorado correlacionado", "<div id='resultadoEleitoradoCards' class='persona-list'></div>", "wide")}
      </div>
    </section>

    <section id="clusters" class="panel">
      <div class="grid">
        {_card("Pessoa do cluster - somente eleitores", "<div id='clusterVoterPersonas' class='persona-list'></div>", "wide")}
        {_card("Pessoa do cluster - eleitores + resultado", "<div id='clusterPersonas' class='persona-list'></div>", "wide")}
        {_card("Cotovelo: escolha de k", "<div id='clusterElbow' class='bar-list'></div>")}
        {_card("Predicao 2026 por cluster", "<div id='clusterPrediction' class='bar-list'></div>")}
        {_card("Valores discriminantes dos clusters", "<div id='clusterDiscriminantsCards' class='persona-list'></div>", "wide")}
        {_card("Entidades por cluster", "<div id='clusterEntitiesCards' class='persona-list'></div>", "wide")}
        {_card("Clusters comportamentais", "<div id='comportamentoClusterCards' class='persona-list'></div>", "wide")}
        {_card("Relatorios dos clusters", f"<p><strong>Somente eleitores:</strong> {cluster_voter_report}</p><p><strong>Eleitores + resultado:</strong> {cluster_report}</p><p><strong>Comportamental:</strong> {behavior_report}</p>", "wide")}
      </div>
    </section>

    <section id="consulta" class="panel">
      <div class="grid">
        {_card("Consulta grafica das tabelas", _table_explorer_html(), "wide")}
      </div>
    </section>

    <section id="arquivos" class="panel">
      <div class="grid">
        {_card("Analises individuais dos arquivos", individual_html, "wide")}
      </div>
    </section>

    <section id="notas" class="panel">
      {_details("Manifesto dos Parquets correlacionados", correlation_manifest)}
      {_details("Estatisticas correlacionadas por ano/codigo", correlation_stats)}
      {_details("Dicionario da correlacao", correlation_dictionary)}
      {_details("Inventario temporal dos arquivos", inventory)}
      {_details("Matriz arquivo x ano", year_matrix)}
      {_details("Catalogo de campos", profiles)}
      {_details("Mapa canonico aprendido", canonical_map)}
      {_details("Similaridade entre tabelas", sim_tables)}
      {_details("Similaridade entre campos", sim_fields)}
      {_details("Correlacoes temporais", corr_temporal)}
      {_details("Correlacoes de entidades entre anos", corr_entities)}
    </section>

    <section class="panel active" style="display:block">
      <h2>Graficos</h2>
      {image_gallery}
    </section>
  </div>
</div>
{dashboard_script}
<script>
function showPanel(id, btn) {{
  document.querySelectorAll('.dash .panel').forEach(p => {{
    if (!p.querySelector('h2') || p.id) p.classList.remove('active');
    if (p.id) p.style.display = 'none';
  }});
  const panel = document.getElementById(id);
  if (panel) {{ panel.classList.add('active'); panel.style.display = 'block'; }}
  document.querySelectorAll('.tabs button').forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');
}}
document.addEventListener('DOMContentLoaded', function() {{
  if (window.initDashboardConsultas) window.initDashboardConsultas();
}});
</script>
"""


def _kpi_card(title: str, value: str, note: str) -> str:
    return f"<div class='kpi'><span>{html.escape(title)}</span><strong>{html.escape(value)}</strong><span>{html.escape(note)}</span></div>"


def _card(title: str, content: str, extra_class: str = "") -> str:
    return f"<div class='card {html.escape(extra_class)}'><h3>{html.escape(title)}</h3>{content}</div>"


def _municipality_selector_html() -> str:
    return """
<div class="controlbar">
  <div>
    <label for="municipioSelect">Municipio</label>
    <select id="municipioSelect"></select>
  </div>
  <div>
    <label for="municipioSearch">Buscar na lista</label>
    <input id="municipioSearch" type="search" placeholder="Digite municipio, UF ou codigo">
  </div>
  <button type="button" onclick="renderMunicipioSelecionado()">Atualizar</button>
</div>
<p class="smallnote">A lista traz todos os municipios detectados na base global. A visualizacao abaixo resume votos, abstencao, comparecimento, evolucao anual e clusters associados.</p>
"""


def _table_explorer_html() -> str:
    return """
<div class="table-tools">
  <select id="tableSelect"></select>
  <input id="tableSearch" type="search" placeholder="Buscar dentro da tabela selecionada">
  <button type="button" onclick="renderTabelaSelecionada()">Consultar</button>
</div>
<div class="grid">
  <div class="card">
    <h3>Grafico da tabela</h3>
    <div id="tableChart" class="bar-list"></div>
  </div>
  <div class="card">
    <h3>Preview filtrado</h3>
    <div id="tablePreview" class="mini-table"></div>
  </div>
</div>
<p class="smallnote">O preview e curto de proposito. Os CSVs e Parquets completos continuam salvos nas pastas do run para consulta pesada.</p>
"""


def _details(title: str, df: pd.DataFrame, max_rows: int = 80) -> str:
    return f"<details><summary>{html.escape(title)}</summary>{df_to_html(df.head(max_rows) if df is not None else pd.DataFrame(), max_rows)}<p class='smallnote'>Preview enxuto; os CSVs/Parquets completos ficam nas pastas do run.</p></details>"


def _image_gallery(images: list[Path], base: Path) -> str:
    if not images:
        return "<p class='smallnote'>Sem graficos gerados neste run.</p>"
    cards = []
    for image in images:
        try:
            src = image.relative_to(base).as_posix()
        except Exception:
            src = image.as_posix()
        cards.append(f"<div class='fig'><img src='{html.escape(src)}'><div>{html.escape(image.name)}</div></div>")
    return "<div class='image-grid'>" + "".join(cards) + "</div>"


def _individual_files_html(results: list[dict[str, Any]], base: Path) -> str:
    rows = []
    for r in results:
        rel = safe_text(r.get("relativo", ""))
        html_path = safe_text(r.get("html", ""))
        try:
            href = Path(html_path).relative_to(base).as_posix()
        except Exception:
            try:
                href = "../" + Path(html_path).relative_to(base.parent).as_posix()
            except Exception:
                href = html_path
        years = ", ".join(sorted(map(str, (r.get("analises_por_ano") or {}).keys())))
        rows.append({
            "arquivo": f"<a href='{html.escape(href)}'>{html.escape(rel)}</a>" if href else html.escape(rel),
            "status": html.escape(safe_text(r.get("status", ""))),
            "dominio": html.escape(safe_text(r.get("dominio_documento", ""))),
            "assunto": html.escape(safe_text(r.get("assunto_documento", ""))),
            "anos": html.escape(years),
        })
    if not rows:
        return "<p class='smallnote'>Sem analises individuais no manifesto.</p>"
    df = pd.DataFrame(rows)
    return df.to_html(index=False, escape=False)


def _build_dashboard_payload(
    municipal: pd.DataFrame,
    evolucao_municipal: pd.DataFrame,
    cluster_personas: pd.DataFrame,
    cluster_municipalities: pd.DataFrame,
    cluster_prediction: pd.DataFrame,
    cluster_elbow: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    cluster_discriminants: pd.DataFrame,
    cluster_year_region: pd.DataFrame,
    cluster_entities: pd.DataFrame,
    cluster_voter_personas: pd.DataFrame,
    cluster_voter_discriminants: pd.DataFrame,
    cluster_voter_year_region: pd.DataFrame,
    cluster_voter_municipalities: pd.DataFrame,
    cluster_voter_elbow: pd.DataFrame,
    timeline_uf: pd.DataFrame,
    timeline_nacional: pd.DataFrame,
    respostas: pd.DataFrame,
    perfil_ano: pd.DataFrame,
    perfil_partido: pd.DataFrame,
    perfil_candidato: pd.DataFrame,
    perfil_do_candidato: pd.DataFrame,
    resultado_eleitorado: pd.DataFrame,
    comparativo_perfil: pd.DataFrame,
    comparativo_partido: pd.DataFrame,
    comparativo_candidato: pd.DataFrame,
    top10_perfis: pd.DataFrame,
    vencedor_secao: pd.DataFrame,
    vencedor_secao_exp: pd.DataFrame,
    quem_vota: pd.DataFrame,
    afinidade: pd.DataFrame,
    comportamento_interp: pd.DataFrame,
) -> dict[str, Any]:
    municipios = _municipality_summary_records(municipal)
    municipal_series = _municipality_series_records(evolucao_municipal)
    cluster_mun = _with_municipality_key(cluster_municipalities)

    tables = {
        "Pessoas dos clusters": _records_for_js(cluster_personas, max_rows=500),
        "Pessoas dos clusters - somente eleitores": _records_for_js(cluster_voter_personas, max_rows=500),
        "Predicao 2026 por cluster": _records_for_js(cluster_prediction, max_rows=800),
        "Clusters por municipio": _records_for_js(cluster_mun, max_rows=1200),
        "Clusters por municipio - somente eleitores": _records_for_js(_with_municipality_key(cluster_voter_municipalities), max_rows=1200),
        "Valores discriminantes": _records_for_js(cluster_discriminants, max_rows=800),
        "Valores discriminantes - somente eleitores": _records_for_js(cluster_voter_discriminants, max_rows=800),
        "Cotovelo dos clusters": _records_for_js(cluster_elbow, max_rows=80),
        "Cotovelo dos clusters - somente eleitores": _records_for_js(cluster_voter_elbow, max_rows=80),
        "Timeline nacional": _records_for_js(timeline_nacional, max_rows=300),
        "Timeline por UF": _records_for_js(timeline_uf, max_rows=900),
        "Respostas eleitorais": _records_for_js(respostas, max_rows=80),
        "Perfil por ano": _records_for_js(perfil_ano, max_rows=500),
        "Perfil por partido": _records_for_js(perfil_partido, max_rows=500),
        "Perfil por candidato": _records_for_js(perfil_candidato, max_rows=500),
        "Perfil do candidato correlacionado": _records_for_js(perfil_do_candidato, max_rows=500),
        "Resultado + eleitorado correlacionado": _records_for_js(resultado_eleitorado, max_rows=500),
        "Comparativo anual perfil eleitor": _records_for_js(comparativo_perfil, max_rows=500),
        "Comparativo anual perfil partido": _records_for_js(comparativo_partido, max_rows=500),
        "Comparativo anual perfil candidato": _records_for_js(comparativo_candidato, max_rows=500),
        "Top 10 perfis por nivel": _records_for_js(top10_perfis, max_rows=500),
        "Vencedor por secao": _records_for_js(vencedor_secao, max_rows=500),
        "Vencedor por secao explicado": _records_for_js(vencedor_secao_exp, max_rows=500),
        "Quem vota em quem": _records_for_js(quem_vota, max_rows=500),
        "Afinidade perfil-candidato": _records_for_js(afinidade, max_rows=500),
        "Clusters comportamentais": _records_for_js(comportamento_interp, max_rows=500),
    }
    return {
        "municipios": municipios,
        "municipalSeries": municipal_series,
        "municipalClusters": _records_for_js(cluster_mun, max_rows=200000),
        "clusterPersonas": _records_for_js(cluster_personas, max_rows=200),
        "clusterVoterPersonas": _records_for_js(cluster_voter_personas, max_rows=200),
        "clusterPrediction": _records_for_js(cluster_prediction, max_rows=1000),
        "clusterElbow": _records_for_js(cluster_elbow, max_rows=80),
        "clusterSummary": _records_for_js(cluster_summary, max_rows=500),
        "clusterDiscriminants": _records_for_js(cluster_discriminants, max_rows=1000),
        "clusterVoterDiscriminants": _records_for_js(cluster_voter_discriminants, max_rows=1000),
        "clusterYearRegion": _records_for_js(cluster_voter_year_region, max_rows=1200),
        "clusterResultYearRegion": _records_for_js(cluster_year_region, max_rows=1200),
        "clusterEntities": _records_for_js(cluster_entities, max_rows=1000),
        "perfilAno": _records_for_js(perfil_ano, max_rows=1000),
        "perfilPartido": _records_for_js(perfil_partido, max_rows=1000),
        "timelineUf": _records_for_js(timeline_uf, max_rows=1000),
        "timelineNacional": _records_for_js(timeline_nacional, max_rows=500),
        "respostas": _records_for_js(respostas, max_rows=100),
        "perfilCandidato": _records_for_js(perfil_candidato, max_rows=1000),
        "perfilDoCandidato": _records_for_js(perfil_do_candidato, max_rows=1000),
        "resultadoEleitorado": _records_for_js(resultado_eleitorado, max_rows=1000),
        "comparativoPerfil": _records_for_js(comparativo_perfil, max_rows=1000),
        "comparativoPartido": _records_for_js(comparativo_partido, max_rows=1000),
        "comparativoCandidato": _records_for_js(comparativo_candidato, max_rows=1000),
        "top10Perfis": _records_for_js(top10_perfis, max_rows=1200),
        "vencedorSecao": _records_for_js(vencedor_secao, max_rows=500),
        "vencedorSecaoExp": _records_for_js(vencedor_secao_exp, max_rows=500),
        "quemVota": _records_for_js(quem_vota, max_rows=500),
        "afinidade": _records_for_js(afinidade, max_rows=500),
        "comportamentoInterp": _records_for_js(comportamento_interp, max_rows=500),
        "tabelas": tables,
    }


def _municipality_summary_records(municipal: pd.DataFrame) -> list[dict[str, Any]]:
    if municipal is None or municipal.empty:
        return []
    df = municipal.copy()
    for col in ["uf", "cd_municipio", "nm_municipio", "ano", "lider"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].map(lambda x: safe_text(x, ""))
    for col in ["votos", "pct_abstencao", "pct_comparecimento"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["_municipio_key"] = df.apply(_municipality_key_from_row, axis=1)
    grouped = df.groupby("_municipio_key", dropna=False).agg(
        uf=("uf", "first"),
        cd_municipio=("cd_municipio", "first"),
        nm_municipio=("nm_municipio", "first"),
        votos=("votos", "sum"),
        abstencao_media=("pct_abstencao", "mean"),
        comparecimento_medio=("pct_comparecimento", "mean"),
        anos=("ano", lambda s: ", ".join(sorted(set(x for x in map(str, s) if safe_text(x))))),
        lider=("lider", lambda s: _dominant_text(s)),
    ).reset_index().rename(columns={"_municipio_key": "key"})
    grouped["label"] = grouped.apply(
        lambda r: " - ".join([x for x in [safe_text(r.get("uf")), safe_text(r.get("nm_municipio")) or safe_text(r.get("cd_municipio"))] if x]),
        axis=1,
    )
    grouped = grouped.sort_values(["uf", "nm_municipio", "cd_municipio"])
    return _records_for_js(grouped, max_rows=200000)


def _municipality_series_records(evolucao_municipal: pd.DataFrame) -> list[dict[str, Any]]:
    if evolucao_municipal is None or evolucao_municipal.empty:
        return []
    cols = [
        "ano", "uf", "cd_municipio", "nm_municipio", "cargo", "turno",
        "votos", "pct_abstencao", "pct_comparecimento",
        "lider", "votos_lider",
    ]
    df = evolucao_municipal.copy()
    for col in cols:
        if col not in df.columns:
            df[col] = ""
    df["_municipio_key"] = df.apply(_municipality_key_from_row, axis=1)
    cols = ["_municipio_key"] + cols
    return _records_for_js(df[cols].rename(columns={"_municipio_key": "key"}), max_rows=200000)


def _with_municipality_key(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for col in ["uf", "cd_municipio", "nm_municipio"]:
        if col not in out.columns:
            out[col] = ""
    out["_municipio_key"] = out.apply(_municipality_key_from_row, axis=1)
    return out.rename(columns={"_municipio_key": "key"})


def _municipality_key_from_row(row: pd.Series) -> str:
    uf = safe_text(row.get("uf", ""))
    code = safe_text(row.get("cd_municipio", ""))
    name = safe_text(row.get("nm_municipio", ""))
    return f"{uf}|{code or name}".strip("|")


def _dominant_text(series: pd.Series) -> str:
    vals = [safe_text(x, "") for x in series if safe_text(x, "")]
    if not vals:
        return ""
    counts = pd.Series(vals).value_counts()
    return safe_text(counts.index[0], "")


def _records_for_js(df: pd.DataFrame | None, max_rows: int | None = None) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    out = df.copy()
    if max_rows is not None:
        out = out.head(max_rows)
    records = out.to_dict(orient="records")
    return [_clean_json_record(r) for r in records]


def _clean_json_record(record: dict[str, Any]) -> dict[str, Any]:
    return {str(k): _clean_json_value(v) for k, v in record.items()}


def _clean_json_value(value: Any) -> Any:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _dashboard_script(payload: dict[str, Any]) -> str:
    payload_json = json.dumps(payload, ensure_ascii=False, allow_nan=False, default=str)
    script = r"""
<script>
window.DASHBOARD_DATA = __PAYLOAD__;

function fmtInt(v) {
  const n = Number(v || 0);
  if (!Number.isFinite(n)) return '0';
  return Math.round(n).toLocaleString('pt-BR');
}
function fmtPct(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return 'sem dado';
  return (n * 100).toLocaleString('pt-BR', {maximumFractionDigits: 1}) + '%';
}
function esc(v) {
  return String(v ?? '').replace(/[&<>"']/g, s => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[s]));
}
function numeric(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n : 0;
}
function meaningful(v) {
  const s = String(v ?? '').trim();
  if (!s) return '';
  const lower = s.toLowerCase();
  const code = lower.replace('codigo ', '').replace('código ', '').replace('.', '').replace(/^[-+]/, '');
  if (['sem valor','sem_valor','nan','none','null','<na>','#nulo#','geral','nao informado','não informado','sem_entidade'].includes(lower)) return '';
  if (lower.endsWith('_sem_valor') || lower.endsWith(' sem valor')) return '';
  if ((lower.startsWith('codigo ') || lower.startsWith('código ')) && /^[0-9]+$/.test(code)) return '';
  return s;
}
function firstMeaningful(row, keys) {
  for (const key of keys) {
    const value = meaningful(row?.[key]);
    if (value) return value;
  }
  return '';
}
function localLabel(row) {
  return [meaningful(row?.uf), meaningful(row?.nm_municipio) || meaningful(row?.cd_municipio), meaningful(row?.zona), meaningful(row?.secao)].filter(Boolean).join(' / ');
}
function hasClusterProfile(r) {
  return ['perfil_faixa_etaria_dominante','perfil_genero_dominante','perfil_instrucao_dominante','perfil_estado_civil_dominante','perfil_raca_cor_dominante']
    .some(k => meaningful(r?.[k]));
}
function clusterTraitChips(r) {
  const traits = [
    ['Faixa', 'perfil_faixa_etaria_dominante'],
    ['Sexo', 'perfil_genero_dominante'],
    ['Escolaridade', 'perfil_instrucao_dominante'],
    ['Estado civil', 'perfil_estado_civil_dominante'],
    ['Raca/cor', 'perfil_raca_cor_dominante'],
    ['Regiao', 'regiao_dominante'],
    ['UF', 'uf_dominante']
  ];
  return traits.map(([label, key]) => {
    const value = meaningful(r?.[key]);
    return value ? `<span class="pill">${esc(label)}: ${esc(value)}</span>` : '';
  }).join('');
}
function clusterTitle(r, opts = {}) {
  const age = meaningful(r?.perfil_faixa_etaria_dominante);
  const gender = meaningful(r?.perfil_genero_dominante);
  const education = meaningful(r?.perfil_instrucao_dominante);
  const titleBits = [age, gender, education].filter(Boolean).slice(0, 3);
  if (titleBits.length) return titleBits.join(' · ');
  return opts.result ? 'Perfil eleitoral do cluster' : 'Perfil do eleitor do cluster';
}
function clusterPoliticalSentence(r) {
  const pred = meaningful(r?.entidade_prevista_2026);
  const winner = meaningful(r?.vencedor_setor_dominante);
  const party = meaningful(r?.partido_vencedor_setor_dominante);
  if (pred) return `Tendencia projetada para 2026: ${pred}.`;
  if (winner) return `Historicamente se aproxima de ${winner}.`;
  if (party) return `Partido mais associado: ${party}.`;
  return 'Sem tendencia politica dominante confiavel.';
}
function uniqueBy(rows, keyFn) {
  const seen = new Set();
  return rows.filter(r => {
    const k = keyFn(r);
    if (seen.has(k)) return false;
    seen.add(k);
    return true;
  });
}
function renderCards(id, rows, cardFn, limit = 24) {
  const el = document.getElementById(id);
  if (!el) return;
  const clean = rows.slice(0, limit);
  el.innerHTML = clean.length ? clean.map(cardFn).join('') : "<p class='smallnote'>Sem dados suficientes para cards.</p>";
}
function renderBars(id, rows, labelFn, valueFn, opts = {}) {
  const el = document.getElementById(id);
  if (!el) return;
  const clean = rows.filter(r => Number.isFinite(Number(valueFn(r)))).slice(0, opts.limit || 18);
  const labels = new Set(clean.map(r => meaningful(labelFn(r))).filter(Boolean));
  if (!clean.length) {
    el.innerHTML = "<p class='smallnote'>Sem dados suficientes para o grafico.</p>";
    return;
  }
  if (!opts.allowSingle && labels.size < 2) {
    el.innerHTML = "<p class='smallnote'>Grafico omitido: existe apenas uma categoria/ano util neste recorte.</p>";
    return;
  }
  const max = Math.max(...clean.map(r => Math.abs(Number(valueFn(r))) || 0), 1);
  el.innerHTML = clean.map(r => {
    const value = Number(valueFn(r)) || 0;
    const width = Math.max(2, Math.min(100, Math.abs(value) / max * 100));
    const textValue = opts.percent ? fmtPct(value) : fmtInt(value);
    return `<div class="bar-row"><div class="bar-label" title="${esc(labelFn(r))}">${esc(labelFn(r))}</div><div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div><div>${esc(textValue)}</div></div>`;
  }).join('');
}
function renderSimpleTable(id, rows, limit = 60) {
  const el = document.getElementById(id);
  if (!el) return;
  const data = rows.slice(0, limit);
  if (!data.length) {
    el.innerHTML = "<p class='smallnote'>Sem linhas para mostrar.</p>";
    return;
  }
  const cols = Object.keys(data[0]).slice(0, 12);
  el.innerHTML = `<table><thead><tr>${cols.map(c => `<th>${esc(c)}</th>`).join('')}</tr></thead><tbody>${data.map(r => `<tr>${cols.map(c => `<td>${esc(r[c])}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
}
function aggregate(rows, keyField, valueField) {
  const m = new Map();
  rows.forEach(r => {
    const k = meaningful(r[keyField]);
    if (!k) return;
    m.set(k, (m.get(k) || 0) + numeric(r[valueField]));
  });
  return [...m.entries()].map(([label, value]) => ({label, value})).sort((a,b) => b.value - a.value);
}
const UF_LAYOUT = [
  ['AC',13,58],['RO',24,62],['AM',28,39],['RR',34,20],['AP',53,22],['PA',49,39],['TO',54,52],
  ['MA',63,43],['PI',68,50],['CE',74,46],['RN',80,49],['PB',79,53],['PE',77,57],['AL',76,61],['SE',74,65],['BA',68,67],
  ['MT',47,63],['MS',52,76],['GO',60,69],['DF',63,71],['MG',66,77],['ES',73,78],['RJ',70,84],['SP',62,84],
  ['PR',58,89],['SC',60,94],['RS',57,98]
];
function latestByYear(rows) {
  const years = rows.map(r => numeric(r.ano)).filter(Number.isFinite);
  const latest = years.length ? Math.max(...years) : null;
  return latest === null ? rows : rows.filter(r => numeric(r.ano) === latest);
}
function profileSummary(row) {
  return meaningful(row?.descricao) || meaningful(row?.perfil_combinado) || meaningful(row?.pessoa_do_partido) || meaningful(row?.pessoa_do_candidato) || meaningful(row?.perfil_predominante) || 'Perfil ainda sem descricao consolidada.';
}
function profileChips(row) {
  return [
    meaningful(row?.ano),
    meaningful(row?.uf),
    meaningful(row?.nm_municipio),
    meaningful(row?.entidade),
    meaningful(row?.partido),
    meaningful(row?.candidato),
    meaningful(row?.padrao_temporal)
  ].filter(Boolean).map(v => `<span class="pill">${esc(v)}</span>`).join('');
}
function renderProfileCards(id, rows, limit = 12) {
  renderCards(id, rows, r => `
    <details class="persona">
      <summary>${esc(meaningful(r.perfil_combinado) || meaningful(r.entidade) || meaningful(r.partido) || meaningful(r.candidato) || 'Perfil')}</summary>
      <p>${esc(profileSummary(r))}</p>
      <p>${profileChips(r)}</p>
      <p class="smallnote">Eleitorado: ${fmtInt(r.eleitorado)} | Votos: ${fmtInt(r.votos)} | Share: ${fmtPct(r.share_perfil || r.share_perfil_na_entidade)}</p>
    </details>`, limit);
}
function renderBrazilMap() {
  const data = window.DASHBOARD_DATA || {};
  const holder = document.getElementById('brasilMap');
  if (!holder) return;
  const top = data.top10Perfis || [];
  const comp = data.comparativoPerfil || [];
  const ufProfile = uf => {
    const stateTop = latestByYear(top.filter(r => String(r.nivel || '').toLowerCase() === 'estado' && meaningful(r.uf) === uf));
    const stateComp = latestByYear(comp.filter(r => String(r.nivel || '').toLowerCase() === 'estado' && meaningful(r.uf) === uf));
    return stateTop[0] || stateComp[0] || {};
  };
  holder.innerHTML = `<div class="br-map-wrap"><div class="br-map" id="brMapCanvas"></div><div class="state-detail" id="brMapDetail"></div></div>`;
  const canvas = document.getElementById('brMapCanvas');
  canvas.innerHTML = UF_LAYOUT.map(([uf,x,y]) => `<button type="button" class="uf-dot" data-uf="${uf}" style="left:${x}%;top:${y}%">${uf}</button>`).join('');
  const detail = document.getElementById('brMapDetail');
  const update = uf => {
    document.querySelectorAll('.uf-dot').forEach(b => b.classList.toggle('active', b.dataset.uf === uf));
    const row = ufProfile(uf);
    const summary = profileSummary(row);
    detail.innerHTML = `
      <h3>${esc(uf)}</h3>
      <p>${esc(summary)}</p>
      <p>${profileChips(row)}</p>
      <p class="smallnote">Clique em outro estado para consultar o perfil dominante daquele recorte.</p>`;
  };
  canvas.querySelectorAll('.uf-dot').forEach(btn => btn.addEventListener('mouseenter', () => update(btn.dataset.uf)));
  canvas.querySelectorAll('.uf-dot').forEach(btn => btn.addEventListener('click', () => update(btn.dataset.uf)));
  update('SP');
}
function populateMunicipios() {
  const data = window.DASHBOARD_DATA || {};
  const select = document.getElementById('municipioSelect');
  const search = document.getElementById('municipioSearch');
  if (!select) return;
  const old = select.value;
  const term = (search?.value || '').toLowerCase();
  const municipios = (data.municipios || []).filter(m => {
    const hay = `${m.label || ''} ${m.uf || ''} ${m.cd_municipio || ''}`.toLowerCase();
    return !term || hay.includes(term);
  });
  select.innerHTML = municipios.map(m => `<option value="${esc(m.key)}">${esc(m.label || m.key)}</option>`).join('');
  if (old && municipios.some(m => m.key === old)) select.value = old;
}
function metric(label, value, note = '') {
  return `<div class="metric"><span>${esc(label)}</span><strong>${esc(value)}</strong><span>${esc(note)}</span></div>`;
}
function renderMunicipioSelecionado() {
  const data = window.DASHBOARD_DATA || {};
  const select = document.getElementById('municipioSelect');
  if (!select) return;
  const key = select.value || (data.municipios?.[0]?.key || '');
  const mun = (data.municipios || []).find(m => m.key === key) || {};
  const series = (data.municipalSeries || []).filter(r => r.key === key);
  const clusters = (data.municipalClusters || []).filter(r => r.key === key);
  const cards = document.getElementById('municipioCards');
  if (cards) {
    const clusterTop = [...clusters].sort((a,b) => numeric(b.qtd_setores) - numeric(a.qtd_setores))[0] || {};
    cards.innerHTML = [
      metric('Municipio', mun.label || key, mun.anos ? `anos: ${mun.anos}` : ''),
      metric('Votos acumulados', fmtInt(mun.votos), 'soma dos recortes carregados'),
      metric('Abstencao media', fmtPct(mun.abstencao_media), 'media dos recortes'),
      metric('Comparecimento medio', fmtPct(mun.comparecimento_medio), 'media dos recortes'),
      metric('Cluster principal', clusterTop.cluster_global_discriminado ?? 'sem cluster', clusterTop.regiao || '')
    ].join('');
  }
  const yearly = aggregate(series, 'ano', 'votos').sort((a,b) => String(a.label).localeCompare(String(b.label)));
  renderBars('municipioTimeline', yearly, r => r.label, r => r.value, {limit: 24});
  const clusterRows = clusters.map(r => ({label: `Cluster ${r.cluster_global_discriminado} | ${r.regiao || ''}`, value: numeric(r.qtd_setores)})).sort((a,b) => b.value-a.value);
  renderBars('municipioClusters', clusterRows, r => r.label, r => r.value, {limit: 16});
  const munTop = latestByYear((data.top10Perfis || []).filter(r => {
    if (String(r.nivel || '').toLowerCase() !== 'municipio') return false;
    const code = meaningful(mun.cd_municipio);
    const name = meaningful(mun.nm_municipio);
    return (code && meaningful(r.cd_municipio) === code) || (name && meaningful(r.nm_municipio) === name);
  }));
  renderProfileCards('top10MunicipioCards', munTop, 10);
  renderCards('municipioRows', series, r => `
    <details class="persona">
      <summary>${esc(r.ano || '')} ${esc(r.cargo || '')} ${esc(r.turno || '')}</summary>
      <p><span class="pill">Votos: ${fmtInt(r.votos)}</span><span class="pill">Abstencao: ${fmtPct(r.pct_abstencao)}</span><span class="pill">Comparecimento: ${fmtPct(r.pct_comparecimento)}</span></p>
      <p class="smallnote">Lider: ${esc(firstMeaningful(r, ['lider','vencedor','entidade','partido','candidato']) || 'sem lider calculado')}</p>
    </details>`, 36);
}
function renderBrasil() {
  const data = window.DASHBOARD_DATA || {};
  renderBrazilMap();
  const perfil = (data.perfilAno || []).filter(r => meaningful(r.valor_perfil) && !String(r.dimensao_perfil || '').toLowerCase().includes('biometr'));
  const latestAno = Math.max(...perfil.map(r => numeric(r.ano)).filter(Number.isFinite), 0);
  const perfilLatest = perfil.filter(r => !latestAno || numeric(r.ano) === latestAno);
  const dims = {};
  perfilLatest.forEach(r => {
    const dim = meaningful(r.dimensao_perfil);
    if (!dim) return;
    const value = numeric(r.eleitorado);
    if (!dims[dim] || value > numeric(dims[dim].eleitorado)) dims[dim] = r;
  });
  const chips = Object.values(dims).map(r => `<span class="pill">${esc(r.dimensao_perfil)}: ${esc(r.valor_perfil)}</span>`).join('');
  renderCards('brasilEleitorMedio', [{
    title: latestAno ? `Eleitor medio nacional ${latestAno}` : 'Eleitor medio nacional',
    body: chips || 'Perfil ainda insuficiente nos JSONs processados.'
  }], r => `<div class="persona"><h4>${esc(r.title)}</h4><p>${r.body}</p></div>`, 1);
  const perfilBars = Object.values(dims).map(r => ({label:`${r.dimensao_perfil}: ${r.valor_perfil}`, value:numeric(r.share_eleitorado_ano)}));
  renderBars('brasilPerfilChart', perfilBars, r => r.label, r => r.value, {percent:true, limit:12});
  renderBars('brasilVotosUf', aggregate(data.timelineUf || [], 'uf', 'votos').filter(r => meaningful(r.label)).slice(0, 27), r => r.label, r => r.value, {limit:27});
  const partidos = (data.perfilPartido || []).filter(r => meaningful(r.partido)).sort((a,b) => numeric(b.votos_partido)-numeric(a.votos_partido));
  renderCards('brasilPartidos', partidos, r => `
    <div class="persona">
      <h4>${esc(r.partido)} ${meaningful(r.ano) ? `- ${esc(r.ano)}` : ''}</h4>
      <p>${esc(r.pessoa_do_partido || 'Perfil indisponivel')}</p>
      <p class="smallnote">Votos: ${fmtInt(r.votos_partido)}</p>
    </div>`, 18);
  const candidatos = (data.perfilCandidato || []).filter(r => meaningful(r.candidato) || meaningful(r.entidade)).sort((a,b) => numeric(b.votos_candidato || b.votos)-numeric(a.votos_candidato || a.votos));
  renderCards('brasilCandidatos', candidatos, r => `
    <details class="persona">
      <summary>${esc(meaningful(r.candidato) || meaningful(r.entidade) || 'Candidato')}</summary>
      <p>${esc(r.pessoa_do_candidato || r.pessoa_do_entidade || profileSummary(r))}</p>
      <p>${profileChips(r)}</p>
      <p class="smallnote">Votos: ${fmtInt(r.votos_candidato || r.votos)}</p>
    </details>`, 18);
  renderProfileCards('top10FederacaoCards', latestByYear((data.top10Perfis || []).filter(r => String(r.nivel || '').toLowerCase() === 'brasil')), 10);
  const nacional = data.timelineNacional || [];
  const totals = [
    {label:'Votos', value:nacional.reduce((s,r)=>s+numeric(r.votos),0)},
    {label:'Abstencao', value:nacional.reduce((s,r)=>s+numeric(r.abstencao_estimado),0)},
    {label:'Comparecimento', value:nacional.reduce((s,r)=>s+numeric(r.comparecimento_estimado),0)}
  ];
  renderBars('brasilTimeline', totals, r => r.label, r => r.value, {limit:3});
  renderCards('brasilRespostas', data.respostas || [], r => `
    <details class="persona" open>
      <summary>${esc(r.pergunta || 'Pergunta')}</summary>
      <p>${esc(r.resposta || '')}</p>
      <p class="smallnote">${esc(r.base_de_evidencia || '')}</p>
    </details>`, 8);
}
function renderEstadoSecaoCards() {
  const data = window.DASHBOARD_DATA || {};
  renderCards('estadoUfCards', data.timelineUf || [], r => `
    <details class="persona">
      <summary>${esc(meaningful(r.uf) || 'UF')} ${meaningful(r.ano) ? '- ' + esc(r.ano) : ''}</summary>
      <p><span class="pill">Votos: ${fmtInt(r.votos)}</span><span class="pill">Abstencao: ${fmtInt(r.abstencao_estimado)}</span><span class="pill">Comparecimento: ${fmtInt(r.comparecimento_estimado)}</span></p>
      <p class="smallnote">${esc(r.cargo || '')} ${esc(r.turno || '')}</p>
    </details>`, 48);
  renderProfileCards('top10EstadoCards', latestByYear((data.top10Perfis || []).filter(r => String(r.nivel || '').toLowerCase() === 'estado')), 30);
  renderProfileCards('estadoPerfilCards', latestByYear((data.comparativoPerfil || []).filter(r => String(r.nivel || '').toLowerCase() === 'estado')), 30);
  renderCards('secaoVencedorCards', (data.vencedorSecao || []).filter(r => meaningful(firstMeaningful(r, ['vencedor_secao','entidade','candidato','partido']))), r => `
    <details class="persona">
      <summary>${esc(localLabel(r) || 'Secao')}</summary>
      <p><strong>${esc(firstMeaningful(r, ['vencedor_secao','entidade','candidato','partido']))}</strong></p>
      <p><span class="pill">Votos: ${fmtInt(r.votos_vencedor || r.votos)}</span><span class="pill">Share: ${fmtPct(r.share_vencedor || r.share)}</span><span class="pill">Margem: ${fmtPct(r.margem_share)}</span></p>
      <p class="smallnote">${esc(r.como_ganhou || '')}</p>
    </details>`, 36);
  renderCards('secaoExpCards', data.vencedorSecaoExp || [], r => `
    <details class="persona">
      <summary>${esc(localLabel(r) || firstMeaningful(r, ['vencedor_secao','entidade']) || 'Explicacao')}</summary>
      <p>${esc(firstMeaningful(r, ['explicacao','motivo_vitoria','como_ganhou','interpretacao']) || firstMeaningful(r, ['vencedor_secao','entidade']) || '')}</p>
    </details>`, 24);
  renderCards('quemVotaCards', data.quemVota || [], r => `
    <details class="persona">
      <summary>${esc(firstMeaningful(r, ['entidade','candidato','partido','entidade_voto']) || 'Entidade')}</summary>
      <p>${esc(firstMeaningful(r, ['perfil_predominante','perfil','pessoa_do_cluster','perfil_eleitor']) || 'Perfil agregado disponivel no CSV.')}</p>
      <p class="smallnote">Votos: ${fmtInt(r.votos || r.votos_perfil || r.votos_proxy_perfil_entidade)}</p>
    </details>`, 24);
  renderCards('afinidadeCards', data.afinidade || [], r => `
    <details class="persona">
      <summary>${esc(firstMeaningful(r, ['entidade','candidato','partido','entidade_voto']) || 'Afinidade')}</summary>
      <p>${esc(firstMeaningful(r, ['perfil','valor_perfil','perfil_predominante','dimensao_perfil']) || '')}</p>
      <p><span class="pill">Lift: ${Number(numeric(r.lift_perfil_entidade_proxy || r.lift || r.lift_vs_global)).toFixed(2)}x</span><span class="pill">Share: ${fmtPct(r.share_proxy_no_perfil || r.share)}</span></p>
    </details>`, 24);
  renderCards('perfilCandidatoCards', data.perfilDoCandidato || [], r => `
    <details class="persona">
      <summary>${esc(firstMeaningful(r, ['candidato','entidade']) || 'Candidato')}</summary>
      <p>${esc(firstMeaningful(r, ['perfil_eleitor_associado','perfil_predominante','pessoa_do_candidato','perfil_resumido']) || profileSummary(r))}</p>
      <p>${profileChips(r)}</p>
      <p class="smallnote">Partido: ${esc(meaningful(r.partido) || 'sem partido calculado')} | Votos: ${fmtInt(r.votos || r.votos_candidato)}</p>
    </details>`, 24);
  renderCards('resultadoEleitoradoCards', data.resultadoEleitorado || [], r => `
    <details class="persona">
      <summary>${esc(firstMeaningful(r, ['entidade','partido','candidato']) || 'Resultado correlacionado')}</summary>
      <p>${esc(profileSummary(r))}</p>
      <p>${profileChips(r)}</p>
      <p class="smallnote">Votos: ${fmtInt(r.votos || r.votos_perfil_entidade)} | Share: ${fmtPct(r.share || r.share_proxy_no_perfil)}</p>
    </details>`, 24);
  renderCards('comportamentoClusterCards', data.comportamentoInterp || [], r => `
    <details class="persona">
      <summary>Cluster ${esc(r.cluster_comportamento_eleitoral ?? r.cluster ?? '')}</summary>
      <p>${esc(firstMeaningful(r, ['interpretacao','pessoa_do_cluster','descricao']) || 'Resumo comportamental disponivel no CSV.')}</p>
      <p class="smallnote">Linhas: ${fmtInt(r.qtd_linhas)} | Votos: ${fmtInt(r.votos_cluster || r.votos)}</p>
    </details>`, 24);
}
function renderClusterList(id, rows, opts = {}) {
  const filtered = uniqueBy((rows || []).filter(r => meaningful(r.pessoa_do_cluster) && hasClusterProfile(r)), r => `${r.cluster_global_discriminado}|${r.pessoa_do_cluster}|${r.entidade_prevista_2026 || ''}`);
  renderCards(id, filtered, r => {
    return `<div class="persona">
      <h4>${esc(clusterTitle(r, opts))}</h4>
      <p>${esc(r.pessoa_do_cluster)}</p>
      <p>${clusterTraitChips(r)}</p>
      ${opts.result ? `<p class="smallnote">${esc(clusterPoliticalSentence(r))}</p>` : `<p class="smallnote">Perfil formado somente por caracteristicas discretas do eleitor.</p>`}
    </div>`;
  }, 24);
}
function renderClusterConsultas() {
  const data = window.DASHBOARD_DATA || {};
  renderClusterList('clusterVoterPersonas', data.clusterVoterPersonas || [], {result:false});
  renderClusterList('clusterPersonas', data.clusterPersonas || [], {result:true});
  const personaByCluster = new Map((data.clusterPersonas || []).filter(r => meaningful(r.pessoa_do_cluster) && hasClusterProfile(r)).map(r => [String(r.cluster_global_discriminado), r]));
  const predCards = uniqueBy(
    (data.clusterPrediction || [])
      .filter(r => numeric(r.rank_pred_2026_cluster) <= 3 && meaningful(r.vencedor_setor) && personaByCluster.has(String(r.cluster_global_discriminado)))
      .sort((a,b) => numeric(a.cluster_global_discriminado)-numeric(b.cluster_global_discriminado) || numeric(a.rank_pred_2026_cluster)-numeric(b.rank_pred_2026_cluster)),
    r => `${r.cluster_global_discriminado}|${r.vencedor_setor}`
  );
  renderCards('clusterPrediction', predCards, r => {
    const persona = personaByCluster.get(String(r.cluster_global_discriminado)) || {};
    return `<div class="persona">
      <h4>${esc(r.vencedor_setor)}</h4>
      <p>${esc(meaningful(persona.pessoa_do_cluster))}</p>
      <p>${clusterTraitChips(persona)}</p>
      <p class="smallnote">Tendencia eleitoral projetada a partir do historico discreto desse perfil.</p>
    </div>`;
  }, 36);
  renderBars('clusterElbow', data.clusterElbow || [], r => `k=${r.k}`, r => numeric(r.inercia), {limit:20});
  renderCards('clusterAnoRegiaoCards', data.clusterResultYearRegion || [], r => `
    <details class="persona">
      <summary>Cluster ${esc(r.cluster_global_discriminado)} - ${esc(r.ano_correlacao)} - ${esc(r.regiao)} / ${esc(r.uf)}</summary>
      <p><span class="pill">Setores: ${fmtInt(r.qtd_setores)}</span><span class="pill">Votos: ${fmtInt(r.votos_total)}</span></p>
      <p><span class="pill">Abstencao: ${fmtPct(r.abstencao_media)}</span><span class="pill">Comparecimento: ${fmtPct(r.comparecimento_medio)}</span></p>
    </details>`, 48);
  renderCards('clusterDiscriminantsCards', (data.clusterDiscriminants || []).filter(r => meaningful(r.valor_discriminado)), r => `
    <div class="persona">
      <h4>${esc(r.campo_discriminado_legivel || r.campo_discriminado)}</h4>
      <p>Esse cluster se diferencia principalmente por <strong>${esc(r.valor_discriminado)}</strong>.</p>
      <p class="smallnote">Este atributo aparece com mais forca neste grupo do que na base geral.</p>
    </div>`, 24);
  renderCards('clusterEntitiesCards', (data.clusterEntities || []).filter(r => meaningful(r.vencedor_setor)), r => `
    <div class="persona">
      <h4>${esc(r.vencedor_setor)}</h4>
      <p>Entidade mais recorrente nos resultados associados a esse perfil de eleitor.</p>
      <p class="smallnote">Use a aba de consulta para ver quantidades e detalhes tabulares.</p>
    </div>`, 24);
}
function renderClusterCards() {
  renderClusterConsultas();
}
function populateTables() {
  const select = document.getElementById('tableSelect');
  const tables = (window.DASHBOARD_DATA || {}).tabelas || {};
  if (!select) return;
  select.innerHTML = Object.keys(tables).map(k => `<option value="${esc(k)}">${esc(k)}</option>`).join('');
}
function renderTabelaSelecionada() {
  const data = window.DASHBOARD_DATA || {};
  const name = document.getElementById('tableSelect')?.value || Object.keys(data.tabelas || {})[0];
  const term = (document.getElementById('tableSearch')?.value || '').toLowerCase();
  const rows = ((data.tabelas || {})[name] || []).filter(r => !term || Object.values(r).join(' ').toLowerCase().includes(term));
  renderSimpleTable('tablePreview', rows, 80);
  if (!rows.length) {
    renderBars('tableChart', [], r => '', r => 0);
    return;
  }
  const numericPriority = ['votos_total_cluster','votos_total','total_votos_cluster','votos','abstencao_media_cluster','taxa_abstencao_media','pct_abstencao','abstencao_media','comparecimento_medio_cluster','taxa_comparecimento','pct_comparecimento','comparecimento_medio','qtd_setores','qtd_municipios','share_previsto_2026'];
  const valueField = numericPriority.find(c => rows.some(r => Number.isFinite(Number(r[c]))));
  const labelField = Object.keys(rows[0]).find(c => c !== valueField && rows.some(r => r[c] !== null && r[c] !== undefined && String(r[c]).trim() !== '')) || Object.keys(rows[0])[0];
  if (valueField) renderBars('tableChart', rows.slice(0, 40), r => `${r[labelField]}`, r => numeric(r[valueField]), {percent: valueField.includes('share') || valueField.includes('pct') || valueField.includes('media')});
  else document.getElementById('tableChart').innerHTML = "<p class='smallnote'>Esta tabela nao tem metrica numerica clara para grafico.</p>";
}
function initDashboardConsultas() {
  populateMunicipios();
  document.getElementById('municipioSearch')?.addEventListener('input', () => { populateMunicipios(); renderMunicipioSelecionado(); });
  document.getElementById('municipioSelect')?.addEventListener('change', renderMunicipioSelecionado);
  renderBrasil();
  renderEstadoSecaoCards();
  renderMunicipioSelecionado();
  renderClusterCards();
  populateTables();
  document.getElementById('tableSearch')?.addEventListener('input', renderTabelaSelecionada);
  document.getElementById('tableSelect')?.addEventListener('change', renderTabelaSelecionada);
  renderTabelaSelecionada();
}
window.initDashboardConsultas = initDashboardConsultas;
window.renderMunicipioSelecionado = renderMunicipioSelecionado;
window.renderTabelaSelecionada = renderTabelaSelecionada;
</script>
"""
    return script.replace("__PAYLOAD__", payload_json)


def _nunique(df: pd.DataFrame, col: str) -> int:
    if df is None or df.empty or col not in df.columns:
        return 0
    return int(df[col].replace("", pd.NA).dropna().nunique())


def _sum_as_int(df: pd.DataFrame, col: str) -> str:
    if df is None or df.empty or col not in df.columns:
        return "0"
    value = pd.to_numeric(df[col], errors="coerce").fillna(0).sum()
    return f"{int(value):,}".replace(",", ".")


def _years_from_inventory(inventory: pd.DataFrame) -> int:
    if inventory is None or inventory.empty:
        return 0
    years = set()
    for col in ["anos_detectados_conteudo", "anos_detectados_nome"]:
        for value in inventory.get(col, pd.Series(dtype=str)).fillna("").astype(str):
            years.update(__import__("re").findall(r"(?:19|20)\d{2}", value))
    return len(years)


def global_story(
    profiles: pd.DataFrame,
    inventory: pd.DataFrame,
    sim_tables: pd.DataFrame,
    sim_fields: pd.DataFrame,
    corr_temporal: pd.DataFrame,
    corr_entities: pd.DataFrame,
    municipal: pd.DataFrame,
    correlation_outputs: dict[str, Any] | None = None,
    correlation_stats: pd.DataFrame | None = None,
) -> str:
    lines = []
    lines.append("A análise global foi construída a partir dos JSONs efetivamente encontrados. O código não assume anos fixos nem lê ZIP/CSV bruto.")
    lines.append("Primeiro foram inventariados arquivos, campos e anos; depois foram aprendidas similaridades entre tabelas/campos; só então foram montadas timelines e correlações.")

    if inventory is not None and not inventory.empty:
        years = sorted(set(
            int(y)
            for col in ["anos_detectados_conteudo", "anos_detectados_nome"]
            for value in inventory.get(col, pd.Series(dtype=str)).fillna("").astype(str)
            for y in __import__("re").findall(r"(?:19|20)\d{2}", value)
        ))
        lines.append("Anos detectados nos dados/nomes dos arquivos: " + (", ".join(map(str, years)) if years else "nenhum ano detectado explicitamente."))

    if profiles is not None and not profiles.empty:
        lines.append(f"Foram perfilados {profiles['arquivo_relativo'].nunique()} arquivos e {profiles[['arquivo_relativo','coluna']].drop_duplicates().shape[0]} campos.")

    if sim_tables is not None and not sim_tables.empty:
        top = sim_tables.head(5)
        lines.append("Tabelas mais parecidas: " + " | ".join(
            f"{__import__('pathlib').Path(str(r['arquivo_1'])).name} ↔ {__import__('pathlib').Path(str(r['arquivo_2'])).name} ({float(r['score_similaridade']):.2f})"
            for _, r in top.iterrows()
        ))

    if sim_fields is not None and not sim_fields.empty:
        lines.append(f"Foram encontrados {len(sim_fields)} pares de campos potencialmente equivalentes entre arquivos.")

    if False and corr_temporal is not None and not corr_temporal.empty:
        lines.append("Correlações temporais mais fortes: " + " | ".join(
            f"{r['metrica']} {int(r['ano_1'])}→{int(r['ano_2'])}: {float(r['pearson']):.2f}"
            for _, r in corr_temporal.head(5).iterrows()
            if pd.notna(r.get("pearson"))
        ))

    if False and corr_entities is not None and not corr_entities.empty:
        lines.append("Correlação de share de entidades entre anos: " + " | ".join(
            f"{int(r['ano_1'])}→{int(r['ano_2'])}: {float(r['pearson_share']):.2f}"
            for _, r in corr_entities.head(5).iterrows()
            if pd.notna(r.get("pearson_share"))
        ))

    if municipal is not None and not municipal.empty:
        lines.append(f"O retrato municipal global tem {len(municipal)} linhas e {municipal['cd_municipio'].replace('', pd.NA).dropna().nunique() if 'cd_municipio' in municipal.columns else 0} municípios detectados.")

    lines.append("A simulação usa esta global como base de evidência, não como chute isolado.")
    if correlation_outputs:
        years = correlation_outputs.get("anos_correlacionados", [])
        if years:
            lines.append("Parquets correlacionados por ano/codigo gerados para: " + ", ".join(map(str, years)))
        parquet_dir = correlation_outputs.get("parquet_por_ano_dir", "")
        if parquet_dir:
            lines.append(f"Diretorio dos Parquets por ano: {parquet_dir}")

    if correlation_stats is not None and not correlation_stats.empty:
        totals = []
        for _, r in correlation_stats.head(8).iterrows():
            totals.append(
                f"{r.get('ano')}: {r.get('setores_eleitorais')} setores, {r.get('municipios')} municipios, {r.get('linhas_base_correlacionada')} linhas correlacionadas"
            )
        if totals:
            lines.append("Resumo da correlacao anual: " + " | ".join(totals))

    return "\n".join(lines)
