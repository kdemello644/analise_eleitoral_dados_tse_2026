from __future__ import annotations

from pathlib import Path
from typing import Any
import copy
import html
import json
import logging
import traceback

import pandas as pd

from .aggregation import aggregate_full_json, aggregate_full_json_to_parquet_parts, aggregate_records_to_gold
from .spark_engine import aggregate_json_with_pyspark
from .json_reader import classify_json_document, iter_json_records, sample_json_file
from .plots import plot_categorical_frequencies, plot_gold_complete_summary, plot_gold_datasets, plot_numeric_distributions
from .profiler import infer_column_profile
from .stats import correlations, numeric_stats
from .electoral_analysis import run_electoral_analysis
from .discrete import discrete_summary
from .utils import (
    df_to_html,
    extract_years_from_value,
    hash_short,
    img_tag,
    safe_name,
    save_csv,
    save_html,
    save_json,
    save_parquet,
)


def process_file(path: Path, root: Path, cfg) -> dict[str, Any]:
    try:
        rel = path.relative_to(root).as_posix()
    except Exception:
        rel = path.name

    ident = safe_name(rel, 120)
    doc_context = classify_json_document(path)
    base_dir = Path(cfg.out) / "individual" / ident
    tables_dir = base_dir / "tabelas"
    plots_dir = base_dir / "plots"
    parquet_dir = base_dir / "parquet"
    logs_dir = base_dir / "logs"
    for d in [tables_dir, plots_dir, parquet_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    try:
        sample_csv = tables_dir / "amostra.csv.gz"
        profile_csv = tables_dir / "perfil_colunas.csv"
        meta_json = tables_dir / "metadados_amostra.json"
        gold_csv = tables_dir / "gold_individual.csv"
        gold_parquet = parquet_dir / "gold_individual.parquet"
        gold_parts_dir = parquet_dir / "gold_individual_parts"
        gold_parts_manifest = gold_parts_dir / "manifesto_partes_gold.csv"
        gold_storage_info: dict[str, Any] = {}

        if cfg.resume and sample_csv.exists() and profile_csv.exists():
            sample = pd.read_csv(sample_csv, sep=";", compression="gzip", dtype=str)
            profile = pd.read_csv(profile_csv, sep=";", dtype=str)
            meta = json.loads(meta_json.read_text(encoding="utf-8")) if meta_json.exists() else {}
        else:
            sample_cfg = diagnostic_sample_config(path, cfg)
            logging.info(
                "Amostrando JSON: %s | modo=%s | max_rows=%s",
                rel,
                getattr(sample_cfg, "sample_mode", ""),
                getattr(sample_cfg, "max_sample_rows", ""),
            )
            sample, meta = sample_json_file(path, sample_cfg)
            if sample.empty:
                raise RuntimeError("Amostra vazia. O JSON pode estar vazio ou em formato não tabular.")
            profile = infer_column_profile(sample)
            sample.to_csv(sample_csv, sep=";", index=False, encoding="utf-8-sig", compression="gzip")
            save_csv(profile, profile_csv)
            save_json(meta, meta_json)

        stats_df = numeric_stats(sample, profile)
        corr_df = correlations(sample, profile)
        discrete_df = discrete_summary(sample, profile)
        save_csv(stats_df, tables_dir / "estatisticas_numericas.csv")
        save_csv(corr_df, tables_dir / "correlacoes_pearson.csv")
        save_csv(discrete_df, tables_dir / "analise_discreta_categorias.csv")

        if cfg.resume and gold_csv.exists():
            gold = pd.read_csv(gold_csv, sep=";", dtype=str)
            if gold_parts_manifest.exists():
                gold_storage_info = {
                    "modo_gold": "streaming_parquet_parts",
                    "gold_parts_dir": str(gold_parts_dir),
                    "gold_parts_manifest": str(gold_parts_manifest),
                    "gold_parts_manifest_parquet": str(gold_parts_dir / "manifesto_partes_gold.parquet"),
                    "gold_plot_summary_parquet": str(gold_parts_dir / "resumo_plot_gold_completo.parquet"),
                    "gold_plot_manifest_parquet": str(gold_parts_dir / "plots_data" / "manifesto_plots_data.parquet"),
                    "linhas_preview_gold": int(len(gold)),
                }
        else:
            if cfg.full_aggregations:
                spark_result = None
                if getattr(cfg, "engine", "pandas") in {"pyspark", "auto"}:
                    spark_result = aggregate_json_with_pyspark(path, rel, gold_parts_dir, cfg)
                if spark_result is not None:
                    gold, gold_storage_info = spark_result
                else:
                    gold, gold_storage_info = aggregate_full_json_to_parquet_parts(
                        path,
                        rel,
                        gold_parts_dir,
                        chunk_rows=getattr(cfg, "aggregate_chunk_rows", 75000),
                        preview_rows=getattr(cfg, "analysis_max_rows", 200000),
                        cfg=cfg,
                    )
                aggregation_mode = "full_json_streaming_parquet_parts"
            else:
                records = sample.to_dict(orient="records")
                gold = aggregate_records_to_gold(records, rel)
                aggregation_mode = "sample_based"
            if not gold.empty:
                gold = normalize_gold_context(gold, rel)
                gold["arquivo_origem"] = rel
                gold["aggregation_mode"] = aggregation_mode
            save_csv(gold.head(getattr(cfg, "gold_csv_max_rows", 150000) or 150000), gold_csv)
            if cfg.parquet and aggregation_mode == "sample_based":
                save_parquet(gold, gold_parquet)
                gold_storage_info = {
                    "modo_gold": "sample_single_parquet",
                    "gold_parquet": str(gold_parquet),
                    "linhas_preview_gold": int(len(gold)),
                }

        electoral_dir = tables_dir / "analise_eleitoral"
        gold = normalize_gold_context(gold, rel)
        electoral_outputs = run_electoral_analysis(gold, electoral_dir, profile)
        year_outputs = save_individual_year_outputs(gold, base_dir, profile, cfg)

        images = []
        full_plot_manifest = (gold_storage_info or {}).get("gold_plot_manifest_parquet", "")
        full_plot_summary = (gold_storage_info or {}).get("gold_plot_summary_parquet", "")
        if full_plot_manifest and Path(full_plot_manifest).exists():
            images.extend(plot_gold_datasets(full_plot_manifest, plots_dir, cfg))
        elif full_plot_summary and Path(full_plot_summary).exists():
            images.extend(plot_gold_complete_summary(full_plot_summary, plots_dir, cfg))
        else:
            images.extend(plot_categorical_frequencies(sample, profile, plots_dir, cfg))
            images.extend(plot_numeric_distributions(sample, profile, plots_dir, cfg))

        story = build_individual_story(rel, sample, profile, gold)
        body = f"""
<h2>Resumo</h2>
<pre>{html.escape(story)}</pre>

<h2>Classificação do documento</h2>
<pre>{html.escape(json.dumps(doc_context, ensure_ascii=False, indent=2, default=str))}</pre>

<h2>Metadados da amostra</h2>
<pre>{html.escape(json.dumps(meta, ensure_ascii=False, indent=2, default=str))}</pre>

<h2>Perfil das colunas</h2>
{df_to_html(profile, cfg.top_n_html)}

<h2>Analise discreta de categorias</h2>
{df_to_html(discrete_df, cfg.top_n_html)}

<h2>Estatísticas numéricas</h2>
{df_to_html(stats_df, cfg.top_n_html)}

<h2>Correlações Pearson</h2>
{df_to_html(corr_df, cfg.top_n_html)}

<h2>Gold individual</h2>
{df_to_html(gold, cfg.top_n_html)}

<h2>Análises eleitorais individuais geradas</h2>
<pre>{html.escape(json.dumps(electoral_outputs, ensure_ascii=False, indent=2, default=str))}</pre>

<h2>Análises individuais por ano</h2>
<pre>{html.escape(json.dumps(year_outputs, ensure_ascii=False, indent=2, default=str))}</pre>

<h2>Respostas eleitorais do arquivo</h2>
{df_to_html(pd.read_csv(electoral_outputs.get("respostas_perguntas_eleitorais"), sep=";", dtype=str) if electoral_outputs.get("respostas_perguntas_eleitorais") else pd.DataFrame(), cfg.top_n_html)}

<h2>Perfil do eleitor por ano</h2>
{df_to_html(pd.read_csv(electoral_outputs.get("perfil_eleitor_por_ano"), sep=";", dtype=str) if electoral_outputs.get("perfil_eleitor_por_ano") else pd.DataFrame(), cfg.top_n_html)}

<h2>Vencedores por seção explicados</h2>
{df_to_html(pd.read_csv(electoral_outputs.get("vencedor_secao_explicado"), sep=";", dtype=str) if electoral_outputs.get("vencedor_secao_explicado") else pd.DataFrame(), cfg.top_n_html)}

<h2>Gráficos</h2>
{''.join(img_tag(img, base_dir) for img in images)}
"""
        html_path = base_dir / "relatorio_individual.html"
        save_html(html_path, f"Relatório individual - {rel}", body)

        return {
            "status": "ok",
            "arquivo": str(path),
            "relativo": rel,
            "base_dir": str(base_dir),
            "perfil_csv": str(profile_csv),
            "analise_discreta_csv": str(tables_dir / "analise_discreta_categorias.csv"),
            "gold_csv": str(gold_csv),
            "gold_parquet": str(gold_parquet) if gold_parquet.exists() else "",
            "gold_parts_dir": str(gold_parts_dir) if gold_parts_dir.exists() else "",
            "gold_parts_manifest": str(gold_parts_manifest) if gold_parts_manifest.exists() else "",
            "gold_parts_manifest_parquet": str(gold_parts_dir / "manifesto_partes_gold.parquet") if (gold_parts_dir / "manifesto_partes_gold.parquet").exists() else "",
            "gold_storage": gold_storage_info,
            "anos_dir": str(base_dir / "anos"),
            "analises_por_ano": year_outputs,
            "html": str(html_path),
            **doc_context,
            "erro": "",
        }

    except Exception as exc:
        err = traceback.format_exc()
        (logs_dir / "erro.txt").write_text(err, encoding="utf-8")
        logging.error("Erro processando %s: %s", rel, exc)
        return {
            "status": "erro",
            "arquivo": str(path),
            "relativo": rel,
            "base_dir": str(base_dir),
            "perfil_csv": "",
            "gold_csv": "",
            "html": "",
            **doc_context,
            "erro": str(exc),
        }


def diagnostic_sample_config(path: Path, cfg):
    sample_cfg = copy.copy(cfg)
    if getattr(cfg, "full_aggregations", False):
        sample_cfg.sample_mode = "head"
        sample_cfg.sample_frac = 1.0
        sample_cfg.max_sample_rows = min(
            int(getattr(cfg, "max_sample_rows", 30000) or 30000),
            int(getattr(cfg, "analysis_max_rows", 120000) or 120000),
        )

    try:
        size_gb = path.stat().st_size / (1024 ** 3)
    except Exception:
        size_gb = 0.0

    if size_gb >= 1:
        sample_cfg.sample_mode = "head"
        sample_cfg.sample_frac = 1.0
        sample_cfg.max_sample_rows = min(int(getattr(sample_cfg, "max_sample_rows", 30000) or 30000), 5000)
        sample_cfg.min_sample_rows = min(int(getattr(sample_cfg, "min_sample_rows", 3000) or 3000), sample_cfg.max_sample_rows)
        logging.info(
            "Arquivo grande detectado (%.2f GB). Amostra diagnostica limitada a %s linhas; dados completos entram no Parquet streaming.",
            size_gb,
            sample_cfg.max_sample_rows,
        )
    return sample_cfg


def normalize_gold_context(gold: pd.DataFrame, rel: str) -> pd.DataFrame:
    if gold is None or gold.empty:
        return pd.DataFrame() if gold is None else gold
    out = gold.copy()
    if "ano" not in out.columns:
        out["ano"] = ""
    out["ano"] = out["ano"].fillna("").astype(str).str.strip()
    missing_year = out["ano"].eq("")
    if missing_year.any():
        years = extract_years_from_value(rel)
        if years:
            out.loc[missing_year, "ano"] = str(years[0])
    if "arquivo_origem" not in out.columns:
        out["arquivo_origem"] = rel
    return out


def save_individual_year_outputs(gold: pd.DataFrame, base_dir: Path, profile: pd.DataFrame, cfg) -> dict[str, Any]:
    years_dir = base_dir / "anos"
    years_dir.mkdir(parents=True, exist_ok=True)

    if gold is None or gold.empty:
        return {}

    work = gold.copy()
    if "ano" not in work.columns:
        work["ano"] = "ano_desconhecido"
    work["ano"] = work["ano"].fillna("").astype(str).str.strip()
    work.loc[work["ano"].eq(""), "ano"] = "ano_desconhecido"

    outputs: dict[str, Any] = {}
    for year, year_gold in work.groupby("ano", dropna=False):
        year_key = safe_name(year or "ano_desconhecido", 40)
        year_dir = years_dir / year_key
        tables_dir = year_dir / "tabelas"
        electoral_dir = tables_dir / "analise_eleitoral"
        parquet_dir = year_dir / "parquet"
        tables_dir.mkdir(parents=True, exist_ok=True)
        parquet_dir.mkdir(parents=True, exist_ok=True)

        gold_path = tables_dir / "gold_individual_ano.csv"
        save_csv(year_gold, gold_path)
        parquet_path = ""
        if cfg.parquet:
            pq = parquet_dir / "gold_individual_ano.parquet"
            if save_parquet(year_gold, pq):
                parquet_path = str(pq)

        electoral_outputs = run_electoral_analysis(year_gold, electoral_dir, profile)
        outputs[str(year)] = {
            "gold_csv": str(gold_path),
            "gold_parquet": parquet_path,
            "analise_eleitoral_dir": str(electoral_dir),
            "analise_eleitoral_outputs": electoral_outputs,
            "linhas_gold": int(len(year_gold)),
        }

    return outputs


def build_individual_story(rel: str, sample: pd.DataFrame, profile: pd.DataFrame, gold: pd.DataFrame) -> str:
    lines = [
        f"Arquivo: {rel}",
        f"Linhas na amostra: {len(sample)}",
        f"Colunas detectadas: {len(profile)}",
    ]

    if profile is not None and not profile.empty:
        types = profile["tipo_inferido"].value_counts().to_dict()
        lines.append("Tipos inferidos: " + ", ".join(f"{k}={v}" for k, v in types.items()))

        years = set()
        for ylist in profile.get("anos_detectados_no_valor", pd.Series(dtype=str)).fillna("").astype(str):
            for token in ylist.replace(",", " ").split():
                if token.isdigit():
                    years.add(int(token))
        if years:
            lines.append("Anos detectados em valores do arquivo: " + ", ".join(map(str, sorted(years))))

    if gold is not None and not gold.empty:
        years_gold = sorted(pd.to_numeric(gold.get("ano", pd.Series(dtype=str)), errors="coerce").dropna().astype(int).unique().tolist())
        if years_gold:
            lines.append("Anos que entraram no gold: " + ", ".join(map(str, years_gold)))
        lines.append(f"Linhas gold: {len(gold)}")

    lines.append("Esta etapa não assume estrutura fixa. Ela descreve o que existe no JSON e só depois tenta construir gold eleitoral quando há campos compatíveis.")
    return "\n".join(lines)
