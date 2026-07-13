from __future__ import annotations

from pathlib import Path
from typing import Any
import logging

import numpy as np
import pandas as pd

from .utils import (
    compact_code,
    extract_years_from_value,
    safe_name,
    safe_text,
    save_csv,
    save_parquet,
)
from .discrete import label_category_value, readable_field_label


PROFILE_COLS = [
    "perfil_faixa_etaria",
    "perfil_genero",
    "perfil_instrucao",
    "perfil_estado_civil",
    "perfil_raca_cor",
]

SECTOR_KEY_COLS = [
    "ano_correlacao",
    "uf",
    "cd_municipio",
    "nm_municipio",
    "zona",
    "secao",
    "local_votacao",
    "bairro",
    "codigo_municipio",
    "codigo_setor_eleitoral",
    "codigo_correlacao_setor_ano",
]

ELECTION_KEY_COLS = SECTOR_KEY_COLS + ["cargo", "turno"]
ENTITY_COLS = ["partido", "candidato", "ideologia", "coalizao"]

NUMERIC_COLS = [
    "votos",
    "eleitorado",
    "comparecimento",
    "abstencao",
    "brancos",
    "nulos",
    "validos",
    "validos_estimados",
    "comparecimento_estimado",
    "abstencao_estimado",
    "pct_comparecimento",
    "pct_abstencao",
    "share_votos_grupo",
    "linhas_origem",
]


def build_correlated_year_parquets(
    global_gold: pd.DataFrame,
    results: list[dict[str, Any]],
    global_dir: Path,
    cfg,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Materialize the global correlation layer requested by year/code.

    The individual stage remains descriptive. Here the global stage attaches
    source metadata, builds territorial/electoral codes, separates electorate,
    candidate and result views, and writes year-scoped Parquet tables.
    """
    corr_dir = global_dir / "correlacao_codigos"
    tables_dir = corr_dir / "tabelas"
    parquet_dir = corr_dir / "parquet"
    year_parquet_dir = parquet_dir / "por_ano"
    year_csv_dir = tables_dir / "por_ano"
    for d in [tables_dir, parquet_dir, year_parquet_dir, year_csv_dir]:
        d.mkdir(parents=True, exist_ok=True)

    annotated = annotate_global_gold(global_gold, results)
    if annotated.empty:
        outputs = _empty_outputs(tables_dir, parquet_dir)
        return annotated, outputs

    electorate_sector, profile_sector = build_electorate_sector_tables(annotated)
    results_entity = build_results_entity_table(annotated)
    candidate_catalog = build_candidate_catalog(annotated)
    sector_summary = build_sector_summary(results_entity, electorate_sector)
    correlated_base = build_correlated_base(results_entity, electorate_sector, candidate_catalog)

    manifest_rows: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []

    tables_by_name = {
        "base_gold_anotada": annotated,
        "base_correlacionada_codigo": correlated_base,
        "eleitorado_setor_codigo": electorate_sector,
        "perfil_eleitorado_setor_codigo": profile_sector,
        "resultados_setor_entidade_codigo": results_entity,
        "candidatos_catalogo_codigo": candidate_catalog,
        "resumo_setor_eleitoral_codigo": sector_summary,
    }

    combined_paths = {}
    for name, table in tables_by_name.items():
        csv_path = tables_dir / f"{name}.csv"
        _save_csv_preview_for_large_table(table, csv_path, cfg)
        combined_paths[f"{name}_csv"] = str(csv_path)

        pq_path = parquet_dir / f"{name}.parquet"
        if cfg.parquet and save_parquet(table, pq_path):
            combined_paths[f"{name}_parquet"] = str(pq_path)
        else:
            combined_paths[f"{name}_parquet"] = ""

    years = _sorted_years(annotated, correlated_base, electorate_sector, results_entity, candidate_catalog)
    for year in years:
        year_key = safe_name(f"ano_{year}", 40)
        pq_year_dir = year_parquet_dir / year_key
        csv_year_dir = year_csv_dir / year_key
        pq_year_dir.mkdir(parents=True, exist_ok=True)
        csv_year_dir.mkdir(parents=True, exist_ok=True)

        year_tables = {
            "base_correlacionada_codigo": _filter_year(correlated_base, year),
            "eleitorado_setor_codigo": _filter_year(electorate_sector, year),
            "perfil_eleitorado_setor_codigo": _filter_year(profile_sector, year),
            "resultados_setor_entidade_codigo": _filter_year(results_entity, year),
            "candidatos_catalogo_codigo": _filter_year(candidate_catalog, year),
            "resumo_setor_eleitoral_codigo": _filter_year(sector_summary, year),
        }

        for table_name, table in year_tables.items():
            if table.empty:
                continue
            file_stem = f"{table_name}_{year}"
            csv_path = csv_year_dir / f"{file_stem}.csv"
            parquet_path = pq_year_dir / f"{file_stem}.parquet"

            # CSV fallback keeps the pipeline useful even without pyarrow.
            csv_value = ""
            if not cfg.parquet:
                save_csv(table, csv_path)
                csv_value = str(csv_path)

            parquet_value = ""
            if cfg.parquet and save_parquet(table, parquet_path):
                parquet_value = str(parquet_path)

            manifest_rows.append({
                "ano": year,
                "tipo_tabela": table_name,
                "linhas": int(len(table)),
                "colunas": int(len(table.columns)),
                "parquet": parquet_value,
                "csv": csv_value,
                "subtitulo": table_name.replace("_", " "),
                "ano_no_arquivo": year,
                "codigos_correlacao": ", ".join(_available_cols(table, [
                    "ano_correlacao",
                    "codigo_municipio",
                    "codigo_setor_eleitoral",
                    "codigo_correlacao_setor_ano",
                    "uf",
                    "cd_municipio",
                    "zona",
                    "secao",
                ])),
            })

        stats_rows.append(_year_stats(
            year=year,
            annotated=_filter_year(annotated, year),
            correlated=year_tables["base_correlacionada_codigo"],
            electorate=year_tables["eleitorado_setor_codigo"],
            results_entity=year_tables["resultados_setor_entidade_codigo"],
            candidate_catalog=year_tables["candidatos_catalogo_codigo"],
        ))

    manifest = pd.DataFrame(manifest_rows)
    stats = pd.DataFrame(stats_rows)
    dictionary = correlation_dictionary()

    manifest_csv = tables_dir / "manifesto_parquets_correlacionados_por_ano.csv"
    stats_csv = tables_dir / "estatisticas_correlacionadas_por_ano.csv"
    dictionary_csv = tables_dir / "dicionario_correlacao_codigos.csv"
    save_csv(manifest, manifest_csv)
    save_csv(stats, stats_csv)
    save_csv(dictionary, dictionary_csv)

    manifest_parquet = parquet_dir / "manifesto_parquets_correlacionados_por_ano.parquet"
    stats_parquet = parquet_dir / "estatisticas_correlacionadas_por_ano.parquet"
    if cfg.parquet:
        save_parquet(manifest, manifest_parquet)
        save_parquet(stats, stats_parquet)

    outputs = {
        "correlacao_codigos_dir": str(corr_dir),
        "manifesto_parquets_correlacionados_csv": str(manifest_csv),
        "manifesto_parquets_correlacionados_parquet": str(manifest_parquet) if cfg.parquet else "",
        "estatisticas_correlacionadas_por_ano_csv": str(stats_csv),
        "estatisticas_correlacionadas_por_ano_parquet": str(stats_parquet) if cfg.parquet else "",
        "dicionario_correlacao_codigos_csv": str(dictionary_csv),
        "parquet_por_ano_dir": str(year_parquet_dir),
        "tabelas_por_ano_dir": str(year_csv_dir),
        "anos_correlacionados": years,
        **combined_paths,
    }
    return annotated, outputs


def _save_csv_preview_for_large_table(table: pd.DataFrame, path: Path, cfg) -> None:
    limit = int(getattr(cfg, "gold_csv_max_rows", 150000) or 150000)
    if getattr(cfg, "parquet", True) and table is not None and len(table) > limit:
        preview = table.head(limit).copy()
        preview["_csv_preview_observacao"] = f"CSV limitado a {limit} linhas; tabela completa salva em Parquet."
        save_csv(preview, path)
    else:
        save_csv(table, path)


def annotate_global_gold(global_gold: pd.DataFrame, results: list[dict[str, Any]]) -> pd.DataFrame:
    if global_gold is None or global_gold.empty:
        return pd.DataFrame()

    df = global_gold.copy()
    if "arquivo_origem" not in df.columns:
        df["arquivo_origem"] = ""

    meta = _result_metadata(results)
    if not meta.empty:
        df = df.merge(meta, on="arquivo_origem", how="left")
    else:
        for col in ["dominio_documento", "assunto_documento", "tipo_arquivo_json"]:
            df[col] = ""

    for col in ["ano", "uf", "cd_municipio", "nm_municipio", "zona", "secao", "local_votacao", "bairro", "cargo", "turno", "partido", "candidato", "ideologia", "coalizao"]:
        if col not in df.columns:
            df[col] = ""
        if col in ["uf", "cd_municipio", "zona", "secao", "turno"]:
            df[col] = df[col].map(compact_code)
        else:
            df[col] = df[col].map(lambda x: safe_text(x, ""))

    for col in PROFILE_COLS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].map(lambda x: safe_text(x, ""))

    for col in ["cargo", "turno"] + PROFILE_COLS:
        if col in df.columns:
            df[col] = df[col].map(lambda x, c=col: _label_if_present(x, c))

    for col in NUMERIC_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["ano_conteudo_original"] = df["ano"].map(lambda x: safe_text(x, ""))
    df["ano_nome_arquivo"] = df["arquivo_origem"].map(_first_year_token)
    df["ano_correlacao"] = df.apply(_choose_correlation_year, axis=1)
    df["ano"] = np.where(df["ano"].astype(str).str.strip().ne(""), df["ano"].astype(str), df["ano_correlacao"])

    df["subtitulo_arquivo"] = df["arquivo_origem"].map(_subtitle_from_path)
    df["codigo_municipio"] = df.apply(lambda r: _join_code_parts([r.get("cd_municipio", "")]), axis=1)
    df["codigo_zona_eleitoral"] = df.apply(lambda r: _join_code_parts([r.get("cd_municipio", ""), r.get("zona", "")]), axis=1)
    df["codigo_setor_eleitoral"] = df.apply(lambda r: _join_code_parts([r.get("cd_municipio", ""), r.get("zona", ""), r.get("secao", "")]), axis=1)
    df["codigo_correlacao_setor_ano"] = df.apply(lambda r: _join_code_parts([r.get("ano_correlacao", ""), r.get("codigo_setor_eleitoral", "")]), axis=1)
    df["tem_codigo_setor"] = df["codigo_setor_eleitoral"].astype(str).str.strip().ne("")

    text_domain = (
        df.get("dominio_documento", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
        + " "
        + df.get("assunto_documento", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
        + " "
        + df.get("arquivo_origem", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    )
    profile_mask = pd.Series(False, index=df.index)
    for col in PROFILE_COLS:
        profile_mask = profile_mask | df[col].astype(str).str.strip().ne("")

    df["eh_resultado"] = (
        text_domain.str.contains("resultado|votacao|vota", regex=True)
        | (df["votos"].fillna(0) > 0)
        | (df["validos_estimados"].fillna(0) > 0)
    )
    df["eh_eleitorado"] = (
        text_domain.str.contains("eleitorado|perfil_eleitor", regex=True)
        | (df["eleitorado"].fillna(0) > 0)
        | profile_mask
    )
    df["eh_candidato"] = (
        text_domain.str.contains("candidato|cand|partido|coligacao|colig", regex=True)
        | df["candidato"].astype(str).str.strip().ne("")
        | df["partido"].astype(str).str.strip().ne("")
    )

    df["familia_dado"] = np.select(
        [df["eh_resultado"], df["eh_eleitorado"], df["eh_candidato"]],
        ["resultado", "eleitorado", "candidato"],
        default="outros",
    )
    return df


def build_electorate_sector_tables(annotated: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = annotated.loc[annotated.get("eh_eleitorado", False)].copy()
    if base.empty:
        return pd.DataFrame(), pd.DataFrame()

    sector_keys = _available_cols(base, SECTOR_KEY_COLS)
    profile_sector = _profile_sector_table(base, sector_keys)
    profile_desc = _profile_description(profile_sector, sector_keys)

    agg_map = {
        "eleitorado": "max",
        "comparecimento_estimado": "max",
        "abstencao_estimado": "max",
        "brancos": "sum",
        "nulos": "sum",
        "linhas_origem": "sum",
        "arquivo_origem": _join_unique_limited,
        "subtitulo_arquivo": _join_unique_limited,
    }
    agg_map = {col: op for col, op in agg_map.items() if col in base.columns}
    sector = base.groupby(sector_keys, dropna=False).agg(agg_map).reset_index()
    sector = sector.rename(columns={
        "eleitorado": "eleitorado_setor",
        "comparecimento_estimado": "comparecimento_setor",
        "abstencao_estimado": "abstencao_setor",
        "brancos": "brancos_setor",
        "nulos": "nulos_setor",
        "linhas_origem": "linhas_origem_eleitorado",
        "arquivo_origem": "arquivos_eleitorado",
        "subtitulo_arquivo": "subtitulos_eleitorado",
    })

    dim_totals = []
    for col in PROFILE_COLS:
        if col not in base.columns or not base[col].astype(str).str.strip().ne("").any():
            continue
        tmp = base.loc[base[col].astype(str).str.strip().ne("")].copy()
        if tmp.empty:
            continue
        tmp = tmp.groupby(sector_keys, dropna=False)["eleitorado"].sum().reset_index()
        tmp = tmp.rename(columns={"eleitorado": f"eleitorado_total_dim_{col.replace('perfil_', '')}"})
        dim_totals.append(tmp)

    for tmp in dim_totals:
        sector = sector.merge(tmp, on=sector_keys, how="left")

    dim_cols = [c for c in sector.columns if c.startswith("eleitorado_total_dim_")]
    if dim_cols:
        sector["eleitorado_total_dimensao_max"] = sector[dim_cols].max(axis=1, skipna=True)
        sector["eleitorado_setor"] = np.maximum(
            pd.to_numeric(sector["eleitorado_setor"], errors="coerce").fillna(0),
            pd.to_numeric(sector["eleitorado_total_dimensao_max"], errors="coerce").fillna(0),
        )

    if "comparecimento_setor" in sector.columns:
        sector["pct_comparecimento_setor"] = np.where(
            sector["eleitorado_setor"] > 0,
            pd.to_numeric(sector["comparecimento_setor"], errors="coerce").fillna(0) / sector["eleitorado_setor"],
            np.nan,
        )
    if "abstencao_setor" in sector.columns:
        sector["pct_abstencao_setor"] = np.where(
            sector["eleitorado_setor"] > 0,
            pd.to_numeric(sector["abstencao_setor"], errors="coerce").fillna(0) / sector["eleitorado_setor"],
            np.nan,
        )

    if not profile_desc.empty:
        sector = sector.merge(profile_desc, on=sector_keys, how="left")
    if "perfil_predominante_setor" not in sector.columns:
        sector["perfil_predominante_setor"] = ""

    return sector, profile_sector


def build_results_entity_table(annotated: pd.DataFrame) -> pd.DataFrame:
    base = annotated.loc[annotated.get("eh_resultado", False)].copy()
    if base.empty:
        return pd.DataFrame()

    group_cols = _available_cols(base, ELECTION_KEY_COLS + ENTITY_COLS)
    metrics = _available_cols(base, [
        "votos",
        "brancos",
        "nulos",
        "validos",
        "validos_estimados",
        "comparecimento_estimado",
        "abstencao_estimado",
        "linhas_origem",
    ])
    agg_map: dict[str, Any] = {col: "sum" for col in metrics}
    agg_map["arquivo_origem"] = _join_unique_limited
    agg_map["subtitulo_arquivo"] = _join_unique_limited
    results = base.groupby(group_cols, dropna=False).agg(agg_map).reset_index()
    results = results.rename(columns={
        "arquivo_origem": "arquivos_resultado",
        "subtitulo_arquivo": "subtitulos_resultado",
        "linhas_origem": "linhas_origem_resultado",
    })

    for col in ["partido", "candidato"]:
        if col not in results.columns:
            results[col] = ""
    results["entidade"] = results.apply(_entity_label, axis=1)

    denom_keys = _available_cols(results, ELECTION_KEY_COLS)
    if "votos" in results.columns and denom_keys:
        total = results.groupby(denom_keys, dropna=False)["votos"].transform("sum")
        results["share_votos_setor"] = np.where(total > 0, results["votos"] / total, np.nan)
        results["rank_entidade_setor"] = results.groupby(denom_keys, dropna=False)["votos"].rank(method="first", ascending=False)
        results["vencedor_setor"] = np.where(results["rank_entidade_setor"].eq(1), results["entidade"], "")

    return results


def build_candidate_catalog(annotated: pd.DataFrame) -> pd.DataFrame:
    base = annotated.loc[annotated.get("eh_candidato", False)].copy()
    if base.empty:
        return pd.DataFrame()

    candidate_keys = _available_cols(base, [
        "ano_correlacao",
        "ano",
        "uf",
        "cargo",
        "turno",
        "partido",
        "candidato",
        "ideologia",
        "coalizao",
    ])
    if not candidate_keys:
        return pd.DataFrame()

    agg_map = {
        "arquivo_origem": _join_unique_limited,
        "subtitulo_arquivo": _join_unique_limited,
        "dominio_documento": _join_unique_limited,
        "assunto_documento": _join_unique_limited,
        "linhas_origem": "sum",
    }
    agg_map = {col: op for col, op in agg_map.items() if col in base.columns}
    out = base.groupby(candidate_keys, dropna=False).agg(agg_map).reset_index()
    out = out.rename(columns={
        "arquivo_origem": "arquivos_candidato",
        "subtitulo_arquivo": "subtitulos_candidato",
        "dominio_documento": "dominios_candidato",
        "assunto_documento": "assuntos_candidato",
        "linhas_origem": "linhas_origem_candidato",
    })
    return out


def build_correlated_base(
    results_entity: pd.DataFrame,
    electorate_sector: pd.DataFrame,
    candidate_catalog: pd.DataFrame,
) -> pd.DataFrame:
    if results_entity is None or results_entity.empty:
        return pd.DataFrame()

    out = results_entity.copy()

    if electorate_sector is not None and not electorate_sector.empty:
        sector_join = _join_keys_with_signal(out, electorate_sector, ["ano_correlacao", "uf", "cd_municipio", "zona", "secao"])
        if sector_join and _has_any_key(sector_join, ["uf", "cd_municipio", "zona", "secao"]):
            keep = sector_join + [
                c for c in [
                    "eleitorado_setor",
                    "comparecimento_setor",
                    "abstencao_setor",
                    "pct_comparecimento_setor",
                    "pct_abstencao_setor",
                    "perfil_predominante_setor",
                    "arquivos_eleitorado",
                    "subtitulos_eleitorado",
                ]
                if c in electorate_sector.columns
            ]
            out = out.merge(electorate_sector[keep].drop_duplicates(sector_join), on=sector_join, how="left")

    if candidate_catalog is not None and not candidate_catalog.empty:
        candidate_join = _join_keys_with_signal(out, candidate_catalog, ["ano_correlacao", "cargo", "partido", "candidato"])
        if candidate_join and _has_any_key(candidate_join, ["partido", "candidato"]):
            add_cols = [
                c for c in [
                    "arquivos_candidato",
                    "subtitulos_candidato",
                    "dominios_candidato",
                    "assuntos_candidato",
                    "linhas_origem_candidato",
                ]
                if c in candidate_catalog.columns
            ]
            out = out.merge(
                candidate_catalog[candidate_join + add_cols].drop_duplicates(candidate_join),
                on=candidate_join,
                how="left",
            )
            out["candidato_presente_catalogo"] = out[add_cols[0]].notna() if add_cols else True
        else:
            out["candidato_presente_catalogo"] = False

    if "eleitorado" not in out.columns:
        out["eleitorado"] = 0.0
    if "eleitorado_setor" in out.columns:
        out["eleitorado"] = np.where(
            pd.to_numeric(out["eleitorado"], errors="coerce").fillna(0) > 0,
            pd.to_numeric(out["eleitorado"], errors="coerce").fillna(0),
            pd.to_numeric(out["eleitorado_setor"], errors="coerce").fillna(0),
        )
    if "comparecimento_estimado" in out.columns and "comparecimento_setor" in out.columns:
        out["comparecimento_estimado"] = np.where(
            pd.to_numeric(out["comparecimento_estimado"], errors="coerce").fillna(0) > 0,
            pd.to_numeric(out["comparecimento_estimado"], errors="coerce").fillna(0),
            pd.to_numeric(out["comparecimento_setor"], errors="coerce").fillna(0),
        )
    if "abstencao_estimado" in out.columns and "abstencao_setor" in out.columns:
        out["abstencao_estimado"] = np.where(
            pd.to_numeric(out["abstencao_estimado"], errors="coerce").fillna(0) > 0,
            pd.to_numeric(out["abstencao_estimado"], errors="coerce").fillna(0),
            pd.to_numeric(out["abstencao_setor"], errors="coerce").fillna(0),
        )

    out["base_correlacao"] = "resultados + eleitorado_setor + candidatos_catalogo"
    return out


def build_sector_summary(results_entity: pd.DataFrame, electorate_sector: pd.DataFrame) -> pd.DataFrame:
    if results_entity is None or results_entity.empty:
        return electorate_sector.copy() if electorate_sector is not None else pd.DataFrame()

    election_keys = _available_cols(results_entity, ELECTION_KEY_COLS)
    metrics = _available_cols(results_entity, ["votos", "brancos", "nulos", "validos_estimados", "comparecimento_estimado", "abstencao_estimado"])
    summary = results_entity.groupby(election_keys, dropna=False)[metrics].sum().reset_index()

    if "rank_entidade_setor" in results_entity.columns:
        winners = results_entity.loc[pd.to_numeric(results_entity["rank_entidade_setor"], errors="coerce").eq(1)].copy()
        winner_cols = _available_cols(winners, election_keys + ["entidade", "partido", "candidato", "votos", "share_votos_setor"])
        winners = winners[winner_cols].rename(columns={
            "entidade": "vencedor_setor",
            "partido": "partido_vencedor_setor",
            "candidato": "candidato_vencedor_setor",
            "votos": "votos_vencedor_setor",
            "share_votos_setor": "share_vencedor_setor",
        })
        summary = summary.merge(winners, on=election_keys, how="left")

    if electorate_sector is not None and not electorate_sector.empty:
        sector_join = _join_keys_with_signal(summary, electorate_sector, ["ano_correlacao", "uf", "cd_municipio", "zona", "secao"])
        if sector_join and _has_any_key(sector_join, ["uf", "cd_municipio", "zona", "secao"]):
            keep = sector_join + [
                c for c in [
                    "eleitorado_setor",
                    "comparecimento_setor",
                    "abstencao_setor",
                    "pct_comparecimento_setor",
                    "pct_abstencao_setor",
                    "perfil_predominante_setor",
                ]
                if c in electorate_sector.columns
            ]
            summary = summary.merge(electorate_sector[keep].drop_duplicates(sector_join), on=sector_join, how="left")

    return summary


def correlation_dictionary() -> pd.DataFrame:
    rows = [
        ("ano_correlacao", "Ano usado para agrupar e cruzar os arquivos; vem do conteudo quando existe, senao do nome do arquivo."),
        ("codigo_municipio", "Chave territorial baseada no codigo do municipio, sem UF."),
        ("codigo_zona_eleitoral", "Chave municipio + zona eleitoral, sem UF."),
        ("codigo_setor_eleitoral", "Chave municipio + zona + secao, sem UF, quando os campos existem."),
        ("codigo_correlacao_setor_ano", "Ano de correlacao + codigo_setor_eleitoral."),
        ("base_correlacionada_codigo", "Tabela por ano que cruza resultados por entidade com eleitorado do setor e catalogo de candidatos."),
        ("eleitorado_setor_codigo", "Tabela por ano com eleitorado e perfil predominante por setor/secao."),
        ("perfil_eleitorado_setor_codigo", "Tabela longa por setor, dimensao de perfil e valor do perfil."),
        ("resultados_setor_entidade_codigo", "Tabela por setor/cargo/turno/candidato ou partido com votos, share e ranking."),
        ("candidatos_catalogo_codigo", "Catalogo anual de candidatos/partidos extraido dos arquivos de candidatos e dos resultados."),
        ("resumo_setor_eleitoral_codigo", "Resumo por setor/cargo/turno com vencedor, votos e eleitorado associado."),
    ]
    return pd.DataFrame(rows, columns=["campo_ou_tabela", "descricao"])


def _profile_sector_table(base: pd.DataFrame, sector_keys: list[str]) -> pd.DataFrame:
    frames = []
    for col in PROFILE_COLS:
        if col not in base.columns or not base[col].astype(str).str.strip().ne("").any():
            continue
        tmp = base.loc[base[col].astype(str).str.strip().ne("")].copy()
        tmp[col] = tmp[col].map(lambda x, c=col: _label_if_present(x, c))
        tmp = tmp.groupby(sector_keys + [col], dropna=False)["eleitorado"].sum().reset_index()
        tmp = tmp.rename(columns={col: "valor_perfil", "eleitorado": "eleitorado_perfil_setor"})
        tmp["dimensao_perfil"] = col.replace("perfil_", "")
        denom = tmp.groupby(sector_keys + ["dimensao_perfil"], dropna=False)["eleitorado_perfil_setor"].transform("sum")
        tmp["share_perfil_setor"] = np.where(denom > 0, tmp["eleitorado_perfil_setor"] / denom, np.nan)
        tmp["rank_perfil_setor"] = tmp.groupby(sector_keys + ["dimensao_perfil"], dropna=False)["eleitorado_perfil_setor"].rank(method="first", ascending=False)
        frames.append(tmp)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _profile_description(profile_sector: pd.DataFrame, sector_keys: list[str]) -> pd.DataFrame:
    if profile_sector is None or profile_sector.empty:
        return pd.DataFrame()
    top = profile_sector.loc[pd.to_numeric(profile_sector["rank_perfil_setor"], errors="coerce").eq(1)].copy()
    if top.empty:
        return pd.DataFrame()
    top["perfil_item"] = top.apply(
        lambda r: f"{readable_field_label('perfil_' + safe_text(r.get('dimensao_perfil', '')).replace('perfil_', ''))}: {r.get('valor_perfil', '')}",
        axis=1,
    )
    return top.groupby(sector_keys, dropna=False)["perfil_item"].agg(
        lambda s: "; ".join([safe_text(x) for x in s if safe_text(x)])
    ).reset_index().rename(columns={"perfil_item": "perfil_predominante_setor"})


def _result_metadata(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in results or []:
        rel = safe_text(r.get("relativo", ""))
        if not rel:
            continue
        rows.append({
            "arquivo_origem": rel,
            "dominio_documento": safe_text(r.get("dominio_documento", "")),
            "assunto_documento": safe_text(r.get("assunto_documento", "")),
            "tipo_arquivo_json": safe_text(r.get("tipo_arquivo_json", "")),
            "arquivo_original_absoluto": safe_text(r.get("arquivo", "")),
        })
    return pd.DataFrame(rows).drop_duplicates("arquivo_origem") if rows else pd.DataFrame()


def _empty_outputs(tables_dir: Path, parquet_dir: Path) -> dict[str, Any]:
    manifest = tables_dir / "manifesto_parquets_correlacionados_por_ano.csv"
    stats = tables_dir / "estatisticas_correlacionadas_por_ano.csv"
    dictionary = tables_dir / "dicionario_correlacao_codigos.csv"
    save_csv(pd.DataFrame(), manifest)
    save_csv(pd.DataFrame(), stats)
    save_csv(correlation_dictionary(), dictionary)
    return {
        "correlacao_codigos_dir": str(tables_dir.parent),
        "manifesto_parquets_correlacionados_csv": str(manifest),
        "manifesto_parquets_correlacionados_parquet": "",
        "estatisticas_correlacionadas_por_ano_csv": str(stats),
        "estatisticas_correlacionadas_por_ano_parquet": "",
        "dicionario_correlacao_codigos_csv": str(dictionary),
        "parquet_por_ano_dir": str(parquet_dir / "por_ano"),
        "anos_correlacionados": [],
    }


def _year_stats(
    year: str,
    annotated: pd.DataFrame,
    correlated: pd.DataFrame,
    electorate: pd.DataFrame,
    results_entity: pd.DataFrame,
    candidate_catalog: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "ano": year,
        "linhas_gold_anotado": int(len(annotated)),
        "linhas_base_correlacionada": int(len(correlated)),
        "linhas_eleitorado_setor": int(len(electorate)),
        "linhas_resultado_entidade": int(len(results_entity)),
        "linhas_catalogo_candidato": int(len(candidate_catalog)),
        "arquivos_origem": int(annotated["arquivo_origem"].nunique()) if "arquivo_origem" in annotated.columns and not annotated.empty else 0,
        "ufs": int(annotated["uf"].replace("", pd.NA).dropna().nunique()) if "uf" in annotated.columns and not annotated.empty else 0,
        "municipios": int(annotated["cd_municipio"].replace("", pd.NA).dropna().nunique()) if "cd_municipio" in annotated.columns and not annotated.empty else 0,
        "setores_eleitorais": int(annotated["codigo_setor_eleitoral"].replace("", pd.NA).dropna().nunique()) if "codigo_setor_eleitoral" in annotated.columns and not annotated.empty else 0,
        "candidatos": int(candidate_catalog["candidato"].replace("", pd.NA).dropna().nunique()) if "candidato" in candidate_catalog.columns and not candidate_catalog.empty else 0,
        "partidos": int(candidate_catalog["partido"].replace("", pd.NA).dropna().nunique()) if "partido" in candidate_catalog.columns and not candidate_catalog.empty else 0,
        "votos_total_resultados": _safe_numeric_sum(results_entity, "votos"),
        "eleitorado_total_setores": _safe_numeric_sum(electorate, "eleitorado_setor"),
    }


def _safe_numeric_sum(df: pd.DataFrame, col: str) -> float:
    if df is None or df.empty or col not in df.columns:
        return 0.0
    return float(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())


def _filter_year(df: pd.DataFrame, year: str) -> pd.DataFrame:
    if df is None or df.empty or "ano_correlacao" not in df.columns:
        return pd.DataFrame()
    return df.loc[df["ano_correlacao"].astype(str).eq(str(year))].copy()


def _sorted_years(*tables: pd.DataFrame) -> list[str]:
    years: set[str] = set()
    for table in tables:
        if table is None or table.empty or "ano_correlacao" not in table.columns:
            continue
        for value in table["ano_correlacao"].dropna().astype(str):
            if safe_text(value):
                years.add(value)
    return sorted(years, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))


def _available_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def _join_keys_with_signal(left: pd.DataFrame, right: pd.DataFrame, candidates: list[str]) -> list[str]:
    keys = []
    for col in candidates:
        if col not in left.columns or col not in right.columns:
            continue
        left_has = left[col].astype(str).str.strip().ne("").any()
        right_has = right[col].astype(str).str.strip().ne("").any()
        if left_has and right_has:
            keys.append(col)
    if "ano_correlacao" in left.columns and "ano_correlacao" in right.columns and "ano_correlacao" not in keys:
        keys.insert(0, "ano_correlacao")
    return keys


def _has_any_key(keys: list[str], required: list[str]) -> bool:
    return any(key in keys for key in required)


def _choose_correlation_year(row: pd.Series) -> str:
    content = safe_text(row.get("ano", ""))
    content_years = extract_years_from_value(content)
    if content_years:
        return str(content_years[0])
    name_year = safe_text(row.get("ano_nome_arquivo", ""))
    if name_year:
        return name_year
    return "ano_desconhecido"


def _label_if_present(value: Any, col: str) -> str:
    text = safe_text(value, "")
    if not text or text.lower() in {"sem valor", "sem_valor"}:
        return ""
    return label_category_value(text, col=col, role=col)


def _first_year_token(value: Any) -> str:
    years = extract_years_from_value(value)
    return str(years[0]) if years else ""


def _subtitle_from_path(value: Any) -> str:
    text = safe_text(value, "")
    if not text:
        return ""
    name = text.replace("\\", "/").split("/")[-1]
    stem = name.rsplit(".", 1)[0]
    return safe_text(stem, safe_name(text, 80))


def _join_code_parts(parts: list[Any]) -> str:
    values = [compact_code(p) for p in parts]
    values = [v for v in values if safe_text(v)]
    return "|".join(values)


def _join_unique_limited(values: pd.Series, limit: int = 20) -> str:
    cleaned = []
    seen = set()
    for value in values:
        text = safe_text(value, "")
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
        if len(cleaned) >= limit:
            break
    return " | ".join(cleaned)


def _entity_label(row: pd.Series) -> str:
    candidato = safe_text(row.get("candidato", ""))
    partido = safe_text(row.get("partido", ""))
    if candidato and partido:
        return f"{candidato} | {partido}"
    return candidato or partido or "GERAL"


def _empty_df_warning(name: str) -> None:
    logging.info("Tabela de correlacao vazia: %s", name)
