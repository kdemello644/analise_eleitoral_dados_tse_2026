from __future__ import annotations

from pathlib import Path
from typing import Any
import logging

import numpy as np
import pandas as pd

from .utils import (
    MATPLOTLIB_OK,
    clean_memory,
    img_tag,
    safe_name,
    safe_text,
    save_csv,
    save_parquet,
)
from .discrete import is_useful_discrete_series, label_category_value, readable_field_label

if MATPLOTLIB_OK:
    import matplotlib.pyplot as plt
else:
    plt = None

try:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction import FeatureHasher
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


HASH_FEATURES = 4096
MAX_CLUSTER_TRAIN_ROWS = 50000

PROFILE_COLS = [
    "perfil_faixa_etaria",
    "perfil_genero",
    "perfil_instrucao",
    "perfil_estado_civil",
    "perfil_raca_cor",
]

REGION_BY_UF = {
    "AC": "Norte",
    "AP": "Norte",
    "AM": "Norte",
    "PA": "Norte",
    "RO": "Norte",
    "RR": "Norte",
    "TO": "Norte",
    "AL": "Nordeste",
    "BA": "Nordeste",
    "CE": "Nordeste",
    "MA": "Nordeste",
    "PB": "Nordeste",
    "PE": "Nordeste",
    "PI": "Nordeste",
    "RN": "Nordeste",
    "SE": "Nordeste",
    "DF": "Centro-Oeste",
    "GO": "Centro-Oeste",
    "MT": "Centro-Oeste",
    "MS": "Centro-Oeste",
    "ES": "Sudeste",
    "MG": "Sudeste",
    "RJ": "Sudeste",
    "SP": "Sudeste",
    "PR": "Sul",
    "RS": "Sul",
    "SC": "Sul",
}

VOTER_ONLY_CATEGORICAL_COLS = [
    "regiao",
    "cargo",
    "turno",
    "perfil_faixa_etaria",
    "perfil_genero",
    "perfil_instrucao",
    "perfil_estado_civil",
    "perfil_raca_cor",
    "perfil_predominante_setor",
    "faixa_votos_setor",
    "faixa_abstencao_setor",
    "faixa_comparecimento_setor",
    "municipio_grupo",
]

VOTER_RESULT_CATEGORICAL_COLS = VOTER_ONLY_CATEGORICAL_COLS + [
    "partido_vencedor_setor",
    "candidato_vencedor_setor",
    "vencedor_setor",
]

BASE_CATEGORICAL_COLS = VOTER_RESULT_CATEGORICAL_COLS

CATEGORY_LIMITS = {
    "candidato_vencedor_setor": 250,
    "vencedor_setor": 250,
    "perfil_predominante_setor": 250,
    "municipio_grupo": 350,
    "partido_vencedor_setor": 120,
}


def run_global_discriminated_cluster_analysis(
    correlation_outputs: dict[str, Any],
    out_dir: Path,
    cfg,
) -> dict[str, Any]:
    """Cluster the correlated global base with focus on discriminated values.

    KMeans is kept as requested, but the feature matrix is categorical-first:
    each sector/election record becomes a bag of discriminated tokens such as
    region, profile, winner, party and discretized electoral ranges.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir / "tabelas"
    parquet_dir = out_dir / "parquet"
    plots_dir = out_dir / "plots"
    for d in [tables_dir, parquet_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)

    base = _load_correlated_base(correlation_outputs)
    sector_base = build_sector_cluster_base(base)
    clustered_voters, summary_voters, discriminants_voters, year_region_voters, entities_voters, municipalities_voters, personas_voters, prediction_voters, elbow_voters, report_voters_md = cluster_discriminated_sector_base(
        sector_base,
        cfg,
        categorical_source_cols=VOTER_ONLY_CATEGORICAL_COLS,
        analysis_label="somente_eleitores",
        include_results=False,
    )
    clustered, summary, discriminants, year_region, entities, municipalities, personas, prediction_2026, elbow, report_md = cluster_discriminated_sector_base(
        sector_base,
        cfg,
        categorical_source_cols=VOTER_RESULT_CATEGORICAL_COLS,
        analysis_label="eleitores_resultado",
        include_results=True,
    )
    images_voters = plot_discriminated_clusters(clustered_voters, summary_voters, discriminants_voters, personas_voters, prediction_voters, elbow_voters, plots_dir / "somente_eleitores", cfg)
    images_result = plot_discriminated_clusters(clustered, summary, discriminants, personas, prediction_2026, elbow, plots_dir / "eleitores_resultado", cfg)
    images = images_voters + images_result

    outputs = {
        "base_setor_cluster_csv": str(tables_dir / "base_setor_cluster.csv"),
        "clusters_globais_eleitores_csv": str(tables_dir / "clusters_globais_eleitores.csv"),
        "clusters_globais_eleitores_resumo_csv": str(tables_dir / "clusters_globais_eleitores_resumo.csv"),
        "clusters_eleitores_personas_csv": str(tables_dir / "clusters_eleitores_personas.csv"),
        "clusters_eleitores_valores_discriminantes_csv": str(tables_dir / "clusters_eleitores_valores_discriminantes.csv"),
        "clusters_eleitores_ano_regiao_csv": str(tables_dir / "clusters_eleitores_ano_regiao.csv"),
        "clusters_eleitores_municipios_csv": str(tables_dir / "clusters_eleitores_municipios.csv"),
        "clusters_eleitores_cotovelo_k_csv": str(tables_dir / "clusters_eleitores_cotovelo_k.csv"),
        "clusters_eleitores_relatorio_md": str(out_dir / "relatorio_clusters_somente_eleitores.md"),
        "clusters_globais_discriminados_csv": str(tables_dir / "clusters_globais_discriminados.csv"),
        "clusters_globais_discriminados_resumo_csv": str(tables_dir / "clusters_globais_discriminados_resumo.csv"),
        "clusters_personas_csv": str(tables_dir / "clusters_personas.csv"),
        "clusters_valores_discriminantes_csv": str(tables_dir / "clusters_valores_discriminantes.csv"),
        "clusters_ano_regiao_csv": str(tables_dir / "clusters_ano_regiao.csv"),
        "clusters_municipios_csv": str(tables_dir / "clusters_municipios.csv"),
        "clusters_entidades_csv": str(tables_dir / "clusters_entidades.csv"),
        "clusters_predicao_2026_csv": str(tables_dir / "clusters_predicao_2026.csv"),
        "clusters_cotovelo_k_csv": str(tables_dir / "clusters_cotovelo_k.csv"),
        "clusters_relatorio_md": str(out_dir / "relatorio_clusters_globais_discriminados.md"),
        "plots": [str(p) for p in images],
    }

    _save_cluster_csv_preview(sector_base, Path(outputs["base_setor_cluster_csv"]), cfg)
    _save_cluster_csv_preview(clustered_voters, Path(outputs["clusters_globais_eleitores_csv"]), cfg)
    _save_cluster_csv_preview(summary_voters, Path(outputs["clusters_globais_eleitores_resumo_csv"]), cfg)
    _save_cluster_csv_preview(personas_voters, Path(outputs["clusters_eleitores_personas_csv"]), cfg)
    _save_cluster_csv_preview(discriminants_voters, Path(outputs["clusters_eleitores_valores_discriminantes_csv"]), cfg)
    _save_cluster_csv_preview(year_region_voters, Path(outputs["clusters_eleitores_ano_regiao_csv"]), cfg)
    _save_cluster_csv_preview(municipalities_voters, Path(outputs["clusters_eleitores_municipios_csv"]), cfg)
    _save_cluster_csv_preview(elbow_voters, Path(outputs["clusters_eleitores_cotovelo_k_csv"]), cfg)
    Path(outputs["clusters_eleitores_relatorio_md"]).write_text(report_voters_md, encoding="utf-8")
    _save_cluster_csv_preview(clustered, Path(outputs["clusters_globais_discriminados_csv"]), cfg)
    _save_cluster_csv_preview(summary, Path(outputs["clusters_globais_discriminados_resumo_csv"]), cfg)
    _save_cluster_csv_preview(personas, Path(outputs["clusters_personas_csv"]), cfg)
    _save_cluster_csv_preview(discriminants, Path(outputs["clusters_valores_discriminantes_csv"]), cfg)
    _save_cluster_csv_preview(year_region, Path(outputs["clusters_ano_regiao_csv"]), cfg)
    _save_cluster_csv_preview(municipalities, Path(outputs["clusters_municipios_csv"]), cfg)
    _save_cluster_csv_preview(entities, Path(outputs["clusters_entidades_csv"]), cfg)
    _save_cluster_csv_preview(prediction_2026, Path(outputs["clusters_predicao_2026_csv"]), cfg)
    _save_cluster_csv_preview(elbow, Path(outputs["clusters_cotovelo_k_csv"]), cfg)
    Path(outputs["clusters_relatorio_md"]).write_text(report_md, encoding="utf-8")

    if cfg.parquet:
        parquet_outputs = {
            "base_setor_cluster_parquet": parquet_dir / "base_setor_cluster.parquet",
            "clusters_globais_eleitores_parquet": parquet_dir / "clusters_globais_eleitores.parquet",
            "clusters_globais_eleitores_resumo_parquet": parquet_dir / "clusters_globais_eleitores_resumo.parquet",
            "clusters_eleitores_personas_parquet": parquet_dir / "clusters_eleitores_personas.parquet",
            "clusters_eleitores_valores_discriminantes_parquet": parquet_dir / "clusters_eleitores_valores_discriminantes.parquet",
            "clusters_eleitores_ano_regiao_parquet": parquet_dir / "clusters_eleitores_ano_regiao.parquet",
            "clusters_eleitores_municipios_parquet": parquet_dir / "clusters_eleitores_municipios.parquet",
            "clusters_eleitores_cotovelo_k_parquet": parquet_dir / "clusters_eleitores_cotovelo_k.parquet",
            "clusters_globais_discriminados_parquet": parquet_dir / "clusters_globais_discriminados.parquet",
            "clusters_globais_discriminados_resumo_parquet": parquet_dir / "clusters_globais_discriminados_resumo.parquet",
            "clusters_personas_parquet": parquet_dir / "clusters_personas.parquet",
            "clusters_valores_discriminantes_parquet": parquet_dir / "clusters_valores_discriminantes.parquet",
            "clusters_ano_regiao_parquet": parquet_dir / "clusters_ano_regiao.parquet",
            "clusters_municipios_parquet": parquet_dir / "clusters_municipios.parquet",
            "clusters_entidades_parquet": parquet_dir / "clusters_entidades.parquet",
            "clusters_predicao_2026_parquet": parquet_dir / "clusters_predicao_2026.parquet",
            "clusters_cotovelo_k_parquet": parquet_dir / "clusters_cotovelo_k.parquet",
        }
        frames = {
            "base_setor_cluster_parquet": sector_base,
            "clusters_globais_eleitores_parquet": clustered_voters,
            "clusters_globais_eleitores_resumo_parquet": summary_voters,
            "clusters_eleitores_personas_parquet": personas_voters,
            "clusters_eleitores_valores_discriminantes_parquet": discriminants_voters,
            "clusters_eleitores_ano_regiao_parquet": year_region_voters,
            "clusters_eleitores_municipios_parquet": municipalities_voters,
            "clusters_eleitores_cotovelo_k_parquet": elbow_voters,
            "clusters_globais_discriminados_parquet": clustered,
            "clusters_globais_discriminados_resumo_parquet": summary,
            "clusters_personas_parquet": personas,
            "clusters_valores_discriminantes_parquet": discriminants,
            "clusters_ano_regiao_parquet": year_region,
            "clusters_municipios_parquet": municipalities,
            "clusters_entidades_parquet": entities,
            "clusters_predicao_2026_parquet": prediction_2026,
            "clusters_cotovelo_k_parquet": elbow,
        }
        for key, path in parquet_outputs.items():
            outputs[key] = str(path) if save_parquet(frames[key], path) else ""

    return outputs


def _save_cluster_csv_preview(df: pd.DataFrame, path: Path, cfg) -> None:
    if getattr(cfg, "parquet", True):
        limit = int(getattr(cfg, "gold_csv_max_rows", 50000) or 50000)
        preview = df.head(limit).copy() if df is not None and not df.empty else df
        if preview is not None and not preview.empty and len(df) > len(preview):
            preview["_csv_preview_observacao"] = f"CSV preview limitado a {limit} linhas; Parquet contem a tabela completa."
        save_csv(preview, path)
        return
    save_csv(df, path)


def build_sector_cluster_base(correlated: pd.DataFrame) -> pd.DataFrame:
    if correlated is None or correlated.empty:
        return pd.DataFrame()

    df = correlated.copy()
    for col in [
        "ano_correlacao",
        "uf",
        "cd_municipio",
        "nm_municipio",
        "zona",
        "secao",
        "codigo_setor_eleitoral",
        "codigo_correlacao_setor_ano",
        "cargo",
        "turno",
        "partido",
        "candidato",
        "entidade",
        "vencedor_setor",
        "perfil_predominante_setor",
        *PROFILE_COLS,
    ]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].map(lambda x, c=col: label_category_value(x, col=c) if c in PROFILE_COLS + ["cargo", "turno"] else safe_text(x, ""))

    for col in [
        "votos",
        "eleitorado",
        "eleitorado_setor",
        "comparecimento_estimado",
        "comparecimento_setor",
        "abstencao_estimado",
        "abstencao_setor",
        "share_votos_setor",
        "pct_abstencao_setor",
        "pct_comparecimento_setor",
        "rank_entidade_setor",
    ]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    keys = [
        c for c in [
            "ano_correlacao",
            "uf",
            "cd_municipio",
            "nm_municipio",
            "zona",
            "secao",
            "codigo_setor_eleitoral",
            "codigo_correlacao_setor_ano",
            "cargo",
            "turno",
        ]
        if c in df.columns
    ]
    if not keys:
        return pd.DataFrame()

    metrics = df.groupby(keys, dropna=False).agg({
        "votos": "sum",
        "eleitorado": "max",
        "eleitorado_setor": "max",
        "comparecimento_estimado": "max",
        "comparecimento_setor": "max",
        "abstencao_estimado": "max",
        "abstencao_setor": "max",
        "pct_abstencao_setor": "max",
        "pct_comparecimento_setor": "max",
    }).reset_index().rename(columns={"votos": "votos_total_setor"})

    winner = df.loc[df["rank_entidade_setor"].eq(1)].copy()
    if winner.empty:
        winner = (
            df.sort_values(keys + ["votos"], ascending=[True] * len(keys) + [False])
            .groupby(keys, dropna=False)
            .head(1)
            .copy()
        )
    winner_cols = keys + [
        "entidade",
        "partido",
        "candidato",
        "share_votos_setor",
        "perfil_predominante_setor",
        "votos",
    ]
    winner_cols = [c for c in winner_cols if c in winner.columns]
    winner = winner[winner_cols].rename(columns={
        "entidade": "vencedor_setor",
        "partido": "partido_vencedor_setor",
        "candidato": "candidato_vencedor_setor",
        "share_votos_setor": "share_vencedor_setor",
        "votos": "votos_vencedor_setor",
    })

    out = metrics.merge(winner.drop_duplicates(keys), on=keys, how="left")
    profile_parts = []
    weight_col = "eleitorado"
    if weight_col not in df.columns or pd.to_numeric(df[weight_col], errors="coerce").fillna(0).sum() <= 0:
        weight_col = "votos"
    for col in PROFILE_COLS:
        if col not in df.columns or not df[col].astype(str).str.strip().ne("").any():
            continue
        p = df.loc[df[col].astype(str).str.strip().ne("")].copy()
        if p.empty:
            continue
        p["_peso_perfil_cluster"] = pd.to_numeric(p.get(weight_col, 0), errors="coerce").fillna(0)
        if p["_peso_perfil_cluster"].sum() <= 0:
            p["_peso_perfil_cluster"] = 1.0
        g = p.groupby(keys + [col], dropna=False)["_peso_perfil_cluster"].sum().reset_index()
        total = g.groupby(keys, dropna=False)["_peso_perfil_cluster"].transform("sum")
        g[f"{col}_share_setor"] = np.where(total > 0, g["_peso_perfil_cluster"] / total, np.nan)
        g = g.sort_values(keys + ["_peso_perfil_cluster"], ascending=[True] * len(keys) + [False])
        top = g.drop_duplicates(keys).rename(columns={col: f"{col}_dominante_setor"})
        profile_parts.append(top[keys + [f"{col}_dominante_setor", f"{col}_share_setor"]])
    for p in profile_parts:
        out = out.merge(p, on=keys, how="left")
    for col in PROFILE_COLS:
        dom_col = f"{col}_dominante_setor"
        if dom_col in out.columns:
            out[col] = out[dom_col].map(lambda x: safe_text(x, ""))
        elif col not in out.columns:
            out[col] = ""

    out["eleitorado_cluster"] = np.where(
        pd.to_numeric(out["eleitorado_setor"], errors="coerce").fillna(0) > 0,
        pd.to_numeric(out["eleitorado_setor"], errors="coerce").fillna(0),
        pd.to_numeric(out["eleitorado"], errors="coerce").fillna(0),
    )
    out["comparecimento_cluster"] = np.where(
        pd.to_numeric(out["comparecimento_setor"], errors="coerce").fillna(0) > 0,
        pd.to_numeric(out["comparecimento_setor"], errors="coerce").fillna(0),
        pd.to_numeric(out["comparecimento_estimado"], errors="coerce").fillna(0),
    )
    out["abstencao_cluster"] = np.where(
        pd.to_numeric(out["abstencao_setor"], errors="coerce").fillna(0) > 0,
        pd.to_numeric(out["abstencao_setor"], errors="coerce").fillna(0),
        pd.to_numeric(out["abstencao_estimado"], errors="coerce").fillna(0),
    )
    out["pct_abstencao_cluster"] = np.where(
        out["eleitorado_cluster"] > 0,
        out["abstencao_cluster"] / out["eleitorado_cluster"],
        pd.to_numeric(out["pct_abstencao_setor"], errors="coerce"),
    )
    out["pct_comparecimento_cluster"] = np.where(
        out["eleitorado_cluster"] > 0,
        out["comparecimento_cluster"] / out["eleitorado_cluster"],
        pd.to_numeric(out["pct_comparecimento_setor"], errors="coerce"),
    )
    out["share_vencedor_setor"] = pd.to_numeric(out.get("share_vencedor_setor", np.nan), errors="coerce")
    out["regiao"] = out["uf"].astype(str).str.upper().map(REGION_BY_UF).fillna("Sem regiao")
    out["municipio_grupo"] = np.where(
        out["cd_municipio"].astype(str).str.strip().ne(""),
        out["uf"].astype(str) + "|" + out["cd_municipio"].astype(str),
        "SEM_MUNICIPIO",
    )

    out = add_discriminated_bins(out)
    return out


def cluster_discriminated_sector_base(
    sector_base: pd.DataFrame,
    cfg,
    categorical_source_cols: list[str] | None = None,
    analysis_label: str = "eleitores_resultado",
    include_results: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    if sector_base is None or sector_base.empty:
        empty = pd.DataFrame([{
            "status": "sem_base_correlacionada",
            "observacao": "Nao ha base correlacionada por codigo para clusterizar."
        }])
        return pd.DataFrame(), empty, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "# Clusters globais\n\nSem base correlacionada para clusterizar.\n"

    clustered = sector_base.copy()
    source_cols = categorical_source_cols or BASE_CATEGORICAL_COLS
    categorical_cols = [
        c for c in source_cols
        if c in clustered.columns and is_useful_discrete_series(clustered[c], col=c, max_categories=1000)
    ]
    categorical = cap_categorical_values(clustered, categorical_cols)
    valid_mask = useful_cluster_rows(categorical[categorical_cols] if categorical_cols else categorical, include_results=include_results)
    discarded_rows = int((~valid_mask).sum()) if len(valid_mask) else 0
    clustered = clustered.loc[valid_mask].copy()
    categorical = categorical.loc[valid_mask].copy()
    for col in categorical_cols:
        clustered[col] = categorical[col]
    token_rows = build_token_rows(categorical, categorical_cols)

    if not SKLEARN_OK or len(clustered) < 5 or not categorical_cols or not token_rows:
        clustered["cluster_global_discriminado"] = 0
        clustered["tipo_analise_cluster"] = analysis_label
        clustered["linhas_descartadas_sem_dado_discreto"] = discarded_rows
        summary, discriminants, year_region, entities, municipalities, personas, prediction_2026 = analyze_discriminated_clusters(clustered, categorical_cols, include_results=include_results, analysis_label=analysis_label)
        elbow = pd.DataFrame([{"k": 1, "inercia": np.nan, "metodo": "cotovelo", "k_escolhido": 1}])
        report = build_cluster_report(summary, discriminants, entities, algorithm_note="Cluster unico: scikit-learn indisponivel ou dados insuficientes.", include_results=include_results, analysis_label=analysis_label)
        return clustered, summary, discriminants, year_region, entities, municipalities, personas, prediction_2026, elbow, report

    hasher = FeatureHasher(n_features=HASH_FEATURES, input_type="string", alternate_sign=False)
    matrix = hasher.transform(token_rows)

    if matrix.shape[0] > MAX_CLUSTER_TRAIN_ROWS:
        rng = np.random.default_rng(42)
        train_idx = np.sort(rng.choice(matrix.shape[0], size=MAX_CLUSTER_TRAIN_ROWS, replace=False))
        train_matrix = matrix[train_idx]
    else:
        train_matrix = matrix

    min_k = max(2, int(getattr(cfg, "cluster_min_k", 4) or 4))
    max_k = max(min_k, int(getattr(cfg, "cluster_max_k", 12) or 12))
    max_k = min(max_k, max(2, train_matrix.shape[0] - 1), max(2, len(clustered) // 10))
    if max_k < min_k:
        min_k = 2
        max_k = min(4, max(2, train_matrix.shape[0] - 1))

    best_k, elbow = choose_k_by_elbow(train_matrix, min_k, max_k)

    model = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(train_matrix)
    clustered["cluster_global_discriminado"] = model.predict(matrix)
    clustered["algoritmo_cluster"] = "KMeans"
    clustered["tipo_analise_cluster"] = analysis_label
    clustered["tipo_features_cluster"] = (
        "valores_discretos_eleitores_hash_onehot"
        if not include_results else "valores_discretos_eleitores_resultado_hash_onehot"
    )
    clustered["metodo_escolha_k"] = "cotovelo_inercia"
    clustered["k_escolhido"] = best_k
    clustered["linhas_treinamento_cluster"] = int(train_matrix.shape[0])
    clustered["features_hash"] = HASH_FEATURES
    clustered["linhas_descartadas_sem_dado_discreto"] = discarded_rows

    summary, discriminants, year_region, entities, municipalities, personas, prediction_2026 = analyze_discriminated_clusters(clustered, categorical_cols, include_results=include_results, analysis_label=analysis_label)
    for df in [summary, discriminants, year_region, entities, municipalities, personas, prediction_2026]:
        if df is not None and not df.empty:
            df["algoritmo_cluster"] = "KMeans"
            df["tipo_analise_cluster"] = analysis_label
            df["tipo_features_cluster"] = (
                "valores_discretos_eleitores_hash_onehot"
                if not include_results else "valores_discretos_eleitores_resultado_hash_onehot"
            )
            df["metodo_escolha_k"] = "cotovelo_inercia"
            df["k_escolhido"] = best_k
            df["linhas_descartadas_sem_dado_discreto"] = discarded_rows
    report = build_cluster_report(summary, discriminants, entities, algorithm_note="KMeans aplicado sobre valores discretos/discriminados; k definido pela tecnica do cotovelo sobre a inercia.", include_results=include_results, analysis_label=analysis_label)
    return clustered, summary, discriminants, year_region, entities, municipalities, personas, prediction_2026, elbow, report


def analyze_discriminated_clusters(
    clustered: pd.DataFrame,
    categorical_cols: list[str],
    include_results: bool = True,
    analysis_label: str = "eleitores_resultado",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if clustered is None or clustered.empty or "cluster_global_discriminado" not in clustered.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    work = clustered.copy()
    for col in ["codigo_setor_eleitoral", "uf", "cd_municipio"]:
        if col not in work.columns:
            work[col] = ""
    if "votos_total_setor" not in work.columns:
        work["votos_total_setor"] = 0
    if "pct_abstencao_cluster" not in work.columns:
        work["pct_abstencao_cluster"] = pd.to_numeric(work.get("pct_abstencao_setor", np.nan), errors="coerce")
    if "pct_comparecimento_cluster" not in work.columns:
        work["pct_comparecimento_cluster"] = pd.to_numeric(work.get("pct_comparecimento_setor", np.nan), errors="coerce")
    cluster_col = "cluster_global_discriminado"
    summary = work.groupby(cluster_col, dropna=False).agg({
        "codigo_setor_eleitoral": "nunique",
        "uf": "nunique",
        "cd_municipio": "nunique",
        "votos_total_setor": "sum",
        "pct_abstencao_cluster": "mean",
        "pct_comparecimento_cluster": "mean",
    }).reset_index().rename(columns={
        "codigo_setor_eleitoral": "qtd_setores",
        "uf": "qtd_ufs",
        "cd_municipio": "qtd_municipios",
        "votos_total_setor": "votos_total_cluster",
        "pct_abstencao_cluster": "abstencao_media_cluster",
        "pct_comparecimento_cluster": "comparecimento_medio_cluster",
    })
    summary["qtd_linhas"] = work.groupby(cluster_col, dropna=False).size().values

    dominant_cols = ["regiao", "uf", "cargo", *PROFILE_COLS, "perfil_predominante_setor"]
    if include_results:
        dominant_cols.extend(["partido_vencedor_setor", "vencedor_setor"])
    for col in dominant_cols:
        if col in work.columns:
            summary[f"{col}_dominante"] = summary[cluster_col].map(
                lambda cl: _dominant_value(work.loc[work[cluster_col].eq(cl), col])
            )
            summary[f"{col}_dominante_share"] = summary[cluster_col].map(
                lambda cl: _dominant_share(work.loc[work[cluster_col].eq(cl), col])
            )

    summary["tipo_analise_cluster"] = analysis_label
    summary["interpretacao"] = summary.apply(lambda row: _summary_interpretation(row, include_results=include_results), axis=1)
    summary = summary.sort_values("qtd_linhas", ascending=False)

    discriminants = build_discriminant_table(work, categorical_cols)
    year_region = build_year_region_table(work)
    entities = build_entities_table(work) if include_results else pd.DataFrame()
    municipalities = build_municipality_cluster_table(work)
    prediction_2026 = build_cluster_prediction_2026(work) if include_results else pd.DataFrame()
    personas = build_cluster_personas(summary, prediction_2026, include_results=include_results)
    for df in [discriminants, year_region, entities, municipalities, personas, prediction_2026]:
        if df is not None and not df.empty:
            df["tipo_analise_cluster"] = analysis_label
    return summary, discriminants, year_region, entities, municipalities, personas, prediction_2026


def build_discriminant_table(work: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    rows = []
    cluster_col = "cluster_global_discriminado"
    total_rows = max(len(work), 1)
    for col in categorical_cols:
        if col not in work.columns:
            continue
        global_values = work[col].map(_meaningful_text)
        global_values = global_values.loc[global_values.ne("")]
        if global_values.empty:
            continue
        global_counts = global_values.value_counts(dropna=False)
        for cluster, g in work.groupby(cluster_col, dropna=False):
            cluster_values = g[col].map(_meaningful_text)
            cluster_values = cluster_values.loc[cluster_values.ne("")]
            counts = cluster_values.value_counts(dropna=False).head(8)
            for value, count in counts.items():
                share_cluster = float(count / max(len(g), 1))
                share_global = float(global_counts.get(value, 0) / total_rows)
                rows.append({
                    "cluster_global_discriminado": cluster,
                    "campo_discriminado": col,
                    "campo_discriminado_legivel": readable_field_label(col),
                    "valor_discriminado": value,
                    "qtd_no_cluster": int(count),
                    "share_no_cluster": share_cluster,
                    "share_global": share_global,
                    "lift_vs_global": share_cluster / share_global if share_global > 0 else np.nan,
                })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["abs_lift"] = pd.to_numeric(out["lift_vs_global"], errors="coerce").abs()
        out = out.sort_values(["cluster_global_discriminado", "abs_lift", "qtd_no_cluster"], ascending=[True, False, False])
    return out


def build_year_region_table(work: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["cluster_global_discriminado", "ano_correlacao", "regiao", "uf"] if c in work.columns]
    if not cols:
        return pd.DataFrame()
    out = work.groupby(cols, dropna=False).agg({
        "codigo_setor_eleitoral": "nunique",
        "votos_total_setor": "sum",
        "pct_abstencao_cluster": "mean",
        "pct_comparecimento_cluster": "mean",
    }).reset_index().rename(columns={
        "codigo_setor_eleitoral": "qtd_setores",
        "votos_total_setor": "votos_total",
        "pct_abstencao_cluster": "abstencao_media",
        "pct_comparecimento_cluster": "comparecimento_medio",
    })
    total_cluster = out.groupby("cluster_global_discriminado", dropna=False)["qtd_setores"].transform("sum")
    out["share_setores_no_cluster"] = np.where(total_cluster > 0, out["qtd_setores"] / total_cluster, np.nan)
    return out.sort_values(["cluster_global_discriminado", "qtd_setores"], ascending=[True, False])


def build_entities_table(work: pd.DataFrame) -> pd.DataFrame:
    if "vencedor_setor" not in work.columns:
        return pd.DataFrame()
    out = work.groupby(["cluster_global_discriminado", "vencedor_setor"], dropna=False).agg({
        "codigo_setor_eleitoral": "nunique",
        "votos_total_setor": "sum",
    }).reset_index().rename(columns={
        "codigo_setor_eleitoral": "qtd_setores",
        "votos_total_setor": "votos_total",
    })
    total_cluster = out.groupby("cluster_global_discriminado", dropna=False)["qtd_setores"].transform("sum")
    out["share_setores_no_cluster"] = np.where(total_cluster > 0, out["qtd_setores"] / total_cluster, np.nan)
    return out.sort_values(["cluster_global_discriminado", "qtd_setores"], ascending=[True, False]).groupby("cluster_global_discriminado").head(15)


def build_municipality_cluster_table(work: pd.DataFrame) -> pd.DataFrame:
    cols = [
        c for c in [
            "cluster_global_discriminado",
            "uf",
            "cd_municipio",
            "nm_municipio",
            "regiao",
        ]
        if c in work.columns
    ]
    if not cols:
        return pd.DataFrame()
    out = work.groupby(cols, dropna=False).agg({
        "codigo_setor_eleitoral": "nunique",
        "votos_total_setor": "sum",
        "pct_abstencao_cluster": "mean",
        "pct_comparecimento_cluster": "mean",
    }).reset_index().rename(columns={
        "codigo_setor_eleitoral": "qtd_setores",
        "votos_total_setor": "votos_total",
        "pct_abstencao_cluster": "abstencao_media",
        "pct_comparecimento_cluster": "comparecimento_medio",
    })
    total_cluster = out.groupby("cluster_global_discriminado", dropna=False)["qtd_setores"].transform("sum")
    out["share_setores_no_cluster"] = np.where(total_cluster > 0, out["qtd_setores"] / total_cluster, np.nan)
    return out.sort_values(["cluster_global_discriminado", "qtd_setores"], ascending=[True, False])


def build_cluster_prediction_2026(work: pd.DataFrame) -> pd.DataFrame:
    needed = {"cluster_global_discriminado", "ano_correlacao", "vencedor_setor", "votos_total_setor"}
    if work is None or work.empty or not needed.issubset(work.columns):
        return pd.DataFrame()

    df = work.copy()
    df["ano_num"] = pd.to_numeric(df["ano_correlacao"], errors="coerce")
    df["votos_total_setor"] = pd.to_numeric(df["votos_total_setor"], errors="coerce").fillna(0)
    df = df.loc[df["ano_num"].notna() & df["vencedor_setor"].astype(str).str.strip().ne("")]
    if df.empty:
        return pd.DataFrame()

    group = ["cluster_global_discriminado", "ano_num", "cargo", "turno", "vencedor_setor"]
    group = [c for c in group if c in df.columns]
    agg = df.groupby(group, dropna=False)["votos_total_setor"].sum().reset_index()
    denom_group = [c for c in ["cluster_global_discriminado", "ano_num", "cargo", "turno"] if c in agg.columns]
    total = agg.groupby(denom_group, dropna=False)["votos_total_setor"].transform("sum")
    agg["share_cluster_ano"] = np.where(total > 0, agg["votos_total_setor"] / total, np.nan)

    rows = []
    entity_cols = [c for c in ["cluster_global_discriminado", "cargo", "turno", "vencedor_setor"] if c in agg.columns]
    for keys, g in agg.groupby(entity_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(entity_cols, keys))
        g = g.sort_values("ano_num")
        last = g.iloc[-1]
        if len(g) >= 2:
            prev = g.iloc[-2]
            year_delta = max(float(last["ano_num"] - prev["ano_num"]), 1.0)
            swing_anual = (float(last["share_cluster_ano"]) - float(prev["share_cluster_ano"])) / year_delta
        else:
            swing_anual = 0.0
        years_to_2026 = max(0.0, 2026.0 - float(last["ano_num"]))
        row.update({
            "ano_base": int(float(last["ano_num"])),
            "share_base_cluster": float(last["share_cluster_ano"]) if pd.notna(last["share_cluster_ano"]) else np.nan,
            "swing_anual_cluster": float(swing_anual),
            "share_previsto_2026_raw": max(0.000001, float(last["share_cluster_ano"] or 0) + swing_anual * years_to_2026),
            "anos_ate_2026": years_to_2026,
        })
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    norm_group = [c for c in ["cluster_global_discriminado", "cargo", "turno"] if c in out.columns]
    total_pred = out.groupby(norm_group, dropna=False)["share_previsto_2026_raw"].transform("sum")
    out["share_previsto_2026"] = np.where(total_pred > 0, out["share_previsto_2026_raw"] / total_pred, np.nan)
    out["rank_pred_2026_cluster"] = out.groupby(norm_group, dropna=False)["share_previsto_2026"].rank(method="first", ascending=False)
    out["interpretacao"] = out.apply(
        lambda r: (
            f"No cluster {r.get('cluster_global_discriminado')}, {r.get('vencedor_setor')} fica projetado para "
            f"{float(r.get('share_previsto_2026')) * 100:.1f}% em 2026 pelo historico discreto do proprio cluster."
            if pd.notna(r.get("share_previsto_2026")) else "Sem base suficiente para previsao do cluster."
        ),
        axis=1,
    )
    return out.sort_values(["cluster_global_discriminado", "rank_pred_2026_cluster"], ascending=[True, True])


def build_cluster_personas(summary: pd.DataFrame, prediction_2026: pd.DataFrame, include_results: bool = True) -> pd.DataFrame:
    if summary is None or summary.empty or "cluster_global_discriminado" not in summary.columns:
        return pd.DataFrame()

    pred_top = pd.DataFrame()
    if prediction_2026 is not None and not prediction_2026.empty and "rank_pred_2026_cluster" in prediction_2026.columns:
        pred_top = prediction_2026.loc[pd.to_numeric(prediction_2026["rank_pred_2026_cluster"], errors="coerce").eq(1)].copy()
        keep = [c for c in ["cluster_global_discriminado", "vencedor_setor", "share_previsto_2026"] if c in pred_top.columns]
        pred_top = pred_top[keep].rename(columns={
            "vencedor_setor": "entidade_prevista_2026",
            "share_previsto_2026": "share_entidade_prevista_2026",
        })
        if "share_entidade_prevista_2026" in pred_top.columns:
            pred_top["share_entidade_prevista_2026"] = pd.to_numeric(pred_top["share_entidade_prevista_2026"], errors="coerce")
            pred_top = pred_top.sort_values(["cluster_global_discriminado", "share_entidade_prevista_2026"], ascending=[True, False])
        pred_top = pred_top.drop_duplicates("cluster_global_discriminado")

    out = summary.copy()
    if not pred_top.empty:
        out = out.merge(pred_top, on="cluster_global_discriminado", how="left")
    else:
        out["entidade_prevista_2026"] = ""
        out["share_entidade_prevista_2026"] = np.nan

    def _persona(row: pd.Series) -> str:
        age = _meaningful_text(row.get("perfil_faixa_etaria_dominante", ""))
        gender = _meaningful_text(row.get("perfil_genero_dominante", ""))
        education = _meaningful_text(row.get("perfil_instrucao_dominante", ""))
        civil = _meaningful_text(row.get("perfil_estado_civil_dominante", ""))
        race = _meaningful_text(row.get("perfil_raca_cor_dominante", ""))
        region = _meaningful_text(row.get("regiao_dominante", ""))
        uf = _meaningful_text(row.get("uf_dominante", ""))
        party = _meaningful_text(row.get("partido_vencedor_setor_dominante", ""))
        entity = _meaningful_text(row.get("vencedor_setor_dominante", ""))
        pred = _meaningful_text(row.get("entidade_prevista_2026", ""))
        traits = []
        if age:
            traits.append(f"na faixa {age}")
        if gender:
            traits.append(gender.lower())
        if education:
            traits.append(f"com {education.lower()}")
        if civil:
            traits.append(civil.lower())
        if race:
            traits.append(f"de raca/cor {race.lower()}")
        where = " / ".join([x for x in [region, uf] if x])
        person = "Eleitor " + ", ".join(traits) if traits else "Eleitor com perfil discreto ainda incompleto"
        if where:
            person += f", mais presente em {where}"
        if include_results:
            vote = pred or entity or party
            if vote:
                return f"{person}. Esse grupo tende a se aproximar de {vote}."
            return f"{person}. Resultado eleitoral dominante ainda sem confianca suficiente."
        return f"{person}. Este cluster descreve apenas o perfil do eleitor, sem usar candidato ou partido."

    out["pessoa_do_cluster"] = out.apply(_persona, axis=1)
    cols = [
        "cluster_global_discriminado",
        "pessoa_do_cluster",
        "perfil_faixa_etaria_dominante",
        "perfil_genero_dominante",
        "perfil_instrucao_dominante",
        "perfil_estado_civil_dominante",
        "perfil_raca_cor_dominante",
        "regiao_dominante",
        "uf_dominante",
        "qtd_setores",
        "qtd_municipios",
        "votos_total_cluster",
        "abstencao_media_cluster",
        "comparecimento_medio_cluster",
    ]
    if include_results:
        cols[7:7] = [
            "partido_vencedor_setor_dominante",
            "vencedor_setor_dominante",
            "entidade_prevista_2026",
            "share_entidade_prevista_2026",
        ]
    cols = [c for c in cols if c in out.columns]
    return out[cols].sort_values("qtd_setores", ascending=False, na_position="last")


def choose_k_by_elbow(matrix, min_k: int, max_k: int) -> tuple[int, pd.DataFrame]:
    rows = []
    if max_k < min_k:
        min_k = max_k
    for k in range(min_k, max_k + 1):
        try:
            model = KMeans(n_clusters=k, random_state=42, n_init=10).fit(matrix)
            rows.append({"k": k, "inercia": float(model.inertia_)})
        except Exception as exc:
            logging.warning("Falha calculando cotovelo k=%s: %s", k, exc)

    elbow = pd.DataFrame(rows)
    if elbow.empty:
        return max(1, min_k), pd.DataFrame([{"k": max(1, min_k), "inercia": np.nan, "metodo": "cotovelo", "k_escolhido": max(1, min_k)}])
    if len(elbow) == 1 or elbow["inercia"].nunique(dropna=True) <= 1:
        best_k = int(elbow["k"].iloc[0])
    else:
        x = elbow["k"].astype(float).to_numpy()
        y = elbow["inercia"].astype(float).to_numpy()
        x_norm = (x - x.min()) / max(x.max() - x.min(), 1e-9)
        y_norm = (y - y.min()) / max(y.max() - y.min(), 1e-9)
        start = np.array([x_norm[0], y_norm[0]])
        end = np.array([x_norm[-1], y_norm[-1]])
        line = end - start
        denom = np.linalg.norm(line) or 1e-9
        distances = [
            abs(line[0] * (yi - start[1]) - line[1] * (xi - start[0])) / denom
            for xi, yi in zip(x_norm, y_norm)
        ]
        best_k = int(elbow.iloc[int(np.argmax(distances))]["k"])
        elbow["distancia_cotovelo"] = distances
    elbow["metodo"] = "cotovelo_inercia"
    elbow["k_escolhido"] = best_k
    return best_k, elbow


def build_cluster_report(
    summary: pd.DataFrame,
    discriminants: pd.DataFrame,
    entities: pd.DataFrame,
    algorithm_note: str = "KMeans aplicado sobre valores discriminados via hash one-hot.",
    include_results: bool = True,
    analysis_label: str = "eleitores_resultado",
) -> str:
    lines = [
        "# Clusters globais - " + ("eleitores + resultado" if include_results else "somente eleitores"),
        "",
        algorithm_note,
        "",
        "Unidade de cluster: setor eleitoral/ano/cargo/turno da base correlacionada.",
        (
            "Este bloco usa perfil do eleitor, territorio e resultado eleitoral."
            if include_results
            else "Este bloco usa perfil do eleitor, territorio e participacao eleitoral; nao usa partido, candidato ou vencedor como feature."
        ),
        "",
    ]

    if summary is None or summary.empty:
        lines.append("Sem clusters para interpretar.")
        return "\n".join(lines)

    for _, row in summary.sort_values("qtd_linhas", ascending=False).iterrows():
        cluster = row.get("cluster_global_discriminado", "")
        lines.extend([
            f"## Cluster {cluster}",
            "",
            f"- Setores: {row.get('qtd_setores', 0)}",
            f"- Municipios: {row.get('qtd_municipios', 0)}",
            f"- Regiao dominante: {row.get('regiao_dominante', '')} ({_pct(row.get('regiao_dominante_share'))})",
            f"- Faixa etaria dominante: {row.get('perfil_faixa_etaria_dominante', '')}",
            f"- Sexo/genero dominante: {row.get('perfil_genero_dominante', '')}",
            f"- Escolaridade dominante: {row.get('perfil_instrucao_dominante', '')}",
            f"- Perfil predominante: {row.get('perfil_predominante_setor_dominante', '')}",
            f"- Votos no cluster: {row.get('votos_total_cluster', 0)}",
            f"- Abstencao media: {_pct(row.get('abstencao_media_cluster'))}",
            f"- Comparecimento medio: {_pct(row.get('comparecimento_medio_cluster'))}",
            "",
            str(row.get("interpretacao", "")),
            "",
        ])
        if include_results:
            result_lines = []
            party = _meaningful_text(row.get("partido_vencedor_setor_dominante", ""))
            entity = _meaningful_text(row.get("vencedor_setor_dominante", ""))
            if party:
                result_lines.append(f"- Partido vencedor dominante: {party}")
            if entity:
                result_lines.append(f"- Entidade vencedora dominante: {entity}")
            insert_at = max(len(lines) - 5, 0)
            lines[insert_at:insert_at] = result_lines

        if discriminants is not None and not discriminants.empty:
            top = discriminants.loc[discriminants["cluster_global_discriminado"].astype(str).eq(str(cluster))].head(8)
            if not top.empty:
                lines.append("Valores discriminantes mais fortes:")
                for _, d in top.iterrows():
                    lines.append(
                        f"- {d.get('campo_discriminado')}={d.get('valor_discriminado')} "
                        f"(share {_pct(d.get('share_no_cluster'))}, lift {float(pd.to_numeric(d.get('lift_vs_global'), errors='coerce')):.2f}x)"
                    )
                lines.append("")

        if include_results and entities is not None and not entities.empty:
            top_ent = entities.loc[entities["cluster_global_discriminado"].astype(str).eq(str(cluster))].head(5)
            if not top_ent.empty:
                lines.append("Entidades vencedoras mais frequentes:")
                for _, e in top_ent.iterrows():
                    lines.append(f"- {e.get('vencedor_setor')}: {e.get('qtd_setores')} setores")
                lines.append("")

    return "\n".join(lines)


def plot_discriminated_clusters(
    clustered: pd.DataFrame,
    summary: pd.DataFrame,
    discriminants: pd.DataFrame,
    personas: pd.DataFrame,
    prediction_2026: pd.DataFrame,
    elbow: pd.DataFrame,
    plots_dir: Path,
    cfg,
) -> list[Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    images: list[Path] = []
    if not MATPLOTLIB_OK:
        return images

    if summary is not None and not summary.empty and "cluster_global_discriminado" in summary.columns:
        tmp = summary.copy()
        tmp["cluster_plot"] = tmp["cluster_global_discriminado"].astype(str)
        tmp["qtd_linhas"] = pd.to_numeric(tmp.get("qtd_linhas", 0), errors="coerce").fillna(0)
        tmp = tmp.sort_values("qtd_linhas", ascending=False)
        if tmp["qtd_linhas"].sum() > 0:
            plt.figure(figsize=(10, 4.8))
            plt.bar(tmp["cluster_plot"], tmp["qtd_linhas"])
            plt.title("Clusters globais discriminados - quantidade de setores")
            plt.xlabel("cluster")
            plt.ylabel("setores/recortes")
            plt.tight_layout()
            path = plots_dir / "clusters_globais_discriminados_tamanho.png"
            plt.savefig(path, dpi=140)
            plt.close()
            images.append(path)

        if "abstencao_media_cluster" in tmp.columns:
            vals = pd.to_numeric(tmp["abstencao_media_cluster"], errors="coerce")
            if vals.notna().any():
                plt.figure(figsize=(10, 4.8))
                plt.bar(tmp["cluster_plot"], vals.fillna(0) * 100)
                plt.title("Clusters globais discriminados - abstencao media")
                plt.xlabel("cluster")
                plt.ylabel("abstencao media (%)")
                plt.tight_layout()
                path = plots_dir / "clusters_globais_discriminados_abstencao.png"
                plt.savefig(path, dpi=140)
                plt.close()
                images.append(path)

    if personas is not None and not personas.empty and "cluster_global_discriminado" in personas.columns:
        for field, title, filename in [
            ("perfil_faixa_etaria_dominante", "Pessoa do cluster - faixa etaria dominante", "clusters_persona_faixa_etaria.png"),
            ("perfil_genero_dominante", "Pessoa do cluster - genero dominante", "clusters_persona_genero.png"),
            ("perfil_instrucao_dominante", "Pessoa do cluster - escolaridade dominante", "clusters_persona_escolaridade.png"),
            ("entidade_prevista_2026", "Pessoa do cluster - tendencia prevista 2026", "clusters_persona_predicao_2026.png"),
        ]:
            if field not in personas.columns:
                continue
            tmp = personas.copy()
            tmp[field] = tmp[field].map(lambda x: safe_text(x, "Sem dado"))
            counts = tmp[field].value_counts().head(20)
            if counts.empty:
                continue
            plt.figure(figsize=(11, max(4.8, len(counts) * 0.34)))
            plt.barh(counts.index.astype(str), counts.values)
            plt.title(title)
            plt.gca().invert_yaxis()
            plt.tight_layout()
            path = plots_dir / filename
            plt.savefig(path, dpi=140)
            plt.close()
            images.append(path)

    if prediction_2026 is not None and not prediction_2026.empty and {"rank_pred_2026_cluster", "vencedor_setor", "share_previsto_2026"}.issubset(prediction_2026.columns):
        pred = prediction_2026.loc[pd.to_numeric(prediction_2026["rank_pred_2026_cluster"], errors="coerce").le(3)].copy()
        pred["share_previsto_2026"] = pd.to_numeric(pred["share_previsto_2026"], errors="coerce")
        pred = pred.sort_values(["cluster_global_discriminado", "rank_pred_2026_cluster"]).head(36)
        if not pred.empty:
            labels = (
                "C" + pred["cluster_global_discriminado"].astype(str)
                + " | "
                + pred["vencedor_setor"].astype(str).str.slice(0, 34)
            )
            plt.figure(figsize=(12, max(5, len(labels) * 0.30)))
            plt.barh(labels, pred["share_previsto_2026"].fillna(0) * 100)
            plt.title("Clusters - predicao 2026 por entidade")
            plt.xlabel("share previsto no cluster (%)")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            path = plots_dir / "clusters_predicao_2026.png"
            plt.savefig(path, dpi=140)
            plt.close()
            images.append(path)

    if elbow is not None and not elbow.empty and {"k", "inercia"}.issubset(elbow.columns):
        tmp = elbow.copy()
        tmp["k"] = pd.to_numeric(tmp["k"], errors="coerce")
        tmp["inercia"] = pd.to_numeric(tmp["inercia"], errors="coerce")
        tmp = tmp.dropna(subset=["k", "inercia"]).sort_values("k")
        if not tmp.empty:
            plt.figure(figsize=(8.8, 4.8))
            plt.plot(tmp["k"], tmp["inercia"], marker="o")
            chosen = pd.to_numeric(tmp.get("k_escolhido", pd.Series(dtype=float)), errors="coerce").dropna()
            if not chosen.empty:
                plt.axvline(float(chosen.iloc[0]), color="red", linestyle="--", linewidth=1)
            plt.title("Tecnica do cotovelo - escolha de k")
            plt.xlabel("quantidade de clusters (k)")
            plt.ylabel("inercia")
            plt.tight_layout()
            path = plots_dir / "clusters_cotovelo_k.png"
            plt.savefig(path, dpi=140)
            plt.close()
            images.append(path)

    if clustered is not None and not clustered.empty and {"cluster_global_discriminado", "regiao"}.issubset(clustered.columns):
        reg = clustered.groupby(["cluster_global_discriminado", "regiao"], dropna=False).size().reset_index(name="qtd")
        reg = reg.sort_values(["cluster_global_discriminado", "qtd"], ascending=[True, False]).groupby("cluster_global_discriminado").head(5)
        if not reg.empty:
            labels = reg["cluster_global_discriminado"].astype(str) + " | " + reg["regiao"].astype(str)
            plt.figure(figsize=(11, max(5, len(labels) * 0.28)))
            plt.barh(labels, pd.to_numeric(reg["qtd"], errors="coerce").fillna(0))
            plt.title("Clusters globais discriminados - principais regioes")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            path = plots_dir / "clusters_globais_discriminados_regioes.png"
            plt.savefig(path, dpi=140)
            plt.close()
            images.append(path)

    if discriminants is not None and not discriminants.empty:
        top = discriminants.copy()
        top["lift_vs_global"] = pd.to_numeric(top["lift_vs_global"], errors="coerce")
        top = top.dropna(subset=["lift_vs_global"]).sort_values("lift_vs_global", ascending=False).head(getattr(cfg, "top_n_plots", 20))
        if not top.empty:
            labels = (
                top["cluster_global_discriminado"].astype(str)
                + " | "
                + top["campo_discriminado"].astype(str)
                + "="
                + top["valor_discriminado"].astype(str).str.slice(0, 40)
            )
            plt.figure(figsize=(12, max(5, len(labels) * 0.32)))
            plt.barh(labels, top["lift_vs_global"])
            plt.title("Clusters globais discriminados - valores com maior lift")
            plt.xlabel("lift contra distribuicao global")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            path = plots_dir / "clusters_globais_discriminados_lift.png"
            plt.savefig(path, dpi=140)
            plt.close()
            images.append(path)

    clean_memory()
    return images


def add_discriminated_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["faixa_votos_setor"] = quantile_band(out["votos_total_setor"], "votos")
    out["faixa_abstencao_setor"] = fixed_rate_band(out["pct_abstencao_cluster"], "abstencao", low=0.12, high=0.25)
    out["faixa_comparecimento_setor"] = fixed_rate_band(out["pct_comparecimento_cluster"], "comparecimento", low=0.65, high=0.82)
    return out


def cap_categorical_values(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        s = out[col].map(lambda x, c=col: label_category_value(x, col=c))
        s = s.map(lambda x: _meaningful_text(x))
        limit = CATEGORY_LIMITS.get(col, 180)
        top = set(s.loc[s.astype(str).str.strip().ne("")].value_counts(dropna=False).head(limit).index.astype(str))
        out[col] = np.where(s.astype(str).isin(top), s.astype(str), "")
    return out


def useful_cluster_rows(df: pd.DataFrame, include_results: bool) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=bool)
    profile_cols = [c for c in PROFILE_COLS if c in df.columns]
    if profile_cols:
        profile_ok = df[profile_cols].apply(lambda col: col.map(_meaningful_text).ne("")).any(axis=1)
    else:
        profile_ok = pd.Series(False, index=df.index)
    token_count = df.apply(lambda row: sum(1 for v in row.values if _meaningful_text(v)), axis=1)
    mask = profile_ok & token_count.ge(3)
    if include_results:
        result_cols = [c for c in ["partido_vencedor_setor", "candidato_vencedor_setor", "vencedor_setor"] if c in df.columns]
        if result_cols:
            result_ok = df[result_cols].apply(lambda col: col.map(_meaningful_text).ne("")).any(axis=1)
        else:
            result_ok = pd.Series(False, index=df.index)
        mask = mask & result_ok
    return mask.fillna(False)


def build_token_rows(df: pd.DataFrame, cols: list[str]) -> list[list[str]]:
    rows = []
    for _, row in df.iterrows():
        tokens = []
        for col in cols:
            value = _meaningful_text(row.get(col, ""))
            if value:
                tokens.append(f"{col}={value}")
        rows.append(tokens)
    return rows


def quantile_band(series: pd.Series, label: str) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() < 4 or s.nunique(dropna=True) <= 1:
        return pd.Series([f"{label}_sem_variacao" if pd.notna(v) else f"{label}_sem_valor" for v in s], index=series.index)
    try:
        return pd.qcut(
            s.rank(method="first"),
            q=4,
            labels=[f"{label}_baixo", f"{label}_medio_baixo", f"{label}_medio_alto", f"{label}_alto"],
            duplicates="drop",
        ).astype(str).replace({"nan": f"{label}_sem_valor"})
    except Exception:
        return pd.Series([f"{label}_sem_valor"] * len(series), index=series.index)


def fixed_rate_band(series: pd.Series, label: str, low: float, high: float) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return pd.Series(np.select(
        [s.isna(), s < low, s <= high, s > high],
        [f"{label}_sem_valor", f"{label}_baixo", f"{label}_medio", f"{label}_alto"],
        default=f"{label}_sem_valor",
    ), index=series.index)


def _load_correlated_base(outputs: dict[str, Any]) -> pd.DataFrame:
    pq_value = safe_text(outputs.get("base_correlacionada_codigo_parquet", ""))
    csv_value = safe_text(outputs.get("base_correlacionada_codigo_csv", ""))
    pq = Path(pq_value) if pq_value else None
    csv = Path(csv_value) if csv_value else None
    needed_cols = [
        "ano_correlacao", "uf", "cd_municipio", "nm_municipio", "zona", "secao",
        "codigo_setor_eleitoral", "codigo_correlacao_setor_ano", "cargo", "turno",
        "partido", "candidato", "entidade", "vencedor_setor", "perfil_predominante_setor",
        *PROFILE_COLS,
        "votos", "eleitorado", "eleitorado_setor", "comparecimento_estimado",
        "comparecimento_setor", "abstencao_estimado", "abstencao_setor",
        "share_votos_setor", "pct_abstencao_setor", "pct_comparecimento_setor",
        "rank_entidade_setor",
    ]
    if pq is not None and pq.exists() and pq.is_file():
        try:
            try:
                return pd.read_parquet(pq, columns=needed_cols)
            except Exception:
                return pd.read_parquet(pq)
        except Exception as exc:
            logging.warning("Falha lendo base correlacionada parquet %s: %s", pq, exc)
    if csv is not None and csv.exists() and csv.is_file():
        try:
            return pd.read_csv(csv, sep=";", dtype=str)
        except Exception as exc:
            logging.warning("Falha lendo base correlacionada csv %s: %s", csv, exc)
    return pd.DataFrame()


def _dominant_value(series: pd.Series) -> str:
    s = series.map(_meaningful_text)
    s = s.loc[s.ne("")]
    vc = s.value_counts(dropna=False)
    return safe_text(vc.index[0], "") if not vc.empty else ""


def _dominant_share(series: pd.Series) -> float:
    s = series.map(_meaningful_text)
    s = s.loc[s.ne("")]
    vc = s.value_counts(dropna=False)
    return float(vc.iloc[0] / max(len(series), 1)) if not vc.empty else np.nan


def _meaningful_text(value: Any) -> str:
    text = safe_text(value, "").strip()
    lower = text.lower()
    code_value = lower.replace("codigo ", "", 1).replace("código ", "", 1).replace(".", "", 1).lstrip("-+")
    if lower in {"", "sem valor", "sem_valor", "nan", "none", "null", "<na>", "#nulo#", "geral", "nao informado", "não informado"}:
        return ""
    if lower.endswith("_sem_valor") or lower.endswith(" sem valor"):
        return ""
    if (lower.startswith("codigo ") or lower.startswith("código ")) and code_value.isdigit():
        return ""
    return text


def _summary_interpretation(row: pd.Series, include_results: bool = True) -> str:
    parts = [
        f"cluster dominado por {_meaningful_text(row.get('regiao_dominante', '')) or 'regiao indefinida'}",
    ]
    profile = _meaningful_text(row.get("perfil_predominante_setor_dominante", ""))
    if profile:
        parts.append(f"perfil predominante {profile}")
    party = _meaningful_text(row.get("partido_vencedor_setor_dominante", ""))
    if include_results and party:
        parts.append(f"partido vencedor recorrente {party}")
    abst = pd.to_numeric(row.get("abstencao_media_cluster"), errors="coerce")
    if pd.notna(abst):
        parts.append(f"abstencao media {abst * 100:.1f}%")
    comp = pd.to_numeric(row.get("comparecimento_medio_cluster"), errors="coerce")
    if pd.notna(comp):
        parts.append(f"comparecimento medio {comp * 100:.1f}%")
    return "Este cluster agrupa setores com " + "; ".join(parts) + "."


def _pct(value: Any) -> str:
    num = pd.to_numeric(value, errors="coerce")
    if pd.isna(num):
        return "sem dado"
    return f"{float(num) * 100:.1f}%"
