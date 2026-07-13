from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import save_csv, save_parquet, safe_text
from .discrete import label_category_value

try:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction import FeatureHasher
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


PROFILE_COLS = [
    "perfil_faixa_etaria",
    "perfil_genero",
    "perfil_instrucao",
    "perfil_estado_civil",
    "perfil_raca_cor",
]

REGION_SECTION_COLS = [
    "uf",
    "cd_municipio",
    "nm_municipio",
    "zona",
    "secao",
    "local_votacao",
    "bairro",
]

ELECTION_COLS = ["ano", "cargo", "turno"]

MAX_BEHAVIOR_ENTITIES = 250
MAX_BEHAVIOR_TRAIN_ROWS = 50000
HASH_FEATURES = 2048


def _clean_gold(gold: pd.DataFrame) -> pd.DataFrame:
    if gold is None or gold.empty:
        return pd.DataFrame()

    df = gold.copy()

    text_cols = ELECTION_COLS + REGION_SECTION_COLS + PROFILE_COLS + [
        "partido",
        "candidato",
        "ideologia",
        "coalizao",
    ]
    for col in text_cols:
        if col not in df.columns:
            df[col] = ""
        if col in PROFILE_COLS or col in {"cargo", "turno"}:
            df[col] = df[col].map(lambda x, c=col: label_category_value(x, col=c))
        else:
            df[col] = df[col].map(lambda x: safe_text(x, ""))

    numeric_cols = [
        "votos",
        "eleitorado",
        "comparecimento_estimado",
        "abstencao_estimado",
        "brancos",
        "nulos",
        "validos_estimados",
    ]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["ano_num"] = pd.to_numeric(df["ano"], errors="coerce")
    return df


def _available_profiles(df: pd.DataFrame) -> list[str]:
    return [
        col for col in PROFILE_COLS
        if col in df.columns and df[col].map(_meaningful_text).ne("").any()
    ]


def _entity_label(df: pd.DataFrame) -> pd.Series:
    cand_ok = "candidato" in df.columns and df["candidato"].astype(str).str.strip().ne("").any()
    part_ok = "partido" in df.columns and df["partido"].astype(str).str.strip().ne("").any()

    if cand_ok and part_ok:
        return df["candidato"].astype(str) + " | " + df["partido"].astype(str)
    if cand_ok:
        return df["candidato"].astype(str)
    if part_ok:
        return df["partido"].astype(str)
    return pd.Series(["GERAL"] * len(df), index=df.index)


def _meaningful_text(value: Any) -> str:
    text = safe_text(value, "").strip()
    lower = text.lower()
    code_value = lower.replace("codigo ", "", 1).replace("código ", "", 1).replace(".", "", 1).lstrip("-+")
    if lower in {"", "sem valor", "sem_valor", "nan", "none", "null", "<na>", "#nulo#", "geral", "sem_entidade", "nao informado", "não informado"}:
        return ""
    if lower.endswith("_sem_valor") or lower.endswith(" sem valor"):
        return ""
    if (lower.startswith("codigo ") or lower.startswith("código ")) and code_value.isdigit():
        return ""
    return text


def _trend_label(first: float, last: float, tolerance: float = 0.03) -> str:
    if pd.isna(first) or pd.isna(last):
        return "sem série suficiente"
    if first == 0 and last > 0:
        return "crescimento"
    if first == 0 and last == 0:
        return "estável"
    delta = (last - first) / abs(first)
    if delta > tolerance:
        return "crescimento"
    if delta < -tolerance:
        return "queda"
    return "estável"


def _crystallization_level(value: float) -> str:
    if pd.isna(value):
        return "sem dados"
    if value >= 0.65:
        return "alto"
    if value >= 0.40:
        return "médio"
    return "baixo"


def build_vote_profile_section_panel(gold: pd.DataFrame) -> pd.DataFrame:
    """
    Base central da análise comportamental:
    ano × cargo × turno × região/seção × perfil do eleitorado × candidato/partido.
    """
    df = _clean_gold(gold)
    if df.empty:
        return pd.DataFrame()

    profile_cols = _available_profiles(df)
    df["entidade_voto"] = _entity_label(df)

    group_cols = ELECTION_COLS + REGION_SECTION_COLS + profile_cols + [
        "partido",
        "candidato",
        "entidade_voto",
    ]
    group_cols = [c for c in group_cols if c in df.columns]

    panel = df.groupby(group_cols, dropna=False).agg({
        "votos": "sum",
        "eleitorado": "max",
        "comparecimento_estimado": "sum",
        "abstencao_estimado": "sum",
        "brancos": "sum",
        "nulos": "sum",
    }).reset_index()

    denominator_group = ELECTION_COLS + REGION_SECTION_COLS + profile_cols
    denominator_group = [c for c in denominator_group if c in panel.columns]
    total = panel.groupby(denominator_group, dropna=False)["votos"].transform("sum")
    panel["share_voto_no_perfil_secao"] = np.where(total > 0, panel["votos"] / total, np.nan)
    panel["taxa_abstencao_perfil_secao"] = np.where(
        panel["eleitorado"] > 0,
        panel["abstencao_estimado"] / panel["eleitorado"],
        np.nan,
    )
    panel["taxa_comparecimento_perfil_secao"] = np.where(
        panel["eleitorado"] > 0,
        panel["comparecimento_estimado"] / panel["eleitorado"],
        np.nan,
    )
    return panel


def build_who_votes_for_whom(panel: pd.DataFrame) -> pd.DataFrame:
    if panel is None or panel.empty:
        return pd.DataFrame()

    profile_cols = _available_profiles(panel)
    group_cols = ELECTION_COLS + REGION_SECTION_COLS + profile_cols
    group_cols = [c for c in group_cols if c in panel.columns]

    out = panel.copy()
    out["rank_entidade_no_perfil_secao"] = out.groupby(group_cols, dropna=False)["votos"].rank(
        method="first",
        ascending=False,
    )
    top = out.loc[out["rank_entidade_no_perfil_secao"].eq(1)].copy()
    top = top.rename(columns={
        "entidade_voto": "entidade_dominante",
        "votos": "votos_entidade_dominante",
        "share_voto_no_perfil_secao": "share_entidade_dominante",
    })

    cols = group_cols + [
        "entidade_dominante",
        "partido",
        "candidato",
        "votos_entidade_dominante",
        "share_entidade_dominante",
        "taxa_abstencao_perfil_secao",
        "taxa_comparecimento_perfil_secao",
    ]
    cols = [c for c in cols if c in top.columns]
    return top[cols].sort_values([c for c in ["ano", "uf", "cd_municipio", "zona", "secao"] if c in top.columns])


def build_candidate_profile_affinity(panel: pd.DataFrame) -> pd.DataFrame:
    if panel is None or panel.empty:
        return pd.DataFrame()

    profile_cols = _available_profiles(panel)
    if not profile_cols:
        return pd.DataFrame([{
            "status": "sem_perfil_eleitor",
            "observacao": "Não há campos de perfil do eleitorado para medir afinidade perfil-candidato."
        }])

    prof_group = ELECTION_COLS + profile_cols + ["entidade_voto"]
    prof = panel.groupby(prof_group, dropna=False)["votos"].sum().reset_index()
    total_profile = prof.groupby(ELECTION_COLS + profile_cols, dropna=False)["votos"].transform("sum")
    prof["share_entidade_no_perfil"] = np.where(total_profile > 0, prof["votos"] / total_profile, np.nan)

    glob = panel.groupby(ELECTION_COLS + ["entidade_voto"], dropna=False)["votos"].sum().reset_index()
    total_global = glob.groupby(ELECTION_COLS, dropna=False)["votos"].transform("sum")
    glob["share_global_entidade"] = np.where(total_global > 0, glob["votos"] / total_global, np.nan)

    out = prof.merge(
        glob[ELECTION_COLS + ["entidade_voto", "share_global_entidade"]],
        on=ELECTION_COLS + ["entidade_voto"],
        how="left",
    )
    out["lift_perfil_entidade"] = np.where(
        out["share_global_entidade"] > 0,
        out["share_entidade_no_perfil"] / out["share_global_entidade"],
        np.nan,
    )
    out["interpretacao"] = out.apply(
        lambda r: (
            f"Este perfil vota {float(r['lift_perfil_entidade']):.2f}x mais em {r['entidade_voto']} do que a média geral."
            if pd.notna(r.get("lift_perfil_entidade")) else "Sem base para cálculo."
        ),
        axis=1,
    )
    return out.sort_values("lift_perfil_entidade", ascending=False, na_position="last")


def build_section_behavior_matrix(panel: pd.DataFrame) -> pd.DataFrame:
    if panel is None or panel.empty:
        return pd.DataFrame()

    profile_cols = _available_profiles(panel)
    index_cols = ELECTION_COLS + REGION_SECTION_COLS + profile_cols
    index_cols = [c for c in index_cols if c in panel.columns]

    work = panel.copy()
    if "entidade_voto" in work.columns:
        top_entities = (
            work.groupby("entidade_voto", dropna=False)["votos"]
            .sum()
            .sort_values(ascending=False)
            .head(MAX_BEHAVIOR_ENTITIES)
            .index
            .astype(str)
            .tolist()
        )
        work["entidade_voto"] = np.where(
            work["entidade_voto"].astype(str).isin(top_entities),
            work["entidade_voto"].astype(str),
            "OUTRAS_ENTIDADES",
        )
        agg_map = {
            "votos": "sum",
            "eleitorado": "max",
            "comparecimento_estimado": "max",
            "abstencao_estimado": "max",
            "brancos": "sum",
            "nulos": "sum",
        }
        agg_map = {col: op for col, op in agg_map.items() if col in work.columns}
        work = work.groupby(index_cols + ["entidade_voto"], dropna=False).agg(agg_map).reset_index()
        total = work.groupby(index_cols, dropna=False)["votos"].transform("sum")
        work["share_voto_no_perfil_secao"] = np.where(total > 0, work["votos"] / total, np.nan)

    pivot = work.pivot_table(
        index=index_cols,
        columns="entidade_voto",
        values="share_voto_no_perfil_secao",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()

    # eleitorado/abstenção/comparecimento aparecem repetidos por entidade dentro do mesmo
    # perfil-seção. Por isso usamos max nesses campos e soma apenas nos votos.
    metrics = work.groupby(index_cols, dropna=False).agg({
        "eleitorado": "max",
        "abstencao_estimado": "max",
        "comparecimento_estimado": "max",
        "votos": "sum",
    }).reset_index()
    metrics["taxa_abstencao"] = np.where(metrics["eleitorado"] > 0, metrics["abstencao_estimado"] / metrics["eleitorado"], np.nan)
    metrics["taxa_comparecimento"] = np.where(metrics["eleitorado"] > 0, metrics["comparecimento_estimado"] / metrics["eleitorado"], np.nan)

    return pivot.merge(
        metrics[index_cols + ["eleitorado", "taxa_abstencao", "taxa_comparecimento", "votos"]],
        on=index_cols,
        how="left",
    )


def cluster_behavior_by_profile_section(matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if matrix is None or matrix.empty:
        return pd.DataFrame(), pd.DataFrame([{
            "status": "sem_matriz",
            "observacao": "Matriz comportamental vazia."
        }])

    out = matrix.copy()
    profile_cols = _available_profiles(out)
    if not profile_cols:
        return pd.DataFrame(), pd.DataFrame([{
            "status": "sem_perfil_discreto",
            "observacao": "Clusters comportamentais nao foram gerados porque faltam perfil discreto de eleitor."
        }])
    profile_ok = out[profile_cols].apply(lambda col: col.map(_meaningful_text).ne("")).any(axis=1)
    out = out.loc[profile_ok].copy()
    categorical_cols = [c for c in ["uf", "cargo", "turno"] + profile_cols if c in out.columns]
    metric_cols = {"votos", "taxa_abstencao", "taxa_comparecimento"}
    excluded = set(ELECTION_COLS + REGION_SECTION_COLS + profile_cols) | metric_cols | {"eleitorado"}
    entity_cols = [
        c for c in out.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(out[c])
    ]

    out["entidade_dominante_cluster"] = ""
    if entity_cols:
        shares = out[entity_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        max_share = shares.max(axis=1)
        dominant = shares.idxmax(axis=1).astype(str)
        out["entidade_dominante_cluster"] = dominant.where(max_share.gt(0), "")

    if "votos" not in out.columns:
        out["votos"] = 0
    if "taxa_abstencao" not in out.columns:
        out["taxa_abstencao"] = np.nan
    if "taxa_comparecimento" not in out.columns:
        out["taxa_comparecimento"] = np.nan
    out["faixa_votos_cluster"] = _quantile_band(out["votos"], "votos")
    out["faixa_abstencao_cluster"] = _fixed_rate_band(out["taxa_abstencao"], "abstencao", low=0.12, high=0.25)
    out["faixa_comparecimento_cluster"] = _fixed_rate_band(out["taxa_comparecimento"], "comparecimento", low=0.65, high=0.82)

    token_cols = categorical_cols + [
        "entidade_dominante_cluster",
        "faixa_votos_cluster",
        "faixa_abstencao_cluster",
        "faixa_comparecimento_cluster",
    ]
    token_rows = [
        [f"{col}={value}" for col in token_cols if col in out.columns and (value := _meaningful_text(row.get(col, "")))]
        for _, row in out.iterrows()
    ]

    if len(out) < 5 or not any(token_rows):
        out["cluster_comportamento_eleitoral"] = 0
        summary = pd.DataFrame([{
            "cluster_comportamento_eleitoral": 0,
            "qtd_linhas": len(out),
            "observacao": "Dados discretos insuficientes para clustering robusto; cluster unico criado."
        }])
        return out, summary

    if not SKLEARN_OK:
        out["cluster_comportamento_eleitoral"] = 0
        summary = pd.DataFrame([{
            "cluster_comportamento_eleitoral": 0,
            "qtd_linhas": len(out),
            "observacao": "scikit-learn nao instalado; cluster unico criado."
        }])
        return out, summary

    hasher = FeatureHasher(n_features=HASH_FEATURES, input_type="string", alternate_sign=False)
    full_matrix = hasher.transform(token_rows)
    if full_matrix.shape[0] > MAX_BEHAVIOR_TRAIN_ROWS:
        rng = np.random.default_rng(42)
        train_idx = np.sort(rng.choice(full_matrix.shape[0], size=MAX_BEHAVIOR_TRAIN_ROWS, replace=False))
        train_matrix = full_matrix[train_idx]
    else:
        train_matrix = full_matrix

    min_k = 2
    max_k = min(10, max(2, len(out) // 20), max(2, train_matrix.shape[0] - 1))
    best_k, elbow = _choose_k_by_elbow(train_matrix, min_k, max_k)

    model = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(train_matrix)
    out["cluster_comportamento_eleitoral"] = model.predict(full_matrix)
    out["algoritmo_cluster"] = "KMeans"
    out["metodo_escolha_k"] = "cotovelo_inercia"
    out["features_cluster"] = "tokens_discretos_perfil_entidade_votos_abstencao_comparecimento"

    summary = out.groupby("cluster_comportamento_eleitoral", dropna=False).agg(
        votos=("votos", "sum"),
        taxa_abstencao=("taxa_abstencao", "mean"),
        taxa_comparecimento=("taxa_comparecimento", "mean"),
    ).reset_index()
    summary["qtd_linhas"] = out.groupby("cluster_comportamento_eleitoral").size().values
    summary["k_escolhido"] = best_k
    if elbow is not None and not elbow.empty and "inercia" in elbow.columns:
        chosen = elbow.loc[pd.to_numeric(elbow["k"], errors="coerce").eq(best_k), "inercia"]
        summary["inercia_k_escolhido"] = float(chosen.iloc[0]) if not chosen.empty else np.nan
    summary["linhas_treinamento_cluster"] = int(train_matrix.shape[0])
    summary["qtd_entidades_usadas_no_cluster"] = min(MAX_BEHAVIOR_ENTITIES, len(entity_cols))
    for col in ["uf", "cargo", "turno", *profile_cols, "entidade_dominante_cluster", "faixa_votos_cluster", "faixa_abstencao_cluster", "faixa_comparecimento_cluster"]:
        if col in out.columns:
            summary[f"{col}_dominante"] = summary["cluster_comportamento_eleitoral"].map(
                lambda cl, field=col: _dominant_value(out.loc[out["cluster_comportamento_eleitoral"].eq(cl), field])
            )
    summary["observacao_memoria"] = (
        "Cluster por tokens discretos: perfil, UF, cargo/turno, entidade dominante e faixas de votos, abstencao e comparecimento. "
        "Ano, datas, horas, eleitorado e demais numeros continuos nao entram no clustering. "
        "K definido pela tecnica do cotovelo; quando necessario, KMeans foi treinado em amostra."
    )
    summary["observacao"] = (
        "Cluster comportamental focado em dados discretos e nas tres metricas mantidas: votos, abstencao e comparecimento."
    )
    return out, summary


def _dominant_value(series: pd.Series) -> str:
    vals = [safe_text(x, "") for x in series if safe_text(x, "")]
    if not vals:
        return ""
    vc = pd.Series(vals).value_counts()
    return safe_text(vc.index[0], "")


def _quantile_band(series: pd.Series, prefix: str) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    out = pd.Series(f"{prefix}_sem_valor", index=series.index, dtype=object)
    valid = s.dropna()
    if valid.empty:
        return out
    low = float(valid.quantile(0.33))
    high = float(valid.quantile(0.66))
    out.loc[s.le(low)] = f"{prefix}_baixo"
    out.loc[s.gt(low) & s.le(high)] = f"{prefix}_medio"
    out.loc[s.gt(high)] = f"{prefix}_alto"
    return out


def _fixed_rate_band(series: pd.Series, prefix: str, low: float, high: float) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return pd.Series(np.select(
        [s.isna(), s < low, s <= high, s > high],
        [f"{prefix}_sem_valor", f"{prefix}_baixo", f"{prefix}_medio", f"{prefix}_alto"],
        default=f"{prefix}_sem_valor",
    ), index=series.index)


def _choose_k_by_elbow(matrix, min_k: int, max_k: int) -> tuple[int, pd.DataFrame]:
    rows = []
    max_k = max(min_k, max_k)
    for k in range(min_k, max_k + 1):
        try:
            model = KMeans(n_clusters=k, random_state=42, n_init=10).fit(matrix)
            rows.append({"k": k, "inercia": float(model.inertia_)})
        except Exception:
            continue
    elbow = pd.DataFrame(rows)
    if elbow.empty:
        return min_k, elbow
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
        elbow["distancia_cotovelo"] = distances
        best_k = int(elbow.iloc[int(np.argmax(distances))]["k"])
    elbow["metodo"] = "cotovelo_inercia"
    elbow["k_escolhido"] = best_k
    return best_k, elbow


def interpret_behavior_clusters(clustered: pd.DataFrame, panel: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if clustered is None or clustered.empty or "cluster_comportamento_eleitoral" not in clustered.columns:
        msg = "Não há clusters comportamentais para interpretar."
        return pd.DataFrame([{"status": "sem_cluster", "observacao": msg}]), msg

    profile_cols = _available_profiles(clustered)
    join_cols = [c for c in ELECTION_COLS + REGION_SECTION_COLS + profile_cols if c in clustered.columns and c in panel.columns]
    if not join_cols:
        msg = "Sem chaves para cruzar cluster com painel de voto."
        return pd.DataFrame([{"status": "sem_chaves", "observacao": msg}]), msg

    merged = panel.merge(
        clustered[join_cols + ["cluster_comportamento_eleitoral"]].drop_duplicates(),
        on=join_cols,
        how="left",
    )
    merged = merged.loc[merged["cluster_comportamento_eleitoral"].notna()].copy()

    if merged.empty:
        msg = "Cluster não cruzou com painel de voto."
        return pd.DataFrame([{"status": "merge_vazio", "observacao": msg}]), msg

    rows = []
    md = [
        "# Relatório interpretável dos clusters eleitorais",
        "",
        "Unidade de análise: seção/região × perfil do eleitorado × ano × candidato/partido.",
        "Objetivo: responder quem vota em quem, onde, com que perfil e como isso muda no tempo.",
        "",
    ]

    for cluster_id, g in merged.groupby("cluster_comportamento_eleitoral", dropna=False):
        cluster_s = str(cluster_id)
        total_votes = float(pd.to_numeric(g["votos"], errors="coerce").fillna(0).sum())

        # eleitorado e abstenção se repetem por entidade; para não inflar a taxa,
        # consolidamos por seção/perfil/ano antes de somar.
        unit_cols = [c for c in join_cols if c in g.columns]
        unit_metrics = g[unit_cols + ["eleitorado", "abstencao_estimado", "comparecimento_estimado"]].drop_duplicates()
        electorate = float(pd.to_numeric(unit_metrics["eleitorado"], errors="coerce").fillna(0).sum())
        abst = float(pd.to_numeric(unit_metrics["abstencao_estimado"], errors="coerce").fillna(0).sum())
        abst_rate = abst / electorate if electorate > 0 else np.nan

        # Perfil eleitoral associado ao cluster
        profile_parts = []
        for col in profile_cols:
            s = g[col].astype(str).str.strip()
            s = s.loc[~s.str.lower().isin(["", "nan", "none", "null", "<na>"])]
            if not s.empty:
                vc = s.value_counts()
                profile_parts.append(f"{col.replace('perfil_', '')}: {vc.index[0]} ({vc.iloc[0] / len(s) * 100:.1f}%)")
        profile_text = "; ".join(profile_parts) if profile_parts else "perfil demográfico não detectado"

        # Entidade dominante
        ent = g.groupby("entidade_voto", dropna=False)["votos"].sum().sort_values(ascending=False)
        dominant_entity = str(ent.index[0]) if len(ent) else ""
        dominant_votes = float(ent.iloc[0]) if len(ent) else np.nan
        dominant_share = dominant_votes / total_votes if total_votes > 0 else np.nan
        top_entities = "; ".join([f"{idx}: {int(v)} ({v / total_votes * 100:.1f}%)" for idx, v in ent.head(8).items()]) if total_votes else ""

        # Candidatos e partidos separados
        top_candidates = ""
        if "candidato" in g.columns and g["candidato"].astype(str).str.strip().ne("").any():
            cv = g.groupby("candidato", dropna=False)["votos"].sum().sort_values(ascending=False)
            top_candidates = "; ".join([f"{idx}: {int(v)} ({v / total_votes * 100:.1f}%)" for idx, v in cv.head(8).items()]) if total_votes else ""

        top_parties = ""
        if "partido" in g.columns and g["partido"].astype(str).str.strip().ne("").any():
            pv = g.groupby("partido", dropna=False)["votos"].sum().sort_values(ascending=False)
            top_parties = "; ".join([f"{idx}: {int(v)} ({v / total_votes * 100:.1f}%)" for idx, v in pv.head(8).items()]) if total_votes else ""

        # Regiões e seções
        region_cols = [c for c in ["uf", "cd_municipio", "nm_municipio", "zona", "secao", "local_votacao", "bairro"] if c in g.columns]
        sec_repr = ""
        if region_cols:
            st = g.groupby(region_cols, dropna=False)["votos"].sum().sort_values(ascending=False).head(10).reset_index()
            parts = []
            for _, r in st.iterrows():
                label = " / ".join([safe_text(r.get(c, "")) for c in region_cols if safe_text(r.get(c, ""))])
                parts.append(f"{label}: {int(r['votos'])}")
            sec_repr = "; ".join(parts)

        # Tendência temporal do cluster e da entidade dominante
        trend_cluster = "sem série suficiente"
        trend_entity = "sem série suficiente"
        series_cluster = ""
        series_entity = ""
        if "ano" in g.columns:
            gy = g.groupby("ano", dropna=False)["votos"].sum().reset_index()
            gy["ano_num"] = pd.to_numeric(gy["ano"], errors="coerce")
            gy = gy.loc[gy["ano_num"].notna()].sort_values("ano_num")
            if len(gy) >= 2:
                first = float(gy.iloc[0]["votos"])
                last = float(gy.iloc[-1]["votos"])
                trend_cluster = _trend_label(first, last)
                series_cluster = "; ".join([f"{int(r['ano_num'])}: {int(r['votos'])}" for _, r in gy.iterrows()])

            ge = g.loc[g["entidade_voto"].astype(str).eq(dominant_entity)].groupby("ano", dropna=False)["votos"].sum().reset_index()
            ge["ano_num"] = pd.to_numeric(ge["ano"], errors="coerce")
            ge = ge.loc[ge["ano_num"].notna()].sort_values("ano_num")
            if len(ge) >= 2:
                first = float(ge.iloc[0]["votos"])
                last = float(ge.iloc[-1]["votos"])
                trend_entity = _trend_label(first, last)
                series_entity = "; ".join([f"{int(r['ano_num'])}: {int(r['votos'])}" for _, r in ge.iterrows()])

        crystallization_score = np.nan
        crystallization_level = "sem dados"
        if dominant_entity and "ano" in g.columns:
            total_y = g.groupby("ano", dropna=False)["votos"].sum().reset_index().rename(columns={"votos": "votos_total_ano"})
            ent_y = g.loc[g["entidade_voto"].astype(str).eq(dominant_entity)].groupby("ano", dropna=False)["votos"].sum().reset_index()
            ey = ent_y.merge(total_y, on="ano", how="left")
            ey["share"] = np.where(ey["votos_total_ano"] > 0, ey["votos"] / ey["votos_total_ano"], np.nan)
            if ey["share"].notna().sum() >= 2:
                crystallization_score = float(ey["share"].mean() * (1 - min(max(float(ey["share"].std()), 0), 1)))
            elif pd.notna(dominant_share):
                crystallization_score = float(dominant_share)
            crystallization_level = _crystallization_level(crystallization_score)

        interpretation = (
            f"Este cluster concentra seções/perfis com {profile_text}. "
            f"A entidade mais associada é {dominant_entity}, com {dominant_share * 100:.1f}% dos votos do cluster. "
            f"A abstenção média é {abst_rate * 100:.1f}%." if pd.notna(dominant_share) and pd.notna(abst_rate)
            else f"Este cluster concentra seções/perfis com {profile_text}. Entidade mais associada: {dominant_entity}."
        )
        if trend_cluster != "sem série suficiente":
            interpretation += f" O volume do cluster está em {trend_cluster}."
        if trend_entity != "sem série suficiente":
            interpretation += f" A entidade dominante no cluster está em {trend_entity}."
        if crystallization_level != "sem dados":
            interpretation += f" O voto neste cluster tem cristalização {crystallization_level}."

        rows.append({
            "cluster": cluster_s,
            "perfil_eleitor_associado": profile_text,
            "entidade_mais_associada": dominant_entity,
            "share_entidade_no_cluster": dominant_share,
            "top_entidades": top_entities,
            "top_candidatos": top_candidates,
            "top_partidos": top_parties,
            "secoes_regioes_representativas": sec_repr,
            "taxa_abstencao_media": abst_rate,
            "total_votos_cluster": total_votes,
            "serie_historica_cluster": series_cluster,
            "tendencia_volume_cluster": trend_cluster,
            "serie_historica_entidade_dominante": series_entity,
            "tendencia_entidade_dominante": trend_entity,
            "indice_cristalizacao": crystallization_score,
            "nivel_cristalizacao": crystallization_level,
            "interpretacao": interpretation,
        })

        md.extend([
            f"## Cluster {cluster_s}",
            "",
            f"- **Perfil eleitor associado:** {profile_text}",
            f"- **Entidade/candidato/partido mais associado:** {dominant_entity or 'não identificado'}",
            f"- **Share da entidade no cluster:** {dominant_share * 100:.2f}%" if pd.notna(dominant_share) else "- **Share da entidade no cluster:** sem dados",
            f"- **Top entidades:** {top_entities or 'sem dados'}",
            f"- **Top candidatos:** {top_candidates or 'sem dados'}",
            f"- **Top partidos:** {top_parties or 'sem dados'}",
            f"- **Seções/regiões representativas:** {sec_repr or 'sem dados'}",
            f"- **Taxa média de abstenção:** {abst_rate * 100:.2f}%" if pd.notna(abst_rate) else "- **Taxa média de abstenção:** sem dados",
            f"- **Série histórica do cluster:** {series_cluster or 'sem série suficiente'}",
            f"- **Tendência do volume do cluster:** {trend_cluster}",
            f"- **Série histórica da entidade dominante:** {series_entity or 'sem série suficiente'}",
            f"- **Tendência da entidade dominante:** {trend_entity}",
            f"- **Nível de cristalização:** {crystallization_level}",
            f"- **Interpretação:** {interpretation}",
            "",
        ])

    result = pd.DataFrame(rows).sort_values("total_votos_cluster", ascending=False, na_position="last")
    return result, "\n".join(md)


def run_behavioral_cluster_analysis(gold: pd.DataFrame, out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_dir = out_dir / "parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)

    panel = build_vote_profile_section_panel(gold)
    who = build_who_votes_for_whom(panel)
    affinity = build_candidate_profile_affinity(panel)
    matrix = build_section_behavior_matrix(panel)
    clustered, cluster_summary = cluster_behavior_by_profile_section(matrix)
    interpretation, markdown = interpret_behavior_clusters(clustered, panel)

    outputs = {
        "base_voto_perfil_secao": str(out_dir / "base_voto_perfil_secao.csv"),
        "quem_vota_em_quem": str(out_dir / "quem_vota_em_quem.csv"),
        "afinidade_perfil_candidato": str(out_dir / "afinidade_perfil_candidato.csv"),
        "matriz_secao_perfil_comportamento": str(out_dir / "matriz_secao_perfil_comportamento.csv"),
        "clusters_comportamento_eleitoral": str(out_dir / "clusters_comportamento_eleitoral.csv"),
        "clusters_comportamento_resumo": str(out_dir / "clusters_comportamento_resumo.csv"),
        "clusters_comportamento_interpretacao": str(out_dir / "clusters_comportamento_interpretacao.csv"),
        "clusters_comportamento_relatorio_md": str(out_dir / "clusters_comportamento_relatorio.md"),
    }

    frames = {
        "base_voto_perfil_secao": panel,
        "quem_vota_em_quem": who,
        "afinidade_perfil_candidato": affinity,
        "matriz_secao_perfil_comportamento": matrix,
        "clusters_comportamento_eleitoral": clustered,
        "clusters_comportamento_resumo": cluster_summary,
        "clusters_comportamento_interpretacao": interpretation,
    }
    for name, df in frames.items():
        save_csv(df, Path(outputs[name]))
        parquet_path = parquet_dir / f"{name}.parquet"
        if save_parquet(df, parquet_path):
            outputs[f"{name}_parquet"] = str(parquet_path)
    Path(outputs["clusters_comportamento_relatorio_md"]).write_text(markdown, encoding="utf-8")

    return outputs
