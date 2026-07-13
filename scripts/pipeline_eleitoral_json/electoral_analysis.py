
from __future__ import annotations

from pathlib import Path
from typing import Any
import math

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

INDIVIDUAL_CLUSTER_PROFILE_COLS = [
    "perfil_faixa_etaria",
    "perfil_genero",
    "perfil_instrucao",
    "perfil_estado_civil",
]

SECTION_COLS = ["ano", "uf", "cd_municipio", "nm_municipio", "zona", "secao", "local_votacao", "bairro", "cargo", "turno"]
ENTITY_COLS = ["partido", "candidato"]
MAX_CLUSTER_ENTITIES = 250
MAX_CLUSTER_TRAIN_ROWS = 50000
PATTERN_PROFILE_COLS = [
    "perfil_faixa_etaria",
    "perfil_genero",
    "perfil_instrucao",
    "perfil_estado_civil",
    "perfil_raca_cor",
]


def sort_existing(df: pd.DataFrame, columns: list[str], ascending=True, na_position: str = "last") -> pd.DataFrame:
    if df is None or df.empty:
        return df
    keep_cols = []
    keep_ascending = []
    asc_list = ascending if isinstance(ascending, list) else None
    for idx, col in enumerate(columns):
        if col in df.columns:
            keep_cols.append(col)
            keep_ascending.append(asc_list[idx] if asc_list and idx < len(asc_list) else ascending)
    if not keep_cols:
        return df
    return df.sort_values(keep_cols, ascending=keep_ascending, na_position=na_position)


def choose_entity(df: pd.DataFrame) -> str:
    for c in ["candidato", "partido"]:
        if c in df.columns and df[c].astype(str).str.strip().ne("").any():
            return c
    return "entidade"


def prepare_gold(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for c in SECTION_COLS + PROFILE_COLS + ["partido", "candidato", "entidade", "ideologia", "coalizao"]:
        if c not in out.columns:
            out[c] = ""
        if c in PROFILE_COLS or c in {"cargo", "turno"}:
            out[c] = out[c].map(lambda x, col=c: label_category_value(x, col=col))
        else:
            out[c] = out[c].map(lambda x: safe_text(x, ""))
    out["entidade"] = out["entidade"].replace("", "GERAL")
    for c in ["votos", "eleitorado", "comparecimento_estimado", "abstencao_estimado", "brancos", "nulos", "validos_estimados"]:
        if c not in out.columns:
            out[c] = 0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out


def add_share(df: pd.DataFrame, group_cols: list[str], vote_col: str = "votos") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    total = out.groupby(group_cols, dropna=False)[vote_col].transform("sum")
    out["share"] = np.where(total > 0, out[vote_col] / total, np.nan)
    return out


def vote_by_section(gold: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = prepare_gold(gold)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    entity = choose_entity(df)
    if entity not in df.columns:
        df[entity] = "GERAL"

    group = [c for c in SECTION_COLS if c in df.columns]
    sec = df.groupby(group + [entity], dropna=False).agg({
        "votos": "sum",
        "eleitorado": "max",
        "comparecimento_estimado": "sum",
        "abstencao_estimado": "sum",
        "brancos": "sum",
        "nulos": "sum",
    }).reset_index().rename(columns={entity: "entidade"})

    sec = add_share(sec, group, "votos")
    sec["taxa_abstencao"] = np.where(sec["eleitorado"] > 0, sec["abstencao_estimado"] / sec["eleitorado"], np.nan)
    sec["rank"] = sec.groupby(group, dropna=False)["votos"].rank(method="first", ascending=False)

    top1 = sec.loc[sec["rank"].eq(1)].copy().drop(columns=["rank"])
    top1 = top1.rename(columns={
        "entidade": "vencedor_secao",
        "votos": "votos_vencedor",
        "share": "share_vencedor",
    })

    top2 = sec.loc[sec["rank"].eq(2), group + ["entidade", "votos", "share"]].copy()
    top2 = top2.rename(columns={
        "entidade": "segundo_colocado",
        "votos": "votos_segundo",
        "share": "share_segundo",
    })

    winner = top1.merge(top2, on=group, how="left")
    winner["votos_segundo"] = pd.to_numeric(winner.get("votos_segundo", 0), errors="coerce").fillna(0)
    winner["share_segundo"] = pd.to_numeric(winner.get("share_segundo", 0), errors="coerce").fillna(0)
    winner["margem_votos"] = winner["votos_vencedor"] - winner["votos_segundo"]
    winner["margem_share"] = winner["share_vencedor"] - winner["share_segundo"]

    def _how_won(row: pd.Series) -> str:
        share = row.get("share_vencedor", np.nan)
        margin = row.get("margem_share", np.nan)
        abst = row.get("taxa_abstencao", np.nan)
        parts = []
        if pd.notna(share):
            if share >= 0.50:
                parts.append("venceu com maioria absoluta dos votos da seção")
            elif share >= 0.35:
                parts.append("venceu com pluralidade competitiva")
            else:
                parts.append("venceu em disputa fragmentada")
        if pd.notna(margin):
            if margin < 0.03:
                parts.append("margem muito apertada")
            elif margin < 0.10:
                parts.append("margem moderada")
            else:
                parts.append("margem confortável")
        if pd.notna(abst):
            if abst >= 0.30:
                parts.append("abstenção alta no recorte")
            elif abst <= 0.12:
                parts.append("comparecimento alto no recorte")
        return "; ".join(parts) if parts else "sem base suficiente para descrever a vitória"

    winner["como_ganhou"] = winner.apply(_how_won, axis=1)
    return sec.drop(columns=["rank"]), winner


def abstention_analysis(gold: pd.DataFrame) -> pd.DataFrame:
    df = prepare_gold(gold)
    if df.empty:
        return pd.DataFrame()
    group = ["ano", "uf", "cd_municipio", "nm_municipio", "zona", "secao", "cargo", "turno"]
    out = df.groupby(group, dropna=False).agg({
        "eleitorado": "max",
        "comparecimento_estimado": "sum",
        "abstencao_estimado": "sum",
        "brancos": "sum",
        "nulos": "sum",
        "votos": "sum",
    }).reset_index()
    out["taxa_abstencao"] = np.where(out["eleitorado"] > 0, out["abstencao_estimado"] / out["eleitorado"], np.nan)
    out["taxa_comparecimento"] = np.where(out["eleitorado"] > 0, out["comparecimento_estimado"] / out["eleitorado"], np.nan)
    out["taxa_brancos_nulos"] = np.where(out["votos"] > 0, (out["brancos"] + out["nulos"]) / out["votos"], np.nan)
    return out.sort_values("taxa_abstencao", ascending=False, na_position="last")


def electorate_profiles(gold: pd.DataFrame) -> pd.DataFrame:
    df = prepare_gold(gold)
    if df.empty:
        return pd.DataFrame()
    entity = choose_entity(df)
    profile_cols = [c for c in INDIVIDUAL_CLUSTER_PROFILE_COLS if c in df.columns and df[c].astype(str).str.strip().ne("").any()]
    if not profile_cols:
        return pd.DataFrame([{
            "status": "sem_perfil_eleitor",
            "observacao": "Não foram detectados campos de perfil do eleitorado nos JSONs processados."
        }])

    group = ["ano", "cargo", "turno"] + profile_cols + [entity]
    out = df.groupby(group, dropna=False).agg({
        "votos": "sum",
        "eleitorado": "max",
        "comparecimento_estimado": "sum",
        "abstencao_estimado": "sum",
    }).reset_index().rename(columns={entity: "entidade"})
    total = out.groupby(["ano", "cargo", "turno"] + profile_cols, dropna=False)["votos"].transform("sum")
    out["share_no_perfil"] = np.where(total > 0, out["votos"] / total, np.nan)
    out["taxa_abstencao_perfil"] = np.where(out["eleitorado"] > 0, out["abstencao_estimado"] / out["eleitorado"], np.nan)
    return sort_existing(out, ["ano", "cargo", "turno", "votos"], ascending=[True, True, True, False])


def electorate_profile_by_year(gold: pd.DataFrame) -> pd.DataFrame:
    df = prepare_gold(gold)
    if df.empty:
        return pd.DataFrame()

    profile_cols = [c for c in PROFILE_COLS if c in df.columns and df[c].astype(str).str.strip().ne("").any()]
    if not profile_cols:
        return pd.DataFrame([{
            "status": "sem_perfil_eleitor",
            "observacao": "Nao foram detectados campos de perfil do eleitorado para descrever o eleitor por ano."
        }])

    profile_mask = pd.Series(False, index=df.index)
    for col in profile_cols:
        profile_mask = profile_mask | df[col].astype(str).str.strip().ne("")

    base = df.loc[profile_mask & (pd.to_numeric(df["eleitorado"], errors="coerce").fillna(0) > 0)].copy()
    if base.empty:
        return pd.DataFrame([{
            "status": "sem_eleitorado_por_perfil",
            "observacao": "Campos de perfil existem, mas nao ha metrica de eleitorado associada."
        }])

    frames = []
    for col in profile_cols:
        sub = base.loc[base[col].astype(str).str.strip().ne("")].copy()
        if sub.empty:
            continue
        tmp = sub.groupby(["ano", col], dropna=False)["eleitorado"].sum().reset_index()
        tmp = tmp.rename(columns={col: "valor_perfil"})
        tmp["dimensao_perfil"] = col.replace("perfil_", "")
        total = tmp.groupby(["ano", "dimensao_perfil"], dropna=False)["eleitorado"].transform("sum")
        tmp["share_eleitorado_ano"] = np.where(total > 0, tmp["eleitorado"] / total, np.nan)
        tmp["rank_dimensao_ano"] = tmp.groupby(["ano", "dimensao_perfil"], dropna=False)["eleitorado"].rank(method="first", ascending=False)
        tmp["interpretacao"] = tmp.apply(
            lambda r: (
                f"Em {r['ano']}, {r['valor_perfil']} representa {float(r['share_eleitorado_ano']) * 100:.2f}% "
                f"da dimensao {r['dimensao_perfil']} nos dados de eleitorado."
                if pd.notna(r.get("share_eleitorado_ano")) else "Sem base percentual."
            ),
            axis=1,
        )
        frames.append(tmp[[
            "ano",
            "dimensao_perfil",
            "valor_perfil",
            "eleitorado",
            "share_eleitorado_ano",
            "rank_dimensao_ano",
            "interpretacao",
        ]])

    out = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    return sort_existing(out, ["ano", "dimensao_perfil", "rank_dimensao_ano"]) if not out.empty else out


def profile_vote_association_proxy(gold: pd.DataFrame, top_entities_per_recorte: int = 3) -> pd.DataFrame:
    df = prepare_gold(gold)
    if df.empty:
        return pd.DataFrame()

    entity = choose_entity(df)
    if entity not in df.columns:
        return pd.DataFrame([{
            "status": "sem_entidade_voto",
            "observacao": "Nao foi detectado candidato/partido para cruzar perfil e voto."
        }])

    profile_cols = [c for c in PROFILE_COLS if c in df.columns and df[c].astype(str).str.strip().ne("").any()]
    if not profile_cols:
        return pd.DataFrame([{
            "status": "sem_perfil_eleitor",
            "observacao": "Nao ha campos de perfil do eleitorado para cruzar com votos."
        }])

    vote_mask = df[entity].astype(str).str.strip().ne("") & (pd.to_numeric(df["votos"], errors="coerce").fillna(0) > 0)
    profile_mask = pd.Series(False, index=df.index)
    for col in profile_cols:
        profile_mask = profile_mask | df[col].astype(str).str.strip().ne("")
    profile_mask = profile_mask & (pd.to_numeric(df["eleitorado"], errors="coerce").fillna(0) > 0)

    votes = df.loc[vote_mask].copy()
    profiles = df.loc[profile_mask].copy()
    if votes.empty or profiles.empty:
        return pd.DataFrame([{
            "status": "sem_cruzamento",
            "observacao": "Foram encontrados votos ou perfis, mas nao ambos com metricas suficientes."
        }])

    candidate_keys = ["ano", "uf", "cd_municipio", "zona", "secao"]
    join_keys = []
    for col in candidate_keys:
        if col not in votes.columns or col not in profiles.columns:
            continue
        if votes[col].astype(str).str.strip().ne("").any() and profiles[col].astype(str).str.strip().ne("").any():
            join_keys.append(col)

    if "ano" not in join_keys and "ano" in votes.columns and "ano" in profiles.columns:
        join_keys.insert(0, "ano")

    if not join_keys:
        return pd.DataFrame([{
            "status": "sem_chave_comum",
            "observacao": "Nao ha chave territorial/temporal comum para cruzar perfil e voto."
        }])

    votes["entidade"] = votes[entity].astype(str)
    vote_group = join_keys + ["cargo", "turno", "entidade"]
    v = votes.groupby(vote_group, dropna=False)["votos"].sum().reset_index()
    denom_group = join_keys + ["cargo", "turno"]
    total = v.groupby(denom_group, dropna=False)["votos"].transform("sum")
    v["share_entidade_recorte"] = np.where(total > 0, v["votos"] / total, np.nan)
    v["rank_entidade_recorte"] = v.groupby(denom_group, dropna=False)["votos"].rank(method="first", ascending=False)
    v = v.loc[v["rank_entidade_recorte"] <= max(1, int(top_entities_per_recorte))].copy()

    global_share = v.groupby(["ano", "cargo", "turno", "entidade"], dropna=False)["votos"].sum().reset_index()
    total_global = global_share.groupby(["ano", "cargo", "turno"], dropna=False)["votos"].transform("sum")
    global_share["share_global_entidade"] = np.where(total_global > 0, global_share["votos"] / total_global, np.nan)
    global_share = global_share[["ano", "cargo", "turno", "entidade", "share_global_entidade"]]

    frames = []
    for col in profile_cols:
        p = profiles.loc[profiles[col].astype(str).str.strip().ne("")].copy()
        if p.empty:
            continue
        p = p.groupby(join_keys + [col], dropna=False)["eleitorado"].sum().reset_index()
        p = p.rename(columns={col: "valor_perfil", "eleitorado": "eleitorado_perfil_recorte"})
        p["dimensao_perfil"] = col.replace("perfil_", "")

        joined = p.merge(v, on=join_keys, how="inner")
        if joined.empty:
            continue
        joined["votos_proxy_perfil_entidade"] = joined["eleitorado_perfil_recorte"] * joined["share_entidade_recorte"].fillna(0)

        out_keys = ["ano", "cargo", "turno", "dimensao_perfil", "valor_perfil", "entidade"]
        agg = joined.groupby(out_keys, dropna=False)["votos_proxy_perfil_entidade"].sum().reset_index()
        denom = p.groupby(["ano", "dimensao_perfil", "valor_perfil"], dropna=False)["eleitorado_perfil_recorte"].sum().reset_index()
        agg = agg.merge(denom, on=["ano", "dimensao_perfil", "valor_perfil"], how="left")
        agg["share_proxy_no_perfil"] = np.where(
            agg["eleitorado_perfil_recorte"] > 0,
            agg["votos_proxy_perfil_entidade"] / agg["eleitorado_perfil_recorte"],
            np.nan,
        )
        agg = agg.merge(global_share, on=["ano", "cargo", "turno", "entidade"], how="left")
        agg["lift_perfil_entidade_proxy"] = np.where(
            agg["share_global_entidade"] > 0,
            agg["share_proxy_no_perfil"] / agg["share_global_entidade"],
            np.nan,
        )
        agg["metodo"] = (
            "Proxy ecologica: cruza perfil do eleitorado por ano/territorio com share dos candidatos/partidos "
            "no mesmo recorte. Nao identifica voto individual nem motivacao declarada."
        )
        agg["interpretacao"] = agg.apply(
            lambda r: (
                f"O perfil {r['dimensao_perfil']}={r['valor_perfil']} aparece associado a {r['entidade']} "
                f"com share proxy de {float(r['share_proxy_no_perfil']) * 100:.2f}% "
                f"e lift {float(r['lift_perfil_entidade_proxy']):.2f}x contra a media do recorte."
                if pd.notna(r.get("share_proxy_no_perfil")) and pd.notna(r.get("lift_perfil_entidade_proxy"))
                else "Sem base suficiente para interpretar o cruzamento."
            ),
            axis=1,
        )
        frames.append(agg)

    result = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if result.empty:
        return pd.DataFrame([{
            "status": "sem_match_perfil_voto",
            "observacao": "Perfis e votos existem, mas nao cruzaram pelas chaves disponiveis."
        }])
    return sort_existing(result, ["ano", "cargo", "turno", "lift_perfil_entidade_proxy"], ascending=[True, True, True, False])


def vote_trends(gold: pd.DataFrame) -> pd.DataFrame:
    df = prepare_gold(gold)
    if df.empty:
        return pd.DataFrame()
    entity = choose_entity(df)
    group = ["ano", "cargo", "turno", entity]
    out = df.groupby(group, dropna=False)["votos"].sum().reset_index().rename(columns={entity: "entidade"})
    total = out.groupby(["ano", "cargo", "turno"], dropna=False)["votos"].transform("sum")
    out["share"] = np.where(total > 0, out["votos"] / total, np.nan)
    out["ano_num"] = pd.to_numeric(out["ano"], errors="coerce")
    out = sort_existing(out, ["cargo", "turno", "entidade", "ano_num"])
    out["votos_lag"] = out.groupby(["cargo", "turno", "entidade"], dropna=False)["votos"].shift(1)
    out["share_lag"] = out.groupby(["cargo", "turno", "entidade"], dropna=False)["share"].shift(1)
    out["delta_votos"] = out["votos"] - out["votos_lag"]
    out["delta_share"] = out["share"] - out["share_lag"]
    out["crescimento_votos_pct"] = np.where(out["votos_lag"].abs() > 0, out["delta_votos"] / out["votos_lag"].abs(), np.nan)
    out["curva"] = np.where(out["delta_share"] > 0, "crescimento", np.where(out["delta_share"] < 0, "queda", "estavel/sem_lag"))
    return out


def annual_profile_patterns(gold: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    df = prepare_gold(gold)
    if df.empty:
        return pd.DataFrame()
    profile_cols = [c for c in PATTERN_PROFILE_COLS if c in df.columns and df[c].map(_meaningful_text).ne("").any()]
    if not profile_cols:
        return pd.DataFrame([{"status": "sem_perfil", "observacao": "Nao ha perfil discreto suficiente para comparar anos."}])
    df = df.copy()
    df["perfil_combinado"] = df.apply(lambda r: _profile_combo(r, profile_cols), axis=1)
    df = df.loc[df["perfil_combinado"].map(_meaningful_text).ne("")]
    if df.empty:
        return pd.DataFrame()
    levels = [
        ("brasil", []),
        ("estado", ["uf"]),
        ("municipio", ["uf", "cd_municipio", "nm_municipio"]),
    ]
    frames = []
    for level, scope_cols in levels:
        group = ["ano", *scope_cols, "perfil_combinado"]
        agg = df.groupby(group, dropna=False)["eleitorado"].sum().reset_index()
        denom_group = ["ano", *scope_cols]
        total = agg.groupby(denom_group, dropna=False)["eleitorado"].transform("sum")
        agg["share_perfil"] = np.where(total > 0, agg["eleitorado"] / total, np.nan)
        agg["rank_perfil_ano"] = agg.groupby(denom_group, dropna=False)["eleitorado"].rank(method="first", ascending=False)
        agg = agg.loc[pd.to_numeric(agg["rank_perfil_ano"], errors="coerce").le(top_n)].copy()
        agg["nivel"] = level
        frames.append(agg)
    out = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    return _add_year_pattern_deltas(out, ["nivel", "uf", "cd_municipio", "nm_municipio", "perfil_combinado"], "share_perfil", "eleitorado")


def annual_entity_profile_patterns(gold: pd.DataFrame, entity_col: str, top_n: int = 10) -> pd.DataFrame:
    df = prepare_gold(gold)
    if df.empty or entity_col not in df.columns or not df[entity_col].astype(str).str.strip().ne("").any():
        return pd.DataFrame([{"status": f"sem_{entity_col}", "observacao": f"Nao ha {entity_col} suficiente para comparar perfil por ano."}])
    profile_cols = [c for c in PATTERN_PROFILE_COLS if c in df.columns and df[c].map(_meaningful_text).ne("").any()]
    if not profile_cols:
        return pd.DataFrame([{"status": "sem_perfil", "observacao": "Nao ha perfil discreto suficiente para comparar voto por ano."}])
    df = df.loc[df[entity_col].map(_meaningful_text).ne("")].copy()
    df["perfil_combinado"] = df.apply(lambda r: _profile_combo(r, profile_cols), axis=1)
    df = df.loc[df["perfil_combinado"].map(_meaningful_text).ne("")]
    if df.empty:
        return pd.DataFrame()
    levels = [
        ("brasil", []),
        ("estado", ["uf"]),
        ("municipio", ["uf", "cd_municipio", "nm_municipio"]),
    ]
    frames = []
    for level, scope_cols in levels:
        group = ["ano", *scope_cols, entity_col, "perfil_combinado"]
        agg = df.groupby(group, dropna=False)["votos"].sum().reset_index().rename(columns={entity_col: "entidade"})
        denom_group = ["ano", *scope_cols, "entidade"]
        total = agg.groupby(denom_group, dropna=False)["votos"].transform("sum")
        agg["share_perfil_na_entidade"] = np.where(total > 0, agg["votos"] / total, np.nan)
        agg["rank_perfil_entidade_ano"] = agg.groupby(denom_group, dropna=False)["votos"].rank(method="first", ascending=False)
        agg = agg.loc[pd.to_numeric(agg["rank_perfil_entidade_ano"], errors="coerce").le(top_n)].copy()
        agg["nivel"] = level
        agg["tipo_entidade"] = entity_col
        frames.append(agg)
    out = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    return _add_year_pattern_deltas(out, ["nivel", "uf", "cd_municipio", "nm_municipio", "tipo_entidade", "entidade", "perfil_combinado"], "share_perfil_na_entidade", "votos")


def top10_profiles_by_scope(gold: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    profiles = annual_profile_patterns(gold, top_n=top_n)
    if profiles is None or profiles.empty or "status" in profiles.columns:
        return profiles
    out = profiles.copy()
    out["escopo"] = np.select(
        [out["nivel"].eq("brasil"), out["nivel"].eq("estado"), out["nivel"].eq("municipio")],
        ["Federacao", out.get("uf", ""), out.get("nm_municipio", "")],
        default="",
    )
    out["descricao"] = out.apply(
        lambda r: (
            f"{r.get('perfil_combinado')} representa {float(r.get('share_perfil')) * 100:.1f}% "
            f"do eleitorado em {r.get('ano')} no nivel {r.get('nivel')}."
            if pd.notna(r.get("share_perfil")) else "Sem percentual calculado."
        ),
        axis=1,
    )
    return sort_existing(out, ["nivel", "uf", "cd_municipio", "ano", "rank_perfil_ano"])


def _profile_combo(row: pd.Series, profile_cols: list[str]) -> str:
    bits = []
    for col in profile_cols:
        value = _meaningful_text(row.get(col, ""))
        if value:
            bits.append(f"{col.replace('perfil_', '')}={value}")
    return "; ".join(bits)


def _add_year_pattern_deltas(df: pd.DataFrame, key_cols: list[str], share_col: str, volume_col: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for col in key_cols:
        if col not in out.columns:
            out[col] = ""
    out["ano_num"] = pd.to_numeric(out.get("ano"), errors="coerce")
    out = sort_existing(out, [*key_cols, "ano_num"])
    group_cols = [c for c in key_cols if c in out.columns]
    out[f"{share_col}_ano_anterior"] = out.groupby(group_cols, dropna=False)[share_col].shift(1)
    out[f"{share_col}_delta"] = out[share_col] - out[f"{share_col}_ano_anterior"]
    out[f"{volume_col}_ano_anterior"] = out.groupby(group_cols, dropna=False)[volume_col].shift(1)
    out[f"{volume_col}_delta"] = out[volume_col] - out[f"{volume_col}_ano_anterior"]
    out["padrao_temporal"] = np.where(
        out[f"{share_col}_delta"] > 0.02,
        "crescimento",
        np.where(out[f"{share_col}_delta"] < -0.02, "queda", "estavel/sem historico"),
    )
    return out


def party_identification(gold: pd.DataFrame) -> pd.DataFrame:
    df = prepare_gold(gold)
    if df.empty:
        return pd.DataFrame()
    if not df["partido"].astype(str).str.strip().ne("").any():
        return pd.DataFrame([{"status": "sem_partido", "observacao": "Não foi detectado campo de partido nos JSONs."}])

    group = ["ano", "uf", "cd_municipio", "cargo", "turno", "partido"]
    extra = []
    if df["ideologia"].astype(str).str.strip().ne("").any():
        extra.append("ideologia")
    if df["coalizao"].astype(str).str.strip().ne("").any():
        extra.append("coalizao")
    group = ["ano", "uf", "cd_municipio", "cargo", "turno"] + extra + ["partido"]

    out = df.groupby(group, dropna=False)["votos"].sum().reset_index()
    total = out.groupby([c for c in group if c not in {"partido", "ideologia", "coalizao"}], dropna=False)["votos"].transform("sum")
    out["share_partidario"] = np.where(total > 0, out["votos"] / total, np.nan)
    out["ideologia_disponivel_no_dado"] = "ideologia" in extra
    out["coalizao_disponivel_no_dado"] = "coalizao" in extra
    return sort_existing(out, ["ano", "uf", "cd_municipio", "share_partidario"], ascending=[True, True, True, False])


def party_voter_profiles(gold: pd.DataFrame) -> pd.DataFrame:
    return entity_voter_profiles(gold, "partido")


def candidate_voter_profiles(gold: pd.DataFrame) -> pd.DataFrame:
    return entity_voter_profiles(gold, "candidato")


def entity_voter_profiles(gold: pd.DataFrame, entity_col: str) -> pd.DataFrame:
    df = prepare_gold(gold)
    if df.empty:
        return pd.DataFrame()
    if entity_col not in df.columns or not df[entity_col].astype(str).str.strip().ne("").any():
        return pd.DataFrame([{"status": f"sem_{entity_col}", "observacao": f"Nao foi detectado campo de {entity_col} nos JSONs."}])

    profile_cols = [c for c in PROFILE_COLS if c in df.columns and df[c].astype(str).str.strip().ne("").any()]
    if not profile_cols:
        return pd.DataFrame([{
            "status": "sem_perfil_eleitor",
            "observacao": f"Ha {entity_col}, mas nao ha campos de perfil do eleitorado para responder quem vota por {entity_col}."
        }])

    work = df.loc[df[entity_col].astype(str).str.strip().ne("")].copy()
    group_base = ["ano", "cargo", "turno", entity_col]
    rows = []
    for keys, g in work.groupby(group_base, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_base, keys))
        total_votes = float(pd.to_numeric(g["votos"], errors="coerce").fillna(0).sum())
        row[f"votos_{entity_col}"] = total_votes
        persona_bits = []
        for col in profile_cols:
            prof = g.groupby(col, dropna=False)["votos"].sum().reset_index()
            prof[col] = prof[col].map(lambda x: safe_text(x, "SEM_VALOR"))
            prof = prof.sort_values("votos", ascending=False)
            if prof.empty:
                continue
            top = prof.iloc[0]
            value = safe_text(top.get(col, ""))
            votes = float(top.get("votos", 0) or 0)
            row[f"{col}_dominante"] = value
            row[f"{col}_share_no_{entity_col}"] = votes / total_votes if total_votes > 0 else np.nan
            persona_bits.append(f"{col.replace('perfil_', '')}: {value}")

        row[f"pessoa_do_{entity_col}"] = (
            f"Eleitor predominante do {entity_col}: " + "; ".join(persona_bits)
            if persona_bits else f"Perfil discreto insuficiente para este {entity_col}."
        )
        row["tipo_entidade"] = entity_col
        row["entidade"] = row.get(entity_col, "")
        row["observacao"] = "Perfil ecologico agregado por votos; nao identifica voto individual declarado."
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return sort_existing(out, ["ano", "cargo", "turno", f"votos_{entity_col}"], ascending=[True, True, True, False])


def candidate_profile_with_electorate(gold: pd.DataFrame) -> pd.DataFrame:
    df = prepare_gold(gold)
    if df.empty or "candidato" not in df.columns or not df["candidato"].astype(str).str.strip().ne("").any():
        return pd.DataFrame([{"status": "sem_candidato", "observacao": "Nao foi detectado candidato para construir perfil do candidato."}])
    profile_cols = [c for c in PROFILE_COLS if c in df.columns and df[c].astype(str).str.strip().ne("").any()]
    group = ["ano", "cargo", "turno", "partido", "candidato"]
    base = df.loc[df["candidato"].astype(str).str.strip().ne("")].copy()
    rows = []
    for keys, g in base.groupby(group, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group, keys))
        row["votos_candidato"] = float(pd.to_numeric(g["votos"], errors="coerce").fillna(0).sum())
        row["ufs_principais"] = _top_join(g, "uf", "votos", 5)
        row["municipios_principais"] = _top_join(g, "nm_municipio", "votos", 5)
        persona_bits = []
        for col in profile_cols:
            top = (
                g.groupby(col, dropna=False)["votos"]
                .sum()
                .reset_index()
                .sort_values("votos", ascending=False)
                .head(1)
            )
            if not top.empty:
                value = safe_text(top.iloc[0].get(col, ""))
                row[f"{col}_eleitorado_associado"] = value
                persona_bits.append(f"{col.replace('perfil_', '')}: {value}")
        row["perfil_eleitorado_associado"] = "; ".join(persona_bits) if persona_bits else "Perfil do eleitorado associado insuficiente."
        row["perfil_do_candidato"] = (
            f"{safe_text(row.get('candidato', 'Candidato'))} | partido {safe_text(row.get('partido', 'sem partido'))} | "
            f"{safe_text(row.get('cargo', 'cargo nao identificado'))}."
        )
        row["interpretacao"] = (
            "Perfil do candidato cruzado com o eleitorado agregado por recorte territorial/eleitoral. "
            "Nao identifica voto individual."
        )
        rows.append(row)
    return sort_existing(pd.DataFrame(rows), ["ano", "cargo", "turno", "votos_candidato"], ascending=[True, True, True, False])


def result_electorate_correlation(gold: pd.DataFrame) -> pd.DataFrame:
    frames = []
    party = profile_vote_association_proxy_by_entity(gold, "partido")
    if party is not None and not party.empty:
        party["tipo_entidade"] = "partido"
        frames.append(party)
    candidate = profile_vote_association_proxy_by_entity(gold, "candidato")
    if candidate is not None and not candidate.empty:
        candidate["tipo_entidade"] = "candidato"
        frames.append(candidate)
    if not frames:
        return pd.DataFrame([{"status": "sem_correlacao", "observacao": "Nao houve chaves suficientes para cruzar resultado e eleitorado."}])
    out = pd.concat(frames, ignore_index=True, sort=False)
    keep = [
        "ano", "cargo", "turno", "tipo_entidade", "entidade", "dimensao_perfil", "valor_perfil",
        "votos_proxy_perfil_entidade", "eleitorado_perfil_recorte", "share_proxy_no_perfil",
        "share_global_entidade", "lift_perfil_entidade_proxy", "metodo", "interpretacao",
    ]
    out = out[[c for c in keep if c in out.columns]]
    return sort_existing(
        out,
        ["ano", "cargo", "turno", "tipo_entidade", "lift_perfil_entidade_proxy"],
        ascending=[True, True, True, True, False],
    )


def profile_vote_association_proxy_by_entity(gold: pd.DataFrame, entity_col: str, top_entities_per_recorte: int = 3) -> pd.DataFrame:
    df = prepare_gold(gold)
    if df.empty or entity_col not in df.columns or not df[entity_col].astype(str).str.strip().ne("").any():
        return pd.DataFrame()
    forced = df.copy()
    if entity_col == "partido":
        forced["candidato"] = ""
    elif entity_col == "candidato":
        forced["partido"] = ""
    out = profile_vote_association_proxy(forced, top_entities_per_recorte=top_entities_per_recorte)
    if not out.empty:
        out["tipo_entidade"] = entity_col
    return out


def _top_join(df: pd.DataFrame, col: str, weight_col: str, limit: int) -> str:
    if col not in df.columns:
        return ""
    tmp = df.groupby(col, dropna=False)[weight_col].sum().sort_values(ascending=False).head(limit)
    return "; ".join(safe_text(idx, "") for idx in tmp.index if safe_text(idx, ""))


def rejection_proxy(gold: pd.DataFrame) -> pd.DataFrame:
    df = prepare_gold(gold)
    if df.empty:
        return pd.DataFrame()
    entity = choose_entity(df)
    group = ["ano", "uf", "cd_municipio", "cargo", "turno", entity]
    out = df.groupby(group, dropna=False)["votos"].sum().reset_index().rename(columns={entity: "entidade"})
    total = out.groupby(["ano", "uf", "cd_municipio", "cargo", "turno"], dropna=False)["votos"].transform("sum")
    out["share"] = np.where(total > 0, out["votos"] / total, np.nan)
    out["proxy_rejeicao_territorial"] = 1 - out["share"]
    out["observacao"] = "Proxy: não é rejeição de pesquisa. É baixa penetração territorial relativa ao total de votos no recorte."
    return out.sort_values("proxy_rejeicao_territorial", ascending=False, na_position="last")


def crystallization(gold: pd.DataFrame) -> pd.DataFrame:
    df = prepare_gold(gold)
    if df.empty:
        return pd.DataFrame()
    entity = choose_entity(df)
    group = ["ano", "uf", "cd_municipio", "cargo", "turno", entity]
    out = df.groupby(group, dropna=False)["votos"].sum().reset_index().rename(columns={entity: "entidade"})
    total = out.groupby(["ano", "uf", "cd_municipio", "cargo", "turno"], dropna=False)["votos"].transform("sum")
    out["share"] = np.where(total > 0, out["votos"] / total, np.nan)
    out["ano_num"] = pd.to_numeric(out["ano"], errors="coerce")
    agg = out.groupby(["uf", "cd_municipio", "cargo", "turno", "entidade"], dropna=False).agg({
        "share": ["mean", "std", "count"],
        "votos": "sum",
    }).reset_index()
    agg.columns = ["uf", "cd_municipio", "cargo", "turno", "entidade", "share_medio", "share_std", "anos_observados", "votos_total"]
    agg["indice_cristalizacao"] = np.where(
        agg["anos_observados"] > 1,
        agg["share_medio"].fillna(0) * (1 - agg["share_std"].fillna(0).clip(0, 1)),
        np.nan,
    )
    agg["interpretacao"] = "Quanto maior, maior força média e menor volatilidade histórica da entidade no território."
    return agg.sort_values("indice_cristalizacao", ascending=False, na_position="last")


def transfer_matrix_proxy(gold: pd.DataFrame) -> pd.DataFrame:
    df = prepare_gold(gold)
    if df.empty:
        return pd.DataFrame()
    entity = choose_entity(df)
    group = ["ano", "uf", "cd_municipio", "cargo", "turno", entity]
    out = df.groupby(group, dropna=False)["votos"].sum().reset_index().rename(columns={entity: "entidade"})
    total = out.groupby(["ano", "uf", "cd_municipio", "cargo", "turno"], dropna=False)["votos"].transform("sum")
    out["share"] = np.where(total > 0, out["votos"] / total, np.nan)
    out["ano_num"] = pd.to_numeric(out["ano"], errors="coerce")
    years = sorted(out["ano_num"].dropna().astype(int).unique().tolist())
    rows = []
    idx_cols = ["uf", "cd_municipio", "cargo", "turno"]
    for i, y1 in enumerate(years):
        for y2 in years[i + 1:]:
            d1 = out.loc[out["ano_num"].eq(y1), idx_cols + ["entidade", "share"]].rename(columns={"entidade": "entidade_origem", "share": "share_origem"})
            d2 = out.loc[out["ano_num"].eq(y2), idx_cols + ["entidade", "share"]].rename(columns={"entidade": "entidade_destino", "share": "share_destino"})
            pair = d1.merge(d2, on=idx_cols, how="inner").dropna(subset=["share_origem", "share_destino"])
            if pair.empty:
                continue
            mat = pair.groupby(["entidade_origem", "entidade_destino"], dropna=False).apply(
                lambda g: float(np.nansum(g["share_origem"] * g["share_destino"]))
            ).reset_index(name="score_coocorrencia")
            mat["ano_origem"] = y1
            mat["ano_destino"] = y2
            total_origin = mat.groupby("entidade_origem")["score_coocorrencia"].transform("sum")
            mat["transferencia_proxy"] = np.where(total_origin > 0, mat["score_coocorrencia"] / total_origin, np.nan)
            mat["observacao"] = "Proxy ecológica por coocorrência territorial; não identifica transferência individual de eleitor."
            rows.append(mat)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def strategic_vote_proxy(gold: pd.DataFrame) -> pd.DataFrame:
    df = prepare_gold(gold)
    if df.empty:
        return pd.DataFrame()
    entity = choose_entity(df)
    group_base = ["ano", "uf", "cd_municipio", "cargo", "turno"]
    out = df.groupby(group_base + [entity], dropna=False)["votos"].sum().reset_index().rename(columns={entity: "entidade"})
    total = out.groupby(group_base, dropna=False)["votos"].transform("sum")
    out["share"] = np.where(total > 0, out["votos"] / total, np.nan)
    out["rank"] = out.groupby(group_base, dropna=False)["share"].rank(method="first", ascending=False)
    summary = out.groupby(group_base, dropna=False).agg(
        top2_share=("share", lambda s: float(np.nansum(sorted(s.dropna(), reverse=True)[:2]))),
        menor_share=("share", lambda s: float(np.nansum([x for x in s.dropna() if x < 0.05]))),
        entidades=("entidade", "nunique"),
    ).reset_index()
    summary["ano_num"] = pd.to_numeric(summary["ano"], errors="coerce")
    summary = sort_existing(summary, ["uf", "cd_municipio", "cargo", "turno", "ano_num"])
    summary["top2_share_lag"] = summary.groupby(["uf", "cd_municipio", "cargo", "turno"], dropna=False)["top2_share"].shift(1)
    summary["menor_share_lag"] = summary.groupby(["uf", "cd_municipio", "cargo", "turno"], dropna=False)["menor_share"].shift(1)
    summary["indicador_voto_util_proxy"] = (summary["top2_share"] - summary["top2_share_lag"]) + (summary["menor_share_lag"] - summary["menor_share"])
    summary["observacao"] = "Proxy: aumento de concentração nos dois primeiros e queda de entidades pequenas pode indicar voto estratégico, não prova intenção individual."
    return summary.sort_values("indicador_voto_util_proxy", ascending=False, na_position="last")


def explain_section_winners(gold: pd.DataFrame, winner: pd.DataFrame | None = None) -> pd.DataFrame:
    df = prepare_gold(gold)
    if df.empty:
        return pd.DataFrame()

    if winner is None or winner.empty:
        _, winner = vote_by_section(df)
    if winner is None or winner.empty:
        return pd.DataFrame()

    out = winner.copy()
    profile_cols = [c for c in PROFILE_COLS if c in df.columns and df[c].astype(str).str.strip().ne("").any()]
    profile_mask = pd.Series(False, index=df.index)
    for col in profile_cols:
        profile_mask = profile_mask | df[col].astype(str).str.strip().ne("")
    profiles = df.loc[profile_mask & (pd.to_numeric(df["eleitorado"], errors="coerce").fillna(0) > 0)].copy()

    join_keys = [
        c for c in ["ano", "uf", "cd_municipio", "nm_municipio", "zona", "secao", "local_votacao", "bairro"]
        if c in out.columns and c in profiles.columns
        and out[c].astype(str).str.strip().ne("").any()
        and profiles[c].astype(str).str.strip().ne("").any()
    ]

    if profile_cols and not profiles.empty and join_keys:
        parts = []
        for col in profile_cols:
            p = profiles.loc[profiles[col].astype(str).str.strip().ne("")].copy()
            if p.empty:
                continue
            g = p.groupby(join_keys + [col], dropna=False)["eleitorado"].sum().reset_index()
            g = g.sort_values(join_keys + ["eleitorado"], ascending=[True] * len(join_keys) + [False])
            top = g.drop_duplicates(join_keys).copy()
            top["perfil_item"] = top.apply(
                lambda r: f"{col.replace('perfil_', '')}: {r[col]} ({int(float(r['eleitorado']))} eleitores)",
                axis=1,
            )
            parts.append(top[join_keys + ["perfil_item"]])

        if parts:
            profile_desc = pd.concat(parts, ignore_index=True, sort=False)
            profile_desc = profile_desc.groupby(join_keys, dropna=False)["perfil_item"].agg(lambda s: "; ".join(map(str, s))).reset_index()
            profile_desc = profile_desc.rename(columns={"perfil_item": "perfil_predominante_secao"})
            out = out.merge(profile_desc, on=join_keys, how="left")

    if "perfil_predominante_secao" not in out.columns:
        out["perfil_predominante_secao"] = ""

    def _why(row: pd.Series) -> str:
        factors = []
        share = row.get("share_vencedor", np.nan)
        margin = row.get("margem_share", np.nan)
        abst = row.get("taxa_abstencao", np.nan)
        profile = safe_text(row.get("perfil_predominante_secao", ""))

        if pd.notna(share):
            factors.append(f"teve {share * 100:.2f}% dos votos validos/registrados no recorte")
        if pd.notna(margin):
            factors.append(f"abriu margem de {margin * 100:.2f} pontos percentuais sobre o segundo")
        if pd.notna(abst):
            factors.append(f"a abstencao observada foi {abst * 100:.2f}%")
        if profile:
            factors.append(f"o perfil predominante da secao foi {profile}")

        if not factors:
            return "Sem evidencias suficientes no gold para explicar a vitoria."
        return (
            "A vitoria e explicada pelos fatores observaveis: "
            + "; ".join(factors)
            + ". Isto e uma explicacao agregada por secao/territorio, nao uma motivacao individual declarada."
        )

    out["por_que_ganhou_proxy"] = out.apply(_why, axis=1)
    return sort_existing(out, ["ano", "uf", "cd_municipio", "zona", "secao"])


def build_question_answer_report(analyses: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, str]:
    rows = []

    profile_year = analyses.get("perfil_eleitor_por_ano", pd.DataFrame())
    if profile_year is not None and not profile_year.empty and "status" not in profile_year.columns:
        top_profiles = profile_year.loc[pd.to_numeric(profile_year.get("rank_dimensao_ano"), errors="coerce").eq(1)].head(12)
        answer = "; ".join(
            f"{r['ano']} {r['dimensao_perfil']}={r['valor_perfil']} ({float(r['share_eleitorado_ano']) * 100:.1f}%)"
            for _, r in top_profiles.iterrows()
            if pd.notna(r.get("share_eleitorado_ano"))
        )
    else:
        answer = "Os JSONs processados ainda nao trouxeram perfil de eleitorado suficiente para responder por ano."
    rows.append({
        "pergunta": "Quem sao os eleitores por ano?",
        "resposta": answer or "Sem perfil dominante detectado.",
        "base_de_evidencia": "perfil_eleitor_por_ano.csv",
        "limite": "Descreve distribuicoes agregadas do eleitorado, nao individuos."
    })

    assoc = analyses.get("perfil_voto_proxy", pd.DataFrame())
    if assoc is not None and not assoc.empty and "status" not in assoc.columns:
        top_assoc = assoc.sort_values("lift_perfil_entidade_proxy", ascending=False, na_position="last").head(12)
        answer = "; ".join(
            f"{r['ano']} {r['dimensao_perfil']}={r['valor_perfil']} -> {r['entidade']} "
            f"(lift {float(r['lift_perfil_entidade_proxy']):.2f}x)"
            for _, r in top_assoc.iterrows()
            if pd.notna(r.get("lift_perfil_entidade_proxy"))
        )
    else:
        answer = "Nao houve cruzamento suficiente entre perfil do eleitorado e votos por territorio."
    rows.append({
        "pergunta": "Por quem esses perfis votam?",
        "resposta": answer or "Sem associacao perfil-voto detectada.",
        "base_de_evidencia": "perfil_voto_proxy.csv",
        "limite": "Proxy ecologica por territorio/secao; nao prova voto individual."
    })

    party_profiles = analyses.get("perfil_eleitor_por_partido", pd.DataFrame())
    if party_profiles is not None and not party_profiles.empty and "status" not in party_profiles.columns:
        top_parties = party_profiles.sort_values("votos_partido", ascending=False, na_position="last").head(12)
        answer = "; ".join(
            f"{r.get('ano')} {r.get('partido')}: {r.get('pessoa_do_partido')}"
            for _, r in top_parties.iterrows()
        )
    else:
        answer = "Nao ha partido e perfil do eleitorado suficientes para descrever quem vota por partido."
    rows.append({
        "pergunta": "Quem vota por partido politico?",
        "resposta": answer or "Sem perfil por partido detectado.",
        "base_de_evidencia": "perfil_eleitor_por_partido.csv",
        "limite": "Perfil agregado/ecologico por votos; nao identifica voto individual declarado."
    })

    winner = analyses.get("vencedor_secao_explicado", analyses.get("vencedor_por_secao", pd.DataFrame()))
    if winner is not None and not winner.empty:
        total_sections = len(winner)
        top_winners = winner.groupby("vencedor_secao", dropna=False).size().sort_values(ascending=False).head(8)
        answer = f"Foram identificados vencedores em {total_sections} recortes de secao/ano/cargo/turno. Mais recorrentes: " + "; ".join(
            f"{idx}: {int(val)}" for idx, val in top_winners.items()
        )
    else:
        answer = "Nao foram detectados votos por secao suficientes para apontar vencedores."
    rows.append({
        "pergunta": "Qual candidato ganhou em cada secao por ano?",
        "resposta": answer,
        "base_de_evidencia": "vencedor_por_secao.csv e vencedor_secao_explicado.csv",
        "limite": "Quando o arquivo traz partido mas nao candidato, a entidade vencedora pode ser partidaria."
    })

    if winner is not None and not winner.empty and "margem_share" in winner.columns:
        margins = pd.to_numeric(winner["margem_share"], errors="coerce").dropna()
        shares = pd.to_numeric(winner.get("share_vencedor", pd.Series(dtype=float)), errors="coerce").dropna()
        answer = (
            f"Share medio vencedor: {shares.mean() * 100:.2f}%; margem media: {margins.mean() * 100:.2f} p.p.; "
            f"vitorias apertadas (<3 p.p.): {int((margins < 0.03).sum())}."
            if not margins.empty and not shares.empty else "Margens insuficientes para resumir como ganharam."
        )
    else:
        answer = "Sem margem calculada."
    rows.append({
        "pergunta": "Como ganhou?",
        "resposta": answer,
        "base_de_evidencia": "vencedor_secao_explicado.csv",
        "limite": "Resume margem, share e abstencao observados."
    })

    rows.append({
        "pergunta": "Por que ganhou?",
        "resposta": (
            "O projeto responde com fatores observaveis: margem sobre o segundo, dominancia territorial, "
            "perfil predominante da secao, abstencao/comparecimento, tendencias e cristalizacao historica. "
            "Nao trata isso como motivacao psicologica individual."
        ),
        "base_de_evidencia": "vencedor_secao_explicado.csv, perfil_voto_proxy.csv, cristalizacao_voto.csv",
        "limite": "Causalidade individual exige pesquisa/survey ou dados declarados que nao existem nos arquivos oficiais."
    })

    report = pd.DataFrame(rows)
    md = ["# Respostas eleitorais orientadas por perguntas", ""]
    for _, row in report.iterrows():
        md.extend([
            f"## {row['pergunta']}",
            "",
            str(row["resposta"]),
            "",
            f"**Base de evidencia:** {row['base_de_evidencia']}",
            "",
            f"**Limite:** {row['limite']}",
            "",
        ])
    return report, "\n".join(md)


def sentiment_engagement_diagnostics(gold: pd.DataFrame, profiles: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    if profiles is not None and not profiles.empty:
        roles = profiles.get("role_sugerido", pd.Series(dtype=str)).fillna("").astype(str)
        has_sent = roles.eq("sentimento").any() or profiles["coluna"].astype(str).str.upper().str.contains("SENTIMENT|SENTIMENTO").any()
        has_eng = roles.eq("engajamento").any() or profiles["coluna"].astype(str).str.upper().str.contains("ENGAJ|LIKE|COMENT|SHARE|COMPART|REAC").any()
        text_cols = profiles.loc[roles.isin(["sentimento", "engajamento", "texto_politico"]), "coluna"].dropna().astype(str).unique().tolist()
    else:
        has_sent = False
        has_eng = False
        text_cols = []

    rows.append({
        "analise": "sentimento",
        "disponivel": bool(has_sent),
        "campos_detectados": ", ".join(text_cols),
        "observacao": "Só é calculável se os JSONs tiverem campos de sentimento/texto/pesquisa/rede social. Dados oficiais de votação normalmente não trazem sentimento.",
    })
    rows.append({
        "analise": "engajamento",
        "disponivel": bool(has_eng),
        "campos_detectados": ", ".join(text_cols),
        "observacao": "Só é calculável se os JSONs tiverem métricas de engajamento, comentários, reações ou campos equivalentes.",
    })
    return pd.DataFrame(rows)


def _legacy_electorate_profile_clusters_unused(gold: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = prepare_gold(gold)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    entity = choose_entity(df)
    profile_cols = [c for c in PROFILE_COLS if c in df.columns and df[c].astype(str).str.strip().ne("").any()]
    if not profile_cols:
        return pd.DataFrame([{
            "status": "sem_perfil_eleitor",
            "observacao": "Sem campos de perfil do eleitorado nos JSONs. Cluster de perfil não pode ser feito com segurança."
        }]), pd.DataFrame()

    base_group = ["ano", "uf", "cd_municipio", "zona", "secao", "cargo", "turno"] + profile_cols
    agg = df.groupby(base_group + [entity], dropna=False)["votos"].sum().reset_index().rename(columns={entity: "entidade"})
    top_entities = (
        agg.groupby("entidade", dropna=False)["votos"]
        .sum()
        .sort_values(ascending=False)
        .head(MAX_CLUSTER_ENTITIES)
        .index
        .astype(str)
        .tolist()
    )
    agg["entidade"] = np.where(agg["entidade"].astype(str).isin(top_entities), agg["entidade"].astype(str), "OUTRAS_ENTIDADES")
    agg = agg.groupby(base_group + ["entidade"], dropna=False)["votos"].sum().reset_index()
    total = agg.groupby(base_group, dropna=False)["votos"].transform("sum")
    agg["share"] = np.where(total > 0, agg["votos"] / total, np.nan)

    pivot = agg.pivot_table(index=base_group, columns="entidade", values="share", fill_value=0, aggfunc="sum").reset_index()
    feature_cols = [c for c in pivot.columns if c not in base_group]

    # One-hot dos perfis + shares de entidades
    X = pd.get_dummies(pivot[profile_cols].astype(str), dummy_na=False)
    for c in feature_cols:
        X[f"share_{c}"] = pd.to_numeric(pivot[c], errors="coerce").fillna(0)

    if X.empty or X.shape[0] < 5:
        pivot["cluster_perfil_eleitorado"] = 0
        resumo = pivot.groupby("cluster_perfil_eleitorado").size().reset_index(name="qtd_grupos")
        return pivot, resumo

    if not SKLEARN_OK:
        pivot["cluster_perfil_eleitorado"] = 0
        resumo = pd.DataFrame([{
            "cluster_perfil_eleitorado": 0,
            "qtd_grupos": len(pivot),
            "observacao": "scikit-learn não instalado; cluster real não executado."
        }])
        return pivot, resumo

    max_k = min(8, max(2, len(pivot) // 10))
    min_k = 2
    best_k = min_k
    best_score = -999
    matrix = X.astype("float32").to_numpy()
    if len(matrix) > MAX_CLUSTER_TRAIN_ROWS:
        rng = np.random.default_rng(42)
        train_idx = np.sort(rng.choice(len(matrix), size=MAX_CLUSTER_TRAIN_ROWS, replace=False))
        train_matrix = matrix[train_idx]
    else:
        train_idx = None
        train_matrix = matrix

    if False:
        rng = np.random.default_rng(43)
        score_idx = None
    else:
        score_idx = None

    for k in range(min_k, max_k + 1):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(train_matrix)
        if len(set(labels)) > 1 and len(train_matrix) > k:
            try:
                if score_idx is not None:
                    score = -999
                else:
                    score = -999
            except Exception:
                score = -999
            if score > best_score:
                best_score = score
                best_k = k

    model = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(train_matrix)
    labels = model.predict(matrix)
    pivot["cluster_perfil_eleitorado"] = labels

    resumo = pivot.groupby("cluster_perfil_eleitorado", dropna=False).size().reset_index(name="qtd_grupos")
    resumo["qtd_entidades_usadas_no_cluster"] = len(top_entities)
    resumo["linhas_treinamento_cluster"] = len(train_matrix)
    resumo["observacao_memoria"] = (
        "Entidades fora do top foram agrupadas como OUTRAS_ENTIDADES; "
        "quando necessario, KMeans foi treinado em amostra e aplicado ao conjunto inteiro."
    )
    # Entidade dominante por cluster
    dom = []
    for cl, g in agg.merge(pivot[base_group + ["cluster_perfil_eleitorado"]], on=base_group, how="left").groupby("cluster_perfil_eleitorado"):
        e = g.groupby("entidade")["votos"].sum().sort_values(ascending=False)
        dom.append({
            "cluster_perfil_eleitorado": cl,
            "entidade_dominante": e.index[0] if len(e) else "",
            "votos_entidade_dominante": float(e.iloc[0]) if len(e) else np.nan,
            "entidades_top": ", ".join([f"{idx}:{int(val)}" for idx, val in e.head(5).items()]),
        })
    resumo = resumo.merge(pd.DataFrame(dom), on="cluster_perfil_eleitorado", how="left")
    resumo["interpretacao"] = "Cluster construído com perfil do eleitorado + share de voto por entidade no recorte seção/município."
    return pivot, resumo


def electorate_profile_clusters(
    gold: pd.DataFrame,
    include_results: bool = True,
    cluster_col: str = "cluster_perfil_eleitorado",
    analysis_label: str = "eleitores_resultado",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = prepare_gold(gold)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    entity = choose_entity(df)
    profile_cols = [c for c in PROFILE_COLS if c in df.columns and df[c].astype(str).str.strip().ne("").any()]
    if not profile_cols:
        return pd.DataFrame([{
            "status": "sem_perfil_eleitor",
            "observacao": "Sem campos de perfil do eleitorado nos JSONs. Cluster de perfil nao pode ser feito com seguranca."
        }]), pd.DataFrame()

    base_group = ["ano", "uf", "cd_municipio", "zona", "secao", "cargo", "turno"] + profile_cols
    agg = df.groupby(base_group + [entity], dropna=False)["votos"].sum().reset_index().rename(columns={entity: "entidade"})
    total = agg.groupby(base_group, dropna=False)["votos"].transform("sum")
    agg["share"] = np.where(total > 0, agg["votos"] / total, np.nan)
    agg = agg.sort_values(base_group + ["votos"], ascending=[True] * len(base_group) + [False])
    base = agg.drop_duplicates(base_group).copy().rename(columns={
        "entidade": "entidade_dominante",
        "share": "share_entidade_dominante",
        "votos": "votos_entidade_dominante",
    })
    base = base[base_group + ["entidade_dominante", "share_entidade_dominante", "votos_entidade_dominante"]]
    profile_mask = base[profile_cols].apply(lambda col: col.map(_meaningful_text).ne("")).any(axis=1)
    result_mask = base["entidade_dominante"].map(_meaningful_text).ne("") if include_results else pd.Series(True, index=base.index)
    base = base.loc[profile_mask & result_mask].copy()

    token_cols = ["cargo", "turno"] + profile_cols
    if include_results:
        token_cols.append("entidade_dominante")
    token_rows = [
        [f"{col}={value}" for col in token_cols if col in base.columns and (value := _meaningful_text(row.get(col, "")))]
        for _, row in base.iterrows()
    ]

    if not SKLEARN_OK or len(base) < 5 or not token_rows:
        base[cluster_col] = 0
        base["tipo_analise_cluster"] = analysis_label
        resumo = pd.DataFrame([{
            cluster_col: 0,
            "qtd_grupos": len(base),
            "tipo_analise_cluster": analysis_label,
            "observacao": "scikit-learn indisponivel ou dados insuficientes; cluster unico."
        }])
        return base, _summarize_profile_clusters(base, resumo, profile_cols, cluster_col, include_results, analysis_label)

    hasher = FeatureHasher(n_features=2048, input_type="string", alternate_sign=False)
    matrix = hasher.transform(token_rows)
    if matrix.shape[0] > MAX_CLUSTER_TRAIN_ROWS:
        rng = np.random.default_rng(42)
        train_idx = np.sort(rng.choice(matrix.shape[0], size=MAX_CLUSTER_TRAIN_ROWS, replace=False))
        train_matrix = matrix[train_idx]
    else:
        train_matrix = matrix

    min_k = 2
    max_k = min(8, max(2, len(base) // 10), max(2, train_matrix.shape[0] - 1))
    best_k, elbow = _choose_k_by_elbow(train_matrix, min_k, max_k)

    model = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(train_matrix)
    base[cluster_col] = model.predict(matrix)
    base["algoritmo_cluster"] = "KMeans"
    base["metodo_escolha_k"] = "cotovelo_inercia"
    base["k_escolhido"] = best_k
    base["tipo_analise_cluster"] = analysis_label
    base["features_cluster"] = (
        "perfil_discreto_eleitor_hash"
        if not include_results
        else "perfil_discreto_eleitor_e_entidade_dominante_hash"
    )

    resumo = base.groupby(cluster_col, dropna=False).size().reset_index(name="qtd_grupos")
    resumo["linhas_treinamento_cluster"] = int(train_matrix.shape[0])
    resumo["metodo_escolha_k"] = "cotovelo_inercia"
    resumo["k_escolhido"] = best_k
    resumo["tipo_analise_cluster"] = analysis_label
    if elbow is not None and not elbow.empty and "inercia" in elbow.columns:
        chosen = elbow.loc[pd.to_numeric(elbow["k"], errors="coerce").eq(best_k), "inercia"]
        resumo["inercia_k_escolhido"] = float(chosen.iloc[0]) if not chosen.empty else np.nan
    resumo["observacao_memoria"] = (
        (
            "Cluster individual por tokens discretos do eleitorado: faixa etaria, sexo/genero, escolaridade e estado civil. "
            if not include_results
            else "Cluster individual por tokens discretos do eleitorado mais resultado: faixa etaria, sexo/genero, escolaridade, estado civil e entidade dominante. "
        )
        + "Ano, datas e horas nao entram no clustering. "
        "K definido pela tecnica do cotovelo; quando necessario, KMeans foi treinado em amostra e aplicado ao conjunto inteiro."
    )
    return base, _summarize_profile_clusters(base, resumo, profile_cols, cluster_col, include_results, analysis_label)


def _summarize_profile_clusters(
    base: pd.DataFrame,
    resumo: pd.DataFrame,
    profile_cols: list[str],
    cluster_col: str = "cluster_perfil_eleitorado",
    include_results: bool = True,
    analysis_label: str = "eleitores_resultado",
) -> pd.DataFrame:
    if base is None or base.empty or resumo is None or resumo.empty:
        return resumo
    out = resumo.copy()
    dominant_cols = ["cargo", "turno", *profile_cols]
    if include_results:
        dominant_cols.append("entidade_dominante")
    for col in dominant_cols:
        if col in base.columns:
            out[f"{col}_dominante"] = out[cluster_col].map(
                lambda cl: _dominant_text(base.loc[base[cluster_col].eq(cl), col])
            )
    out["tipo_analise_cluster"] = analysis_label
    out["pessoa_do_cluster"] = out.apply(_profile_cluster_persona, axis=1)
    out["interpretacao"] = (
        "Cluster individual construido somente com faixa etaria, sexo/genero, escolaridade e estado civil. UF e biometria nao entram no cluster."
        if not include_results
        else "Cluster individual construido com faixa etaria, sexo/genero, escolaridade, estado civil e resultado/entidade dominante. UF e biometria nao entram no cluster."
    )
    return out


def _profile_cluster_persona(row: pd.Series) -> str:
    age = _meaningful_text(row.get("perfil_faixa_etaria_dominante", ""))
    gender = _meaningful_text(row.get("perfil_genero_dominante", ""))
    education = _meaningful_text(row.get("perfil_instrucao_dominante", ""))
    civil = _meaningful_text(row.get("perfil_estado_civil_dominante", ""))
    entity = _meaningful_text(row.get("entidade_dominante_dominante", ""))
    bits = [x for x in [age, gender, education, civil] if x]
    person = "; ".join(bits) if bits else "perfil discreto incompleto"
    if entity:
        return f"Eleitor predominante: {person}. Tendencia/voto dominante no recorte: {entity}."
    return f"Eleitor predominante: {person}. Cluster definido sem usar candidato, partido ou vencedor."


def _dominant_text(series: pd.Series) -> str:
    vals = [_meaningful_text(x) for x in series if _meaningful_text(x)]
    if not vals:
        return ""
    counts = pd.Series(vals).value_counts()
    return safe_text(counts.index[0], "")


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


def run_electoral_analysis(gold: pd.DataFrame, out_dir: Path, profiles: pd.DataFrame | None = None) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_dir = out_dir / "parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}

    analyses = {}

    sec, winner = vote_by_section(gold)
    analyses["como_votou_cada_secao"] = sec
    analyses["vencedor_por_secao"] = winner
    analyses["vencedor_secao_explicado"] = explain_section_winners(gold, winner)
    analyses["taxa_abstencao"] = abstention_analysis(gold)
    analyses["perfis_eleitorado"] = electorate_profiles(gold)
    analyses["perfil_eleitor_por_ano"] = electorate_profile_by_year(gold)
    analyses["perfil_voto_proxy"] = profile_vote_association_proxy(gold)
    analyses["tendencias_voto_partido_candidato"] = vote_trends(gold)
    analyses["identificacao_partidaria_ideologica"] = party_identification(gold)
    analyses["perfil_eleitor_por_partido"] = party_voter_profiles(gold)
    analyses["perfil_eleitor_por_candidato"] = candidate_voter_profiles(gold)
    analyses["perfil_do_candidato_correlacionado_eleitorado"] = candidate_profile_with_electorate(gold)
    analyses["resultado_eleitorado_correlacionado"] = result_electorate_correlation(gold)
    analyses["comparativo_anual_perfil_eleitor"] = annual_profile_patterns(gold, top_n=10)
    analyses["comparativo_anual_perfil_partido"] = annual_entity_profile_patterns(gold, "partido", top_n=10)
    analyses["comparativo_anual_perfil_candidato"] = annual_entity_profile_patterns(gold, "candidato", top_n=10)
    analyses["top10_perfis_federacao_estado_municipio"] = top10_profiles_by_scope(gold, top_n=10)
    analyses["proxy_rejeicao"] = rejection_proxy(gold)
    analyses["cristalizacao_voto"] = crystallization(gold)
    analyses["matriz_transferencia_votos_proxy"] = transfer_matrix_proxy(gold)
    analyses["voto_util_proxy"] = strategic_vote_proxy(gold)
    analyses["sentimento_engajamento_diagnostico"] = sentiment_engagement_diagnostics(gold, profiles)

    clusters, clusters_resumo = electorate_profile_clusters(
        gold,
        include_results=False,
        cluster_col="cluster_eleitores",
        analysis_label="somente_eleitores",
    )
    clusters_resultado, clusters_resultado_resumo = electorate_profile_clusters(
        gold,
        include_results=True,
        cluster_col="cluster_eleitores_resultado",
        analysis_label="eleitores_resultado",
    )
    analyses["clusters_perfil_eleitorado"] = clusters
    analyses["clusters_perfil_eleitorado_resumo"] = clusters_resumo
    analyses["clusters_perfil_eleitorado_resultado"] = clusters_resultado
    analyses["clusters_perfil_eleitorado_resultado_resumo"] = clusters_resultado_resumo
    question_report, question_md = build_question_answer_report(analyses)
    analyses["respostas_perguntas_eleitorais"] = question_report

    for name, df in analyses.items():
        path = out_dir / f"{name}.csv"
        save_csv(df, path)
        outputs[name] = str(path)
        parquet_path = parquet_dir / f"{name}.parquet"
        if save_parquet(df, parquet_path):
            outputs[f"{name}_parquet"] = str(parquet_path)

    md_path = out_dir / "respostas_perguntas_eleitorais.md"
    md_path.write_text(question_md, encoding="utf-8")
    outputs["respostas_perguntas_eleitorais_md"] = str(md_path)

    return outputs
