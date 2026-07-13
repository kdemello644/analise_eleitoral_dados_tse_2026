from __future__ import annotations

from pathlib import Path
from typing import Any

import html
import numpy as np
import pandas as pd

from .explainability import build_explainability
from .plots import plot_prediction
from .utils import df_to_html, img_tag, save_csv, save_html, save_parquet, safe_text


SECTION_MODEL_COLS = ["uf", "cd_municipio", "nm_municipio", "zona", "secao", "local_votacao", "bairro"]


def choose_entity_column(df: pd.DataFrame, preferred: str = "auto") -> str:
    if preferred and preferred != "auto" and preferred in df.columns:
        return preferred
    for c in ["candidato", "partido"]:
        if c in df.columns and df[c].astype(str).str.strip().ne("").any():
            return c
    return "entidade"


PROFILE_COLS = [
    "perfil_faixa_etaria",
    "perfil_genero",
    "perfil_instrucao",
    "perfil_estado_civil",
    "perfil_raca_cor",
]

PREDICTION_GOLD_COLS = [
    "ano",
    *SECTION_MODEL_COLS,
    "turno",
    "cargo",
    "partido",
    "candidato",
    "entidade",
    "votos",
    "eleitorado",
    "comparecimento_estimado",
    "abstencao_estimado",
    *PROFILE_COLS,
]


def read_prediction_parquet(path: Path) -> pd.DataFrame:
    if path.is_dir():
        try:
            import pyarrow.dataset as ds

            dataset = ds.dataset(str(path), format="parquet", partitioning="hive")
            available = set(dataset.schema.names)
            selected = [c for c in PREDICTION_GOLD_COLS if c in available]
            return dataset.to_table(columns=selected or None).to_pandas()
        except Exception:
            return pd.read_parquet(path, columns=PREDICTION_GOLD_COLS)
    try:
        import pyarrow.parquet as pq

        available = set(pq.ParquetFile(path).schema.names)
        selected = [c for c in PREDICTION_GOLD_COLS if c in available]
        return pd.read_parquet(path, columns=selected or None)
    except Exception:
        try:
            return pd.read_parquet(path, columns=PREDICTION_GOLD_COLS)
        except Exception:
            return pd.read_parquet(path)


def read_global_gold_for_prediction(global_info: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    gold_parquet_value = str(global_info.get("global_gold_parquet", "") or "")
    gold_parquet = Path(gold_parquet_value) if gold_parquet_value else None
    gold_path = Path(global_info.get("global_gold_csv", ""))
    if gold_parquet is not None and gold_parquet.exists():
        return read_prediction_parquet(gold_parquet), str(gold_parquet)
    if gold_path.exists():
        try:
            header = pd.read_csv(gold_path, sep=";", nrows=0, encoding="utf-8-sig")
            usecols = [c for c in PREDICTION_GOLD_COLS if c in header.columns]
            return pd.read_csv(gold_path, sep=";", dtype=str, usecols=usecols or None, encoding="utf-8-sig"), str(gold_path)
        except UnicodeDecodeError:
            header = pd.read_csv(gold_path, sep=";", nrows=0, encoding="latin1")
            usecols = [c for c in PREDICTION_GOLD_COLS if c in header.columns]
            return pd.read_csv(gold_path, sep=";", dtype=str, usecols=usecols or None, encoding="latin1"), str(gold_path)
    return pd.DataFrame(), ""


def prepare_model_base(global_gold: pd.DataFrame, cfg, entity_override: str | None = None) -> pd.DataFrame:
    if global_gold is None or global_gold.empty:
        return pd.DataFrame()

    df = global_gold.copy()

    if cfg.prediction_cargo_filter and "cargo" in df.columns:
        mask = df["cargo"].astype(str).str.upper().str.contains(cfg.prediction_cargo_filter.upper(), na=False)
        if mask.any():
            df = df.loc[mask].copy()

    entity_col = entity_override or choose_entity_column(df, cfg.prediction_entity)
    if entity_col not in df.columns:
        entity_col = choose_entity_column(df, cfg.prediction_entity)
    if entity_col == "entidade" and "entidade" not in df.columns:
        df["entidade"] = "GERAL"

    for c in ["ano", *SECTION_MODEL_COLS, "turno", "cargo", entity_col]:
        if c not in df.columns:
            df[c] = ""

    for c in ["votos", "eleitorado", "comparecimento_estimado", "abstencao_estimado"]:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    group = ["ano", *SECTION_MODEL_COLS, "turno", "cargo", entity_col]
    model = df.groupby(group, dropna=False).agg({
        "votos": "sum",
        "eleitorado": "max",
        "comparecimento_estimado": "sum",
        "abstencao_estimado": "sum",
    }).reset_index()

    section_group = ["ano", *SECTION_MODEL_COLS, "turno", "cargo"]
    total = model.groupby(section_group, dropna=False)["votos"].transform("sum")
    model["share"] = np.where(total > 0, model["votos"] / total, np.nan)
    model["ano_num"] = pd.to_numeric(model["ano"], errors="coerce")
    model = model.rename(columns={entity_col: "entidade"})
    model["entidade"] = model["entidade"].astype(str).replace({"": "GERAL", "nan": "GERAL", "None": "GERAL"})
    return model


def backtest_model(model_df: pd.DataFrame) -> pd.DataFrame:
    if model_df is None or model_df.empty:
        return pd.DataFrame()

    years = sorted(pd.to_numeric(model_df["ano_num"], errors="coerce").dropna().astype(int).unique().tolist())
    if len(years) < 2:
        return pd.DataFrame([{
            "status": "insuficiente",
            "observacao": "Backtesting exige pelo menos dois anos comparáveis no modelo.",
        }])

    keys = SECTION_MODEL_COLS + ["turno", "cargo", "entidade"]
    rows = []

    for target_year in years[1:]:
        base_year = max(y for y in years if y < target_year)
        pred = model_df.loc[model_df["ano_num"].eq(base_year), keys + ["share"]].rename(columns={"share": "share_pred"})
        real = model_df.loc[model_df["ano_num"].eq(target_year), keys + ["share"]].rename(columns={"share": "share_real"})
        comp = real.merge(pred, on=keys, how="inner").dropna(subset=["share_real", "share_pred"])
        if comp.empty:
            continue
        comp["erro"] = comp["share_pred"] - comp["share_real"]
        rows.append({
            "base_year": int(base_year),
            "target_year": int(target_year),
            "n": int(len(comp)),
            "mae_share": float(comp["erro"].abs().mean()),
            "rmse_share": float(np.sqrt((comp["erro"] ** 2).mean())),
        })

    return pd.DataFrame(rows)


def estimate_temporal_confidence(model_df: pd.DataFrame) -> float:
    if model_df is None or model_df.empty:
        return 0.35

    df = model_df.copy()
    keys = SECTION_MODEL_COLS + ["turno", "cargo", "entidade"]
    years = sorted(pd.to_numeric(df["ano_num"], errors="coerce").dropna().astype(int).unique().tolist())
    vals = []

    for i, y1 in enumerate(years):
        for y2 in years[i+1:]:
            d1 = df.loc[df["ano_num"].eq(y1), keys + ["share"]].rename(columns={"share": "s1"})
            d2 = df.loc[df["ano_num"].eq(y2), keys + ["share"]].rename(columns={"share": "s2"})
            pair = d1.merge(d2, on=keys, how="inner").dropna(subset=["s1", "s2"])
            if len(pair) >= 10:
                corr = spearman_corr(pair["s1"], pair["s2"])
                if pd.notna(corr):
                    vals.append(abs(float(corr)))

    if not vals:
        return 0.35
    return float(np.clip(np.nanmean(vals), 0.05, 0.95))


def historical_swing(model_df: pd.DataFrame) -> pd.DataFrame:
    keys = SECTION_MODEL_COLS + ["turno", "cargo", "entidade"]
    df = model_df.sort_values(keys + ["ano_num"]).copy()
    df["share_lag"] = df.groupby(keys, dropna=False)["share"].shift(1)
    df["ano_lag"] = df.groupby(keys, dropna=False)["ano_num"].shift(1)
    df["swing"] = df["share"] - df["share_lag"]
    df["anos_delta"] = df["ano_num"] - df["ano_lag"]
    df["swing_anual"] = np.where(df["anos_delta"] > 0, df["swing"] / df["anos_delta"], np.nan)
    return df


def build_scenarios(model_df: pd.DataFrame, cfg) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if model_df is None or model_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = model_df.loc[model_df["ano_num"].notna()].copy()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    base_year = int(df["ano_num"].max())
    base = df.loc[df["ano_num"].eq(base_year)].copy()

    swing_df = historical_swing(df)
    swing_entity = swing_df.dropna(subset=["swing_anual"]).groupby(["cargo", "turno", "entidade"], dropna=False)["swing_anual"].agg(["mean", "std", "count"]).reset_index()
    swing_entity = swing_entity.rename(columns={"mean": "swing_entidade_mean", "std": "swing_entidade_std", "count": "swing_entidade_n"})

    base = base.merge(swing_entity, on=["cargo", "turno", "entidade"], how="left")
    base["swing_anual_estimado"] = pd.to_numeric(base["swing_entidade_mean"], errors="coerce").fillna(0.0)

    confidence = estimate_temporal_confidence(model_df)
    sigma_multiplier = float(np.clip(1.60 - confidence, 0.70, 1.80))
    base["confianca_temporal_modelo"] = confidence
    base["sigma_multiplicador_temporal"] = sigma_multiplier
    base["sigma_estimado"] = pd.to_numeric(base["swing_entidade_std"], errors="coerce").fillna(cfg.monte_carlo_sigma) * sigma_multiplier
    base["anos_ate_2026"] = max(0, min(12, 2026 - base_year))

    scenarios = {
        "base": {"trend_weight": 0.60, "turnout_shift": 0.00, "noise_mult": 1.00},
        "continuidade_historica": {"trend_weight": 0.30, "turnout_shift": 0.00, "noise_mult": 0.80},
        "impulso_territorial": {"trend_weight": 0.90, "turnout_shift": 0.00, "noise_mult": 1.15},
        "alta_abstencao": {"trend_weight": 0.50, "turnout_shift": -0.04, "noise_mult": 1.25},
        "baixa_abstencao": {"trend_weight": 0.70, "turnout_shift": 0.03, "noise_mult": 1.10},
        "maior_volatilidade": {"trend_weight": 0.80, "turnout_shift": 0.00, "noise_mult": 1.80},
    }

    frames = []
    for name, params in scenarios.items():
        tmp = base.copy()
        tmp["cenario"] = name
        tmp["share_pred_raw"] = (
            tmp["share"].fillna(0)
            + params["trend_weight"] * tmp["swing_anual_estimado"].fillna(0) * tmp["anos_ate_2026"]
        ).clip(lower=0.000001)

        group = SECTION_MODEL_COLS + ["turno", "cargo"]
        total = tmp.groupby(group, dropna=False)["share_pred_raw"].transform("sum")
        tmp["share_pred_2026"] = np.where(total > 0, tmp["share_pred_raw"] / total, np.nan)

        turnout = np.where(tmp["eleitorado"] > 0, tmp["comparecimento_estimado"] / tmp["eleitorado"], np.nan)
        tmp["pct_comparecimento_pred"] = pd.Series(turnout).fillna(0.75).clip(0.30, 0.98).values + params["turnout_shift"]
        tmp["pct_comparecimento_pred"] = tmp["pct_comparecimento_pred"].clip(0.30, 0.98)
        tmp["votos_pred_2026"] = tmp["share_pred_2026"] * tmp["eleitorado"].fillna(0) * tmp["pct_comparecimento_pred"]

        frames.append(tmp[[
            "cenario", "ano", "ano_num", *SECTION_MODEL_COLS, "turno", "cargo",
            "entidade", "share", "share_pred_2026", "eleitorado", "pct_comparecimento_pred",
            "votos_pred_2026", "swing_anual_estimado", "sigma_estimado",
            "confianca_temporal_modelo", "sigma_multiplicador_temporal",
        ]])

    scenarios_df = pd.concat(frames, ignore_index=True, sort=False)

    nacional = scenarios_df.groupby(["cenario", "cargo", "turno", "entidade"], dropna=False).agg({
        "votos_pred_2026": "sum",
        "eleitorado": "sum",
    }).reset_index()

    total_n = nacional.groupby(["cenario", "cargo", "turno"], dropna=False)["votos_pred_2026"].transform("sum")
    nacional["share_nacional_pred_2026"] = np.where(total_n > 0, nacional["votos_pred_2026"] / total_n, np.nan)
    nacional = nacional.sort_values(["cenario", "cargo", "turno", "share_nacional_pred_2026"], ascending=[True, True, True, False])

    mc = monte_carlo(base, scenarios, cfg)
    return scenarios_df, nacional, mc


def monte_carlo(base: pd.DataFrame, scenarios: dict[str, dict[str, float]], cfg) -> pd.DataFrame:
    if base is None or base.empty or cfg.cenarios <= 0:
        return pd.DataFrame()

    rng = np.random.default_rng(cfg.seed)
    working = base.sample(300000, random_state=cfg.seed).copy() if len(base) > 300000 else base.copy()

    records = []
    for scenario, params in scenarios.items():
        result = {}

        for _ in range(int(cfg.cenarios)):
            tmp = working.copy()
            noise = rng.normal(
                0,
                tmp["sigma_estimado"].fillna(cfg.monte_carlo_sigma).values * params["noise_mult"],
            )
            tmp["share_raw"] = (
                tmp["share"].fillna(0).values
                + params["trend_weight"] * tmp["swing_anual_estimado"].fillna(0).values * tmp["anos_ate_2026"].fillna(0).values
                + noise
            )
            tmp["share_raw"] = np.clip(tmp["share_raw"], 0.000001, None)

            group = SECTION_MODEL_COLS + ["turno", "cargo"]
            total = tmp.groupby(group, dropna=False)["share_raw"].transform("sum")
            tmp["share_sim"] = np.where(total > 0, tmp["share_raw"] / total, np.nan)

            turnout = np.where(tmp["eleitorado"] > 0, tmp["comparecimento_estimado"] / tmp["eleitorado"], np.nan)
            turnout = pd.Series(turnout).fillna(0.75).clip(0.30, 0.98).values + params["turnout_shift"]
            turnout = np.clip(turnout, 0.30, 0.98)

            tmp["votos_sim"] = tmp["share_sim"] * tmp["eleitorado"].fillna(0).values * turnout

            agg = tmp.groupby(["cargo", "turno", "entidade"], dropna=False)["votos_sim"].sum().reset_index()
            total_sim = agg.groupby(["cargo", "turno"], dropna=False)["votos_sim"].transform("sum")
            agg["share"] = np.where(total_sim > 0, agg["votos_sim"] / total_sim, np.nan)

            for _, r in agg.iterrows():
                key = (str(r["cargo"]), str(r["turno"]), str(r["entidade"]))
                result.setdefault(key, []).append(float(r["share"]))

        for (cargo, turno, entidade), vals in result.items():
            arr = np.array(vals, dtype=float)
            records.append({
                "cenario": scenario,
                "cargo": cargo,
                "turno": turno,
                "entidade": entidade,
                "n_simulacoes": len(arr),
                "share_medio": float(np.nanmean(arr)),
                "share_p05": float(np.nanquantile(arr, 0.05)),
                "share_p50": float(np.nanquantile(arr, 0.50)),
                "share_p95": float(np.nanquantile(arr, 0.95)),
                "share_desvio": float(np.nanstd(arr)),
            })

    out = pd.DataFrame(records)
    return out.sort_values(["cenario", "cargo", "turno", "share_medio"], ascending=[True, True, True, False]) if not out.empty else out


def build_party_2026_tables(global_gold: pd.DataFrame, cfg) -> dict[str, pd.DataFrame]:
    if global_gold is None or global_gold.empty or "partido" not in global_gold.columns:
        return {
            "partidos_2026_brasil": pd.DataFrame(),
            "partidos_2026_estados": pd.DataFrame(),
            "partidos_2026_municipios": pd.DataFrame(),
            "partidos_2026_correlacao_historica": pd.DataFrame(),
        }

    party_model = prepare_model_base(global_gold, cfg, entity_override="partido")
    if party_model.empty or "entidade" not in party_model.columns:
        return {
            "partidos_2026_brasil": pd.DataFrame(),
            "partidos_2026_estados": pd.DataFrame(),
            "partidos_2026_municipios": pd.DataFrame(),
            "partidos_2026_correlacao_historica": pd.DataFrame(),
        }

    party_model = party_model.loc[party_model["entidade"].map(_meaningful_text).ne("")].copy()
    if party_model.empty:
        return {
            "partidos_2026_brasil": pd.DataFrame(),
            "partidos_2026_estados": pd.DataFrame(),
            "partidos_2026_municipios": pd.DataFrame(),
            "partidos_2026_correlacao_historica": pd.DataFrame(),
        }

    party_scenarios, _, _ = build_scenarios(party_model, cfg)
    history = build_party_history_by_level(party_model)
    profiles = build_party_profile_reference(global_gold)
    trend = build_party_trends(history)
    corr = build_party_historical_correlation(history)

    brasil = aggregate_party_scenario_level(party_scenarios, "brasil", [], profiles, trend, corr)
    estados = aggregate_party_scenario_level(party_scenarios, "estado", ["uf"], profiles, trend, corr)
    municipios = aggregate_party_scenario_level(party_scenarios, "municipio", ["uf", "cd_municipio", "nm_municipio"], profiles, trend, corr)

    return {
        "partidos_2026_brasil": brasil,
        "partidos_2026_estados": estados,
        "partidos_2026_municipios": municipios,
        "partidos_2026_correlacao_historica": corr,
    }


def build_party_history_by_level(model_df: pd.DataFrame) -> pd.DataFrame:
    if model_df is None or model_df.empty:
        return pd.DataFrame()
    df = model_df.copy()
    df["partido"] = df["entidade"].map(_meaningful_text)
    df = df.loc[df["partido"].ne("")].copy()
    if df.empty:
        return pd.DataFrame()
    for col in ["votos", "eleitorado"]:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)
    levels = [
        ("brasil", []),
        ("estado", ["uf"]),
        ("municipio", ["uf", "cd_municipio", "nm_municipio"]),
    ]
    frames = []
    for level, scope_cols in levels:
        for col in scope_cols:
            if col not in df.columns:
                df[col] = ""
        base_group = ["ano_num", *scope_cols, "cargo", "turno"]
        group = [*base_group, "partido"]
        agg = df.groupby(group, dropna=False).agg(
            votos_historicos=("votos", "sum"),
            eleitorado_historico=("eleitorado", "sum"),
        ).reset_index()
        total = agg.groupby(base_group, dropna=False)["votos_historicos"].transform("sum")
        agg["share_historico"] = np.where(total > 0, agg["votos_historicos"] / total, np.nan)
        agg["nivel"] = level
        frames.append(_ensure_scope_cols(agg))
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def build_party_profile_reference(global_gold: pd.DataFrame) -> pd.DataFrame:
    if global_gold is None or global_gold.empty or "partido" not in global_gold.columns:
        return pd.DataFrame()
    df = global_gold.copy()
    df["partido"] = df["partido"].map(_meaningful_text)
    df = df.loc[df["partido"].ne("")].copy()
    if df.empty:
        return pd.DataFrame()
    for col in ["votos", "eleitorado"]:
        df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)
    if "ano" not in df.columns:
        df["ano"] = ""
    df["_peso_perfil_partido"] = np.where(df["votos"] > 0, df["votos"], df["eleitorado"])
    if df["_peso_perfil_partido"].sum() <= 0:
        df["_peso_perfil_partido"] = 1.0
    for col in PROFILE_COLS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].map(_meaningful_text)

    levels = [
        ("brasil", []),
        ("estado", ["uf"]),
        ("municipio", ["uf", "cd_municipio", "nm_municipio"]),
    ]
    frames = []
    for level, scope_cols in levels:
        for col in scope_cols:
            if col not in df.columns:
                df[col] = ""
        base_cols = [*scope_cols, "partido"]
        ref = df.groupby(base_cols, dropna=False).agg(
            votos_historicos_perfil=("votos", "sum"),
            eleitorado_historico_perfil=("eleitorado", "sum"),
            anos_historicos_perfil=("ano", lambda s: ", ".join(sorted(set(_meaningful_text(x) for x in s if _meaningful_text(x))))),
        ).reset_index()
        for profile_col in PROFILE_COLS:
            p = df.loc[df[profile_col].map(_meaningful_text).ne("")].copy()
            if p.empty:
                continue
            g = p.groupby(base_cols + [profile_col], dropna=False)["_peso_perfil_partido"].sum().reset_index()
            total = g.groupby(base_cols, dropna=False)["_peso_perfil_partido"].transform("sum")
            g[f"{profile_col}_share_associado"] = np.where(total > 0, g["_peso_perfil_partido"] / total, np.nan)
            g = g.sort_values(base_cols + ["_peso_perfil_partido"], ascending=[True] * len(base_cols) + [False])
            top = g.drop_duplicates(base_cols).rename(columns={profile_col: f"{profile_col}_associado"})
            ref = ref.merge(top[base_cols + [f"{profile_col}_associado", f"{profile_col}_share_associado"]], on=base_cols, how="left")
        ref["nivel"] = level
        ref = _ensure_scope_cols(ref)
        ref["perfil_eleitor_2026"] = ref.apply(profile_sentence, axis=1)
        frames.append(ref)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def build_party_trends(history: pd.DataFrame) -> pd.DataFrame:
    if history is None or history.empty:
        return pd.DataFrame()
    h = _ensure_scope_cols(history).copy()
    keys = ["nivel", "uf", "cd_municipio", "nm_municipio", "cargo", "turno", "partido"]
    h["ano_num"] = pd.to_numeric(h["ano_num"], errors="coerce")
    h = h.dropna(subset=["ano_num"]).sort_values(keys + ["ano_num"])
    if h.empty:
        return pd.DataFrame()
    first = h.groupby(keys, dropna=False).head(1).copy()
    last = h.groupby(keys, dropna=False).tail(1).copy()
    years = h.groupby(keys, dropna=False)["ano_num"].agg(
        anos_historicos=lambda s: ", ".join(str(int(x)) for x in sorted(set(s.dropna()))),
        qtd_anos_historicos="nunique",
    ).reset_index()
    out = last[keys + ["ano_num", "share_historico", "votos_historicos"]].rename(columns={
        "ano_num": "ano_base_historico",
        "share_historico": "share_historico_recente",
        "votos_historicos": "votos_historicos_recentes",
    })
    first = first[keys + ["ano_num", "share_historico"]].rename(columns={
        "ano_num": "ano_inicial_historico",
        "share_historico": "share_historico_inicial",
    })
    out = out.merge(first, on=keys, how="left").merge(years, on=keys, how="left")
    delta_years = pd.to_numeric(out["ano_base_historico"], errors="coerce") - pd.to_numeric(out["ano_inicial_historico"], errors="coerce")
    out["delta_share_historico"] = pd.to_numeric(out["share_historico_recente"], errors="coerce") - pd.to_numeric(out["share_historico_inicial"], errors="coerce")
    out["tendencia_anual_share"] = np.where(delta_years > 0, out["delta_share_historico"] / delta_years, np.nan)
    out["tendencia_partido"] = np.where(
        out["delta_share_historico"] > 0.02,
        "crescimento",
        np.where(out["delta_share_historico"] < -0.02, "queda", "estavel/sem historico"),
    )
    return out


def build_party_historical_correlation(history: pd.DataFrame) -> pd.DataFrame:
    if history is None or history.empty:
        return pd.DataFrame()
    h = _ensure_scope_cols(history).copy()
    h["ano_num"] = pd.to_numeric(h["ano_num"], errors="coerce")
    h["share_historico"] = pd.to_numeric(h["share_historico"], errors="coerce")
    group_cols = ["nivel", "uf", "cd_municipio", "nm_municipio", "cargo", "turno"]
    rows = []
    for keys, g in h.dropna(subset=["ano_num", "share_historico"]).groupby(group_cols, dropna=False):
        pivot = g.pivot_table(index="partido", columns="ano_num", values="share_historico", aggfunc="sum")
        years = sorted([int(y) for y in pivot.columns if pd.notna(y)])
        vals = []
        pairs = []
        for y1, y2 in zip(years, years[1:]):
            pair = pivot[[y1, y2]].dropna()
            if len(pair) < 2 or pair[y1].nunique(dropna=True) <= 1 or pair[y2].nunique(dropna=True) <= 1:
                continue
            corr = spearman_corr(pair[y1], pair[y2])
            if pd.notna(corr):
                vals.append(float(corr))
                pairs.append(f"{y1}-{y2}")
        corr_mean = float(np.nanmean(vals)) if vals else np.nan
        rows.append({
            **dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,))),
            "correlacao_historica_share_partidos": corr_mean,
            "pares_anos_correlacionados": ", ".join(pairs),
            "qtd_anos_historicos_recorte": len(years),
            "anos_historicos_recorte": ", ".join(map(str, years)),
            "forca_correlacao_historica": correlation_strength(corr_mean),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["justificativa_correlacao"] = out.apply(correlation_sentence, axis=1)
    return out


def aggregate_party_scenario_level(
    scenarios: pd.DataFrame,
    level: str,
    scope_cols: list[str],
    profiles: pd.DataFrame,
    trend: pd.DataFrame,
    corr: pd.DataFrame,
) -> pd.DataFrame:
    if scenarios is None or scenarios.empty:
        return pd.DataFrame()
    df = scenarios.copy()
    for col in scope_cols:
        if col not in df.columns:
            df[col] = ""
    df["partido"] = df["entidade"].map(_meaningful_text)
    df = df.loc[df["partido"].ne("")].copy()
    if df.empty:
        return pd.DataFrame()

    group_base = ["cenario", *scope_cols, "cargo", "turno"]
    group = [*group_base, "partido"]
    agg = df.groupby(group, dropna=False).agg(votos_pred_2026=("votos_pred_2026", "sum")).reset_index()
    total_votes = agg.groupby(group_base, dropna=False)["votos_pred_2026"].transform("sum")
    agg["share_pred_2026"] = np.where(total_votes > 0, agg["votos_pred_2026"] / total_votes, np.nan)
    agg["pct_votos_partido_2026"] = agg["share_pred_2026"]

    section_keys = ["cenario", *SECTION_MODEL_COLS, "cargo", "turno"]
    section_keys = [c for c in section_keys if c in df.columns]
    electorate_scope = df.drop_duplicates(section_keys)
    electorate = electorate_scope.groupby(group_base, dropna=False)["eleitorado"].sum().reset_index().rename(columns={"eleitorado": "eleitorado_total_modelado"})
    agg = agg.merge(electorate, on=group_base, how="left")
    agg["votos_validos_total_modelado"] = total_votes
    agg["nivel"] = level
    agg = _ensure_scope_cols(agg)

    join_cols = ["nivel", "uf", "cd_municipio", "nm_municipio", "partido"]
    if profiles is not None and not profiles.empty:
        agg = agg.merge(_ensure_scope_cols(profiles), on=join_cols, how="left")
    trend_cols = ["nivel", "uf", "cd_municipio", "nm_municipio", "cargo", "turno", "partido"]
    if trend is not None and not trend.empty:
        agg = agg.merge(_ensure_scope_cols(trend), on=trend_cols, how="left")
    corr_cols = ["nivel", "uf", "cd_municipio", "nm_municipio", "cargo", "turno"]
    if corr is not None and not corr.empty:
        agg = agg.merge(_ensure_scope_cols(corr), on=corr_cols, how="left")

    if "perfil_eleitor_2026" not in agg.columns:
        agg["perfil_eleitor_2026"] = ""
    agg["perfil_eleitor_2026"] = agg["perfil_eleitor_2026"].map(_meaningful_text)
    agg["perfil_eleitor_2026"] = np.where(
        agg["perfil_eleitor_2026"].astype(str).str.strip().ne(""),
        agg["perfil_eleitor_2026"],
        "Perfil de eleitor ainda insuficiente neste recorte.",
    )
    agg["justificativa_previsao_partido_2026"] = agg.apply(party_prediction_sentence, axis=1)
    agg["ano_simulado"] = 2026
    agg = agg.sort_values(["cenario", "nivel", "uf", "nm_municipio", "cargo", "turno", "share_pred_2026"], ascending=[True, True, True, True, True, True, False])
    return agg


def _ensure_scope_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["uf", "cd_municipio", "nm_municipio"]:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].map(lambda x: safe_text(x, ""))
    return out


def _meaningful_text(value: Any) -> str:
    text = safe_text(value, "").strip()
    lower = text.lower()
    code = lower.replace("codigo ", "", 1).replace("código ", "", 1).replace("cã³digo ", "", 1).replace(".", "", 1).lstrip("-+")
    if lower in {"", "sem valor", "sem_valor", "nan", "none", "null", "<na>", "#nulo#", "geral", "sem_entidade", "nao informado", "não informado", "nÃ£o informado"}:
        return ""
    if lower.endswith("_sem_valor") or lower.endswith(" sem valor"):
        return ""
    if (lower.startswith("codigo ") or lower.startswith("código ") or lower.startswith("cã³digo ")) and code.isdigit():
        return ""
    return text


def profile_sentence(row: pd.Series) -> str:
    labels = {
        "perfil_faixa_etaria": "faixa etaria",
        "perfil_genero": "sexo/genero",
        "perfil_instrucao": "escolaridade",
        "perfil_estado_civil": "estado civil",
        "perfil_raca_cor": "raca/cor",
    }
    bits = []
    for col, label in labels.items():
        value = _meaningful_text(row.get(f"{col}_associado", ""))
        if value:
            bits.append(f"{label}: {value}")
    return "; ".join(bits) if bits else "Perfil de eleitor ainda insuficiente neste recorte."


def correlation_strength(value: Any) -> str:
    try:
        v = float(value)
    except Exception:
        return "sem historico suficiente"
    av = abs(v)
    if av >= 0.75:
        return "forte"
    if av >= 0.45:
        return "moderada"
    if av >= 0.20:
        return "fraca"
    return "baixa/instavel"


def spearman_corr(left: pd.Series, right: pd.Series) -> float:
    pair = pd.concat(
        [
            pd.to_numeric(left, errors="coerce"),
            pd.to_numeric(right, errors="coerce"),
        ],
        axis=1,
    ).dropna()
    if len(pair) < 2 or pair.iloc[:, 0].nunique(dropna=True) <= 1 or pair.iloc[:, 1].nunique(dropna=True) <= 1:
        return float("nan")
    return float(pair.iloc[:, 0].rank(method="average").corr(pair.iloc[:, 1].rank(method="average"), method="pearson"))


def correlation_sentence(row: pd.Series) -> str:
    strength = _meaningful_text(row.get("forca_correlacao_historica", "")) or "sem historico suficiente"
    corr = pd.to_numeric(row.get("correlacao_historica_share_partidos"), errors="coerce")
    years = _meaningful_text(row.get("anos_historicos_recorte", ""))
    pairs = _meaningful_text(row.get("pares_anos_correlacionados", ""))
    if pd.notna(corr):
        return f"Correlacao historica {strength} entre shares partidarios dos anos analisados ({years}); pares usados: {pairs or 'sem pares comparaveis'}; Spearman medio {float(corr):.3f}."
    return f"Sem anos/partidos suficientes para medir correlacao historica robusta neste recorte ({years or 'sem anos'})."


def party_prediction_sentence(row: pd.Series) -> str:
    party = _meaningful_text(row.get("partido", "")) or "partido"
    share = pd.to_numeric(row.get("share_pred_2026"), errors="coerce")
    share_txt = f"{float(share) * 100:.1f}%" if pd.notna(share) else "share indefinido"
    trend = _meaningful_text(row.get("tendencia_partido", ""))
    profile = _meaningful_text(row.get("perfil_eleitor_2026", ""))
    corr = _meaningful_text(row.get("justificativa_correlacao", ""))
    base = f"{party} aparece com {share_txt} dos votos validos modelados em 2026"
    if trend:
        base += f", com tendencia historica de {trend}"
    if profile:
        base += f". Perfil associado: {profile}"
    if corr:
        base += f". {corr}"
    return base + "."


def save_prediction_table(
    df: pd.DataFrame,
    name: str,
    tables_dir: Path,
    parquet_dir: Path,
    cfg,
    csv_preview_rows: int | None = None,
) -> dict[str, str]:
    csv_path = tables_dir / f"{name}.csv"
    parquet_path = parquet_dir / f"{name}.parquet"
    to_csv = df
    if csv_preview_rows and df is not None and len(df) > csv_preview_rows:
        to_csv = df.head(csv_preview_rows).copy()
        to_csv["_csv_preview_observacao"] = f"CSV preview limitado a {csv_preview_rows} linhas; Parquet contem a tabela completa."
    save_csv(to_csv, csv_path)
    out = {f"{name}_csv": str(csv_path)}
    if cfg.parquet and save_parquet(df, parquet_path):
        out[f"{name}_parquet"] = str(parquet_path)
    return out


def party_prediction_cards(df: pd.DataFrame, level_label: str, limit: int = 18) -> str:
    if df is None or df.empty:
        return f"<p>Sem cenario partidario para {html.escape(level_label)}.</p>"
    work = df.copy()
    if "cenario" in work.columns and work["cenario"].astype(str).eq("base").any():
        work = work.loc[work["cenario"].astype(str).eq("base")].copy()
    work["share_pred_2026"] = pd.to_numeric(work.get("share_pred_2026"), errors="coerce")
    work = work.loc[work.get("partido", pd.Series(dtype=str)).map(_meaningful_text).ne("")]
    work = work.sort_values("share_pred_2026", ascending=False).head(limit)
    cards = []
    for _, r in work.iterrows():
        loc = " / ".join(x for x in [
            _meaningful_text(r.get("uf", "")),
            _meaningful_text(r.get("nm_municipio", "")) or _meaningful_text(r.get("cd_municipio", "")),
        ] if x)
        cards.append(
            "<details class='persona'>"
            f"<summary>{html.escape(_meaningful_text(r.get('partido', '')))} - {_pct_value(r.get('share_pred_2026'))}</summary>"
            f"<p><span class='pill'>{html.escape(_meaningful_text(r.get('cargo', '')))}</span>"
            f"<span class='pill'>turno {html.escape(_meaningful_text(r.get('turno', '')))}</span>"
            f"<span class='pill'>{html.escape(loc or level_label)}</span></p>"
            f"<p>{html.escape(_meaningful_text(r.get('perfil_eleitor_2026', '')))}</p>"
            f"<p class='smallnote'>{html.escape(_meaningful_text(r.get('justificativa_previsao_partido_2026', '')))}</p>"
            "</details>"
        )
    return "".join(cards) or f"<p>Sem partidos validos para {html.escape(level_label)}.</p>"


def _pct_value(value: Any) -> str:
    num = pd.to_numeric(value, errors="coerce")
    if pd.isna(num):
        return "sem percentual"
    return f"{float(num) * 100:.1f}%"


def decisive_municipalities(scenarios: pd.DataFrame) -> pd.DataFrame:
    if scenarios is None or scenarios.empty:
        return pd.DataFrame()

    df = scenarios.copy()
    group = ["cenario", *SECTION_MODEL_COLS, "cargo", "turno"]
    df["rank"] = df.groupby(group, dropna=False)["share_pred_2026"].rank(method="first", ascending=False)

    top1 = df.loc[df["rank"].eq(1), group + ["entidade", "share_pred_2026", "votos_pred_2026"]].rename(
        columns={"entidade": "lider_pred", "share_pred_2026": "share_lider_pred", "votos_pred_2026": "votos_lider_pred"}
    )
    top2 = df.loc[df["rank"].eq(2), group + ["entidade", "share_pred_2026", "votos_pred_2026"]].rename(
        columns={"entidade": "segundo_pred", "share_pred_2026": "share_segundo_pred", "votos_pred_2026": "votos_segundo_pred"}
    )

    out = top1.merge(top2, on=group, how="left")
    out["margem_pred"] = out["share_lider_pred"] - out["share_segundo_pred"].fillna(0)
    out["indice_decisivo"] = (1 - out["margem_pred"].clip(0, 1)) * np.log1p(
        out["votos_lider_pred"].fillna(0) + out["votos_segundo_pred"].fillna(0)
    )
    return out.sort_values("indice_decisivo", ascending=False)


def run_prediction(global_info: dict[str, Any], cfg) -> dict[str, Any]:
    pred_dir = Path(cfg.out) / "preditivo_2026"
    tables_dir = pred_dir / "tabelas"
    plots_dir = pred_dir / "plots"
    parquet_dir = pred_dir / "parquet"
    for d in [tables_dir, plots_dir, parquet_dir]:
        d.mkdir(parents=True, exist_ok=True)

    global_gold, gold_source = read_global_gold_for_prediction(global_info)
    if global_gold.empty:
        msg = "Base gold global não encontrada. Execute primeiro modo global ou completo."
        (pred_dir / "predicao_nao_executada.txt").write_text(msg, encoding="utf-8")
        return {"status": "sem_base", "html": "", "erro": msg}

    for c in ["votos", "eleitorado", "comparecimento_estimado", "abstencao_estimado"]:
        if c in global_gold.columns:
            global_gold[c] = pd.to_numeric(global_gold[c], errors="coerce")

    model_df = prepare_model_base(global_gold, cfg)
    save_csv(model_df, tables_dir / "base_modelagem.csv")
    if cfg.parquet:
        save_parquet(model_df, parquet_dir / "base_modelagem.parquet")

    backtest = backtest_model(model_df)
    save_csv(backtest, tables_dir / "backtesting.csv")

    scenarios, nacional, mc = build_scenarios(model_df, cfg)
    save_csv(scenarios, tables_dir / "cenarios_municipais.csv")
    save_csv(nacional, tables_dir / "cenarios_nacionais.csv")
    save_csv(mc, tables_dir / "monte_carlo.csv")

    decisive = decisive_municipalities(scenarios)
    save_csv(decisive, tables_dir / "secoes_municipios_decisivos.csv")
    save_csv(decisive, tables_dir / "municipios_decisivos.csv")

    if cfg.parquet:
        save_parquet(scenarios, parquet_dir / "cenarios_municipais.parquet")
        save_parquet(nacional, parquet_dir / "cenarios_nacionais.parquet")
        save_parquet(mc, parquet_dir / "monte_carlo.parquet")
        save_parquet(decisive, parquet_dir / "secoes_municipios_decisivos.parquet")
        save_parquet(decisive, parquet_dir / "municipios_decisivos.parquet")

    party_tables = build_party_2026_tables(global_gold, cfg)
    party_outputs: dict[str, str] = {}
    party_preview_rows = max(500, int(getattr(cfg, "top_n_html", 250) or 250) * 20)
    for table_name, table_df in party_tables.items():
        party_outputs.update(save_prediction_table(table_df, table_name, tables_dir, parquet_dir, cfg, csv_preview_rows=party_preview_rows))

    explanation = build_explainability(global_info, global_gold, model_df, backtest, scenarios, nacional, mc, pred_dir)
    images = plot_prediction(nacional, mc, plots_dir, cfg)

    story = prediction_story(model_df, backtest, nacional, party_tables.get("partidos_2026_brasil", pd.DataFrame()))
    body = f"""
<h2>Storytelling da simulação</h2>
<pre>{html.escape(story)}</pre>

<h2>Explicação detalhada</h2>
<pre>{html.escape(explanation.get("markdown", ""))}</pre>

<h2>Rastreabilidade dos arquivos individuais</h2>
{df_to_html(explanation.get("trace"), cfg.top_n_html)}

<h2>Evidências globais usadas</h2>
{df_to_html(explanation.get("evidence"), cfg.top_n_html)}

<h2>Contribuição por entidade</h2>
{df_to_html(explanation.get("contribution"), cfg.top_n_html)}

<h2>Cenários nacionais</h2>
{df_to_html(nacional, cfg.top_n_html)}

<h2>Cenario 2026 por partido</h2>
<p>Esta secao ignora candidatos e resume a possivel porcentagem de votos por partido, com perfil de eleitor associado e justificativa historica.</p>
<h3>Brasil</h3>
<div class="persona-list">{party_prediction_cards(party_tables.get("partidos_2026_brasil", pd.DataFrame()), "Brasil", cfg.top_n_html)}</div>
<h3>Estados</h3>
<div class="persona-list">{party_prediction_cards(party_tables.get("partidos_2026_estados", pd.DataFrame()), "Estados", cfg.top_n_html)}</div>
<h3>Municipios</h3>
<div class="persona-list">{party_prediction_cards(party_tables.get("partidos_2026_municipios", pd.DataFrame()), "Municipios", cfg.top_n_html)}</div>

<h2>Monte Carlo</h2>
{df_to_html(mc, cfg.top_n_html)}

<h2>Backtesting</h2>
{df_to_html(backtest, cfg.top_n_html)}

<h2>Seções e municípios decisivos</h2>
{df_to_html(decisive, cfg.top_n_html)}

<h2>Gráficos</h2>
{''.join(img_tag(img, pred_dir) for img in images)}
"""
    html_path = pred_dir / "relatorio_simulacao.html"
    save_html(html_path, "Simulação eleitoral 2026 - explicada pelos dados", body)

    return {
        "status": "ok",
        "html": str(html_path),
        "base_modelagem_csv": str(tables_dir / "base_modelagem.csv"),
        "base_modelagem_parquet": str(parquet_dir / "base_modelagem.parquet") if cfg.parquet else "",
        "cenarios_nacionais_csv": str(tables_dir / "cenarios_nacionais.csv"),
        "cenarios_nacionais_parquet": str(parquet_dir / "cenarios_nacionais.parquet") if cfg.parquet else "",
        "cenarios_secao_municipio_csv": str(tables_dir / "cenarios_municipais.csv"),
        "cenarios_secao_municipio_parquet": str(parquet_dir / "cenarios_municipais.parquet") if cfg.parquet else "",
        "secoes_municipios_decisivos_csv": str(tables_dir / "secoes_municipios_decisivos.csv"),
        "secoes_municipios_decisivos_parquet": str(parquet_dir / "secoes_municipios_decisivos.parquet") if cfg.parquet else "",
        "monte_carlo_csv": str(tables_dir / "monte_carlo.csv"),
        "monte_carlo_parquet": str(parquet_dir / "monte_carlo.parquet") if cfg.parquet else "",
        "explicacao_md": explanation.get("markdown_path", ""),
        **party_outputs,
    }


def prediction_story(model_df: pd.DataFrame, backtest: pd.DataFrame, nacional: pd.DataFrame, party_brasil: pd.DataFrame | None = None) -> str:
    lines = []
    lines.append("A simulação parte da base gold global, que por sua vez foi construída a partir das análises individuais dos JSONs.")
    lines.append("O código detecta os anos existentes nos dados; não há ano histórico fixado manualmente.")

    if model_df is not None and not model_df.empty:
        years = sorted(pd.to_numeric(model_df["ano_num"], errors="coerce").dropna().astype(int).unique().tolist())
        lines.append("Anos efetivamente usados na modelagem: " + (", ".join(map(str, years)) if years else "nenhum ano numérico detectado."))

    if backtest is not None and not backtest.empty and "mae_share" in backtest.columns:
        vals = pd.to_numeric(backtest["mae_share"], errors="coerce").dropna()
        if not vals.empty:
            lines.append(f"Erro médio observado no backtesting: {vals.mean()*100:.2f} pontos percentuais.")

    if nacional is not None and not nacional.empty:
        base = nacional.loc[nacional["cenario"].astype(str).eq("base")].copy()
        if base.empty:
            base = nacional.copy()
        top = base.sort_values("share_nacional_pred_2026", ascending=False).head(5)
        if not top.empty:
            lines.append("Cenário base: " + ", ".join(
                f"{r['entidade']} ({float(r['share_nacional_pred_2026'])*100:.2f}%)"
                for _, r in top.iterrows()
            ))

    lines.append("A explicação detalhada está em preditivo_2026/explicabilidade/explicacao_detalhada_simulacao.md.")
    if party_brasil is not None and not party_brasil.empty:
        pb = party_brasil.copy()
        if "cenario" in pb.columns and pb["cenario"].astype(str).eq("base").any():
            pb = pb.loc[pb["cenario"].astype(str).eq("base")].copy()
        pb["share_pred_2026"] = pd.to_numeric(pb.get("share_pred_2026"), errors="coerce")
        if "partido" in pb.columns:
            top_party = pb.loc[pb["partido"].map(_meaningful_text).ne("")].sort_values("share_pred_2026", ascending=False).head(8)
            if not top_party.empty:
                lines.append("Cenario partidario 2026 no Brasil: " + ", ".join(
                    f"{r.get('partido')} ({float(r.get('share_pred_2026'))*100:.2f}%)"
                    for _, r in top_party.iterrows()
                    if pd.notna(r.get("share_pred_2026"))
                ))
                corr = top_party.get("justificativa_correlacao", pd.Series(dtype=str)).dropna().astype(str).head(1)
                if not corr.empty:
                    lines.append("Justificativa historica partidaria: " + corr.iloc[0])

    return "\n".join(lines)
