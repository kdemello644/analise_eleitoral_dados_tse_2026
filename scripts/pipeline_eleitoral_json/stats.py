from __future__ import annotations

import pandas as pd

from .utils import parse_number


def numeric_stats(df: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or profile is None or profile.empty:
        return pd.DataFrame()

    cols = profile.loc[
        profile["usar_como_metrica"].astype(str).isin(["True", "true", "1"]),
        "coluna"
    ].astype(str).tolist()

    records = []
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col].map(parse_number).dropna().astype(float)
        if s.empty:
            continue
        records.append({
            "coluna": col,
            "n": int(len(s)),
            "media": float(s.mean()),
            "desvio": float(s.std()) if len(s) > 1 else 0.0,
            "min": float(s.min()),
            "p05": float(s.quantile(0.05)),
            "p25": float(s.quantile(0.25)),
            "mediana": float(s.median()),
            "p75": float(s.quantile(0.75)),
            "p95": float(s.quantile(0.95)),
            "max": float(s.max()),
            "soma": float(s.sum()),
        })
    return pd.DataFrame(records)


def correlations(df: pd.DataFrame, profile: pd.DataFrame, max_cols: int = 30) -> pd.DataFrame:
    if df is None or df.empty or profile is None or profile.empty:
        return pd.DataFrame()

    cols = profile.loc[
        profile["usar_como_metrica"].astype(str).isin(["True", "true", "1"]),
        "coluna"
    ].astype(str).tolist()[:max_cols]

    if len(cols) < 2:
        return pd.DataFrame()

    num = pd.DataFrame({c: df[c].map(parse_number) for c in cols if c in df.columns})
    num = num.dropna(axis=1, thresh=max(5, int(len(num) * 0.2)))
    num = num.loc[:, num.nunique(dropna=True) > 1]

    if num.shape[1] < 2:
        return pd.DataFrame()

    mat = num.corr(method="pearson", min_periods=5)
    records = []
    columns = list(mat.columns)
    for i, c1 in enumerate(columns):
        for c2 in columns[i+1:]:
            val = mat.loc[c1, c2]
            if pd.notna(val):
                records.append({
                    "coluna_1": c1,
                    "coluna_2": c2,
                    "pearson": float(val),
                    "abs_pearson": abs(float(val)),
                })

    out = pd.DataFrame(records)
    return out.sort_values("abs_pearson", ascending=False) if not out.empty else out
