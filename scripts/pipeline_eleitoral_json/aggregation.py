from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
import logging

import numpy as np
import pandas as pd

from .json_reader import iter_json_records
from .profiler import role_by_name
from .utils import clean_memory, compact_code, extract_years_from_value, parse_number, safe_name, safe_text, save_csv, save_parquet
from .discrete import label_category_value


GOLD_KEYS = ["ano", "uf", "cd_municipio", "nm_municipio", "zona", "secao", "local_votacao", "bairro", "turno", "cargo", "partido", "candidato", "ideologia", "coalizao", "perfil_faixa_etaria", "perfil_genero", "perfil_instrucao", "perfil_estado_civil", "perfil_raca_cor"]
GOLD_METRICS = ["votos", "eleitorado", "comparecimento", "abstencao", "brancos", "nulos", "validos"]
GOLD_PLOT_CATEGORY_COLS = [
    "ano",
    "uf",
    "cargo",
    "turno",
    "partido",
    "candidato",
    "perfil_faixa_etaria",
    "perfil_genero",
    "perfil_instrucao",
    "perfil_estado_civil",
    "perfil_raca_cor",
]
GOLD_PLOT_METRICS = ["votos", "eleitorado", "comparecimento_estimado", "abstencao_estimado"]
ROLE_FIELDS = {
    "ano",
    "uf",
    "cd_municipio",
    "nm_municipio",
    "zona",
    "secao",
    "local_votacao",
    "bairro",
    "turno",
    "cargo",
    "partido",
    "candidato",
    "ideologia",
    "coalizao",
    "perfil_faixa_etaria",
    "perfil_genero",
    "perfil_instrucao",
    "perfil_estado_civil",
    "perfil_raca_cor",
}

GOLD_PARQUET_COLS = [
    *GOLD_KEYS,
    "votos",
    "eleitorado",
    "comparecimento_estimado",
    "abstencao_estimado",
    "brancos",
    "nulos",
    "validos_estimados",
    "pct_comparecimento",
    "pct_abstencao",
    "share_votos_grupo",
    "linhas_origem",
    "arquivo_origem",
    "aggregation_mode",
]

ELECTORATE_PROFILE_COLS = [
    "perfil_faixa_etaria",
    "perfil_genero",
    "perfil_instrucao",
    "perfil_estado_civil",
    "perfil_raca_cor",
]

ELECTORATE_PARQUET_COLS = [
    "ano",
    "uf",
    "cd_municipio",
    "nm_municipio",
    "zona",
    "secao",
    "cargo",
    "turno",
    *ELECTORATE_PROFILE_COLS,
    "eleitorado",
    "comparecimento_estimado",
    "abstencao_estimado",
    "pct_comparecimento",
    "pct_abstencao",
    "arquivo_origem",
]


def first_by_roles(record: dict[str, Any], roles: set[str]) -> Any:
    candidates = []
    for col, val in record.items():
        role = role_by_name(col)
        if role in roles and safe_text(val):
            candidates.append((_field_value_score(col, val, role), val))
    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]
    return ""


def _field_value_score(col: str, value: Any, role: str) -> int:
    c = str(col).replace("\ufeff", "").strip().upper()
    text = safe_text(value, "")
    numeric_like = parse_number(text)
    is_numeric = pd.notna(numeric_like)
    score = 0

    if role in {"partido", "candidato", "coalizao", "ideologia", "nm_municipio", "local_votacao", "bairro"}:
        score += 60 if not is_numeric else -30
        if c.startswith(("NM_", "DS_", "SG_")) or "NOME" in c or "DESCR" in c:
            score += 30
        if role == "partido" and ("SG_PARTIDO" in c or c.startswith("SG_")):
            score += 40
        if role == "candidato" and ("URNA" in c or "NM_" in c or "NOME" in c):
            score += 40
        if c.startswith(("CD_", "NR_", "SQ_", "ID_")) or "CODIGO" in c:
            score -= 40
    elif role in {
        "cargo",
        "turno",
        "perfil_faixa_etaria",
        "perfil_genero",
        "perfil_instrucao",
        "perfil_estado_civil",
        "perfil_raca_cor",
    }:
        if c.startswith(("DS_", "NM_")) or "DESCR" in c:
            score += 40
        if c.startswith(("CD_", "NR_", "TP_")):
            score -= 5
    return score


def detect_record_year(record: dict[str, Any]) -> str:
    direct = first_by_roles(record, {"ano"})
    if safe_text(direct):
        years = extract_years_from_value(direct)
        if years:
            return str(years[0])
        return compact_code(direct)

    # fallback data-driven: procura qualquer coluna cujo valor pareça ano.
    for value in record.values():
        years = extract_years_from_value(value)
        if len(years) == 1:
            return str(years[0])
    return ""


def label_if_present(value: Any, col: str, role: str) -> str:
    return label_category_value(value, col=col, role=role) if safe_text(value) else ""


def record_to_gold(record: dict[str, Any]) -> dict[str, Any]:
    row = {
        "ano": detect_record_year(record),
        "uf": compact_code(first_by_roles(record, {"uf"})),
        "cd_municipio": compact_code(first_by_roles(record, {"cd_municipio"})),
        "nm_municipio": safe_text(first_by_roles(record, {"nm_municipio"})),
        "zona": compact_code(first_by_roles(record, {"zona"})),
        "secao": compact_code(first_by_roles(record, {"secao"})),
        "local_votacao": safe_text(first_by_roles(record, {"local_votacao"})),
        "bairro": safe_text(first_by_roles(record, {"bairro"})),
        "turno": label_if_present(first_by_roles(record, {"turno"}), col="turno", role="turno"),
        "cargo": label_if_present(first_by_roles(record, {"cargo"}), col="cargo", role="cargo"),
        "partido": safe_text(first_by_roles(record, {"partido"})),
        "candidato": safe_text(first_by_roles(record, {"candidato"})),
        "ideologia": safe_text(first_by_roles(record, {"ideologia"})),
        "coalizao": safe_text(first_by_roles(record, {"coalizao"})),
        "perfil_faixa_etaria": label_if_present(first_by_roles(record, {"perfil_faixa_etaria"}), col="perfil_faixa_etaria", role="perfil_faixa_etaria"),
        "perfil_genero": label_if_present(first_by_roles(record, {"perfil_genero"}), col="perfil_genero", role="perfil_genero"),
        "perfil_instrucao": label_if_present(first_by_roles(record, {"perfil_instrucao"}), col="perfil_instrucao", role="perfil_instrucao"),
        "perfil_estado_civil": label_if_present(first_by_roles(record, {"perfil_estado_civil"}), col="perfil_estado_civil", role="perfil_estado_civil"),
        "perfil_raca_cor": label_if_present(first_by_roles(record, {"perfil_raca_cor"}), col="perfil_raca_cor", role="perfil_raca_cor"),
        "linhas_origem": 1,
    }
    for metric in GOLD_METRICS:
        row[metric] = 0.0

    for col, value in record.items():
        role = role_by_name(col)
        if role in GOLD_METRICS:
            num = parse_number(value)
            if pd.notna(num):
                row[role] += float(num)

    return row


def record_to_gold_cached(record: dict[str, Any], role_cache: dict[str, str]) -> dict[str, Any]:
    candidates: dict[str, list[tuple[int, Any]]] = defaultdict(list)
    metrics = {metric: 0.0 for metric in GOLD_METRICS}

    for col, value in record.items():
        role = role_cache.get(col)
        if role is None:
            role = role_by_name(col)
            role_cache[col] = role
        if not role:
            continue
        if role in GOLD_METRICS:
            num = parse_number(value)
            if pd.notna(num):
                metrics[role] += float(num)
        if role in ROLE_FIELDS and safe_text(value):
            candidates[role].append((_field_value_score(col, value, role), value))

    def first(role: str) -> Any:
        vals = candidates.get(role, [])
        if not vals:
            return ""
        vals.sort(key=lambda item: item[0], reverse=True)
        return vals[0][1]

    ano = ""
    direct_year = first("ano")
    if safe_text(direct_year):
        years = extract_years_from_value(direct_year)
        ano = str(years[0]) if years else compact_code(direct_year)
    if not ano:
        for value in record.values():
            years = extract_years_from_value(value)
            if len(years) == 1:
                ano = str(years[0])
                break

    row = {
        "ano": ano,
        "uf": compact_code(first("uf")),
        "cd_municipio": compact_code(first("cd_municipio")),
        "nm_municipio": safe_text(first("nm_municipio")),
        "zona": compact_code(first("zona")),
        "secao": compact_code(first("secao")),
        "local_votacao": safe_text(first("local_votacao")),
        "bairro": safe_text(first("bairro")),
        "turno": label_if_present(first("turno"), col="turno", role="turno"),
        "cargo": label_if_present(first("cargo"), col="cargo", role="cargo"),
        "partido": safe_text(first("partido")),
        "candidato": safe_text(first("candidato")),
        "ideologia": safe_text(first("ideologia")),
        "coalizao": safe_text(first("coalizao")),
        "perfil_faixa_etaria": label_if_present(first("perfil_faixa_etaria"), col="perfil_faixa_etaria", role="perfil_faixa_etaria"),
        "perfil_genero": label_if_present(first("perfil_genero"), col="perfil_genero", role="perfil_genero"),
        "perfil_instrucao": label_if_present(first("perfil_instrucao"), col="perfil_instrucao", role="perfil_instrucao"),
        "perfil_estado_civil": label_if_present(first("perfil_estado_civil"), col="perfil_estado_civil", role="perfil_estado_civil"),
        "perfil_raca_cor": label_if_present(first("perfil_raca_cor"), col="perfil_raca_cor", role="perfil_raca_cor"),
        "linhas_origem": 1,
        **metrics,
    }
    return row


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()

    for c in GOLD_KEYS:
        if c not in df.columns:
            df[c] = ""

    for c in GOLD_METRICS + ["linhas_origem"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["validos_estimados"] = np.where(df["validos"] > 0, df["validos"], df["votos"])
    df["comparecimento_estimado"] = np.where(
        df["comparecimento"] > 0,
        df["comparecimento"],
        df["validos_estimados"] + df["brancos"] + df["nulos"],
    )
    df["abstencao_estimado"] = np.where(
        df["abstencao"] > 0,
        df["abstencao"],
        np.maximum(df["eleitorado"] - df["comparecimento_estimado"], 0),
    )
    df["pct_comparecimento"] = np.where(df["eleitorado"] > 0, df["comparecimento_estimado"] / df["eleitorado"], np.nan)
    df["pct_abstencao"] = np.where(df["eleitorado"] > 0, df["abstencao_estimado"] / df["eleitorado"], np.nan)

    group = ["ano", "uf", "cd_municipio", "turno", "cargo"]
    total = df.groupby(group, dropna=False)["votos"].transform("sum")
    df["share_votos_grupo"] = np.where(total > 0, df["votos"] / total, np.nan)
    return df


def _aggregate_gold_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    group = GOLD_KEYS
    agg_cols = GOLD_METRICS + ["linhas_origem"]
    return df.groupby(group, dropna=False)[agg_cols].sum().reset_index()


def _merge_gold_partials(partials: list[pd.DataFrame]) -> pd.DataFrame:
    valid = [p for p in partials if p is not None and not p.empty]
    if not valid:
        return pd.DataFrame()
    merged = pd.concat(valid, ignore_index=True, sort=False)
    group = GOLD_KEYS
    agg_cols = GOLD_METRICS + ["linhas_origem"]
    return merged.groupby(group, dropna=False)[agg_cols].sum().reset_index()


def aggregate_records_to_gold(records: list[dict[str, Any]], arquivo_origem: str) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()

    rows = [record_to_gold(r) for r in records]
    gold = _aggregate_gold_rows(rows)
    if not gold.empty:
        gold["arquivo_origem"] = arquivo_origem
    return add_derived_metrics(gold)


def aggregate_full_json(path: Path, arquivo_origem: str, chunk_rows: int = 75000) -> pd.DataFrame:
    chunk_rows = max(1000, int(chunk_rows or 75000))
    rows: list[dict[str, Any]] = []
    partials: list[pd.DataFrame] = []
    seen = 0
    role_cache: dict[str, str] = {}

    for record in iter_json_records(path):
        rows.append(record_to_gold_cached(record, role_cache))
        seen += 1
        if len(rows) >= chunk_rows:
            partials.append(_aggregate_gold_rows(rows))
            rows = []
            if len(partials) >= 12:
                partials = [_merge_gold_partials(partials)]
                clean_memory()
            if seen % (chunk_rows * 10) == 0:
                logging.info("Agregacao streaming %s: %s registros lidos, %s grupos parciais", arquivo_origem, seen, sum(len(p) for p in partials if p is not None))

    if rows:
        partials.append(_aggregate_gold_rows(rows))
        rows = []

    gold = _merge_gold_partials(partials)
    if not gold.empty:
        gold["arquivo_origem"] = arquivo_origem
    out = add_derived_metrics(gold)
    clean_memory()
    return out


def aggregate_full_json_to_parquet_parts(
    path: Path,
    arquivo_origem: str,
    parts_dir: Path,
    chunk_rows: int = 75000,
    preview_rows: int = 200000,
    cfg: Any | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Stream a large JSON into compact Parquet gold parts.

    This keeps all source records in the persisted analytical layer, but only
    returns a bounded preview DataFrame for HTML and per-file diagnostics.
    Duplicate gold keys may appear across parts; downstream global grouping
    sums them again without dropping data.
    """
    parts_dir.mkdir(parents=True, exist_ok=True)
    chunk_rows = max(1000, int(chunk_rows or 75000))
    preview_rows = max(0, int(preview_rows or 0))

    rows: list[dict[str, Any]] = []
    preview_parts: list[pd.DataFrame] = []
    part_rows: list[dict[str, Any]] = []
    plot_acc = _new_gold_plot_accumulator()
    seen = 0
    part_idx = 0
    role_cache: dict[str, str] = {}

    for record in iter_json_records(path):
        rows.append(record_to_gold_cached(record, role_cache))
        seen += 1
        if len(rows) >= chunk_rows:
            part_idx += 1
            part, meta = _write_gold_part(rows, arquivo_origem, parts_dir, part_idx, cfg=cfg)
            part_rows.append(meta)
            _update_gold_plot_accumulator(plot_acc, part)
            if preview_rows and sum(len(p) for p in preview_parts) < preview_rows and part is not None and not part.empty:
                need = preview_rows - sum(len(p) for p in preview_parts)
                preview_parts.append(part.head(need))
            rows = []
            clean_memory()
            if seen % (chunk_rows * 10) == 0:
                logging.info("Parquet gold streaming %s: %s registros lidos, %s partes", arquivo_origem, seen, part_idx)

    if rows:
        part_idx += 1
        part, meta = _write_gold_part(rows, arquivo_origem, parts_dir, part_idx, cfg=cfg)
        part_rows.append(meta)
        _update_gold_plot_accumulator(plot_acc, part)
        if preview_rows and sum(len(p) for p in preview_parts) < preview_rows and part is not None and not part.empty:
            need = preview_rows - sum(len(p) for p in preview_parts)
            preview_parts.append(part.head(need))

    manifest = pd.DataFrame(part_rows)
    save_csv(manifest, parts_dir / "manifesto_partes_gold.csv")
    save_parquet(manifest, parts_dir / "manifesto_partes_gold.parquet")
    plot_summary = _gold_plot_summary_frame(plot_acc)
    save_parquet(plot_summary, parts_dir / "resumo_plot_gold_completo.parquet")
    save_csv(plot_summary.head(5000), parts_dir / "resumo_plot_gold_completo_preview.csv")
    plot_manifest = _write_gold_plot_datasets(plot_summary, parts_dir / "plots_data")
    preview = pd.concat(preview_parts, ignore_index=True, sort=False) if preview_parts else pd.DataFrame()
    info = {
        "modo_gold": "streaming_parquet_parts",
        "linhas_json_lidas": int(seen),
        "partes_gold": int(part_idx),
        "gold_parts_dir": str(parts_dir),
        "gold_parts_manifest": str(parts_dir / "manifesto_partes_gold.csv"),
        "gold_parts_manifest_parquet": str(parts_dir / "manifesto_partes_gold.parquet"),
        "gold_plot_summary_parquet": str(parts_dir / "resumo_plot_gold_completo.parquet"),
        "gold_plot_datasets_dir": str(parts_dir / "plots_data"),
        "gold_plot_manifest_parquet": str(plot_manifest.get("parquet", "")),
        "gold_plot_manifest_csv": str(plot_manifest.get("csv", "")),
        "linhas_preview_gold": int(len(preview)),
    }
    clean_memory()
    return preview, info


def _write_gold_part(
    rows: list[dict[str, Any]],
    arquivo_origem: str,
    parts_dir: Path,
    part_idx: int,
    cfg: Any | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    part = _aggregate_gold_rows(rows)
    if not part.empty:
        part["arquivo_origem"] = arquivo_origem
        part["aggregation_mode"] = "full_json_streaming_parquet_parts"
        part = add_derived_metrics(part)
    parquet_path = parts_dir / f"gold_part_{part_idx:05d}.parquet"
    csv_path = parts_dir / f"gold_part_{part_idx:05d}.csv.gz"
    parquet_part = clean_gold_for_parquet(part)
    parquet_ok = save_parquet(parquet_part, parquet_path)
    csv_value = ""
    if not parquet_ok:
        parquet_part.to_csv(csv_path, sep=";", index=False, encoding="utf-8-sig", compression="gzip")
        csv_value = str(csv_path)
    electorate_partition_count = 0
    if cfg is not None and getattr(cfg, "partition_electorate_by_state", True):
        electorate_partition_count = write_electorate_state_partitions(part, parts_dir, part_idx, cfg)
    return part, {
        "parte": part_idx,
        "linhas": int(len(parquet_part)),
        "parquet": str(parquet_path) if parquet_ok else "",
        "csv": csv_value,
        "particoes_eleitorado_uf": int(electorate_partition_count),
    }


def clean_gold_for_parquet(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for col in GOLD_PARQUET_COLS:
        if col not in out.columns:
            out[col] = 0.0 if col in {"votos", "eleitorado", "comparecimento_estimado", "abstencao_estimado", "brancos", "nulos", "validos_estimados", "pct_comparecimento", "pct_abstencao", "share_votos_grupo", "linhas_origem"} else ""
    return out[GOLD_PARQUET_COLS].copy()


def write_electorate_state_partitions(part: pd.DataFrame | None, parts_dir: Path, part_idx: int, cfg) -> int:
    if part is None or part.empty:
        return 0
    work = part.copy()
    for col in ELECTORATE_PARQUET_COLS:
        if col not in work.columns:
            work[col] = 0.0 if col in {"eleitorado", "comparecimento_estimado", "abstencao_estimado", "pct_comparecimento", "pct_abstencao"} else ""
    profile_mask = pd.Series(False, index=work.index)
    for col in ELECTORATE_PROFILE_COLS:
        if col in work.columns:
            profile_mask = profile_mask | work[col].map(lambda x: safe_text(x, "")).ne("")
    electorate_mask = pd.to_numeric(work.get("eleitorado", 0), errors="coerce").fillna(0).gt(0)
    work = work.loc[profile_mask | electorate_mask, ELECTORATE_PARQUET_COLS].copy()
    if work.empty:
        return 0

    out_root = parts_dir / "eleitorado_por_uf"
    out_root.mkdir(parents=True, exist_ok=True)
    max_rows = max(1000, int(getattr(cfg, "parquet_partition_rows", 250000) or 250000))
    workers = max(1, int(getattr(cfg, "workers_parquet", 2) or 2))
    tasks: list[tuple[pd.DataFrame, Path]] = []
    for uf, g in work.groupby("uf", dropna=False):
        uf_name = safe_name(safe_text(uf, "UF_DESCONHECIDA") or "UF_DESCONHECIDA", 40)
        uf_dir = out_root / f"uf={uf_name}"
        for chunk_idx, start in enumerate(range(0, len(g), max_rows), start=1):
            chunk = g.iloc[start:start + max_rows].copy()
            path = uf_dir / f"eleitorado_part_{part_idx:05d}_{chunk_idx:03d}.parquet"
            tasks.append((chunk, path))

    def _save(task: tuple[pd.DataFrame, Path]) -> bool:
        chunk, path = task
        return save_parquet(chunk, path)

    ok = 0
    if workers > 1 and len(tasks) > 1:
        with ThreadPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
            futures = [pool.submit(_save, task) for task in tasks]
            for future in as_completed(futures):
                ok += 1 if future.result() else 0
    else:
        for task in tasks:
            ok += 1 if _save(task) else 0
    return ok


def _new_gold_plot_accumulator() -> dict[str, Any]:
    return {
        "cat_linhas": defaultdict(Counter),
        "cat_votos": defaultdict(Counter),
        "cat_eleitorado": defaultdict(Counter),
        "ano_metric": defaultdict(Counter),
        "uf_metric": defaultdict(Counter),
    }


def _update_gold_plot_accumulator(acc: dict[str, Any], part: pd.DataFrame | None) -> None:
    if part is None or part.empty:
        return
    work = part.copy()
    for metric in ["linhas_origem", *GOLD_PLOT_METRICS]:
        if metric not in work.columns:
            work[metric] = 0.0
        work[metric] = pd.to_numeric(work[metric], errors="coerce").fillna(0.0)

    for col in GOLD_PLOT_CATEGORY_COLS:
        if col not in work.columns:
            continue
        tmp = work[[col, "linhas_origem", "votos", "eleitorado"]].copy()
        tmp[col] = tmp[col].map(lambda x: safe_text(x, ""))
        tmp = tmp.loc[tmp[col].astype(str).str.strip().ne("")]
        if tmp.empty:
            continue
        grouped = tmp.groupby(col, dropna=False)[["linhas_origem", "votos", "eleitorado"]].sum()
        for value, row in grouped.iterrows():
            key = safe_text(value, "")
            if not key:
                continue
            acc["cat_linhas"][col][key] += float(row.get("linhas_origem", 0) or 0)
            acc["cat_votos"][col][key] += float(row.get("votos", 0) or 0)
            acc["cat_eleitorado"][col][key] += float(row.get("eleitorado", 0) or 0)

    for key_col, bucket in [("ano", "ano_metric"), ("uf", "uf_metric")]:
        if key_col not in work.columns:
            continue
        tmp = work[[key_col, *GOLD_PLOT_METRICS]].copy()
        tmp[key_col] = tmp[key_col].map(lambda x: safe_text(x, ""))
        tmp = tmp.loc[tmp[key_col].astype(str).str.strip().ne("")]
        if tmp.empty:
            continue
        grouped = tmp.groupby(key_col, dropna=False)[GOLD_PLOT_METRICS].sum()
        for value, row in grouped.iterrows():
            key = safe_text(value, "")
            if not key:
                continue
            for metric in GOLD_PLOT_METRICS:
                acc[bucket][metric][key] += float(row.get(metric, 0) or 0)


def _gold_plot_summary_frame(acc: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for col, counter in acc["cat_linhas"].items():
        for value, total in counter.items():
            rows.append({
                "tipo_resumo": "categoria",
                "campo": col,
                "valor": value,
                "metrica": "linhas_origem",
                "total": float(total),
                "votos": float(acc["cat_votos"][col].get(value, 0)),
                "eleitorado": float(acc["cat_eleitorado"][col].get(value, 0)),
            })
    for bucket, tipo in [("ano_metric", "metrica_por_ano"), ("uf_metric", "metrica_por_uf")]:
        for metric, counter in acc[bucket].items():
            for value, total in counter.items():
                rows.append({
                    "tipo_resumo": tipo,
                    "campo": "ano" if bucket == "ano_metric" else "uf",
                    "valor": value,
                    "metrica": metric,
                    "total": float(total),
                    "votos": np.nan,
                    "eleitorado": np.nan,
                })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["tipo_resumo", "campo", "metrica", "total"], ascending=[True, True, True, False])
    return out


def _write_gold_plot_datasets(summary: pd.DataFrame, out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    if summary is None or summary.empty:
        manifest = pd.DataFrame(columns=["chart_id", "tipo_grafico", "campo", "metrica", "parquet", "csv"])
        manifest_csv = out_dir / "manifesto_plots_data.csv"
        manifest_parquet = out_dir / "manifesto_plots_data.parquet"
        save_csv(manifest, manifest_csv)
        save_parquet(manifest, manifest_parquet)
        return {"csv": str(manifest_csv), "parquet": str(manifest_parquet)}

    cat = summary.loc[summary["tipo_resumo"].astype(str).eq("categoria")].copy()
    for field, group in cat.groupby("campo", dropna=False):
        metric = "votos" if pd.to_numeric(group.get("votos"), errors="coerce").fillna(0).sum() > 0 else "eleitorado" if pd.to_numeric(group.get("eleitorado"), errors="coerce").fillna(0).sum() > 0 else "total"
        data = group[["valor", "total", "votos", "eleitorado"]].copy()
        data["valor"] = data["valor"].astype(str)
        data[metric] = pd.to_numeric(data[metric], errors="coerce").fillna(0)
        data = data.sort_values(metric, ascending=False)
        chart_id = f"categoria_{safe_name(field)}"
        parquet_path = out_dir / f"plot_{chart_id}.parquet"
        csv_path = out_dir / f"plot_{chart_id}.csv"
        parquet_value = str(parquet_path) if save_parquet(data, parquet_path) else ""
        csv_value = ""
        if not parquet_value:
            save_csv(data.head(5000), csv_path)
            csv_value = str(csv_path)
        manifest_rows.append({
            "chart_id": chart_id,
            "tipo_grafico": "categoria",
            "campo": field,
            "metrica": metric,
            "parquet": parquet_value,
            "csv": csv_value,
        })

    for tipo, suffix in [("metrica_por_ano", "por_ano"), ("metrica_por_uf", "por_uf")]:
        sub = summary.loc[summary["tipo_resumo"].astype(str).eq(tipo)].copy()
        for metric, group in sub.groupby("metrica", dropna=False):
            data = group[["valor", "total"]].copy()
            data["valor"] = data["valor"].astype(str)
            data["total"] = pd.to_numeric(data["total"], errors="coerce").fillna(0)
            if suffix == "por_ano":
                data["_ord"] = pd.to_numeric(data["valor"], errors="coerce")
                data = data.sort_values("_ord").drop(columns=["_ord"])
            else:
                data = data.sort_values("total", ascending=False)
            chart_id = f"{safe_name(metric)}_{suffix}"
            parquet_path = out_dir / f"plot_{chart_id}.parquet"
            csv_path = out_dir / f"plot_{chart_id}.csv"
            parquet_value = str(parquet_path) if save_parquet(data, parquet_path) else ""
            csv_value = ""
            if not parquet_value:
                save_csv(data.head(5000), csv_path)
                csv_value = str(csv_path)
            manifest_rows.append({
                "chart_id": chart_id,
                "tipo_grafico": suffix,
                "campo": "ano" if suffix == "por_ano" else "uf",
                "metrica": metric,
                "parquet": parquet_value,
                "csv": csv_value,
            })

    manifest = pd.DataFrame(manifest_rows)
    manifest_csv = out_dir / "manifesto_plots_data.csv"
    manifest_parquet = out_dir / "manifesto_plots_data.parquet"
    save_csv(manifest, manifest_csv)
    save_parquet(manifest, manifest_parquet)
    return {"csv": str(manifest_csv), "parquet": str(manifest_parquet)}


def consolidate_municipal(global_gold: pd.DataFrame) -> pd.DataFrame:
    if global_gold is None or global_gold.empty:
        return pd.DataFrame()

    df = add_derived_metrics(global_gold.copy())
    for c in ["ano", "uf", "cd_municipio", "nm_municipio", "cargo", "turno"]:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].map(lambda x: safe_text(x))

    metrics = ["votos", "eleitorado", "comparecimento_estimado", "abstencao_estimado", "brancos", "nulos", "validos_estimados"]
    group = ["ano", "uf", "cd_municipio", "nm_municipio", "cargo", "turno"]
    municipal = df.groupby(group, dropna=False)[metrics].sum().reset_index()
    municipal["pct_comparecimento"] = np.where(municipal["eleitorado"] > 0, municipal["comparecimento_estimado"] / municipal["eleitorado"], np.nan)
    municipal["pct_abstencao"] = np.where(municipal["eleitorado"] > 0, municipal["abstencao_estimado"] / municipal["eleitorado"], np.nan)

    entity_col = "partido" if "partido" in df.columns and df["partido"].astype(str).str.strip().ne("").any() else "candidato"
    if entity_col in df.columns:
        ent = df.groupby(group + [entity_col], dropna=False)["votos"].sum().reset_index()
        ent["rank"] = ent.groupby(group, dropna=False)["votos"].rank(method="first", ascending=False)

        top1 = ent.loc[ent["rank"].eq(1), group + [entity_col, "votos"]].rename(
            columns={entity_col: "lider", "votos": "votos_lider"}
        )
        top2 = ent.loc[ent["rank"].eq(2), group + [entity_col, "votos"]].rename(
            columns={entity_col: "segundo", "votos": "votos_segundo"}
        )

        municipal = municipal.merge(top1, on=group, how="left").merge(top2, on=group, how="left")
        municipal["margem_lider"] = np.where(
            municipal["votos"] > 0,
            (municipal["votos_lider"].fillna(0) - municipal["votos_segundo"].fillna(0)) / municipal["votos"],
            np.nan,
        )
        municipal["competitividade"] = 1 - municipal["margem_lider"].clip(0, 1)

    municipal["ano_num"] = pd.to_numeric(municipal["ano"], errors="coerce")
    municipal = municipal.sort_values(["uf", "cd_municipio", "cargo", "turno", "ano_num"])
    for c in ["votos", "eleitorado", "pct_comparecimento", "pct_abstencao", "competitividade"]:
        if c in municipal.columns:
            municipal[f"{c}_lag"] = municipal.groupby(["uf", "cd_municipio", "cargo", "turno"], dropna=False)[c].shift(1)
            municipal[f"{c}_delta"] = municipal[c] - municipal[f"{c}_lag"]
            municipal[f"{c}_delta_pct"] = np.where(
                municipal[f"{c}_lag"].abs() > 0,
                municipal[f"{c}_delta"] / municipal[f"{c}_lag"].abs(),
                np.nan,
            )

    return municipal


def build_file_temporal_inventory(results: list[dict[str, Any]], global_gold: pd.DataFrame) -> pd.DataFrame:
    rows = []
    gold = global_gold.copy() if global_gold is not None else pd.DataFrame()

    if not gold.empty:
        if "arquivo_origem" not in gold.columns:
            gold["arquivo_origem"] = ""
        if "ano" not in gold.columns:
            gold["ano"] = ""

        for file, g in gold.groupby("arquivo_origem", dropna=False):
            years_content = sorted([
                int(float(x)) for x in pd.to_numeric(g["ano"], errors="coerce").dropna().unique()
                if 1900 <= int(float(x)) <= 2100
            ])
            years_name = extract_years_from_value(file)
            rows.append({
                "arquivo_origem": file,
                "anos_detectados_conteudo": ", ".join(map(str, years_content)),
                "anos_detectados_nome": ", ".join(map(str, years_name)),
                "ano_min": min(years_content) if years_content else (min(years_name) if years_name else np.nan),
                "ano_max": max(years_content) if years_content else (max(years_name) if years_name else np.nan),
                "qtd_anos": len(years_content),
                "ufs": g["uf"].replace("", np.nan).dropna().nunique() if "uf" in g.columns else 0,
                "municipios": g["cd_municipio"].replace("", np.nan).dropna().nunique() if "cd_municipio" in g.columns else 0,
                "cargos": ", ".join(sorted([safe_text(x) for x in g.get("cargo", pd.Series(dtype=str)).dropna().unique() if safe_text(x)])[:30]),
                "linhas_gold": len(g),
            })

    seen = set(r["arquivo_origem"] for r in rows)
    for r in results:
        rel = r.get("relativo", "")
        if rel in seen:
            continue
        years_name = extract_years_from_value(rel)
        rows.append({
            "arquivo_origem": rel,
            "anos_detectados_conteudo": "",
            "anos_detectados_nome": ", ".join(map(str, years_name)),
            "ano_min": min(years_name) if years_name else np.nan,
            "ano_max": max(years_name) if years_name else np.nan,
            "qtd_anos": 0,
            "ufs": 0,
            "municipios": 0,
            "cargos": "",
            "linhas_gold": 0,
        })

    out = pd.DataFrame(rows)
    return out.sort_values(["ano_min", "arquivo_origem"], na_position="last") if not out.empty else out


def build_file_year_matrix(inventory: pd.DataFrame) -> pd.DataFrame:
    if inventory is None or inventory.empty:
        return pd.DataFrame()

    years = sorted(set(
        y
        for col in ["anos_detectados_conteudo", "anos_detectados_nome"]
        for value in inventory.get(col, pd.Series(dtype=str)).fillna("").astype(str)
        for y in extract_years_from_value(value)
    ))

    rows = []
    for _, r in inventory.iterrows():
        present = set(extract_years_from_value(r.get("anos_detectados_conteudo", "")))
        if not present:
            present = set(extract_years_from_value(r.get("anos_detectados_nome", "")))
        row = {"arquivo_origem": r.get("arquivo_origem", "")}
        for year in years:
            row[str(year)] = int(year in present)
        rows.append(row)

    return pd.DataFrame(rows)


def build_timelines(global_gold: pd.DataFrame, municipal: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {k: pd.DataFrame() for k in ["timeline_nacional", "timeline_uf", "timeline_municipal", "timeline_entidades", "evolucao_municipal"]}
    if global_gold is None or global_gold.empty:
        return out

    df = add_derived_metrics(global_gold.copy())
    metrics = ["votos", "eleitorado", "comparecimento_estimado", "abstencao_estimado", "brancos", "nulos", "validos_estimados"]

    for c in ["ano", "uf", "cd_municipio", "nm_municipio", "cargo", "turno", "partido", "candidato"]:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].map(lambda x: safe_text(x))

    tn = df.groupby(["ano", "cargo", "turno"], dropna=False)[metrics].sum().reset_index()
    tn["pct_comparecimento"] = np.where(tn["eleitorado"] > 0, tn["comparecimento_estimado"] / tn["eleitorado"], np.nan)
    tn["pct_abstencao"] = np.where(tn["eleitorado"] > 0, tn["abstencao_estimado"] / tn["eleitorado"], np.nan)

    tuf = df.groupby(["ano", "uf", "cargo", "turno"], dropna=False)[metrics].sum().reset_index()
    tuf["pct_comparecimento"] = np.where(tuf["eleitorado"] > 0, tuf["comparecimento_estimado"] / tuf["eleitorado"], np.nan)
    tuf["pct_abstencao"] = np.where(tuf["eleitorado"] > 0, tuf["abstencao_estimado"] / tuf["eleitorado"], np.nan)

    tm = municipal.copy() if municipal is not None and not municipal.empty else consolidate_municipal(df)

    entity_col = "partido" if df["partido"].astype(str).str.strip().ne("").any() else "candidato"
    te = df.groupby(["ano", "cargo", "turno", entity_col], dropna=False)["votos"].sum().reset_index()
    total = te.groupby(["ano", "cargo", "turno"], dropna=False)["votos"].transform("sum")
    te["share"] = np.where(total > 0, te["votos"] / total, np.nan)
    te = te.rename(columns={entity_col: "entidade"})

    out["timeline_nacional"] = tn
    out["timeline_uf"] = tuf
    out["timeline_municipal"] = tm
    out["timeline_entidades"] = te
    out["evolucao_municipal"] = tm
    return out


def temporal_correlations(timeline_municipal: pd.DataFrame) -> pd.DataFrame:
    if timeline_municipal is None or timeline_municipal.empty:
        return pd.DataFrame()

    df = timeline_municipal.copy()
    df["ano_num"] = pd.to_numeric(df.get("ano", np.nan), errors="coerce")
    df = df.loc[df["ano_num"].notna()].copy()

    keys = ["uf", "cd_municipio", "cargo", "turno"]
    for c in keys:
        if c not in df.columns:
            df[c] = ""

    metrics = [
        c for c in [
            "votos", "eleitorado", "comparecimento_estimado", "abstencao_estimado",
            "pct_comparecimento", "pct_abstencao", "competitividade", "margem_lider"
        ]
        if c in df.columns
    ]

    years = sorted(df["ano_num"].dropna().astype(int).unique().tolist())
    records = []

    for metric in metrics:
        sub = df[keys + ["ano_num", metric]].copy()
        sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
        wide = sub.pivot_table(index=keys, columns="ano_num", values=metric, aggfunc="mean")

        for i, y1 in enumerate(years):
            for y2 in years[i + 1:]:
                if y1 not in wide.columns or y2 not in wide.columns:
                    continue
                pair = wide[[y1, y2]].dropna()
                if len(pair) < 5:
                    continue
                records.append({
                    "tipo": "mesma_metrica_entre_anos",
                    "metrica": metric,
                    "ano_1": int(y1),
                    "ano_2": int(y2),
                    "n": int(len(pair)),
                    "pearson": pair[y1].corr(pair[y2], method="pearson"),
                    "spearman": pair[y1].corr(pair[y2], method="spearman"),
                })

    out = pd.DataFrame(records)
    if not out.empty:
        out["abs_pearson"] = out["pearson"].abs()
        out = out.sort_values(["abs_pearson", "n"], ascending=[False, False])
    return out


def entity_share_correlations(timeline_entidades: pd.DataFrame) -> pd.DataFrame:
    if timeline_entidades is None or timeline_entidades.empty:
        return pd.DataFrame()

    df = timeline_entidades.copy()
    df["ano_num"] = pd.to_numeric(df.get("ano", np.nan), errors="coerce")
    df["share"] = pd.to_numeric(df.get("share", np.nan), errors="coerce")
    years = sorted(df["ano_num"].dropna().astype(int).unique().tolist())

    if len(years) < 2:
        return pd.DataFrame()

    wide = df.pivot_table(index=["cargo", "turno", "entidade"], columns="ano_num", values="share", aggfunc="mean")
    records = []
    for i, y1 in enumerate(years):
        for y2 in years[i+1:]:
            if y1 not in wide.columns or y2 not in wide.columns:
                continue
            pair = wide[[y1, y2]].dropna()
            if len(pair) < 3:
                continue
            records.append({
                "ano_1": int(y1),
                "ano_2": int(y2),
                "n_entidades": int(len(pair)),
                "pearson_share": pair[y1].corr(pair[y2], method="pearson"),
                "spearman_share": pair[y1].corr(pair[y2], method="spearman"),
            })

    out = pd.DataFrame(records)
    if not out.empty:
        out["abs_pearson_share"] = out["pearson_share"].abs()
        out = out.sort_values(["abs_pearson_share", "n_entidades"], ascending=[False, False])
    return out
