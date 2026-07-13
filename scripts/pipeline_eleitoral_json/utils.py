from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import ctypes
import gc
import hashlib
import html
import json
import logging
import re

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_OK = True
except Exception:
    plt = None
    MATPLOTLIB_OK = False

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_OK = True
except Exception:
    PYARROW_OK = False


def setup_logging(out: Path, level: str) -> Path:
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / "pipeline_eleitoral_json.log"

    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return log_path


def clean_memory() -> None:
    if plt is not None:
        plt.close("all")
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def safe_text(value: Any, fallback: str = "") -> str:
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null", "<na>"}:
        return fallback
    return s


def normalize_col(col: Any) -> str:
    s = str(col).replace("\ufeff", "").strip().upper()
    s = re.sub(r"\s+", "_", s)
    return s


def compact_code(value: Any) -> str:
    s = safe_text(value, "")
    if s.endswith(".0") and s.replace(".0", "").isdigit():
        return s[:-2]
    return s


def safe_name(value: Any, limit: int = 140) -> str:
    s = str(value)
    s = s.replace("\\", "__").replace("/", "__")
    s = re.sub(r'[<>:"|?*\n\r\t]', "_", s)
    s = re.sub(r"\s+", "_", s).strip("._-")
    return (s or "arquivo")[:limit]


def hash_short(value: Any, n: int = 12) -> str:
    return hashlib.sha1(str(value).encode("utf-8", errors="ignore")).hexdigest()[:n]


def parse_number(value: Any):
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return np.nan if pd.isna(value) else float(value)
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null", "na", "n/a"}:
        return np.nan
    s = s.replace("\u00a0", "").replace(" ", "")

    # Formato brasileiro: 1.234,56
    if re.fullmatch(r"[-+]?\d{1,3}(\.\d{3})+,\d+", s):
        s = s.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"[-+]?\d+,\d+", s):
        s = s.replace(",", ".")
    elif re.fullmatch(r"[-+]?\d{1,3}(\.\d{3})+", s):
        s = s.replace(".", "")

    try:
        return float(s)
    except Exception:
        return np.nan


def extract_years_from_value(value: Any) -> list[int]:
    years = []
    for token in re.findall(r"(?:19|20)\d{2}", str(value)):
        try:
            y = int(token)
            if 1900 <= y <= 2100:
                years.append(y)
        except Exception:
            pass
    return sorted(set(years))


def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    seen = {}
    cols = []
    for c in df.columns:
        n = normalize_col(c)
        seen[n] = seen.get(n, 0) + 1
        if seen[n] > 1:
            n = f"{n}_{seen[n]}"
        cols.append(n)
    df.columns = cols
    return df


def save_csv(df: pd.DataFrame | None, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df is None:
        df = pd.DataFrame()
    if df.empty and len(df.columns) == 0:
        df = pd.DataFrame(columns=["status"])
    df.to_csv(path, sep=";", index=index, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def save_parquet(df: pd.DataFrame | None, path: Path) -> bool:
    if df is None or df.empty or not PYARROW_OK:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, path, compression="snappy")
        return True
    except Exception as exc:
        logging.warning("Falha ao salvar parquet %s: %s", path, exc)
        return False


def df_to_html(df: pd.DataFrame | None, max_rows: int = 200) -> str:
    if df is None or df.empty:
        return "<p><em>Sem dados.</em></p>"
    return df.head(max_rows).to_html(index=False, escape=True)


def img_tag(path: Path, base: Path) -> str:
    try:
        src = path.relative_to(base).as_posix()
    except Exception:
        src = path.as_posix()
    return f'<div class="figure"><img src="{html.escape(src)}"><div class="small">{html.escape(path.name)}</div></div>'


def save_html(path: Path, title: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = f"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body {{ font-family: Arial, Helvetica, sans-serif; margin: 32px; line-height: 1.45; color: #222; }}
h1, h2, h3 {{ color: #111; }}
table {{ border-collapse: collapse; width: 100%; font-size: 12px; margin: 12px 0 24px 0; }}
th, td {{ border: 1px solid #ddd; padding: 6px; vertical-align: top; }}
th {{ background: #f2f2f2; }}
pre {{ white-space: pre-wrap; background: #fafafa; border: 1px solid #ddd; padding: 12px; }}
.figure {{ margin: 22px 0; }}
.figure img {{ max-width: 100%; border: 1px solid #ddd; }}
.small {{ font-size: 12px; color: #555; }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
{body}
</body>
</html>"""
    path.write_text(doc, encoding="utf-8")
