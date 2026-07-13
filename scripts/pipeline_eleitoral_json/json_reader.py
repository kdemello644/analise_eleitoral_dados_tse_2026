from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator
import gzip
import json
import logging
import random

import pandas as pd

try:
    import ijson
    IJSON_OK = True
except Exception:
    IJSON_OK = False

from .utils import normalize_col, normalize_dataframe_columns


SUPPORTED_JSON_EXT = {".json", ".jsonl", ".ndjson"}
MAX_JSONL_LINE_CHARS = 64 * 1024 * 1024
METADATA_JSON_NAMES = {
    "manifesto_processamento.json",
    "resumo_processamento_json.json",
    "resumo_global.json",
    "resumo_global_amostral.json",
}


def is_metadata_json(path: Path) -> bool:
    name = path.name.lower()
    return name in METADATA_JSON_NAMES or name.startswith("manifesto_") or name.startswith("resumo_")


def classify_json_document(path: Path) -> dict[str, str]:
    text = path.as_posix().lower()
    if (
        "canidatos" in text
        or "candidatos" in text
        or "consulta_cand" in text
        or "bem_candidato" in text
        or "consulta_coligacao" in text
        or "consulta_vagas" in text
        or "motivo_cassacao" in text
    ):
        domain = "candidatos"
    elif (
        "eleitorado" in text
        or "perfil_eleitor" in text
        or "transferencia_temporaria" in text
        or "transf_temporaria" in text
    ):
        domain = "eleitorado"
    elif "resultados" in text or "votacao" in text or "totalizacao" in text:
        domain = "resultados"
    elif is_metadata_json(path):
        domain = "metadados_processamento"
    else:
        domain = "outros"

    if "perfil_eleitor_secao" in text:
        subject = "perfil_eleitor_secao"
    elif "perfil_eleitorado" in text:
        subject = "perfil_eleitorado"
    elif "votacao_secao" in text:
        subject = "votacao_secao"
    elif "votacao_candidato" in text:
        subject = "votacao_candidato"
    elif "votacao_partido" in text:
        subject = "votacao_partido"
    elif "consulta_cand" in text:
        subject = "consulta_candidato"
    elif "bem_candidato" in text:
        subject = "bens_candidato"
    elif "coligacao" in text:
        subject = "coligacao"
    elif "consulta_vagas" in text:
        subject = "consulta_vagas"
    elif "motivo_cassacao" in text:
        subject = "motivo_cassacao"
    elif "transf_temporaria" in text or "transferencia_temporaria" in text:
        subject = "transferencia_temporaria"
    else:
        subject = path.stem

    return {
        "dominio_documento": domain,
        "assunto_documento": subject,
        "tipo_arquivo_json": "metadado" if is_metadata_json(path) else "dados",
    }


def find_json_files(root: Path, include_metadata: bool = False) -> list[Path]:
    files = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.name.startswith("_"):
            continue
        if p.suffix.lower() in SUPPORTED_JSON_EXT:
            if not include_metadata and (p.parent.name.upper() != "JSON" or is_metadata_json(p)):
                continue
            files.append(p)
    return sorted(files)


def open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def normalize_record(rec: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in rec.items():
        out[normalize_col(k)] = v
    return out


def first_non_ws_char(path: Path) -> str:
    try:
        with open_text(path) as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    return ""
                stripped = chunk.lstrip()
                if stripped:
                    return stripped[0]
    except Exception:
        return ""


def iter_json_records(path: Path) -> Iterator[dict[str, Any]]:
    suffix = path.suffix.lower()

    if suffix in {".jsonl", ".ndjson"}:
        if first_non_ws_char(path) == "[" and IJSON_OK:
            try:
                with open_text(path) as f:
                    for obj in ijson.items(f, "item"):
                        if isinstance(obj, dict):
                            yield normalize_record(obj)
                return
            except Exception as exc:
                logging.warning("Falha lendo %s como array JSON streaming: %s. Tentando JSONL seguro.", path, exc)

        with open_text(path) as f:
            while True:
                line = f.readline(MAX_JSONL_LINE_CHARS + 1)
                if not line:
                    break
                if len(line) > MAX_JSONL_LINE_CHARS and not line.endswith("\n"):
                    logging.warning(
                        "Linha JSONL maior que %s MiB em %s. Arquivo parece nao ser JSONL real; instale/ative ijson ou renomeie como JSON array.",
                        MAX_JSONL_LINE_CHARS // (1024 * 1024),
                        path,
                    )
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        yield normalize_record(obj)
                except Exception:
                    continue
        return

    if IJSON_OK:
        try:
            with open_text(path) as f:
                for obj in ijson.items(f, "item"):
                    if isinstance(obj, dict):
                        yield normalize_record(obj)
            return
        except Exception:
            pass

    try:
        with open_text(path) as f:
            data = json.load(f)
    except Exception as exc:
        logging.warning("Falha lendo JSON %s: %s", path, exc)
        return

    if isinstance(data, list):
        for obj in data:
            if isinstance(obj, dict):
                yield normalize_record(obj)
    elif isinstance(data, dict):
        yielded = False
        for value in data.values():
            if isinstance(value, list):
                for obj in value:
                    if isinstance(obj, dict):
                        yielded = True
                        yield normalize_record(obj)
        if not yielded:
            yield normalize_record(data)


def sample_json_file(path: Path, cfg) -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = random.Random(cfg.seed)
    seen = 0
    rows = []

    if cfg.sample_mode == "head":
        for rec in iter_json_records(path):
            seen += 1
            rows.append(rec)
            if len(rows) >= cfg.max_sample_rows:
                break
        df = normalize_dataframe_columns(pd.DataFrame(rows)) if rows else pd.DataFrame()
        return df, {"sample_mode": "head", "linhas_lidas": seen, "linhas_amostra": len(df)}

    # Reservoir data-driven. Não depende de UF, município ou qualquer schema prévio.
    target = max(cfg.min_sample_rows, cfg.max_sample_rows)
    for rec in iter_json_records(path):
        seen += 1
        keep = True
        if cfg.sample_frac < 1.0 and len(rows) >= cfg.min_sample_rows:
            keep = rng.random() <= cfg.sample_frac

        if not keep:
            continue

        if len(rows) < target:
            rows.append(rec)
        else:
            j = rng.randint(0, seen - 1)
            if j < target:
                rows[j] = rec

    if len(rows) > cfg.max_sample_rows:
        rng.shuffle(rows)
        rows = rows[: cfg.max_sample_rows]

    df = normalize_dataframe_columns(pd.DataFrame(rows)) if rows else pd.DataFrame()
    return df, {
        "sample_mode": "reservoir",
        "linhas_lidas": seen,
        "linhas_amostra": len(df),
        "sample_frac": cfg.sample_frac,
    }
