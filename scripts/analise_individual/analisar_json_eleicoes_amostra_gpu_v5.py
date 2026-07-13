#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analisar_json_eleicoes_amostra_gpu_v5.py

Versão segura para computadores locais/WSL que estavam travando ou reiniciando.

Atualização v5:
- Spearman/Kendall continuam sempre amostrais.
- Plots podem usar 100% das linhas via agregação streaming, sem carregar tudo em memória.
- Após cada plot e cada arquivo há limpeza explícita de matplotlib, gc, CuPy/Torch CUDA quando disponível.

Objetivo:
- Processar todos os arquivos encontrados em pastas chamadas JSON.
- NÃO importar o JSON inteiro para DuckDB.
- NÃO carregar o arquivo inteiro em memória.
- Ler registros em streaming e manter apenas uma amostra controlada.
- Identificar campos-código, campos-descritivos e campos quantitativos.
- Ignorar códigos nas análises pesadas, correlações, eta², Cramér's V e plots.
- Fazer correlação em ~5% das linhas, com limite máximo configurável.
- Gerar gráficos, tabelas CSV, amostras salvas e relatório HTML.
- Salvar Parquet analítico sem códigos para reuso eficiente.
- Usar GPU opcionalmente apenas para a matriz de correlação Pearson.

Dependências mínimas:
    pip install pandas numpy matplotlib

Recomendadas:
    pip install ijson pyarrow scipy

GPU opcional:
    pip install cupy-cuda12x
    # ou torch com CUDA, se já estiver instalado

Exemplo seguro no WSL:
    python3 scripts/analise_individual/analisar_json_eleicoes_amostra_gpu_v5.py ./dados \
      --out analise_individual_v2 \
      --sample-frac 0.05 \
      --max-sample-rows 300000 \
      --scan-mode reservoir \
      --gpu auto \
      --top-n-plots 15 \
      --max-corr-cols 40

Para emergência, lendo só o começo de cada arquivo:
    python3 scripts/analise_individual/analisar_json_eleicoes_amostra_gpu_v5.py ./dados \
      --out analise_individual_v2 \
      --scan-mode head \
      --max-sample-rows 200000 \
      --gpu off
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import html
import json
import logging
import math
import os
import random
import re
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


IGNORAR_ARQUIVOS = {
    "manifesto_processamento.json",
    "resumo_processamento_json.json",
    "resumo_global.json",
}

KEY_LIKE_COLUMNS = {
    "ANO_ELEICAO", "CD_ELEICAO", "NR_TURNO", "SG_UF", "CD_MUNICIPIO",
    "NM_MUNICIPIO", "NR_ZONA", "NR_SECAO", "SQ_CANDIDATO", "NR_CANDIDATO",
    "SG_PARTIDO", "CD_CARGO", "DS_CARGO", "CD_TIPO_ELEICAO", "CD_PLEITO",
    "SQ_COLIGACAO", "SQ_PARTIDO", "NR_PARTIDO", "NR_URNA_EFETIVADA",
}

# Campos quantitativos: entram em estatísticas, correlação e plots.
# Em dados eleitorais do TSE, campos realmente analíticos geralmente começam por QT/VL/VR/PC/PCT
# ou contêm termos como voto, eleitorado, comparecimento e abstenção.
QUANT_PREFIXES = ("QT_", "QTD_", "VR_", "VL_", "PCT_", "PC_", "TX_")
QUANT_KEYWORDS = (
    "VOTO", "VOTOS", "VOTACAO", "VOTAÇÃO", "COMPARECIMENTO", "ABSTEN", "ELEITORADO",
    "ELEITORES", "APTOS", "VALIDOS", "VÁLIDOS", "NOMINAIS", "LEGENDA", "BRANCOS",
    "NULOS", "ANULADOS", "TOTAL", "PERCENTUAL", "PORCENTAGEM", "QTD", "QUANTIDADE",
)

# Campos-código: ficam documentados no perfil, mas são ignorados nos cálculos pesados.
# Ex.: CD_MUNICIPIO, NR_SECAO, SQ_CANDIDATO, SG_UF, NR_TURNO etc.
CODE_PREFIXES = ("CD_", "SQ_", "ID_", "SG_")
CODE_NUMERIC_PREFIXES = ("NR_", "HH_", "MM_")
TEXT_DIM_PREFIXES = ("NM_", "DS_")
DATE_PREFIXES = ("DT_", "DATA_")


@dataclass
class Config:
    root: Path
    out_dir: Path
    sample_frac: float
    max_sample_rows: int
    min_sample_rows: int
    sample_seed: int
    scan_mode: str
    max_columns: int
    max_corr_cols: int
    max_categories: int
    top_n_plots: int
    top_n_html: int
    gpu: str
    gpu_max_mb: int
    gerar_spearman: bool
    gerar_kendall: bool
    gerar_scatter: bool
    plots_full: bool
    full_plot_bins: int
    full_density_pairs: int
    salvar_parquet: bool
    salvar_amostra_csv: bool
    salvar_parquet_completo_analitico: bool
    parquet_chunk_rows: int
    json_array_backend: str
    flatten_depth: int
    resume: bool


def safe_name(value: Any, limit: int = 160) -> str:
    s = str(value)
    s = s.replace("\\", "__").replace("/", "__")
    s = re.sub(r'[<>:"|?*\n\r\t]', "_", s)
    s = re.sub(r"\s+", "_", s)
    s = s.strip("._-") or "arquivo"
    return s[:limit]


def hash_curto(texto: str) -> str:
    return hashlib.sha1(texto.encode("utf-8", errors="ignore")).hexdigest()[:12]


def normalizar_coluna(col: Any) -> str:
    return str(col).strip().replace("\ufeff", "").upper()


def parece_quantidade_ou_valor(col: Any) -> bool:
    """Detecta campos que representam medida/quantia, não identificador."""
    c = normalizar_coluna(col)
    return c.startswith(QUANT_PREFIXES) or any(p in c for p in QUANT_KEYWORDS)


def parece_codigo(col: Any) -> bool:
    """Detecta códigos/dimensões técnicas que não devem entrar em correlação/estatística pesada."""
    c = normalizar_coluna(col)
    if parece_quantidade_ou_valor(c):
        return False
    # NM_/DS_ são dimensões textuais/descritivas, não códigos. Podem ser usadas em frequências leves.
    if c.startswith(TEXT_DIM_PREFIXES):
        return False
    return (
        c in KEY_LIKE_COLUMNS
        or c.startswith(CODE_PREFIXES)
        or c.startswith(CODE_NUMERIC_PREFIXES)
    )


def parece_chave(col: Any) -> bool:
    # Mantido por compatibilidade com o resto do script.
    return parece_codigo(col)


def parece_data(col: Any) -> bool:
    c = normalizar_coluna(col)
    return c.startswith(DATE_PREFIXES) or c in {"DT", "DATA", "DATE"} or "DATA" in c


def parece_descricao_nome(col: Any) -> bool:
    c = normalizar_coluna(col)
    return c.startswith(TEXT_DIM_PREFIXES)


def configurar_logging(out_dir: Path) -> Path:
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "analise_amostra_gpu.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return log_path


def cleanup_runtime(label: str = "") -> None:
    """Libera memória/cache depois de cada plot e entre arquivos.

    A função é defensiva: se Torch/CuPy não estiverem instalados, simplesmente ignora.
    """
    try:
        plt.close("all")
    except Exception:
        pass

    try:
        gc.collect()
    except Exception:
        pass

    # Limpeza opcional de cache CUDA via Torch.
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass

    # Limpeza opcional de cache CUDA via CuPy.
    try:
        import cupy as cp  # type: ignore
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass

    if label:
        logging.debug("Memória/cache limpos: %s", label)


def encontrar_jsons(root: Path) -> List[Path]:
    arquivos: List[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.parent.name.upper() != "JSON":
            continue
        if p.name.lower() in IGNORAR_ARQUIVOS:
            continue
        if p.name.startswith("_"):
            continue
        if p.suffix.lower() not in {".json", ".jsonl", ".ndjson"}:
            continue
        arquivos.append(p)
    return sorted(arquivos)


def flatten_record(obj: Any, max_depth: int = 2, prefix: str = "", depth: int = 0) -> Dict[str, Any]:
    """Achata dicts pequenos. Listas/dicts muito profundos viram JSON compacto."""
    out: Dict[str, Any] = {}
    if not isinstance(obj, dict):
        return {"valor": obj}

    for k, v in obj.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict) and depth < max_depth:
            out.update(flatten_record(v, max_depth=max_depth, prefix=key, depth=depth + 1))
        elif isinstance(v, (list, dict)):
            try:
                out[key] = json.dumps(v, ensure_ascii=False, separators=(",", ":"))
            except Exception:
                out[key] = str(v)
        else:
            out[key] = v
    return out


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                logging.warning("Linha JSONL inválida em %s:%s: %s", path, line_no, exc)
                continue
            if isinstance(obj, dict):
                yield obj
            else:
                yield {"valor": obj}


def iter_json_array_ijson(path: Path) -> Iterator[Dict[str, Any]]:
    try:
        import ijson  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Para JSON grande em formato array, instale ijson: python3 -m pip install ijson"
        ) from exc

    with path.open("rb") as f:
        for obj in ijson.items(f, "item"):
            if isinstance(obj, dict):
                yield obj
            else:
                yield {"valor": obj}


def iter_json_array_stdlib(path: Path, chunk_size: int = 1024 * 1024, max_buffer: int = 512 * 1024 * 1024) -> Iterator[Dict[str, Any]]:
    """
    Parser streaming simples para array JSON top-level usando stdlib.
    Evita json.load(). Se o arquivo for complexo demais, prefira ijson.
    """
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8", errors="replace") as f:
        buf = ""
        pos = 0
        started = False
        eof = False

        def ensure_data() -> bool:
            nonlocal buf, pos, eof
            while pos >= len(buf) and not eof:
                chunk = f.read(chunk_size)
                if not chunk:
                    eof = True
                    return False
                buf = chunk
                pos = 0
            return pos < len(buf)

        while True:
            if not ensure_data():
                return

            # Mantém o buffer pequeno quando possível.
            if pos > chunk_size and pos > len(buf) // 2:
                buf = buf[pos:]
                pos = 0

            while True:
                while pos < len(buf) and buf[pos].isspace():
                    pos += 1
                if pos < len(buf):
                    break
                chunk = f.read(chunk_size)
                if not chunk:
                    return
                buf = ""
                pos = 0
                buf += chunk

            if not started:
                if buf[pos] != "[":
                    raise RuntimeError(
                        f"{path} não parece ser array JSON top-level. Use JSONL/NDJSON ou ijson se houver estrutura diferente."
                    )
                started = True
                pos += 1
                continue

            # separadores
            while True:
                while pos < len(buf) and buf[pos].isspace():
                    pos += 1
                if pos >= len(buf):
                    chunk = f.read(chunk_size)
                    if not chunk:
                        return
                    buf = buf[pos:] + chunk
                    pos = 0
                    continue
                if buf[pos] == ",":
                    pos += 1
                    continue
                if buf[pos] == "]":
                    return
                break

            # tenta decodificar; se faltar conteúdo, lê mais
            while True:
                try:
                    obj, end = decoder.raw_decode(buf, pos)
                    pos = end
                    if isinstance(obj, dict):
                        yield obj
                    else:
                        yield {"valor": obj}
                    break
                except json.JSONDecodeError:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        raise
                    # preserva somente a parte ainda não processada
                    if pos > 0:
                        buf = buf[pos:] + chunk
                        pos = 0
                    else:
                        buf += chunk
                    if len(buf) > max_buffer:
                        raise RuntimeError(
                            f"Objeto JSON muito grande ou parser sem progresso em {path}. Instale ijson."
                        )


def iter_records(path: Path, cfg: Config) -> Iterator[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        yield from iter_jsonl(path)
        return

    if cfg.json_array_backend in {"auto", "ijson"}:
        try:
            yield from iter_json_array_ijson(path)
            return
        except Exception as exc:
            if cfg.json_array_backend == "ijson":
                raise
            logging.warning("ijson não disponível/falhou para %s. Tentando parser stdlib. Motivo: %s", path, exc)

    yield from iter_json_array_stdlib(path)


def streaming_sample(path: Path, cfg: Config) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Retorna amostra em memória limitada.

    scan_mode=reservoir:
      - percorre o arquivo inteiro, mas mantém no máximo max_sample_rows.
      - estatisticamente melhor e não explode memória.

    scan_mode=head:
      - lê só o início até max_sample_rows.
      - mais rápido, mas pode enviesar se o arquivo vier ordenado.
    """
    rng = random.Random(cfg.sample_seed + int(hash_curto(str(path)), 16) % 10_000_000)
    rows: List[Dict[str, Any]] = []
    total = 0
    accepted_by_frac = 0
    sample_cap = max(1, int(cfg.max_sample_rows))
    frac = min(max(float(cfg.sample_frac), 0.0), 1.0)

    for rec in iter_records(path, cfg):
        total += 1
        flat = flatten_record(rec, max_depth=cfg.flatten_depth)

        if cfg.scan_mode == "head":
            rows.append(flat)
            if len(rows) >= sample_cap:
                break
            continue

        # reservoir com tentativa de respeitar frac, mas com cap de memória.
        if rng.random() <= frac:
            accepted_by_frac += 1
            if len(rows) < sample_cap:
                rows.append(flat)
            else:
                # Depois de bater o cap, mantém uma amostra aleatória dos aceitos.
                j = rng.randint(0, accepted_by_frac - 1)
                if j < sample_cap:
                    rows[j] = flat

        # Garante amostra mínima no começo, útil se frac for pequeno e arquivo também.
        elif len(rows) < cfg.min_sample_rows:
            rows.append(flat)

    meta = {
        "arquivo": str(path),
        "linhas_lidas": total,
        "linhas_amostra": len(rows),
        "sample_frac_solicitada": frac,
        "max_sample_rows": sample_cap,
        "scan_mode": cfg.scan_mode,
        "observacao": (
            "reservoir percorre o arquivo inteiro com memória limitada" if cfg.scan_mode == "reservoir"
            else "head lê somente o começo do arquivo; mais rápido, porém menos representativo"
        ),
    }
    return rows, meta


def limitar_colunas(df: pd.DataFrame, max_columns: int) -> pd.DataFrame:
    if df.empty or df.shape[1] <= max_columns:
        return df

    # Prioridade: quantidades/valores e campos não-código. Códigos só entram se sobrar espaço.
    scores = []
    n = max(len(df), 1)
    for col in df.columns:
        non_null = float(df[col].notna().sum()) / n
        bonus = 0.0
        if parece_quantidade_ou_valor(col):
            bonus += 2.0
        if parece_descricao_nome(col) and not parece_codigo(col):
            bonus += 0.4
        if parece_codigo(col):
            bonus -= 1.0
        scores.append((non_null + bonus, str(col)))

    keep = [c for _, c in sorted(scores, reverse=True)[:max_columns]]
    keep_set = set(keep)
    return df[[c for c in df.columns if c in keep_set]].copy()


def to_numeric_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")

    ss = s.astype("string").str.strip()
    ss = ss.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA, "null": pd.NA, "NULL": pd.NA})

    # Trata formato brasileiro simples e milhares: 1.234,56 -> 1234.56
    has_comma = ss.str.contains(",", regex=False, na=False)
    has_dot = ss.str.contains(".", regex=False, na=False)
    both = has_comma & has_dot

    out = ss.copy()
    out.loc[both] = out.loc[both].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    out.loc[has_comma & ~has_dot] = out.loc[has_comma & ~has_dot].str.replace(",", ".", regex=False)
    return pd.to_numeric(out, errors="coerce")


def inferir_perfil_amostra(df: pd.DataFrame) -> pd.DataFrame:
    """
    Infere perfil e, principalmente, separa:
      - quantidade/valor: entra em estatística, correlação e plot.
      - codigo/codigo_provavel: fica documentado, mas não entra em análise pesada.
      - dimensao_textual/categorico/texto/data: pode entrar em frequências/resumos leves se não for código.
    """
    records: List[Dict[str, Any]] = []
    total = len(df)
    for col in df.columns:
        s = df[col]
        s_str = s.astype("string")
        non_null = s.notna() & (s_str.str.strip() != "")
        qtd_nao_nulos = int(non_null.sum())
        qtd_nulos = int(total - qtd_nao_nulos)
        nunique = int(s[non_null].astype("string").nunique(dropna=True)) if qtd_nao_nulos else 0

        sn = to_numeric_series(s)
        qtd_numericos = int(sn.notna().sum())
        ratio_num = qtd_numericos / max(qtd_nao_nulos, 1)
        ratio_unicos = nunique / max(qtd_nao_nulos, 1)

        nome_quantidade = parece_quantidade_ou_valor(col)
        nome_codigo = parece_codigo(col)
        nome_data = parece_data(col)
        nome_descricao = parece_descricao_nome(col)

        # Classe semântica decide se a coluna será processada ou ignorada.
        if qtd_nao_nulos == 0:
            tipo = "vazio"
            classe = "vazio"
            processar = False
            motivo = "coluna vazia na amostra"
        elif nome_codigo:
            # Regras eleitorais: CD_/SQ_/SG_/NR_ costumam ser identificadores/códigos.
            tipo = "codigo_numerico" if ratio_num >= 0.80 else "codigo_categorico"
            classe = "codigo"
            processar = False
            motivo = "identificada pelo nome como código/chave/dimensão técnica"
        elif nome_quantidade and ratio_num >= 0.80:
            tipo = "numerico_analitico_quantidade"
            classe = "quantidade"
            processar = True
            motivo = "campo quantitativo/valor detectado pelo nome e convertido como numérico"
        elif nome_data:
            tipo = "data_ou_temporal"
            classe = "data"
            processar = False
            motivo = "campo temporal; não entra em correlação numérica"
        elif ratio_num >= 0.95:
            # Número sem nome de quantidade, com poucos valores únicos, quase sempre é código.
            if nunique <= 50 or ratio_unicos <= 0.05:
                tipo = "codigo_provavel_numerico"
                classe = "codigo_provavel"
                processar = False
                motivo = "numérico, mas com poucos valores únicos/baixa cardinalidade e sem nome de quantia"
            else:
                tipo = "numerico_analitico"
                classe = "numerico"
                processar = True
                motivo = "numérico contínuo sem padrão claro de código"
        elif nome_descricao:
            if nunique <= 120 or ratio_unicos <= 0.10:
                tipo = "dimensao_textual"
                classe = "dimensao"
                processar = True
                motivo = "nome/descrição com cardinalidade tratável; entra só em frequência/categórico leve"
            else:
                tipo = "texto_livre"
                classe = "texto"
                processar = False
                motivo = "texto/descritivo com alta cardinalidade; não entra em análise pesada"
        else:
            if nunique <= 80 or ratio_unicos <= 0.05:
                tipo = "categorico"
                classe = "categorico"
                processar = True
                motivo = "categórico não identificado como código"
            else:
                tipo = "texto"
                classe = "texto"
                processar = False
                motivo = "texto ou alta cardinalidade; não entra em análise pesada"

        records.append({
            "coluna": col,
            "coluna_normalizada": normalizar_coluna(col),
            "tipo_inferido_amostra": tipo,
            "classe_semantica": classe,
            "processar_analise": bool(processar),
            "motivo_decisao": motivo,
            "linhas_amostra": total,
            "qtd_nao_nulos_amostra": qtd_nao_nulos,
            "qtd_nulos_amostra": qtd_nulos,
            "pct_nulos_amostra": qtd_nulos / max(total, 1),
            "qtd_unicos_amostra": nunique,
            "ratio_unicos_amostra": ratio_unicos,
            "qtd_numericos_amostra": qtd_numericos,
            "ratio_numerico_amostra": ratio_num,
            "parece_chave": nome_codigo,
            "parece_codigo": nome_codigo,
            "parece_quantidade_ou_valor": nome_quantidade,
            "parece_data": nome_data,
            "parece_descricao_nome": nome_descricao,
        })
    return pd.DataFrame(records)


def colunas_processadas(profile: pd.DataFrame) -> List[str]:
    if profile.empty or "processar_analise" not in profile.columns:
        return []
    return profile.loc[profile["processar_analise"].astype(bool), "coluna"].astype(str).tolist()


def colunas_ignoradas_codigos(profile: pd.DataFrame) -> List[str]:
    if profile.empty or "classe_semantica" not in profile.columns:
        return []
    return profile.loc[
        profile["classe_semantica"].isin(["codigo", "codigo_provavel"]),
        "coluna"
    ].astype(str).tolist()


def selecionar_numericas_analiticas(profile: pd.DataFrame, max_corr_cols: int) -> List[str]:
    if profile.empty:
        return []
    cand = profile.loc[
        profile["tipo_inferido_amostra"].isin(["numerico_analitico_quantidade", "numerico_analitico"])
        & profile["processar_analise"].astype(bool)
        & profile["qtd_numericos_amostra"].gt(2),
        :
    ].copy()

    if cand.empty:
        return []

    cand["score"] = (
        cand["parece_quantidade_ou_valor"].astype(float) * 5.0
        + cand["qtd_numericos_amostra"].astype(float) / max(float(cand["qtd_numericos_amostra"].max()), 1.0)
        + cand["qtd_unicos_amostra"].clip(upper=1000).astype(float) / 1000.0
        - cand["parece_codigo"].astype(float) * 10.0
    )
    cand = cand.sort_values("score", ascending=False)
    return cand["coluna"].astype(str).head(max_corr_cols).tolist()


def selecionar_categoricas(profile: pd.DataFrame, max_categories: int) -> List[str]:
    if profile.empty:
        return []
    cat = profile.loc[
        profile["tipo_inferido_amostra"].isin(["categorico", "dimensao_textual"])
        & profile["processar_analise"].astype(bool)
        & profile["qtd_unicos_amostra"].between(2, max_categories),
        :
    ].copy()
    if cat.empty:
        return []
    cat["score"] = (
        (1.0 / cat["qtd_unicos_amostra"].clip(lower=1).astype(float))
        + cat["qtd_nao_nulos_amostra"].astype(float) / max(float(cat["qtd_nao_nulos_amostra"].max()), 1.0)
        - cat["parece_codigo"].astype(float) * 10.0
    )
    return cat.sort_values("score", ascending=False)["coluna"].astype(str).tolist()


def preparar_matriz_numerica(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in cols:
        if col in df.columns:
            out[col] = to_numeric_series(df[col]).astype("float32")
    out = out.replace([np.inf, -np.inf], np.nan)
    # Remove colunas vazias ou constantes.
    keep = []
    for col in out.columns:
        s = out[col]
        if s.notna().sum() >= 3 and s.nunique(dropna=True) > 1:
            keep.append(col)
    return out[keep]


def corr_pearson_gpu_ou_cpu(df_num: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, str]:
    if df_num.empty or df_num.shape[1] < 2:
        return pd.DataFrame(), "sem_colunas_suficientes"

    n, p = df_num.shape
    est_mb = int((n * p * 4 + p * p * 4 * 4) / (1024 ** 2))
    use_gpu = cfg.gpu in {"auto", "on"} and est_mb <= cfg.gpu_max_mb

    # Prepara NaNs no CPU para não complicar pairwise-missing na GPU.
    X = df_num.to_numpy(dtype=np.float32, copy=True)
    col_means = np.nanmean(X, axis=0)
    bad_means = ~np.isfinite(col_means)
    if bad_means.any():
        col_means[bad_means] = 0.0
    inds = np.where(~np.isfinite(X))
    if len(inds[0]):
        X[inds] = np.take(col_means, inds[1])

    if use_gpu:
        # 1) CuPy
        try:
            import cupy as cp  # type: ignore
            if cp.cuda.runtime.getDeviceCount() > 0:
                gx = cp.asarray(X, dtype=cp.float32)
                gx = gx - gx.mean(axis=0, keepdims=True)
                denom = cp.sqrt(cp.sum(gx * gx, axis=0))
                denom = cp.where(denom == 0, cp.nan, denom)
                corr = (gx.T @ gx) / (denom[:, None] * denom[None, :])
                arr = cp.asnumpy(corr)
                del gx, corr
                cp.get_default_memory_pool().free_all_blocks()
                return pd.DataFrame(arr, index=df_num.columns, columns=df_num.columns), f"gpu_cupy_est_mb={est_mb}"
        except Exception as exc:
            if cfg.gpu == "on":
                logging.warning("GPU CuPy falhou; tentando Torch. Erro: %s", exc)

        # 2) Torch CUDA
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                device = torch.device("cuda")
                tx = torch.tensor(X, dtype=torch.float32, device=device)
                tx = tx - tx.mean(dim=0, keepdim=True)
                denom = torch.sqrt(torch.sum(tx * tx, dim=0))
                denom = torch.where(denom == 0, torch.tensor(float("nan"), device=device), denom)
                corr = (tx.T @ tx) / (denom[:, None] * denom[None, :])
                arr = corr.detach().cpu().numpy()
                del tx, corr
                torch.cuda.empty_cache()
                return pd.DataFrame(arr, index=df_num.columns, columns=df_num.columns), f"gpu_torch_est_mb={est_mb}"
        except Exception as exc:
            if cfg.gpu == "on":
                logging.warning("GPU Torch falhou; usando CPU. Erro: %s", exc)

    if cfg.gpu == "on":
        logging.warning("GPU solicitada, mas indisponível ou matriz grande demais. Usando CPU.")

    corr = pd.DataFrame(X, columns=df_num.columns).corr(method="pearson", min_periods=3)
    return corr, f"cpu_pandas_est_mb={est_mb}"


def top_corr_pairs(corr: pd.DataFrame, value_col: str = "pearson") -> pd.DataFrame:
    if corr.empty or corr.shape[1] < 2:
        return pd.DataFrame(columns=["coluna_1", "coluna_2", value_col, f"abs_{value_col}"])
    records: List[Dict[str, Any]] = []
    cols = list(corr.columns)
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            val = corr.loc[c1, c2]
            if pd.isna(val):
                continue
            records.append({
                "coluna_1": c1,
                "coluna_2": c2,
                value_col: float(val),
                f"abs_{value_col}": abs(float(val)),
            })
    out = pd.DataFrame(records)
    if not out.empty:
        out = out.sort_values(f"abs_{value_col}", ascending=False)
    return out


def salvar_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, sep=";", encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)


def salvar_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def salvar_amostra(df: pd.DataFrame, base_dir: Path, cfg: Config) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if cfg.salvar_amostra_csv:
        csv_path = base_dir / "amostra_linhas.csv.gz"
        df.to_csv(csv_path, index=False, sep=";", encoding="utf-8-sig", compression="gzip")
        out["amostra_csv_gz"] = str(csv_path)

    if cfg.salvar_parquet:
        pq_path = base_dir / "amostra_linhas.parquet"
        try:
            df.to_parquet(pq_path, index=False)
            out["amostra_parquet"] = str(pq_path)
        except Exception as exc:
            logging.warning("Não consegui salvar Parquet em %s. Instale pyarrow. Erro: %s", pq_path, exc)
    return out




def montar_df_analitico(df: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    """Retorna somente colunas que podem ser processadas; códigos ficam fora."""
    cols = [c for c in colunas_processadas(profile) if c in df.columns]
    return df[cols].copy() if cols else pd.DataFrame(index=df.index)


def salvar_amostra_analitica(df: pd.DataFrame, profile: pd.DataFrame, base_dir: Path, cfg: Config) -> Dict[str, str]:
    """Salva uma versão da amostra sem códigos, otimizada para análise futura."""
    out: Dict[str, str] = {}
    df_analitico = montar_df_analitico(df, profile)
    if df_analitico.empty:
        return out

    if cfg.salvar_parquet:
        pq_path = base_dir / "amostra_analitica_sem_codigos.parquet"
        try:
            df_analitico.to_parquet(pq_path, index=False)
            out["amostra_analitica_parquet"] = str(pq_path)
        except Exception as exc:
            logging.warning("Não consegui salvar Parquet analítico em %s. Instale pyarrow. Erro: %s", pq_path, exc)

    if cfg.salvar_amostra_csv:
        csv_path = base_dir / "amostra_analitica_sem_codigos.csv.gz"
        df_analitico.to_csv(csv_path, index=False, sep=";", encoding="utf-8-sig", compression="gzip")
        out["amostra_analitica_csv_gz"] = str(csv_path)
    return out


def _perfil_por_coluna(profile: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    if profile.empty:
        return {}
    return {str(row["coluna"]): dict(row) for _, row in profile.iterrows()}


def preparar_chunk_parquet_analitico(rows: List[Dict[str, Any]], profile: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Prepara chunk com schema estável: quantidades numéricas; demais colunas analíticas como string."""
    cols = colunas_processadas(profile)
    info = _perfil_por_coluna(profile)
    if not cols:
        return pd.DataFrame()

    chunk = pd.DataFrame(rows)
    for col in cols:
        if col not in chunk.columns:
            chunk[col] = pd.NA
    chunk = chunk[cols]

    for col in cols:
        tipo = str(info.get(col, {}).get("tipo_inferido_amostra", ""))
        if tipo in {"numerico_analitico_quantidade", "numerico_analitico"}:
            chunk[col] = to_numeric_series(chunk[col]).astype("float64")
        else:
            chunk[col] = chunk[col].astype("string")
    return chunk


def converter_parquet_completo_analitico(path: Path, profile: pd.DataFrame, base_dir: Path, cfg: Config) -> Optional[Path]:
    """
    Converte o arquivo inteiro para Parquet analítico SEM códigos, em chunks.
    Faz uma segunda leitura do JSON, mas com memória limitada.
    Use quando quiser reprocessar depois sem reler JSON bruto.
    """
    cols = colunas_processadas(profile)
    if not cols:
        logging.info("Sem colunas analíticas para Parquet completo: %s", path)
        return None

    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except Exception as exc:
        logging.warning("Parquet completo analítico exige pyarrow. Instale: python3 -m pip install pyarrow. Erro: %s", exc)
        return None

    parquet_dir = base_dir / "parquet_analitico"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    pq_path = parquet_dir / "dados_analiticos_sem_codigos.parquet"
    if cfg.resume and pq_path.exists():
        return pq_path

    if pq_path.exists():
        pq_path.unlink()

    writer = None
    buffer: List[Dict[str, Any]] = []
    total_lido = 0
    total_escrito = 0
    chunk_rows = max(1000, int(cfg.parquet_chunk_rows))

    try:
        for rec in iter_records(path, cfg):
            total_lido += 1
            flat = flatten_record(rec, max_depth=cfg.flatten_depth)
            buffer.append(flat)
            if len(buffer) >= chunk_rows:
                chunk_df = preparar_chunk_parquet_analitico(buffer, profile, cfg)
                buffer.clear()
                if chunk_df.empty:
                    continue
                table = pa.Table.from_pandas(chunk_df, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(pq_path, table.schema, compression="zstd")
                else:
                    # Garante schema igual ao primeiro chunk.
                    table = table.cast(writer.schema)
                writer.write_table(table)
                total_escrito += len(chunk_df)
                del chunk_df, table
                cleanup_runtime(f"parquet_chunk:{path.name}:{total_escrito}")
                del chunk_df, table
                gc.collect()

        if buffer:
            chunk_df = preparar_chunk_parquet_analitico(buffer, profile, cfg)
            buffer.clear()
            if not chunk_df.empty:
                table = pa.Table.from_pandas(chunk_df, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(pq_path, table.schema, compression="zstd")
                else:
                    table = table.cast(writer.schema)
                writer.write_table(table)
                total_escrito += len(chunk_df)
                del chunk_df, table
                cleanup_runtime(f"parquet_chunk:{path.name}:{total_escrito}")
                del chunk_df, table

    finally:
        if writer is not None:
            writer.close()

    meta = {
        "arquivo_origem": str(path),
        "parquet_analitico": str(pq_path),
        "linhas_lidas": total_lido,
        "linhas_escritas": total_escrito,
        "colunas_analiticas": len(cols),
        "codigos_excluidos": len(colunas_ignoradas_codigos(profile)),
        "observacao": "Parquet completo sem campos-código, gerado em chunks para reuso eficiente.",
    }
    salvar_json(meta, parquet_dir / "metadados_parquet_analitico.json")
    logging.info("Parquet analítico completo gerado: %s | linhas=%s", pq_path, total_escrito)
    return pq_path


def limitar_label(value: Any, limit: int = 45) -> str:
    s = str(value)
    return s if len(s) <= limit else s[:limit - 3] + "..."


def plot_heatmap(mat: pd.DataFrame, path: Path, title: str, vmin: Optional[float] = None, vmax: Optional[float] = None) -> Optional[Path]:
    if mat.empty or mat.shape[0] < 2 or mat.shape[1] < 2:
        return None
    try:
        labels_x = [limitar_label(c, 28) for c in mat.columns]
        labels_y = [limitar_label(i, 28) for i in mat.index]
        w = max(8, min(24, 5 + 0.35 * len(labels_x)))
        h = max(7, min(24, 4 + 0.35 * len(labels_y)))
        plt.figure(figsize=(w, h))
        plt.imshow(mat.values.astype(float), aspect="auto", vmin=vmin, vmax=vmax)
        plt.title(title)
        plt.colorbar()
        plt.xticks(range(len(labels_x)), labels_x, rotation=90)
        plt.yticks(range(len(labels_y)), labels_y)
        plt.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150)
        return path
    finally:
        cleanup_runtime(f"plot_heatmap:{path.name}")

def gerar_estatisticas_numericas(df_num: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for col in df_num.columns:
        s = df_num[col].dropna()
        if len(s) == 0:
            continue
        records.append({
            "coluna": col,
            "n_amostra": int(len(s)),
            "media": float(s.mean()),
            "desvio_padrao": float(s.std()) if len(s) > 1 else None,
            "min": float(s.min()),
            "p01": float(s.quantile(0.01)),
            "p05": float(s.quantile(0.05)),
            "p25": float(s.quantile(0.25)),
            "mediana": float(s.quantile(0.50)),
            "p75": float(s.quantile(0.75)),
            "p95": float(s.quantile(0.95)),
            "p99": float(s.quantile(0.99)),
            "max": float(s.max()),
            "soma_amostra": float(s.sum()),
        })
    return pd.DataFrame(records)


def _salvar_plot(path: Path, label: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    cleanup_runtime(label)
    return path


def parse_float_light(value: Any) -> Optional[float]:
    """Conversão leve para streaming: aceita número, decimal brasileiro e texto simples."""
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        v = float(value)
        return v if math.isfinite(v) else None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null", "na", "n/a"}:
        return None
    # Formato brasileiro: 1.234,56 -> 1234.56; 12,3 -> 12.3
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _bin_index(v: float, mn: float, mx: float, bins: int) -> Optional[int]:
    if not math.isfinite(v) or not math.isfinite(mn) or not math.isfinite(mx) or mx <= mn:
        return None
    if v < mn or v > mx:
        # Como mn/mx vêm da primeira passada em 100% das linhas, isso quase nunca ocorre.
        return None
    if v == mx:
        return bins - 1
    idx = int((v - mn) / (mx - mn) * bins)
    if idx < 0 or idx >= bins:
        return None
    return idx


def gerar_graficos(
    df: pd.DataFrame,
    df_num: pd.DataFrame,
    profile: pd.DataFrame,
    pearson: pd.DataFrame,
    top_pairs: pd.DataFrame,
    base_dir: Path,
    cfg: Config,
) -> List[Path]:
    """Gera gráficos a partir da amostra controlada."""
    plots_dir = base_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    imagens: List[Path] = []

    # Heatmap Pearson amostral
    p = plot_heatmap(pearson.fillna(0), plots_dir / "heatmap_pearson_amostra.png", "Pearson na amostra", vmin=-1, vmax=1)
    if p:
        imagens.append(p)

    # Histogramas numéricos amostrais
    numeric_priority = list(df_num.columns)[: cfg.top_n_plots]
    for col in numeric_priority:
        s = df_num[col].dropna()
        if len(s) < 3:
            continue
        try:
            plt.figure(figsize=(9, 4))
            plt.hist(s.to_numpy(), bins=50)
            plt.title(f"Histograma amostral - {col}")
            plt.xlabel(col)
            plt.ylabel("frequência na amostra")
            img = plots_dir / f"histograma_amostra_{safe_name(col, 80)}.png"
            imagens.append(_salvar_plot(img, f"histograma_amostra:{col}"))
        finally:
            cleanup_runtime(f"histograma_amostra_finally:{col}")

    # Frequências categóricas amostrais
    cat_cols = selecionar_categoricas(profile, cfg.max_categories)[: cfg.top_n_plots]
    for col in cat_cols:
        if col not in df.columns:
            continue
        freq = df[col].astype("string").fillna("(nulo)").value_counts(dropna=False).head(cfg.top_n_plots)
        if freq.empty:
            continue
        try:
            plt.figure(figsize=(10, max(4, len(freq) * 0.35)))
            plt.barh([limitar_label(x, 55) for x in freq.index.astype(str)], freq.values)
            plt.title(f"Top categorias amostrais - {col}")
            plt.xlabel("frequência na amostra")
            plt.ylabel(col)
            plt.gca().invert_yaxis()
            img = plots_dir / f"freq_amostra_{safe_name(col, 80)}.png"
            imagens.append(_salvar_plot(img, f"freq_amostra:{col}"))
        finally:
            cleanup_runtime(f"freq_amostra_finally:{col}")

    # Scatterplots amostrais dos pares mais correlacionados.
    # Mesmo no modo amostral, limita pontos para não derrubar a máquina.
    if cfg.gerar_scatter and not top_pairs.empty:
        for _, row in top_pairs.head(min(12, cfg.top_n_plots)).iterrows():
            c1, c2 = str(row["coluna_1"]), str(row["coluna_2"])
            if c1 not in df_num.columns or c2 not in df_num.columns:
                continue
            data = df_num[[c1, c2]].dropna()
            if len(data) < 10:
                continue
            if len(data) > 20000:
                data = data.sample(20000, random_state=cfg.sample_seed)
            try:
                plt.figure(figsize=(8, 5.5))
                plt.scatter(data[c1], data[c2], s=8, alpha=0.35)
                plt.title(f"Scatter amostral: {c1} x {c2}")
                plt.xlabel(c1)
                plt.ylabel(c2)
                img = plots_dir / f"scatter_amostra_{safe_name(c1, 55)}__{safe_name(c2, 55)}.png"
                imagens.append(_salvar_plot(img, f"scatter_amostra:{c1}:{c2}"))
            finally:
                cleanup_runtime(f"scatter_amostra_finally:{c1}:{c2}")

    cleanup_runtime("gerar_graficos_amostra_fim")
    return imagens


def gerar_graficos_100pct_streaming(
    path: Path,
    df_num: pd.DataFrame,
    profile: pd.DataFrame,
    pearson: pd.DataFrame,
    top_pairs: pd.DataFrame,
    base_dir: Path,
    cfg: Config,
) -> Tuple[List[Path], pd.DataFrame]:
    """Gera plots usando 100% das linhas, sem carregar todas em memória.

    Histograma e frequência são contados em streaming. Para pares numéricos, em vez de
    scatter com milhões de pontos, gera densidade 2D/heatmap de contagem, também em streaming.
    Spearman/Kendall NÃO são calculados aqui; continuam amostrais no pipeline principal.
    """
    plots_dir = base_dir / "plots_100pct_streaming"
    plots_dir.mkdir(parents=True, exist_ok=True)
    imagens: List[Path] = []
    meta_records: List[Dict[str, Any]] = []

    bins = max(10, int(cfg.full_plot_bins))
    numeric_cols = list(df_num.columns)[: cfg.top_n_plots]
    cat_cols = selecionar_categoricas(profile, cfg.max_categories)[: cfg.top_n_plots]

    pairs: List[Tuple[str, str]] = []
    if cfg.gerar_scatter and not top_pairs.empty:
        for _, row in top_pairs.head(max(0, int(cfg.full_density_pairs))).iterrows():
            c1, c2 = str(row["coluna_1"]), str(row["coluna_2"])
            if c1 in df_num.columns and c2 in df_num.columns and c1 != c2:
                pairs.append((c1, c2))

    needed_numeric = sorted(set(numeric_cols + [c for pair in pairs for c in pair]))
    if not needed_numeric and not cat_cols:
        return imagens, pd.DataFrame([{"modo_plots": "100pct_streaming", "status": "sem_colunas_para_plotar"}])

    # Passada 1: min/max numérico + frequências categóricas completas.
    num_stats: Dict[str, Dict[str, float]] = {
        col: {"n": 0.0, "min": float("inf"), "max": float("-inf")} for col in needed_numeric
    }
    cat_counts: Dict[str, Counter] = {col: Counter() for col in cat_cols}
    total_pass1 = 0
    max_counter_keys = 20000  # proteção contra classificação errada de coluna textual de alta cardinalidade

    logging.info("Plots 100%%: passada 1/2 em %s", path)
    for rec in iter_records(path, cfg):
        total_pass1 += 1
        flat = flatten_record(rec, max_depth=cfg.flatten_depth)

        for col in needed_numeric:
            v = parse_float_light(flat.get(col))
            if v is None:
                continue
            st = num_stats[col]
            st["n"] += 1.0
            if v < st["min"]:
                st["min"] = v
            if v > st["max"]:
                st["max"] = v

        for col in cat_cols:
            value = flat.get(col, "(nulo)")
            key = "(nulo)" if value is None or str(value).strip() == "" else str(value)
            counter = cat_counts[col]
            if key in counter or len(counter) < max_counter_keys:
                counter[key] += 1
            else:
                counter["(outras_categorias_limite_memoria)"] += 1

        if total_pass1 % 500000 == 0:
            cleanup_runtime(f"plots_100pct_pass1_{total_pass1}")

    # Plota frequências categóricas de 100%.
    for col, counter in cat_counts.items():
        if not counter:
            continue
        top = counter.most_common(cfg.top_n_plots)
        labels = [limitar_label(k, 55) for k, _ in top]
        vals = [v for _, v in top]
        try:
            plt.figure(figsize=(10, max(4, len(top) * 0.35)))
            plt.barh(labels, vals)
            plt.title(f"Top categorias 100% streaming - {col}")
            plt.xlabel("frequência em 100% das linhas")
            plt.ylabel(col)
            plt.gca().invert_yaxis()
            img = plots_dir / f"freq_100pct_{safe_name(col, 80)}.png"
            imagens.append(_salvar_plot(img, f"freq_100pct:{col}"))
            meta_records.append({"plot": img.name, "tipo": "frequencia", "coluna": col, "linhas_usadas": total_pass1, "modo": "100pct_streaming"})
        finally:
            cleanup_runtime(f"freq_100pct_finally:{col}")

    # Prepara contadores de histograma/densidade.
    valid_num = {
        col: st for col, st in num_stats.items()
        if st["n"] >= 3 and math.isfinite(st["min"]) and math.isfinite(st["max"]) and st["max"] > st["min"]
    }
    hist_counts: Dict[str, np.ndarray] = {col: np.zeros(bins, dtype=np.int64) for col in numeric_cols if col in valid_num}
    density_counts: Dict[Tuple[str, str], np.ndarray] = {}
    for c1, c2 in pairs:
        if c1 in valid_num and c2 in valid_num:
            density_counts[(c1, c2)] = np.zeros((bins, bins), dtype=np.int64)

    # Passada 2: histogramas e densidade 2D com 100% das linhas.
    total_pass2 = 0
    if hist_counts or density_counts:
        logging.info("Plots 100%%: passada 2/2 em %s", path)
        for rec in iter_records(path, cfg):
            total_pass2 += 1
            flat = flatten_record(rec, max_depth=cfg.flatten_depth)
            parsed: Dict[str, Optional[float]] = {}
            for col in needed_numeric:
                parsed[col] = parse_float_light(flat.get(col))

            for col, counts in hist_counts.items():
                v = parsed.get(col)
                if v is None:
                    continue
                st = valid_num[col]
                idx = _bin_index(v, st["min"], st["max"], bins)
                if idx is not None:
                    counts[idx] += 1

            for (c1, c2), mat in density_counts.items():
                x = parsed.get(c1)
                y = parsed.get(c2)
                if x is None or y is None:
                    continue
                sx = valid_num[c1]
                sy = valid_num[c2]
                ix = _bin_index(x, sx["min"], sx["max"], bins)
                iy = _bin_index(y, sy["min"], sy["max"], bins)
                if ix is not None and iy is not None:
                    mat[ix, iy] += 1

            if total_pass2 % 500000 == 0:
                cleanup_runtime(f"plots_100pct_pass2_{total_pass2}")

    # Plota histogramas de 100%.
    for col, counts in hist_counts.items():
        st = valid_num[col]
        edges = np.linspace(st["min"], st["max"], bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2
        width = (edges[1] - edges[0]) if len(edges) > 1 else 1.0
        try:
            plt.figure(figsize=(9, 4))
            plt.bar(centers, counts, width=width)
            plt.title(f"Histograma 100% streaming - {col}")
            plt.xlabel(col)
            plt.ylabel("frequência em 100% das linhas")
            img = plots_dir / f"histograma_100pct_{safe_name(col, 80)}.png"
            imagens.append(_salvar_plot(img, f"histograma_100pct:{col}"))
            meta_records.append({"plot": img.name, "tipo": "histograma", "coluna": col, "linhas_usadas": total_pass2 or total_pass1, "modo": "100pct_streaming"})
        finally:
            cleanup_runtime(f"histograma_100pct_finally:{col}")

    # Plota densidade 2D de 100% dos pares principais. É o substituto seguro do scatter com milhões de pontos.
    for (c1, c2), mat in density_counts.items():
        sx = valid_num[c1]
        sy = valid_num[c2]
        try:
            plt.figure(figsize=(8.5, 6))
            plt.imshow(
                mat.T,
                origin="lower",
                aspect="auto",
                extent=[sx["min"], sx["max"], sy["min"], sy["max"]],
            )
            plt.title(f"Densidade 2D 100% streaming: {c1} x {c2}")
            plt.xlabel(c1)
            plt.ylabel(c2)
            plt.colorbar(label="contagem de linhas")
            img = plots_dir / f"densidade2d_100pct_{safe_name(c1, 55)}__{safe_name(c2, 55)}.png"
            imagens.append(_salvar_plot(img, f"densidade2d_100pct:{c1}:{c2}"))
            meta_records.append({"plot": img.name, "tipo": "densidade2d", "coluna": f"{c1} x {c2}", "linhas_usadas": total_pass2 or total_pass1, "modo": "100pct_streaming"})
        finally:
            cleanup_runtime(f"densidade2d_100pct_finally:{c1}:{c2}")

    meta_records.append({
        "plot": "_controle",
        "tipo": "metadado",
        "coluna": "",
        "linhas_usadas": total_pass1,
        "linhas_passada_2": total_pass2,
        "modo": "100pct_streaming",
        "observacao": "Spearman/Kendall permanecem amostrais; plots 100% são agregados streaming, não scatter bruto."
    })
    cleanup_runtime("gerar_graficos_100pct_streaming_fim")
    return imagens, pd.DataFrame(meta_records)

def gerar_resumo_categorico(df: pd.DataFrame, profile: pd.DataFrame, cfg: Config, tables_dir: Path) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    freq_dir = tables_dir / "frequencias_amostra"
    freq_dir.mkdir(parents=True, exist_ok=True)
    for col in selecionar_categoricas(profile, cfg.max_categories):
        if col not in df.columns:
            continue
        freq = df[col].astype("string").fillna("(nulo)").value_counts(dropna=False).reset_index()
        freq.columns = ["valor", "frequencia_amostra"]
        salvar_csv(freq, freq_dir / f"freq_amostra_{safe_name(col, 90)}.csv")
        total = max(int(freq["frequencia_amostra"].sum()), 1)
        records.append({
            "coluna": col,
            "qtd_categorias_amostra": int(len(freq)),
            "moda_amostra": str(freq.iloc[0]["valor"]) if not freq.empty else "",
            "freq_moda_amostra": int(freq.iloc[0]["frequencia_amostra"]) if not freq.empty else 0,
            "pct_moda_amostra": float(freq.iloc[0]["frequencia_amostra"] / total) if not freq.empty else 0.0,
        })
    return pd.DataFrame(records)


def cramers_v_sample(df: pd.DataFrame, cols: List[str], max_pairs: int = 400) -> pd.DataFrame:
    try:
        from scipy import stats  # type: ignore
    except Exception:
        return pd.DataFrame(columns=["coluna_1", "coluna_2", "cramers_v_amostra", "n_amostra", "observacao"])

    records: List[Dict[str, Any]] = []
    pair_count = 0
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            pair_count += 1
            if pair_count > max_pairs:
                break
            data = df[[c1, c2]].astype("string").fillna("(nulo)")
            pivot = pd.crosstab(data[c1], data[c2])
            if pivot.shape[0] < 2 or pivot.shape[1] < 2:
                continue
            try:
                chi2, _, _, _ = stats.chi2_contingency(pivot)
                n = pivot.to_numpy().sum()
                r, k = pivot.shape
                denom = max(min(k - 1, r - 1), 1)
                v = math.sqrt((chi2 / max(n, 1)) / denom)
                records.append({
                    "coluna_1": c1,
                    "coluna_2": c2,
                    "cramers_v_amostra": float(v),
                    "n_amostra": int(n),
                    "observacao": "calculado somente na amostra para evitar varrer JSON gigante",
                })
            except Exception:
                continue
        if pair_count > max_pairs:
            break
    out = pd.DataFrame(records)
    if not out.empty:
        out["abs_cramers_v_amostra"] = out["cramers_v_amostra"].abs()
        out = out.sort_values("abs_cramers_v_amostra", ascending=False)
    return out


def eta2_sample(df: pd.DataFrame, cat_cols: List[str], num_df: pd.DataFrame, max_pairs: int = 400) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    pair_count = 0
    for cat in cat_cols:
        if cat not in df.columns:
            continue
        cats = df[cat].astype("string").fillna("(nulo)")
        for num in num_df.columns:
            pair_count += 1
            if pair_count > max_pairs:
                break
            x = num_df[num]
            data = pd.DataFrame({"cat": cats, "x": x}).dropna()
            if len(data) < 3 or data["cat"].nunique() < 2:
                continue
            grand = data["x"].mean()
            ss_total = ((data["x"] - grand) ** 2).sum()
            if ss_total <= 0:
                continue
            grouped = data.groupby("cat", observed=True)["x"].agg(["count", "mean"])
            ss_between = (grouped["count"] * ((grouped["mean"] - grand) ** 2)).sum()
            records.append({
                "coluna_categorica": cat,
                "coluna_numerica": num,
                "eta_squared_amostra": float(ss_between / ss_total),
                "n_amostra": int(len(data)),
                "qtd_categorias_amostra": int(data["cat"].nunique()),
            })
        if pair_count > max_pairs:
            break
    out = pd.DataFrame(records)
    if not out.empty:
        out = out.sort_values("eta_squared_amostra", ascending=False)
    return out


def render_html(path: Path, title: str, resumo: str, sections: List[Tuple[str, Any]], images: List[Path], max_rows: int) -> None:
    body = [f"<h1>{html.escape(title)}</h1>", "<h2>Resumo</h2>", f"<pre>{html.escape(resumo)}</pre>"]
    for name, content in sections:
        body.append(f"<h2>{html.escape(name)}</h2>")
        if isinstance(content, pd.DataFrame):
            if content.empty:
                body.append("<p><em>Sem dados.</em></p>")
            else:
                body.append(content.head(max_rows).to_html(index=False, escape=True))
        else:
            body.append(f"<pre>{html.escape(str(content))}</pre>")

    if images:
        body.append("<h2>Gráficos</h2>")
        for img in images:
            rel = img.relative_to(path.parent).as_posix() if img.is_relative_to(path.parent) else img.as_posix()
            body.append(
                f'<figure><img src="{html.escape(rel)}" alt="{html.escape(img.name)}">'
                f'<figcaption>{html.escape(img.stem)}</figcaption></figure>'
            )

    doc = f"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body {{ font-family: Arial, Helvetica, sans-serif; margin: 32px; color: #222; line-height: 1.35; }}
table {{ border-collapse: collapse; width: 100%; font-size: 12px; margin: 12px 0 24px 0; }}
th, td {{ border: 1px solid #ddd; padding: 6px; text-align: left; vertical-align: top; }}
th {{ background: #f0f0f0; }}
pre {{ white-space: pre-wrap; background: #fafafa; border: 1px solid #ddd; padding: 12px; }}
figure {{ margin: 24px 0; page-break-inside: avoid; }}
img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
figcaption {{ font-size: 12px; color: #555; }}
</style>
</head>
<body>
{''.join(body)}
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")


def processar_arquivo(path: Path, cfg: Config) -> Dict[str, Any]:
    rel = path.relative_to(cfg.root).as_posix()
    base_name = f"{hash_curto(rel)}_{safe_name(rel.replace('.', '_'), 100)}"
    base_dir = cfg.out_dir / "individual" / base_name
    tables_dir = base_dir / "tabelas"
    tables_dir.mkdir(parents=True, exist_ok=True)

    done_marker = base_dir / "_OK.json"
    if cfg.resume and done_marker.exists():
        try:
            return json.loads(done_marker.read_text(encoding="utf-8"))
        except Exception:
            pass

    logging.info("Processando arquivo: %s", path)
    try:
        rows, meta = streaming_sample(path, cfg)
        if not rows:
            raise RuntimeError("Nenhuma linha válida encontrada.")

        df = pd.DataFrame(rows)
        df = limitar_colunas(df, cfg.max_columns)
        df = df.convert_dtypes()
        meta["colunas_amostra_apos_limite"] = int(df.shape[1])

        salvar_json(meta, tables_dir / "metadados_amostragem.json")

        profile = inferir_perfil_amostra(df)
        salvar_csv(profile, tables_dir / "perfil_colunas_amostra.csv")

        # Catálogos explícitos: o que foi processado e o que foi ignorado como código.
        colunas_analiticas_df = profile.loc[profile["processar_analise"].astype(bool)].copy()
        colunas_ignoradas_df = profile.loc[profile["classe_semantica"].isin(["codigo", "codigo_provavel", "vazio", "data", "texto"])].copy()
        salvar_csv(colunas_analiticas_df, tables_dir / "colunas_processadas_analiticas.csv")
        salvar_csv(colunas_ignoradas_df, tables_dir / "colunas_ignoradas_na_analise.csv")

        salvar_amostra_paths = salvar_amostra(df, base_dir, cfg)
        salvar_amostra_paths.update(salvar_amostra_analitica(df, profile, base_dir, cfg))

        if cfg.salvar_parquet_completo_analitico:
            pq_full = converter_parquet_completo_analitico(path, profile, base_dir, cfg)
            if pq_full:
                salvar_amostra_paths["parquet_completo_analitico_sem_codigos"] = str(pq_full)

        num_cols = selecionar_numericas_analiticas(profile, cfg.max_corr_cols)
        df_num = preparar_matriz_numerica(df, num_cols)
        stats_num = gerar_estatisticas_numericas(df_num)
        salvar_csv(stats_num, tables_dir / "estatisticas_numericas_amostra.csv")

        pearson, corr_engine = corr_pearson_gpu_ou_cpu(df_num, cfg)
        salvar_csv(pearson, tables_dir / "matriz_pearson_amostra.csv", index=True)
        top_pearson = top_corr_pairs(pearson, "pearson_amostra")
        salvar_csv(top_pearson, tables_dir / "top_correlacoes_pearson_amostra.csv")

        if cfg.gerar_spearman and not df_num.empty and df_num.shape[1] >= 2:
            spearman = df_num.corr(method="spearman", min_periods=3)
        else:
            spearman = pd.DataFrame()
        salvar_csv(spearman, tables_dir / "matriz_spearman_amostra.csv", index=True)
        top_spearman = top_corr_pairs(spearman, "spearman_amostra")
        salvar_csv(top_spearman, tables_dir / "top_correlacoes_spearman_amostra.csv")

        if cfg.gerar_kendall and not df_num.empty and df_num.shape[1] >= 2 and len(df_num) <= 50000:
            kendall = df_num.corr(method="kendall", min_periods=3)
        else:
            kendall = pd.DataFrame()
        salvar_csv(kendall, tables_dir / "matriz_kendall_amostra.csv", index=True)
        top_kendall = top_corr_pairs(kendall, "kendall_amostra")
        salvar_csv(top_kendall, tables_dir / "top_correlacoes_kendall_amostra.csv")

        resumo_cat = gerar_resumo_categorico(df, profile, cfg, tables_dir)
        salvar_csv(resumo_cat, tables_dir / "resumo_categorico_amostra.csv")

        cat_cols = selecionar_categoricas(profile, cfg.max_categories)[:40]
        cv = cramers_v_sample(df, cat_cols, max_pairs=400)
        salvar_csv(cv, tables_dir / "associacao_categorica_cramersv_amostra.csv")

        eta2 = eta2_sample(df, cat_cols[:25], df_num, max_pairs=400)
        salvar_csv(eta2, tables_dir / "associacao_num_cat_eta2_amostra.csv")

        if cfg.plots_full:
            images, meta_plots = gerar_graficos_100pct_streaming(
                path=path,
                df_num=df_num,
                profile=profile,
                pearson=pearson,
                top_pairs=top_pearson,
                base_dir=base_dir,
                cfg=cfg,
            )
        else:
            images = gerar_graficos(df, df_num, profile, pearson, top_pearson, base_dir, cfg)
            meta_plots = pd.DataFrame([{
                "modo_plots": "amostra_controlada",
                "linhas_usadas": len(df),
                "observacao": "plots gerados a partir da amostra controlada"
            }])
        salvar_csv(meta_plots, tables_dir / "metadados_plots.csv")

        resumo = (
            f"Arquivo: {path}\n"
            f"Relativo: {rel}\n"
            f"Linhas lidas: {meta.get('linhas_lidas')}\n"
            f"Linhas mantidas na amostra: {meta.get('linhas_amostra')}\n"
            f"Fração solicitada: {cfg.sample_frac:.4f}\n"
            f"Cap de amostra: {cfg.max_sample_rows}\n"
            f"Modo de leitura: {cfg.scan_mode}\n"
            f"Colunas na amostra: {df.shape[1]}\n"
            f"Colunas analíticas processadas: {int(profile['processar_analise'].astype(bool).sum())}\n"
            f"Colunas ignoradas como código/provável código: {len(colunas_ignoradas_codigos(profile))}\n"
            f"Colunas numéricas usadas na correlação: {df_num.shape[1]}\n"
            f"Motor de correlação Pearson: {corr_engine}\n"
            f"Parquet analítico completo: {'sim' if cfg.salvar_parquet_completo_analitico else 'não'}\n"
            f"Modo dos plots: {'100% streaming agregado' if cfg.plots_full else 'amostra controlada'}\n"
            f"Observação: códigos/chaves foram documentados no perfil, mas ignorados nas análises pesadas. Spearman/Kendall são amostrais; plots 100% usam agregação streaming quando --plots-full está ativo.\n"
        )

        sections = [
            ("Metadados da amostragem", pd.DataFrame([meta | salvar_amostra_paths | {"corr_engine": corr_engine}])),
            ("Metadados dos plots", meta_plots),
            ("Perfil das colunas na amostra", profile),
            ("Colunas processadas analíticas", colunas_analiticas_df),
            ("Colunas ignoradas/códigos/texto pesado", colunas_ignoradas_df),
            ("Estatísticas numéricas na amostra", stats_num),
            ("Top Pearson na amostra", top_pearson),
            ("Top Spearman na amostra", top_spearman),
            ("Top Kendall na amostra", top_kendall),
            ("Resumo categórico na amostra", resumo_cat),
            ("Cramér's V categórico na amostra", cv),
            ("Eta² numérico x categórico na amostra", eta2),
        ]
        html_path = base_dir / "relatorio_amostra.html"
        render_html(html_path, f"Relatório amostral seguro - {rel}", resumo, sections, images, cfg.top_n_html)

        result = {
            "status": "ok",
            "arquivo": str(path),
            "relativo": rel,
            "linhas_lidas": int(meta.get("linhas_lidas") or 0),
            "linhas_amostra": int(meta.get("linhas_amostra") or 0),
            "colunas_amostra": int(df.shape[1]),
            "colunas_analiticas_processadas": int(profile["processar_analise"].astype(bool).sum()),
            "colunas_ignoradas_codigos": int(len(colunas_ignoradas_codigos(profile))),
            "corr_engine": corr_engine,
            "relatorio_html": str(html_path),
            "base_dir": str(base_dir),
            "erro": "",
        }
        salvar_json(result, done_marker)

        # Libera memória agressivamente entre arquivos.
        del rows, df, profile, df_num, stats_num, pearson, top_pearson, spearman, kendall, resumo_cat, cv, eta2, colunas_analiticas_df, colunas_ignoradas_df, meta_plots, images
        gc.collect()
        cleanup_runtime(f"fim_arquivo:{rel}")
        return result

    except Exception as exc:
        erro = traceback.format_exc()
        err_path = base_dir / "erro.txt"
        err_path.parent.mkdir(parents=True, exist_ok=True)
        err_path.write_text(erro, encoding="utf-8")
        logging.error("Erro processando %s: %s", path, exc)
        return {
            "status": "erro",
            "arquivo": str(path),
            "relativo": rel,
            "linhas_lidas": 0,
            "linhas_amostra": 0,
            "colunas_amostra": 0,
            "corr_engine": "",
            "relatorio_html": "",
            "base_dir": str(base_dir),
            "erro": str(exc),
        }


def construir_global(resultados: List[Dict[str, Any]], cfg: Config) -> None:
    global_dir = cfg.out_dir / "global"
    global_dir.mkdir(parents=True, exist_ok=True)
    catalogo = pd.DataFrame(resultados)
    salvar_csv(catalogo, global_dir / "catalogo_arquivos.csv")

    resumo = (
        f"Raiz: {cfg.root}\n"
        f"Saída: {cfg.out_dir}\n"
        f"Arquivos processados: {len(resultados)}\n"
        f"OK: {sum(1 for r in resultados if r.get('status') == 'ok')}\n"
        f"Erro: {sum(1 for r in resultados if r.get('status') != 'ok')}\n"
        f"Amostra solicitada: {cfg.sample_frac:.4f}\n"
        f"Cap por arquivo: {cfg.max_sample_rows}\n"
        f"Modo de leitura: {cfg.scan_mode}\n"
        f"Gerado em: {datetime.now().isoformat(timespec='seconds')}\n"
    )
    render_html(
        global_dir / "relatorio_global_amostral.html",
        "Relatório global amostral seguro",
        resumo,
        [("Catálogo de arquivos", catalogo)],
        [],
        cfg.top_n_html,
    )
    salvar_json({
        "root": str(cfg.root),
        "out_dir": str(cfg.out_dir),
        "processados": len(resultados),
        "ok": sum(1 for r in resultados if r.get("status") == "ok"),
        "erro": sum(1 for r in resultados if r.get("status") != "ok"),
        "relatorio_global_html": str(global_dir / "relatorio_global_amostral.html"),
    }, global_dir / "resumo_global_amostral.json")


def normalizar_saida_resultados(value: str) -> Path:
    out = Path(value).expanduser()
    if out.is_absolute():
        return out.resolve()
    parts = out.parts
    if parts and parts[0].lower() == "resultados":
        return (Path.cwd() / out).resolve()
    return (Path.cwd() / "resultados" / out).resolve()


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Análise eleitoral segura por amostragem streaming com GPU opcional.")
    parser.add_argument("dados", help="Pasta dados. O script busca arquivos em subpastas chamadas JSON.")
    parser.add_argument("--out", default="analise_individual_v2", help="Nome do run dentro de resultados/.")
    parser.add_argument("--sample-frac", type=float, default=0.05, help="Fração desejada para amostra. Ex.: 0.05 = 5%%.")
    parser.add_argument("--max-sample-rows", type=int, default=300000, help="Máximo de linhas mantidas em memória por arquivo.")
    parser.add_argument("--min-sample-rows", type=int, default=1000, help="Mínimo de linhas a tentar manter em arquivos pequenos.")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--scan-mode", choices=["reservoir", "head"], default="reservoir", help="reservoir lê tudo com memória limitada; head lê só o começo.")
    parser.add_argument("--max-columns", type=int, default=160, help="Máximo de colunas mantidas por amostra.")
    parser.add_argument("--max-corr-cols", type=int, default=40, help="Máximo de colunas numéricas na matriz de correlação.")
    parser.add_argument("--max-categories", type=int, default=80, help="Máximo de categorias únicas para analisar coluna categórica.")
    parser.add_argument("--top-n-plots", type=int, default=15)
    parser.add_argument("--top-n-html", type=int, default=100)
    parser.add_argument("--gpu", choices=["auto", "on", "off"], default="auto", help="GPU opcional para Pearson: auto/on/off.")
    parser.add_argument("--gpu-max-mb", type=int, default=2048, help="Limite estimado de memória para mandar matriz à GPU.")
    parser.add_argument("--sem-spearman", action="store_true", help="Desativa Spearman amostral.")
    parser.add_argument("--kendall", action="store_true", help="Ativa Kendall amostral; só roda até 50k linhas para evitar travar.")
    parser.add_argument("--sem-scatter", action="store_true", help="Desativa scatterplots amostrais/densidade 2D 100%.")
    parser.add_argument("--plots-full", action="store_true", help="Gera plots com 100% das linhas por agregação streaming. Spearman/Kendall continuam amostrais.")
    parser.add_argument("--full-plot-bins", type=int, default=80, help="Número de bins para histogramas/densidade 2D no modo --plots-full.")
    parser.add_argument("--full-density-pairs", type=int, default=6, help="Quantidade de pares principais para densidade 2D 100% no modo --plots-full.")
    parser.add_argument("--sem-parquet", action="store_true", help="Não salva amostra em Parquet.")
    parser.add_argument("--sem-amostra-csv", action="store_true", help="Não salva amostra em CSV gzip.")
    parser.add_argument("--parquet-completo-analitico", action="store_true", help="Além da amostra, relê o JSON e salva Parquet completo sem códigos, em chunks.")
    parser.add_argument("--parquet-chunk-rows", type=int, default=50000, help="Linhas por chunk ao gerar Parquet completo analítico.")
    parser.add_argument("--json-array-backend", choices=["auto", "ijson", "stdlib"], default="auto", help="Backend para .json array.")
    parser.add_argument("--flatten-depth", type=int, default=2)
    parser.add_argument("--resume", action="store_true", help="Pula arquivos que já têm _OK.json.")

    args = parser.parse_args()
    root = Path(args.dados).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"[ERRO] Pasta não encontrada: {root}", file=sys.stderr)
        sys.exit(1)
    out_dir = normalizar_saida_resultados(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    return Config(
        root=root,
        out_dir=out_dir,
        sample_frac=float(args.sample_frac),
        max_sample_rows=int(args.max_sample_rows),
        min_sample_rows=int(args.min_sample_rows),
        sample_seed=int(args.sample_seed),
        scan_mode=str(args.scan_mode),
        max_columns=int(args.max_columns),
        max_corr_cols=int(args.max_corr_cols),
        max_categories=int(args.max_categories),
        top_n_plots=int(args.top_n_plots),
        top_n_html=int(args.top_n_html),
        gpu=str(args.gpu),
        gpu_max_mb=int(args.gpu_max_mb),
        gerar_spearman=not bool(args.sem_spearman),
        gerar_kendall=bool(args.kendall),
        gerar_scatter=not bool(args.sem_scatter),
        plots_full=bool(args.plots_full),
        full_plot_bins=int(args.full_plot_bins),
        full_density_pairs=int(args.full_density_pairs),
        salvar_parquet=not bool(args.sem_parquet),
        salvar_amostra_csv=not bool(args.sem_amostra_csv),
        salvar_parquet_completo_analitico=bool(args.parquet_completo_analitico),
        parquet_chunk_rows=int(args.parquet_chunk_rows),
        json_array_backend=str(args.json_array_backend),
        flatten_depth=int(args.flatten_depth),
        resume=bool(args.resume),
    )


def main() -> None:
    cfg = parse_args()
    log_path = configurar_logging(cfg.out_dir)
    logging.info("Log: %s", log_path)
    logging.info("Config: %s", cfg)

    arquivos = encontrar_jsons(cfg.root)
    logging.info("Arquivos JSON encontrados: %s", len(arquivos))
    if not arquivos:
        logging.warning("Nenhum arquivo encontrado. O script procura arquivos dentro de pastas chamadas JSON.")
        construir_global([], cfg)
        return

    resultados: List[Dict[str, Any]] = []
    for i, path in enumerate(arquivos, start=1):
        logging.info("[%s/%s] %s", i, len(arquivos), path)
        res = processar_arquivo(path, cfg)
        resultados.append(res)
        construir_global(resultados, cfg)
        gc.collect()
        cleanup_runtime(f"main_loop:{path.name}")

    construir_global(resultados, cfg)
    logging.info("Finalizado. Relatório global: %s", cfg.out_dir / "global" / "relatorio_global_amostral.html")


if __name__ == "__main__":
    main()
