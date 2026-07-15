from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import argparse
import json
import sys

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


NULL_WORDS = {
    "",
    "nan",
    "none",
    "null",
    "<na>",
    "#nulo#",
    "sem valor",
    "sem_valor",
    "nao informado",
    "sem entidade",
    "geral",
}

PROFILE_FIELDS = [
    "perfil_faixa_etaria_dominante",
    "perfil_genero_dominante",
    "perfil_instrucao_dominante",
    "perfil_estado_civil_dominante",
    "perfil_raca_cor_dominante",
]

TABLE_CANDIDATES: dict[str, list[str]] = {
    "catalogo": [
        "global/tabelas/catalogo_processamento.csv",
    ],
    "municipal": [
        "ouro/retrato_municipal",
        "ouro/retrato_municipal.parquet",
        "global/parquet/retrato_municipal_global.parquet",
        "global/tabelas/retrato_municipal_global.csv",
    ],
    "timeline_nacional": [
        "ouro/timeline_nacional.parquet",
        "global/parquet/timeline_nacional.parquet",
        "global/timeline/timeline_nacional.csv",
    ],
    "timeline_uf": [
        "ouro/timeline_uf",
        "ouro/timeline_uf.parquet",
        "global/parquet/timeline_uf.parquet",
        "global/timeline/timeline_uf.csv",
    ],
    "timeline_municipal": [
        "ouro/timeline_municipal",
        "ouro/timeline_municipal.parquet",
        "global/parquet/timeline_municipal.parquet",
        "global/timeline/timeline_municipal.csv",
    ],
    "timeline_entidades": [
        "global/parquet/timeline_entidades.parquet",
        "global/timeline/timeline_entidades.csv",
    ],
    "perfil_ano": [
        "ouro/perfil_eleitor_por_ano",
        "ouro/perfil_eleitor_por_ano.parquet",
        "global/analise_eleitoral/parquet/perfil_eleitor_por_ano.parquet",
        "global/analise_eleitoral/perfil_eleitor_por_ano.csv",
    ],
    "perfil_partido": [
        "ouro/perfil_eleitor_por_partido",
        "ouro/perfil_eleitor_por_partido.parquet",
        "global/analise_eleitoral/parquet/perfil_eleitor_por_partido.parquet",
        "global/analise_eleitoral/perfil_eleitor_por_partido.csv",
    ],
    "perfil_candidato": [
        "ouro/perfil_eleitor_por_candidato",
        "ouro/perfil_eleitor_por_candidato.parquet",
        "global/analise_eleitoral/parquet/perfil_eleitor_por_candidato.parquet",
        "global/analise_eleitoral/perfil_eleitor_por_candidato.csv",
    ],
    "perfil_do_candidato": [
        "global/analise_eleitoral/parquet/perfil_do_candidato_correlacionado_eleitorado.parquet",
        "global/analise_eleitoral/perfil_do_candidato_correlacionado_eleitorado.csv",
    ],
    "resultado_eleitorado": [
        "ouro/resultado_eleitorado_por_secao",
        "ouro/resultado_eleitorado_por_secao.parquet",
        "global/analise_eleitoral/parquet/resultado_eleitorado_correlacionado.parquet",
        "global/analise_eleitoral/resultado_eleitorado_correlacionado.csv",
    ],
    "resultado_partido": [
        "ouro/brasil/resultado_partido",
        "ouro/estadual/resultado_partido",
        "ouro/municipal/resultado_partido",
    ],
    "contagem_colunas_resultado_partido": [
        "ouro/brasil/contagem_colunas_resultado_partido",
        "ouro/estadual/contagem_colunas_resultado_partido",
        "ouro/municipal/contagem_colunas_resultado_partido",
    ],
    "resultado_candidato": [
        "ouro/brasil/resultado_candidato",
        "ouro/estadual/resultado_candidato",
        "ouro/municipal/resultado_candidato",
    ],
    "contagem_colunas_resultado_candidato": [
        "ouro/brasil/contagem_colunas_resultado_candidato",
        "ouro/estadual/contagem_colunas_resultado_candidato",
        "ouro/municipal/contagem_colunas_resultado_candidato",
    ],
    "comparativo_perfil": [
        "ouro/top10_perfis_federacao_estado_municipio",
        "ouro/top10_perfis_federacao_estado_municipio.parquet",
        "global/analise_eleitoral/parquet/comparativo_anual_perfil_eleitor.parquet",
        "global/analise_eleitoral/comparativo_anual_perfil_eleitor.csv",
    ],
    "comparativo_partido": [
        "ouro/comparativo_anual_perfil_partido",
        "ouro/comparativo_anual_perfil_partido.parquet",
        "global/analise_eleitoral/parquet/comparativo_anual_perfil_partido.parquet",
        "global/analise_eleitoral/comparativo_anual_perfil_partido.csv",
    ],
    "comparativo_candidato": [
        "ouro/comparativo_anual_perfil_candidato",
        "ouro/comparativo_anual_perfil_candidato.parquet",
        "global/analise_eleitoral/parquet/comparativo_anual_perfil_candidato.parquet",
        "global/analise_eleitoral/comparativo_anual_perfil_candidato.csv",
    ],
    "top10_perfis": [
        "ouro/top10_perfis_federacao_estado_municipio",
        "ouro/top10_perfis_federacao_estado_municipio.parquet",
        "global/analise_eleitoral/parquet/top10_perfis_federacao_estado_municipio.parquet",
        "global/analise_eleitoral/top10_perfis_federacao_estado_municipio.csv",
    ],
    "respostas": [
        "global/analise_eleitoral/parquet/respostas_perguntas_eleitorais.parquet",
        "global/analise_eleitoral/respostas_perguntas_eleitorais.csv",
    ],
    "vencedor_secao": [
        "ouro/resultados_vencedores_secao",
        "global/analise_eleitoral/parquet/vencedor_por_secao.parquet",
        "global/analise_eleitoral/vencedor_por_secao.csv",
    ],
    "quem_vota": [
        "global/comportamento_eleitoral/parquet/quem_vota_em_quem.parquet",
        "global/comportamento_eleitoral/quem_vota_em_quem.csv",
    ],
    "afinidade": [
        "global/comportamento_eleitoral/parquet/afinidade_perfil_candidato.parquet",
        "global/comportamento_eleitoral/afinidade_perfil_candidato.csv",
    ],
    "cluster_voter_personas": [
        "global/correlacao_codigos/clusters/parquet/clusters_eleitores_personas.parquet",
        "global/correlacao_codigos/clusters/tabelas/clusters_eleitores_personas.csv",
    ],
    "cluster_voter_year_region": [
        "global/correlacao_codigos/clusters/parquet/clusters_eleitores_ano_regiao.parquet",
        "global/correlacao_codigos/clusters/tabelas/clusters_eleitores_ano_regiao.csv",
    ],
    "cluster_voter_discriminants": [
        "global/correlacao_codigos/clusters/parquet/clusters_eleitores_valores_discriminantes.parquet",
        "global/correlacao_codigos/clusters/tabelas/clusters_eleitores_valores_discriminantes.csv",
    ],
    "cluster_result_personas": [
        "global/correlacao_codigos/clusters/parquet/clusters_personas.parquet",
        "global/correlacao_codigos/clusters/tabelas/clusters_personas.csv",
    ],
    "cluster_result_year_region": [
        "global/correlacao_codigos/clusters/parquet/clusters_ano_regiao.parquet",
        "global/correlacao_codigos/clusters/tabelas/clusters_ano_regiao.csv",
    ],
    "cluster_result_discriminants": [
        "global/correlacao_codigos/clusters/parquet/clusters_valores_discriminantes.parquet",
        "global/correlacao_codigos/clusters/tabelas/clusters_valores_discriminantes.csv",
    ],
    "cluster_result_prediction": [
        "global/correlacao_codigos/clusters/parquet/clusters_predicao_2026.parquet",
        "global/correlacao_codigos/clusters/tabelas/clusters_predicao_2026.csv",
    ],
    "cluster_result_entities": [
        "global/correlacao_codigos/clusters/parquet/clusters_entidades.parquet",
        "global/correlacao_codigos/clusters/tabelas/clusters_entidades.csv",
    ],
    "cluster_elbow": [
        "global/correlacao_codigos/clusters/parquet/clusters_cotovelo_k.parquet",
        "global/correlacao_codigos/clusters/tabelas/clusters_cotovelo_k.csv",
    ],
    "correlacao_manifesto": [
        "global/correlacao_codigos/parquet/manifesto_parquets_correlacionados_por_ano.parquet",
        "global/correlacao_codigos/tabelas/manifesto_parquets_correlacionados_por_ano.csv",
    ],
    "correlacao_stats": [
        "global/correlacao_codigos/parquet/estatisticas_correlacionadas_por_ano.parquet",
        "global/correlacao_codigos/tabelas/estatisticas_correlacionadas_por_ano.csv",
    ],
    "sim_nacional": [
        "preditivo_2026/parquet/cenarios_nacionais.parquet",
        "preditivo_2026/tabelas/cenarios_nacionais.csv",
    ],
    "sim_monte_carlo": [
        "preditivo_2026/parquet/monte_carlo.parquet",
        "preditivo_2026/tabelas/monte_carlo.csv",
    ],
    "sim_decisive": [
        "preditivo_2026/parquet/secoes_municipios_decisivos.parquet",
        "preditivo_2026/tabelas/secoes_municipios_decisivos.csv",
    ],
    "sim_partidos_brasil": [
        "preditivo_2026/parquet/partidos_2026_brasil.parquet",
        "preditivo_2026/tabelas/partidos_2026_brasil.csv",
    ],
    "sim_partidos_estados": [
        "preditivo_2026/parquet/partidos_2026_estados.parquet",
        "preditivo_2026/tabelas/partidos_2026_estados.csv",
    ],
    "sim_partidos_municipios": [
        "preditivo_2026/parquet/partidos_2026_municipios.parquet",
        "preditivo_2026/tabelas/partidos_2026_municipios.csv",
    ],
    "sim_partidos_correlacao": [
        "preditivo_2026/parquet/partidos_2026_correlacao_historica.parquet",
        "preditivo_2026/tabelas/partidos_2026_correlacao_historica.csv",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run", default="", help="Pasta do run em resultados/.")
    parser.add_argument("--preview-rows", type=int, default=2500)
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


def resolve_run_path(value: str) -> Path:
    if not value:
        latest = latest_run()
        return latest if latest is not None else Path.cwd() / "resultados"
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    if path.parts and path.parts[0].lower() == "resultados":
        return (Path.cwd() / path).resolve()
    candidate = (Path.cwd() / "resultados" / path).resolve()
    if candidate.exists():
        return candidate
    return (Path.cwd() / path).resolve()


def latest_run() -> Path | None:
    root = Path.cwd() / "resultados"
    if not root.exists():
        return None
    runs = [p for p in root.iterdir() if p.is_dir()]
    if not runs:
        return None
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def list_runs() -> list[Path]:
    root = Path.cwd() / "resultados"
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)


def inject_css() -> None:
    st.markdown(
        """
<style>
:root {
  --panel: #ffffff;
  --ink: #111827;
  --muted: #475569;
  --line: #d8e0ea;
  --blue: #2563eb;
  --blue-soft: #dbeafe;
  --green-soft: #dcfce7;
  --amber-soft: #fef3c7;
}
.block-container { padding-top: 1.2rem; max-width: 1500px; }
div[data-testid="stMetric"] {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px 16px;
  box-shadow: 0 6px 18px rgba(15, 23, 42, .05);
}
div[data-testid="stVegaLiteChart"],
div[data-testid="stPlotlyChart"],
div[data-testid="stPyplot"] {
  transition: transform .18s ease, box-shadow .18s ease;
  transform-origin: center;
}
div[data-testid="stVegaLiteChart"]:hover,
div[data-testid="stPlotlyChart"]:hover,
div[data-testid="stPyplot"]:hover {
  transform: scale(1.035);
  z-index: 5;
  box-shadow: 0 14px 34px rgba(15, 23, 42, .18);
}
.card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 12px;
  margin: 8px 0 18px 0;
}
.info-card {
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 14px 16px;
  box-shadow: 0 8px 22px rgba(15, 23, 42, .05);
}
.info-card h4 { margin: 0 0 8px 0; font-size: 1rem; color: var(--ink); }
.info-card p { margin: 6px 0; color: var(--muted); line-height: 1.35; }
.chip {
  display: inline-block;
  margin: 5px 5px 0 0;
  padding: 4px 8px;
  border-radius: 999px;
  background: var(--blue-soft);
  color: #0f3a7c;
  font-size: .78rem;
  border: 1px solid #bfdbfe;
}
.chip.green { background: var(--green-soft); color: #14532d; border-color: #bbf7d0; }
.chip.amber { background: var(--amber-soft); color: #78350f; border-color: #fde68a; }
.muted { color: var(--muted); font-size: .9rem; }
.path-box {
  background: #0f172a;
  color: #f8fafc;
  border-radius: 8px;
  padding: 10px 12px;
  font-family: Consolas, monospace;
  overflow-x: auto;
}
.br-map-wrap {
  display: grid;
  grid-template-columns: minmax(320px, 1.2fr) minmax(260px, .8fr);
  gap: 14px;
  align-items: stretch;
}
.br-map {
  position: relative;
  min-height: 420px;
  border: 1px solid #d8e0ea;
  border-radius: 8px;
  background: linear-gradient(180deg, #f8fbff, #eef7f6);
  overflow: hidden;
}
.br-map:before {
  content: "";
  position: absolute;
  inset: 28px 46px;
  border-radius: 46% 54% 52% 48%;
  border: 2px dashed rgba(15, 118, 110, .22);
  transform: rotate(-12deg);
}
.uf-dot {
  position: absolute;
  transform: translate(-50%, -50%);
  min-width: 42px;
  min-height: 34px;
  border: 1px solid #93c5fd;
  border-radius: 8px;
  background: #fff;
  color: #0f172a;
  font-weight: 700;
  cursor: pointer;
  box-shadow: 0 8px 22px rgba(15, 23, 42, .08);
  animation: pulseState 2.6s ease-in-out infinite;
}
.uf-dot:hover, .uf-dot.active {
  background: #2563eb;
  color: #fff;
  border-color: #2563eb;
  transform: translate(-50%, -50%) scale(1.12);
}
.state-detail {
  border: 1px solid #d8e0ea;
  border-radius: 8px;
  padding: 14px 16px;
  background: #fff;
  min-height: 160px;
}
@keyframes pulseState {
  0%, 100% { box-shadow: 0 0 0 0 rgba(37, 99, 235, .18); }
  50% { box-shadow: 0 0 0 8px rgba(37, 99, 235, 0); }
}
@media (max-width: 760px) { .br-map-wrap { grid-template-columns: 1fr; } }
</style>
""",
        unsafe_allow_html=True,
    )


def path_for(run_dir: Path, name: str) -> Path | None:
    for rel in TABLE_CANDIDATES.get(name, []):
        path = run_dir / rel
        if path.is_dir():
            if next(path.rglob("*.parquet"), None) is not None:
                return path
        elif path.exists():
            return path
    return None


def existing_tables(run_dir: Path) -> dict[str, Path]:
    out = {}
    for name in TABLE_CANDIDATES:
        path = path_for(run_dir, name)
        if path is not None:
            out[name] = path
    return out


def read_table(run_dir: Path, name: str, columns: Iterable[str] | None = None, max_rows: int | None = 2500) -> pd.DataFrame:
    path = path_for(run_dir, name)
    if path is None:
        return pd.DataFrame()
    return read_table_path(path, tuple(columns or ()), max_rows)


def read_table_path(path: Path, columns: tuple[str, ...] = (), max_rows: int | None = 2500) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    stat = path.stat()
    return _read_table_cached(str(path), columns, max_rows if max_rows and max_rows > 0 else None, stat.st_mtime_ns, stat.st_size)


@st.cache_data(show_spinner=False)
def _read_table_cached(path_str: str, columns: tuple[str, ...], max_rows: int | None, mtime_ns: int, size_bytes: int) -> pd.DataFrame:
    path = Path(path_str)
    try:
        if path.is_dir() or path.suffix.lower() == ".parquet":
            return _read_parquet_limited(path, list(columns), max_rows)
        return _read_csv_limited(path, list(columns), max_rows)
    except Exception as exc:
        st.warning(f"Nao consegui ler {path.name}: {exc}")
        return pd.DataFrame()


def _read_parquet_limited(path: Path, columns: list[str], max_rows: int | None) -> pd.DataFrame:
    if path.is_dir():
        return _read_parquet_dataset_limited(path, columns, max_rows)
    try:
        import pyarrow.parquet as pq
    except Exception:
        if path.stat().st_size > 256 * 1024 * 1024:
            return pd.DataFrame({"status": ["Instale pyarrow para consultar este Parquet grande sem carregar tudo."]})
        return pd.read_parquet(path, columns=columns or None).head(max_rows)

    pf = pq.ParquetFile(path)
    available = set(pf.schema.names)
    selected = [c for c in columns if c in available] if columns else None
    if max_rows is None:
        return pf.read(columns=selected).to_pandas()

    frames = []
    remaining = max(int(max_rows), 1)
    batch_size = min(max(remaining, 1), 50000)
    for batch in pf.iter_batches(batch_size=batch_size, columns=selected):
        part = batch.to_pandas()
        frames.append(part)
        remaining -= len(part)
        if remaining <= 0:
            break
    if not frames:
        return pd.DataFrame(columns=selected or [])
    return pd.concat(frames, ignore_index=True).head(max_rows)


def _read_parquet_dataset_limited(path: Path, columns: list[str], max_rows: int | None) -> pd.DataFrame:
    try:
        import pyarrow.dataset as ds
    except Exception:
        return pd.read_parquet(path, columns=columns or None).head(max_rows)

    dataset = ds.dataset(str(path), format="parquet", partitioning="hive")
    available = set(dataset.schema.names)
    selected = [c for c in columns if c in available] if columns else None
    if max_rows is None:
        return dataset.to_table(columns=selected).to_pandas()

    frames = []
    remaining = max(int(max_rows), 1)
    scanner = dataset.scanner(columns=selected, batch_size=min(max(remaining, 1), 50000))
    for batch in scanner.to_batches():
        part = batch.to_pandas()
        frames.append(part)
        remaining -= len(part)
        if remaining <= 0:
            break
    if not frames:
        return pd.DataFrame(columns=selected or [])
    return pd.concat(frames, ignore_index=True).head(max_rows)


def _read_csv_limited(path: Path, columns: list[str], max_rows: int | None) -> pd.DataFrame:
    kwargs: dict[str, Any] = {"sep": ";", "dtype": str, "nrows": max_rows}
    if columns:
        try:
            header = pd.read_csv(path, sep=";", nrows=0, encoding="utf-8-sig")
            usecols = [c for c in columns if c in header.columns]
            if usecols:
                kwargs["usecols"] = usecols
        except Exception:
            pass
    try:
        return pd.read_csv(path, encoding="utf-8-sig", **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1", **kwargs)


def load_json_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return pd.DataFrame(obj if isinstance(obj, list) else [obj])
    except Exception:
        return pd.DataFrame()


def meaningful(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if text.endswith(".0") and text[:-2].replace("-", "", 1).isdigit():
        text = text[:-2]
    lower = text.lower()
    code_value = lower.replace("codigo ", "", 1).replace(".", "", 1).lstrip("-+")
    if lower in NULL_WORDS or lower.endswith("_sem_valor") or lower.endswith(" sem valor"):
        return ""
    if lower.startswith("codigo ") and code_value.isdigit():
        return ""
    return text


def numeric(series: pd.Series | Any) -> pd.Series:
    return pd.to_numeric(series, errors="coerce") if isinstance(series, pd.Series) else pd.to_numeric(pd.Series(series), errors="coerce")


def fmt_int(value: Any) -> str:
    num = pd.to_numeric(value, errors="coerce")
    if pd.isna(num):
        return "0"
    return f"{int(float(num)):,}".replace(",", ".")


def fmt_pct(value: Any) -> str:
    num = pd.to_numeric(value, errors="coerce")
    if pd.isna(num):
        return "sem dado"
    val = float(num)
    if val <= 1.5:
        val *= 100
    return f"{val:.1f}%"


def normalize_year(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["ano", "ano_correlacao", "ano_num"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64").astype(str).replace("<NA>", "")
    return out


def metric_sum(df: pd.DataFrame, *cols: str) -> float:
    for col in cols:
        if col in df.columns:
            return float(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())
    return 0.0


def metric_nunique(df: pd.DataFrame, *cols: str) -> int:
    for col in cols:
        if col in df.columns:
            return int(df[col].astype(str).replace("", np.nan).dropna().nunique())
    return 0


def html_card(title: str, body: str, chips: Iterable[str] = (), note: str = "") -> str:
    chip_html = "".join(f"<span class='chip'>{escape_html(chip)}</span>" for chip in chips if meaningful(chip))
    note_html = f"<p class='muted'>{escape_html(note)}</p>" if note else ""
    return (
        "<div class='info-card'>"
        f"<h4>{escape_html(title)}</h4>"
        f"<p>{escape_html(body)}</p>"
        f"{chip_html}{note_html}"
        "</div>"
    )


def escape_html(text: Any) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_cards(cards: list[str]) -> None:
    if not cards:
        st.info("Sem dados suficientes para montar cards nesta etapa.")
        return
    st.markdown("<div class='card-grid'>" + "".join(cards) + "</div>", unsafe_allow_html=True)


UF_LAYOUT = [
    ("AC", 13, 58), ("RO", 24, 62), ("AM", 28, 39), ("RR", 34, 20), ("AP", 53, 22), ("PA", 49, 39), ("TO", 54, 52),
    ("MA", 63, 43), ("PI", 68, 50), ("CE", 74, 46), ("RN", 80, 49), ("PB", 79, 53), ("PE", 77, 57), ("AL", 76, 61), ("SE", 74, 65), ("BA", 68, 67),
    ("MT", 47, 63), ("MS", 52, 76), ("GO", 60, 69), ("DF", 63, 71), ("MG", 66, 77), ("ES", 73, 78), ("RJ", 70, 84), ("SP", 62, 84),
    ("PR", 58, 89), ("SC", 60, 94), ("RS", 57, 98),
]


def latest_by_year(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "ano" not in df.columns:
        return df if df is not None else pd.DataFrame()
    out = df.copy()
    out["_ano_num"] = pd.to_numeric(out["ano"], errors="coerce")
    if out["_ano_num"].notna().any():
        out = out.loc[out["_ano_num"].eq(out["_ano_num"].max())].copy()
    return out


def profile_summary_text(row: pd.Series | dict[str, Any]) -> str:
    getter = row.get if hasattr(row, "get") else lambda key, default="": default
    return (
        meaningful(getter("descricao", ""))
        or meaningful(getter("perfil_combinado", ""))
        or meaningful(getter("pessoa_do_partido", ""))
        or meaningful(getter("pessoa_do_candidato", ""))
        or meaningful(getter("perfil_predominante", ""))
        or "Perfil ainda sem descricao consolidada."
    )


def render_top10_profile_cards(df: pd.DataFrame, title: str, limit: int = 10) -> None:
    if df is None or df.empty:
        st.info(f"{title}: sem dados neste run.")
        return
    work = latest_by_year(df).copy()
    if "rank_perfil_ano" in work.columns:
        work["_rank"] = pd.to_numeric(work["rank_perfil_ano"], errors="coerce").fillna(999)
        work = work.sort_values("_rank")
    cards = []
    for _, row in work.head(limit).iterrows():
        chips = [
            meaningful(row.get("ano", "")),
            meaningful(row.get("uf", "")),
            meaningful(row.get("nm_municipio", "")),
            meaningful(row.get("padrao_temporal", "")),
        ]
        body = profile_summary_text(row)
        note = f"Eleitorado: {fmt_int(row.get('eleitorado', 0))} | Share: {fmt_pct(row.get('share_perfil', np.nan))}"
        cards.append(html_card(meaningful(row.get("perfil_combinado", "")) or title, body, chips, note))
    st.subheader(title)
    render_cards(cards)


def render_brazil_map(data: dict[str, pd.DataFrame]) -> None:
    top = data.get("top10_perfis", pd.DataFrame())
    comp = data.get("comparativo_perfil", pd.DataFrame())
    profiles: dict[str, dict[str, Any]] = {}
    for uf, _, _ in UF_LAYOUT:
        rows = pd.DataFrame()
        if top is not None and not top.empty and {"nivel", "uf"}.issubset(top.columns):
            rows = top.loc[top["nivel"].astype(str).str.lower().eq("estado") & top["uf"].astype(str).eq(uf)].copy()
        if rows.empty and comp is not None and not comp.empty and {"nivel", "uf"}.issubset(comp.columns):
            rows = comp.loc[comp["nivel"].astype(str).str.lower().eq("estado") & comp["uf"].astype(str).eq(uf)].copy()
        rows = latest_by_year(rows)
        if "rank_perfil_ano" in rows.columns:
            rows["_rank"] = pd.to_numeric(rows["rank_perfil_ano"], errors="coerce").fillna(999)
            rows = rows.sort_values("_rank")
        row = rows.iloc[0].to_dict() if not rows.empty else {}
        profiles[uf] = {
            "summary": profile_summary_text(row),
            "ano": meaningful(row.get("ano", "")),
            "perfil": meaningful(row.get("perfil_combinado", "")),
            "share": fmt_pct(row.get("share_perfil", np.nan)),
            "eleitorado": fmt_int(row.get("eleitorado", 0)),
        }

    html_doc = """
<style>
.br-map-wrap { display:grid; grid-template-columns:minmax(320px,1.2fr) minmax(260px,.8fr); gap:14px; align-items:stretch; font-family:Arial, sans-serif; color:#111827; }
.br-map { position:relative; min-height:420px; border:1px solid #d8e0ea; border-radius:8px; background:linear-gradient(180deg,#f8fbff,#eef7f6); overflow:hidden; }
.br-map:before { content:""; position:absolute; inset:28px 46px; border-radius:46% 54% 52% 48%; border:2px dashed rgba(15,118,110,.22); transform:rotate(-12deg); }
.uf-dot { position:absolute; transform:translate(-50%,-50%); min-width:42px; min-height:34px; border:1px solid #93c5fd; border-radius:8px; background:#fff; color:#0f172a; font-weight:700; cursor:pointer; box-shadow:0 8px 22px rgba(15,23,42,.08); animation:pulseState 2.6s ease-in-out infinite; }
.uf-dot:hover,.uf-dot.active { background:#2563eb; color:#fff; border-color:#2563eb; transform:translate(-50%,-50%) scale(1.12); }
.state-detail { border:1px solid #d8e0ea; border-radius:8px; padding:14px 16px; background:#fff; min-height:160px; }
.chip { display:inline-block; margin:5px 5px 0 0; padding:4px 8px; border-radius:999px; background:#dbeafe; color:#0f3a7c; font-size:.78rem; border:1px solid #bfdbfe; }
.muted { color:#475569; font-size:.9rem; }
@keyframes pulseState { 0%,100% { box-shadow:0 0 0 0 rgba(37,99,235,.18); } 50% { box-shadow:0 0 0 8px rgba(37,99,235,0); } }
@media (max-width:760px) { .br-map-wrap { grid-template-columns:1fr; } }
</style>
<div class="br-map-wrap">
  <div class="br-map" id="brMapCanvas"></div>
  <div class="state-detail" id="brMapDetail"></div>
</div>
<script>
const layout = __LAYOUT__;
const profiles = __PROFILES__;
function esc(v){ return String(v ?? '').replace(/[&<>"']/g, s => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[s])); }
const canvas = document.getElementById('brMapCanvas');
const detail = document.getElementById('brMapDetail');
canvas.innerHTML = layout.map(([uf,x,y]) => `<button type="button" class="uf-dot" data-uf="${uf}" style="left:${x}%;top:${y}%">${uf}</button>`).join('');
function update(uf){
  canvas.querySelectorAll('.uf-dot').forEach(b => b.classList.toggle('active', b.dataset.uf === uf));
  const p = profiles[uf] || {};
  detail.innerHTML = `<h3>${esc(uf)}</h3><p>${esc(p.summary || 'Sem perfil calculado para este estado.')}</p><p><span class="chip">${esc(p.ano || 'ano indefinido')}</span><span class="chip">${esc(p.share || 'sem share')}</span><span class="chip">Eleitorado: ${esc(p.eleitorado || '0')}</span></p><p class="muted">${esc(p.perfil || '')}</p>`;
}
canvas.querySelectorAll('.uf-dot').forEach(btn => btn.addEventListener('mouseenter', () => update(btn.dataset.uf)));
canvas.querySelectorAll('.uf-dot').forEach(btn => btn.addEventListener('click', () => update(btn.dataset.uf)));
update('SP');
</script>
"""
    html_doc = html_doc.replace("__LAYOUT__", json.dumps(UF_LAYOUT, ensure_ascii=False))
    html_doc = html_doc.replace("__PROFILES__", json.dumps(profiles, ensure_ascii=False))
    components.html(html_doc, height=520)


def bar_chart(df: pd.DataFrame, x: str, y: str, title: str, limit: int = 20) -> None:
    if df is None or df.empty or x not in df.columns or y not in df.columns:
        st.caption(f"{title}: sem dados.")
        return
    chart = df[[x, y]].copy()
    chart[x] = chart[x].map(meaningful)
    chart[y] = pd.to_numeric(chart[y], errors="coerce").fillna(0)
    chart = chart.loc[chart[x].ne("")].groupby(x, dropna=False)[y].sum().sort_values(ascending=False).head(limit)
    if chart.empty:
        st.caption(f"{title}: sem dados uteis.")
        return
    if chart.index.astype(str).nunique() <= 1:
        st.caption(f"{title}: omitido porque ha apenas uma categoria util.")
        return
    st.subheader(title)
    st.bar_chart(chart)


def line_chart_by_year(df: pd.DataFrame, y_cols: list[str], title: str, year_col: str = "ano") -> None:
    if df is None or df.empty or year_col not in df.columns:
        st.caption(f"{title}: sem serie temporal.")
        return
    cols = [c for c in y_cols if c in df.columns]
    if not cols:
        st.caption(f"{title}: metricas nao encontradas.")
        return
    work = df[[year_col, *cols]].copy()
    work[year_col] = pd.to_numeric(work[year_col], errors="coerce")
    for col in cols:
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0)
    work = work.dropna(subset=[year_col]).groupby(year_col, dropna=False)[cols].sum().sort_index()
    if work.empty:
        st.caption(f"{title}: sem dados uteis.")
        return
    if work.index.nunique() <= 1:
        st.caption(f"{title}: omitido porque ha apenas um ano util.")
        return
    st.subheader(title)
    st.line_chart(work)


def kpi_row(results: pd.DataFrame, municipal: pd.DataFrame, timeline_national: pd.DataFrame, personas: pd.DataFrame) -> None:
    cols = st.columns(5)
    ok = int(results.get("status", pd.Series(dtype=str)).astype(str).eq("ok").sum()) if not results.empty else 0
    total_files = len(results) if not results.empty else 0
    years = metric_nunique(timeline_national, "ano") or metric_nunique(municipal, "ano")
    municipalities = metric_nunique(municipal, "cd_municipio", "nm_municipio")
    votes = metric_sum(timeline_national, "votos", "votos_total")
    cluster_count = metric_nunique(personas, "cluster_global_discriminado")
    cols[0].metric("Arquivos OK", fmt_int(ok), f"{fmt_int(total_files)} no manifesto")
    cols[1].metric("Anos", fmt_int(years), "base global")
    cols[2].metric("Municipios", fmt_int(municipalities), "consulta por municipio")
    cols[3].metric("Votos agregados", fmt_int(votes), "quando disponivel")
    cols[4].metric("Clusters", fmt_int(cluster_count), "perfis discretos")


def average_voter_card(profile_year: pd.DataFrame, personas: pd.DataFrame) -> None:
    title = "Eleitor medio no Brasil"
    traits: dict[str, str] = {}
    if profile_year is not None and not profile_year.empty:
        df = profile_year.copy()
        if "ano" in df.columns:
            df["_ano"] = pd.to_numeric(df["ano"], errors="coerce")
            if df["_ano"].notna().any():
                df = df.loc[df["_ano"].eq(df["_ano"].max())]
        weight_col = next((c for c in ["eleitorado", "votos", "peso", "qtd"] if c in df.columns), "")
        if "dimensao_perfil" in df.columns and "valor_perfil" in df.columns:
            for dim, g in df.groupby("dimensao_perfil", dropna=False):
                if "biometr" in str(dim).lower():
                    continue
                g = g.copy()
                if weight_col:
                    g["_peso"] = pd.to_numeric(g[weight_col], errors="coerce").fillna(0)
                    g = g.sort_values("_peso", ascending=False)
                val = meaningful(g.iloc[0].get("valor_perfil", "")) if not g.empty else ""
                if val:
                    traits[str(dim)] = val
    if not traits and personas is not None and not personas.empty:
        row = personas.iloc[0]
        mapping = {
            "faixa_etaria": "perfil_faixa_etaria_dominante",
            "genero": "perfil_genero_dominante",
            "instrucao": "perfil_instrucao_dominante",
            "estado_civil": "perfil_estado_civil_dominante",
            "raca_cor": "perfil_raca_cor_dominante",
        }
        traits = {k: meaningful(row.get(v, "")) for k, v in mapping.items() if meaningful(row.get(v, ""))}

    if traits:
        sentence = "; ".join(f"{label.replace('_', ' ')}: {value}" for label, value in traits.items())
        chips = list(traits.values())[:8]
    else:
        sentence = "Perfil medio ainda nao foi calculado neste run."
        chips = []
    render_cards([html_card(title, sentence, chips, "Perfil agregado: nao representa voto individual declarado.")])


def render_overview(run_dir: Path, data: dict[str, pd.DataFrame]) -> None:
    st.header("Visao geral")
    kpi_row(data["results"], data["municipal"], data["timeline_nacional"], data["cluster_result_personas"])
    st.markdown("")
    average_voter_card(data["perfil_ano"], data["cluster_voter_personas"])
    st.subheader("Mapa do perfil do eleitor por estado")
    render_brazil_map(data)

    c1, c2 = st.columns(2)
    with c1:
        line_chart_by_year(data["timeline_nacional"], ["votos", "eleitorado", "comparecimento_estimado", "abstencao_estimado"], "Evolucao nacional")
    with c2:
        bar_chart(data["perfil_partido"], "partido", "votos_partido", "Quem vota por partido politico", limit=15)

    if not data["perfil_candidato"].empty:
        st.subheader("Quem vota por candidato")
        work = data["perfil_candidato"].copy()
        sort_col = "votos_candidato" if "votos_candidato" in work.columns else "votos"
        if sort_col in work.columns:
            work["_sort"] = pd.to_numeric(work[sort_col], errors="coerce").fillna(0)
            work = work.sort_values("_sort", ascending=False)
        cards = []
        for _, row in work.head(12).iterrows():
            title = meaningful(row.get("candidato", "")) or meaningful(row.get("entidade", "")) or "Candidato"
            body = meaningful(row.get("pessoa_do_candidato", "")) or profile_summary_text(row)
            chips = [meaningful(row.get("ano", "")), meaningful(row.get("partido", "")), meaningful(row.get("cargo", ""))]
            cards.append(html_card(title, body, chips, f"Votos: {fmt_int(row.get(sort_col, 0))}"))
        render_cards(cards)

    top_fed = data["top10_perfis"]
    if not top_fed.empty and "nivel" in top_fed.columns:
        render_top10_profile_cards(top_fed.loc[top_fed["nivel"].astype(str).str.lower().eq("brasil")], "Top 10 perfis na federacao", limit=10)

    answers = data["respostas"]
    if answers is not None and not answers.empty:
        cards = []
        for _, row in answers.head(8).iterrows():
            question = meaningful(row.get("pergunta", "")) or "Pergunta eleitoral"
            answer = meaningful(row.get("resposta", "")) or "Sem resposta calculada."
            cards.append(html_card(question, shorten(answer, 360), [meaningful(row.get("base_de_evidencia", ""))]))
        st.subheader("Respostas eleitorais")
        render_cards(cards)

    st.markdown("**Relatorios HTML/PDF do mesmo run**")
    report_cards = report_link_cards(run_dir)
    render_cards(report_cards)


def render_states(data: dict[str, pd.DataFrame]) -> None:
    st.header("Estados")
    timeline_uf = data["timeline_uf"]
    if timeline_uf.empty:
        st.info("Sem timeline por UF neste run.")
        return
    timeline_uf = normalize_year(timeline_uf)
    ufs = sorted([x for x in timeline_uf.get("uf", pd.Series(dtype=str)).map(meaningful).unique() if x])
    selected = st.selectbox("Escolha o estado", ["Brasil agregado"] + ufs)
    if selected != "Brasil agregado":
        work = timeline_uf.loc[timeline_uf["uf"].astype(str).eq(selected)].copy()
    else:
        work = timeline_uf.copy()

    c1, c2 = st.columns(2)
    with c1:
        line_chart_by_year(work, ["votos", "eleitorado", "comparecimento_estimado", "abstencao_estimado"], "Serie do estado")
    with c2:
        metric = "votos" if "votos" in timeline_uf.columns else next((c for c in ["eleitorado", "qtd_setores"] if c in timeline_uf.columns), "")
        if metric:
            bar_chart(timeline_uf, "uf", metric, "Ranking por UF", limit=27)

    cards = []
    latest = work.copy()
    if "ano" in latest.columns:
        latest["_ano"] = pd.to_numeric(latest["ano"], errors="coerce")
        if latest["_ano"].notna().any():
            latest = latest.loc[latest["_ano"].eq(latest["_ano"].max())]
    cards.append(html_card("Resumo do estado", state_sentence(latest), [selected]))
    render_cards(cards)

    top = data["top10_perfis"]
    if not top.empty and "nivel" in top.columns:
        state_top = top.loc[top["nivel"].astype(str).str.lower().eq("estado")].copy()
        if selected != "Brasil agregado" and "uf" in state_top.columns:
            state_top = state_top.loc[state_top["uf"].astype(str).eq(selected)]
        render_top10_profile_cards(state_top, "Top 10 perfis por estado", limit=10 if selected != "Brasil agregado" else 24)

    comp = data["comparativo_perfil"]
    if not comp.empty and {"nivel", "uf"}.issubset(comp.columns):
        state_comp = comp.loc[comp["nivel"].astype(str).str.lower().eq("estado")].copy()
        if selected != "Brasil agregado":
            state_comp = state_comp.loc[state_comp["uf"].astype(str).eq(selected)]
        render_top10_profile_cards(state_comp, "Padrao anual do perfil no estado", limit=10 if selected != "Brasil agregado" else 24)


def render_municipalities(data: dict[str, pd.DataFrame]) -> None:
    st.header("Municipios")
    municipal = data["municipal"]
    if municipal.empty:
        st.info("Sem retrato municipal neste run.")
        return
    municipal = municipal.copy()
    municipal["municipio_label"] = municipal.apply(municipality_label, axis=1)
    labels = sorted([x for x in municipal["municipio_label"].map(meaningful).unique() if x])
    if not labels:
        st.info("A base municipal existe, mas nao trouxe nome/codigo de municipio para consulta.")
        return
    selected = st.selectbox("Escolha um municipio", labels)
    selected_df = municipal.loc[municipal["municipio_label"].eq(selected)].copy()

    cards = []
    if not selected_df.empty:
        row = aggregate_municipality(selected_df)
        cards.append(html_card("Municipio selecionado", municipality_sentence(row), municipality_chips(row)))
    render_cards(cards)

    c1, c2 = st.columns(2)
    with c1:
        bar_chart(municipal, "municipio_label", "votos", "Municipios por votos", limit=20)
    with c2:
        if "eleitorado" in municipal.columns:
            bar_chart(municipal, "municipio_label", "eleitorado", "Municipios por eleitorado", limit=20)

    timeline_municipal = data["timeline_municipal"]
    if not timeline_municipal.empty and selected:
        code = meaningful(selected_df.iloc[0].get("cd_municipio", "")) if not selected_df.empty else ""
        name = meaningful(selected_df.iloc[0].get("nm_municipio", "")) if not selected_df.empty else ""
        mask = pd.Series(False, index=timeline_municipal.index)
        if code and "cd_municipio" in timeline_municipal.columns:
            mask = mask | timeline_municipal["cd_municipio"].astype(str).eq(code)
        if name and "nm_municipio" in timeline_municipal.columns:
            mask = mask | timeline_municipal["nm_municipio"].astype(str).eq(name)
        work = timeline_municipal.loc[mask].copy()
        if not work.empty:
            line_chart_by_year(work, ["votos", "eleitorado", "comparecimento_estimado", "abstencao_estimado"], "Evolucao do municipio")

    top = data["top10_perfis"]
    if not top.empty and "nivel" in top.columns and not selected_df.empty:
        code = meaningful(selected_df.iloc[0].get("cd_municipio", ""))
        name = meaningful(selected_df.iloc[0].get("nm_municipio", ""))
        mun_top = top.loc[top["nivel"].astype(str).str.lower().eq("municipio")].copy()
        mask = pd.Series(False, index=mun_top.index)
        if code and "cd_municipio" in mun_top.columns:
            mask = mask | mun_top["cd_municipio"].astype(str).eq(code)
        if name and "nm_municipio" in mun_top.columns:
            mask = mask | mun_top["nm_municipio"].astype(str).eq(name)
        render_top10_profile_cards(mun_top.loc[mask], "Top 10 perfis do municipio", limit=10)


def render_sections(data: dict[str, pd.DataFrame]) -> None:
    st.header("Secoes eleitorais")
    decisive = data["sim_decisive"]
    winners = data["vencedor_secao"]
    source = decisive if not decisive.empty else winners
    if source.empty:
        st.info("Sem base de secoes em preview. Rode a simulacao ou gere vencedor_por_secao.")
        return

    source = source.copy()
    source["municipio_label"] = source.apply(municipality_label, axis=1)
    labels = ["Todas"] + sorted([x for x in source["municipio_label"].map(meaningful).unique() if x])
    selected = st.selectbox("Filtrar secoes por municipio", labels)
    if selected != "Todas":
        source = source.loc[source["municipio_label"].eq(selected)].copy()

    cards = []
    sort_col = "indice_decisivo" if "indice_decisivo" in source.columns else "votos"
    if sort_col in source.columns:
        source["_rank_metric"] = pd.to_numeric(source[sort_col], errors="coerce").fillna(0)
        source = source.sort_values("_rank_metric", ascending=False)
    for _, row in source.head(16).iterrows():
        title = "Secao " + " / ".join(x for x in [meaningful(row.get("zona", "")), meaningful(row.get("secao", ""))] if x)
        leader = meaningful(row.get("lider_pred", "")) or meaningful(row.get("entidade", "")) or meaningful(row.get("candidato", "")) or "sem lider calculado"
        second = meaningful(row.get("segundo_pred", ""))
        body = f"Lider: {leader}" + (f". Segundo: {second}" if second else "")
        chips = [meaningful(row.get("uf", "")), meaningful(row.get("nm_municipio", "")), meaningful(row.get("cargo", "")), meaningful(row.get("turno", ""))]
        cards.append(html_card(title or "Secao", body, chips, section_note(row)))
    render_cards(cards)

    if "indice_decisivo" in source.columns:
        source["secao_label"] = source.apply(lambda r: " / ".join(x for x in [meaningful(r.get("nm_municipio", "")), meaningful(r.get("zona", "")), meaningful(r.get("secao", ""))] if x), axis=1)
        bar_chart(source, "secao_label", "indice_decisivo", "Secoes/municipios mais decisivos", limit=20)

    extra_tabs = st.tabs(["Perfil do candidato", "Resultado + eleitorado"])
    with extra_tabs[0]:
        df = data["perfil_do_candidato"]
        if df.empty:
            st.info("Sem perfil de candidato correlacionado neste run.")
        else:
            cards = []
            for _, row in df.head(24).iterrows():
                title = meaningful(row.get("candidato", "")) or meaningful(row.get("entidade", "")) or "Candidato"
                body = profile_summary_text(row)
                chips = [meaningful(row.get("ano", "")), meaningful(row.get("partido", "")), meaningful(row.get("cargo", ""))]
                cards.append(html_card(title, body, chips, f"Votos: {fmt_int(row.get('votos', row.get('votos_candidato', 0)))}"))
            render_cards(cards)
    with extra_tabs[1]:
        df = data["resultado_eleitorado"]
        if df.empty:
            st.info("Sem resultado + eleitorado correlacionado neste run.")
        else:
            cards = []
            for _, row in df.head(24).iterrows():
                title = meaningful(row.get("entidade", "")) or meaningful(row.get("partido", "")) or meaningful(row.get("candidato", "")) or "Resultado correlacionado"
                body = profile_summary_text(row)
                chips = [meaningful(row.get("ano", "")), meaningful(row.get("tipo_entidade", "")), meaningful(row.get("perfil_combinado", ""))]
                cards.append(html_card(title, body, chips, f"Votos: {fmt_int(row.get('votos', row.get('votos_perfil_entidade', 0)))}"))
            render_cards(cards)


def render_clusters(data: dict[str, pd.DataFrame]) -> None:
    st.header("Clusters")
    left, right = st.tabs(["Somente eleitores", "Eleitores + resultado"])
    with left:
        render_cluster_block(
            "Clusters somente com dados discretos dos eleitores",
            data["cluster_voter_personas"],
            data["cluster_voter_year_region"],
            data["cluster_voter_discriminants"],
            pd.DataFrame(),
        )
    with right:
        render_cluster_block(
            "Clusters com eleitor + resultado",
            data["cluster_result_personas"],
            data["cluster_result_year_region"],
            data["cluster_result_discriminants"],
            data["cluster_result_prediction"],
        )


def render_cluster_block(title: str, personas: pd.DataFrame, year_region: pd.DataFrame, discriminants: pd.DataFrame, prediction: pd.DataFrame) -> None:
    st.subheader(title)
    personas = clean_persona_df(personas)
    if personas.empty:
        st.info("Sem clusters com perfil discreto util neste run.")
        return

    cluster_col = "cluster_global_discriminado"
    labels = [str(x) for x in personas[cluster_col].dropna().astype(str).unique()] if cluster_col in personas.columns else []
    if not labels:
        st.info("Os clusters existem, mas nao trouxeram identificador de cluster para consulta.")
        return
    selected = st.selectbox("Escolha um cluster para consultar", labels, key=title)
    if selected and cluster_col in personas.columns:
        selected_df = personas.loc[personas[cluster_col].astype(str).eq(selected)]
    else:
        selected_df = personas.head(1)

    cards = []
    for _, row in selected_df.head(1).iterrows():
        cards.append(cluster_card(row, detailed=True))
    for _, row in personas.loc[~personas.index.isin(selected_df.index)].head(10).iterrows():
        cards.append(cluster_card(row))
    render_cards(cards)

    c1, c2 = st.columns(2)
    with c1:
        for field, label in [
            ("perfil_faixa_etaria_dominante", "Clusters por faixa etaria"),
            ("perfil_genero_dominante", "Clusters por genero"),
            ("perfil_instrucao_dominante", "Clusters por escolaridade"),
        ]:
            if field in personas.columns:
                tmp = personas.copy()
                tmp["qtd"] = 1
                bar_chart(tmp, field, "qtd", label, limit=12)
                break
    with c2:
        if year_region is not None and not year_region.empty:
            metric = "qtd_setores" if "qtd_setores" in year_region.columns else "votos_total"
            if "regiao" in year_region.columns and metric in year_region.columns:
                bar_chart(year_region, "regiao", metric, "Clusters por regiao", limit=10)

    if discriminants is not None and not discriminants.empty:
        disc = discriminants.copy()
        disc["valor_legivel"] = disc.apply(
            lambda r: f"{meaningful(r.get('campo_discriminado_legivel', ''))}: {meaningful(r.get('valor_discriminado', ''))}",
            axis=1,
        )
        metric = "qtd_no_cluster" if "qtd_no_cluster" in disc.columns else "share_no_cluster"
        bar_chart(disc, "valor_legivel", metric, "Valores discretos que definem os clusters", limit=20)

    if prediction is not None and not prediction.empty:
        pred = prediction.copy()
        if "vencedor_setor" not in pred.columns:
            pred = pd.DataFrame()
        else:
            pred = pred.loc[pred["vencedor_setor"].map(meaningful).ne("")]
        if not pred.empty:
            pred["cluster_entidade"] = pred.apply(
                lambda r: f"Cluster {meaningful(r.get('cluster_global_discriminado', ''))}: {meaningful(r.get('vencedor_setor', ''))}",
                axis=1,
            )
            metric = "share_previsto_2026" if "share_previsto_2026" in pred.columns else "share_previsto_2026_raw"
            bar_chart(pred, "cluster_entidade", metric, "Predicao 2026 por cluster", limit=18)


def clean_persona_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "pessoa_do_cluster" in out.columns:
        out = out.loc[out["pessoa_do_cluster"].map(meaningful).ne("")]
        out = out.loc[~out["pessoa_do_cluster"].astype(str).str.lower().str.contains("sem valor", na=False)]
    for field in PROFILE_FIELDS:
        if field in out.columns:
            out[field] = out[field].map(meaningful)
    if "qtd_setores" in out.columns:
        out["_qtd_sort"] = pd.to_numeric(out["qtd_setores"], errors="coerce").fillna(0)
        out = out.sort_values("_qtd_sort", ascending=False)
    return out


def cluster_card(row: pd.Series, detailed: bool = False) -> str:
    age = meaningful(row.get("perfil_faixa_etaria_dominante", ""))
    gender = meaningful(row.get("perfil_genero_dominante", ""))
    education = meaningful(row.get("perfil_instrucao_dominante", ""))
    region = meaningful(row.get("regiao_dominante", ""))
    uf = meaningful(row.get("uf_dominante", ""))
    party = meaningful(row.get("partido_vencedor_setor_dominante", ""))
    winner = meaningful(row.get("entidade_prevista_2026", "")) or meaningful(row.get("vencedor_setor_dominante", ""))
    title_bits = [age, gender, education]
    title = " | ".join([x for x in title_bits if x]) or f"Cluster {meaningful(row.get('cluster_global_discriminado', ''))}"
    body = meaningful(row.get("pessoa_do_cluster", "")) or "Perfil discreto insuficiente para descrever este cluster."
    chips = [age, gender, education, meaningful(row.get("perfil_estado_civil_dominante", "")), meaningful(row.get("perfil_raca_cor_dominante", "")), region, uf]
    if party:
        chips.append(f"Partido: {party}")
    if winner:
        chips.append(f"Tendencia: {winner}")
    note = "Consulta detalhada do cluster selecionado." if detailed else ""
    return html_card(title, body, chips, note)


def render_party_simulation_cards(df: pd.DataFrame, title: str, limit: int = 18) -> None:
    if df is None or df.empty:
        st.info(f"{title}: sem simulacao partidaria neste run.")
        return
    work = df.copy()
    if "cenario" in work.columns and work["cenario"].astype(str).eq("base").any():
        work = work.loc[work["cenario"].astype(str).eq("base")].copy()
    work["share_pred_2026"] = pd.to_numeric(work.get("share_pred_2026"), errors="coerce")
    work = work.loc[work.get("partido", pd.Series(dtype=str)).map(meaningful).ne("")]
    work = work.sort_values("share_pred_2026", ascending=False)
    st.subheader(title)
    chart = work.copy()
    chart["votos_pred_2026"] = pd.to_numeric(chart.get("votos_pred_2026"), errors="coerce").fillna(0)
    if chart["votos_pred_2026"].sum() > 0:
        by_party = chart.groupby("partido", dropna=False)["votos_pred_2026"].sum().sort_values(ascending=False).head(limit)
        by_party = by_party / by_party.sum()
    else:
        by_party = chart.groupby("partido", dropna=False)["share_pred_2026"].mean().sort_values(ascending=False).head(limit)
    by_party = by_party.loc[by_party.index.map(meaningful).ne("")]
    if by_party.index.nunique() > 1:
        st.subheader(f"{title} - percentual por partido")
        st.bar_chart(by_party)
    else:
        st.caption(f"{title}: grafico omitido porque ha apenas um partido util.")
    cards = []
    for _, row in work.head(limit).iterrows():
        party = meaningful(row.get("partido", "")) or "Partido"
        body = meaningful(row.get("perfil_eleitor_2026", "")) or "Perfil de eleitor ainda insuficiente neste recorte."
        chips = [
            fmt_pct(row.get("share_pred_2026")),
            meaningful(row.get("uf", "")),
            meaningful(row.get("nm_municipio", "")),
            meaningful(row.get("tendencia_partido", "")),
            meaningful(row.get("forca_correlacao_historica", "")),
        ]
        note = meaningful(row.get("justificativa_previsao_partido_2026", "")) or meaningful(row.get("justificativa_correlacao", ""))
        cards.append(html_card(party, body, chips, note))
    render_cards(cards)


def render_simulation(data: dict[str, pd.DataFrame]) -> None:
    st.header("Simulacao 2026")
    party_br = data["sim_partidos_brasil"]
    party_uf = data["sim_partidos_estados"]
    party_mun = data["sim_partidos_municipios"]
    if not party_br.empty or not party_uf.empty or not party_mun.empty:
        tab_br, tab_uf, tab_mun = st.tabs(["Partidos - Brasil", "Partidos - estados", "Partidos - municipios"])
        with tab_br:
            render_party_simulation_cards(party_br, "Cenario partidario 2026 no Brasil", limit=20)
        with tab_uf:
            if party_uf.empty:
                st.info("Sem simulacao partidaria por estado.")
            else:
                ufs = sorted([x for x in party_uf.get("uf", pd.Series(dtype=str)).map(meaningful).unique() if x])
                selected_uf = st.selectbox("Estado para simulacao partidaria", ["Todos"] + ufs)
                work = party_uf if selected_uf == "Todos" else party_uf.loc[party_uf["uf"].astype(str).eq(selected_uf)].copy()
                render_party_simulation_cards(work, "Cenario partidario 2026 por estado", limit=24 if selected_uf == "Todos" else 20)
        with tab_mun:
            if party_mun.empty:
                st.info("Sem simulacao partidaria por municipio.")
            else:
                mun = party_mun.copy()
                mun["municipio_label"] = mun.apply(municipality_label, axis=1)
                labels = sorted([x for x in mun["municipio_label"].map(meaningful).unique() if x])
                selected_mun = st.selectbox("Municipio para simulacao partidaria", labels[:5000] if labels else [""])
                work = mun.loc[mun["municipio_label"].eq(selected_mun)].copy() if selected_mun else mun.head(0)
                render_party_simulation_cards(work, "Cenario partidario 2026 por municipio", limit=20)
        st.divider()

    nacional = data["sim_nacional"]
    mc = data["sim_monte_carlo"]
    decisive = data["sim_decisive"]
    if nacional.empty and mc.empty:
        st.info("A simulacao ainda nao foi gerada neste run.")
        return

    if not nacional.empty:
        df = nacional.copy()
        if "cenario" in df.columns and df["cenario"].astype(str).eq("base").any():
            df = df.loc[df["cenario"].astype(str).eq("base")]
        metric = "share_nacional_pred_2026" if "share_nacional_pred_2026" in df.columns else "votos_pred_2026"
        cards = []
        sort = pd.to_numeric(df.get(metric), errors="coerce").fillna(0)
        df = df.assign(_sort=sort).sort_values("_sort", ascending=False)
        if "entidade" not in df.columns:
            df = pd.DataFrame()
        if not df.empty:
            valid_entities = df.loc[df["entidade"].map(meaningful).ne("")]
            for _, row in valid_entities.head(12).iterrows():
                entity = meaningful(row.get("entidade", "")) or "Entidade"
                body = f"Cenario base: {fmt_pct(row.get(metric)) if 'share' in metric else fmt_int(row.get(metric))}"
                chips = [meaningful(row.get("cargo", "")), meaningful(row.get("turno", "")), meaningful(row.get("cenario", ""))]
                cards.append(html_card(entity, body, chips))
        render_cards(cards)
        bar_chart(df, "entidade", metric, "Cenario nacional 2026", limit=15)

    c1, c2 = st.columns(2)
    with c1:
        if not mc.empty:
            metric = "share_medio" if "share_medio" in mc.columns else "share_p50"
            bar_chart(mc, "entidade", metric, "Monte Carlo por entidade", limit=15)
    with c2:
        if not decisive.empty:
            decisive["local"] = decisive.apply(lambda r: " / ".join(x for x in [meaningful(r.get("uf", "")), meaningful(r.get("nm_municipio", "")), meaningful(r.get("zona", "")), meaningful(r.get("secao", ""))] if x), axis=1)
            metric = "indice_decisivo" if "indice_decisivo" in decisive.columns else "margem_pred"
            bar_chart(decisive, "local", metric, "Locais decisivos", limit=15)


def render_files(run_dir: Path, data: dict[str, pd.DataFrame]) -> None:
    st.header("Arquivos e analises individuais")
    results = data["results"]
    if results.empty:
        st.info("Nao encontrei logs/resultados_individuais.json neste run.")
    else:
        text = st.text_input("Buscar arquivo", "")
        work = results.copy()
        if text:
            mask = work.astype(str).apply(lambda col: col.str.contains(text, case=False, na=False)).any(axis=1)
            work = work.loc[mask]
        cards = []
        for _, row in work.head(40).iterrows():
            rel = meaningful(row.get("arquivo_relativo", "")) or meaningful(row.get("arquivo", "")) or meaningful(row.get("path", ""))
            status = meaningful(row.get("status", ""))
            body = f"Status: {status or 'sem status'}"
            if meaningful(row.get("html", "")):
                body += f". HTML: {row.get('html')}"
            chips = [meaningful(row.get("dominio_documento", "")), meaningful(row.get("assunto_documento", ""))]
            cards.append(html_card(shorten(rel, 80) or "Arquivo", body, chips))
        render_cards(cards)

    st.subheader("Artefatos do run")
    tables = existing_tables(run_dir)
    cards = [html_card(name, str(path.relative_to(run_dir)), [path.suffix.lstrip(".").upper()]) for name, path in sorted(tables.items())]
    render_cards(cards)


def render_table_explorer(run_dir: Path) -> None:
    st.header("Consulta de tabelas")
    st.caption("As tabelas ficam atras desta aba. O dashboard principal usa cards e graficos; aqui voce consulta previews.")
    tables = existing_tables(run_dir)
    if not tables:
        st.info("Nenhuma tabela conhecida encontrada neste run.")
        return
    selected = st.selectbox("Tabela", sorted(tables.keys()))
    path = tables[selected]
    max_rows = st.slider("Linhas para preview", 50, 5000, 500, step=50)
    df = read_table_path(path, (), max_rows)
    st.markdown(f"<div class='path-box'>{escape_html(path)}</div>", unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, hide_index=True)


def load_data(run_dir: Path, preview_rows: int) -> dict[str, pd.DataFrame]:
    data = {name: read_table(run_dir, name, max_rows=preview_rows) for name in TABLE_CANDIDATES}
    data["results"] = load_json_table(run_dir / "logs" / "resultados_individuais.json")
    return data


def report_link_cards(run_dir: Path) -> list[str]:
    reports = [
        ("Indice HTML", run_dir / "index.html"),
        ("Relatorio global HTML", run_dir / "global" / "relatorio_global.html"),
        ("Relatorio simulacao HTML", run_dir / "preditivo_2026" / "relatorio_simulacao.html"),
        ("Relatorio executivo HTML", run_dir / "relatorio_executivo" / "relatorio_eleicoes_simulacao.html"),
        ("Relatorio executivo PDF", run_dir / "relatorio_executivo" / "relatorio_eleicoes_simulacao.pdf"),
    ]
    cards = []
    for title, path in reports:
        if path.exists():
            cards.append(html_card(title, str(path), [path.suffix.lstrip(".").upper()]))
    return cards


def municipality_label(row: pd.Series) -> str:
    uf = meaningful(row.get("uf", ""))
    name = meaningful(row.get("nm_municipio", ""))
    code = meaningful(row.get("cd_municipio", "")) or meaningful(row.get("codigo_municipio", ""))
    base = name or code or "Municipio"
    return f"{uf} - {base}" if uf else base


def aggregate_municipality(df: pd.DataFrame) -> pd.Series:
    row = df.iloc[0].copy()
    for col in ["votos", "eleitorado", "comparecimento_estimado", "abstencao_estimado"]:
        if col in df.columns:
            row[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).sum()
    return row


def municipality_sentence(row: pd.Series) -> str:
    parts = []
    if meaningful(row.get("nm_municipio", "")):
        parts.append(meaningful(row.get("nm_municipio", "")))
    if meaningful(row.get("uf", "")):
        parts.append(meaningful(row.get("uf", "")))
    intro = " / ".join(parts) or "Municipio"
    return (
        f"{intro}. Votos: {fmt_int(row.get('votos', 0))}. "
        f"Eleitorado: {fmt_int(row.get('eleitorado', row.get('eleitorado_setor', 0)))}. "
        f"Comparecimento: {fmt_int(row.get('comparecimento_estimado', row.get('comparecimento_setor', 0)))}. "
        f"Abstencao: {fmt_int(row.get('abstencao_estimado', row.get('abstencao_setor', 0)))}."
    )


def municipality_chips(row: pd.Series) -> list[str]:
    return [
        meaningful(row.get("uf", "")),
        meaningful(row.get("cd_municipio", "")),
        meaningful(row.get("regiao", "")),
        meaningful(row.get("cargo", "")),
        meaningful(row.get("turno", "")),
    ]


def state_sentence(df: pd.DataFrame) -> str:
    if df.empty:
        return "Sem dados para o estado selecionado."
    return (
        f"Votos: {fmt_int(metric_sum(df, 'votos', 'votos_total'))}. "
        f"Eleitorado: {fmt_int(metric_sum(df, 'eleitorado'))}. "
        f"Comparecimento: {fmt_int(metric_sum(df, 'comparecimento_estimado', 'comparecimento'))}. "
        f"Abstencao: {fmt_int(metric_sum(df, 'abstencao_estimado', 'abstencao'))}."
    )


def section_note(row: pd.Series) -> str:
    bits = []
    if "indice_decisivo" in row.index:
        bits.append(f"indice decisivo {fmt_pct(row.get('indice_decisivo')) if pd.to_numeric(row.get('indice_decisivo'), errors='coerce') <= 1 else meaningful(row.get('indice_decisivo'))}")
    if "margem_pred" in row.index:
        bits.append(f"margem {fmt_pct(row.get('margem_pred'))}")
    return "; ".join(bits)


def shorten(text: Any, limit: int = 180) -> str:
    value = meaningful(text)
    if len(value) <= limit:
        return value
    return value[: max(limit - 1, 1)].rstrip() + "..."


def main() -> None:
    args = parse_args()
    st.set_page_config(page_title="Dashboard Global Eleitoral", layout="wide")
    inject_css()

    runs = list_runs()
    default_run = resolve_run_path(args.run)
    options = [str(p) for p in runs]
    if str(default_run) not in options:
        options.insert(0, str(default_run))

    with st.sidebar:
        st.title("Dashboard eleitoral")
        selected_run = st.selectbox("Run", options, index=options.index(str(default_run)) if str(default_run) in options else 0)
        preview_rows = st.slider("Linhas maximas por leitura", 500, 20000, int(args.preview_rows), step=500)
        st.caption("Leitura preferencial em Parquet. CSV entra apenas como fallback ou preview.")

    run_dir = Path(selected_run)
    st.title("Dashboard Global Eleitoral")
    st.markdown(f"<div class='path-box'>{escape_html(run_dir)}</div>", unsafe_allow_html=True)

    if not run_dir.exists():
        st.error("Pasta do run nao existe. Rode o pipeline primeiro ou informe --run corretamente.")
        st.code("streamlit run scripts/dashboard_streamlit_eleitoral.py -- --run resultados/NOME_DO_RUN", language="bash")
        return

    data = load_data(run_dir, preview_rows)

    tabs = st.tabs(["Brasil", "Estados", "Municipios", "Secoes", "Clusters", "Simulacao 2026", "Arquivos", "Consulta"])
    with tabs[0]:
        render_overview(run_dir, data)
    with tabs[1]:
        render_states(data)
    with tabs[2]:
        render_municipalities(data)
    with tabs[3]:
        render_sections(data)
    with tabs[4]:
        render_clusters(data)
    with tabs[5]:
        render_simulation(data)
    with tabs[6]:
        render_files(run_dir, data)
    with tabs[7]:
        render_table_explorer(run_dir)


if __name__ == "__main__":
    main()
