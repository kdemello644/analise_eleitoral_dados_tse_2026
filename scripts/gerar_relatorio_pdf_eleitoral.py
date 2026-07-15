from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import duckdb
    import pandas as pd
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
except ModuleNotFoundError as exc:
    missing = exc.name or "dependencia"
    print(
        f"Dependencia ausente: {missing}\n"
        "Instale com:\n"
        "  python3 -m pip install -r scripts/pipeline_eleitoral_json/requirements.txt\n",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc

try:
    import polars as pl
except ModuleNotFoundError:  # pragma: no cover - ambiente sem Polars usa pandas/DuckDB
    pl = None  # type: ignore[assignment]

try:
    from parquet_query_polars_eleitoral import (
        ANALYSIS_MODES,
        MODE_LABELS,
        PolarsStore,
        modalidade_allows,
        modalidade_info,
        normalize_modalidade,
        records as polars_records,
        polars_available,
    )
except Exception:  # pragma: no cover - fallback quando polars nao esta instalado
    ANALYSIS_MODES = ["completa", "estados_brasil", "eleitor", "candidato", "eleitor_partido", "eleitor_candidato_partido"]
    MODE_LABELS = {x: x for x in ANALYSIS_MODES}
    PolarsStore = None  # type: ignore[assignment]

    def normalize_modalidade(value: Any) -> str:
        text = str(value or "completa").strip().lower()
        return text if text in ANALYSIS_MODES else "completa"

    def modalidade_allows(value: Any, feature: str) -> bool:
        mode = normalize_modalidade(value)
        if mode == "completa":
            return True
        if mode == "estados_brasil":
            return feature in {"brasil", "estado", "perfil", "partido", "simulacao"}
        if mode == "eleitor":
            return feature in {"brasil", "estado", "municipio", "perfil"}
        if mode == "candidato":
            return feature in {"brasil", "estado", "municipio", "candidato"}
        if mode == "eleitor_partido":
            return feature in {"brasil", "estado", "municipio", "perfil", "partido", "simulacao"}
        if mode == "eleitor_candidato_partido":
            return feature in {"brasil", "estado", "municipio", "perfil", "partido", "candidato", "simulacao"}
        return False

    def modalidade_info(value: Any) -> dict[str, Any]:
        mode = normalize_modalidade(value)
        return {"modalidade": mode, "label": MODE_LABELS.get(mode, mode)}

    def polars_records(_: Any) -> list[dict[str, Any]]:
        return []

    def polars_available() -> bool:
        return False


TABLE_CANDIDATES: dict[str, list[str]] = {
    "municipal": ["ouro/municipal/resumo", "ouro/retrato_municipal", "ouro/retrato_municipal.parquet", "global/parquet/retrato_municipal_global.parquet"],
    "timeline_nacional": ["ouro/brasil/resumo", "ouro/timeline_nacional.parquet", "global/parquet/timeline_nacional.parquet"],
    "timeline_uf": ["ouro/estadual/resumo", "ouro/timeline_uf", "ouro/timeline_uf.parquet", "global/parquet/timeline_uf.parquet"],
    "timeline_municipal": ["ouro/municipal/resumo", "ouro/timeline_municipal", "ouro/timeline_municipal.parquet", "global/parquet/timeline_municipal.parquet"],
    "perfil_ano": ["ouro/perfil_eleitor_por_ano", "ouro/brasil/perfil_eleitor", "ouro/estadual/perfil_eleitor", "ouro/municipal/perfil_eleitor", "ouro/perfil_eleitor_por_ano.parquet"],
    "contagem_colunas_perfil_eleitor": ["ouro/brasil/contagem_colunas_perfil_eleitor", "ouro/estadual/contagem_colunas_perfil_eleitor", "ouro/municipal/contagem_colunas_perfil_eleitor"],
    "perfil_partido": ["ouro/brasil/perfil_partido", "ouro/estadual/perfil_partido", "ouro/municipal/perfil_partido", "ouro/perfil_eleitor_por_partido", "ouro/perfil_eleitor_por_partido.parquet"],
    "contagem_colunas_perfil_partido": ["ouro/brasil/contagem_colunas_perfil_partido", "ouro/estadual/contagem_colunas_perfil_partido", "ouro/municipal/contagem_colunas_perfil_partido"],
    "perfil_candidato": ["ouro/brasil/perfil_candidato", "ouro/estadual/perfil_candidato", "ouro/municipal/perfil_candidato", "ouro/perfil_eleitor_por_candidato", "ouro/perfil_eleitor_por_candidato.parquet"],
    "contagem_colunas_perfil_candidato": ["ouro/brasil/contagem_colunas_perfil_candidato", "ouro/estadual/contagem_colunas_perfil_candidato", "ouro/municipal/contagem_colunas_perfil_candidato"],
    "resultado_partido": ["ouro/brasil/resultado_partido", "ouro/estadual/resultado_partido", "ouro/municipal/resultado_partido"],
    "resultado_candidato": ["ouro/brasil/resultado_candidato", "ouro/estadual/resultado_candidato", "ouro/municipal/resultado_candidato"],
    "top10_perfis": ["ouro/top10_perfis_federacao_estado_municipio", "ouro/brasil/perfil_eleitor", "ouro/estadual/perfil_eleitor", "ouro/municipal/perfil_eleitor", "ouro/top10_perfis_federacao_estado_municipio.parquet"],
    "vencedor_secao": ["ouro/resultados_vencedores_secao", "ouro/resultados_vencedores_secao.parquet"],
    "resultado_eleitorado": ["ouro/resultado_eleitorado_por_secao", "ouro/resultado_eleitorado_por_secao.parquet"],
    "base_gold": ["ouro/base_gold_global", "ouro/base_gold_global.parquet"],
    "perfil_candidatos": ["ouro/perfil_candidatos", "ouro/perfil_candidatos.parquet"],
    "sim_partidos_brasil": ["preditivo_2026/parquet/partidos_2026_brasil.parquet", "preditivo_2026/tabelas/partidos_2026_brasil.csv"],
    "sim_partidos_estados": ["preditivo_2026/parquet/partidos_2026_estados.parquet", "preditivo_2026/tabelas/partidos_2026_estados.csv"],
    "sim_partidos_municipios": ["preditivo_2026/parquet/partidos_2026_municipios.parquet", "preditivo_2026/tabelas/partidos_2026_municipios.csv"],
    "sim_partidos_correlacao": ["preditivo_2026/parquet/partidos_2026_correlacao_historica.parquet", "preditivo_2026/tabelas/partidos_2026_correlacao_historica.csv"],
    "cluster_voter_personas": [
        "ouro/brasil/contagem_colunas_clusters_eleitores",
        "ouro/estadual/contagem_colunas_clusters_eleitores",
        "ouro/municipal/contagem_colunas_clusters_eleitores",
        "ouro/brasil/clusters_eleitores",
        "ouro/estadual/clusters_eleitores",
        "ouro/municipal/clusters_eleitores",
        "global/correlacao_codigos/clusters/parquet/clusters_eleitores_personas.parquet",
        "global/correlacao_codigos/clusters/tabelas/clusters_eleitores_personas.csv",
    ],
    "cluster_result_personas": [
        "ouro/brasil/contagem_colunas_clusters_eleitores_resultado",
        "ouro/estadual/contagem_colunas_clusters_eleitores_resultado",
        "ouro/municipal/contagem_colunas_clusters_eleitores_resultado",
        "ouro/brasil/clusters_eleitores_resultado",
        "ouro/estadual/clusters_eleitores_resultado",
        "ouro/municipal/clusters_eleitores_resultado",
        "global/correlacao_codigos/clusters/parquet/clusters_personas.parquet",
        "global/correlacao_codigos/clusters/tabelas/clusters_personas.csv",
    ],
    "cluster_elbow": [
        "global/correlacao_codigos/clusters/parquet/clusters_cotovelo_k.parquet",
        "global/correlacao_codigos/clusters/tabelas/clusters_cotovelo_k.csv",
    ],
}

NULL_WORDS = {"", "nan", "none", "null", "<na>", "#nulo#", "sem valor", "sem_valor", "geral", "nao informado"}


class PdfRunLogger:
    def __init__(self, out_pdf: Path, verbose: bool = True, log_dir: Path | None = None):
        root = log_dir or out_pdf.parent / "logs"
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = out_pdf.stem
        self.jsonl_path = root / f"{stem}_{stamp}_eventos.jsonl"
        self.text_path = root / f"{stem}_{stamp}.log"
        self.graphs_path = root / f"{stem}_{stamp}_graficos.jsonl"
        self.verbose = bool(verbose)
        logging.basicConfig(
            level=logging.INFO if verbose else logging.WARNING,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(self.text_path, encoding="utf-8"),
            ],
            force=True,
        )

    def event(self, etapa: str, evento: str, **data: Any) -> None:
        payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "epoch": round(time.time(), 3),
            "etapa": etapa,
            "evento": evento,
            **data,
        }
        with self.jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        msg = f"[{etapa}] {evento}"
        if data:
            msg += " | " + " | ".join(f"{k}={v}" for k, v in data.items() if k not in {"sql"})
        logging.info(msg)

    def graph(self, **data: Any) -> None:
        payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "epoch": round(time.time(), 3),
            **data,
        }
        with self.graphs_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        self.event("grafico", str(data.get("evento", "grafico")), **{k: v for k, v in data.items() if k != "evento"})


def lit(value: Any) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def as_posix(path: Path) -> str:
    return path.as_posix()


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


def fmt_int(value: Any) -> str:
    num = pd.to_numeric(value, errors="coerce")
    if pd.isna(num):
        return "0"
    return f"{int(float(num)):,}".replace(",", ".")


def fmt_pct(value: Any) -> str:
    num = pd.to_numeric(value, errors="coerce")
    if pd.isna(num):
        return "0,0%"
    return f"{float(num) * 100:.1f}%".replace(".", ",")


def choose_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    if df is None or df.empty:
        return None
    lower_map = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        found = lower_map.get(candidate.lower())
        if found is not None:
            return str(found)
    return None


def unique_texts(df: pd.DataFrame, col: str | None, limit: int = 8) -> list[str]:
    if df is None or df.empty or not col or col not in df.columns:
        return []
    out: list[str] = []
    for value in df[col].tolist():
        text = meaningful(value)
        if text and text not in out:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def profile_title(row: pd.Series) -> str:
    for col in ["perfil_combinado", "perfil_eleitor", "descricao_perfil", "valor_perfil"]:
        if col in row and meaningful(row.get(col)):
            return meaningful(row.get(col))
    parts = []
    for col in [
        "perfil_faixa_etaria",
        "perfil_genero",
        "perfil_instrucao",
        "perfil_estado_civil",
        "perfil_raca_cor",
        "genero",
        "sexo",
        "faixa_etaria",
        "idade_faixa",
        "grau_instrucao",
        "escolaridade",
        "estado_civil",
        "raca_cor",
    ]:
        if col in row and meaningful(row.get(col)):
            parts.append(meaningful(row.get(col)))
    return " | ".join(parts) if parts else "Perfil eleitoral"


def profile_chips(text: str, max_items: int = 5) -> list[str]:
    raw = str(text or "").replace(";", "|").replace(",", "|").replace(" - ", "|")
    chips: list[str] = []
    for part in raw.split("|"):
        clean = meaningful(part)
        if not clean:
            continue
        if "=" in clean:
            clean = clean.split("=", 1)[1].strip()
        if "->" in clean:
            clean = clean.split("->", 1)[-1].strip()
        if clean and clean not in chips:
            chips.append(clean)
        if len(chips) >= max_items:
            break
    return chips


def sort_by_numeric(df: pd.DataFrame, candidates: list[str], ascending: bool = False) -> pd.DataFrame:
    col = choose_col(df, candidates)
    if not col:
        return df
    work = df.copy()
    work["_sort_value"] = pd.to_numeric(work[col], errors="coerce").fillna(0)
    return work.sort_values("_sort_value", ascending=ascending).drop(columns=["_sort_value"], errors="ignore")


def split_result_status_df(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()
    work = df.copy()
    if "resultado_eleitoral" in work.columns:
        status = work["resultado_eleitoral"].astype(str).str.lower()
        winners = work[status.str.contains("vencedor|eleito|ganhou", regex=True, na=False)].copy()
        losers = work[~work.index.isin(winners.index)].copy()
        return winners, losers
    if "rank_entidade" in work.columns:
        rank = pd.to_numeric(work["rank_entidade"], errors="coerce")
        return work[rank == 1].copy(), work[rank != 1].copy()
    return pd.DataFrame(), work


def sql_meaningful(column: str) -> str:
    nulls = ", ".join(lit(x) for x in sorted(NULL_WORDS))
    expr = f"lower(trim(cast({column} as varchar)))"
    return f"{expr} not in ({nulls}) and {expr} not like '%sem valor%'"


class DuckStore:
    def __init__(self, run_path: Path, threads: int = 4, logger: PdfRunLogger | None = None):
        self.run_path = run_path
        self.logger = logger
        self.con = duckdb.connect(database=":memory:")
        self.con.execute(f"PRAGMA threads={max(1, int(threads or 1))}")
        self.con.execute("PRAGMA preserve_insertion_order=false")
        if self.logger:
            self.logger.event("duckdb", "conexao_iniciada", run=str(run_path), threads=max(1, int(threads or 1)))

    def close(self) -> None:
        self.con.close()

    def path_for(self, key: str) -> Path | None:
        for rel in TABLE_CANDIDATES.get(key, []):
            path = self.run_path / rel
            if path.is_dir():
                if next(path.rglob("*.parquet"), None) is not None:
                    if self.logger:
                        self.logger.event("fonte", "dataset_encontrado", chave=key, caminho=str(path), tipo="diretorio_parquet")
                    return path
            elif path.exists():
                if self.logger:
                    self.logger.event("fonte", "dataset_encontrado", chave=key, caminho=str(path), tipo=path.suffix.lower() or "arquivo")
                return path
        if self.logger:
            self.logger.event("fonte", "dataset_ausente", chave=key)
        return None

    def expr(self, key: str) -> str | None:
        path = self.path_for(key)
        if path is None:
            return None
        if path.is_dir():
            glob = lit(as_posix(path / "**" / "*.parquet"))
            return f"read_parquet({glob}, union_by_name=true, hive_partitioning=true)"
        quoted = lit(as_posix(path))
        if path.suffix.lower() == ".parquet":
            return f"read_parquet({quoted}, union_by_name=true)"
        return f"read_csv_auto({quoted}, delim=';', header=true, all_varchar=true, ignore_errors=true)"

    def query(self, sql: str) -> pd.DataFrame:
        started = time.perf_counter()
        if self.logger:
            compact_sql = " ".join(str(sql).split())
            self.logger.event("consulta", "inicio", sql=compact_sql[:2000])
        try:
            df = self.con.execute(sql).fetchdf()
            if self.logger:
                self.logger.event("consulta", "fim", linhas=len(df), colunas=len(df.columns), duracao_segundos=round(time.perf_counter() - started, 3))
            return df
        except Exception as exc:
            if self.logger:
                self.logger.event("consulta", "erro", erro=str(exc), duracao_segundos=round(time.perf_counter() - started, 3))
            return pd.DataFrame({"erro": [str(exc)]})

    def table(self, key: str, limit: int = 1000) -> pd.DataFrame:
        expr = self.expr(key)
        if expr is None:
            return pd.DataFrame()
        return self.query(f"select * from {expr} limit {int(limit)}")

    def count_rows(self, key: str) -> int:
        expr = self.expr(key)
        if expr is None:
            return 0
        df = self.query(f"select count(*) as n from {expr}")
        return int(pd.to_numeric(df.get("n"), errors="coerce").fillna(0).iloc[0]) if "n" in df else 0

    def distinct(self, key: str, column: str, limit: int = 10000, where: str = "") -> list[str]:
        expr = self.expr(key)
        if expr is None:
            return []
        where_sql = f"where {where}" if where else ""
        df = self.query(
            f"""
            select distinct cast({column} as varchar) as value
            from {expr}
            {where_sql}
            order by 1
            limit {int(limit)}
            """
        )
        if "value" not in df.columns:
            return []
        return [meaningful(x) for x in df["value"].tolist() if meaningful(x)]


class PdfReport:
    def __init__(self, path: Path, title: str, max_pages: int = 1000, logger: PdfRunLogger | None = None):
        self.path = path
        self.title = title
        self.max_pages = max(1, int(max_pages or 1000))
        self.logger = logger
        self.width, self.height = A4
        self.canvas = canvas.Canvas(str(path), pagesize=A4)
        self.page = 0
        self.y = 0.0
        self.margin = 42
        self.palette = [colors.HexColor(x) for x in ["#2563eb", "#059669", "#dc2626", "#7c3aed", "#ea580c", "#0891b2"]]

    def close(self) -> None:
        if self.page == 0:
            self.new_page("Relatorio vazio")
        if self.logger:
            self.logger.event("pdf", "salvando", arquivo=str(self.path), paginas=self.page)
        self.canvas.save()

    def can_add_page(self) -> bool:
        return self.page < self.max_pages

    def new_page(self, heading: str = "") -> bool:
        if self.page >= self.max_pages:
            if self.logger:
                self.logger.event("pdf", "limite_paginas_atingido", max_pages=self.max_pages, heading=heading)
            return False
        if self.page:
            self.canvas.showPage()
        self.page += 1
        if self.logger:
            self.logger.event("pdf", "nova_pagina", pagina=self.page, titulo=heading)
        self.y = self.height - self.margin
        self.canvas.setFillColor(colors.HexColor("#111827"))
        self.canvas.setFont("Helvetica-Bold", 9)
        self.canvas.drawString(self.margin, self.height - 24, self.title[:90])
        self.canvas.setFont("Helvetica", 8)
        self.canvas.setFillColor(colors.HexColor("#6b7280"))
        self.canvas.drawRightString(self.width - self.margin, 22, f"Pagina {self.page}")
        self.canvas.setStrokeColor(colors.HexColor("#e5e7eb"))
        self.canvas.line(self.margin, self.height - 32, self.width - self.margin, self.height - 32)
        if heading:
            self.heading(heading)
        return True

    def ensure_space(self, needed: float, heading: str = "") -> bool:
        if self.y - needed < 46:
            return self.new_page(heading)
        return True

    def heading(self, text: str) -> None:
        self.canvas.setFillColor(colors.HexColor("#111827"))
        self.canvas.setFont("Helvetica-Bold", 17)
        self.canvas.drawString(self.margin, self.y, text[:85])
        self.y -= 26

    def subheading(self, text: str) -> None:
        self.ensure_space(34)
        self.canvas.setFillColor(colors.HexColor("#1f2937"))
        self.canvas.setFont("Helvetica-Bold", 12)
        self.canvas.drawString(self.margin, self.y, text[:95])
        self.y -= 18

    def paragraph(self, text: str, size: int = 9, width_chars: int = 96, leading: int = 12) -> None:
        if not text:
            self.y -= 8
            return
        lines = []
        for raw in str(text).splitlines():
            lines.extend(textwrap.wrap(raw, width=width_chars) or [""])
        self.ensure_space(max(16, len(lines) * leading + 4))
        self.canvas.setFillColor(colors.HexColor("#374151"))
        self.canvas.setFont("Helvetica", size)
        for line in lines:
            self.canvas.drawString(self.margin, self.y, line)
            self.y -= leading

    def hero_panel(self, title: str, subtitle: str, kicker: str = "RELATORIO ELEITORAL") -> None:
        panel_h = 96
        self.ensure_space(panel_h + 18)
        x = self.margin
        y = self.y - panel_h
        w = self.width - 2 * self.margin
        self.canvas.setFillColor(colors.HexColor("#0f172a"))
        self.canvas.roundRect(x, y, w, panel_h, 10, stroke=0, fill=1)
        self.canvas.setFillColor(colors.HexColor("#0f766e"))
        self.canvas.rect(x + w - 150, y, 150, panel_h, stroke=0, fill=1)
        self.canvas.setFillColor(colors.HexColor("#22c55e"))
        self.canvas.roundRect(x + 16, y + panel_h - 30, 104, 17, 7, stroke=0, fill=1)
        self.canvas.setFillColor(colors.white)
        self.canvas.setFont("Helvetica-Bold", 7)
        self.canvas.drawString(x + 24, y + panel_h - 25, kicker[:28])
        self.canvas.setFont("Helvetica-Bold", 27)
        self.canvas.drawString(x + 18, y + 42, title[:32])
        self.canvas.setFont("Helvetica", 10)
        self.canvas.drawString(x + 20, y + 23, subtitle[:105])
        self.canvas.setFillColor(colors.HexColor("#bbf7d0"))
        self.canvas.setFont("Helvetica-Bold", 9)
        self.canvas.drawRightString(x + w - 18, y + 21, "camada ouro")
        self.y -= panel_h + 18

    def insight_cards(self, items: list[tuple[str, str, str]], columns: int = 4) -> None:
        if not items:
            return
        card_w = (self.width - 2 * self.margin - (columns - 1) * 10) / columns
        card_h = 76
        accent = ["#2563eb", "#0f766e", "#dc2626", "#7c3aed", "#ea580c", "#0891b2"]
        for i, (title, value, caption) in enumerate(items):
            if i % columns == 0:
                self.ensure_space(card_h + 14)
            col = i % columns
            x = self.margin + col * (card_w + 10)
            y = self.y - card_h
            self.canvas.setFillColor(colors.HexColor("#ffffff"))
            self.canvas.roundRect(x, y, card_w, card_h, 7, stroke=0, fill=1)
            self.canvas.setStrokeColor(colors.HexColor("#dbeafe"))
            self.canvas.roundRect(x, y, card_w, card_h, 7, stroke=1, fill=0)
            self.canvas.setFillColor(colors.HexColor(accent[i % len(accent)]))
            self.canvas.roundRect(x, y + card_h - 5, card_w, 5, 3, stroke=0, fill=1)
            self.canvas.setFillColor(colors.HexColor("#475569"))
            self.canvas.setFont("Helvetica-Bold", 7)
            self.canvas.drawString(x + 10, y + card_h - 22, title[:28].upper())
            self.canvas.setFillColor(colors.HexColor("#0f172a"))
            self.canvas.setFont("Helvetica-Bold", 17)
            self.canvas.drawString(x + 10, y + 30, str(value)[:19])
            self.canvas.setFillColor(colors.HexColor("#64748b"))
            self.canvas.setFont("Helvetica", 7)
            self.canvas.drawString(x + 10, y + 14, str(caption)[:34])
            if col == columns - 1 or i == len(items) - 1:
                self.y -= card_h + 14

    def chips(self, x: float, y: float, chips: list[str], max_width: float) -> float:
        cx = x
        cy = y
        self.canvas.setFont("Helvetica-Bold", 6)
        for idx, chip in enumerate(chips[:5]):
            text = str(chip)[:18]
            chip_w = min(max_width, max(34, 5.2 * len(text) + 14))
            if cx + chip_w > x + max_width:
                cx = x
                cy -= 15
            self.canvas.setFillColor(colors.HexColor(["#dbeafe", "#dcfce7", "#fee2e2", "#ede9fe", "#ffedd5"][idx % 5]))
            self.canvas.roundRect(cx, cy - 8, chip_w, 12, 6, stroke=0, fill=1)
            self.canvas.setFillColor(colors.HexColor("#0f172a"))
            self.canvas.drawString(cx + 6, cy - 4, text)
            cx += chip_w + 5
        return cy

    def profile_cards(self, title: str, df: pd.DataFrame, limit: int = 6) -> None:
        self.ensure_space(40)
        self.subheading(title)
        if df is None or df.empty:
            self.paragraph("Sem perfis processados para este recorte.")
            return
        work = sort_by_numeric(df, ["share_perfil", "share", "eleitorado", "peso"], ascending=False).head(limit)
        cols = 2
        gap = 12
        card_w = (self.width - 2 * self.margin - gap) / cols
        card_h = 106
        for i, (_, row) in enumerate(work.iterrows()):
            if i % cols == 0:
                self.ensure_space(card_h + 12)
            col = i % cols
            x = self.margin + col * (card_w + gap)
            y = self.y - card_h
            accent = ["#2563eb", "#0f766e", "#7c3aed", "#ea580c", "#dc2626", "#0891b2"][i % 6]
            self.canvas.setFillColor(colors.HexColor("#ffffff"))
            self.canvas.roundRect(x, y, card_w, card_h, 8, stroke=0, fill=1)
            self.canvas.setStrokeColor(colors.HexColor("#d1d5db"))
            self.canvas.roundRect(x, y, card_w, card_h, 8, stroke=1, fill=0)
            self.canvas.setFillColor(colors.HexColor(accent))
            self.canvas.circle(x + 18, y + card_h - 21, 11, stroke=0, fill=1)
            self.canvas.setFillColor(colors.white)
            self.canvas.setFont("Helvetica-Bold", 9)
            self.canvas.drawCentredString(x + 18, y + card_h - 24, str(i + 1))
            label = profile_title(row)
            share_col = choose_col(pd.DataFrame([row]), ["share_perfil", "share", "share_eleitorado_ano", "share_perfil_na_entidade"])
            weight_col = choose_col(pd.DataFrame([row]), ["eleitorado", "peso", "votos_proxy", "votos"])
            self.canvas.setFillColor(colors.HexColor("#0f172a"))
            self.canvas.setFont("Helvetica-Bold", 10)
            self.canvas.drawString(x + 36, y + card_h - 18, f"Perfil {i + 1}")
            self.canvas.setFont("Helvetica", 7)
            wrapped = textwrap.wrap(label, width=42)
            yy = y + card_h - 35
            for line in wrapped[:2]:
                self.canvas.drawString(x + 12, yy, line)
                yy -= 10
            self.chips(x + 12, yy - 2, profile_chips(label), card_w - 24)
            self.canvas.setFillColor(colors.HexColor("#475569"))
            self.canvas.setFont("Helvetica-Bold", 7)
            share_text = fmt_pct(row.get(share_col)) if share_col else "-"
            peso_text = fmt_int(row.get(weight_col)) if weight_col else "-"
            self.canvas.drawString(x + 12, y + 13, f"Participacao: {share_text} | Base: {peso_text}")
            if col == cols - 1 or i == len(work) - 1:
                self.y -= card_h + 12

    def donut(self, title: str, rows: list[tuple[str, float, str]], max_rows: int = 8) -> None:
        rows = [(label, float(value or 0), suffix) for label, value, suffix in rows if meaningful(label) and float(value or 0) > 0][:max_rows]
        if not rows:
            self.paragraph(f"{title}: sem dados para grafico.")
            return
        self.ensure_space(190, title)
        self.subheading(title)
        total = sum(value for _, value, _ in rows) or 1
        cx = self.margin + 82
        cy = self.y - 76
        radius = 60
        start = 90
        for idx, (_, value, _) in enumerate(rows):
            extent = 360 * value / total
            self.canvas.setFillColor(self.palette[idx % len(self.palette)])
            self.canvas.wedge(cx - radius, cy - radius, cx + radius, cy + radius, start, extent, stroke=0, fill=1)
            start += extent
        self.canvas.setFillColor(colors.white)
        self.canvas.circle(cx, cy, 32, stroke=0, fill=1)
        self.canvas.setFillColor(colors.HexColor("#0f172a"))
        self.canvas.setFont("Helvetica-Bold", 11)
        self.canvas.drawCentredString(cx, cy + 2, "Top")
        self.canvas.setFont("Helvetica", 8)
        self.canvas.drawCentredString(cx, cy - 12, str(len(rows)))
        lx = self.margin + 175
        ly = self.y - 18
        self.canvas.setFont("Helvetica", 8)
        for idx, (label, _, suffix) in enumerate(rows):
            y = ly - idx * 16
            self.canvas.setFillColor(self.palette[idx % len(self.palette)])
            self.canvas.rect(lx, y - 7, 9, 9, stroke=0, fill=1)
            self.canvas.setFillColor(colors.HexColor("#0f172a"))
            self.canvas.drawString(lx + 15, y - 5, f"{label[:44]}  {suffix}")
        self.y -= 162

    def cards(self, items: list[tuple[str, str]], columns: int = 3) -> None:
        if not items:
            return
        if self.logger:
            self.logger.event("pdf", "cards", quantidade=len(items), colunas=columns)
        card_w = (self.width - 2 * self.margin - (columns - 1) * 10) / columns
        card_h = 58
        for i, (title, value) in enumerate(items):
            if i % columns == 0:
                self.ensure_space(card_h + 12)
            col = i % columns
            x = self.margin + col * (card_w + 10)
            y = self.y - card_h
            self.canvas.setFillColor(colors.HexColor("#f8fafc"))
            self.canvas.roundRect(x, y, card_w, card_h, 6, stroke=0, fill=1)
            self.canvas.setStrokeColor(colors.HexColor("#d1d5db"))
            self.canvas.roundRect(x, y, card_w, card_h, 6, stroke=1, fill=0)
            self.canvas.setFillColor(colors.HexColor("#475569"))
            self.canvas.setFont("Helvetica", 7)
            self.canvas.drawString(x + 10, y + card_h - 18, str(title)[:34])
            self.canvas.setFillColor(colors.HexColor("#0f172a"))
            self.canvas.setFont("Helvetica-Bold", 14)
            self.canvas.drawString(x + 10, y + 18, str(value)[:28])
            if col == columns - 1 or i == len(items) - 1:
                self.y -= card_h + 12

    def hbar(self, title: str, rows: list[tuple[str, float, str]], max_rows: int = 12) -> None:
        original_rows = len(rows)
        rows = [(a, float(b or 0), c) for a, b, c in rows if meaningful(a)][:max_rows]
        if not rows:
            if self.logger:
                self.logger.graph(
                    evento="grafico_sem_dados",
                    tipo="barra_horizontal_reportlab",
                    titulo=title,
                    linhas_recebidas=original_rows,
                    linhas_usadas=0,
                    metodo="sem desenho; dados vazios apos filtro de valores nulos/sem valor",
                )
            self.paragraph(f"{title}: sem dados disponiveis.")
            return
        chart_h = 24 + len(rows) * 18
        self.ensure_space(chart_h + 16, title)
        self.subheading(title)
        max_val = max([abs(v) for _, v, _ in rows] or [1]) or 1
        label_w = 160
        bar_w = self.width - 2 * self.margin - label_w - 70
        if self.logger:
            self.logger.graph(
                evento="grafico_gerado",
                tipo="barra_horizontal_reportlab",
                titulo=title,
                pagina=self.page,
                linhas_recebidas=original_rows,
                linhas_usadas=len(rows),
                top_n=max_rows,
                max_valor=max_val,
                largura_util_barras=round(bar_w, 2),
                metodo="ReportLab canvas.rect; largura = largura_util_barras * valor / max_valor; uma barra por linha",
                dados=[{"label": a, "valor": b, "sufixo": c} for a, b, c in rows[:max_rows]],
            )
        self.canvas.setFont("Helvetica", 8)
        for idx, (label, value, suffix) in enumerate(rows):
            y = self.y - idx * 18
            color = self.palette[idx % len(self.palette)]
            self.canvas.setFillColor(colors.HexColor("#374151"))
            self.canvas.drawString(self.margin, y, str(label)[:34])
            x = self.margin + label_w
            width = max(2, bar_w * (value / max_val))
            self.canvas.setFillColor(color)
            self.canvas.rect(x, y - 3, width, 9, stroke=0, fill=1)
            self.canvas.setFillColor(colors.HexColor("#111827"))
            self.canvas.drawString(x + width + 5, y - 1, suffix)
        self.y -= len(rows) * 18 + 8

    def table(self, title: str, df: pd.DataFrame, cols: list[str], limit: int = 12) -> None:
        if df is None or df.empty:
            if self.logger:
                self.logger.event("pdf", "tabela_sem_dados", titulo=title)
            self.paragraph(f"{title}: sem dados disponiveis.")
            return
        if self.logger:
            self.logger.event("pdf", "tabela", titulo=title, linhas_recebidas=len(df), limite=limit, colunas_solicitadas=",".join(cols))
        self.ensure_space(80, title)
        self.subheading(title)
        work = df.head(limit).copy()
        usable = [c for c in cols if c in work.columns][:5]
        if not usable:
            usable = list(work.columns[:5])
        col_w = (self.width - 2 * self.margin) / max(1, len(usable))
        self.canvas.setFillColor(colors.HexColor("#e5e7eb"))
        self.canvas.rect(self.margin, self.y - 15, self.width - 2 * self.margin, 17, stroke=0, fill=1)
        self.canvas.setFillColor(colors.HexColor("#111827"))
        self.canvas.setFont("Helvetica-Bold", 7)
        for i, col in enumerate(usable):
            self.canvas.drawString(self.margin + i * col_w + 3, self.y - 10, col[:18])
        self.y -= 20
        self.canvas.setFont("Helvetica", 7)
        for _, row in work.iterrows():
            self.ensure_space(14)
            for i, col in enumerate(usable):
                self.canvas.drawString(self.margin + i * col_w + 3, self.y, str(row.get(col, ""))[:24])
            self.y -= 12
        self.y -= 6


def polars_to_pandas(data: Any) -> pd.DataFrame:
    return pd.DataFrame(polars_records(data))


def read_parquet_file_small(path: Path, limit: int) -> pd.DataFrame:
    if pl is not None:
        frame = pl.scan_parquet(str(path), hive_partitioning=False).limit(max(1, int(limit))).collect(engine="streaming")
        return pd.DataFrame(frame.to_dicts())
    return pd.read_parquet(path).head(limit)


def read_small_parquet(path: Path, limit: int = 50) -> pd.DataFrame:
    """Le um Parquet pequeno da camada ouro, inclusive quando o arquivo nao tem extensao."""
    if not path.exists():
        return pd.DataFrame()
    try:
        if path.is_dir():
            files = sorted(path.rglob("*.parquet"))
            frames: list[pd.DataFrame] = []
            remaining = max(1, int(limit))
            for file in files:
                if remaining <= 0:
                    break
                frame = read_parquet_file_small(file, remaining)
                if not frame.empty:
                    frames.append(frame.head(remaining))
                    remaining -= len(frames[-1])
            return pd.concat(frames, ignore_index=True).head(limit) if frames else pd.DataFrame()
        return read_parquet_file_small(path, limit)
    except Exception as exc:
        return pd.DataFrame({"erro": [str(exc)], "arquivo": [str(path)]})


def read_ouro_brasil(run: Path, name: str, limit: int = 50) -> pd.DataFrame:
    return read_small_parquet(run / "ouro" / "brasil" / name, limit=limit)


def read_ouro_level(run: Path, level: str, name: str, limit: int = 500, uf: str = "", municipio: str = "") -> pd.DataFrame:
    path = run / "ouro" / level / name
    if not path.exists():
        return pd.DataFrame()
    if pl is not None:
        try:
            lf = pl.scan_parquet(str(path / "**" / "*.parquet"), hive_partitioning=True)
            if uf and "uf" in lf.collect_schema().names():
                lf = lf.filter(pl.col("uf").cast(pl.Utf8) == str(uf).upper())
            if municipio and "cd_municipio" in lf.collect_schema().names():
                lf = lf.filter(pl.col("cd_municipio").cast(pl.Utf8) == str(municipio))
            frame = lf.limit(max(1, int(limit))).collect(engine="streaming")
            return pd.DataFrame(frame.to_dicts())
        except Exception:
            pass
    df = read_small_parquet(path, limit=max(limit, 1000))
    if uf and "uf" in df.columns:
        df = df[df["uf"].astype(str).str.upper() == str(uf).upper()]
    if municipio and "cd_municipio" in df.columns:
        df = df[df["cd_municipio"].astype(str) == str(municipio)]
    return df.head(limit)


def add_histogram_table_section(pdf: PdfReport, title: str, df: pd.DataFrame, top_n: int, value_col: str = "qtd_pessoas") -> None:
    if df is None or df.empty or "dimensao_perfil" not in df.columns or "valor_perfil" not in df.columns:
        return
    if value_col not in df.columns:
        value_col = "qtd_votos" if "qtd_votos" in df.columns else ""
    if not value_col:
        return
    work = df.copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce").fillna(0)
    work = work[(work[value_col] > 0) & (work["valor_perfil"].astype(str).str.strip() != "")]
    if work.empty:
        return
    pdf.new_page(title)
    for dim in ["perfil_combinado", "faixa_etaria", "sexo_genero", "escolaridade", "estado_civil", "raca_cor"]:
        sub = work[work["dimensao_perfil"].astype(str) == dim].copy()
        if sub.empty:
            continue
        sub = sub.groupby("valor_perfil", as_index=False)[value_col].sum()
        sub = sort_by_numeric(sub, [value_col], ascending=False).head(top_n)
        rows = [(str(row.get("valor_perfil", "")), pd.to_numeric(row.get(value_col), errors="coerce"), fmt_int(row.get(value_col))) for _, row in sub.iterrows()]
        if not rows:
            continue
        pdf.hbar(f"{title} - {dim}", rows, max_rows=min(top_n, 14))


def add_party_result_section(pdf: PdfReport, title: str, df: pd.DataFrame, top_n: int) -> None:
    if df is None or df.empty:
        pdf.paragraph(f"{title}: sem dados processados.")
        return
    should_split = "quem ganhou" not in title.lower() and "quem nao ganhou" not in title.lower()
    winners, losers = split_result_status_df(df) if should_split else (pd.DataFrame(), pd.DataFrame())
    if should_split and not winners.empty:
        add_party_result_section(pdf, f"{title} - quem ganhou", winners, top_n)
        if not losers.empty:
            add_party_result_section(pdf, f"{title} - quem nao ganhou", losers, min(top_n, 12))
        return
    label_col = choose_col(df, ["partido", "sg_partido", "sigla_partido", "entidade", "partido_vencedor", "nm_partido", "nr_partido"])
    value_col = choose_col(df, ["share_votos", "share_pred_2026", "votos", "votos_total"])
    if not label_col:
        pdf.paragraph(f"{title}: a tabela existe, mas nao tem coluna de partido/entidade identificavel.")
        return
    if not value_col:
        pdf.paragraph(f"{title}: a tabela existe, mas nao tem coluna numerica de votos/share identificavel.")
        return
    work = sort_by_numeric(df, [value_col], ascending=False).head(top_n)
    rows = []
    for _, row in work.iterrows():
        value = pd.to_numeric(row.get(value_col), errors="coerce")
        label_value = fmt_pct(value) if "share" in value_col else fmt_int(value)
        rows.append((str(row.get(label_col, "")), value, label_value))
    pdf.donut(f"{title} - distribuicao", rows, max_rows=min(top_n, 8))
    pdf.hbar(title, rows, max_rows=top_n)


def add_profile_result_section(pdf: PdfReport, title: str, df: pd.DataFrame, top_n: int) -> None:
    if df is None or df.empty:
        pdf.paragraph(f"{title}: sem perfis processados.")
        return
    label_col = choose_col(df, ["perfil_combinado", "perfil_eleitor", "valor_perfil", "descricao_perfil"])
    value_col = choose_col(df, ["histograma_qtd_pessoas", "qtd_eleitores_perfil", "eleitorado", "peso", "share_perfil", "share", "share_eleitorado_ano"])
    if not label_col or not value_col:
        return
    work = sort_by_numeric(df, [value_col], ascending=False).head(top_n)
    rows = []
    for _, row in work.iterrows():
        value = pd.to_numeric(row.get(value_col), errors="coerce")
        suffix = fmt_pct(value) if "share" in value_col else fmt_int(value)
        rows.append((profile_title(row), value, suffix))
    pdf.donut(f"{title} - grupos principais", rows, max_rows=min(top_n, 8))
    pdf.hbar(title, rows, max_rows=min(top_n, 10))


def add_polars_party_section(pdf: PdfReport, title: str, data: Any, top_n: int) -> None:
    df = polars_to_pandas(data)
    pdf.table(title, df, ["partido", "share_pred_2026", "votos_pred_2026", "perfil_eleitor_2026"], limit=top_n)
    if not df.empty and "partido" in df.columns:
        rows = [
            (
                str(r.get("partido", "")),
                pd.to_numeric(r.get("share_pred_2026"), errors="coerce"),
                fmt_pct(r.get("share_pred_2026")),
            )
            for _, r in df.head(top_n).iterrows()
        ]
        pdf.hbar(title, rows, max_rows=top_n)


def add_polars_cluster_section(pdf: PdfReport, title: str, data: Any, top_n: int) -> None:
    df = polars_to_pandas(data)
    pdf.table(title, df, ["ano", "cluster_id", "perfil_combinado", "partido", "share_cluster", "eleitorado", "votos_proxy"], limit=top_n)
    if df.empty or "cluster_id" not in df.columns:
        return
    value_col = next((col for col in ["share_cluster", "votos_proxy", "eleitorado"] if col in df.columns), None)
    if not value_col:
        return
    rows = [
        (
            f"Cluster {r.get('cluster_id', '')}",
            pd.to_numeric(r.get(value_col), errors="coerce"),
            str(r.get("perfil_combinado") or r.get("descricao") or "")[:80],
        )
        for _, r in df.head(top_n).iterrows()
    ]
    pdf.hbar(title, rows, max_rows=top_n)


def selected_ufs_polars(store: Any, args: argparse.Namespace, progress: dict[str, Any] | None = None) -> list[str]:
    if str(getattr(args, "ufs", "") or "").strip():
        return [x.strip().upper() for x in str(args.ufs).split(",") if x.strip()]
    progress = progress or {}
    ufs = [str(x).upper() for x in (progress.get("ufs_concluidas") or []) if str(x).strip()]
    if ufs:
        return [uf for uf in ufs if uf not in {"BR", "ZZ", "SEM_UF"}]
    try:
        metrics = store.metrics_by_year("timeline_uf")
        if getattr(metrics, "height", 0) and "uf" in metrics.columns:
            return [str(x).upper() for x in metrics.select("uf").unique().sort("uf").to_series().to_list() if str(x).strip()]
    except Exception:
        return []
    return []


def add_polars_common_intro(pdf: PdfReport, run: Path, modalidade: str, logger: PdfRunLogger) -> None:
    add_cover(pdf, run)
    add_methodology(pdf)
    add_graph_generation_methodology(pdf, logger)
    pdf.new_page("Arquitetura de consulta")
    pdf.paragraph(
        "Este PDF foi gerado com Polars LazyFrame como engine principal. "
        "Os Parquets sao escaneados de forma preguiçosa, com filtros por ano, UF e municipio aplicados antes do collect."
    )
    pdf.cards(
        [
            ("Engine", "Polars LazyFrame"),
            ("Modalidade", MODE_LABELS.get(modalidade, modalidade)),
            ("Baixo nivel", "PyArrow/Parquet"),
            ("Fallback", "DuckDB"),
        ],
        columns=2,
    )
    pdf.paragraph("Recursos ativos nesta modalidade: " + ", ".join(modalidade_info(modalidade).get("features", [])))


def add_polars_progress_page(pdf: PdfReport, progress: dict[str, Any]) -> None:
    pdf.new_page("Progresso da camada ouro")
    pdf.cards(
        [
            ("Fatias totais", fmt_int(progress.get("total"))),
            ("Concluidas", fmt_int(progress.get("concluidas"))),
            ("Pendentes", fmt_int(progress.get("pendentes"))),
            ("UFs pendentes", fmt_int(len(progress.get("ufs_pendentes") or []))),
        ],
        columns=2,
    )
    pdf.paragraph("UFs pendentes: " + ", ".join(progress.get("ufs_pendentes") or []) if progress.get("ufs_pendentes") else "Sem pendencias registradas no manifesto.")


def add_polars_brasil_pages(pdf: PdfReport, store: Any, args: argparse.Namespace, modalidade: str) -> None:
    pdf.new_page("")
    logger = getattr(pdf, "logger", None)
    run_path = Path(getattr(store, "run_path", getattr(args, "run", "."))).expanduser()
    pdf.hero_panel(
        "Brasil",
        "Perfil eleitoral nacional, partidos e leitura executiva gerados direto dos Parquets da camada ouro.",
        "DASHBOARD NACIONAL",
    )
    if logger:
        logger.event("consulta_direta", "inicio", nivel="brasil", tabela="ouro/brasil/resumo")
    resumo = read_ouro_brasil(run_path, "resumo", limit=max(args.top_n, 10000))
    if logger:
        logger.event("consulta_direta", "fim", nivel="brasil", tabela="ouro/brasil/resumo", linhas=len(resumo))
    anos = unique_texts(resumo, choose_col(resumo, ["ano"]), limit=8)
    eleitorado_col = choose_col(resumo, ["eleitorado", "eleitorado_total", "qtd_eleitores"])
    eleitorado_val = pd.to_numeric(resumo[eleitorado_col], errors="coerce").max() if eleitorado_col else 0
    partidos_preview = pd.DataFrame()
    if modalidade_allows(modalidade, "partido"):
        if logger:
            logger.event("consulta_direta", "inicio", nivel="brasil", tabela="ouro/brasil/resultado_partido")
        partidos_preview = read_ouro_brasil(run_path, "resultado_partido", limit=max(args.top_n, 10000))
        if logger:
            logger.event("consulta_direta", "fim", nivel="brasil", tabela="ouro/brasil/resultado_partido", linhas=len(partidos_preview))
    perfis_preview = pd.DataFrame()
    hist_perfis = pd.DataFrame()
    hist_partidos = pd.DataFrame()
    if modalidade_allows(modalidade, "perfil"):
        if logger:
            logger.event("consulta_direta", "inicio", nivel="brasil", tabela="ouro/brasil/perfil_eleitor")
        perfis_preview = read_ouro_brasil(run_path, "perfil_eleitor", limit=max(args.top_n, 10000))
        hist_perfis = read_ouro_brasil(run_path, "contagem_colunas_perfil_eleitor", limit=max(args.top_n * 20, 1000))
        if logger:
            logger.event("consulta_direta", "fim", nivel="brasil", tabela="ouro/brasil/perfil_eleitor", linhas=len(perfis_preview))
    if modalidade_allows(modalidade, "partido"):
        hist_partidos = read_ouro_brasil(run_path, "contagem_colunas_perfil_partido", limit=max(args.top_n * 20, 1000))
    party_label_col = choose_col(partidos_preview, ["partido", "sg_partido", "sigla_partido", "entidade", "partido_vencedor", "nm_partido", "nr_partido"])
    top_party = "-"
    if party_label_col and not partidos_preview.empty:
        winners_preview, _ = split_result_status_df(partidos_preview)
        party_source = winners_preview if not winners_preview.empty else partidos_preview
        sorted_party = sort_by_numeric(party_source, ["share_votos", "share_pred_2026", "votos", "votos_total"], ascending=False)
        top_party = meaningful(sorted_party.iloc[0].get(party_label_col)) if not sorted_party.empty else "-"
    profile_label = "-"
    if not perfis_preview.empty:
        sorted_profiles = sort_by_numeric(perfis_preview, ["share_perfil", "share", "eleitorado", "peso"], ascending=False)
        profile_label = profile_title(sorted_profiles.iloc[0]) if not sorted_profiles.empty else "-"
    pdf.insight_cards(
        [
            ("Anos", ", ".join(anos) if anos else "-"),
            ("Eleitorado base", fmt_int(eleitorado_val)),
            ("Perfil dominante", profile_label[:22]),
            ("Partido destaque", top_party[:22]),
        ],
        columns=4,
    )
    pdf.paragraph(
        "Leitura rapida: os blocos abaixo mostram primeiro quem e o eleitor predominante no Brasil "
        "e depois como os votos por partido aparecem na camada ouro ja processada.",
        size=9,
        width_chars=102,
    )
    if modalidade_allows(modalidade, "perfil"):
        pdf.profile_cards("Quem e o eleitor medio no Brasil", perfis_preview, limit=min(args.top_n, 6))
        if not hist_perfis.empty:
            add_histogram_table_section(pdf, "Histogramas do eleitor no Brasil", hist_perfis, min(args.top_n, 14), "qtd_pessoas")
        else:
            add_profile_result_section(pdf, "Perfis mais fortes no Brasil", perfis_preview, min(args.top_n, 10))
    if modalidade_allows(modalidade, "partido"):
        pdf.paragraph("Fonte dos partidos: ouro/brasil/resultado_partido.", size=8)
        add_party_result_section(pdf, "Brasil por partido", partidos_preview, min(args.top_n, 12))
        add_histogram_table_section(pdf, "Perfil do eleitor por partido - Brasil", hist_partidos, min(args.top_n, 14), "qtd_votos")
    if modalidade_allows(modalidade, "candidato"):
        pdf.table("Resultado por candidato - Brasil", polars_to_pandas(store.entity_results(entity="candidato", nivel="brasil", limit=args.top_n)), ["ano", "entidade", "votos", "share_votos", "rank_entidade"], limit=args.top_n)
        pdf.table("Perfil por candidato - Brasil", polars_to_pandas(store.entity_profiles(entity="candidato", nivel="brasil", limit=args.top_n)), ["ano", "entidade", "perfil_combinado", "share_perfil_na_entidade"], limit=args.top_n)
    if modalidade_allows(modalidade, "cluster"):
        add_polars_cluster_section(pdf, "Clusters Brasil - eleitorado", store.cluster_personas(tipo="eleitores", nivel="brasil", limit=args.top_n), args.top_n)
        add_polars_cluster_section(pdf, "Clusters Brasil - eleitorado + partido", store.cluster_personas(tipo="resultado", nivel="brasil", limit=args.top_n), args.top_n)


def add_polars_state_pages(pdf: PdfReport, store: Any, args: argparse.Namespace, modalidade: str, uf: str) -> None:
    pdf.new_page("")
    pdf.hero_panel(
        f"Estado {uf}",
        "Resumo estadual com perfil dominante do eleitor e distribuicao partidaria da camada ouro.",
        "DASHBOARD ESTADUAL",
    )
    logger = getattr(pdf, "logger", None)
    if logger:
        logger.event("consulta_polars", "inicio", nivel="estado", uf=uf, tabela="ouro/estadual/resumo")
    metrics = polars_to_pandas(store.metrics_by_year("timeline_uf", uf=uf))
    if logger:
        logger.event("consulta_polars", "fim", nivel="estado", uf=uf, tabela="ouro/estadual/resumo", linhas=len(metrics))
    anos = unique_texts(metrics, choose_col(metrics, ["ano"]), limit=6)
    eleitorado_col = choose_col(metrics, ["eleitorado", "eleitorado_total", "qtd_eleitores"])
    eleitorado_val = pd.to_numeric(metrics[eleitorado_col], errors="coerce").max() if eleitorado_col else 0
    state_parties = pd.DataFrame()
    state_profiles = pd.DataFrame()
    state_hist_profiles = pd.DataFrame()
    state_hist_parties = pd.DataFrame()
    if modalidade_allows(modalidade, "partido"):
        if logger:
            logger.event("consulta_polars", "inicio", nivel="estado", uf=uf, tabela="ouro/estadual/resultado_partido")
        state_parties = store.quick_party_results(nivel="estado", uf=uf, limit=args.top_n)
        if logger:
            logger.event("consulta_polars", "fim", nivel="estado", uf=uf, tabela="ouro/estadual/resultado_partido", linhas=getattr(state_parties, "height", 0))
    if modalidade_allows(modalidade, "perfil"):
        if logger:
            logger.event("consulta_polars", "inicio", nivel="estado", uf=uf, tabela="ouro/estadual/perfil_eleitor")
        state_profiles = polars_to_pandas(store.top_profiles("estado", uf=uf, limit=args.top_n))
        state_hist_profiles = read_ouro_level(Path(getattr(store, "run_path", getattr(args, "run", "."))).expanduser(), "estadual", "contagem_colunas_perfil_eleitor", limit=max(args.top_n * 20, 1000), uf=uf)
        if logger:
            logger.event("consulta_polars", "fim", nivel="estado", uf=uf, tabela="ouro/estadual/perfil_eleitor", linhas=len(state_profiles))
    if modalidade_allows(modalidade, "partido"):
        state_hist_parties = read_ouro_level(Path(getattr(store, "run_path", getattr(args, "run", "."))).expanduser(), "estadual", "contagem_colunas_perfil_partido", limit=max(args.top_n * 20, 1000), uf=uf)
    party_df = polars_to_pandas(state_parties) if hasattr(state_parties, "to_dicts") else state_parties
    party_label_col = choose_col(party_df, ["partido", "sg_partido", "sigla_partido", "entidade", "partido_vencedor", "nm_partido", "nr_partido"])
    top_party = "-"
    if party_label_col and not party_df.empty:
        sorted_party = sort_by_numeric(party_df, ["share_votos", "share_pred_2026", "votos", "votos_total"], ascending=False)
        top_party = meaningful(sorted_party.iloc[0].get(party_label_col)) if not sorted_party.empty else "-"
    profile_label = "-"
    if not state_profiles.empty:
        sorted_profiles = sort_by_numeric(state_profiles, ["share_perfil", "share", "eleitorado", "peso"], ascending=False)
        profile_label = profile_title(sorted_profiles.iloc[0]) if not sorted_profiles.empty else "-"
    pdf.insight_cards(
        [
            ("Anos", ", ".join(anos) if anos else "-"),
            ("Eleitorado base", fmt_int(eleitorado_val)),
            ("Perfil dominante", profile_label[:22]),
            ("Partido destaque", top_party[:22]),
        ],
        columns=4,
    )
    if modalidade_allows(modalidade, "perfil"):
        pdf.profile_cards(f"Quem e o eleitor medio em {uf}", state_profiles, limit=min(args.top_n, 6))
        if not state_hist_profiles.empty:
            add_histogram_table_section(pdf, f"Histogramas do eleitor em {uf}", state_hist_profiles, min(args.top_n, 14), "qtd_pessoas")
        else:
            add_profile_result_section(pdf, f"Perfis mais fortes em {uf}", state_profiles, min(args.top_n, 10))
    if modalidade_allows(modalidade, "partido"):
        pdf.paragraph(f"Fonte dos partidos em {uf}: ouro/estadual/resultado_partido.", size=8)
        add_party_result_section(pdf, f"{uf} por partido", party_df, min(args.top_n, 12))
        add_histogram_table_section(pdf, f"Perfil do eleitor por partido - {uf}", state_hist_parties, min(args.top_n, 14), "qtd_votos")
    if modalidade_allows(modalidade, "candidato"):
        pdf.table(f"Resultado por candidato - {uf}", polars_to_pandas(store.entity_results(entity="candidato", nivel="estado", uf=uf, limit=args.top_n)), ["ano", "entidade", "votos", "share_votos", "rank_entidade"], limit=args.top_n)
        pdf.table(f"Perfil por candidato - {uf}", polars_to_pandas(store.entity_profiles(entity="candidato", nivel="estado", uf=uf, limit=args.top_n)), ["ano", "entidade", "perfil_combinado", "share_perfil_na_entidade"], limit=args.top_n)
    if modalidade_allows(modalidade, "cluster"):
        add_polars_cluster_section(pdf, f"Clusters {uf} - eleitorado", store.cluster_personas(tipo="eleitores", nivel="estado", uf=uf, limit=args.top_n), args.top_n)
        add_polars_cluster_section(pdf, f"Clusters {uf} - eleitorado + partido", store.cluster_personas(tipo="resultado", nivel="estado", uf=uf, limit=args.top_n), args.top_n)


def add_polars_municipality_pages(pdf: PdfReport, store: Any, args: argparse.Namespace, modalidade: str, uf: str, municipio: dict[str, str]) -> None:
    label = municipio.get("label", "")
    value = municipio.get("value", "")
    pdf.new_page(f"Municipio {label} - {uf}")
    if modalidade_allows(modalidade, "partido"):
        mun_parties = store.party_prediction("sim_partidos_municipios", uf=uf, municipio=value, cenario="base", limit=args.top_n)
        mun_source = "simulacao_2026"
        if getattr(mun_parties, "height", 0) == 0:
            mun_parties = store.historical_party_results(uf=uf, municipio=value, limit=args.top_n)
            mun_source = "historico_processado"
        pdf.paragraph(f"Fonte dos partidos em {label}: {mun_source}.")
        add_polars_party_section(pdf, f"{label} por partido", mun_parties, args.top_n)
    if modalidade_allows(modalidade, "perfil"):
        mun_profiles = polars_to_pandas(store.top_profiles("municipio", uf=uf, municipio=value, limit=args.top_n))
        pdf.table(f"Top perfis {label}", mun_profiles, ["ano", "perfil_combinado", "share_perfil", "eleitorado"], limit=args.top_n)
    if modalidade_allows(modalidade, "candidato"):
        pdf.table(f"Resultado por candidato - {label}", polars_to_pandas(store.entity_results(entity="candidato", nivel="municipio", uf=uf, municipio=value, limit=args.top_n)), ["ano", "entidade", "votos", "share_votos", "rank_entidade"], limit=args.top_n)
        pdf.table(f"Perfil por candidato - {label}", polars_to_pandas(store.entity_profiles(entity="candidato", nivel="municipio", uf=uf, municipio=value, limit=args.top_n)), ["ano", "entidade", "perfil_combinado", "share_perfil_na_entidade"], limit=args.top_n)
    if modalidade_allows(modalidade, "cluster"):
        add_polars_cluster_section(pdf, f"Clusters {label} - eleitorado", store.cluster_personas(tipo="eleitores", nivel="municipio", uf=uf, municipio=value, limit=args.top_n), args.top_n)
        add_polars_cluster_section(pdf, f"Clusters {label} - eleitorado + partido", store.cluster_personas(tipo="resultado", nivel="municipio", uf=uf, municipio=value, limit=args.top_n), args.top_n)


def safe_pdf_name(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(text).strip().lower())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "sem_nome"


def build_report_polars_split(args: argparse.Namespace, run: Path, out: Path) -> Path:
    modalidade = normalize_modalidade(getattr(args, "modalidade_analise", "completa"))
    out_dir = out if out.suffix.lower() != ".pdf" else out.with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)
    store = PolarsStore(run)  # type: ignore[misc]
    progress = store.ouro_resultados_summary()
    generated: list[str] = []

    brasil_out = out_dir / "00_brasil.pdf"
    logger = PdfRunLogger(brasil_out, verbose=not args.quiet, log_dir=Path(args.log_dir).expanduser() if args.log_dir else None)
    logger.event("relatorio_split", "inicio_brasil", saida=str(brasil_out), modalidade=modalidade)
    pdf = PdfReport(brasil_out, f"Relatorio Brasil - {MODE_LABELS.get(modalidade, modalidade)}", max_pages=args.max_pages, logger=logger)
    try:
        add_polars_common_intro(pdf, run, modalidade, logger)
        add_polars_progress_page(pdf, progress)
        add_polars_brasil_pages(pdf, store, args, modalidade)
        pdf.new_page("Notas finais")
        pdf.paragraph("Este arquivo foi gerado primeiro no modo separado por nivel: Brasil -> estados -> municipios.")
    finally:
        pdf.close()
    generated.append(str(brasil_out))

    ufs = selected_ufs_polars(store, args, progress)
    for idx, uf in enumerate(ufs, start=1):
        state_out = out_dir / f"{idx:02d}_estado_{safe_pdf_name(uf)}.pdf"
        logger = PdfRunLogger(state_out, verbose=not args.quiet, log_dir=Path(args.log_dir).expanduser() if args.log_dir else None)
        logger.event("relatorio_split", "inicio_estado", uf=uf, saida=str(state_out), modalidade=modalidade)
        pdf = PdfReport(state_out, f"Relatorio Estado {uf} - {MODE_LABELS.get(modalidade, modalidade)}", max_pages=args.max_pages, logger=logger)
        try:
            add_polars_common_intro(pdf, run, modalidade, logger)
            add_polars_state_pages(pdf, store, args, modalidade, uf)
            pdf.new_page("Notas finais")
            pdf.paragraph(f"Relatorio estadual gerado a partir da camada ouro para {uf}.")
        finally:
            pdf.close()
        generated.append(str(state_out))

        if modalidade_allows(modalidade, "municipio") and int(args.municipios_por_uf or 0) > 0:
            municipios = store.municipios(uf)[: int(args.municipios_por_uf or 0)]
            for mun_idx, municipio in enumerate(municipios, start=1):
                value = municipio.get("value", "")
                label = municipio.get("label", "")
                cd = value.split("|", 1)[0] if "|" in value else safe_pdf_name(label)
                mun_out = out_dir / f"{idx:02d}_{mun_idx:03d}_municipio_{safe_pdf_name(uf)}_{safe_pdf_name(cd)}.pdf"
                logger = PdfRunLogger(mun_out, verbose=not args.quiet, log_dir=Path(args.log_dir).expanduser() if args.log_dir else None)
                logger.event("relatorio_split", "inicio_municipio", uf=uf, municipio=label, saida=str(mun_out), modalidade=modalidade)
                pdf = PdfReport(mun_out, f"Relatorio Municipio {label} - {MODE_LABELS.get(modalidade, modalidade)}", max_pages=args.max_pages, logger=logger)
                try:
                    add_polars_common_intro(pdf, run, modalidade, logger)
                    add_polars_municipality_pages(pdf, store, args, modalidade, uf, municipio)
                    pdf.new_page("Notas finais")
                    pdf.paragraph(f"Relatorio municipal gerado a partir da camada ouro para {label} - {uf}.")
                finally:
                    pdf.close()
                generated.append(str(mun_out))

    manifest = out_dir / "manifesto_pdfs.json"
    manifest.write_text(json.dumps({"modalidade": modalidade, "ordem": "brasil_estado_municipio", "arquivos": generated}, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir


def build_report_polars(args: argparse.Namespace, run: Path, out: Path) -> Path:
    modalidade = normalize_modalidade(getattr(args, "modalidade_analise", "completa"))
    log_dir = Path(args.log_dir).expanduser() if args.log_dir else None
    if log_dir and not log_dir.is_absolute():
        log_dir = (Path.cwd() / log_dir).resolve()
    logger = PdfRunLogger(out, verbose=not args.quiet, log_dir=log_dir)
    logger.event("relatorio", "inicio", run=str(run), saida=str(out), query_engine="polars", modalidade=modalidade, max_pages=args.max_pages, top_n=args.top_n)
    store = PolarsStore(run)  # type: ignore[misc]
    pdf = PdfReport(out, f"Relatorio eleitoral - {MODE_LABELS.get(modalidade, modalidade)}", max_pages=args.max_pages, logger=logger)
    try:
        add_cover(pdf, run)
        add_methodology(pdf)
        add_graph_generation_methodology(pdf, logger)
        pdf.new_page("Arquitetura de consulta")
        pdf.paragraph(
            "Este PDF foi gerado com Polars LazyFrame como engine principal. "
            "Os Parquets sao escaneados de forma preguiçosa, com filtros por ano, UF e municipio aplicados antes do collect."
        )
        pdf.cards(
            [
                ("Engine", "Polars LazyFrame"),
                ("Modalidade", MODE_LABELS.get(modalidade, modalidade)),
                ("Baixo nivel", "PyArrow/Parquet"),
                ("Batch pesado", "Spark opcional"),
                ("Fallback", "DuckDB"),
            ],
            columns=2,
        )
        pdf.paragraph("Recursos ativos nesta modalidade: " + ", ".join(modalidade_info(modalidade).get("features", [])))

        progress = store.ouro_resultados_summary()
        pdf.new_page("Progresso da camada ouro")
        pdf.cards(
            [
                ("Fatias totais", fmt_int(progress.get("total"))),
                ("Concluidas", fmt_int(progress.get("concluidas"))),
                ("Pendentes", fmt_int(progress.get("pendentes"))),
                ("UFs pendentes", fmt_int(len(progress.get("ufs_pendentes") or []))),
            ],
            columns=2,
        )
        pdf.paragraph("UFs pendentes: " + ", ".join(progress.get("ufs_pendentes") or []) if progress.get("ufs_pendentes") else "Sem pendencias registradas no manifesto.")

        pdf.new_page("Brasil")
        if modalidade_allows(modalidade, "partido"):
            partidos = store.party_prediction("sim_partidos_brasil", cenario="base", limit=args.top_n)
            fonte = "simulacao_2026"
            if getattr(partidos, "height", 0) == 0:
                partidos = store.historical_party_results(limit=args.top_n)
                fonte = "historico_processado"
            pdf.paragraph(f"Fonte dos partidos: {fonte}.")
            add_polars_party_section(pdf, "Brasil por partido", partidos, args.top_n)
        if modalidade_allows(modalidade, "perfil"):
            perfis = polars_to_pandas(store.top_profiles("brasil", limit=args.top_n))
            pdf.table("Top perfis Brasil", perfis, ["ano", "perfil_combinado", "share_perfil", "eleitorado"], limit=args.top_n)
            perfil_discreto = polars_to_pandas(store.profile_distribution(limit=args.top_n * 2))
            pdf.table("Perfil discreto Brasil", perfil_discreto, ["ano", "dimensao_perfil", "valor_perfil", "share", "peso"], limit=args.top_n)
        if modalidade_allows(modalidade, "candidato"):
            pdf.table("Resultado por candidato - Brasil", polars_to_pandas(store.entity_results(entity="candidato", nivel="brasil", limit=args.top_n)), ["ano", "entidade", "votos", "share_votos", "rank_entidade"], limit=args.top_n)
            pdf.table("Perfil por candidato - Brasil", polars_to_pandas(store.entity_profiles(entity="candidato", nivel="brasil", limit=args.top_n)), ["ano", "entidade", "perfil_combinado", "share_perfil_na_entidade"], limit=args.top_n)
        if modalidade_allows(modalidade, "cluster"):
            add_polars_cluster_section(pdf, "Clusters Brasil - eleitorado", store.cluster_personas(tipo="eleitores", nivel="brasil", limit=args.top_n), args.top_n)
            add_polars_cluster_section(pdf, "Clusters Brasil - eleitorado + partido", store.cluster_personas(tipo="resultado", nivel="brasil", limit=args.top_n), args.top_n)

        ufs = [x.strip().upper() for x in str(args.ufs or "").split(",") if x.strip()]
        if not ufs:
            ufs = list(progress.get("ufs_concluidas") or [])[:10]
        for uf in ufs:
            if not pdf.can_add_page():
                break
            pdf.new_page(f"Estado {uf}")
            metrics = polars_to_pandas(store.metrics_by_year("timeline_uf", uf=uf))
            pdf.table(f"Metricas {uf}", metrics, ["ano", "eleitorado", "comparecimento_estimado", "abstencao_estimado"], limit=args.top_n)
            if modalidade_allows(modalidade, "partido"):
                state_parties = store.party_prediction("sim_partidos_estados", uf=uf, cenario="base", limit=args.top_n)
                state_source = "simulacao_2026"
                if getattr(state_parties, "height", 0) == 0:
                    state_parties = store.historical_party_results(uf=uf, limit=args.top_n)
                    state_source = "historico_processado"
                pdf.paragraph(f"Fonte dos partidos em {uf}: {state_source}.")
                add_polars_party_section(pdf, f"{uf} por partido", state_parties, args.top_n)
            if modalidade_allows(modalidade, "perfil"):
                state_profiles = polars_to_pandas(store.top_profiles("estado", uf=uf, limit=args.top_n))
                pdf.table(f"Top perfis {uf}", state_profiles, ["ano", "perfil_combinado", "share_perfil", "eleitorado"], limit=args.top_n)
            if modalidade_allows(modalidade, "candidato"):
                pdf.table(f"Resultado por candidato - {uf}", polars_to_pandas(store.entity_results(entity="candidato", nivel="estado", uf=uf, limit=args.top_n)), ["ano", "entidade", "votos", "share_votos", "rank_entidade"], limit=args.top_n)
                pdf.table(f"Perfil por candidato - {uf}", polars_to_pandas(store.entity_profiles(entity="candidato", nivel="estado", uf=uf, limit=args.top_n)), ["ano", "entidade", "perfil_combinado", "share_perfil_na_entidade"], limit=args.top_n)
            if modalidade_allows(modalidade, "cluster"):
                add_polars_cluster_section(pdf, f"Clusters {uf} - eleitorado", store.cluster_personas(tipo="eleitores", nivel="estado", uf=uf, limit=args.top_n), args.top_n)
                add_polars_cluster_section(pdf, f"Clusters {uf} - eleitorado + partido", store.cluster_personas(tipo="resultado", nivel="estado", uf=uf, limit=args.top_n), args.top_n)

            municipios = store.municipios(uf)[: int(args.municipios_por_uf or 0)] if modalidade_allows(modalidade, "municipio") else []
            for municipio in municipios:
                if not pdf.can_add_page():
                    break
                label = municipio.get("label", "")
                value = municipio.get("value", "")
                pdf.new_page(f"Municipio {label} - {uf}")
                if modalidade_allows(modalidade, "partido"):
                    mun_parties = store.party_prediction("sim_partidos_municipios", uf=uf, municipio=value, cenario="base", limit=args.top_n)
                    mun_source = "simulacao_2026"
                    if getattr(mun_parties, "height", 0) == 0:
                        mun_parties = store.historical_party_results(uf=uf, municipio=value, limit=args.top_n)
                        mun_source = "historico_processado"
                    pdf.paragraph(f"Fonte dos partidos em {label}: {mun_source}.")
                    add_polars_party_section(pdf, f"{label} por partido", mun_parties, args.top_n)
                if modalidade_allows(modalidade, "perfil"):
                    mun_profiles = polars_to_pandas(store.top_profiles("municipio", uf=uf, municipio=value, limit=args.top_n))
                    pdf.table(f"Top perfis {label}", mun_profiles, ["ano", "perfil_combinado", "share_perfil", "eleitorado"], limit=args.top_n)
                if modalidade_allows(modalidade, "candidato"):
                    pdf.table(f"Resultado por candidato - {label}", polars_to_pandas(store.entity_results(entity="candidato", nivel="municipio", uf=uf, municipio=value, limit=args.top_n)), ["ano", "entidade", "votos", "share_votos", "rank_entidade"], limit=args.top_n)
                    pdf.table(f"Perfil por candidato - {label}", polars_to_pandas(store.entity_profiles(entity="candidato", nivel="municipio", uf=uf, municipio=value, limit=args.top_n)), ["ano", "entidade", "perfil_combinado", "share_perfil_na_entidade"], limit=args.top_n)
                if modalidade_allows(modalidade, "cluster"):
                    add_polars_cluster_section(pdf, f"Clusters {label} - eleitorado", store.cluster_personas(tipo="eleitores", nivel="municipio", uf=uf, municipio=value, limit=args.top_n), args.top_n)
                    add_polars_cluster_section(pdf, f"Clusters {label} - eleitorado + partido", store.cluster_personas(tipo="resultado", nivel="municipio", uf=uf, municipio=value, limit=args.top_n), args.top_n)

        pdf.new_page("Notas finais")
        pdf.paragraph(
            "O modo Polars do PDF usa a mesma arquitetura da API: consultas lazy sobre Parquet, filtros territoriais antes da materializacao e graficos desenhados com ReportLab."
        )
    finally:
        pdf.close()
        logger.event("relatorio", "fim", saida=str(out), paginas=pdf.page, query_engine="polars", modalidade=modalidade, log_texto=str(logger.text_path), log_eventos=str(logger.jsonl_path), log_graficos=str(logger.graphs_path))
    return out


def build_report(args: argparse.Namespace) -> Path:
    modalidade = normalize_modalidade(getattr(args, "modalidade_analise", "completa"))
    run = Path(args.run).expanduser().resolve()
    out = Path(args.out).expanduser() if args.out else run / "relatorios" / "relatorio_completo_eleitoral.pdf"
    if not out.is_absolute():
        out = (Path.cwd() / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if getattr(args, "query_engine", "polars") == "polars" and polars_available():
        if getattr(args, "pdf_separado_por_nivel", False):
            return build_report_polars_split(args, run, out)
        return build_report_polars(args, run, out)

    log_dir = Path(args.log_dir).expanduser() if args.log_dir else None
    if log_dir and not log_dir.is_absolute():
        log_dir = (Path.cwd() / log_dir).resolve()
    logger = PdfRunLogger(out, verbose=not args.quiet, log_dir=log_dir)
    logger.event("relatorio", "inicio", run=str(run), saida=str(out), modalidade=modalidade, max_pages=args.max_pages, top_n=args.top_n)

    store = DuckStore(run, threads=args.duckdb_threads, logger=logger)
    pdf = PdfReport(out, f"Relatorio eleitoral - {MODE_LABELS.get(modalidade, modalidade)}", max_pages=args.max_pages, logger=logger)

    def run_section(name: str, fn: Any, *fn_args: Any) -> None:
        started = time.perf_counter()
        logger.event("secao", "inicio", nome=name)
        try:
            fn(*fn_args)
            logger.event("secao", "fim", nome=name, duracao_segundos=round(time.perf_counter() - started, 3), pagina_atual=pdf.page)
        except Exception as exc:
            logger.event("secao", "erro", nome=name, erro=str(exc), duracao_segundos=round(time.perf_counter() - started, 3))
            pdf.new_page(f"Erro na secao {name}")
            pdf.paragraph(f"A secao {name} nao foi gerada por erro: {exc}")

    try:
        run_section("capa", add_cover, pdf, run)
        run_section("metodologia", add_methodology, pdf)
        run_section("como_graficos_sao_gerados", add_graph_generation_methodology, pdf, logger)
        run_section("inventario", add_inventory, pdf, store)
        run_section("analise_nacional", add_national_analysis, pdf, store, args)
        if modalidade_allows(modalidade, "perfil"):
            run_section("perfil_eleitor", add_profile_analysis, pdf, store, args)
        if modalidade_allows(modalidade, "partido") or modalidade_allows(modalidade, "candidato"):
            run_section("partidos_candidatos", add_party_and_candidate_analysis, pdf, store, args)
        if modalidade_allows(modalidade, "cluster"):
            run_section("clusters", add_cluster_analysis, pdf, store, args)
        if modalidade_allows(modalidade, "simulacao"):
            run_section("simulacao", add_simulation_analysis, pdf, store, args)
        run_section("estados", add_state_pages, pdf, store, args)
        if modalidade_allows(modalidade, "municipio"):
            run_section("municipios", add_municipality_pages, pdf, store, args)
        if args.incluir_secoes and modalidade_allows(modalidade, "secao"):
            run_section("secoes", add_section_pages, pdf, store, args)
        pdf.new_page("Notas finais")
        pdf.paragraph(
            "Este relatorio foi gerado automaticamente a partir dos dados tratados em Parquet. "
            "As associacoes entre perfil eleitoral e voto sao agregadas por territorio e nao representam voto individual declarado."
        )
    finally:
        pdf.close()
        store.close()
        logger.event("relatorio", "fim", saida=str(out), paginas=pdf.page, modalidade=modalidade, log_texto=str(logger.text_path), log_eventos=str(logger.jsonl_path), log_graficos=str(logger.graphs_path))
    return out


def add_cover(pdf: PdfReport, run: Path) -> None:
    pdf.new_page("")
    pdf.canvas.setFillColor(colors.HexColor("#0f172a"))
    pdf.canvas.rect(0, 0, pdf.width, pdf.height, stroke=0, fill=1)
    pdf.canvas.setFillColor(colors.white)
    pdf.canvas.setFont("Helvetica-Bold", 26)
    pdf.canvas.drawString(pdf.margin, pdf.height - 145, "Relatorio Completo Eleitoral")
    pdf.canvas.setFont("Helvetica", 13)
    pdf.canvas.drawString(pdf.margin, pdf.height - 175, "Analise data-driven dos Dados Abertos do TSE")
    pdf.canvas.drawString(pdf.margin, pdf.height - 200, f"Base: {run}")
    pdf.canvas.drawString(pdf.margin, pdf.height - 225, f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    pdf.canvas.setFont("Helvetica", 9)
    pdf.canvas.drawString(pdf.margin, 72, "Fonte dos dados: Tribunal Superior Eleitoral - Portal de Dados Abertos do TSE")
    pdf.canvas.drawString(pdf.margin, 56, "Nota: dados agregados; correlacoes territoriais nao indicam voto individual.")


def add_methodology(pdf: PdfReport) -> None:
    pdf.new_page("Metodologia")
    paragraphs = [
        "Os dados foram obtidos dos Dados Abertos do TSE, extraidos de arquivos ZIP oficiais, convertidos de CSV para JSON/JSONL e posteriormente transformados em Parquet.",
        "A organizacao segue arquitetura bronze, prata e ouro. A bronze preserva dados normalizados proximos da origem; a prata limpa e padroniza dominios como eleitorado, candidatos e resultados; a ouro contem tabelas prontas para consulta, graficos, correlacoes e simulacao.",
        "A correlacao usa codigos eleitorais: ano, UF, codigo de municipio, zona, secao, cargo e turno. Isso permite relacionar eleitorado, resultados e candidatos sem depender de texto livre.",
        "As analises de perfil priorizam variaveis discretas: faixa etaria, sexo/genero, escolaridade, estado civil, raca/cor quando disponivel, partido, candidato e recortes territoriais.",
        "Clusters sao gerados com foco em variaveis discretas. A tecnica do cotovelo orienta a escolha de K quando os artefatos de clustering estao disponiveis.",
        "A simulacao de 2026 usa historico de votos, shares, swing temporal e Monte Carlo. Ela gera cenarios por partido para Brasil, UF e municipio quando os dados tratados existem.",
    ]
    for text in paragraphs:
        pdf.paragraph(text)


def add_graph_generation_methodology(pdf: PdfReport, logger: PdfRunLogger) -> None:
    pdf.new_page("Como os graficos do PDF sao gerados")
    pdf.paragraph(
        "Os graficos do PDF sao criados diretamente com ReportLab, sem transformar o PDF em HTML. "
        "O gerador consulta os Parquets com DuckDB, transforma cada resultado em linhas agregadas e desenha barras, cards e tabelas no canvas do PDF."
    )
    pdf.subheading("Fluxo de cada grafico")
    steps = [
        "1. Escolhe a camada de dados: normalmente ouro; quando a simulacao nao existe, usa os dados historicos ja processados.",
        "2. Monta uma consulta DuckDB sobre read_parquet ou read_csv_auto, sem carregar a base inteira antes da consulta.",
        "3. Converte o resultado em DataFrame pequeno, limitado por --top-n e pelos limites de cada secao.",
        "4. Remove labels nulos, 'sem valor', codigos sem legenda e categorias vazias.",
        "5. Para grafico de barra horizontal, calcula max_valor e desenha cada barra com canvas.rect.",
        "6. A largura da barra e proporcional: largura = largura_util * valor / max_valor.",
        "7. Cada grafico grava um evento em *_graficos.jsonl com titulo, linhas usadas, escala, metodo e dados plotados.",
    ]
    for step in steps:
        pdf.paragraph(step)
    pdf.subheading("Arquivos de log gerados")
    pdf.paragraph(f"Log textual: {logger.text_path}")
    pdf.paragraph(f"Eventos estruturados: {logger.jsonl_path}")
    pdf.paragraph(f"Manifesto dos graficos: {logger.graphs_path}")
    pdf.paragraph(
        "O manifesto dos graficos e o arquivo mais importante para auditoria visual: cada linha JSONL descreve um grafico, "
        "incluindo o tipo, pagina, top_n, quantidade de linhas recebidas/usadas e os valores efetivamente desenhados."
    )


def add_inventory(pdf: PdfReport, store: DuckStore) -> None:
    pdf.new_page("Inventario da base")
    items = []
    for key in ["municipal", "timeline_nacional", "timeline_uf", "timeline_municipal", "perfil_ano", "perfil_partido", "vencedor_secao", "base_gold"]:
        items.append((key, fmt_int(store.count_rows(key))))
    pdf.cards(items, columns=2)
    ufs = store.distinct("municipal", "uf", limit=40) or store.distinct("timeline_uf", "uf", limit=40)
    years = store.distinct("timeline_nacional", "ano", limit=40) or store.distinct("timeline_uf", "ano", limit=40)
    pdf.paragraph(f"UFs detectadas: {', '.join(ufs) if ufs else 'nao disponivel'}")
    pdf.paragraph(f"Anos detectados: {', '.join(years) if years else 'nao disponivel'}")


def add_national_analysis(pdf: PdfReport, store: DuckStore, args: argparse.Namespace) -> None:
    pdf.new_page("Analise nacional")
    expr = store.expr("timeline_nacional")
    if expr:
        df = store.query(f"select * from {expr} order by ano")
        if not df.empty and "erro" not in df.columns:
            cards = []
            for col in ["eleitorado", "comparecimento_estimado", "abstencao_estimado"]:
                if col in df.columns:
                    cards.append((col, fmt_int(pd.to_numeric(df[col], errors="coerce").sum())))
            pdf.cards(cards, columns=3)
            rows = []
            for _, r in df.iterrows():
                val = pd.to_numeric(r.get("eleitorado"), errors="coerce")
                rows.append((str(r.get("ano", "")), float(val if pd.notna(val) else 0), fmt_int(val)))
            pdf.hbar("Eleitorado por ano", rows, max_rows=args.top_n)
            pdf.table("Timeline nacional", df, ["ano", "eleitorado", "comparecimento_estimado", "abstencao_estimado"], limit=args.top_n)
        else:
            pdf.paragraph("Timeline nacional ainda nao disponivel.")
    else:
        pdf.paragraph("Tabela timeline_nacional nao encontrada.")


def add_profile_analysis(pdf: PdfReport, store: DuckStore, args: argparse.Namespace) -> None:
    pdf.new_page("Perfil do eleitor")
    expr = store.expr("perfil_ano")
    if expr:
        df = store.query(
            f"""
            select ano, dimensao_perfil, valor_perfil, eleitorado, share_eleitorado_ano, rank_dimensao_ano
            from {expr}
            where rank_dimensao_ano <= {int(args.top_n)}
            order by ano, dimensao_perfil, rank_dimensao_ano
            limit {int(args.top_n * 20)}
            """
        )
        pdf.table("Perfil do eleitor por ano", df, ["ano", "dimensao_perfil", "valor_perfil", "share_eleitorado_ano", "eleitorado"], limit=args.top_n * 4)
        if not df.empty and "share_eleitorado_ano" in df.columns:
            best = df.sort_values("share_eleitorado_ano", ascending=False).head(args.top_n)
            rows = [(f"{r.get('ano')} {r.get('dimensao_perfil')}={r.get('valor_perfil')}", pd.to_numeric(r.get("share_eleitorado_ano"), errors="coerce"), fmt_pct(r.get("share_eleitorado_ano"))) for _, r in best.iterrows()]
            pdf.hbar("Maiores perfis por dimensao", rows, max_rows=args.top_n)
    else:
        pdf.paragraph("Perfil por ano nao encontrado.")

    expr = store.expr("top10_perfis")
    if expr:
        df = store.query(
            f"""
            select nivel, ano, uf, cd_municipio, nm_municipio, perfil_combinado, eleitorado, share_perfil, rank_perfil_ano
            from {expr}
            order by nivel, ano, uf, cd_municipio, rank_perfil_ano
            limit {int(args.top_n * 30)}
            """
        )
        pdf.table("Top perfis Brasil/UF/municipio", df, ["nivel", "ano", "uf", "nm_municipio", "perfil_combinado"], limit=args.top_n * 4)


def add_party_and_candidate_analysis(pdf: PdfReport, store: DuckStore, args: argparse.Namespace) -> None:
    modalidade = normalize_modalidade(getattr(args, "modalidade_analise", "completa"))
    pdf.new_page("Partidos e candidatos")
    for key, label in [("perfil_partido", "Perfil do eleitor por partido"), ("perfil_candidato", "Perfil do eleitor por candidato")]:
        if key == "perfil_partido" and not modalidade_allows(modalidade, "partido"):
            continue
        if key == "perfil_candidato" and not modalidade_allows(modalidade, "candidato"):
            continue
        expr = store.expr(key)
        if not expr:
            pdf.paragraph(f"{label}: tabela nao encontrada.")
            continue
        df = store.query(
            f"""
            select *
            from {expr}
            where {sql_meaningful('entidade') if key in {'perfil_partido', 'perfil_candidato'} else '1=1'}
            limit {int(args.top_n * 20)}
            """
        )
        pdf.table(label, df, ["nivel", "ano", "uf", "nm_municipio", "entidade", "perfil_combinado", "share_perfil_na_entidade"], limit=args.top_n * 3)


def add_cluster_analysis(pdf: PdfReport, store: DuckStore, args: argparse.Namespace) -> None:
    pdf.new_page("Clusters")
    any_cluster = False
    for key, label in [("cluster_voter_personas", "Clusters do eleitorado"), ("cluster_result_personas", "Clusters eleitorado + resultado"), ("cluster_elbow", "Tecnica do cotovelo")]:
        df = store.table(key, limit=args.top_n * 10)
        if df.empty:
            continue
        any_cluster = True
        pdf.table(label, df, list(df.columns[:6]), limit=args.top_n)
    if not any_cluster:
        pdf.paragraph("Clusters ainda nao encontrados nesta base. Gere a analise global/clustering para preencher esta secao.")


def add_simulation_analysis(pdf: PdfReport, store: DuckStore, args: argparse.Namespace) -> None:
    pdf.new_page("Simulacao 2026")
    for key, label in [
        ("sim_partidos_brasil", "Partidos 2026 - Brasil"),
        ("sim_partidos_estados", "Partidos 2026 - Estados"),
        ("sim_partidos_municipios", "Partidos 2026 - Municipios"),
    ]:
        expr = store.expr(key)
        if not expr:
            pdf.paragraph(f"{label}: tabela nao encontrada.")
            continue
        df = store.query(
            f"""
            select *
            from {expr}
            where {sql_meaningful('partido')}
            order by share_pred_2026 desc nulls last
            limit {int(args.top_n * 5)}
            """
        )
        pdf.table(label, df, ["cenario", "uf", "nm_municipio", "partido", "share_pred_2026", "perfil_eleitor_2026"], limit=args.top_n)
        if "share_pred_2026" in df.columns:
            rows = [(str(r.get("partido", "")), pd.to_numeric(r.get("share_pred_2026"), errors="coerce"), fmt_pct(r.get("share_pred_2026"))) for _, r in df.head(args.top_n).iterrows()]
            pdf.hbar(label, rows, max_rows=args.top_n)


def selected_ufs(store: DuckStore, args: argparse.Namespace) -> list[str]:
    if args.ufs:
        return [x.strip().upper() for x in args.ufs.split(",") if x.strip()]
    return store.distinct("timeline_uf", "uf", limit=30) or store.distinct("municipal", "uf", limit=30)


def add_state_pages(pdf: PdfReport, store: DuckStore, args: argparse.Namespace) -> None:
    modalidade = normalize_modalidade(getattr(args, "modalidade_analise", "completa"))
    for uf in selected_ufs(store, args):
        if not pdf.can_add_page():
            return
        pdf.new_page(f"Estado {uf}")
        expr = store.expr("timeline_uf")
        if expr:
            df = store.query(f"select * from {expr} where uf = {lit(uf)} order by ano limit 100")
            pdf.table(f"Timeline {uf}", df, ["ano", "uf", "eleitorado", "comparecimento_estimado", "abstencao_estimado"], limit=args.top_n)
        expr = store.expr("sim_partidos_estados")
        if expr and modalidade_allows(modalidade, "partido"):
            df = store.query(
                f"""
                select *
                from {expr}
                where uf = {lit(uf)} and {sql_meaningful('partido')}
                order by share_pred_2026 desc nulls last
                limit {int(args.top_n)}
                """
            )
            pdf.table(f"Simulacao partidaria {uf}", df, ["cenario", "partido", "share_pred_2026", "perfil_eleitor_2026"], limit=args.top_n)


def add_municipality_pages(pdf: PdfReport, store: DuckStore, args: argparse.Namespace) -> None:
    modalidade = normalize_modalidade(getattr(args, "modalidade_analise", "completa"))
    if not modalidade_allows(modalidade, "municipio"):
        return
    expr = store.expr("municipal")
    if not expr:
        return
    for uf in selected_ufs(store, args):
        municipios = store.query(
            f"""
            select uf, cd_municipio, nm_municipio, sum(eleitorado) as eleitorado
            from {expr}
            where uf = {lit(uf)}
            group by all
            order by eleitorado desc nulls last
            limit {int(args.municipios_por_uf)}
            """
        )
        for _, mun in municipios.iterrows():
            if not pdf.can_add_page():
                return
            cd = meaningful(mun.get("cd_municipio"))
            name = meaningful(mun.get("nm_municipio")) or cd
            pdf.new_page(f"Municipio {name} - {uf}")
            pdf.cards([("UF", uf), ("Municipio", name), ("Eleitorado", fmt_int(mun.get("eleitorado")))], columns=3)
            sim_expr = store.expr("sim_partidos_municipios")
            if sim_expr and cd and modalidade_allows(modalidade, "partido"):
                df = store.query(
                    f"""
                    select *
                    from {sim_expr}
                    where uf = {lit(uf)} and cast(cd_municipio as varchar) = {lit(cd)} and {sql_meaningful('partido')}
                    order by share_pred_2026 desc nulls last
                    limit {int(args.top_n)}
                    """
                )
                pdf.table("Simulacao partidaria municipal", df, ["cenario", "partido", "share_pred_2026", "perfil_eleitor_2026"], limit=args.top_n)
            top_expr = store.expr("top10_perfis")
            if top_expr and cd and modalidade_allows(modalidade, "perfil"):
                df = store.query(
                    f"""
                    select *
                    from {top_expr}
                    where nivel = 'municipio' and uf = {lit(uf)} and cast(cd_municipio as varchar) = {lit(cd)}
                    order by ano, rank_perfil_ano
                    limit {int(args.top_n)}
                    """
                )
                pdf.table("Top perfis do municipio", df, ["ano", "perfil_combinado", "share_perfil", "eleitorado"], limit=args.top_n)


def add_section_pages(pdf: PdfReport, store: DuckStore, args: argparse.Namespace) -> None:
    expr = store.expr("vencedor_secao")
    if not expr:
        return
    for uf in selected_ufs(store, args):
        df = store.query(
            f"""
            select *
            from {expr}
            where uf = {lit(uf)}
            order by ano, cd_municipio, zona, secao
            limit {int(args.secoes_por_uf)}
            """
        )
        if df.empty:
            continue
        if not pdf.can_add_page():
            return
        pdf.new_page(f"Amostra de secoes - {uf}")
        pdf.table("Vencedores por secao", df, ["ano", "cd_municipio", "zona", "secao", "partido_vencedor", "votos_vencedor", "share_vencedor"], limit=args.secoes_por_uf)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera relatorio PDF completo a partir da camada ouro do banco eleitoral.")
    parser.add_argument("--run", default="dados/banco_eleitoral", help="Pasta do banco/run. Ex.: dados/banco_eleitoral")
    parser.add_argument("--out", default="", help="Arquivo PDF de saida. Padrao: <run>/relatorios/relatorio_completo_eleitoral.pdf")
    parser.add_argument("--modalidade-analise", choices=ANALYSIS_MODES, default="completa", help="Recorte do PDF: completa, estados_brasil, eleitor, candidato, eleitor_partido ou eleitor_candidato_partido.")
    parser.add_argument("--max-pages", type=int, default=1000, help="Limite maximo de paginas do PDF.")
    parser.add_argument("--top-n", type=int, default=15, help="Quantidade de itens por ranking/grafico.")
    parser.add_argument("--ufs", default="", help="Lista de UFs separadas por virgula. Vazio usa todas detectadas.")
    parser.add_argument("--municipios-por-uf", type=int, default=20, help="Quantidade de municipios detalhados por UF.")
    parser.add_argument("--incluir-secoes", action="store_true", help="Inclui paginas com amostra de secoes eleitorais.")
    parser.add_argument("--secoes-por-uf", type=int, default=80, help="Quantidade de secoes listadas por UF quando --incluir-secoes.")
    parser.add_argument("--duckdb-threads", type=int, default=4, help="Threads DuckDB usadas nas consultas do relatorio.")
    parser.add_argument("--query-engine", choices=["polars", "duckdb"], default="polars", help="Engine de consulta do PDF. Padrao: polars; duckdb fica como fallback legado.")
    parser.add_argument("--log-dir", default="", help="Pasta para logs detalhados do PDF. Padrao: <saida_pdf>/logs.")
    parser.add_argument("--pdf-separado-por-nivel", action="store_true", help="Gera PDFs separados: primeiro Brasil, depois um por estado e, se aplicavel, um por municipio.")
    parser.add_argument("--quiet", action="store_true", help="Reduz logs no terminal, mantendo arquivos de log.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = build_report(args)
    print(f"PDF gerado: {out}")


if __name__ == "__main__":
    main()
