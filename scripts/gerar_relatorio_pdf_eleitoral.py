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
    "perfil_partido": ["ouro/brasil/perfil_partido", "ouro/estadual/perfil_partido", "ouro/municipal/perfil_partido", "ouro/perfil_eleitor_por_partido", "ouro/perfil_eleitor_por_partido.parquet"],
    "perfil_candidato": ["ouro/brasil/perfil_candidato", "ouro/estadual/perfil_candidato", "ouro/municipal/perfil_candidato", "ouro/perfil_eleitor_por_candidato", "ouro/perfil_eleitor_por_candidato.parquet"],
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
        "ouro/brasil/clusters_eleitores",
        "ouro/estadual/clusters_eleitores",
        "ouro/municipal/clusters_eleitores",
        "global/correlacao_codigos/clusters/parquet/clusters_eleitores_personas.parquet",
        "global/correlacao_codigos/clusters/tabelas/clusters_eleitores_personas.csv",
    ],
    "cluster_result_personas": [
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
    parser.add_argument("--quiet", action="store_true", help="Reduz logs no terminal, mantendo arquivos de log.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = build_report(args)
    print(f"PDF gerado: {out}")


if __name__ == "__main__":
    main()
