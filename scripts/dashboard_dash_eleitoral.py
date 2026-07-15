from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import duckdb
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from dash import Dash, Input, Output, State, dash_table, dcc, html
    from flask import jsonify, request
except ModuleNotFoundError as exc:
    missing = exc.name or "dependencia"
    print(
        f"Dependencia ausente: {missing}\n"
        "Instale com:\n"
        "  python3 -m pip install dash plotly duckdb pyarrow pandas\n",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


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
    "nao informado.",
    "sem entidade",
    "sem_entidade",
    "geral",
}

TABLE_CANDIDATES: dict[str, list[str]] = {
    "catalogo": ["metadados/manifesto_arquivos.parquet", "global/tabelas/catalogo_processamento.csv"],
    "municipal": [
        "ouro/municipal/resumo",
        "ouro/retrato_municipal",
        "ouro/retrato_municipal.parquet",
        "global/parquet/retrato_municipal_global.parquet",
        "global/tabelas/retrato_municipal_global.csv",
    ],
    "timeline_nacional": [
        "ouro/brasil/resumo",
        "ouro/timeline_nacional.parquet",
        "global/parquet/timeline_nacional.parquet",
        "global/timeline/timeline_nacional.csv",
    ],
    "timeline_uf": [
        "ouro/estadual/resumo",
        "ouro/timeline_uf",
        "ouro/timeline_uf.parquet",
        "global/parquet/timeline_uf.parquet",
        "global/timeline/timeline_uf.csv",
    ],
    "timeline_municipal": [
        "ouro/municipal/resumo",
        "ouro/timeline_municipal",
        "ouro/timeline_municipal.parquet",
        "global/parquet/timeline_municipal.parquet",
        "global/timeline/timeline_municipal.csv",
    ],
    "perfil_ano": [
        "ouro/perfil_eleitor_por_ano",
        "ouro/brasil/perfil_eleitor",
        "ouro/estadual/perfil_eleitor",
        "ouro/municipal/perfil_eleitor",
        "ouro/perfil_eleitor_por_ano.parquet",
        "global/analise_eleitoral/parquet/perfil_eleitor_por_ano.parquet",
        "global/analise_eleitoral/perfil_eleitor_por_ano.csv",
    ],
    "perfil_partido": [
        "ouro/perfil_eleitor_por_partido",
        "ouro/brasil/perfil_partido",
        "ouro/estadual/perfil_partido",
        "ouro/municipal/perfil_partido",
        "ouro/perfil_eleitor_por_partido.parquet",
        "global/analise_eleitoral/parquet/perfil_eleitor_por_partido.parquet",
        "global/analise_eleitoral/perfil_eleitor_por_partido.csv",
    ],
    "perfil_candidato": [
        "ouro/perfil_eleitor_por_candidato",
        "ouro/brasil/perfil_candidato",
        "ouro/estadual/perfil_candidato",
        "ouro/municipal/perfil_candidato",
        "ouro/perfil_eleitor_por_candidato.parquet",
        "global/analise_eleitoral/parquet/perfil_eleitor_por_candidato.parquet",
        "global/analise_eleitoral/perfil_eleitor_por_candidato.csv",
    ],
    "perfil_do_candidato": [
        "global/analise_eleitoral/parquet/perfil_do_candidato_correlacionado_eleitorado.parquet",
        "global/analise_eleitoral/perfil_do_candidato_correlacionado_eleitorado.csv",
    ],
    "resultado_eleitorado": [
        "ouro/municipal/resultado_eleitorado_por_secao",
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
        "ouro/brasil/perfil_eleitor",
        "ouro/estadual/perfil_eleitor",
        "ouro/municipal/perfil_eleitor",
        "ouro/top10_perfis_federacao_estado_municipio.parquet",
        "global/analise_eleitoral/parquet/top10_perfis_federacao_estado_municipio.parquet",
        "global/analise_eleitoral/top10_perfis_federacao_estado_municipio.csv",
    ],
    "vencedor_secao": [
        "ouro/resultados_vencedores_secao",
        "global/analise_eleitoral/parquet/vencedor_por_secao.parquet",
        "global/analise_eleitoral/vencedor_por_secao.csv",
    ],
    "banco_prata_eleitorado": ["prata/eleitorado"],
    "banco_prata_candidatos": ["prata/candidatos"],
    "banco_prata_resultados": ["prata/resultados_votos"],
    "banco_ouro_base_gold": ["ouro/municipal/base_secao", "ouro/base_gold_global", "ouro/base_gold_global.parquet"],
    "cluster_voter_personas": [
        "ouro/brasil/clusters_eleitores",
        "ouro/estadual/clusters_eleitores",
        "ouro/municipal/clusters_eleitores",
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
        "ouro/brasil/clusters_eleitores_resultado",
        "ouro/estadual/clusters_eleitores_resultado",
        "ouro/municipal/clusters_eleitores_resultado",
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
    "cluster_elbow": [
        "global/correlacao_codigos/clusters/parquet/clusters_cotovelo_k.parquet",
        "global/correlacao_codigos/clusters/tabelas/clusters_cotovelo_k.csv",
    ],
    "correlacao_stats": [
        "global/correlacao_codigos/parquet/estatisticas_correlacionadas_por_ano.parquet",
        "global/correlacao_codigos/tabelas/estatisticas_correlacionadas_por_ano.csv",
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

TABLE_LABELS: dict[str, str] = {
    "catalogo": "Metadados - catalogo dos arquivos",
    "municipal": "Ouro - retrato municipal",
    "timeline_nacional": "Ouro - timeline nacional",
    "timeline_uf": "Ouro - timeline por UF",
    "timeline_municipal": "Ouro - timeline municipal",
    "perfil_ano": "Ouro - perfil do eleitor por ano",
    "perfil_partido": "Ouro - perfil do eleitor por partido",
    "perfil_candidato": "Ouro - perfil do eleitor por candidato",
    "perfil_do_candidato": "Ouro - perfil do candidato correlacionado",
    "resultado_eleitorado": "Ouro - resultado + eleitorado por secao",
    "comparativo_perfil": "Ouro - comparativo anual de perfis",
    "comparativo_partido": "Ouro - comparativo anual por partido",
    "comparativo_candidato": "Ouro - comparativo anual por candidato",
    "top10_perfis": "Ouro - top 10 perfis Brasil/UF/municipio",
    "vencedor_secao": "Ouro - vencedores por secao ja concluidos",
    "banco_prata_eleitorado": "Prata - eleitorado limpo",
    "banco_prata_candidatos": "Prata - candidatos limpos",
    "banco_prata_resultados": "Prata - resultados/votos limpos",
    "banco_ouro_base_gold": "Ouro - base gold global",
    "cluster_voter_personas": "Clusters - personas dos eleitores",
    "cluster_voter_year_region": "Clusters - eleitores por ano/regiao",
    "cluster_voter_discriminants": "Clusters - discriminantes dos eleitores",
    "cluster_result_personas": "Clusters - personas resultado + eleitor",
    "cluster_result_year_region": "Clusters - resultado por ano/regiao",
    "cluster_result_discriminants": "Clusters - discriminantes resultado + eleitor",
    "cluster_result_prediction": "Clusters - predicao 2026",
    "cluster_elbow": "Clusters - cotovelo K",
    "correlacao_stats": "Correlacao - estatisticas por ano",
    "sim_partidos_brasil": "Simulacao 2026 - partidos Brasil",
    "sim_partidos_estados": "Simulacao 2026 - partidos Estados",
    "sim_partidos_municipios": "Simulacao 2026 - partidos Municipios",
    "sim_partidos_correlacao": "Simulacao 2026 - correlacao historica",
}

UF_POINTS: dict[str, tuple[float, float, str]] = {
    "AC": (-8.77, -70.55, "Acre"),
    "AL": (-9.62, -36.82, "Alagoas"),
    "AP": (1.41, -51.77, "Amapa"),
    "AM": (-3.47, -65.10, "Amazonas"),
    "BA": (-12.96, -41.70, "Bahia"),
    "CE": (-5.20, -39.53, "Ceara"),
    "DF": (-15.83, -47.86, "Distrito Federal"),
    "ES": (-19.19, -40.34, "Espirito Santo"),
    "GO": (-15.98, -49.86, "Goias"),
    "MA": (-5.42, -45.44, "Maranhao"),
    "MT": (-12.64, -55.42, "Mato Grosso"),
    "MS": (-20.51, -54.54, "Mato Grosso do Sul"),
    "MG": (-18.10, -44.38, "Minas Gerais"),
    "PA": (-3.79, -52.48, "Para"),
    "PB": (-7.28, -36.72, "Paraiba"),
    "PR": (-24.89, -51.55, "Parana"),
    "PE": (-8.38, -37.86, "Pernambuco"),
    "PI": (-6.60, -42.28, "Piaui"),
    "RJ": (-22.25, -42.66, "Rio de Janeiro"),
    "RN": (-5.81, -36.59, "Rio Grande do Norte"),
    "RS": (-30.17, -53.50, "Rio Grande do Sul"),
    "RO": (-10.83, -63.34, "Rondonia"),
    "RR": (1.99, -61.33, "Roraima"),
    "SC": (-27.45, -50.95, "Santa Catarina"),
    "SP": (-22.19, -48.79, "Sao Paulo"),
    "SE": (-10.57, -37.45, "Sergipe"),
    "TO": (-10.25, -48.25, "Tocantins"),
}

UF_NAME_TO_CODE = {
    code.lower(): code for code in UF_POINTS
} | {
    name.lower(): code for code, (_, _, name) in UF_POINTS.items()
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dashboard Dash/DuckDB para os Parquets do pipeline eleitoral.")
    parser.add_argument("--run", default="", help="Pasta do run. Ex: resultados/completo ou completo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def resolve_run_path(value: str) -> Path:
    if not value:
        latest = latest_run()
        return latest if latest is not None else (Path.cwd() / "resultados").resolve()
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
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)[0] if runs else None


def qident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def lit(value: Any) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def as_posix(path: Path) -> str:
    return path.resolve().as_posix()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    lower = text.lower()
    if lower in NULL_WORDS:
        return ""
    if lower.endswith("_sem_valor") or lower.endswith(" sem valor"):
        return ""
    if lower.startswith("codigo ") and lower.replace("codigo ", "", 1).replace(".", "", 1).isdigit():
        return ""
    return text


def is_meaningful(value: Any) -> bool:
    return bool(clean_text(value))


def fmt_int(value: Any) -> str:
    try:
        return f"{float(value):,.0f}".replace(",", ".")
    except Exception:
        return "0"


def fmt_pct(value: Any) -> str:
    try:
        number = float(value)
    except Exception:
        return "-"
    if number > 1.5:
        return f"{number:.1f}%"
    return f"{number * 100:.1f}%"


def pct_expr(column: str) -> str:
    return f"try_cast(replace(cast({qident(column)} as varchar), ',', '.') as double)"


def year_sort_key(value: Any) -> tuple[int, Any]:
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)


def meaningful_sql(column: str) -> str:
    expr = f"lower(trim(cast({qident(column)} as varchar)))"
    nulls = ", ".join(lit(x) for x in sorted(NULL_WORDS))
    return f"{expr} not in ({nulls}) and {expr} not like '%sem valor%'"


class DuckStore:
    def __init__(self, run_path: Path, threads: int = 4):
        self.run_path = run_path
        self.con = duckdb.connect(database=":memory:")
        self.con.execute(f"PRAGMA threads={max(1, int(threads or 1))}")
        self.con.execute("PRAGMA preserve_insertion_order=false")

    def __enter__(self) -> "DuckStore":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def close(self) -> None:
        try:
            self.con.close()
        except Exception:
            pass

    def path_for(self, key: str) -> Path | None:
        for rel in TABLE_CANDIDATES.get(key, []):
            path = self.run_path / rel
            if path.is_dir():
                if next(path.rglob("*.parquet"), None) is not None:
                    return path
                continue
            elif path.exists():
                return path
        return None

    def local_path_from_manifest(self, value: Any) -> Path | None:
        text = str(value or "").strip()
        if not text:
            return None
        direct = Path(text)
        if direct.exists():
            return direct
        normalized = text.replace("\\", "/")
        if normalized.startswith("/mnt/") and len(normalized) > 7:
            drive = normalized[5].upper()
            converted = Path(f"{drive}:/" + normalized[7:])
            if converted.exists():
                return converted
        marker = "/dados/banco_eleitoral/"
        if marker in normalized:
            candidate = self.run_path / normalized.split(marker, 1)[1]
            if candidate.exists():
                return candidate
        marker = "/ouro/"
        if marker in normalized:
            candidate = self.run_path / ("ouro/" + normalized.split(marker, 1)[1])
            if candidate.exists():
                return candidate
        return direct

    def ouro_resultados_status(self) -> dict[str, Any]:
        path = self.run_path / "logs" / "ouro" / "ouro_resultados_status_fatias.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            return {"erro": str(exc), "status": []}

    def completed_resultado_paths(self) -> list[Path]:
        status = self.ouro_resultados_status()
        rows = status.get("status") if isinstance(status, dict) else None
        if not isinstance(rows, list):
            return []
        paths: list[Path] = []
        for row in rows:
            if not isinstance(row, dict) or str(row.get("status", "")).lower() != "concluido":
                continue
            path = self.local_path_from_manifest(row.get("saida"))
            if path is not None and path.exists():
                if path.is_dir():
                    paths.append(path)
                elif path.suffix.lower() == ".parquet":
                    paths.append(path)
        return sorted(set(paths), key=lambda p: as_posix(p))

    def ouro_resultados_summary(self) -> dict[str, Any]:
        status = self.ouro_resultados_status()
        rows = status.get("status") if isinstance(status, dict) else None
        rows = rows if isinstance(rows, list) else []
        concluidas = [r for r in rows if isinstance(r, dict) and str(r.get("status", "")).lower() == "concluido"]
        pendentes = [r for r in rows if isinstance(r, dict) and str(r.get("status", "")).lower() != "concluido"]
        ufs_concluidas = sorted({str(r.get("uf", "")) for r in concluidas if r.get("uf")})
        ufs_pendentes = sorted({str(r.get("uf", "")) for r in pendentes if r.get("uf")})
        return {
            "total": int(status.get("total") or len(rows) or 0) if isinstance(status, dict) else len(rows),
            "concluidas": int(status.get("concluidas") or len(concluidas)) if isinstance(status, dict) else len(concluidas),
            "pendentes": int(status.get("pendentes") or len(pendentes)) if isinstance(status, dict) else len(pendentes),
            "ufs_concluidas": ufs_concluidas,
            "ufs_pendentes": ufs_pendentes,
            "linhas_status": len(rows),
            "erro": status.get("erro", "") if isinstance(status, dict) else "",
        }

    def available_tables(self) -> list[str]:
        return [key for key in TABLE_CANDIDATES if self.path_for(key) is not None]

    def expr(self, key: str) -> str | None:
        if key == "vencedor_secao":
            completed = self.completed_resultado_paths()
            if completed:
                globs = [
                    as_posix(path / "**" / "*.parquet") if path.is_dir() else as_posix(path)
                    for path in completed
                ]
                return f"read_parquet([{', '.join(lit(g) for g in globs)}], union_by_name=true, hive_partitioning=true)"
        path = self.path_for(key)
        if path is None:
            return None
        quoted = lit(as_posix(path))
        if path.is_dir():
            glob = lit(as_posix(path / "**" / "*.parquet"))
            return f"read_parquet({glob}, union_by_name=true, hive_partitioning=true)"
        if path.suffix.lower() == ".parquet":
            return f"read_parquet({quoted}, union_by_name=true)"
        return f"read_csv_auto({quoted}, delim=';', header=true, all_varchar=true, ignore_errors=true)"

    def query(self, sql: str) -> pd.DataFrame:
        try:
            return self.con.execute(sql).fetchdf()
        except Exception as exc:
            return pd.DataFrame({"erro": [str(exc)]})

    @lru_cache(maxsize=256)
    def columns(self, key: str) -> tuple[str, ...]:
        expr = self.expr(key)
        if expr is None:
            return tuple()
        try:
            df = self.con.execute(f"DESCRIBE SELECT * FROM {expr} LIMIT 0").fetchdf()
            return tuple(str(x) for x in df["column_name"].tolist())
        except Exception:
            return tuple()

    @lru_cache(maxsize=256)
    def count_rows(self, key: str) -> int:
        expr = self.expr(key)
        if expr is None:
            return 0
        df = self.query(f"SELECT count(*) as n FROM {expr}")
        return int(pd.to_numeric(df.get("n"), errors="coerce").fillna(0).iloc[0]) if "n" in df else 0

    @lru_cache(maxsize=256)
    def distinct_values(self, key: str, column: str, where: str = "", limit: int = 7000) -> tuple[str, ...]:
        expr = self.expr(key)
        cols = self.columns(key)
        if expr is None or column not in cols:
            return tuple()
        clauses = [meaningful_sql(column)]
        if where:
            clauses.append(where)
        sql = (
            f"SELECT distinct cast({qident(column)} as varchar) as value "
            f"FROM {expr} WHERE {' and '.join(clauses)} "
            f"ORDER BY value LIMIT {int(limit)}"
        )
        df = self.query(sql)
        if "value" not in df:
            return tuple()
        return tuple(clean_text(x) for x in df["value"].tolist() if is_meaningful(x))


def table_query(store: DuckStore, key: str, limit: int = 200) -> pd.DataFrame:
    expr = store.expr(key)
    if expr is None:
        return pd.DataFrame()
    return store.query(f"SELECT * FROM {expr} LIMIT {int(limit)}")


def filter_clause(cols: tuple[str, ...], uf: str | None = None, municipio: str | None = None, ano: str | None = None, cenario: str | None = None) -> str:
    clauses: list[str] = []
    if uf and "uf" in cols:
        clauses.append(f"cast({qident('uf')} as varchar) = {lit(uf)}")
    if municipio:
        mun_parts = municipio.split("|", 1)
        if len(mun_parts) == 2 and "cd_municipio" in cols:
            clauses.append(f"cast({qident('cd_municipio')} as varchar) = {lit(mun_parts[0])}")
        elif "nm_municipio" in cols:
            clauses.append(f"cast({qident('nm_municipio')} as varchar) = {lit(municipio)}")
    if ano:
        if "ano" in cols:
            clauses.append(f"cast({qident('ano')} as varchar) = {lit(ano)}")
        elif "ano_correlacao" in cols:
            clauses.append(f"cast({qident('ano_correlacao')} as varchar) = {lit(ano)}")
        elif "ano_num" in cols:
            clauses.append(f"cast(try_cast({qident('ano_num')} as integer) as varchar) = {lit(ano)}")
    if cenario and "cenario" in cols:
        clauses.append(f"cast({qident('cenario')} as varchar) = {lit(cenario)}")
    return " and ".join(clauses)


def first_col(cols: tuple[str, ...], names: list[str]) -> str | None:
    for name in names:
        if name in cols:
            return name
    return None


def make_card(title: str, value: str, subtitle: str = "", tone: str = "") -> html.Div:
    return html.Div(
        className=f"metric-card {tone}",
        children=[
            html.Div(title, className="metric-title"),
            html.Div(value, className="metric-value"),
            html.Div(subtitle, className="metric-subtitle") if subtitle else None,
        ],
    )


def card_grid(cards: list[Any]) -> html.Div:
    return html.Div(cards, className="card-grid")


def render_search_prompt(tab: str = "") -> list[Any]:
    title = "Consulta sob demanda"
    subtitle = "A visao Brasil e pre-carregada. Para estados, municipios, clusters e simulacao, escolha os filtros no topo e clique em Buscar dados."
    return [
        section(
            title,
            [
                card_grid(
                    [
                        make_card("Status", "Aguardando busca", "nenhuma consulta pesada executada"),
                        make_card("Como usar", "1 clique", "selecione UF/municipio/ano/cenario e clique em Buscar dados"),
                        make_card("Modo", "leve", "os Parquets ficam no disco ate voce pedir"),
                    ]
                )
            ],
            subtitle,
        )
    ]


def section(title: str, children: list[Any], subtitle: str = "") -> html.Section:
    return html.Section(
        className="panel",
        children=[
            html.Div(
                className="panel-head",
                children=[
                    html.H2(title),
                    html.P(subtitle) if subtitle else None,
                ],
            ),
            *children,
        ],
    )


def graph_card(title: str, figure: go.Figure, subtitle: str = "") -> html.Div:
    return html.Div(
        className="graph-card",
        children=[
            html.Div([html.H3(title), html.P(subtitle) if subtitle else None], className="graph-head"),
            dcc.Graph(figure=figure, config={"displaylogo": False, "responsive": True}),
        ],
    )


def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    fig.update_layout(template="plotly_white", height=360, margin=dict(l=20, r=20, t=30, b=20))
    return fig


def style_figure(fig: go.Figure, height: int = 390) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=30, r=20, t=45, b=35),
        font=dict(family="Inter, Segoe UI, Arial", size=12, color="#172033"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hoverlabel=dict(bgcolor="#0f172a", font_color="white"),
    )
    return fig


def top_profiles(store: DuckStore, level: str, uf: str | None = None, municipio: str | None = None, ano: str | None = None, limit: int = 10) -> pd.DataFrame:
    key = "top10_perfis"
    expr = store.expr(key)
    cols = store.columns(key)
    if expr is None or "nivel" not in cols:
        return pd.DataFrame()
    selected = [
        c
        for c in [
            "nivel",
            "ano",
            "uf",
            "cd_municipio",
            "nm_municipio",
            "perfil_combinado",
            "share_perfil",
            "eleitorado",
            "rank_perfil_ano",
            "padrao_temporal",
            "descricao",
        ]
        if c in cols
    ]
    where = [f"lower(cast({qident('nivel')} as varchar)) = {lit(level.lower())}"]
    extra = filter_clause(cols, uf=uf, municipio=municipio, ano=ano)
    if extra:
        where.append(extra)
    if "perfil_combinado" in cols:
        where.append(meaningful_sql("perfil_combinado"))
    order = []
    if "rank_perfil_ano" in cols:
        order.append(f"{pct_expr('rank_perfil_ano')} asc")
    if "share_perfil" in cols:
        order.append(f"{pct_expr('share_perfil')} desc")
    if "eleitorado" in cols:
        order.append(f"{pct_expr('eleitorado')} desc")
    sql = (
        f"SELECT {', '.join(qident(c) for c in selected)} FROM {expr} "
        f"WHERE {' and '.join(where)} "
        f"ORDER BY {', '.join(order) if order else '1'} LIMIT {int(limit)}"
    )
    return store.query(sql)


def profile_distribution(store: DuckStore, ano: str | None = None, limit: int = 24) -> pd.DataFrame:
    key = "perfil_ano"
    expr = store.expr(key)
    cols = store.columns(key)
    needed = {"dimensao_perfil", "valor_perfil"}
    if expr is None or not needed.issubset(cols):
        return pd.DataFrame()
    metric_col = "eleitorado" if "eleitorado" in cols else None
    share_col = "share_eleitorado_ano" if "share_eleitorado_ano" in cols else None
    where = [
        meaningful_sql("valor_perfil"),
        "lower(cast(\"dimensao_perfil\" as varchar)) not like '%biometria%'",
        "lower(cast(\"dimensao_perfil\" as varchar)) not like '%data%'",
        "lower(cast(\"dimensao_perfil\" as varchar)) not like '%hora%'",
    ]
    extra = filter_clause(cols, ano=ano)
    if extra:
        where.append(extra)
    selected_year = "cast(\"ano\" as varchar) as ano," if "ano" in cols else "'' as ano,"
    metric_expr = f"sum({pct_expr(metric_col)})" if metric_col else "count(*)"
    share_expr = f"avg({pct_expr(share_col)})" if share_col else "null"
    sql = (
        f"SELECT {selected_year} cast(\"dimensao_perfil\" as varchar) as dimensao_perfil, "
        f"cast(\"valor_perfil\" as varchar) as valor_perfil, {metric_expr} as peso, {share_expr} as share "
        f"FROM {expr} WHERE {' and '.join(where)} "
        f"GROUP BY all HAVING count(distinct cast(\"valor_perfil\" as varchar)) > 0 "
        f"ORDER BY peso desc LIMIT {int(limit)}"
    )
    return store.query(sql)


def entity_profile_cards(store: DuckStore, key: str, level: str = "brasil", uf: str | None = None, municipio: str | None = None, ano: str | None = None, limit: int = 10) -> pd.DataFrame:
    expr = store.expr(key)
    cols = store.columns(key)
    if expr is None:
        return pd.DataFrame()
    entity_col = first_col(cols, ["entidade", "partido", "candidato"])
    profile_col = first_col(cols, ["perfil_combinado", "valor_perfil", "perfil_eleitorado_associado"])
    share_col = first_col(cols, ["share_perfil_na_entidade", "share_proxy_no_perfil", "share_perfil", "pct_votos_partido_2026", "share_pred_2026"])
    lift_col = first_col(cols, ["lift_perfil_entidade_proxy", "lift"])
    if entity_col is None or profile_col is None:
        return pd.DataFrame()
    selected = [
        c
        for c in [
            "nivel",
            "ano",
            "uf",
            "cd_municipio",
            "nm_municipio",
            entity_col,
            profile_col,
            share_col,
            lift_col,
            "padrao_temporal",
            "cargo",
            "turno",
            "interpretacao",
        ]
        if c and c in cols
    ]
    where = [meaningful_sql(entity_col), meaningful_sql(profile_col)]
    if "nivel" in cols:
        where.append(f"lower(cast({qident('nivel')} as varchar)) = {lit(level.lower())}")
    extra = filter_clause(cols, uf=uf, municipio=municipio, ano=ano)
    if extra:
        where.append(extra)
    order = []
    if lift_col:
        order.append(f"{pct_expr(lift_col)} desc nulls last")
    if share_col:
        order.append(f"{pct_expr(share_col)} desc nulls last")
    sql = (
        f"SELECT {', '.join(qident(c) for c in dict.fromkeys(selected))} FROM {expr} "
        f"WHERE {' and '.join(where)} "
        f"ORDER BY {', '.join(order) if order else '1'} LIMIT {int(limit)}"
    )
    df = store.query(sql)
    rename = {}
    if entity_col in df:
        rename[entity_col] = "entidade"
    if profile_col in df:
        rename[profile_col] = "perfil"
    if share_col in df:
        rename[share_col] = "share"
    if lift_col in df:
        rename[lift_col] = "lift"
    return df.rename(columns=rename)


def party_prediction(store: DuckStore, key: str, uf: str | None = None, municipio: str | None = None, cenario: str | None = "base", limit: int = 20) -> pd.DataFrame:
    expr = store.expr(key)
    cols = store.columns(key)
    if expr is None or "partido" not in cols:
        return pd.DataFrame()
    selected = [
        c
        for c in [
            "cenario",
            "nivel",
            "uf",
            "cd_municipio",
            "nm_municipio",
            "cargo",
            "turno",
            "partido",
            "share_pred_2026",
            "votos_pred_2026",
            "perfil_eleitor_2026",
            "tendencia_partido",
            "forca_correlacao_historica",
            "justificativa_previsao_partido_2026",
        ]
        if c in cols
    ]
    where = [meaningful_sql("partido")]
    extra = filter_clause(cols, uf=uf, municipio=municipio, cenario=cenario)
    if extra:
        where.append(extra)
    order_col = "share_pred_2026" if "share_pred_2026" in cols else ("votos_pred_2026" if "votos_pred_2026" in cols else "partido")
    order_expr = pct_expr(order_col) if order_col != "partido" else qident("partido")
    sql = (
        f"SELECT {', '.join(qident(c) for c in selected)} FROM {expr} "
        f"WHERE {' and '.join(where)} ORDER BY {order_expr} desc nulls last LIMIT {int(limit)}"
    )
    return store.query(sql)


def historical_party_results(store: DuckStore, uf: str | None = None, municipio: str | None = None, ano: str | None = None, limit: int = 20) -> pd.DataFrame:
    for key in ["vencedor_secao", "banco_prata_resultados"]:
        expr = store.expr(key)
        cols = store.columns(key)
        if expr is None:
            continue
        party_col = first_col(cols, ["partido_vencedor", "partido", "sg_partido", "nm_partido"])
        metric_col = first_col(cols, ["votos_vencedor", "votos", "qt_votos", "votos_total_secao", "qt_votos_nominais"])
        if party_col is None:
            continue
        metric_expr = f"sum({pct_expr(metric_col)})" if metric_col else "count(*)"
        where = [meaningful_sql(party_col)]
        extra = filter_clause(cols, uf=uf, municipio=municipio, ano=ano)
        if extra:
            where.append(extra)
        sql = f"""
        with agg as (
          select cast({qident(party_col)} as varchar) as partido,
                 {metric_expr} as votos_pred_2026
          from {expr}
          where {' and '.join(where)}
          group by 1
        )
        select partido,
               votos_pred_2026,
               votos_pred_2026 / nullif(sum(votos_pred_2026) over(), 0) as share_pred_2026,
               'historico_processado' as tendencia_partido,
               'dados reais ja processados' as forca_correlacao_historica,
               'Resultado historico parcial/total conforme fatias concluidas da camada ouro ou prata.' as perfil_eleitor_2026
        from agg
        where votos_pred_2026 > 0
        order by share_pred_2026 desc nulls last
        limit {int(limit)}
        """
        df = store.query(sql)
        if not df.empty and "erro" not in df:
            return df
    return pd.DataFrame()


def historical_state_party_map(store: DuckStore, ano: str | None = None) -> pd.DataFrame:
    expr = store.expr("vencedor_secao") or store.expr("banco_prata_resultados")
    key = "vencedor_secao" if store.expr("vencedor_secao") else "banco_prata_resultados"
    cols = store.columns(key)
    if expr is None or "uf" not in cols:
        return pd.DataFrame()
    party_col = first_col(cols, ["partido_vencedor", "partido", "sg_partido", "nm_partido"])
    metric_col = first_col(cols, ["votos_vencedor", "votos", "qt_votos", "votos_total_secao", "qt_votos_nominais"])
    if party_col is None:
        return pd.DataFrame()
    metric_expr = f"sum({pct_expr(metric_col)})" if metric_col else "count(*)"
    where = [meaningful_sql("uf"), meaningful_sql(party_col)]
    extra = filter_clause(cols, ano=ano)
    if extra:
        where.append(extra)
    sql = f"""
    with agg as (
      select cast({qident('uf')} as varchar) as uf,
             cast({qident(party_col)} as varchar) as partido,
             {metric_expr} as votos_pred_2026
      from {expr}
      where {' and '.join(where)}
      group by 1, 2
    ),
    ranked as (
      select *,
             votos_pred_2026 / nullif(sum(votos_pred_2026) over(partition by uf), 0) as share_pred_2026,
             row_number() over(partition by uf order by votos_pred_2026 desc) as rn
      from agg
    )
    select uf, partido, votos_pred_2026, share_pred_2026,
           'Resultado historico processado' as perfil_eleitor_2026
    from ranked
    where rn = 1
    """
    df = store.query(sql)
    if df.empty or "erro" in df:
        return pd.DataFrame()
    df = df.copy()
    df["uf_code"] = df["uf"].map(uf_code)
    df = df.loc[df["uf_code"].map(bool)].copy()
    if df.empty:
        return df
    df["lat"] = df["uf_code"].map(lambda x: UF_POINTS[x][0])
    df["lon"] = df["uf_code"].map(lambda x: UF_POINTS[x][1])
    df["uf_nome"] = df["uf_code"].map(lambda x: UF_POINTS[x][2])
    return df


def metrics_by_year(store: DuckStore, key: str, uf: str | None = None, municipio: str | None = None) -> pd.DataFrame:
    expr = store.expr(key)
    cols = store.columns(key)
    if expr is None:
        return pd.DataFrame()
    year_col = first_col(cols, ["ano", "ano_correlacao", "ano_num"])
    if year_col is None:
        return pd.DataFrame()
    metrics = [c for c in ["votos", "eleitorado", "comparecimento_estimado", "abstencao_estimado"] if c in cols]
    if not metrics:
        metrics = [c for c in ["votos_total", "eleitorado_total", "comparecimento_medio", "abstencao_media"] if c in cols]
    if not metrics:
        return pd.DataFrame()
    where = filter_clause(cols, uf=uf, municipio=municipio)
    select_metrics = ", ".join(f"sum({pct_expr(c)}) as {qident(c)}" for c in metrics)
    sql = (
        f"SELECT cast({qident(year_col)} as varchar) as ano, {select_metrics} FROM {expr} "
        f"{'WHERE ' + where if where else ''} GROUP BY 1 ORDER BY 1"
    )
    return store.query(sql)


def state_map_data(store: DuckStore, cenario: str | None = "base") -> pd.DataFrame:
    df = party_prediction(store, "sim_partidos_estados", cenario=cenario, limit=5000)
    if df.empty or "uf" not in df:
        return historical_state_party_map(store)
    df = df.copy()
    df["uf_code"] = df["uf"].map(uf_code)
    df = df.loc[df["uf_code"].map(bool)].copy()
    if df.empty:
        return df
    df["share_pred_2026"] = pd.to_numeric(df.get("share_pred_2026"), errors="coerce")
    df = df.sort_values(["uf_code", "share_pred_2026"], ascending=[True, False])
    df = df.drop_duplicates("uf_code")
    df["lat"] = df["uf_code"].map(lambda x: UF_POINTS[x][0])
    df["lon"] = df["uf_code"].map(lambda x: UF_POINTS[x][1])
    df["uf_nome"] = df["uf_code"].map(lambda x: UF_POINTS[x][2])
    return df


def uf_code(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    return UF_NAME_TO_CODE.get(text.lower(), text.upper() if text.upper() in UF_POINTS else "")


def make_map_figure(df: pd.DataFrame) -> go.Figure:
    if df is None or df.empty:
        return empty_figure("Sem dados estaduais de simulacao para o mapa.")
    fig = go.Figure()
    size = (pd.to_numeric(df.get("share_pred_2026"), errors="coerce").fillna(0) * 55 + 12).clip(10, 70)
    fig.add_trace(
        go.Scattergeo(
            lon=df["lon"],
            lat=df["lat"],
            text=df.get("uf_nome", df.get("uf", "")),
            mode="markers+text",
            textposition="top center",
            marker=dict(
                size=size,
                color=pd.to_numeric(df.get("share_pred_2026"), errors="coerce").fillna(0),
                colorscale="Tealgrn",
                colorbar=dict(title="Share"),
                line=dict(width=1, color="#0f172a"),
                opacity=0.88,
            ),
            customdata=df[["partido", "share_pred_2026", "perfil_eleitor_2026"]].fillna("").to_numpy()
            if {"partido", "share_pred_2026", "perfil_eleitor_2026"}.issubset(df.columns)
            else None,
            hovertemplate="<b>%{text}</b><br>Partido: %{customdata[0]}<br>Share: %{customdata[1]:.1%}<br>%{customdata[2]}<extra></extra>",
        )
    )
    fig.update_geos(
        scope="south america",
        projection_type="mercator",
        lataxis_range=[-35, 7],
        lonaxis_range=[-76, -32],
        showland=True,
        landcolor="#edf7f5",
        showcountries=True,
        countrycolor="#94a3b8",
        showocean=True,
        oceancolor="#f8fafc",
    )
    fig.update_layout(title="Mapa interativo por UF: partido projetado em 2026", height=560)
    return style_figure(fig, height=560)


def bar_profile_figure(df: pd.DataFrame, title: str) -> go.Figure:
    if df is None or df.empty:
        return empty_figure("Sem perfil discreto suficiente para graficar.")
    work = df.copy()
    for col in ["valor_perfil", "dimensao_perfil"]:
        if col not in work:
            return empty_figure("Tabela de perfil sem colunas esperadas.")
    work = work.loc[work["valor_perfil"].map(is_meaningful)]
    if work["valor_perfil"].nunique(dropna=True) <= 1:
        return empty_figure("Grafico omitido: apenas uma categoria relevante.")
    work["label"] = work["dimensao_perfil"].astype(str) + ": " + work["valor_perfil"].astype(str)
    ycol = "share" if "share" in work and pd.to_numeric(work["share"], errors="coerce").notna().any() else "peso"
    work[ycol] = pd.to_numeric(work[ycol], errors="coerce").fillna(0)
    work = work.sort_values(ycol, ascending=True).tail(18)
    fig = px.bar(work, x=ycol, y="label", color="dimensao_perfil", orientation="h", title=title)
    fig.update_traces(hovertemplate="%{y}<br>%{x:.2%}<extra></extra>" if ycol == "share" else "%{y}<br>%{x:,.0f}<extra></extra>")
    return style_figure(fig, height=430)


def line_metrics_figure(df: pd.DataFrame, title: str) -> go.Figure:
    if df is None or df.empty or "ano" not in df:
        return empty_figure("Sem serie temporal disponivel.")
    if df["ano"].nunique(dropna=True) <= 1:
        return empty_figure("Grafico omitido: serie temporal tem apenas um ano.")
    metrics = [c for c in df.columns if c != "ano" and pd.to_numeric(df[c], errors="coerce").notna().any()]
    if not metrics:
        return empty_figure("Sem metricas numericas essenciais para timeline.")
    work = df.melt(id_vars=["ano"], value_vars=metrics, var_name="metrica", value_name="valor")
    work["valor"] = pd.to_numeric(work["valor"], errors="coerce")
    fig = px.line(work, x="ano", y="valor", color="metrica", markers=True, title=title)
    return style_figure(fig, height=390)


def party_bar_figure(df: pd.DataFrame, title: str) -> go.Figure:
    if df is None or df.empty or "partido" not in df:
        return empty_figure("Sem dados de partido para graficar.")
    work = df.copy()
    work["share_pred_2026"] = pd.to_numeric(work.get("share_pred_2026"), errors="coerce").fillna(0)
    work = work.loc[work["partido"].map(is_meaningful)].sort_values("share_pred_2026", ascending=True).tail(15)
    if work["partido"].nunique(dropna=True) <= 1:
        return empty_figure("Grafico omitido: apenas um partido relevante.")
    fig = px.bar(work, x="share_pred_2026", y="partido", color="partido", orientation="h", title=title)
    fig.update_traces(hovertemplate="%{y}<br>%{x:.1%}<extra></extra>")
    fig.update_layout(showlegend=False)
    return style_figure(fig, height=430)


def cluster_discriminant_figure(store: DuckStore, key: str, title: str) -> go.Figure:
    expr = store.expr(key)
    cols = store.columns(key)
    if expr is None:
        return empty_figure("Sem tabela de discriminantes dos clusters.")
    value_col = first_col(cols, ["valor_legivel", "valor", "categoria_valor"])
    metric_col = first_col(cols, ["lift", "abs_lift", "qtd_no_cluster", "qtd_setores"])
    cluster_col = first_col(cols, ["cluster_global_discriminado", "cluster_perfil_eleitorado"])
    if not value_col or not metric_col:
        return empty_figure("Tabela de cluster sem valores discretos discriminantes.")
    selected = [c for c in [cluster_col, "campo", "variavel", value_col, metric_col] if c and c in cols]
    where = [meaningful_sql(value_col)]
    sql = (
        f"SELECT {', '.join(qident(c) for c in dict.fromkeys(selected))} FROM {expr} "
        f"WHERE {' and '.join(where)} ORDER BY {pct_expr(metric_col)} desc nulls last LIMIT 30"
    )
    df = store.query(sql)
    if df.empty or "erro" in df:
        return empty_figure("Sem discriminantes validos.")
    df = df.rename(columns={value_col: "valor", metric_col: "peso", cluster_col or "": "cluster"})
    df["valor"] = df["valor"].map(clean_text)
    df = df.loc[df["valor"].ne("")]
    df["peso"] = pd.to_numeric(df["peso"], errors="coerce").fillna(0)
    df["label"] = "C" + df.get("cluster", "").astype(str) + " | " + df["valor"].astype(str)
    if df["label"].nunique(dropna=True) <= 1:
        return empty_figure("Grafico omitido: uma unica categoria de cluster.")
    df = df.sort_values("peso", ascending=True).tail(20)
    fig = px.bar(df, x="peso", y="label", color="cluster", orientation="h", title=title)
    return style_figure(fig, height=470)


def profile_cards_from_df(df: pd.DataFrame, title: str, limit: int = 10) -> html.Div:
    cards: list[Any] = []
    if df is None or df.empty or "erro" in df:
        return html.Div([make_card(title, "Sem dados", "O run ainda nao gerou esta tabela.")], className="card-grid")
    work = df.head(limit).copy()
    for _, row in work.iterrows():
        profile = clean_text(row.get("perfil_combinado", "")) or clean_text(row.get("perfil", "")) or clean_text(row.get("descricao", ""))
        if not profile:
            continue
        year = clean_text(row.get("ano", ""))
        share = fmt_pct(row.get("share_perfil", row.get("share", "")))
        volume = fmt_int(row.get("eleitorado", ""))
        trend = clean_text(row.get("padrao_temporal", ""))
        scope = clean_text(row.get("uf", "")) or clean_text(row.get("nm_municipio", "")) or "Brasil"
        cards.append(
            html.Div(
                className="info-card",
                children=[
                    html.Div([html.Strong(scope), html.Span(year)], className="card-line"),
                    html.P(profile),
                    html.Div(
                        [
                            html.Span(f"share {share}", className="pill"),
                            html.Span(f"eleitorado {volume}", className="pill") if volume != "0" else None,
                            html.Span(trend, className="pill muted") if trend else None,
                        ],
                        className="pill-row",
                    ),
                ],
            )
        )
    if not cards:
        cards.append(make_card(title, "Sem perfil valido", "Dados nulos/sem valor foram filtrados."))
    return html.Div(cards, className="cards-compact")


def entity_cards(df: pd.DataFrame, title: str, entity_label: str = "Entidade", limit: int = 10) -> html.Div:
    cards: list[Any] = []
    if df is None or df.empty or "erro" in df:
        return html.Div([make_card(title, "Sem dados", "O run ainda nao gerou esta tabela.")], className="card-grid")
    for _, row in df.head(limit).iterrows():
        entity = clean_text(row.get("entidade", ""))
        profile = clean_text(row.get("perfil", ""))
        if not entity or not profile:
            continue
        share = fmt_pct(row.get("share", ""))
        lift = clean_text(row.get("lift", ""))
        interp = clean_text(row.get("interpretacao", ""))
        cards.append(
            html.Div(
                className="info-card",
                children=[
                    html.Div([html.Strong(entity), html.Span(entity_label)], className="card-line"),
                    html.P(profile),
                    html.Div(
                        [
                            html.Span(f"share {share}", className="pill"),
                            html.Span(f"lift {float(lift):.2f}x", className="pill") if lift else None,
                        ],
                        className="pill-row",
                    ),
                    html.Small(interp) if interp else None,
                ],
            )
        )
    if not cards:
        cards.append(make_card(title, "Sem perfil valido", "Dados nulos/sem valor foram filtrados."))
    return html.Div(cards, className="cards-compact")


def prediction_cards(df: pd.DataFrame, title: str, limit: int = 12) -> html.Div:
    cards: list[Any] = []
    if df is None or df.empty or "erro" in df:
        return html.Div([make_card(title, "Sem simulacao", "Rode o pipeline com --predict-2026.")], className="card-grid")
    for _, row in df.head(limit).iterrows():
        party = clean_text(row.get("partido", ""))
        if not party:
            continue
        loc = clean_text(row.get("nm_municipio", "")) or clean_text(row.get("uf", "")) or "Brasil"
        profile = clean_text(row.get("perfil_eleitor_2026", ""))
        justification = clean_text(row.get("justificativa_previsao_partido_2026", ""))
        cards.append(
            html.Div(
                className="info-card prediction",
                children=[
                    html.Div([html.Strong(party), html.Span(fmt_pct(row.get("share_pred_2026", "")))], className="card-line"),
                    html.P(loc),
                    html.Div(
                        [
                            html.Span(clean_text(row.get("tendencia_partido", "")) or "sem tendencia", className="pill"),
                            html.Span(clean_text(row.get("forca_correlacao_historica", "")) or "sem historico", className="pill muted"),
                        ],
                        className="pill-row",
                    ),
                    html.Small(profile),
                    html.Details([html.Summary("Justificativa"), html.P(justification)]) if justification else None,
                ],
            )
        )
    if not cards:
        cards.append(make_card(title, "Sem partido valido", "Dados nulos/sem valor foram filtrados."))
    return html.Div(cards, className="cards-compact")


def cluster_cards(store: DuckStore, key: str, title: str, limit: int = 12) -> html.Div:
    expr = store.expr(key)
    cols = store.columns(key)
    if expr is None:
        return html.Div([make_card(title, "Sem clusters", "Tabela nao encontrada.")], className="card-grid")
    selected = [
        c
        for c in [
            "cluster_global_discriminado",
            "qtd_setores",
            "qtd_municipios",
            "perfil_faixa_etaria_dominante",
            "perfil_genero_dominante",
            "perfil_instrucao_dominante",
            "perfil_estado_civil_dominante",
            "perfil_raca_cor_dominante",
            "regiao_dominante",
            "uf_dominante",
            "partido_vencedor_setor_dominante",
            "vencedor_setor_dominante",
            "persona_cluster",
        ]
        if c in cols
    ]
    if not selected:
        return html.Div([make_card(title, "Sem colunas", "Tabela sem persona de cluster.")], className="card-grid")
    order_col = "qtd_setores" if "qtd_setores" in cols else selected[0]
    sql = f"SELECT {', '.join(qident(c) for c in selected)} FROM {expr} ORDER BY {pct_expr(order_col)} desc nulls last LIMIT {int(limit)}"
    df = store.query(sql)
    cards: list[Any] = []
    for _, row in df.iterrows():
        cid = clean_text(row.get("cluster_global_discriminado", ""))
        bits = [
            ("Faixa", row.get("perfil_faixa_etaria_dominante", "")),
            ("Sexo", row.get("perfil_genero_dominante", "")),
            ("Escolaridade", row.get("perfil_instrucao_dominante", "")),
            ("Civil", row.get("perfil_estado_civil_dominante", "")),
            ("Raca/cor", row.get("perfil_raca_cor_dominante", "")),
        ]
        pills = [html.Span(f"{label}: {clean_text(value)}", className="pill") for label, value in bits if clean_text(value)]
        if not pills:
            continue
        result = clean_text(row.get("partido_vencedor_setor_dominante", "")) or clean_text(row.get("vencedor_setor_dominante", ""))
        loc = clean_text(row.get("regiao_dominante", "")) or clean_text(row.get("uf_dominante", ""))
        cards.append(
            html.Div(
                className="info-card cluster",
                children=[
                    html.Div([html.Strong(f"Cluster {cid}"), html.Span(loc)], className="card-line"),
                    html.Div(pills, className="pill-row"),
                    html.P(f"Tendencia eleitoral dominante: {result}") if result else html.P("Cluster definido apenas pelo perfil discreto do eleitorado."),
                    html.Small(f"Setores: {fmt_int(row.get('qtd_setores', ''))} | Municipios: {fmt_int(row.get('qtd_municipios', ''))}"),
                ],
            )
        )
    if not cards:
        cards.append(make_card(title, "Sem persona valida", "Clusters com dados nulos foram ocultados."))
    return html.Div(cards, className="cards-compact")


def render_brasil(store: DuckStore, ano: str | None, cenario: str | None) -> list[Any]:
    profiles = top_profiles(store, "brasil", ano=ano, limit=10)
    party_profiles = entity_profile_cards(store, "comparativo_partido", "brasil", ano=ano, limit=10)
    candidate_profiles = entity_profile_cards(store, "comparativo_candidato", "brasil", ano=ano, limit=8)
    prediction = party_prediction(store, "sim_partidos_brasil", cenario=cenario, limit=15)
    prediction_is_sim = not prediction.empty and "erro" not in prediction
    if not prediction_is_sim:
        prediction = historical_party_results(store, ano=ano, limit=15)
    distribution = profile_distribution(store, ano=ano)
    metrics = metrics_by_year(store, "timeline_nacional")
    return [
        section(
            "Eleitor medio no Brasil",
            [
                profile_cards_from_df(profiles, "Perfil Brasil", limit=10),
                html.Div(
                    [
                        graph_card("Perfil discreto do eleitorado", bar_profile_figure(distribution, "Perfil do eleitor por ano")),
                        graph_card("Metricas essenciais por ano", line_metrics_figure(metrics, "Votos, eleitorado, comparecimento e abstencao")),
                    ],
                    className="graph-grid",
                ),
            ],
            "Somente dados discretos relevantes entram no perfil. Biometria, datas e horas ficam fora.",
        ),
        section("Quem vota por partido", [entity_cards(party_profiles, "Partidos", "Partido")]),
        section("Quem vota por candidato", [entity_cards(candidate_profiles, "Candidatos", "Candidato")]),
        section(
            "Simulacao 2026 por partido" if prediction_is_sim else "Resultado historico por partido",
            [
                html.Div([graph_card("Partidos projetados" if prediction_is_sim else "Partidos no historico processado", party_bar_figure(prediction, "Possivel porcentagem de votos por partido - Brasil" if prediction_is_sim else "Participacao historica por partido - Brasil"))], className="graph-grid one"),
                prediction_cards(prediction, "Simulacao Brasil" if prediction_is_sim else "Historico Brasil", limit=15),
            ],
            "Fallback historico usando ouro/resultados_vencedores_secao e prata/resultados_votos enquanto a simulacao 2026 nao estiver pronta." if not prediction_is_sim else "",
        ),
    ]


def render_estados(store: DuckStore, uf: str | None, ano: str | None, cenario: str | None) -> list[Any]:
    profiles = top_profiles(store, "estado", uf=uf, ano=ano, limit=10 if uf else 24)
    party = party_prediction(store, "sim_partidos_estados", uf=uf, cenario=cenario, limit=20)
    party_is_sim = not party.empty and "erro" not in party
    if not party_is_sim:
        party = historical_party_results(store, uf=uf, ano=ano, limit=20)
    metrics = metrics_by_year(store, "timeline_uf", uf=uf)
    map_df = state_map_data(store, cenario=cenario)
    return [
        section(
            "Mapa do Brasil por estado",
            [
                html.Div(
                    [
                        graph_card("Mapa interativo", make_map_figure(map_df), "Passe o mouse nos pontos para ver partido, share e perfil associado."),
                        graph_card("Simulacao estadual" if party_is_sim else "Historico estadual", party_bar_figure(party, "Possivel porcentagem por partido no estado" if party_is_sim else "Participacao historica por partido no estado")),
                    ],
                    className="graph-grid",
                )
            ],
        ),
        section("Top perfis por estado", [profile_cards_from_df(profiles, "Perfis por UF", limit=24)]),
        section("Metricas essenciais do estado", [html.Div([graph_card("Timeline estadual", line_metrics_figure(metrics, "Votos, eleitorado, comparecimento e abstencao por UF"))], className="graph-grid one")]),
        section("Eleitor 2026 por partido no estado" if party_is_sim else "Partidos no historico do estado", [prediction_cards(party, "Simulacao estadual" if party_is_sim else "Historico estadual", limit=20)]),
    ]


def render_municipios(store: DuckStore, uf: str | None, municipio: str | None, ano: str | None, cenario: str | None) -> list[Any]:
    profiles = top_profiles(store, "municipio", uf=uf, municipio=municipio, ano=ano, limit=10 if municipio else 24)
    party = party_prediction(store, "sim_partidos_municipios", uf=uf, municipio=municipio, cenario=cenario, limit=20)
    party_is_sim = not party.empty and "erro" not in party
    if not party_is_sim:
        party = historical_party_results(store, uf=uf, municipio=municipio, ano=ano, limit=20)
    metrics = metrics_by_year(store, "timeline_municipal", uf=uf, municipio=municipio)
    return [
        section(
            "Consulta municipal",
            [
                profile_cards_from_df(profiles, "Top perfis municipais", limit=24),
                html.Div(
                    [
                        graph_card("Partidos projetados no municipio" if party_is_sim else "Partidos no historico do municipio", party_bar_figure(party, "Possivel porcentagem por partido" if party_is_sim else "Participacao historica por partido")),
                        graph_card("Metricas essenciais do municipio", line_metrics_figure(metrics, "Timeline municipal")),
                    ],
                    className="graph-grid",
                ),
            ],
            "Use a caixa de municipios no topo para procurar qualquer municipio do run.",
        ),
        section("Eleitor 2026 por partido no municipio" if party_is_sim else "Partidos no historico do municipio", [prediction_cards(party, "Simulacao municipal" if party_is_sim else "Historico municipal", limit=20)]),
    ]


def render_clusters(store: DuckStore) -> list[Any]:
    return [
        section(
            "Clusters somente de eleitores",
            [
                cluster_cards(store, "cluster_voter_personas", "Clusters eleitores"),
                html.Div(
                    [graph_card("Valores discretos que definem os clusters", cluster_discriminant_figure(store, "cluster_voter_discriminants", "Discriminantes dos clusters de eleitores"))],
                    className="graph-grid one",
                ),
            ],
            "KMeans sobre tokens discretos: faixa etaria, sexo/genero, escolaridade, estado civil e outros discretos uteis. Biometria fica fora.",
        ),
        section(
            "Clusters eleitores + resultado",
            [
                cluster_cards(store, "cluster_result_personas", "Clusters resultado"),
                html.Div(
                    [graph_card("Valores discretos + resultado", cluster_discriminant_figure(store, "cluster_result_discriminants", "Discriminantes dos clusters com resultado"))],
                    className="graph-grid one",
                ),
            ],
        ),
    ]


def render_simulacao(store: DuckStore, uf: str | None, municipio: str | None, cenario: str | None) -> list[Any]:
    br = party_prediction(store, "sim_partidos_brasil", cenario=cenario, limit=20)
    st = party_prediction(store, "sim_partidos_estados", uf=uf, cenario=cenario, limit=20)
    mu = party_prediction(store, "sim_partidos_municipios", uf=uf, municipio=municipio, cenario=cenario, limit=20)
    br_is_sim = not br.empty and "erro" not in br
    st_is_sim = not st.empty and "erro" not in st
    mu_is_sim = not mu.empty and "erro" not in mu
    if not br_is_sim:
        br = historical_party_results(store, limit=20)
    if not st_is_sim:
        st = historical_party_results(store, uf=uf, limit=20)
    if not mu_is_sim:
        mu = historical_party_results(store, uf=uf, municipio=municipio, limit=20)
    return [
        section(
            "Cenario 2026 - Brasil" if br_is_sim else "Historico processado - Brasil",
            [
                html.Div([graph_card("Brasil por partido", party_bar_figure(br, "Porcentagem possivel por partido - Brasil" if br_is_sim else "Participacao historica por partido - Brasil"))], className="graph-grid one"),
                prediction_cards(br, "Brasil" if br_is_sim else "Historico Brasil", limit=20),
            ],
            "" if br_is_sim else "A simulacao 2026 ainda nao foi encontrada; exibindo dados historicos ja processados.",
        ),
        section(
            "Cenario 2026 - estados" if st_is_sim else "Historico processado - estados",
            [
                html.Div([graph_card("Estado por partido", party_bar_figure(st, "Porcentagem possivel por partido - Estado" if st_is_sim else "Participacao historica por partido - Estado"))], className="graph-grid one"),
                prediction_cards(st, "Estados" if st_is_sim else "Historico Estados", limit=20),
            ],
        ),
        section(
            "Cenario 2026 - municipios" if mu_is_sim else "Historico processado - municipios",
            [
                html.Div([graph_card("Municipio por partido", party_bar_figure(mu, "Porcentagem possivel por partido - Municipio" if mu_is_sim else "Participacao historica por partido - Municipio"))], className="graph-grid one"),
                prediction_cards(mu, "Municipios" if mu_is_sim else "Historico Municipios", limit=20),
            ],
        ),
    ]


def kpi_cards(store: DuckStore) -> list[Any]:
    resultados = store.ouro_resultados_summary()
    total_resultados = int(resultados.get("total") or 0)
    concluidas_resultados = int(resultados.get("concluidas") or 0)
    anos = sorted(
        {
            str(row.get("ano"))
            for row in (store.ouro_resultados_status().get("status") or [])
            if isinstance(row, dict) and row.get("ano")
        },
        key=year_sort_key,
    )
    tabelas = store.available_tables()
    prata_count = sum(1 for key in tabelas if key.startswith("banco_prata_"))
    ouro_count = sum(1 for key in tabelas if key.startswith("banco_ouro_") or key in {"municipal", "timeline_nacional", "timeline_uf", "timeline_municipal", "perfil_ano", "top10_perfis", "vencedor_secao"})
    return [
        make_card("Tabelas encontradas", str(len(tabelas)), "caminhos tratados detectados"),
        make_card("Anos", str(len(anos)), ", ".join(anos[:8])),
        make_card("Camada prata", fmt_int(prata_count), "bases limpas consultaveis"),
        make_card("Camada ouro", fmt_int(ouro_count), "bases analiticas consultaveis"),
        make_card("Resultados por secao", f"{concluidas_resultados}/{total_resultados}" if total_resultados else "sem manifesto", "fatias UF/ano concluidas"),
        make_card("Modo", "consulta direta", "sem carregar tudo no HTML"),
    ]


def table_options(store: DuckStore) -> list[dict[str, str]]:
    return [{"label": TABLE_LABELS.get(key, key), "value": key} for key in store.available_tables()]


def df_records(df: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    work = df.head(int(limit)) if limit else df
    return json.loads(work.to_json(orient="records", force_ascii=False, date_format="iso"))


def dropdown_options(values: list[str] | tuple[str, ...]) -> list[dict[str, str]]:
    return [{"label": v, "value": v} for v in values if is_meaningful(v)]


def layer_cards(store: DuckStore, keys: list[str]) -> list[Any]:
    cards: list[Any] = []
    for key in keys:
        path = store.path_for(key)
        label = TABLE_LABELS.get(key, key)
        cards.append(
            make_card(
                label,
                "Disponivel" if path else "Ausente",
                str(path.relative_to(store.run_path)) if path and path.is_relative_to(store.run_path) else (str(path) if path else "nao encontrado"),
            )
        )
    return cards


def grouped_resultados_status(store: DuckStore, wanted_status: str) -> dict[str, list[str]]:
    data = store.ouro_resultados_status()
    rows = data.get("status") if isinstance(data, dict) else []
    grouped: dict[str, list[str]] = {}
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        status = str(row.get("status", "")).lower()
        if status != wanted_status:
            continue
        uf = clean_text(row.get("uf")) or "SEM_UF"
        ano = clean_text(row.get("ano")) or "sem ano"
        grouped.setdefault(uf, []).append(ano)
    return {uf: sorted(set(anos), key=year_sort_key) for uf, anos in sorted(grouped.items())}


def status_cards(grouped: dict[str, list[str]], limit: int = 24) -> list[Any]:
    cards: list[Any] = []
    for uf, years in list(grouped.items())[:limit]:
        cards.append(
            html.Div(
                className="info-card compact",
                children=[
                    html.Div([html.Strong(uf), html.Span(f"{len(years)} ano(s)")], className="card-line"),
                    html.Div([html.Span(year, className="pill muted") for year in years], className="pill-row"),
                ],
            )
        )
    if len(grouped) > limit:
        cards.append(
            html.Div(
                className="info-card compact",
                children=[
                    html.Div([html.Strong("Mais UFs"), html.Span(str(len(grouped) - limit))], className="card-line"),
                    html.Small("Use o manifesto completo nos logs para ver a lista inteira."),
                ],
            )
        )
    return cards


def query_municipios(store: DuckStore, uf: str | None) -> list[dict[str, str]]:
    if not uf:
        return []
    expr = store.expr("municipal") or store.expr("sim_partidos_municipios")
    key = "municipal" if store.expr("municipal") else "sim_partidos_municipios"
    cols = store.columns(key)
    if expr is None or "nm_municipio" not in cols:
        return []
    clauses = [meaningful_sql("nm_municipio")]
    if "uf" in cols:
        clauses.append(f"cast({qident('uf')} as varchar) = {lit(uf)}")
    select_code = "cast(\"cd_municipio\" as varchar) || '|' || cast(\"nm_municipio\" as varchar)" if "cd_municipio" in cols else "cast(\"nm_municipio\" as varchar)"
    sql = (
        f"SELECT distinct {select_code} as value, cast(\"nm_municipio\" as varchar) as label "
        f"FROM {expr} WHERE {' and '.join(clauses)} ORDER BY label LIMIT 8000"
    )
    df = store.query(sql)
    if df.empty or "value" not in df:
        return []
    return [
        {"label": clean_text(r.get("label", "")), "value": clean_text(r.get("value", ""))}
        for _, r in df.iterrows()
        if is_meaningful(r.get("label", ""))
    ]


def register_api_routes(app: Dash, store: DuckStore) -> None:
    server = app.server

    @server.get("/api/health")
    def api_health():
        return jsonify({"status": "ok", "run": str(store.run_path), "tabelas": store.available_tables()})

    @server.get("/api/progresso")
    def api_progresso():
        summary = store.ouro_resultados_summary()
        return jsonify(
            {
                "status": "ok",
                "ouro_resultados": summary,
                "pendentes": grouped_resultados_status(store, "pendente"),
                "concluidos": grouped_resultados_status(store, "concluido"),
            }
        )

    @server.get("/api/municipios")
    def api_municipios():
        return jsonify({"status": "ok", "uf": request.args.get("uf", ""), "municipios": query_municipios(store, request.args.get("uf"))})

    @server.get("/api/partidos")
    def api_partidos():
        escopo = (request.args.get("escopo") or "brasil").lower()
        uf = request.args.get("uf") or None
        municipio = request.args.get("municipio") or None
        ano = request.args.get("ano") or None
        cenario = request.args.get("cenario") or "base"
        limit = max(1, min(int(request.args.get("limit", 20)), 200))
        if escopo == "estado":
            data = party_prediction(store, "sim_partidos_estados", uf=uf, cenario=cenario, limit=limit)
            fonte = "simulacao_2026"
            if data.empty or "erro" in data:
                data = historical_party_results(store, uf=uf, ano=ano, limit=limit)
                fonte = "historico_processado"
        elif escopo == "municipio":
            data = party_prediction(store, "sim_partidos_municipios", uf=uf, municipio=municipio, cenario=cenario, limit=limit)
            fonte = "simulacao_2026"
            if data.empty or "erro" in data:
                data = historical_party_results(store, uf=uf, municipio=municipio, ano=ano, limit=limit)
                fonte = "historico_processado"
        else:
            data = party_prediction(store, "sim_partidos_brasil", cenario=cenario, limit=limit)
            fonte = "simulacao_2026"
            if data.empty or "erro" in data:
                data = historical_party_results(store, ano=ano, limit=limit)
                fonte = "historico_processado"
        return jsonify({"status": "ok", "fonte": fonte, "escopo": escopo, "dados": df_records(data)})

    @server.get("/api/brasil")
    def api_brasil():
        ano = request.args.get("ano") or None
        cenario = request.args.get("cenario") or "base"
        partidos = party_prediction(store, "sim_partidos_brasil", cenario=cenario, limit=20)
        fonte_partidos = "simulacao_2026"
        if partidos.empty or "erro" in partidos:
            partidos = historical_party_results(store, ano=ano, limit=20)
            fonte_partidos = "historico_processado"
        return jsonify(
            {
                "status": "ok",
                "fonte_partidos": fonte_partidos,
                "perfis": df_records(top_profiles(store, "brasil", ano=ano, limit=10)),
                "perfil_discreto": df_records(profile_distribution(store, ano=ano, limit=30)),
                "metricas": df_records(metrics_by_year(store, "timeline_nacional")),
                "partidos": df_records(partidos),
            }
        )

    @server.get("/api/tabela")
    def api_tabela():
        key = request.args.get("key") or ""
        limit = max(1, min(int(request.args.get("limit", 100)), 1000))
        if key not in store.available_tables():
            return jsonify({"status": "erro", "erro": "tabela_nao_encontrada", "tabelas": store.available_tables()}), 404
        return jsonify({"status": "ok", "tabela": key, "label": TABLE_LABELS.get(key, key), "dados": df_records(table_query(store, key, limit=limit), limit=limit)})


def create_app(store: DuckStore) -> Dash:
    app = Dash(__name__, title="Dashboard Eleitoral DuckDB", suppress_callback_exceptions=True)
    status_rows = store.ouro_resultados_status().get("status") or []
    uf_values = tuple(sorted({str(r.get("uf")) for r in status_rows if isinstance(r, dict) and is_meaningful(r.get("uf"))} or set(UF_POINTS)))
    year_values = sorted(
        {str(r.get("ano")) for r in status_rows if isinstance(r, dict) and is_meaningful(r.get("ano"))} or {"2014", "2018", "2022", "2024"},
        key=year_sort_key,
    )
    cenario_values = ("base",)

    app.index_string = INDEX_TEMPLATE
    app.layout = html.Div(
        className="app-shell",
        children=[
            html.Header(
                className="hero",
                children=[
                    html.Div(
                        [
                            html.P("DuckDB + Plotly + Dash", className="eyebrow"),
                            html.H1("Dashboard Eleitoral consultando Parquet direto"),
                            html.P(
                                "Graficos, cards, mapa interativo e consulta leve sem embutir tabelas gigantes no HTML.",
                                className="hero-copy",
                            ),
                            html.Code(str(store.run_path), className="run-path"),
                        ]
                    ),
                    html.Div(kpi_cards(store), className="hero-kpis"),
                ],
            ),
            html.Div(
                className="filters",
                children=[
                    html.Div([html.Label("Estado"), dcc.Dropdown(id="uf-filter", options=dropdown_options(uf_values), placeholder="Todos os estados", clearable=True)], className="filter-box"),
                    html.Div([html.Label("Municipio"), dcc.Dropdown(id="municipio-filter", options=[], placeholder="Escolha um municipio", clearable=True, searchable=True)], className="filter-box wide"),
                    html.Div([html.Label("Ano"), dcc.Dropdown(id="ano-filter", options=dropdown_options(year_values), placeholder="Todos os anos", clearable=True)], className="filter-box"),
                    html.Div([html.Label("Cenario"), dcc.Dropdown(id="cenario-filter", options=dropdown_options(cenario_values), value="base" if "base" in cenario_values else (cenario_values[0] if cenario_values else None), clearable=False)], className="filter-box"),
                    html.Div([html.Label("Acao"), html.Button("Buscar dados", id="search-button", n_clicks=0, className="search-button")], className="filter-box action-box"),
                ],
            ),
            dcc.Store(id="search-state"),
            dcc.Tabs(
                id="main-tabs",
                value="brasil",
                className="tabs",
                children=[
                    dcc.Tab(label="Brasil", value="brasil"),
                    dcc.Tab(label="Estados", value="estados"),
                    dcc.Tab(label="Municipios", value="municipios"),
                    dcc.Tab(label="Clusters", value="clusters"),
                    dcc.Tab(label="Simulacao 2026", value="simulacao"),
                    dcc.Tab(label="Progresso", value="progresso"),
                    dcc.Tab(label="Consulta", value="consulta"),
                ],
            ),
            html.Main(id="tab-content", className="content"),
        ],
    )

    @app.callback(
        Output("municipio-filter", "options"),
        Output("municipio-filter", "value"),
        Input("uf-filter", "value"),
        State("municipio-filter", "value"),
    )
    def update_municipios(uf: str | None, current: str | None):
        if not uf:
            return [], None
        options = query_municipios(store, uf)
        values = {o["value"] for o in options}
        return options, current if current in values else None

    @app.callback(
        Output("search-state", "data"),
        Input("search-button", "n_clicks"),
        State("uf-filter", "value"),
        State("municipio-filter", "value"),
        State("ano-filter", "value"),
        State("cenario-filter", "value"),
    )
    def store_search(n_clicks: int | None, uf: str | None, municipio: str | None, ano: str | None, cenario: str | None):
        if not n_clicks:
            return None
        return {
            "clicks": int(n_clicks or 0),
            "uf": uf,
            "municipio": municipio,
            "ano": ano,
            "cenario": cenario,
        }

    @app.callback(
        Output("tab-content", "children"),
        Input("main-tabs", "value"),
        Input("search-state", "data"),
    )
    def render_tab(tab: str, search_state: dict[str, Any] | None):
        if tab == "progresso":
            return render_progresso(store)
        if tab == "consulta":
            return render_consulta(store)
        if not search_state:
            if tab == "brasil":
                return render_brasil(store, None, "base")
            return render_search_prompt(tab)
        uf = search_state.get("uf")
        municipio = search_state.get("municipio")
        ano = search_state.get("ano")
        cenario = search_state.get("cenario")
        if tab == "estados":
            return render_estados(store, uf, ano, cenario)
        if tab == "municipios":
            return render_municipios(store, uf, municipio, ano, cenario)
        if tab == "clusters":
            return render_clusters(store)
        if tab == "simulacao":
            return render_simulacao(store, uf, municipio, cenario)
        return render_brasil(store, ano, cenario)

    @app.callback(
        Output("table-preview", "children"),
        Input("table-search-button", "n_clicks"),
        State("table-select", "value"),
        State("table-limit", "value"),
    )
    def update_table_preview(n_clicks: int | None, table_key: str | None, limit: Any):
        if not n_clicks:
            return html.Div("Escolha uma tabela e clique em Buscar tabela.")
        if not table_key:
            return html.Div("Escolha uma tabela tratada.")
        try:
            n = max(10, min(int(limit or 100), 1000))
        except Exception:
            n = 100
        df = table_query(store, table_key, limit=n)
        if df.empty:
            return html.Div("Tabela vazia ou nao encontrada.")
        return dash_table.DataTable(
            data=df.to_dict("records"),
            columns=[{"name": c, "id": c} for c in df.columns],
            page_size=min(n, 25),
            sort_action="native",
            filter_action="native",
            style_table={"overflowX": "auto", "maxHeight": "560px", "overflowY": "auto"},
            style_cell={"fontFamily": "Consolas, monospace", "fontSize": 12, "padding": "8px", "maxWidth": 260, "whiteSpace": "normal"},
            style_header={"fontWeight": "700", "backgroundColor": "#eef4f8"},
        )

    register_api_routes(app, store)
    return app


def render_consulta(store: DuckStore) -> list[Any]:
    options = table_options(store)
    default = options[0]["value"] if options else None
    return [
        section(
            "Camadas consultaveis",
            [
                card_grid(
                    layer_cards(
                        store,
                        [
                            "banco_prata_eleitorado",
                            "banco_prata_candidatos",
                            "banco_prata_resultados",
                            "municipal",
                            "perfil_ano",
                            "vencedor_secao",
                        ],
                    )
                )
            ],
            "A consulta usa DuckDB direto nos Parquets. A prata fica disponivel aqui para auditoria; a ouro alimenta os graficos principais.",
        ),
        section(
            "Consulta dos dados tratados",
            [
                html.Div(
                    [
                        html.Div([html.Label("Tabela"), dcc.Dropdown(id="table-select", options=options, value=default, clearable=False)], className="filter-box wide"),
                        html.Div([html.Label("Linhas"), dcc.Input(id="table-limit", type="number", value=100, min=10, max=1000, step=10)], className="filter-box small"),
                        html.Div([html.Label("Acao"), html.Button("Buscar tabela", id="table-search-button", n_clicks=0, className="search-button")], className="filter-box action-box"),
                    ],
                    className="inline-controls",
                ),
                html.Div(id="table-preview", className="table-preview"),
            ],
            "Esta aba e a unica que mostra tabela, sempre com limite. As demais abas sao graficos e cards.",
        )
    ]


def render_progresso(store: DuckStore) -> list[Any]:
    summary = store.ouro_resultados_summary()
    completed_paths = store.completed_resultado_paths()
    total = int(summary.get("total") or 0)
    concluidas = int(summary.get("concluidas") or 0)
    pendentes = int(summary.get("pendentes") or 0)
    cobertura = (concluidas / total) if total else 0
    pending = grouped_resultados_status(store, "pendente")
    completed = grouped_resultados_status(store, "concluido")
    cards = [
        make_card("Fatias UF/ano", fmt_int(total), "planejadas no manifesto ouro_resultados"),
        make_card("Concluidas", fmt_int(concluidas), fmt_pct(cobertura)),
        make_card("Pendentes", fmt_int(pendentes), f"{len(pending)} UFs com pendencia"),
        make_card("Lidas no dashboard", fmt_int(len(completed_paths)), "saidas concluidas usadas em vencedor_secao"),
    ]
    if summary.get("erro"):
        cards.append(make_card("Manifesto", "Erro de leitura", str(summary.get("erro"))))
    return [
        section(
            "Progresso da camada ouro",
            [
                card_grid(cards),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3("Pendentes por UF"),
                                html.Div(status_cards(pending, limit=36), className="cards-compact"),
                            ],
                            className="progress-column",
                        ),
                        html.Div(
                            [
                                html.H3("Concluidos por UF"),
                                html.Div(status_cards(completed, limit=36), className="cards-compact"),
                            ],
                            className="progress-column",
                        ),
                    ],
                    className="progress-grid",
                ),
            ],
            "Esta tela le os manifestos/status em logs/ouro. O dataset vencedor_secao usa preferencialmente apenas as fatias marcadas como concluidas.",
        ),
        section(
            "Prata e ouro disponiveis",
            [
                card_grid(
                    layer_cards(
                        store,
                        [
                            "banco_prata_eleitorado",
                            "banco_prata_candidatos",
                            "banco_prata_resultados",
                            "timeline_uf",
                            "timeline_municipal",
                            "retrato_municipal" if "retrato_municipal" in TABLE_CANDIDATES else "municipal",
                            "perfil_ano",
                            "top10_perfis",
                            "vencedor_secao",
                        ],
                    )
                )
            ],
            "Use a aba Consulta para abrir uma amostra limitada dessas camadas sem carregar a base inteira.",
        ),
    ]


INDEX_TEMPLATE = """
<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
      :root {
        --bg: #f5f7fb;
        --ink: #142033;
        --muted: #5b6678;
        --line: #d8e0eb;
        --panel: #ffffff;
        --soft: #eef6f4;
        --accent: #146c5f;
        --accent2: #2563eb;
        --dark: #0f172a;
      }
      * { box-sizing: border-box; }
      body { margin: 0; background: var(--bg); color: var(--ink); font-family: Inter, "Segoe UI", Arial, sans-serif; }
      .app-shell { min-height: 100vh; }
      .hero { background: linear-gradient(135deg, #0f172a 0%, #17324d 52%, #0f766e 100%); color: white; padding: 28px 36px 30px; display: grid; grid-template-columns: minmax(320px, 1fr) minmax(360px, 760px); gap: 28px; align-items: end; }
      .eyebrow { margin: 0 0 8px; text-transform: uppercase; letter-spacing: .08em; color: #9ee8d9; font-size: 12px; font-weight: 800; }
      h1 { margin: 0; font-size: clamp(30px, 4vw, 54px); line-height: 1.02; letter-spacing: 0; }
      .hero-copy { max-width: 780px; color: #dbeafe; font-size: 16px; line-height: 1.5; }
      .run-path { display: inline-block; margin-top: 4px; max-width: 100%; padding: 8px 10px; border-radius: 8px; background: rgba(15, 23, 42, .55); color: #d8fff7; white-space: normal; word-break: break-all; }
      .hero-kpis { display: grid; grid-template-columns: repeat(2, minmax(150px, 1fr)); gap: 12px; }
      .metric-card { background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.20); border-radius: 8px; padding: 16px; box-shadow: 0 18px 50px rgba(0,0,0,.14); }
      .content .metric-card { background: white; border-color: var(--line); color: var(--ink); }
      .metric-title { color: inherit; opacity: .78; font-size: 12px; font-weight: 800; text-transform: uppercase; }
      .metric-value { font-size: 30px; font-weight: 900; margin-top: 6px; }
      .metric-subtitle { color: inherit; opacity: .76; font-size: 12px; margin-top: 4px; }
      .filters { max-width: 1440px; margin: 24px auto 12px; padding: 0 24px; display: grid; grid-template-columns: 1fr 2fr 1fr 1fr auto; gap: 12px; align-items: end; }
      .filter-box label { display: block; font-size: 12px; font-weight: 800; text-transform: uppercase; color: var(--muted); margin-bottom: 6px; }
      .filter-box input { width: 100%; min-height: 38px; border: 1px solid var(--line); border-radius: 8px; padding: 8px; }
      .filter-box.small { max-width: 140px; }
      .action-box { min-width: 140px; }
      .search-button { width: 100%; min-height: 38px; border: 0; border-radius: 8px; background: var(--accent2); color: white; font-weight: 900; padding: 0 14px; cursor: pointer; box-shadow: 0 10px 24px rgba(37,99,235,.24); }
      .search-button:hover { filter: brightness(.96); transform: translateY(-1px); }
      .tabs { max-width: 1440px; margin: 12px auto 0; padding: 0 24px; }
      .content { max-width: 1440px; margin: 0 auto; padding: 18px 24px 60px; }
      .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 18px; margin-bottom: 18px; box-shadow: 0 14px 45px rgba(15,23,42,.06); }
      .panel-head { display: flex; justify-content: space-between; gap: 16px; align-items: end; margin-bottom: 14px; }
      .panel h2 { margin: 0; font-size: 20px; letter-spacing: 0; }
      .panel-head p { margin: 0; color: var(--muted); max-width: 720px; line-height: 1.45; }
      .card-grid, .cards-compact { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 12px; }
      .cards-compact { grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); }
      .info-card { border: 1px solid var(--line); border-radius: 8px; padding: 14px; background: linear-gradient(180deg, #ffffff 0%, #f9fbfd 100%); min-height: 158px; transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease; }
      .info-card:hover { transform: translateY(-3px); box-shadow: 0 18px 42px rgba(15,23,42,.13); border-color: rgba(20,108,95,.35); }
      .info-card p { color: #334155; line-height: 1.42; margin: 10px 0; }
      .info-card small { color: var(--muted); line-height: 1.35; display: block; }
      .card-line { display: flex; justify-content: space-between; gap: 10px; align-items: center; }
      .card-line span { color: var(--accent); font-weight: 800; }
      .pill-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
      .pill { border-radius: 999px; background: #dff8f2; color: #07584d; padding: 5px 8px; font-size: 11px; font-weight: 800; }
      .pill.muted { background: #eef2ff; color: #3949a5; }
      .graph-grid { display: grid; grid-template-columns: repeat(2, minmax(320px, 1fr)); gap: 14px; margin-top: 14px; }
      .graph-grid.one { grid-template-columns: minmax(320px, 1fr); }
      .graph-card { border: 1px solid var(--line); border-radius: 8px; background: white; padding: 12px; min-height: 380px; transition: transform .2s ease, box-shadow .2s ease, z-index .2s ease; position: relative; }
      .graph-card:hover { transform: scale(1.025); z-index: 5; box-shadow: 0 24px 70px rgba(15,23,42,.18); }
      .graph-head { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; padding: 0 4px; }
      .graph-head h3 { margin: 0; font-size: 15px; }
      .graph-head p { margin: 0; color: var(--muted); font-size: 12px; }
      .inline-controls { display: flex; gap: 12px; align-items: end; flex-wrap: wrap; margin-bottom: 12px; }
      .inline-controls .wide { min-width: 340px; flex: 1; }
      .progress-grid { display: grid; grid-template-columns: repeat(2, minmax(320px, 1fr)); gap: 18px; margin-top: 18px; }
      .progress-column h3 { margin: 0 0 10px; font-size: 16px; }
      .info-card.compact { min-height: 98px; }
      .table-preview { border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
      details summary { cursor: pointer; font-weight: 800; color: var(--accent2); }
      @media (max-width: 980px) {
        .hero, .filters, .graph-grid, .progress-grid { grid-template-columns: 1fr; }
        .hero { padding: 24px; }
        .hero-kpis { grid-template-columns: 1fr; }
      }
    </style>
  </head>
  <body>
    {%app_entry%}
    <footer>
      {%config%}
      {%scripts%}
      {%renderer%}
    </footer>
  </body>
</html>
"""


def main() -> None:
    args = parse_args()
    run_path = resolve_run_path(args.run)
    if not run_path.exists():
        print(f"Run nao encontrado: {run_path}", file=sys.stderr)
        raise SystemExit(2)
    store = DuckStore(run_path)
    if not store.available_tables():
        print(f"Nenhum Parquet/CSV tratado encontrado em: {run_path}", file=sys.stderr)
        raise SystemExit(2)
    app = create_app(store)
    print(f"Dashboard Dash lendo dados tratados de: {run_path}")
    print(f"Abra: http://{args.host}:{args.port}")
    if hasattr(app, "run"):
        app.run(host=args.host, port=args.port, debug=args.debug)
    else:
        app.run_server(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
