from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    import polars as pl
except ModuleNotFoundError:  # pragma: no cover
    pl = None  # type: ignore[assignment]

from dashboard_dash_eleitoral import (
    DuckStore,
    df_records,
    grouped_resultados_status,
    historical_party_results,
    metrics_by_year,
    party_prediction,
    profile_distribution,
    query_municipios,
    table_query,
    top_profiles,
)
from gerar_relatorio_pdf_eleitoral import build_report
from parquet_query_polars_eleitoral import (
    ANALYSIS_MODES,
    MODE_LABELS,
    POLARS_IMPORT_ERROR,
    PolarsStore,
    TABLE_LABELS,
    modalidade_allows,
    modalidade_info,
    normalize_modalidade,
    polars_available,
    records as polars_records,
)


class PdfJobRequest(BaseModel):
    modalidade_analise: str = Field(default="completa")
    out: str = ""
    log_dir: str = ""
    max_pages: int = Field(default=300, ge=1, le=5000)
    top_n: int = Field(default=15, ge=1, le=200)
    ufs: str = ""
    municipios_por_uf: int = Field(default=5, ge=0, le=200)
    incluir_secoes: bool = False
    secoes_por_uf: int = Field(default=30, ge=0, le=1000)
    duckdb_threads: int = Field(default=2, ge=1, le=32)
    engine: str = Field(default="polars", pattern="^(polars|duckdb)$")
    separado_por_nivel: bool = False
    quiet: bool = False


class AnalysisJobRequest(BaseModel):
    modalidade_analise: str = Field(default="estados_brasil")
    ufs: str = ""
    somente_estados_brasil: bool = False
    max_municipios_por_uf: int = Field(default=0, ge=0, le=10000)
    cenarios: int = Field(default=100, ge=0, le=100000)
    cluster_min_k: int = Field(default=2, ge=2, le=50)
    cluster_max_k: int = Field(default=10, ge=2, le=80)
    banco_ouro_workers: int = Field(default=1, ge=1, le=64)
    banco_duckdb_threads: int = Field(default=1, ge=1, le=64)
    skip_heavy_analyses: bool = True
    skip_clusters: bool = True
    predict_2026: bool = True
    paralelo_agressivo: bool = False


class ApiState:
    def __init__(self, run_path: Path, engine: str = "polars"):
        self.run_path = run_path
        self.engine = engine
        self.jobs: dict[str, dict[str, Any]] = {}
        self.jobs_lock = threading.Lock()

    def store(self, threads: int = 2) -> Any:
        if self.engine == "polars":
            if not polars_available():
                raise RuntimeError(f"Polars nao instalado: {POLARS_IMPORT_ERROR}")
            return PolarsStore(self.run_path)
        return DuckStore(self.run_path, threads=threads)


def to_records(data: Any, limit: int | None = None) -> list[dict[str, Any]]:
    if hasattr(data, "to_dicts"):
        rows = polars_records(data)
        return rows[: int(limit)] if limit else rows
    return df_records(data, limit=limit)


def is_empty(data: Any) -> bool:
    if data is None:
        return True
    if hasattr(data, "height"):
        return int(data.height) == 0
    if isinstance(data, pd.DataFrame):
        return data.empty or "erro" in data
    return False


def call_grouped_status(store: Any, status: str) -> dict[str, list[str]]:
    if hasattr(store, "grouped_resultados_status"):
        return store.grouped_resultados_status(status)
    return grouped_resultados_status(store, status)


def call_municipios(store: Any, uf: str) -> list[dict[str, str]]:
    if hasattr(store, "municipios"):
        return store.municipios(uf)
    return query_municipios(store, uf)


def call_top_profiles(store: Any, nivel: str, **kwargs: Any) -> Any:
    if hasattr(store, "top_profiles"):
        return store.top_profiles(nivel, **kwargs)
    return top_profiles(store, nivel, **kwargs)


def call_profile_distribution(store: Any, **kwargs: Any) -> Any:
    if hasattr(store, "profile_distribution"):
        return store.profile_distribution(**kwargs)
    return profile_distribution(store, **kwargs)


def call_party_prediction(store: Any, key: str, **kwargs: Any) -> Any:
    if hasattr(store, "party_prediction"):
        return store.party_prediction(key, **kwargs)
    return party_prediction(store, key, **kwargs)


def call_historical_party_results(store: Any, **kwargs: Any) -> Any:
    if hasattr(store, "historical_party_results"):
        return store.historical_party_results(**kwargs)
    return historical_party_results(store, **kwargs)


def call_metrics_by_year(store: Any, key: str, **kwargs: Any) -> Any:
    if hasattr(store, "metrics_by_year"):
        return store.metrics_by_year(key, **kwargs)
    return metrics_by_year(store, key, **kwargs)


def call_table(store: Any, key: str, limit: int) -> Any:
    if hasattr(store, "table"):
        return store.table(key, limit=limit)
    return table_query(store, key, limit=limit)


def read_parquet_file_small(path: Path, limit: int) -> pd.DataFrame:
    if pl is not None:
        frame = pl.scan_parquet(str(path), hive_partitioning=False).limit(max(1, int(limit))).collect(engine="streaming")
        return pd.DataFrame(frame.to_dicts())
    return pd.read_parquet(path).head(limit)


def read_small_parquet(path: Path, limit: int = 50) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        if path.is_dir():
            frames: list[pd.DataFrame] = []
            remaining = max(1, int(limit))
            for file in sorted(path.rglob("*.parquet")):
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


def read_ouro_brasil(run_path: Path, name: str, limit: int = 50) -> pd.DataFrame:
    return read_small_parquet(run_path / "ouro" / "brasil" / name, limit=limit)


def call_state_party_map(store: Any, **kwargs: Any) -> Any:
    if hasattr(store, "state_party_map"):
        return store.state_party_map(**kwargs)
    return pd.DataFrame()


def call_cluster_personas(store: Any, **kwargs: Any) -> Any:
    if hasattr(store, "cluster_personas"):
        return store.cluster_personas(**kwargs)
    return pd.DataFrame()


def call_entity_results(store: Any, **kwargs: Any) -> Any:
    if hasattr(store, "entity_results"):
        return store.entity_results(**kwargs)
    return pd.DataFrame()


def call_entity_profiles(store: Any, **kwargs: Any) -> Any:
    if hasattr(store, "entity_profiles"):
        return store.entity_profiles(**kwargs)
    return pd.DataFrame()


def call_quick_party_results(store: Any, **kwargs: Any) -> Any:
    if hasattr(store, "quick_party_results"):
        return store.quick_party_results(**kwargs)
    return pd.DataFrame()


def disabled_response(modalidade: str, feature: str, **extra: Any) -> dict[str, Any]:
    info = modalidade_info(modalidade)
    return {
        "status": "ok",
        "modalidade": info,
        "habilitado": False,
        "aviso": f"A modalidade {info['modalidade']} nao gera/consulta {feature}.",
        **extra,
    }


def resolve_run_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path.cwd() / path).resolve()


def tail_text(path: Path, max_lines: int = 80, max_bytes: int = 200_000) -> list[str]:
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            handle.seek(max(0, size - int(max_bytes)))
            data = handle.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return [f"erro lendo {path.name}: {exc}"]
    lines = data.splitlines()
    return lines[-int(max_lines):]


def read_json_file(path: Path, max_bytes: int = 2_000_000) -> Any:
    try:
        if path.stat().st_size > max_bytes:
            return {"arquivo_grande": True, "ultimas_linhas": tail_text(path, max_lines=60)}
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        return {"erro": str(exc), "arquivo": str(path)}


def safe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def build_analysis_command(run_path: Path, payload: AnalysisJobRequest) -> list[str]:
    modalidade = normalize_modalidade(payload.modalidade_analise)
    script_path = Path(__file__).resolve().with_name("run_pipeline_eleitoral_json.py")
    cmd = [
        sys.executable,
        str(script_path),
        str(run_path),
        "--modo",
        "analise_banco",
        "--banco-out",
        str(run_path),
        "--resume",
        "--banco-modalidade-analise",
        modalidade,
        "--cenarios",
        str(int(payload.cenarios)),
        "--cluster-min-k",
        str(int(payload.cluster_min_k)),
        "--cluster-max-k",
        str(int(payload.cluster_max_k)),
        "--banco-ouro-workers",
        str(int(payload.banco_ouro_workers)),
        "--banco-duckdb-threads",
        str(int(payload.banco_duckdb_threads)),
    ]
    if payload.predict_2026:
        cmd.append("--predict-2026")
    if payload.ufs.strip():
        cmd.extend(["--banco-ufs", payload.ufs.strip()])
    if payload.somente_estados_brasil or modalidade == "estados_brasil":
        cmd.append("--banco-somente-estados-brasil")
    if payload.max_municipios_por_uf > 0:
        cmd.extend(["--banco-max-municipios-por-uf", str(int(payload.max_municipios_por_uf))])
    if payload.skip_heavy_analyses:
        cmd.append("--banco-skip-heavy-analyses")
    if payload.skip_clusters:
        cmd.extend(["--banco-skip-clusters", "--sem-clustering"])
    if payload.paralelo_agressivo:
        cmd.append("--banco-ouro-paralelo-agressivo")
    return cmd


def recent_entries_from_root(logs_root: Path, relative_root: Path, max_files: int = 12, max_lines: int = 60) -> list[dict[str, Any]]:
    if not logs_root.exists():
        return []
    files = [path for path in logs_root.rglob("*") if path.is_file()]
    files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    entries: list[dict[str, Any]] = []
    for path in files[: int(max_files)]:
        stat = path.stat()
        entry: dict[str, Any] = {
            "nome": path.name,
            "relativo": safe_relative(path, relative_root),
            "tamanho_bytes": int(stat.st_size),
            "modificado_em": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        }
        suffix = path.suffix.lower()
        if suffix == ".json":
            entry["json"] = read_json_file(path)
        elif suffix in {".log", ".jsonl", ".txt"}:
            entry["linhas"] = tail_text(path, max_lines=max_lines)
        else:
            entry["linhas"] = tail_text(path, max_lines=min(max_lines, 20))
        entries.append(entry)
    return entries


def recent_log_entries(run_path: Path, max_files: int = 12, max_lines: int = 60) -> list[dict[str, Any]]:
    return recent_entries_from_root(run_path / "logs", run_path, max_files=max_files, max_lines=max_lines)


def priority_processing_events(run_path: Path) -> dict[str, Any]:
    candidates = {
        "evento_atual": run_path / "logs" / "evento_atual.json",
        "ouro_resultados_evento_atual": run_path / "logs" / "ouro" / "ouro_resultados_evento_atual.json",
        "ouro_eleitorado_evento_atual": run_path / "logs" / "ouro" / "ouro_eleitorado_evento_atual.json",
        "ouro_resultados_status_fatias": run_path / "logs" / "ouro" / "ouro_resultados_status_fatias.json",
        "ouro_resultados_pendentes": run_path / "logs" / "ouro" / "ouro_resultados_pendentes.json",
    }
    eventos: dict[str, Any] = {}
    for key, path in candidates.items():
        if path.exists():
            eventos[key] = read_json_file(path)
    return eventos


def make_app(run_path: Path, engine: str = "polars") -> FastAPI:
    if engine == "polars" and not polars_available():
        engine = "duckdb"
    state = ApiState(run_path, engine=engine)
    app = FastAPI(
        title="API Eleitoral TSE",
        description="Consulta Parquet com Polars por padrao, DuckDB como fallback, e geracao de PDF para o dashboard eleitoral.",
        version="1.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        with state.store() as store:
            return {
                "status": "ok",
                "run": str(state.run_path),
                "engine": state.engine,
                "tabelas": store.available_tables(),
                "modalidades": [{"key": key, "label": MODE_LABELS.get(key, key), **modalidade_info(key)} for key in ANALYSIS_MODES],
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }

    @app.get("/api/modalidades")
    def modalidades() -> dict[str, Any]:
        return {
            "status": "ok",
            "modalidades": [{"key": key, "label": MODE_LABELS.get(key, key), **modalidade_info(key)} for key in ANALYSIS_MODES],
        }

    @app.get("/api/tabelas")
    def tabelas() -> dict[str, Any]:
        with state.store() as store:
            keys = store.available_tables()
            return {
                "status": "ok",
                "tabelas": [{"key": key, "label": TABLE_LABELS.get(key, key), "path": str(store.path_for(key) or "")} for key in keys],
            }

    @app.get("/api/progresso")
    def progresso() -> dict[str, Any]:
        with state.store() as store:
            return {
                "status": "ok",
                "ouro_resultados": store.ouro_resultados_summary(),
                "pendentes": call_grouped_status(store, "pendente"),
                "concluidos": call_grouped_status(store, "concluido"),
            }

    @app.get("/api/logs")
    def logs(max_files: int = Query(12, ge=1, le=100), max_lines: int = Query(80, ge=1, le=500)) -> dict[str, Any]:
        return {
            "status": "ok",
            "run": str(state.run_path),
            "eventos": priority_processing_events(state.run_path),
            "arquivos": recent_log_entries(state.run_path, max_files=max_files, max_lines=max_lines),
        }

    @app.get("/api/processamento")
    def processamento(max_files: int = Query(8, ge=1, le=50), max_lines: int = Query(60, ge=1, le=300)) -> dict[str, Any]:
        with state.store() as store:
            resumo = {
                "ouro_resultados": store.ouro_resultados_summary(),
                "pendentes": call_grouped_status(store, "pendente"),
                "concluidos": call_grouped_status(store, "concluido"),
            }
        return {
            "status": "ok",
            "resumo": resumo,
            "eventos": priority_processing_events(state.run_path),
            "logs_recentes": recent_log_entries(state.run_path, max_files=max_files, max_lines=max_lines),
        }

    @app.get("/api/municipios")
    def municipios(uf: str = Query("", description="UF, exemplo SP"), modalidade: str = "completa") -> dict[str, Any]:
        modalidade = normalize_modalidade(modalidade)
        if not modalidade_allows(modalidade, "municipio"):
            return disabled_response(modalidade, "municipios", uf=uf, municipios=[])
        with state.store() as store:
            return {"status": "ok", "modalidade": modalidade_info(modalidade), "uf": uf, "municipios": call_municipios(store, uf)}

    @app.get("/api/mapa/estados")
    def mapa_estados(ano: str = "", cenario: str = "base", modalidade: str = "completa", limit: int = Query(80, ge=1, le=200)) -> dict[str, Any]:
        modalidade = normalize_modalidade(modalidade)
        if not modalidade_allows(modalidade, "estado"):
            return disabled_response(modalidade, "estados", ano=ano, cenario=cenario, dados=[])
        with state.store() as store:
            data = call_state_party_map(store, ano=ano or None, cenario=cenario, limit=limit)
            return {"status": "ok", "modalidade": modalidade_info(modalidade), "ano": ano, "cenario": cenario, "dados": to_records(data)}

    @app.get("/api/brasil")
    def brasil(ano: str = "", cenario: str = "base", modalidade: str = "completa", limit: int = Query(20, ge=1, le=200)) -> dict[str, Any]:
        modalidade = normalize_modalidade(modalidade)
        with state.store() as store:
            partidos = pd.DataFrame()
            fonte = "desativado_na_modalidade"
            if modalidade_allows(modalidade, "partido"):
                partidos = call_party_prediction(store, "sim_partidos_brasil", cenario=cenario, limit=limit)
                fonte = "simulacao_2026"
                if is_empty(partidos):
                    partidos = call_historical_party_results(store, ano=ano or None, limit=limit)
                    fonte = "historico_processado"
            perfis = call_top_profiles(store, "brasil", ano=ano or None, limit=10) if modalidade_allows(modalidade, "perfil") else pd.DataFrame()
            perfil_discreto = call_profile_distribution(store, ano=ano or None, limit=30) if modalidade_allows(modalidade, "perfil") else pd.DataFrame()
            return {
                "status": "ok",
                "modalidade": modalidade_info(modalidade),
                "fonte_partidos": fonte,
                "perfis": to_records(perfis),
                "perfil_discreto": to_records(perfil_discreto),
                "metricas": to_records(call_metrics_by_year(store, "timeline_nacional")),
                "partidos": to_records(partidos),
            }

    @app.get("/api/brasil/rapido")
    def brasil_rapido(ano: str = "", modalidade: str = "estados_brasil", limit: int = Query(20, ge=1, le=200)) -> dict[str, Any]:
        modalidade = normalize_modalidade(modalidade)
        with state.store() as store:
            partidos = call_quick_party_results(store, nivel="brasil", ano=ano or None, limit=limit) if modalidade_allows(modalidade, "partido") else pd.DataFrame()
            perfis = call_top_profiles(store, "brasil", ano=ano or None, limit=min(limit, 10)) if modalidade_allows(modalidade, "perfil") else pd.DataFrame()
            metricas = call_metrics_by_year(store, "timeline_nacional")
            return {
                "status": "ok",
                "modalidade": modalidade_info(modalidade),
                "fonte_partidos": "ouro_brasil_resultado_partido",
                "partidos": to_records(partidos),
                "perfis": to_records(perfis),
                "metricas": to_records(metricas),
            }

    @app.get("/api/brasil/tabelas")
    def brasil_tabelas(limit: int = Query(50, ge=1, le=500)) -> dict[str, Any]:
        resumo = read_ouro_brasil(state.run_path, "resumo", limit=limit)
        perfil = read_ouro_brasil(state.run_path, "perfil_eleitor", limit=limit)
        resultado = read_ouro_brasil(state.run_path, "resultado_partido", limit=limit)
        hist_perfil = read_ouro_brasil(state.run_path, "contagem_colunas_perfil_eleitor", limit=limit)
        hist_partido = read_ouro_brasil(state.run_path, "contagem_colunas_perfil_partido", limit=limit)
        hist_candidato = read_ouro_brasil(state.run_path, "contagem_colunas_perfil_candidato", limit=limit)
        hist_clusters = read_ouro_brasil(state.run_path, "contagem_colunas_clusters_eleitores", limit=limit)
        hist_clusters_resultado = read_ouro_brasil(state.run_path, "contagem_colunas_clusters_eleitores_resultado", limit=limit)
        return {
            "status": "ok",
            "fonte": "ouro/brasil",
            "tabelas": {
                "resumo": to_records(resumo, limit=limit),
                "perfil_eleitor": to_records(perfil, limit=limit),
                "resultado_partido": to_records(resultado, limit=limit),
                "contagem_colunas_perfil_eleitor": to_records(hist_perfil, limit=limit),
                "contagem_colunas_perfil_partido": to_records(hist_partido, limit=limit),
                "contagem_colunas_perfil_candidato": to_records(hist_candidato, limit=limit),
                "contagem_colunas_clusters_eleitores": to_records(hist_clusters, limit=limit),
                "contagem_colunas_clusters_eleitores_resultado": to_records(hist_clusters_resultado, limit=limit),
            },
        }

    @app.get("/api/perfis")
    def perfis(
        nivel: str = Query("brasil", pattern="^(brasil|estado|municipio)$"),
        uf: str = "",
        municipio: str = "",
        ano: str = "",
        modalidade: str = "completa",
        limit: int = Query(20, ge=1, le=200),
    ) -> dict[str, Any]:
        modalidade = normalize_modalidade(modalidade)
        if not modalidade_allows(modalidade, "perfil") or not modalidade_allows(modalidade, nivel):
            return disabled_response(modalidade, f"perfil_{nivel}", nivel=nivel, dados=[])
        with state.store() as store:
            data = call_top_profiles(
                store,
                nivel,
                uf=uf or None,
                municipio=municipio or None,
                ano=ano or None,
                limit=limit,
            )
            return {"status": "ok", "modalidade": modalidade_info(modalidade), "nivel": nivel, "dados": to_records(data)}

    @app.get("/api/clusters")
    def clusters(
        tipo: str = Query("eleitores", pattern="^(eleitores|resultado|eleitores_resultado)$"),
        nivel: str = Query("brasil", pattern="^(brasil|estado|municipio)$"),
        uf: str = "",
        municipio: str = "",
        ano: str = "",
        modalidade: str = "completa",
        limit: int = Query(20, ge=1, le=200),
    ) -> dict[str, Any]:
        modalidade = normalize_modalidade(modalidade)
        if not modalidade_allows(modalidade, "cluster") or not modalidade_allows(modalidade, nivel):
            return disabled_response(modalidade, f"clusters_{nivel}", tipo=tipo, nivel=nivel, dados=[])
        with state.store() as store:
            data = call_cluster_personas(
                store,
                tipo=tipo,
                nivel=nivel,
                uf=uf or None,
                municipio=municipio or None,
                ano=ano or None,
                limit=limit,
            )
            return {"status": "ok", "modalidade": modalidade_info(modalidade), "tipo": tipo, "nivel": nivel, "dados": to_records(data)}

    @app.get("/api/partidos")
    def partidos(
        escopo: str = Query("brasil", pattern="^(brasil|estado|municipio)$"),
        uf: str = "",
        municipio: str = "",
        ano: str = "",
        cenario: str = "base",
        modalidade: str = "completa",
        limit: int = Query(20, ge=1, le=200),
    ) -> dict[str, Any]:
        modalidade = normalize_modalidade(modalidade)
        if not modalidade_allows(modalidade, "partido") or not modalidade_allows(modalidade, escopo):
            return disabled_response(modalidade, f"partidos_{escopo}", fonte="desativado_na_modalidade", escopo=escopo, dados=[])
        with state.store() as store:
            if escopo == "estado":
                data = call_party_prediction(store, "sim_partidos_estados", uf=uf or None, cenario=cenario, limit=limit)
                fonte = "simulacao_2026"
                if is_empty(data):
                    data = call_historical_party_results(store, uf=uf or None, ano=ano or None, limit=limit)
                    fonte = "historico_processado"
            elif escopo == "municipio":
                data = call_party_prediction(store, "sim_partidos_municipios", uf=uf or None, municipio=municipio or None, cenario=cenario, limit=limit)
                fonte = "simulacao_2026"
                if is_empty(data):
                    data = call_historical_party_results(store, uf=uf or None, municipio=municipio or None, ano=ano or None, limit=limit)
                    fonte = "historico_processado"
            else:
                data = call_party_prediction(store, "sim_partidos_brasil", cenario=cenario, limit=limit)
                fonte = "simulacao_2026"
                if is_empty(data):
                    data = call_historical_party_results(store, ano=ano or None, limit=limit)
                    fonte = "historico_processado"
            return {"status": "ok", "modalidade": modalidade_info(modalidade), "fonte": fonte, "escopo": escopo, "dados": to_records(data)}

    @app.get("/api/candidatos")
    def candidatos(
        escopo: str = Query("brasil", pattern="^(brasil|estado|municipio)$"),
        tipo: str = Query("resultado", pattern="^(resultado|perfil)$"),
        uf: str = "",
        municipio: str = "",
        ano: str = "",
        modalidade: str = "completa",
        limit: int = Query(20, ge=1, le=200),
    ) -> dict[str, Any]:
        modalidade = normalize_modalidade(modalidade)
        if not modalidade_allows(modalidade, "candidato") or not modalidade_allows(modalidade, escopo):
            return disabled_response(modalidade, f"candidatos_{escopo}", escopo=escopo, tipo=tipo, dados=[])
        with state.store() as store:
            if tipo == "perfil":
                data = call_entity_profiles(store, entity="candidato", nivel=escopo, uf=uf or None, municipio=municipio or None, ano=ano or None, limit=limit)
            else:
                data = call_entity_results(store, entity="candidato", nivel=escopo, uf=uf or None, municipio=municipio or None, ano=ano or None, limit=limit)
            return {"status": "ok", "modalidade": modalidade_info(modalidade), "escopo": escopo, "tipo": tipo, "dados": to_records(data)}

    @app.get("/api/metricas")
    def metricas(
        escopo: str = Query("brasil", pattern="^(brasil|estado|municipio)$"),
        uf: str = "",
        municipio: str = "",
        modalidade: str = "completa",
    ) -> dict[str, Any]:
        modalidade = normalize_modalidade(modalidade)
        if not modalidade_allows(modalidade, escopo):
            return disabled_response(modalidade, f"metricas_{escopo}", escopo=escopo, dados=[])
        key = "timeline_nacional" if escopo == "brasil" else ("timeline_uf" if escopo == "estado" else "timeline_municipal")
        with state.store() as store:
            data = call_metrics_by_year(store, key, uf=uf or None, municipio=municipio or None)
            return {"status": "ok", "modalidade": modalidade_info(modalidade), "escopo": escopo, "dados": to_records(data)}

    @app.get("/api/tabela")
    def tabela(key: str, limit: int = Query(100, ge=1, le=1000)) -> dict[str, Any]:
        with state.store() as store:
            available = store.available_tables()
            if key not in available:
                raise HTTPException(status_code=404, detail={"erro": "tabela_nao_encontrada", "tabelas": available})
            data = call_table(store, key, limit=limit)
            return {"status": "ok", "tabela": key, "label": TABLE_LABELS.get(key, key), "dados": to_records(data, limit=limit)}

    @app.post("/api/analises/jobs")
    def create_analysis_job(payload: AnalysisJobRequest) -> dict[str, Any]:
        job_id = uuid.uuid4().hex[:12]
        modalidade = normalize_modalidade(payload.modalidade_analise)
        log_dir = state.run_path / "logs" / "api_jobs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"analise_{job_id}_{modalidade}.log"
        cmd = build_analysis_command(state.run_path, payload)
        job = {
            "id": job_id,
            "tipo": "analise_banco",
            "status": "queued",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "modalidade_analise": modalidade,
            "log_file": str(log_file),
            "command": cmd,
            "erro": "",
        }
        with state.jobs_lock:
            state.jobs[job_id] = job

        def runner() -> None:
            started = time.perf_counter()
            cwd = Path(__file__).resolve().parent.parent
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            with state.jobs_lock:
                state.jobs[job_id].update(status="running", updated_at=datetime.now().isoformat(timespec="seconds"))
            try:
                with log_file.open("w", encoding="utf-8", errors="replace") as log:
                    log.write(f"Inicio: {datetime.now().isoformat(timespec='seconds')}\n")
                    log.write("Comando:\n")
                    log.write(" ".join(cmd) + "\n\n")
                    log.flush()
                    process = subprocess.Popen(
                        cmd,
                        cwd=str(cwd),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        env=env,
                    )
                    with state.jobs_lock:
                        state.jobs[job_id].update(pid=process.pid, updated_at=datetime.now().isoformat(timespec="seconds"))
                    assert process.stdout is not None
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                    return_code = process.wait()
                    log.write(f"\nFim: {datetime.now().isoformat(timespec='seconds')} | returncode={return_code}\n")
                with state.jobs_lock:
                    state.jobs[job_id].update(
                        status="done" if return_code == 0 else "error",
                        returncode=return_code,
                        updated_at=datetime.now().isoformat(timespec="seconds"),
                        duracao_segundos=round(time.perf_counter() - started, 3),
                        erro="" if return_code == 0 else f"processo retornou codigo {return_code}",
                    )
            except Exception as exc:
                with state.jobs_lock:
                    state.jobs[job_id].update(
                        status="error",
                        erro=str(exc),
                        updated_at=datetime.now().isoformat(timespec="seconds"),
                        duracao_segundos=round(time.perf_counter() - started, 3),
                    )

        threading.Thread(target=runner, name=f"analysis-job-{job_id}", daemon=True).start()
        return {"status": "ok", "job": job}

    @app.get("/api/analises/jobs")
    def list_analysis_jobs() -> dict[str, Any]:
        with state.jobs_lock:
            jobs = [job for job in state.jobs.values() if job.get("tipo") == "analise_banco"]
        return {"status": "ok", "jobs": jobs}

    @app.get("/api/analises/jobs/{job_id}")
    def get_analysis_job(job_id: str) -> dict[str, Any]:
        with state.jobs_lock:
            job = state.jobs.get(job_id)
        if not job or job.get("tipo") != "analise_banco":
            raise HTTPException(status_code=404, detail="job_analise_nao_encontrado")
        return {"status": "ok", "job": job}

    @app.get("/api/analises/jobs/{job_id}/logs")
    def get_analysis_job_logs(job_id: str, max_lines: int = Query(160, ge=1, le=1000)) -> dict[str, Any]:
        with state.jobs_lock:
            job = state.jobs.get(job_id)
        if not job or job.get("tipo") != "analise_banco":
            raise HTTPException(status_code=404, detail="job_analise_nao_encontrado")
        log_file = Path(str(job.get("log_file") or ""))
        return {
            "status": "ok",
            "job_id": job_id,
            "log_file": str(log_file),
            "linhas": tail_text(log_file, max_lines=max_lines) if log_file.exists() else [],
            "job": job,
        }

    @app.post("/api/pdf/jobs")
    def create_pdf_job(payload: PdfJobRequest) -> dict[str, Any]:
        job_id = uuid.uuid4().hex[:12]
        out = Path(payload.out).expanduser() if payload.out else state.run_path / "relatorios" / f"relatorio_api_{job_id}.pdf"
        if not out.is_absolute():
            out = (Path.cwd() / out).resolve()
        log_dir = Path(payload.log_dir).expanduser() if payload.log_dir else out.parent / "logs"
        if not log_dir.is_absolute():
            log_dir = (Path.cwd() / log_dir).resolve()
        job = {
            "id": job_id,
            "status": "queued",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "out": str(out),
            "log_dir": str(log_dir),
            "engine": payload.engine,
            "modalidade_analise": normalize_modalidade(payload.modalidade_analise),
            "separado_por_nivel": bool(payload.separado_por_nivel),
            "erro": "",
        }
        with state.jobs_lock:
            state.jobs[job_id] = job

        def runner() -> None:
            started = time.perf_counter()
            with state.jobs_lock:
                state.jobs[job_id].update(status="running", updated_at=datetime.now().isoformat(timespec="seconds"))
            try:
                args = Namespace(
                    run=str(state.run_path),
                    out=str(out),
                    modalidade_analise=normalize_modalidade(payload.modalidade_analise),
                    max_pages=payload.max_pages,
                    top_n=payload.top_n,
                    ufs=payload.ufs,
                    municipios_por_uf=payload.municipios_por_uf,
                    incluir_secoes=payload.incluir_secoes,
                    secoes_por_uf=payload.secoes_por_uf,
                    duckdb_threads=payload.duckdb_threads,
                    query_engine=payload.engine,
                    pdf_separado_por_nivel=bool(payload.separado_por_nivel),
                    log_dir=str(log_dir),
                    quiet=payload.quiet,
                )
                result = build_report(args)
                with state.jobs_lock:
                    state.jobs[job_id].update(
                        status="done",
                        out=str(result),
                        updated_at=datetime.now().isoformat(timespec="seconds"),
                        duracao_segundos=round(time.perf_counter() - started, 3),
                    )
            except Exception as exc:
                with state.jobs_lock:
                    state.jobs[job_id].update(
                        status="error",
                        erro=str(exc),
                        updated_at=datetime.now().isoformat(timespec="seconds"),
                        duracao_segundos=round(time.perf_counter() - started, 3),
                    )

        threading.Thread(target=runner, name=f"pdf-job-{job_id}", daemon=True).start()
        return {"status": "ok", "job": job}

    @app.get("/api/pdf/jobs")
    def list_pdf_jobs() -> dict[str, Any]:
        with state.jobs_lock:
            jobs = list(state.jobs.values())
        return {"status": "ok", "jobs": jobs}

    @app.get("/api/pdf/jobs/{job_id}")
    def get_pdf_job(job_id: str) -> dict[str, Any]:
        with state.jobs_lock:
            job = state.jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job_nao_encontrado")
        return {"status": "ok", "job": job}

    @app.get("/api/pdf/jobs/{job_id}/logs")
    def get_pdf_job_logs(job_id: str, max_files: int = Query(12, ge=1, le=100), max_lines: int = Query(80, ge=1, le=500)) -> dict[str, Any]:
        with state.jobs_lock:
            job = state.jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job_nao_encontrado")
        log_dir = Path(str(job.get("log_dir") or ""))
        return {
            "status": "ok",
            "job_id": job_id,
            "log_dir": str(log_dir),
            "arquivos": recent_entries_from_root(log_dir, log_dir, max_files=max_files, max_lines=max_lines),
        }

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="API FastAPI para consultar o banco eleitoral Parquet e gerar PDF.")
    parser.add_argument("--run", default="dados/banco_eleitoral", help="Pasta do banco/run.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8055)
    parser.add_argument("--engine", choices=["polars", "duckdb"], default="polars", help="Engine de consulta da API. Padrao: polars.")
    parser.add_argument("--reload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_path = resolve_run_path(args.run)
    if not run_path.exists():
        raise SystemExit(f"Banco/run nao encontrado: {run_path}")
    app = make_app(run_path, engine=args.engine)
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
