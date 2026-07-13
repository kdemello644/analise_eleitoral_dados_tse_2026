from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import argparse
import ctypes
import os


@dataclass
class Config:
    dados: Path
    out: Path
    banco_out: Path
    modo: str
    resume: bool
    seed: int

    sample_mode: str
    max_sample_rows: int
    min_sample_rows: int
    sample_frac: float
    full_aggregations: bool
    aggregate_chunk_rows: int
    analysis_max_rows: int
    global_max_gold_rows: int
    gold_csv_max_rows: int
    workers_individual: int
    workers_large_files: int
    workers_parquet: int
    large_file_threshold_gb: float
    parquet_partition_rows: int
    partition_electorate_by_state: bool
    engine: str
    spark_master: str
    incluir_metadados_json: bool
    banco_overwrite: bool
    banco_chunk_rows: int
    banco_max_files: int
    banco_workers: int
    banco_workers_large_files: int
    banco_large_file_threshold_gb: float
    banco_ouro_workers: int
    banco_duckdb_threads: int
    banco_skip_heavy_analyses: bool
    banco_delete_source_after_success: bool
    banco_use_all_workers: bool
    banco_ouro_parallel_aggressive: bool
    banco_auto_tune_info: dict[str, Any]

    clustering: bool
    cluster_min_k: int
    cluster_max_k: int

    predict_2026: bool
    cenarios: int
    monte_carlo_sigma: float
    prediction_entity: str
    prediction_cargo_filter: str

    parquet: bool
    top_n_html: int
    top_n_plots: int
    log_level: str


def normalize_output_path(value: str) -> Path:
    out = Path(value).expanduser()
    if out.is_absolute():
        return out.resolve()

    base = Path.cwd()
    parts = out.parts
    if parts and parts[0].lower() == "resultados":
        return (base / out).resolve()
    return (base / "resultados" / out).resolve()


def normalize_database_path(value: str, dados: Path) -> Path:
    if not value:
        base = dados.parent if dados.name.lower() == "json" else dados
        return (base / "banco_eleitoral").resolve()
    out = Path(value).expanduser()
    if out.is_absolute():
        return out.resolve()
    return (Path.cwd() / out).resolve()


def detect_total_memory_gb() -> float:
    try:
        if hasattr(os, "sysconf"):
            page_size = os.sysconf("SC_PAGE_SIZE")
            pages = os.sysconf("SC_PHYS_PAGES")
            if page_size and pages:
                return float(page_size * pages) / float(1024 ** 3)
    except (AttributeError, OSError, ValueError):
        pass

    try:
        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return float(status.ullTotalPhys) / float(1024 ** 3)
    except Exception:
        pass

    return 0.0


def auto_tune_banco_settings(args: argparse.Namespace) -> tuple[dict[str, int | float], dict[str, Any]]:
    cpu_total = max(1, int(os.cpu_count() or 1))
    memory_gb = detect_total_memory_gb()
    effective_memory_gb = memory_gb if memory_gb > 0 else 16.0

    if cpu_total <= 2:
        reserve_cpu = 0
    elif cpu_total <= 4:
        reserve_cpu = 1
    elif cpu_total <= 8:
        reserve_cpu = 2
    else:
        reserve_cpu = max(2, int(round(cpu_total * 0.20)))
    usable_cpu = max(1, cpu_total - reserve_cpu)

    memory_reserve_gb = max(2.0, min(10.0, effective_memory_gb * 0.20))
    memory_for_workers_gb = max(1.0, effective_memory_gb - memory_reserve_gb)
    memory_worker_cap = max(1, int(memory_for_workers_gb // 2.5))
    worker_cap = max(1, min(usable_cpu, memory_worker_cap))

    if effective_memory_gb < 12:
        auto_workers = 1
        auto_large_workers = 1
        auto_chunk_rows = 30_000
        auto_ouro_workers = 1
        auto_duckdb_threads = min(usable_cpu, 2)
        auto_large_threshold_gb = 1.0
    elif effective_memory_gb < 24:
        auto_workers = min(4, worker_cap)
        auto_large_workers = min(3, auto_workers)
        auto_chunk_rows = 40_000
        auto_ouro_workers = min(3, worker_cap)
        auto_duckdb_threads = min(usable_cpu, 4)
        auto_large_threshold_gb = 1.0
    elif effective_memory_gb < 48:
        auto_workers = min(4, worker_cap)
        auto_large_workers = min(4, auto_workers)
        auto_chunk_rows = 60_000
        auto_ouro_workers = min(3, worker_cap)
        auto_duckdb_threads = min(usable_cpu, 6)
        auto_large_threshold_gb = 1.5
    elif effective_memory_gb < 96:
        auto_workers = min(6, worker_cap)
        auto_large_workers = min(4, auto_workers)
        auto_chunk_rows = 75_000
        auto_ouro_workers = min(4, worker_cap)
        auto_duckdb_threads = min(usable_cpu, 10)
        auto_large_threshold_gb = 2.0
    else:
        auto_workers = min(8, worker_cap)
        auto_large_workers = min(6, auto_workers)
        auto_chunk_rows = 100_000
        auto_ouro_workers = min(5, worker_cap)
        auto_duckdb_threads = min(usable_cpu, 14)
        auto_large_threshold_gb = 2.0

    selected = {
        "banco_chunk_rows": max(1_000, int(args.banco_chunk_rows or auto_chunk_rows)),
        "banco_workers": max(1, int(args.banco_workers or auto_workers)),
        "banco_workers_large_files": max(1, int(args.banco_workers_large_files or auto_large_workers)),
        "banco_large_file_threshold_gb": max(0.1, float(args.banco_large_file_threshold_gb or auto_large_threshold_gb)),
        "banco_ouro_workers": max(1, int(args.banco_ouro_workers or auto_ouro_workers)),
        "banco_duckdb_threads": max(1, int(args.banco_duckdb_threads or auto_duckdb_threads)),
    }
    if getattr(args, "banco_use_all_workers", False):
        selected["banco_workers"] = max(1, usable_cpu)
        selected["banco_workers_large_files"] = max(1, usable_cpu)
        if not args.banco_chunk_rows:
            selected["banco_chunk_rows"] = min(int(selected["banco_chunk_rows"]), 25_000)
    info = {
        "modo": "auto_com_override_manual",
        "cpu_total_logico": cpu_total,
        "cpu_reservado_para_sistema": reserve_cpu,
        "cpu_usavel_pipeline": usable_cpu,
        "memoria_total_gb_detectada": round(memory_gb, 2) if memory_gb > 0 else None,
        "memoria_base_calculo_gb": round(effective_memory_gb, 2),
        "memoria_reservada_para_sistema_gb": round(memory_reserve_gb, 2),
        "limite_workers_por_memoria": memory_worker_cap,
        "valores_auto": {
            "banco_chunk_rows": auto_chunk_rows,
            "banco_workers": auto_workers,
            "banco_workers_large_files": auto_large_workers,
            "banco_large_file_threshold_gb": auto_large_threshold_gb,
            "banco_ouro_workers": auto_ouro_workers,
            "banco_duckdb_threads": auto_duckdb_threads,
        },
        "valores_selecionados": selected,
        "usar_todos_workers": bool(getattr(args, "banco_use_all_workers", False)),
        "observacao": "Passe valores explicitos nas flags --banco-* para sobrescrever o auto-tuning.",
    }
    return selected, info


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Pipeline eleitoral JSON: data-driven, modular, análise global profunda e simulação explicada."
    )

    parser.add_argument("dados", help="Pasta raiz contendo somente JSON/JSONL/NDJSON já preparados.")
    parser.add_argument(
        "--out",
        default="pipeline_eleitoral_json",
        help="Pasta do run dentro de resultados/. Ex.: --out teste_rapido cria resultados/teste_rapido.",
    )
    parser.add_argument(
        "--modo",
        choices=["inventario", "individual", "global", "preditivo", "completo", "banco", "analise_banco"],
        default="completo",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--sample-mode", choices=["head", "reservoir"], default="reservoir")
    parser.add_argument("--max-sample-rows", type=int, default=300000)
    parser.add_argument("--min-sample-rows", type=int, default=5000)
    parser.add_argument("--sample-frac", type=float, default=0.03)
    parser.add_argument("--full-aggregations", action="store_true")
    parser.add_argument("--workers-individual", type=int, default=1)
    parser.add_argument("--workers-large-files", type=int, default=1)
    parser.add_argument("--workers-parquet", type=int, default=2)
    parser.add_argument("--large-file-threshold-gb", type=float, default=10.0)
    parser.add_argument("--parquet-partition-rows", type=int, default=250000)
    parser.add_argument("--sem-particionar-eleitorado-uf", dest="partition_electorate_by_state", action="store_false")
    parser.set_defaults(partition_electorate_by_state=True)
    parser.add_argument("--engine", choices=["pandas", "pyspark", "auto"], default="pandas")
    parser.add_argument("--spark-master", default="local[*]")
    parser.add_argument(
        "--aggregate-chunk-rows",
        type=int,
        default=75000,
        help="Quantidade de registros JSON processados por parte na agregacao completa. Menor usa menos memoria.",
    )
    parser.add_argument(
        "--analysis-max-rows",
        type=int,
        default=200000,
        help="Limite de linhas usadas em analises individuais pesadas/HTML; o Parquet completo continua salvo.",
    )
    parser.add_argument(
        "--global-max-gold-rows",
        type=int,
        default=0,
        help="Limite opcional de linhas carregadas da base gold global para etapas em memoria. 0 processa tudo.",
    )
    parser.add_argument(
        "--gold-csv-max-rows",
        type=int,
        default=150000,
        help="Acima deste tamanho, o gold individual completo fica em Parquet e o CSV vira preview para nao gastar horas escrevendo texto.",
    )
    parser.add_argument(
        "--incluir-metadados-json",
        action="store_true",
        help="Inclui manifestos/resumos .json fora das pastas JSON. Por padrao, o pipeline analisa apenas JSONs de dados.",
    )
    parser.add_argument(
        "--banco-out",
        default="",
        help="Pasta da base bronze/prata/ouro. Padrao: <dados>/banco_eleitoral, ou <pai>/banco_eleitoral quando a entrada for dados/json.",
    )
    parser.add_argument("--banco-overwrite", action="store_true", help="Recria a base limpa se ela ja existir.")
    parser.add_argument("--banco-chunk-rows", type=int, default=0, help="Linhas por parte Parquet no banco limpo. 0 calcula automaticamente.")
    parser.add_argument("--banco-max-files", type=int, default=0, help="Limite de arquivos JSON para teste do modo banco. 0 usa todos.")
    parser.add_argument("--banco-workers", type=int, default=0, help="Workers paralelos para criar bronze/prata com arquivos pequenos e medios. 0 calcula automaticamente.")
    parser.add_argument("--banco-workers-large-files", type=int, default=0, help="Workers paralelos para arquivos grandes no modo banco. 0 calcula automaticamente.")
    parser.add_argument("--banco-large-file-threshold-gb", type=float, default=0.0, help="Acima deste tamanho, o arquivo entra na fila de arquivos grandes do modo banco. 0 calcula automaticamente.")
    parser.add_argument("--banco-ouro-workers", type=int, default=0, help="Queries independentes da camada ouro executadas em paralelo. 0 calcula automaticamente.")
    parser.add_argument("--banco-duckdb-threads", type=int, default=0, help="Total aproximado de threads DuckDB usadas na camada ouro. 0 calcula automaticamente.")
    parser.add_argument("--banco-skip-heavy-analyses", action="store_true", help="Pula analises ouro mais pesadas, como candidato por perfil.")
    parser.add_argument(
        "--banco-ouro-paralelo-agressivo",
        dest="banco_ouro_parallel_aggressive",
        action="store_true",
        help="Permite rodar varias queries pesadas da camada ouro ao mesmo tempo. Mais rapido, mas pode estourar RAM.",
    )
    parser.add_argument(
        "--banco-usar-todos-workers",
        dest="banco_use_all_workers",
        action="store_true",
        help="Forca o modo banco a usar todos os CPUs logicos disponiveis por arquivo. Mais rapido, mas mais agressivo em RAM/IO.",
    )
    parser.add_argument(
        "--banco-apagar-json-apos-processar",
        "--banco-delete-source-after-success",
        dest="banco_delete_source_after_success",
        action="store_true",
        help="Apaga cada JSON/JSONL/NDJSON original somente depois que o Parquet foi gravado com sucesso. Use com cuidado.",
    )

    parser.add_argument("--clustering", action="store_true", default=True)
    parser.add_argument("--sem-clustering", dest="clustering", action="store_false")
    parser.add_argument("--cluster-min-k", type=int, default=4)
    parser.add_argument("--cluster-max-k", type=int, default=12)

    parser.add_argument("--predict-2026", action="store_true")
    parser.add_argument("--cenarios", type=int, default=3000)
    parser.add_argument("--monte-carlo-sigma", type=float, default=0.035)
    parser.add_argument("--prediction-entity", default="auto")
    parser.add_argument("--prediction-cargo-filter", default="")

    parser.add_argument("--parquet", action="store_true", default=True)
    parser.add_argument("--sem-parquet", dest="parquet", action="store_false")
    parser.add_argument("--top-n-html", type=int, default=250)
    parser.add_argument("--top-n-plots", type=int, default=15)
    parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args()

    dados = Path(args.dados).expanduser()
    if not dados.is_absolute():
        dados = Path.cwd() / dados

    out = normalize_output_path(args.out)
    banco_out = normalize_database_path(args.banco_out, dados)
    banco_auto, banco_auto_info = auto_tune_banco_settings(args)

    return Config(
        dados=dados.resolve(),
        out=out.resolve(),
        banco_out=banco_out.resolve(),
        modo=args.modo,
        resume=args.resume,
        seed=args.seed,
        sample_mode=args.sample_mode,
        max_sample_rows=args.max_sample_rows,
        min_sample_rows=args.min_sample_rows,
        sample_frac=args.sample_frac,
        full_aggregations=args.full_aggregations,
        aggregate_chunk_rows=args.aggregate_chunk_rows,
        analysis_max_rows=args.analysis_max_rows,
        global_max_gold_rows=args.global_max_gold_rows,
        gold_csv_max_rows=args.gold_csv_max_rows,
        workers_individual=max(1, int(args.workers_individual or 1)),
        workers_large_files=max(1, int(args.workers_large_files or 1)),
        workers_parquet=max(1, int(args.workers_parquet or 1)),
        large_file_threshold_gb=max(0.0, float(args.large_file_threshold_gb or 0.0)),
        parquet_partition_rows=max(1000, int(args.parquet_partition_rows or 250000)),
        partition_electorate_by_state=bool(args.partition_electorate_by_state),
        engine=args.engine,
        spark_master=args.spark_master,
        incluir_metadados_json=args.incluir_metadados_json,
        banco_overwrite=args.banco_overwrite,
        banco_chunk_rows=int(banco_auto["banco_chunk_rows"]),
        banco_max_files=max(0, int(args.banco_max_files or 0)),
        banco_workers=int(banco_auto["banco_workers"]),
        banco_workers_large_files=int(banco_auto["banco_workers_large_files"]),
        banco_large_file_threshold_gb=float(banco_auto["banco_large_file_threshold_gb"]),
        banco_ouro_workers=int(banco_auto["banco_ouro_workers"]),
        banco_duckdb_threads=int(banco_auto["banco_duckdb_threads"]),
        banco_skip_heavy_analyses=bool(args.banco_skip_heavy_analyses),
        banco_delete_source_after_success=bool(args.banco_delete_source_after_success),
        banco_use_all_workers=bool(args.banco_use_all_workers),
        banco_ouro_parallel_aggressive=bool(args.banco_ouro_parallel_aggressive),
        banco_auto_tune_info=banco_auto_info,
        clustering=args.clustering,
        cluster_min_k=args.cluster_min_k,
        cluster_max_k=args.cluster_max_k,
        predict_2026=args.predict_2026,
        cenarios=args.cenarios,
        monte_carlo_sigma=args.monte_carlo_sigma,
        prediction_entity=args.prediction_entity,
        prediction_cargo_filter=args.prediction_cargo_filter,
        parquet=args.parquet,
        top_n_html=args.top_n_html,
        top_n_plots=args.top_n_plots,
        log_level=args.log_level,
    )
