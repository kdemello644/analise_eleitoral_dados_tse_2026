from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable
from collections import defaultdict
import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
import uuid

import pandas as pd

from .aggregation import record_to_gold_cached
from .discrete import label_category_value
from .json_reader import (
    MAX_JSONL_LINE_CHARS,
    SUPPORTED_JSON_EXT,
    classify_json_document,
    find_json_files,
    first_non_ws_char,
    iter_json_records,
    normalize_record,
)
from .utils import clean_memory, compact_code, extract_years_from_value, parse_number, safe_name, safe_text, save_json

try:
    import pyarrow  # noqa: F401
    PARQUET_ENGINE_OK = True
except Exception:
    PARQUET_ENGINE_OK = False


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
    "geral",
}

PROFILE_COLS = [
    "perfil_faixa_etaria",
    "perfil_genero",
    "perfil_instrucao",
    "perfil_estado_civil",
    "perfil_raca_cor",
]

KEY_COLS = ["ano", "uf", "cd_municipio", "nm_municipio", "zona", "secao", "local_votacao", "bairro", "cargo", "turno"]

ELEITORADO_COLS = [
    *KEY_COLS,
    *PROFILE_COLS,
    "eleitorado",
    "comparecimento_estimado",
    "abstencao_estimado",
    "brancos",
    "nulos",
    "validos_estimados",
    "qtd_registros",
    "schema_id",
    "arquivo_origem",
]

CANDIDATOS_COLS = [
    "ano",
    "uf",
    "cd_municipio",
    "nm_municipio",
    "zona",
    "secao",
    "cargo",
    "turno",
    "partido",
    "candidato",
    "nr_candidato",
    "sq_candidato",
    *PROFILE_COLS,
    "situacao_candidatura",
    "resultado_candidatura",
    "schema_id",
    "arquivo_origem",
]

RESULTADOS_COLS = [
    *KEY_COLS,
    "partido",
    "candidato",
    "nr_votavel",
    "sq_candidato",
    "votos",
    "brancos",
    "nulos",
    "validos_estimados",
    "qtd_registros",
    "schema_id",
    "arquivo_origem",
]

NUMERIC_COLS = {
    "eleitorado": {
        "eleitorado",
        "comparecimento_estimado",
        "abstencao_estimado",
        "brancos",
        "nulos",
        "validos_estimados",
        "qtd_registros",
    },
    "candidatos": set(),
    "resultados_votos": {"votos", "brancos", "nulos", "validos_estimados", "qtd_registros"},
}

TABLE_COLUMNS = {
    "eleitorado": ELEITORADO_COLS,
    "candidatos": CANDIDATOS_COLS,
    "resultados_votos": RESULTADOS_COLS,
}

FILE_PROGRESS_EVERY_ROWS = 500_000
BATCH_PARALLEL_MAX_ROWS = 2_000
DATABASE_MARKER_NAME = "_banco_eleitoral.json"
LEGACY_DATABASE_MARKER_NAME = "_banco_eleitoral_limpo.json"
HEAVY_OURO_LABELS = {"ouro_eleitorado", "ouro_correlacoes"}
RESULTADOS_HASH_BUCKETS = 64
YEAR_UF_PARTITION_COLS = ["ano", "uf"]
MUNICIPIO_PARTITION_COLS = ["ano", "uf", "cd_municipio"]
ZONE_PARTITION_COLS = ["ano", "uf", "cd_municipio", "zona"]
SECTION_PARTITION_COLS = ["ano", "uf", "cd_municipio", "zona", "secao"]
PRATA_MINIMA_LAYOUT_VERSION = 5
PRATA_MINIMA_SECTION_PARTITION_COLS = ["uf"]
PRATA_MINIMA_STREAM_BATCH_ROWS = 20_000
OURO_MUNICIPIO_PARALLEL_MAX = 4
OURO_MUNICIPIO_LARGE_ROWS_THRESHOLD = 250_000
OURO_MUNICIPIO_LARGE_ELEITORADO_THRESHOLD = 1_000_000
ENTITY_PARTITION_COLS = ["nivel", "ano", "uf", "cd_municipio"]
RESULTADOS_SPLIT_LEVELS = [
    ("municipio_bucket", ["cd_municipio"]),
    ("cargo_turno", ["cargo", "turno"]),
    ("zona", ["zona"]),
]


@dataclass
class CleanDatabaseConfig:
    dados: Path
    out: Path
    chunk_rows: int = 100_000
    max_files: int = 0
    workers: int = 1
    workers_large_files: int = 1
    large_file_threshold_gb: float = 8.0
    ouro_workers: int = 2
    duckdb_threads: int = 4
    overwrite: bool = False
    resume: bool = False
    include_metadata: bool = False
    skip_heavy_analyses: bool = False
    only_states_brasil: bool = False
    uf_filter: tuple[str, ...] = ()
    skip_clusters: bool = False
    analysis_mode: str = "completa"
    max_municipios_por_uf: int = 0
    delete_source_after_success: bool = False
    ouro_parallel_aggressive: bool = False
    auto_tune_info: dict[str, Any] | None = None
    log_level: str = "INFO"


def ouro_task_enabled(cfg: CleanDatabaseConfig, task: str) -> bool:
    mode = safe_text(getattr(cfg, "analysis_mode", "completa"), "completa") or "completa"
    if task in {"resumo", "perfil_eleitor"}:
        return mode in {"completa", "eleitor", "candidato", "eleitor_partido", "eleitor_candidato_partido", "estados_brasil"}
    if task in {"resultado_partido", "perfil_partido"}:
        return mode in {"completa", "eleitor_partido", "eleitor_candidato_partido", "estados_brasil"}
    if task in {"resultado_candidato", "perfil_candidato"}:
        if mode == "completa":
            return not cfg.skip_heavy_analyses
        return mode in {"candidato", "eleitor_candidato_partido"}
    if task in {"clusters_eleitores", "clusters_eleitores_resultado"}:
        return mode == "completa" and not cfg.skip_clusters
    return True


class PartitionedParquetWriter:
    def __init__(self, root: Path, chunk_rows: int = 100_000, shard_id: str = "main"):
        self.root = root
        self.chunk_rows = max(1_000, int(chunk_rows or 100_000))
        self.shard_id = safe_name(shard_id or "main", 80)
        self.buffers: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.part_counter: dict[tuple[str, str], int] = defaultdict(int)
        self.rows_written: dict[str, int] = defaultdict(int)

    def add(self, table: str, row: dict[str, Any]) -> None:
        self.buffers[table].append(row)
        if len(self.buffers[table]) >= self.chunk_rows:
            self.flush(table)

    def flush_all(self) -> None:
        for table in list(self.buffers):
            self.flush(table)

    def flush(self, table: str) -> None:
        rows = self.buffers.get(table) or []
        if not rows:
            return
        self.buffers[table] = []
        df = pd.DataFrame(rows)
        df = normalize_table_frame(table, df)
        if df.empty:
            return
        for uf, group in df.groupby("uf", dropna=False, sort=False):
            uf_value = partition_value(uf)
            out_dir = self.root / table / f"uf={uf_value}" / f"shard={self.shard_id}"
            out_dir.mkdir(parents=True, exist_ok=True)
            part_id = self.part_counter[(table, uf_value)]
            self.part_counter[(table, uf_value)] += 1
            path = out_dir / f"part-{part_id:06d}.parquet"
            group.to_parquet(path, index=False, compression="snappy")
            self.rows_written[table] += len(group)
            logging.info("Parquet prata gravado: tabela=%s uf=%s shard=%s parte=%s linhas=%s -> %s", table, uf_value, self.shard_id, part_id, len(group), path)
            write_pipeline_event(
                self.root.parent,
                "parquet_prata",
                "part_gravado",
                tabela=table,
                uf=uf_value,
                shard=self.shard_id,
                parte=part_id,
                linhas=len(group),
                caminho=str(path),
            )


class BronzeParquetWriter:
    def __init__(self, root: Path, chunk_rows: int = 100_000, shard_id: str = "main"):
        self.root = root
        self.chunk_rows = max(1_000, int(chunk_rows or 100_000))
        self.shard_id = safe_name(shard_id or "main", 80)
        self.buffers: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        self.part_counter: dict[tuple[str, str, str], int] = defaultdict(int)
        self.rows_written: int = 0

    def add(self, domain: str, schema_id: str, uf: str, row: dict[str, Any]) -> None:
        uf_value = partition_value(uf)
        key = (domain, schema_id, uf_value)
        self.buffers[key].append(row)
        if len(self.buffers[key]) >= self.chunk_rows:
            self.flush_key(key)

    def flush_all(self) -> None:
        for key in list(self.buffers):
            self.flush_key(key)

    def flush_key(self, key: tuple[str, str, str]) -> None:
        rows = self.buffers.get(key) or []
        if not rows:
            return
        self.buffers[key] = []
        domain, schema_id, uf_value = key
        out_dir = self.root / domain / f"schema_id={safe_name(schema_id, 40)}" / f"uf={uf_value}" / f"shard={self.shard_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        part_id = self.part_counter[key]
        self.part_counter[key] += 1
        df = pd.DataFrame(rows)
        path = out_dir / f"part-{part_id:06d}.parquet"
        df.to_parquet(path, index=False, compression="snappy")
        self.rows_written += len(df)
        logging.info("Parquet bronze gravado: dominio=%s schema=%s uf=%s shard=%s parte=%s linhas=%s -> %s", domain, schema_id, uf_value, self.shard_id, part_id, len(df), path)
        write_pipeline_event(
            self.root.parent,
            "parquet_bronze",
            "part_gravado",
            dominio=domain,
            schema_id=schema_id,
            uf=uf_value,
            shard=self.shard_id,
            parte=part_id,
            linhas=len(df),
            caminho=str(path),
        )


def build_clean_database(cfg: CleanDatabaseConfig) -> dict[str, Any]:
    ensure_parquet_engine()
    prepare_output_dir(cfg)
    setup_clean_database_logging(cfg.out, cfg.log_level)
    marker = cfg.out / DATABASE_MARKER_NAME
    save_json({"status": "em_processamento", **asdict(cfg)}, marker)

    files = find_json_files(cfg.dados, include_metadata=cfg.include_metadata)
    if cfg.max_files and cfg.max_files > 0:
        files = files[: cfg.max_files]

    previous_manifests = load_resume_manifests(cfg) if cfg.resume else []
    manifests: list[dict[str, Any]] = list(previous_manifests)
    schema_rows: dict[str, dict[str, Any]] = load_resume_schemas(cfg) if cfg.resume else {}
    rows_written: dict[str, int] = defaultdict(int)
    bronze_rows_written = 0

    logging.info("Banco eleitoral limpo iniciado.")
    write_pipeline_event(cfg.out, "banco", "inicio", entrada=str(cfg.dados), saida=str(cfg.out), arquivos_encontrados=len(files))
    logging.info("Entrada JSON: %s", cfg.dados)
    logging.info("Saida Parquet: %s", cfg.out)
    logging.info("Arquivos encontrados: %s", len(files))
    logging.info(
        "Auto-tuning banco: chunk_rows=%s, workers=%s, workers_grandes=%s, threshold_gb=%s, ouro_workers=%s, duckdb_threads=%s",
        cfg.chunk_rows,
        cfg.workers,
        cfg.workers_large_files,
        cfg.large_file_threshold_gb,
        cfg.ouro_workers,
        cfg.duckdb_threads,
    )
    if cfg.auto_tune_info:
        logging.info("Auto-tuning detalhes: %s", json.dumps(cfg.auto_tune_info, ensure_ascii=False, default=str))

    work_items = build_database_work_items(files, cfg)
    ignored = [item for item in work_items if item["dominio"] not in {"eleitorado", "candidatos", "resultados"}]
    for item in ignored:
        logging.info("Ignorando dominio %s: %s", item["dominio"], item["relativo"])
    process_items_all = [item for item in work_items if item["dominio"] in {"eleitorado", "candidatos", "resultados"}]
    processed_files = {
        str(m.get("arquivo_origem", ""))
        for m in previous_manifests
        if (
            m.get("arquivo_origem")
            and not safe_text(m.get("erro", ""))
            and not safe_text(m.get("erros_faixas", ""))
            and not safe_text(m.get("erros_lotes", ""))
        )
    }
    if processed_files:
        logging.info("Resume ativo: %s documentos ja concluidos serao pulados.", len(processed_files))
    process_items = [item for item in process_items_all if str(item.get("relativo", "")) not in processed_files]
    ordered_items = sorted(
        process_items,
        key=lambda item: (float(item.get("tamanho_gb", 0) or 0), str(item.get("relativo", ""))),
    )
    total_with_resume = len(processed_files) + len(process_items)
    threshold = float(cfg.large_file_threshold_gb or 0)
    small_items = [item for item in ordered_items if float(item.get("tamanho_gb", 0)) < threshold]
    large_items = [item for item in ordered_items if float(item.get("tamanho_gb", 0)) >= threshold]

    logging.info(
        "Arquivos validos para banco: %s pequenos/medios, %s grandes. Fila menor->maior, arquivo-a-arquivo, ate %s workers internos por arquivo.",
        len(small_items),
        len(large_items),
        max(int(cfg.workers or 1), int(cfg.workers_large_files or 1)),
    )
    save_json(
        {
            "status": "planejado",
            "total_documentos_validos": total_with_resume,
            "documentos_ja_concluidos_resume": len(processed_files),
            "documentos_pendentes": len(process_items),
            "pequenos_medios": len(small_items),
            "grandes": len(large_items),
            "ordem_processamento": "menor_para_maior",
            "documentos": [
                {
                    "ordem": idx,
                    "arquivo": item.get("relativo", ""),
                    "dominio": item.get("dominio", ""),
                    "assunto": item.get("assunto", ""),
                    "ano": item.get("ano_detectado_nome", ""),
                    "tamanho_gb": round(float(item.get("tamanho_gb", 0) or 0), 3),
                    "fila": "grandes" if item in large_items else "pequenos_medios",
                }
                for idx, item in enumerate(ordered_items, start=1)
            ],
        },
        cfg.out / "logs" / "documentos_banco_planejados.json",
    )
    write_pipeline_event(
        cfg.out,
        "banco",
        "fila_planejada",
        total_documentos_validos=total_with_resume,
        documentos_ja_concluidos_resume=len(processed_files),
        documentos_pendentes=len(process_items),
        pequenos_medios=len(small_items),
        grandes=len(large_items),
    )

    done = len(processed_files)
    total = total_with_resume
    completed_documents: list[dict[str, Any]] = [
        {
            "arquivo": m.get("arquivo_origem", ""),
            "dominio": m.get("dominio", ""),
            "linhas_lidas": m.get("linhas_lidas", 0),
            "linhas_gravadas": m.get("linhas_gravadas", 0),
            "processamento": m.get("processamento", "resume_concluido"),
        }
        for m in previous_manifests[-50:]
    ]

    def save_progress(fase: str, atual: str = "") -> None:
        write_pipeline_event(
            cfg.out,
            "banco",
            fase,
            documento_atual=atual,
            total_documentos=total,
            documentos_concluidos=done,
            documentos_pendentes=max(0, total - done),
        )
        save_json(
            {
                "status": "em_processamento" if done < total else "ok",
                "fase": fase,
                "documento_atual": atual,
                "total_documentos": total,
                "documentos_concluidos": done,
                "documentos_pendentes": max(0, total - done),
                "concluidos": completed_documents[-50:],
            },
            cfg.out / "logs" / "progresso_banco.json",
        )

    def merge_result(result: dict[str, Any]) -> None:
        nonlocal done, bronze_rows_written
        done += 1
        manifest = result.get("manifest", {})
        current_file = str(manifest.get("arquivo_origem", ""))
        if current_file:
            manifests[:] = [m for m in manifests if str(m.get("arquivo_origem", "")) != current_file]
        completed_documents.append({
            "arquivo": manifest.get("arquivo_origem", ""),
            "dominio": manifest.get("dominio", ""),
            "linhas_lidas": manifest.get("linhas_lidas", 0),
            "linhas_gravadas": manifest.get("linhas_gravadas", 0),
            "processamento": manifest.get("processamento", "streaming"),
        })
        manifests.append(manifest)
        for schema_id, row in (result.get("schemas") or {}).items():
            schema_rows.setdefault(schema_id, row)
        for table, count in (result.get("linhas_por_tabela") or {}).items():
            rows_written[table] += int(count or 0)
        bronze_rows_written += int(result.get("linhas_bronze", 0) or 0)
        logging.info(
            "(%s/%s) Banco concluido: %s | lidas=%s gravadas=%s | modo=%s",
            done,
            total,
            manifest.get("arquivo_origem", ""),
            manifest.get("linhas_lidas", 0),
            manifest.get("linhas_gravadas", 0),
            manifest.get("processamento", "streaming"),
        )
        write_pipeline_event(
            cfg.out,
            "banco",
            "documento_concluido",
            arquivo=manifest.get("arquivo_origem", ""),
            dominio=manifest.get("dominio", ""),
            linhas_lidas=manifest.get("linhas_lidas", 0),
            linhas_gravadas=manifest.get("linhas_gravadas", 0),
            modo=manifest.get("processamento", "streaming"),
            documento_indice=done,
            total_documentos=total,
        )
        maybe_delete_source_after_success(cfg, manifest)
        save_json(manifests, cfg.out / "logs" / "manifesto_arquivos_parcial.json")
        save_progress("processando", str(manifest.get("arquivo_origem", "")))
        clean_memory()

    save_progress("inicio")
    for idx, item in enumerate(ordered_items, start=1):
        save_progress("processando_documento", str(item.get("relativo", "")))
        try:
            result = process_one_file_with_all_workers(item, cfg, idx=idx, total=len(ordered_items))
        except Exception as exc:
            logging.exception("Documento falhou depois dos retries e sera marcado com erro: %s", item.get("relativo", ""))
            result = error_result_for_item(item, exc)
        merge_result(result)
    save_progress("finalizado")

    if cfg.resume:
        existing_counts = count_existing_silver_rows(cfg.out / "prata")
        rows_written = defaultdict(int, {k: int(v) for k, v in existing_counts.items()})
        bronze_rows_written = count_parquet_rows(cfg.out / "bronze")

    manifest = pd.DataFrame(manifests)
    schemas = pd.DataFrame(schema_rows.values())
    write_small_parquet(manifest, cfg.out / "metadados" / "manifesto_arquivos.parquet")
    write_small_parquet(schemas, cfg.out / "metadados" / "manifesto_esquemas.parquet")

    analysis_info = {
        "status": "pendente",
        "observacao": "Execute --modo analise_banco para gerar a camada ouro, analises e simulacao.",
    }
    summary = {
        "status": "ok",
        "entrada": str(cfg.dados),
        "saida": str(cfg.out),
        "arquivos_processados": int(len(manifests)),
        "linhas_por_tabela": dict(rows_written),
        "linhas_bronze": int(bronze_rows_written),
        "paralelismo": {
            "workers_bronze_prata": int(cfg.workers),
            "workers_arquivos_grandes": int(cfg.workers_large_files),
            "large_file_threshold_gb": float(cfg.large_file_threshold_gb),
            "chunk_rows": int(cfg.chunk_rows),
            "ouro_workers": int(cfg.ouro_workers),
            "duckdb_threads": int(cfg.duckdb_threads),
            "auto_tune": cfg.auto_tune_info or {},
        },
        "camadas": {
            "bronze": str(cfg.out / "bronze"),
            "prata": str(cfg.out / "prata"),
            "ouro": str(cfg.out / "ouro"),
        },
        "datasets": {
            "eleitorado": str(cfg.out / "prata" / "eleitorado"),
            "candidatos": str(cfg.out / "prata" / "candidatos"),
            "resultados_votos": str(cfg.out / "prata" / "resultados_votos"),
            "resultados_vencedores_secao": str(cfg.out / "ouro" / "resultados_vencedores_secao"),
            "analises": str(cfg.out / "ouro"),
        },
        "analises": analysis_info,
    }
    save_json(summary, marker)
    logging.info("Banco eleitoral limpo finalizado: %s", cfg.out)
    write_pipeline_event(cfg.out, "banco", "fim", saida=str(cfg.out), arquivos_processados=len(manifests), linhas_por_tabela=dict(rows_written))
    return summary


def build_database_work_items(files: list[Path], cfg: CleanDatabaseConfig) -> list[dict[str, Any]]:
    items = []
    for path in files:
        rel = safe_rel(path, cfg.dados)
        cls = classify_json_document(path)
        items.append({
            "path": str(path),
            "relativo": rel,
            "dominio": cls.get("dominio_documento", "outros"),
            "assunto": cls.get("assunto_documento", ""),
            "ano_detectado_nome": ",".join(map(str, extract_years_from_value(rel))),
            "tamanho_gb": file_size_gb(path),
            "shard_id": shard_id_for_file(rel),
        })
    return items


def file_size_gb(path: Path) -> float:
    try:
        return path.stat().st_size / (1024 ** 3)
    except Exception:
        return 0.0


def load_resume_manifests(cfg: CleanDatabaseConfig) -> list[dict[str, Any]]:
    partial = cfg.out / "logs" / "manifesto_arquivos_parcial.json"
    complete = cfg.out / "metadados" / "manifesto_arquivos.parquet"
    rows: list[dict[str, Any]] = []
    if partial.exists():
        try:
            data = json.loads(partial.read_text(encoding="utf-8"))
            if isinstance(data, list):
                rows = [r for r in data if isinstance(r, dict)]
        except Exception as exc:
            logging.warning("Nao consegui ler manifesto parcial para resume: %s", exc)
    elif complete.exists():
        try:
            rows = pd.read_parquet(complete).to_dict("records")
        except Exception as exc:
            logging.warning("Nao consegui ler manifesto Parquet para resume: %s", exc)

    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("arquivo_origem", ""))
        if key:
            deduped[key] = row
    return list(deduped.values())


def load_resume_schemas(cfg: CleanDatabaseConfig) -> dict[str, dict[str, Any]]:
    path = cfg.out / "metadados" / "manifesto_esquemas.parquet"
    if not path.exists():
        return {}
    try:
        rows = pd.read_parquet(path).to_dict("records")
    except Exception as exc:
        logging.warning("Nao consegui ler schemas anteriores para resume: %s", exc)
        return {}
    return {
        str(row.get("schema_id", "")): row
        for row in rows
        if row.get("schema_id")
    }


def count_parquet_rows(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        import pyarrow.parquet as pq
    except Exception:
        return 0
    total = 0
    for parquet_file in path.rglob("*.parquet"):
        try:
            total += int(pq.ParquetFile(parquet_file).metadata.num_rows)
        except Exception:
            continue
    return total


def count_existing_silver_rows(root: Path) -> dict[str, int]:
    return {
        table: count_parquet_rows(root / table)
        for table in TABLE_COLUMNS
    }


def remove_existing_shard_outputs(cfg: CleanDatabaseConfig, shard_id: str, domain: str, table: str | None) -> None:
    if table:
        table_root = cfg.out / "prata" / table
        if table_root.exists():
            for shard_dir in table_root.glob(f"uf=*/shard={shard_id}"):
                if shard_dir.is_dir():
                    shutil.rmtree(shard_dir, ignore_errors=True)

    bronze_root = cfg.out / "bronze" / domain
    if bronze_root.exists():
        for shard_dir in bronze_root.glob(f"schema_id=*/uf=*/shard={shard_id}"):
            if shard_dir.is_dir():
                shutil.rmtree(shard_dir, ignore_errors=True)


def remove_existing_shard_prefix_outputs(cfg: CleanDatabaseConfig, shard_prefix: str, domain: str, table: str | None) -> None:
    prefix = f"shard={shard_prefix}"
    if table:
        table_root = cfg.out / "prata" / table
        if table_root.exists():
            for shard_dir in table_root.glob("uf=*/shard=*"):
                if shard_dir.is_dir() and shard_dir.name.startswith(prefix):
                    shutil.rmtree(shard_dir, ignore_errors=True)

    bronze_root = cfg.out / "bronze" / domain
    if bronze_root.exists():
        for shard_dir in bronze_root.glob("schema_id=*/uf=*/shard=*"):
            if shard_dir.is_dir() and shard_dir.name.startswith(prefix):
                shutil.rmtree(shard_dir, ignore_errors=True)


def maybe_delete_source_after_success(cfg: CleanDatabaseConfig, manifest: dict[str, Any]) -> None:
    if not cfg.delete_source_after_success:
        return
    if safe_text(manifest.get("erro", "")) or safe_text(manifest.get("erros_faixas", "")) or safe_text(manifest.get("erros_lotes", "")):
        manifest["arquivo_origem_apagado"] = False
        manifest["arquivo_origem_apagado_motivo"] = "processamento_com_erro"
        return

    rel = safe_text(manifest.get("arquivo_origem", ""))
    if not rel:
        return
    source = (cfg.dados / rel).resolve()
    root = cfg.dados.resolve()
    try:
        source.relative_to(root)
    except ValueError:
        manifest["arquivo_origem_apagado"] = False
        manifest["arquivo_origem_apagado_motivo"] = "fora_da_pasta_dados"
        logging.warning("Recusei apagar arquivo fora da pasta dados: %s", source)
        return

    if source.suffix.lower() not in SUPPORTED_JSON_EXT:
        manifest["arquivo_origem_apagado"] = False
        manifest["arquivo_origem_apagado_motivo"] = "extensao_nao_suportada"
        return
    if not source.exists():
        manifest["arquivo_origem_apagado"] = False
        manifest["arquivo_origem_apagado_motivo"] = "arquivo_ja_nao_existe"
        return

    try:
        source.unlink()
        manifest["arquivo_origem_apagado"] = True
        manifest["arquivo_origem_apagado_motivo"] = "apagado_apos_parquet_ok"
        logging.info("Arquivo original apagado apos sucesso: %s", source)
    except Exception as exc:
        manifest["arquivo_origem_apagado"] = False
        manifest["arquivo_origem_apagado_motivo"] = str(exc)
        logging.warning("Nao consegui apagar arquivo original %s: %s", source, exc)


def shard_id_for_file(rel: str) -> str:
    digest = hashlib.sha1(rel.encode("utf-8", errors="ignore")).hexdigest()[:14]
    return f"{digest}_{safe_name(Path(rel).stem, 48)}"


def describe_work_item(item: dict[str, Any]) -> str:
    size_gb = float(item.get("tamanho_gb", 0) or 0)
    return (
        f"{item.get('relativo', '')} | dominio={item.get('dominio', '')} "
        f"| assunto={item.get('assunto', '')} | ano={item.get('ano_detectado_nome', '')} "
        f"| tamanho={size_gb:.3f} GB"
    )


def error_result_for_item(item: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        "manifest": {
            "arquivo_origem": item.get("relativo", ""),
            "dominio": item.get("dominio", ""),
            "assunto": item.get("assunto", ""),
            "ano_detectado_nome": item.get("ano_detectado_nome", ""),
            "tamanho_gb": item.get("tamanho_gb", 0),
            "shard_id": item.get("shard_id", ""),
            "linhas_lidas": 0,
            "linhas_gravadas": 0,
            "schema_id": "",
            "processamento": "erro_depois_dos_retries",
            "erro": str(exc),
        },
        "schemas": {},
        "linhas_por_tabela": {},
        "linhas_bronze": 0,
    }


def process_one_file_with_all_workers(
    item: dict[str, Any],
    cfg: CleanDatabaseConfig,
    idx: int,
    total: int,
) -> dict[str, Any]:
    requested_workers = max(1, int(max(cfg.workers or 1, cfg.workers_large_files or 1)))
    path = Path(str(item["path"]))
    logging.info(
        "Arquivo %s/%s tentando ate %s workers: %s",
        idx,
        total,
        requested_workers,
        describe_work_item(item),
    )
    write_pipeline_event(
        cfg.out,
        "arquivo",
        "inicio",
        arquivo=item.get("relativo", ""),
        indice=idx,
        total=total,
        workers_solicitados=requested_workers,
        dominio=item.get("dominio", ""),
        tamanho_gb=item.get("tamanho_gb", 0),
    )
    last_exc: Exception | None = None
    for workers in worker_retry_ladder(requested_workers):
        try:
            logging.info("Tentativa de processamento com %s worker(s): %s", workers, item.get("relativo", ""))
            write_pipeline_event(cfg.out, "arquivo", "tentativa_workers", arquivo=item.get("relativo", ""), workers=workers)
            if workers <= 1:
                return process_json_file_item(item, cfg)
            if can_parallelize_jsonl(path):
                return process_large_jsonl_file_parallel(item, cfg, max_workers=workers)
            return process_json_file_parallel_batches(item, cfg, max_workers=workers)
        except (BrokenProcessPool, MemoryError, OSError, RuntimeError) as exc:
            last_exc = exc
            logging.exception(
                "Falha com %s worker(s) em %s. Vou reduzir o paralelismo e tentar novamente.",
                workers,
                item.get("relativo", ""),
            )
            write_pipeline_event(cfg.out, "arquivo", "erro_tentativa_workers", arquivo=item.get("relativo", ""), workers=workers, erro=str(exc))
            clean_memory()
            continue
    raise RuntimeError(f"Falha processando {item.get('relativo', '')}: {last_exc}")


def worker_retry_ladder(requested_workers: int) -> list[int]:
    workers = max(1, int(requested_workers or 1))
    candidates = [workers, max(1, workers // 2), max(1, workers // 4), 2, 1]
    out: list[int] = []
    for value in candidates:
        if value not in out:
            out.append(value)
    return out


def process_work_items_parallel(
    items: list[dict[str, Any]],
    cfg: CleanDatabaseConfig,
    max_workers: int,
    label: str,
) -> Iterable[dict[str, Any]]:
    if not items:
        return
    workers = min(max(1, int(max_workers or 1)), len(items))
    if workers <= 1:
        logging.info("Processando %s arquivos %s em serie.", len(items), label)
        write_pipeline_event(cfg.out, "fila_arquivos", "inicio_serie", label=label, total=len(items), workers=workers)
        for idx, item in enumerate(items, start=1):
            logging.info(
                "Iniciando documento %s/%s [%s]: %s",
                idx,
                len(items),
                label,
                describe_work_item(item),
            )
            write_pipeline_event(cfg.out, "fila_arquivos", "documento_inicio", label=label, indice=idx, total=len(items), arquivo=item.get("relativo", ""))
            yield process_json_file_item(item, cfg)
        return

    logging.info("Processando %s arquivos %s em paralelo com %s workers.", len(items), label, workers)
    write_pipeline_event(cfg.out, "fila_arquivos", "inicio_paralelo", label=label, total=len(items), workers=workers)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_map = {}
        for idx, item in enumerate(items, start=1):
            logging.info(
                "Enfileirando documento %s/%s [%s]: %s",
                idx,
                len(items),
                label,
                describe_work_item(item),
            )
            write_pipeline_event(cfg.out, "fila_arquivos", "documento_enfileirado", label=label, indice=idx, total=len(items), arquivo=item.get("relativo", ""))
            future_map[pool.submit(process_json_file_item, item, cfg)] = item
        for future in as_completed(future_map):
            item = future_map[future]
            try:
                result = future.result()
                manifest = result.get("manifest", {})
                logging.info(
                    "Documento finalizado [%s]: %s | lidas=%s gravadas=%s",
                    label,
                    manifest.get("arquivo_origem", item.get("relativo", "")),
                    manifest.get("linhas_lidas", 0),
                    manifest.get("linhas_gravadas", 0),
                )
                write_pipeline_event(cfg.out, "fila_arquivos", "documento_finalizado", label=label, arquivo=manifest.get("arquivo_origem", item.get("relativo", "")), linhas_lidas=manifest.get("linhas_lidas", 0), linhas_gravadas=manifest.get("linhas_gravadas", 0))
                yield result
            except Exception as exc:
                logging.exception("Erro processando banco %s: %s", item.get("relativo"), exc)
                write_pipeline_event(cfg.out, "fila_arquivos", "documento_erro", label=label, arquivo=item.get("relativo", ""), erro=str(exc))
                yield {
                    "manifest": {
                        "arquivo_origem": item.get("relativo", ""),
                        "dominio": item.get("dominio", ""),
                        "assunto": item.get("assunto", ""),
                        "ano_detectado_nome": item.get("ano_detectado_nome", ""),
                        "linhas_lidas": 0,
                        "linhas_gravadas": 0,
                        "schema_id": "",
                        "erro": str(exc),
                    },
                    "schemas": {},
                    "linhas_por_tabela": {},
                    "linhas_bronze": 0,
                }


def process_large_items_adaptive(items: list[dict[str, Any]], cfg: CleanDatabaseConfig) -> Iterable[dict[str, Any]]:
    if not items:
        return

    total = len(items)
    inner_workers = max(1, int(max(cfg.workers or 1, cfg.workers_large_files or 1)))
    for idx, item in enumerate(items, start=1):
        path = Path(str(item["path"]))
        if can_parallelize_jsonl(path) and inner_workers > 1:
            logging.info(
                "Iniciando documento grande %s/%s com %s workers internos por faixa JSONL: %s",
                idx,
                total,
                inner_workers,
                describe_work_item(item),
            )
            write_pipeline_event(cfg.out, "arquivo_grande", "inicio_jsonl_paralelo", arquivo=item.get("relativo", ""), indice=idx, total=total, workers=inner_workers)
            yield process_large_jsonl_file_parallel(item, cfg, max_workers=inner_workers)
        else:
            reason = "formato nao particionavel" if not can_parallelize_jsonl(path) else "worker interno unico"
            logging.info(
                "Iniciando documento grande %s/%s em streaming serial (%s): %s",
                idx,
                total,
                reason,
                describe_work_item(item),
            )
            write_pipeline_event(cfg.out, "arquivo_grande", "inicio_streaming_serial", arquivo=item.get("relativo", ""), indice=idx, total=total, motivo=reason)
            yield process_json_file_item(item, cfg)


def can_parallelize_jsonl(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix not in {".jsonl", ".ndjson"}:
        return False
    return first_non_ws_char(path) != "["


def process_large_jsonl_file_parallel(item: dict[str, Any], cfg: CleanDatabaseConfig, max_workers: int) -> dict[str, Any]:
    started = time.perf_counter()
    path = Path(str(item["path"]))
    file_size = path.stat().st_size
    base_shard = str(item.get("shard_id") or shard_id_for_file(str(item.get("relativo", ""))))
    domain = str(item.get("dominio", ""))
    table = domain_table(domain)
    remove_existing_shard_prefix_outputs(cfg, base_shard, domain, table)
    ranges = build_byte_ranges(file_size, max_workers)
    if len(ranges) <= 1:
        return process_json_file_item(item, cfg)

    schemas: dict[str, dict[str, Any]] = {}
    rows_written: dict[str, int] = defaultdict(int)
    bronze_rows_written = 0
    read_rows = 0
    written_rows = 0
    errors: list[str] = []
    workers = min(max(1, int(max_workers or 1)), len(ranges))

    logging.info(
        "Arquivo grande particionado: %s | partes=%s | workers=%s | tamanho=%.3f GB",
        item.get("relativo", ""),
        len(ranges),
        workers,
        file_size / (1024 ** 3),
    )
    write_pipeline_event(
        cfg.out,
        "jsonl_faixas",
        "inicio",
        arquivo=item.get("relativo", ""),
        partes=len(ranges),
        workers=workers,
        tamanho_gb=round(file_size / (1024 ** 3), 6),
    )

    chunk_items = []
    for range_index, (start, end) in enumerate(ranges):
        chunk = dict(item)
        chunk["range_start"] = int(start)
        chunk["range_end"] = int(end)
        chunk["range_index"] = int(range_index)
        chunk["range_total"] = int(len(ranges))
        chunk["shard_id"] = f"{base_shard}_p{range_index:03d}"
        chunk_items.append(chunk)

    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_map = {}
        for chunk in chunk_items:
            logging.info(
                "Enfileirando faixa %s/%s de %s | bytes %s-%s",
                int(chunk.get("range_index", 0)) + 1,
                int(chunk.get("range_total", 0)),
                item.get("relativo", ""),
                chunk.get("range_start", 0),
                chunk.get("range_end", 0),
            )
            write_pipeline_event(
                cfg.out,
                "jsonl_faixas",
                "faixa_enfileirada",
                arquivo=item.get("relativo", ""),
                faixa=int(chunk.get("range_index", 0)) + 1,
                total_faixas=int(chunk.get("range_total", 0)),
                byte_inicio=chunk.get("range_start", 0),
                byte_fim=chunk.get("range_end", 0),
            )
            future_map[pool.submit(process_jsonl_range_item, chunk, cfg)] = chunk
        for future in as_completed(future_map):
            chunk = future_map[future]
            try:
                result = future.result()
            except Exception as exc:
                logging.exception(
                    "Erro processando faixa %s de %s: %s",
                    chunk.get("range_index"),
                    item.get("relativo"),
                    exc,
                )
                errors.append(f"faixa {chunk.get('range_index')}: {exc}")
                write_pipeline_event(cfg.out, "jsonl_faixas", "faixa_erro", arquivo=item.get("relativo", ""), faixa=chunk.get("range_index"), erro=str(exc))
                continue

            manifest = result.get("manifest", {})
            logging.info(
                "Faixa concluida %s/%s de %s | lidas=%s gravadas=%s",
                int(chunk.get("range_index", 0)) + 1,
                int(chunk.get("range_total", 0)),
                item.get("relativo", ""),
                manifest.get("linhas_lidas", 0),
                manifest.get("linhas_gravadas", 0),
            )
            write_pipeline_event(
                cfg.out,
                "jsonl_faixas",
                "faixa_concluida",
                arquivo=item.get("relativo", ""),
                faixa=int(chunk.get("range_index", 0)) + 1,
                total_faixas=int(chunk.get("range_total", 0)),
                linhas_lidas=manifest.get("linhas_lidas", 0),
                linhas_gravadas=manifest.get("linhas_gravadas", 0),
            )
            read_rows += int(manifest.get("linhas_lidas", 0) or 0)
            written_rows += int(manifest.get("linhas_gravadas", 0) or 0)
            for schema_id, row in (result.get("schemas") or {}).items():
                schemas.setdefault(schema_id, row)
            for table, count in (result.get("linhas_por_tabela") or {}).items():
                rows_written[table] += int(count or 0)
            bronze_rows_written += int(result.get("linhas_bronze", 0) or 0)

    if errors:
        write_pipeline_event(cfg.out, "jsonl_faixas", "erro", arquivo=item.get("relativo", ""), erros=errors[:20])
        raise RuntimeError(f"Falha em faixas de {item.get('relativo', '')}: {' | '.join(errors[:5])}")

    manifest = {
        "arquivo_origem": item.get("relativo", ""),
        "dominio": item.get("dominio", ""),
        "assunto": item.get("assunto", ""),
        "ano_detectado_nome": item.get("ano_detectado_nome", ""),
        "tamanho_gb": item.get("tamanho_gb", 0),
        "shard_id": item.get("shard_id", ""),
        "linhas_lidas": read_rows,
        "linhas_gravadas": written_rows,
        "schema_id": ",".join(sorted(schemas.keys())[:10]),
        "processamento": "jsonl_paralelo_por_faixa",
        "partes_arquivo": len(ranges),
        "workers_internos": workers,
        "duracao_segundos": round(time.perf_counter() - started, 3),
        "mb_por_segundo": round((file_size / (1024 ** 2)) / max(0.001, time.perf_counter() - started), 3),
        "erros_faixas": " | ".join(errors),
    }
    write_pipeline_event(
        cfg.out,
        "jsonl_faixas",
        "fim",
        arquivo=item.get("relativo", ""),
        partes=len(ranges),
        workers=workers,
        linhas_lidas=read_rows,
        linhas_gravadas=written_rows,
        duracao_segundos=manifest["duracao_segundos"],
    )
    return {
        "manifest": manifest,
        "schemas": schemas,
        "linhas_por_tabela": dict(rows_written),
        "linhas_bronze": int(bronze_rows_written),
    }


def build_byte_ranges(file_size: int, max_workers: int) -> list[tuple[int, int]]:
    workers = max(1, int(max_workers or 1))
    if file_size <= 0 or workers <= 1:
        return [(0, file_size)]
    chunk_size = max(1, file_size // workers)
    ranges: list[tuple[int, int]] = []
    start = 0
    for idx in range(workers):
        end = file_size if idx == workers - 1 else min(file_size, start + chunk_size)
        if start < end:
            ranges.append((start, end))
        start = end
    return ranges


def process_json_file_parallel_batches(item: dict[str, Any], cfg: CleanDatabaseConfig, max_workers: int) -> dict[str, Any]:
    started = time.perf_counter()
    path = Path(str(item["path"]))
    rel = str(item["relativo"])
    domain = str(item["dominio"])
    base_shard = str(item.get("shard_id") or shard_id_for_file(rel))
    table = domain_table(domain)
    remove_existing_shard_prefix_outputs(cfg, base_shard, domain, table)

    workers = max(1, int(max_workers or 1))
    batch_rows = max(1_000, min(int(cfg.chunk_rows or BATCH_PARALLEL_MAX_ROWS), BATCH_PARALLEL_MAX_ROWS))
    max_pending = max(1, workers)
    schemas: dict[str, dict[str, Any]] = {}
    rows_written: dict[str, int] = defaultdict(int)
    bronze_rows_written = 0
    read_rows = 0
    written_rows = 0
    errors: list[str] = []
    batch_index = 0
    submitted_batches = 0
    pending = set()
    future_map: dict[Any, dict[str, Any]] = {}

    logging.info(
        "Arquivo nao particionavel por bytes: usando leitor streaming + lotes paralelos | workers=%s | batch_rows=%s | %s",
        workers,
        batch_rows,
        describe_work_item(item),
    )
    write_pipeline_event(cfg.out, "json_lotes", "inicio", arquivo=rel, workers=workers, batch_rows=batch_rows, dominio=domain)

    def merge_batch_result(result: dict[str, Any], chunk: dict[str, Any]) -> None:
        nonlocal bronze_rows_written, read_rows, written_rows
        manifest = result.get("manifest", {})
        read_rows += int(manifest.get("linhas_lidas", 0) or 0)
        written_rows += int(manifest.get("linhas_gravadas", 0) or 0)
        for schema_id, row in (result.get("schemas") or {}).items():
            schemas.setdefault(schema_id, row)
        for table_name, count in (result.get("linhas_por_tabela") or {}).items():
            rows_written[table_name] += int(count or 0)
        bronze_rows_written += int(result.get("linhas_bronze", 0) or 0)
        logging.info(
            "Lote concluido %s de %s | lidas=%s gravadas=%s",
            int(chunk.get("batch_index", 0)) + 1,
            rel,
            manifest.get("linhas_lidas", 0),
            manifest.get("linhas_gravadas", 0),
        )
        write_pipeline_event(
            cfg.out,
            "json_lotes",
            "lote_concluido",
            arquivo=rel,
            lote=int(chunk.get("batch_index", 0)) + 1,
            linhas_lidas=manifest.get("linhas_lidas", 0),
            linhas_gravadas=manifest.get("linhas_gravadas", 0),
        )

    def drain_completed(done_futures: set[Any]) -> None:
        for future in done_futures:
            chunk = future_map.pop(future, {})
            try:
                merge_batch_result(future.result(), chunk)
            except Exception as exc:
                logging.exception("Erro processando lote %s de %s: %s", chunk.get("batch_index"), rel, exc)
                errors.append(f"lote {chunk.get('batch_index')}: {exc}")
                write_pipeline_event(cfg.out, "json_lotes", "lote_erro", arquivo=rel, lote=chunk.get("batch_index"), erro=str(exc))

    with ProcessPoolExecutor(max_workers=workers) as pool:
        batch: list[dict[str, Any]] = []
        for rec in iter_json_records(path):
            batch.append(rec)
            if len(batch) < batch_rows:
                continue

            chunk = build_batch_item(item, batch, base_shard, batch_index)
            logging.info("Enfileirando lote %s de %s | linhas=%s", batch_index + 1, rel, len(batch))
            write_pipeline_event(cfg.out, "json_lotes", "lote_enfileirado", arquivo=rel, lote=batch_index + 1, linhas=len(batch))
            future = pool.submit(process_record_batch_item, chunk, cfg)
            pending.add(future)
            future_map[future] = chunk
            batch_index += 1
            submitted_batches += 1
            batch = []

            if len(pending) >= max_pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                drain_completed(done)

        if batch:
            chunk = build_batch_item(item, batch, base_shard, batch_index)
            logging.info("Enfileirando lote %s de %s | linhas=%s", batch_index + 1, rel, len(batch))
            write_pipeline_event(cfg.out, "json_lotes", "lote_enfileirado", arquivo=rel, lote=batch_index + 1, linhas=len(batch))
            future = pool.submit(process_record_batch_item, chunk, cfg)
            pending.add(future)
            future_map[future] = chunk
            submitted_batches += 1

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            drain_completed(done)

    if errors:
        write_pipeline_event(cfg.out, "json_lotes", "erro", arquivo=rel, erros=errors[:20])
        raise RuntimeError(f"Falha em lotes de {rel}: {' | '.join(errors[:5])}")

    size_mb = float(item.get("tamanho_gb", 0) or 0) * 1024
    duration = time.perf_counter() - started
    write_pipeline_event(
        cfg.out,
        "json_lotes",
        "fim",
        arquivo=rel,
        lotes=submitted_batches,
        workers=workers,
        linhas_lidas=read_rows,
        linhas_gravadas=written_rows,
        duracao_segundos=round(duration, 3),
    )
    return {
        "manifest": {
            "arquivo_origem": rel,
            "dominio": domain,
            "assunto": item.get("assunto", ""),
            "ano_detectado_nome": item.get("ano_detectado_nome", ""),
            "tamanho_gb": item.get("tamanho_gb", 0),
            "shard_id": base_shard,
            "linhas_lidas": read_rows,
            "linhas_gravadas": written_rows,
            "schema_id": ",".join(sorted(schemas.keys())[:10]),
            "processamento": "streaming_lotes_paralelos",
            "lotes": submitted_batches,
            "workers_internos": workers,
            "duracao_segundos": round(duration, 3),
            "mb_por_segundo": round(size_mb / max(0.001, duration), 3),
            "erros_lotes": " | ".join(errors),
        },
        "schemas": schemas,
        "linhas_por_tabela": dict(rows_written),
        "linhas_bronze": int(bronze_rows_written),
    }


def build_batch_item(item: dict[str, Any], rows: list[dict[str, Any]], base_shard: str, batch_index: int) -> dict[str, Any]:
    chunk = dict(item)
    chunk["records"] = rows
    chunk["batch_index"] = int(batch_index)
    chunk["shard_id"] = f"{base_shard}_b{batch_index:06d}"
    return chunk


def process_record_batch_item(item: dict[str, Any], cfg: CleanDatabaseConfig) -> dict[str, Any]:
    local_schema_rows: dict[str, dict[str, Any]] = {}
    rel = str(item["relativo"])
    domain = str(item["dominio"])
    shard_id = str(item.get("shard_id") or shard_id_for_file(rel))
    table = domain_table(domain)
    remove_existing_shard_outputs(cfg, shard_id, domain, table)
    writer = PartitionedParquetWriter(cfg.out / "prata", chunk_rows=cfg.chunk_rows, shard_id=shard_id)
    bronze_writer = BronzeParquetWriter(cfg.out / "bronze", chunk_rows=cfg.chunk_rows, shard_id=shard_id)
    try:
        write_pipeline_event(cfg.out, "worker_lote", "inicio", arquivo=rel, lote=item.get("batch_index", 0), shard=shard_id, dominio=domain)
        counters = process_records_iterable(
            item.get("records") or [],
            rel,
            domain,
            writer,
            bronze_writer,
            local_schema_rows,
            progress_label=f"{rel} lote {int(item.get('batch_index', 0)) + 1}",
        )
        writer.flush_all()
        bronze_writer.flush_all()
        write_pipeline_event(
            cfg.out,
            "worker_lote",
            "fim",
            arquivo=rel,
            lote=item.get("batch_index", 0),
            shard=shard_id,
            linhas_lidas=counters.get("linhas_lidas", 0),
            linhas_gravadas=counters.get("linhas_gravadas", 0),
        )
        return {
            "manifest": {
                "arquivo_origem": rel,
                "dominio": domain,
                "assunto": item.get("assunto", ""),
                "ano_detectado_nome": item.get("ano_detectado_nome", ""),
                "tamanho_gb": item.get("tamanho_gb", 0),
                "shard_id": shard_id,
                "batch_index": item.get("batch_index", 0),
                **counters,
            },
            "schemas": local_schema_rows,
            "linhas_por_tabela": dict(writer.rows_written),
            "linhas_bronze": int(bronze_writer.rows_written),
        }
    finally:
        clean_memory()


def process_json_file_item(item: dict[str, Any], cfg: CleanDatabaseConfig) -> dict[str, Any]:
    started = time.perf_counter()
    local_schema_rows: dict[str, dict[str, Any]] = {}
    path = Path(str(item["path"]))
    rel = str(item["relativo"])
    domain = str(item["dominio"])
    shard_id = str(item.get("shard_id") or shard_id_for_file(rel))
    table = domain_table(domain)
    remove_existing_shard_outputs(cfg, shard_id, domain, table)
    writer = PartitionedParquetWriter(cfg.out / "prata", chunk_rows=cfg.chunk_rows, shard_id=shard_id)
    bronze_writer = BronzeParquetWriter(cfg.out / "bronze", chunk_rows=cfg.chunk_rows, shard_id=shard_id)
    try:
        logging.info("Worker iniciou documento: %s", describe_work_item(item))
        write_pipeline_event(cfg.out, "worker_documento", "inicio", arquivo=rel, dominio=domain, shard=shard_id, tamanho_gb=item.get("tamanho_gb", 0))
        counters = process_json_file(path, rel, domain, writer, bronze_writer, local_schema_rows)
        writer.flush_all()
        bronze_writer.flush_all()
        duration = time.perf_counter() - started
        size_mb = float(item.get("tamanho_gb", 0) or 0) * 1024
        manifest = {
            "arquivo_origem": rel,
            "dominio": domain,
            "assunto": item.get("assunto", ""),
            "ano_detectado_nome": item.get("ano_detectado_nome", ""),
            "tamanho_gb": item.get("tamanho_gb", 0),
            "shard_id": shard_id,
            "duracao_segundos": round(duration, 3),
            "mb_por_segundo": round(size_mb / max(0.001, duration), 3),
            **counters,
        }
        write_pipeline_event(
            cfg.out,
            "worker_documento",
            "fim",
            arquivo=rel,
            dominio=domain,
            shard=shard_id,
            linhas_lidas=manifest.get("linhas_lidas", 0),
            linhas_gravadas=manifest.get("linhas_gravadas", 0),
            duracao_segundos=manifest.get("duracao_segundos", 0),
        )
        return {
            "manifest": manifest,
            "schemas": local_schema_rows,
            "linhas_por_tabela": dict(writer.rows_written),
            "linhas_bronze": int(bronze_writer.rows_written),
        }
    finally:
        clean_memory()


def process_jsonl_range_item(item: dict[str, Any], cfg: CleanDatabaseConfig) -> dict[str, Any]:
    local_schema_rows: dict[str, dict[str, Any]] = {}
    path = Path(str(item["path"]))
    rel = str(item["relativo"])
    domain = str(item["dominio"])
    shard_id = str(item.get("shard_id") or shard_id_for_file(rel))
    table = domain_table(domain)
    remove_existing_shard_outputs(cfg, shard_id, domain, table)
    writer = PartitionedParquetWriter(cfg.out / "prata", chunk_rows=cfg.chunk_rows, shard_id=shard_id)
    bronze_writer = BronzeParquetWriter(cfg.out / "bronze", chunk_rows=cfg.chunk_rows, shard_id=shard_id)
    try:
        write_pipeline_event(
            cfg.out,
            "worker_faixa",
            "inicio",
            arquivo=rel,
            dominio=domain,
            shard=shard_id,
            range_index=item.get("range_index", 0),
            range_total=item.get("range_total", 0),
            range_start=item.get("range_start", 0),
            range_end=item.get("range_end", 0),
        )
        counters = process_jsonl_range(
            path,
            rel,
            domain,
            int(item.get("range_start", 0)),
            int(item.get("range_end", 0)),
            writer,
            bronze_writer,
            local_schema_rows,
        )
        writer.flush_all()
        bronze_writer.flush_all()
        manifest = {
            "arquivo_origem": rel,
            "dominio": domain,
            "assunto": item.get("assunto", ""),
            "ano_detectado_nome": item.get("ano_detectado_nome", ""),
            "tamanho_gb": item.get("tamanho_gb", 0),
            "shard_id": shard_id,
            "range_start": item.get("range_start", 0),
            "range_end": item.get("range_end", 0),
            "range_index": item.get("range_index", 0),
            **counters,
        }
        write_pipeline_event(
            cfg.out,
            "worker_faixa",
            "fim",
            arquivo=rel,
            dominio=domain,
            shard=shard_id,
            range_index=item.get("range_index", 0),
            range_total=item.get("range_total", 0),
            linhas_lidas=manifest.get("linhas_lidas", 0),
            linhas_gravadas=manifest.get("linhas_gravadas", 0),
        )
        return {
            "manifest": manifest,
            "schemas": local_schema_rows,
            "linhas_por_tabela": dict(writer.rows_written),
            "linhas_bronze": int(bronze_writer.rows_written),
        }
    finally:
        clean_memory()


def setup_clean_database_logging(out: Path, level: str) -> None:
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / "build_banco_eleitoral.log"
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


def logs_dir_from_base(base: Path) -> Path:
    base = Path(base)
    if base.name == "logs":
        return base
    if base.parent.name == "logs":
        return base.parent
    if base.name in {"bronze", "prata", "ouro", "metadados"}:
        return base.parent / "logs"
    return base / "logs"


def write_pipeline_event(base: Path, categoria: str, evento: str, **data: Any) -> None:
    try:
        logs = logs_dir_from_base(Path(base))
        logs.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "epoch": round(time.time(), 3),
            "pid": os.getpid(),
            "categoria": categoria,
            "evento": evento,
            **data,
        }
        line = json.dumps(payload, ensure_ascii=False, default=str)
        with (logs / "eventos_pipeline.jsonl").open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        save_json(payload, logs / "evento_atual.json")
    except Exception:
        return


def write_ouro_event(progress_dir: Path, label: str, evento: str, **data: Any) -> None:
    payload = {
        "label": label,
        "evento": evento,
        **data,
    }
    try:
        save_json(
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "epoch": round(time.time(), 3),
                "pid": os.getpid(),
                **payload,
            },
            progress_dir / f"{safe_name(label, 80)}_evento_atual.json",
        )
    except Exception:
        pass
    write_pipeline_event(progress_dir.parent, f"ouro:{label}", evento, **data)


def prepare_output_dir(cfg: CleanDatabaseConfig) -> None:
    if cfg.out.exists() and any(cfg.out.iterdir()):
        marker = cfg.out / DATABASE_MARKER_NAME
        legacy_marker = cfg.out / LEGACY_DATABASE_MARKER_NAME
        partial = cfg.out / "logs" / "manifesto_arquivos_parcial.json"
        complete_manifest = cfg.out / "metadados" / "manifesto_arquivos.parquet"
        if cfg.resume and (marker.exists() or legacy_marker.exists() or partial.exists() or complete_manifest.exists()):
            cfg.out.mkdir(parents=True, exist_ok=True)
            return
        if not cfg.overwrite:
            raise FileExistsError(
                f"A pasta de saida ja existe e nao esta vazia: {cfg.out}. "
                "Use --banco-overwrite para recriar do zero ou --resume para continuar."
            )
        if not marker.exists() and not legacy_marker.exists():
            raise RuntimeError(
                f"Recusei apagar {cfg.out} porque ela nao parece uma base criada por este script."
            )
        shutil.rmtree(cfg.out)
    cfg.out.mkdir(parents=True, exist_ok=True)


def process_json_file(
    path: Path,
    rel: str,
    domain: str,
    writer: PartitionedParquetWriter,
    bronze_writer: BronzeParquetWriter,
    schema_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return process_records_iterable(
        iter_json_records(path),
        rel,
        domain,
        writer,
        bronze_writer,
        schema_rows,
        progress_label=rel,
    )


def process_records_iterable(
    records: Iterable[dict[str, Any]],
    rel: str,
    domain: str,
    writer: PartitionedParquetWriter,
    bronze_writer: BronzeParquetWriter,
    schema_rows: dict[str, dict[str, Any]],
    progress_label: str = "",
) -> dict[str, Any]:
    role_cache: dict[str, str] = {}
    year = year_from_path(rel)
    schema_id = ""
    read_rows = 0
    written_rows = 0
    started = time.perf_counter()
    table = domain_table(domain)
    if table is None:
        return {"linhas_lidas": 0, "linhas_gravadas": 0, "schema_id": ""}
    event_base = writer.root.parent
    write_pipeline_event(event_base, "leitura_registros", "inicio", arquivo=rel, dominio=domain, tabela=table, progress_label=progress_label)

    for rec in records:
        read_rows += 1
        if not schema_id:
            schema_id = schema_hash(rec.keys(), domain)
            schema_rows[schema_id] = {
                "schema_id": schema_id,
                "dominio": domain,
                "qtd_campos": len(rec.keys()),
                "campos": ", ".join(sorted(map(str, rec.keys()))),
                "primeiro_arquivo": rel,
            }

        if domain == "eleitorado":
            row = electorate_row(rec, role_cache, year, rel, schema_id)
        elif domain == "candidatos":
            row = candidate_row(rec, role_cache, year, rel, schema_id)
        else:
            row = result_row(rec, role_cache, year, rel, schema_id)

        if row:
            bronze_writer.add(domain, schema_id, row.get("uf", ""), bronze_row(rec, row, domain, year, rel, schema_id))
            writer.add(table, row)
            written_rows += 1

        if progress_label and read_rows % FILE_PROGRESS_EVERY_ROWS == 0:
            elapsed = max(0.001, time.perf_counter() - started)
            logging.info(
                "Progresso %s | lidas=%s gravadas=%s velocidade=%.0f linhas/s",
                progress_label,
                read_rows,
                written_rows,
                read_rows / elapsed,
            )
            write_pipeline_event(
                event_base,
                "leitura_registros",
                "progresso",
                arquivo=rel,
                dominio=domain,
                tabela=table,
                progress_label=progress_label,
                linhas_lidas=read_rows,
                linhas_gravadas=written_rows,
                linhas_por_segundo=round(read_rows / elapsed, 3),
            )

    write_pipeline_event(
        event_base,
        "leitura_registros",
        "fim",
        arquivo=rel,
        dominio=domain,
        tabela=table,
        progress_label=progress_label,
        linhas_lidas=read_rows,
        linhas_gravadas=written_rows,
        duracao_segundos=round(time.perf_counter() - started, 3),
    )
    return {"linhas_lidas": read_rows, "linhas_gravadas": written_rows, "schema_id": schema_id}


def process_jsonl_range(
    path: Path,
    rel: str,
    domain: str,
    start: int,
    end: int,
    writer: PartitionedParquetWriter,
    bronze_writer: BronzeParquetWriter,
    schema_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return process_records_iterable(
        iter_jsonl_records_byte_range(path, start, end),
        rel,
        domain,
        writer,
        bronze_writer,
        schema_rows,
        progress_label=f"{rel} bytes {start}-{end}",
    )


def iter_jsonl_records_byte_range(path: Path, start: int, end: int) -> Iterable[dict[str, Any]]:
    with open(path, "rb") as f:
        f.seek(max(0, int(start)))
        if start > 0:
            f.readline(MAX_JSONL_LINE_CHARS + 1)

        while True:
            pos = f.tell()
            if end > 0 and pos >= end:
                break
            line = f.readline(MAX_JSONL_LINE_CHARS + 1)
            if not line:
                break
            if len(line) > MAX_JSONL_LINE_CHARS and not line.endswith(b"\n"):
                logging.warning(
                    "Linha JSONL maior que %s MiB em %s entre bytes %s-%s. Pulando linha longa.",
                    MAX_JSONL_LINE_CHARS // (1024 * 1024),
                    path,
                    start,
                    end,
                )
                continue
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line.decode("utf-8", errors="replace"))
            except Exception:
                continue
            if isinstance(obj, dict):
                yield normalize_record(obj)


def bronze_row(record: dict[str, Any], silver_row: dict[str, Any], domain: str, year: str, rel: str, schema_id: str) -> dict[str, Any]:
    row = {str(k): v for k, v in record.items()}
    row["_dominio_documento"] = domain
    row["_schema_id"] = schema_id
    row["_ano_arquivo"] = year
    row["_uf_particao"] = silver_row.get("uf", "")
    row["_arquivo_origem"] = rel
    return row


def electorate_row(record: dict[str, Any], role_cache: dict[str, str], year: str, rel: str, schema_id: str) -> dict[str, Any]:
    gold = record_to_gold_cached(record, role_cache)
    return {
        **base_key_values(gold, year),
        "perfil_faixa_etaria": meaningful_profile(gold.get("perfil_faixa_etaria", "")),
        "perfil_genero": meaningful_profile(gold.get("perfil_genero", "")),
        "perfil_instrucao": meaningful_profile(gold.get("perfil_instrucao", "")),
        "perfil_estado_civil": meaningful_profile(gold.get("perfil_estado_civil", "")),
        "perfil_raca_cor": meaningful_profile(gold.get("perfil_raca_cor", "")),
        "eleitorado": metric_value(gold.get("eleitorado")),
        "comparecimento_estimado": metric_value(gold.get("comparecimento")),
        "abstencao_estimado": metric_value(gold.get("abstencao")),
        "brancos": metric_value(gold.get("brancos")),
        "nulos": metric_value(gold.get("nulos")),
        "validos_estimados": metric_value(gold.get("validos")),
        "qtd_registros": 1,
        "schema_id": schema_id,
        "arquivo_origem": rel,
    }


def candidate_row(record: dict[str, Any], role_cache: dict[str, str], year: str, rel: str, schema_id: str) -> dict[str, Any]:
    gold = record_to_gold_cached(record, role_cache)
    age_value = direct_value(record, ["IDADE", "IDADE_DATA_POSSE", "NR_IDADE_DATA_POSSE", "FAIXA_ETARIA", "DS_FAIXA_ETARIA"])
    return {
        **base_key_values(gold, year),
        "partido": clean_value(gold.get("partido", "")),
        "candidato": candidate_name(record, gold),
        "nr_candidato": compact_code(direct_value(record, ["NR_CANDIDATO", "NUMERO_CANDIDATO", "NR_VOTAVEL"])),
        "sq_candidato": compact_code(direct_value(record, ["SQ_CANDIDATO", "CD_CANDIDATO", "ID_CANDIDATO"])),
        "perfil_faixa_etaria": age_band(age_value) or meaningful_profile(gold.get("perfil_faixa_etaria", "")),
        "perfil_genero": meaningful_profile(gold.get("perfil_genero", "")),
        "perfil_instrucao": meaningful_profile(gold.get("perfil_instrucao", "")),
        "perfil_estado_civil": meaningful_profile(gold.get("perfil_estado_civil", "")),
        "perfil_raca_cor": meaningful_profile(gold.get("perfil_raca_cor", "")),
        "situacao_candidatura": clean_value(direct_value(record, ["DS_SITUACAO_CANDIDATURA", "DS_SIT_TOT_TURNO", "SITUACAO_CANDIDATURA"])),
        "resultado_candidatura": clean_value(direct_value(record, ["DS_SIT_TOT_TURNO", "DS_RESULTADO", "RESULTADO"])),
        "schema_id": schema_id,
        "arquivo_origem": rel,
    }


def result_row(record: dict[str, Any], role_cache: dict[str, str], year: str, rel: str, schema_id: str) -> dict[str, Any]:
    gold = record_to_gold_cached(record, role_cache)
    votes = metric_value(gold.get("votos"))
    if votes <= 0:
        votes = metric_value(direct_value(record, ["QT_VOTOS", "QTD_VOTOS", "VOTOS", "QT_VOTOS_NOMINAIS"]))
    return {
        **base_key_values(gold, year),
        "partido": clean_value(gold.get("partido", "")),
        "candidato": candidate_name(record, gold),
        "nr_votavel": compact_code(direct_value(record, ["NR_VOTAVEL", "NR_CANDIDATO", "NUMERO_CANDIDATO"])),
        "sq_candidato": compact_code(direct_value(record, ["SQ_CANDIDATO", "CD_CANDIDATO", "ID_CANDIDATO"])),
        "votos": votes,
        "brancos": metric_value(gold.get("brancos")),
        "nulos": metric_value(gold.get("nulos")),
        "validos_estimados": metric_value(gold.get("validos")),
        "qtd_registros": 1,
        "schema_id": schema_id,
        "arquivo_origem": rel,
    }


def base_key_values(gold: dict[str, Any], year: str) -> dict[str, str]:
    detected_year = clean_value(gold.get("ano", ""))
    return {
        "ano": year or detected_year,
        "uf": clean_uf(gold.get("uf", "")),
        "cd_municipio": compact_code(gold.get("cd_municipio", "")),
        "nm_municipio": clean_value(gold.get("nm_municipio", "")),
        "zona": compact_code(gold.get("zona", "")),
        "secao": compact_code(gold.get("secao", "")),
        "local_votacao": clean_value(gold.get("local_votacao", "")),
        "bairro": clean_value(gold.get("bairro", "")),
        "cargo": clean_value(gold.get("cargo", "")),
        "turno": clean_value(gold.get("turno", "")),
    }


def domain_table(domain: str) -> str | None:
    if domain == "eleitorado":
        return "eleitorado"
    if domain == "candidatos":
        return "candidatos"
    if domain == "resultados":
        return "resultados_votos"
    return None


def normalize_table_frame(table: str, df: pd.DataFrame) -> pd.DataFrame:
    cols = TABLE_COLUMNS[table]
    for col in cols:
        if col not in df.columns:
            df[col] = 0.0 if col in NUMERIC_COLS.get(table, set()) else ""
    df = df[cols].copy()
    for col in NUMERIC_COLS.get(table, set()):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    for col in [c for c in cols if c not in NUMERIC_COLS.get(table, set())]:
        df[col] = df[col].map(clean_value)
    df["uf"] = df["uf"].map(clean_uf)
    return reduce_chunk(table, df)


def reduce_chunk(table: str, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    numeric = list(NUMERIC_COLS.get(table, set()))
    if table == "candidatos":
        return df.drop_duplicates()
    group_cols = [c for c in df.columns if c not in numeric]
    if not numeric:
        return df.drop_duplicates()
    return df.groupby(group_cols, dropna=False, as_index=False)[numeric].sum()


def run_clean_database_analyses(cfg: CleanDatabaseConfig) -> dict[str, Any]:
    ensure_parquet_engine()
    write_pipeline_event(cfg.out, "analise_banco", "inicio", banco=str(cfg.out))
    try:
        import duckdb
    except Exception as exc:
        logging.warning("DuckDB indisponivel; analises derivadas nao foram geradas: %s", exc)
        write_pipeline_event(cfg.out, "analise_banco", "erro_duckdb_indisponivel", erro=str(exc))
        return {"status": "duckdb_indisponivel", "erro": str(exc)}

    analyses = cfg.out / "ouro"
    analyses.mkdir(parents=True, exist_ok=True)

    prata = cfg.out / "prata"
    e = dataset_expr(prata / "eleitorado")
    r = dataset_expr(prata / "resultados_votos")
    c = dataset_expr(prata / "candidatos")
    outputs: dict[str, str] = {}

    if parquet_dataset_exists(prata / "eleitorado") and parquet_dataset_exists(prata / "resultados_votos"):
        uf_parts = [(uf, e_path) for uf, e_path in list_uf_partition_dirs(prata / "eleitorado") if parquet_dataset_exists(prata / "resultados_votos" / f"uf={uf}")]
        if cfg.uf_filter:
            allowed_ufs = {safe_text(uf, "").upper() for uf in cfg.uf_filter if safe_text(uf, "")}
            before_count = len(uf_parts)
            uf_parts = [(uf, e_path) for uf, e_path in uf_parts if safe_text(uf, "").upper() in allowed_ufs]
            logging.info("Filtro de UFs ativo: %s de %s UFs serao processadas (%s).", len(uf_parts), before_count, ", ".join(sorted(allowed_ufs)))
            write_pipeline_event(cfg.out, "analise_banco", "filtro_ufs", ufs=sorted(allowed_ufs), selecionadas=len(uf_parts), total_original=before_count)
        logging.info(
            "Etapa 1/3: organizando prata_minima direto da prata por UF, sem planejamento de anos.",
        )
        write_pipeline_event(
            cfg.out,
            "analise_banco",
            "inicio_etapa_prata_minima",
            etapa="prata_minima",
            ordem="1/3",
            ufs=len(uf_parts),
        )
        plan = build_correlacao_uf_year_plan(cfg, prata, uf_parts)
        outputs.update(run_prata_minima_correlacoes_stage(cfg, prata, analyses, plan=plan, uf_parts=uf_parts))
        if cfg.only_states_brasil:
            logging.info("Etapa 2/3: modo curto ativo; gerando apenas estadual -> Brasil a partir da prata_minima.")
            write_pipeline_event(cfg.out, "analise_banco", "chamando_etapa", etapa="ouro_estados_brasil")
            outputs.update(run_ouro_estados_brasil_analyses(cfg, analyses, plan))
        else:
            logging.info("Etapa 2/3: iniciando ouro nivelado municipal -> estadual -> Brasil a partir da prata_minima organizada.")
            write_pipeline_event(cfg.out, "analise_banco", "chamando_etapa", etapa="ouro_nivelado")
            outputs.update(run_ouro_nivelado_analyses(cfg, analyses, plan))

    if parquet_dataset_exists(prata / "candidatos"):
        write_pipeline_event(cfg.out, "analise_banco", "chamando_etapa", etapa="ouro_candidatos")
        outputs.update(run_candidatos_ouro_partitioned(cfg, prata, analyses))

    write_pipeline_event(cfg.out, "analise_banco", "fim", outputs=outputs)
    return {
        "status": "ok",
        "outputs": outputs,
        "paralelismo": {
            "ouro_workers": int(cfg.ouro_workers),
            "duckdb_threads": int(cfg.duckdb_threads),
        },
    }


def run_eleitorado_ouro_partitioned(cfg: CleanDatabaseConfig, prata: Path, analyses: Path) -> dict[str, str]:
    label = "ouro_eleitorado"
    progress_dir = prepare_ouro_progress(cfg, label)
    write_ouro_event(progress_dir, label, "inicio_etapa", etapa=label)
    if cfg.resume and ouro_eleitorado_stage_ready(analyses):
        logging.info("Pulando etapa %s inteira: artefatos principais ja existem.", label)
        outputs = {
            "timeline_nacional": str(analyses / "timeline_nacional.parquet"),
            "timeline_uf": str(analyses / "timeline_uf"),
            "timeline_municipal": str(analyses / "timeline_municipal"),
            "retrato_municipal": str(analyses / "retrato_municipal"),
            "perfil_eleitor_por_ano": str(analyses / "perfil_eleitor_por_ano"),
            "top10_perfis_parts": str(analyses / "_work" / "top10_perfis_parts"),
        }
        save_json(
            {
                "label": label,
                "status": "ok",
                "resume_skip_etapa": True,
                "motivo": "Artefatos principais de eleitorado ja existem; pulando varredura de fatias.",
                "outputs": outputs,
            },
            progress_dir / f"{label}_progresso.json",
        )
        return outputs
    targets = [
        analyses / "timeline_nacional.parquet",
        analyses / "timeline_uf",
        analyses / "timeline_municipal",
        analyses / "retrato_municipal",
        analyses / "perfil_eleitor_por_ano",
        analyses / "top10_perfis_federacao_estado_municipio",
        analyses / "_work" / "perfil_eleitor_por_ano_parts",
        analyses / "_work" / "top10_perfis_parts",
    ]
    reset_ouro_targets(targets, resume=cfg.resume)

    outputs: dict[str, str] = {}
    uf_parts = list_uf_partition_dirs(prata / "eleitorado")
    logging.info("Gerando %s por UF: %s particoes de eleitorado.", label, len(uf_parts))
    save_json({"label": label, "status": "processando", "ufs_total": len(uf_parts)}, progress_dir / f"{label}_progresso.json")

    for index, (uf, e_path) in enumerate(uf_parts, start=1):
        write_ouro_event(progress_dir, label, "processando_uf", etapa=label, indice_uf=index, total_ufs=len(uf_parts), uf=uf, entrada=str(e_path))
        years = list_years_for_dataset(e_path, cfg)
        logging.info("Ouro eleitorado UF %s/%s: %s | anos=%s", index, len(uf_parts), uf, ", ".join(years) or "sem_ano")
        for year_index, year in enumerate(years, start=1):
            slice_key = slice_name(uf, year)
            logging.info("Ouro eleitorado fatia %s/%s UF %s ano %s", year_index, len(years), uf, year)
            write_ouro_event(
                progress_dir,
                label,
                "processando_fatia",
                etapa=label,
                uf=uf,
                ano=year,
                indice_uf=index,
                total_ufs=len(uf_parts),
                indice_ano=year_index,
                total_anos=len(years),
            )
            save_json(
                {
                    "label": label,
                    "status": "processando",
                    "uf_atual": uf,
                    "ano_atual": year,
                    "indice_uf": index,
                    "total_ufs": len(uf_parts),
                    "indice_ano": year_index,
                    "total_anos_uf": len(years),
                },
                progress_dir / f"{label}_progresso.json",
            )
            e_slice = filtered_year_expr(dataset_expr(e_path), year)
            r_path = prata / "resultados_votos" / f"uf={uf}"
            r_slice = filtered_year_expr(dataset_expr(r_path), year) if parquet_dataset_exists(r_path) else ""
            tasks = [
                copy_task(f"timeline_uf_{slice_key}", timeline_sql(e_slice, "uf", "timeline_uf"), chunk_output(analyses / "timeline_uf", slice_key), partition_by=YEAR_UF_PARTITION_COLS),
                copy_task(f"timeline_municipal_{slice_key}", timeline_sql(e_slice, "uf, cd_municipio, nm_municipio", "timeline_municipal"), chunk_output(analyses / "timeline_municipal", slice_key), partition_by=MUNICIPIO_PARTITION_COLS),
                copy_task(f"retrato_municipal_{slice_key}", retrato_municipal_sql(e_slice, r_slice), chunk_output(analyses / "retrato_municipal", slice_key), partition_by=MUNICIPIO_PARTITION_COLS),
                copy_task(f"perfil_eleitor_por_ano_parts_{slice_key}", perfil_eleitor_por_ano_parts_sql(e_slice), chunk_output(analyses / "_work" / "perfil_eleitor_por_ano_parts", slice_key), partition_by=["ano"]),
                copy_task(f"top10_perfis_parts_{slice_key}", top10_perfis_parts_sql(e_slice), chunk_output(analyses / "_work" / "top10_perfis_parts", slice_key), partition_by=MUNICIPIO_PARTITION_COLS),
            ]
            outputs.update(execute_copy_tasks(tasks, cfg, label))
            clean_memory()

    final_tasks = []
    if parquet_dataset_exists(analyses / "timeline_uf"):
        timeline_uf_expr = dataset_expr(analyses / "timeline_uf")
        final_tasks.append(copy_task("timeline_nacional", timeline_nacional_from_timeline_uf_sql(timeline_uf_expr), analyses / "timeline_nacional.parquet"))
    if parquet_dataset_exists(analyses / "_work" / "perfil_eleitor_por_ano_parts"):
        perfil_parts_expr = dataset_expr(analyses / "_work" / "perfil_eleitor_por_ano_parts")
        final_tasks.append(copy_task("perfil_eleitor_por_ano", perfil_eleitor_por_ano_final_sql(perfil_parts_expr), analyses / "perfil_eleitor_por_ano", partition_by=["ano"]))
    if parquet_dataset_exists(analyses / "_work" / "top10_perfis_parts"):
        top10_parts_expr = dataset_expr(analyses / "_work" / "top10_perfis_parts")
        final_tasks.append(copy_task("top10_perfis", top10_perfis_from_parts_sql(top10_parts_expr), analyses / "top10_perfis_federacao_estado_municipio", partition_by=ENTITY_PARTITION_COLS))
    final_cfg = replace(cfg, resume=False)
    for task in final_tasks:
        outputs[task["name"]] = execute_ouro_task(task, final_cfg, label, progress_dir)
        clean_memory()

    save_json({"label": label, "status": "ok", "outputs": outputs}, progress_dir / f"{label}_progresso.json")
    write_ouro_event(progress_dir, label, "fim_etapa", etapa=label, outputs=outputs)
    return outputs


def ouro_eleitorado_stage_ready(analyses: Path) -> bool:
    required = [
        analyses / "timeline_nacional.parquet",
        analyses / "timeline_uf",
        analyses / "timeline_municipal",
        analyses / "retrato_municipal",
        analyses / "perfil_eleitor_por_ano",
        analyses / "_work" / "perfil_eleitor_por_ano_parts",
        analyses / "_work" / "top10_perfis_parts",
    ]
    return all(path.exists() for path in required)


def run_resultados_ouro_partitioned(cfg: CleanDatabaseConfig, prata: Path, analyses: Path) -> dict[str, str]:
    label = "ouro_resultados"
    progress_dir = prepare_ouro_progress(cfg, label)
    write_ouro_event(progress_dir, label, "inicio_etapa", etapa=label)
    reset_ouro_targets([analyses / "resultados_vencedores_secao"], resume=cfg.resume)
    outputs: dict[str, str] = {}
    plan = load_or_build_resultados_plan(cfg, prata, progress_dir)
    pending_plan = report_resultados_plan_status(plan, progress_dir, analyses / "resultados_vencedores_secao")
    total_ufs = len({safe_text(item.get("uf", "")) for item in plan})
    done_count = len(plan) - len(pending_plan)
    logging.info(
        "Manifesto %s lido: %s fatias UF/ano em %s UFs | concluidas=%s | pendentes=%s.",
        label,
        len(plan),
        total_ufs,
        done_count,
        len(pending_plan),
    )
    if pending_plan:
        preview = ", ".join(str(item.get("tarefa", "")) for item in pending_plan[:10])
        suffix = " ..." if len(pending_plan) > 10 else ""
        logging.info("Pendencias %s: %s%s", label, preview, suffix)
    else:
        logging.info("Nenhuma pendencia em %s; seguindo para a proxima etapa.", label)
        return outputs
    pending_years_by_uf: dict[str, list[str]] = defaultdict(list)
    for pending_item in pending_plan:
        pending_uf = safe_text(pending_item.get("uf", "")) or "SEM_UF"
        pending_year = safe_text(pending_item.get("ano", ""))
        if pending_year and pending_year not in pending_years_by_uf[pending_uf]:
            pending_years_by_uf[pending_uf].append(pending_year)
    for index, item in enumerate(pending_plan, start=1):
        uf = safe_text(item.get("uf", "")) or "SEM_UF"
        year = safe_text(item.get("ano", ""))
        r_path = prata / "resultados_votos" / f"uf={uf}"
        if not parquet_dataset_exists(r_path):
            r_path = Path(safe_text(item.get("path", "")))
        size_gb = float(item.get("tamanho_gb", 0) or 0)
        slice_key = slice_name(uf, year)
        out_root = chunk_output(analyses / "resultados_vencedores_secao", slice_key)
        task_root_name = f"resultados_vencedores_secao_{slice_key}"
        if cfg.resume and resultado_slice_is_done(progress_dir, out_root, task_root_name):
            logging.info("Pulando fatia resultado ja concluida %s/%s: %s", index, len(plan), task_root_name)
            outputs[task_root_name] = str(out_root)
            continue
        logging.info(
            "Ouro resultados fatia %s/%s: %s ano=%s | tamanho_uf=%.2f GB | modo=fatiado_por_municipio",
            index,
            len(pending_plan),
            uf,
            year or "sem_ano",
            size_gb,
        )
        write_ouro_event(
            progress_dir,
            label,
            "processando_fatia",
            indice_fatia=index,
            total_fatias=len(pending_plan),
            uf=uf,
            ano=year,
            tarefa=task_root_name,
            saida=str(out_root),
        )
        outputs.update(run_resultados_vencedores_fatiado(cfg, progress_dir, r_path, year, out_root, slice_key, pending_years_by_uf.get(uf, [year])))
        report_resultados_plan_status(plan, progress_dir, analyses / "resultados_vencedores_secao")
        clean_memory()
    save_json({"label": label, "status": "ok", "outputs": outputs}, progress_dir / f"{label}_progresso.json")
    write_ouro_event(progress_dir, label, "fim_etapa", etapa=label, outputs=outputs)
    return outputs


def load_or_build_resultados_plan(cfg: CleanDatabaseConfig, prata: Path, progress_dir: Path) -> list[dict[str, Any]]:
    plan_path = progress_dir / "ouro_resultados_plano.json"
    manifesto_path = progress_dir / "ouro_resultados_manifesto.json"
    if cfg.resume and plan_path.exists():
        try:
            with plan_path.open("r", encoding="utf-8-sig") as f:
                plan = json.load(f)
            if isinstance(plan, list) and plan:
                logging.info("Manifesto ouro_resultados carregado: %s fatias UF/ano.", len(plan))
                ordered_plan = sort_uf_year_plan(plan)
                if ordered_plan != plan:
                    logging.info("Manifesto ouro_resultados reordenado por UF/ano para processamento alfabetico.")
                    save_json(ordered_plan, plan_path)
                    save_json(ordered_plan, manifesto_path)
                return ordered_plan
        except Exception as exc:
            logging.warning("Nao foi possivel ler manifesto ouro_resultados; recriando. Erro: %s", exc)

    plan: list[dict[str, Any]] = []
    uf_parts = list_uf_partition_dirs(prata / "resultados_votos")
    logging.info("Criando plano ouro_resultados: %s particoes de UF.", len(uf_parts))
    for index, (uf, r_path) in enumerate(uf_parts, start=1):
        years = list_years_for_dataset(r_path, cfg)
        size_gb = parquet_dataset_size_gb(r_path)
        logging.info(
            "Plano ouro_resultados UF %s/%s: %s | anos=%s | tamanho=%.2f GB",
            index,
            len(uf_parts),
            uf,
            ", ".join(years) or "sem_ano",
            size_gb,
        )
        for year in years:
            plan.append(
                {
                    "uf": uf,
                    "ano": year,
                    "path": str(r_path),
                    "tamanho_gb": round(size_gb, 6),
                    "tarefa": f"resultados_vencedores_secao_{slice_name(uf, year)}",
                }
            )
    plan = sort_uf_year_plan(plan)
    save_json(plan, plan_path)
    save_json(plan, manifesto_path)
    return plan


def report_resultados_plan_status(plan: list[dict[str, Any]], progress_dir: Path, out_base: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    done_count = 0
    for item in plan:
        uf = safe_text(item.get("uf", "")) or "SEM_UF"
        year = safe_text(item.get("ano", ""))
        slice_key = slice_name(uf, year)
        task_name = f"resultados_vencedores_secao_{slice_key}"
        out_root = chunk_output(out_base, slice_key)
        done = resultado_slice_is_done(progress_dir, out_root, task_name)
        error_marker = progress_dir / f"{safe_name(task_name, 80)}.error.json"
        status = "concluido" if done else ("erro" if error_marker.exists() else "pendente")
        row = {
            "uf": uf,
            "ano": year,
            "tarefa": task_name,
            "status": status,
            "saida": str(out_root),
        }
        rows.append(row)
        if done:
            done_count += 1
        else:
            pending.append(row)
    save_json(
        {
            "total": len(plan),
            "concluidas": done_count,
            "pendentes": len(plan) - done_count,
            "status": rows,
        },
        progress_dir / "ouro_resultados_status_fatias.json",
    )
    save_json(pending, progress_dir / "ouro_resultados_pendentes.json")
    return pending


def resultado_slice_is_done(progress_dir: Path, out_root: Path, task_root_name: str) -> bool:
    done_marker = progress_dir / f"{safe_name(task_root_name, 80)}_fatiado.done.json"
    direct_marker = progress_dir / f"{safe_name(task_root_name, 80)}.done.json"
    return out_root.exists() and (done_marker.exists() or direct_marker.exists())


def run_resultados_vencedores_fatiado(
    cfg: CleanDatabaseConfig,
    progress_dir: Path,
    r_path: Path,
    year: str,
    out_root: Path,
    slice_key: str,
    years_for_uf: list[str] | None = None,
) -> dict[str, str]:
    label = "ouro_resultados"
    task_root_name = f"resultados_vencedores_secao_{slice_key}"
    done_marker = progress_dir / f"{safe_name(task_root_name, 80)}_fatiado.done.json"
    if cfg.resume and resultado_slice_is_done(progress_dir, out_root, task_root_name):
        logging.info("Pulando vencedores fatiados ja concluidos [%s]: %s", label, slice_key)
        return {task_root_name: str(out_root)}

    remove_path_if_exists(out_root)
    base_expr = filtered_year_expr(dataset_expr(r_path), year)
    outputs: dict[str, str] = {}
    logging.info("Reprocessando %s em fatias para reduzir memoria.", task_root_name)
    write_ouro_event(
        progress_dir,
        label,
        "iniciando_fatiamento",
        tarefa=task_root_name,
        slice=slice_key,
        ano=year,
        entrada=str(r_path),
        saida=str(out_root),
    )
    bucket_root = materialize_resultados_hash_buckets(cfg, progress_dir, r_path, year, base_expr, slice_key, years_for_uf=years_for_uf)
    if bucket_root is not None and parquet_dataset_exists(bucket_root):
        process_resultados_bucketed(
            cfg,
            progress_dir,
            bucket_root,
            out_root,
            task_root_name,
            slice_key,
            outputs,
        )
    else:
        logging.warning("Buckets fisicos indisponiveis para %s; usando fatiamento SQL recursivo.", task_root_name)
        process_resultados_split_level(
            cfg,
            progress_dir,
            base_expr,
            out_root,
            task_root_name,
            slice_key,
            0,
            outputs,
        )
    has_errors = any(str(value).startswith("ERRO:") for value in outputs.values())
    has_data = out_root.exists()
    status = "ok" if has_data and not has_errors else "erro"
    marker_payload = {
        "label": label,
        "tarefa": task_root_name,
        "status": status,
        "saida": str(out_root),
        "outputs": outputs,
    }
    if status == "ok":
        remove_path_if_exists(progress_dir / f"{safe_name(task_root_name, 80)}.error.json")
        remove_path_if_exists(progress_dir / f"{safe_name(task_root_name, 80)}_fatiado.error.json")
        save_json(marker_payload, done_marker)
        write_ouro_event(progress_dir, label, "fatia_concluida", tarefa=task_root_name, saida=str(out_root))
        return outputs or {task_root_name: str(out_root)}

    save_json(marker_payload, progress_dir / f"{safe_name(task_root_name, 80)}_fatiado.error.json")
    save_json(
        marker_payload,
        progress_dir / f"{safe_name(task_root_name, 80)}.error.json",
    )
    write_ouro_event(progress_dir, label, "fatia_com_erro", tarefa=task_root_name, saida=str(out_root), outputs=outputs)
    return outputs or {task_root_name: "ERRO: fatiamento sem saida valida"}


def materialize_resultados_hash_buckets(
    cfg: CleanDatabaseConfig,
    progress_dir: Path,
    r_path: Path,
    year: str,
    base_expr: str,
    slice_key: str,
    years_for_uf: list[str] | None = None,
) -> Path | None:
    label = "ouro_resultados"
    try:
        uf_streamed = materialize_resultados_uf_year_buckets_streaming(
            cfg,
            progress_dir,
            r_path,
            years_for_uf or ([year] if year else []),
            slice_key,
        )
        year_root = resultados_uf_year_bucket_root(cfg, r_path, year)
        if uf_streamed is not None and parquet_dataset_exists(year_root):
            logging.info("Usando cache de buckets por UF/ano para %s: %s", slice_key, year_root)
            write_ouro_event(progress_dir, label, "buckets_uf_ano_reutilizados", slice=slice_key, ano=year, saida=str(year_root))
            return year_root
    except Exception as exc:
        logging.warning("Cache UF/ano/bucket falhou em %s; usando fluxo antigo por ano. Erro: %s", slice_key, exc)
        write_ouro_event(progress_dir, label, "buckets_uf_ano_fallback", slice=slice_key, erro=str(exc))

    bucket_root = chunk_output(cfg.out / "ouro" / "_work" / "resultados_buckets", slice_key)
    task_name = f"resultados_buckets_{slice_key}"
    if cfg.resume and parquet_dataset_exists(bucket_root):
        logging.info("Buckets de resultados ja existem para %s: %s. Reusando cestas fisicas.", slice_key, bucket_root)
        write_ouro_event(
            progress_dir,
            label,
            "buckets_reutilizados",
            tarefa=task_name,
            slice=slice_key,
            buckets=RESULTADOS_HASH_BUCKETS,
            saida=str(bucket_root),
        )
        return bucket_root

    try:
        streamed = materialize_resultados_hash_buckets_streaming(cfg, progress_dir, r_path, year, bucket_root, slice_key)
        if streamed is not None and parquet_dataset_exists(streamed):
            return streamed
    except Exception as exc:
        logging.warning("Streaming por buckets falhou em %s; usando DuckDB como fallback. Erro: %s", slice_key, exc)
        write_ouro_event(progress_dir, label, "buckets_streaming_fallback_duckdb", slice=slice_key, erro=str(exc))

    logging.info(
        "Materializando cestas de resultados para %s via DuckDB: %s buckets em uma leitura sequencial.",
        slice_key,
        RESULTADOS_HASH_BUCKETS,
    )
    write_ouro_event(
        progress_dir,
        label,
        "buckets_materializacao_inicio",
        tarefa=task_name,
        slice=slice_key,
        buckets=RESULTADOS_HASH_BUCKETS,
        saida=str(bucket_root),
    )
    sql = f"""
    select *,
           cast(hash({discrete_sql_value('cd_municipio')}) % {RESULTADOS_HASH_BUCKETS} as integer) as bucket_municipio
    from {base_expr}
    where {metric_sql('votos')} > 0
    """
    result = execute_ouro_task(
        copy_task(task_name, sql, bucket_root, partition_by=["bucket_municipio"]),
        cfg,
        label,
        progress_dir,
    )
    if str(result).startswith("ERRO:"):
        logging.error("Nao foi possivel materializar buckets de %s: %s", slice_key, result)
        write_ouro_event(progress_dir, label, "buckets_materializacao_erro", tarefa=task_name, slice=slice_key, erro=str(result))
        return None
    write_ouro_event(progress_dir, label, "buckets_materializacao_fim", tarefa=task_name, slice=slice_key, saida=str(bucket_root))
    return bucket_root


def resultados_uf_from_path(r_path: Path) -> str:
    name = Path(r_path).name
    if name.startswith("uf="):
        return safe_text(name.split("=", 1)[1], "SEM_UF") or "SEM_UF"
    return "SEM_UF"


def resultados_uf_bucket_root(cfg: CleanDatabaseConfig, r_path: Path) -> Path:
    uf = resultados_uf_from_path(r_path)
    return cfg.out / "ouro" / "_work" / "resultados_buckets_por_uf" / f"uf={safe_name(uf, 20) or 'SEM_UF'}"


def resultados_uf_year_bucket_root(cfg: CleanDatabaseConfig, r_path: Path, year: str) -> Path:
    year_key = safe_name(safe_text(year, "SEM_ANO") or "SEM_ANO", 20) or "SEM_ANO"
    return resultados_uf_bucket_root(cfg, r_path) / f"ano={year_key}"


def flush_resultados_bucket_buffer(
    buffers: dict[tuple[str, int], list[pd.DataFrame]],
    buffer_rows: dict[tuple[str, int], int],
    part_counter: dict[tuple[str, int], int],
    root: Path,
    key: tuple[str, int],
) -> int:
    frames = buffers.get(key) or []
    if not frames:
        return 0
    year_value, bucket_id = key
    out_dir = root / f"ano={safe_name(year_value, 20) or 'SEM_ANO'}" / f"bucket_municipio={bucket_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    part_id = part_counter[key]
    part_counter[key] += 1
    out_file = out_dir / f"part-{part_id:06d}.parquet"
    df_out = pd.concat(frames, ignore_index=True)
    df_out.to_parquet(out_file, index=False, compression="snappy")
    rows = int(len(df_out))
    buffers[key] = []
    buffer_rows[key] = 0
    return rows


def materialize_resultados_uf_year_buckets_streaming(
    cfg: CleanDatabaseConfig,
    progress_dir: Path,
    r_path: Path,
    years: list[str],
    slice_key: str,
) -> Path | None:
    import pyarrow.dataset as ds

    label = "ouro_resultados"
    if not parquet_dataset_exists(r_path):
        return None

    requested_years = sorted({safe_text(y) for y in years if safe_text(y)})
    if not requested_years:
        return None

    root = resultados_uf_bucket_root(cfg, r_path)
    missing_years = [y for y in requested_years if not parquet_dataset_exists(root / f"ano={safe_name(y, 20) or 'SEM_ANO'}")]
    if not missing_years:
        logging.info(
            "Cache UF/ano/bucket ja existe para %s | anos=%s | raiz=%s",
            resultados_uf_from_path(r_path),
            ", ".join(requested_years),
            root,
        )
        return root

    for missing in missing_years:
        remove_path_if_exists(root / f"ano={safe_name(missing, 20) or 'SEM_ANO'}")
    root.mkdir(parents=True, exist_ok=True)

    dataset = ds.dataset(str(r_path), format="parquet", partitioning="hive")
    available = set(dataset.schema.names)
    columns = [c for c in RESULTADOS_COLS if c in available]
    required = {"ano", "cd_municipio", "votos"}
    if not required.issubset(set(columns)):
        missing_cols = sorted(required - set(columns))
        raise RuntimeError(f"Colunas obrigatorias ausentes para cache UF/ano/bucket: {missing_cols}")

    batch_rows = max(10_000, min(max(10_000, int(cfg.chunk_rows or 100_000)), 200_000))
    flush_rows = max(5_000, min(max(5_000, batch_rows // 3), 20_000))
    missing_set = set(missing_years)
    uf = resultados_uf_from_path(r_path)
    logging.info(
        "Materializando cache UF/ano/bucket | uf=%s | anos_pendentes=%s | batch_rows=%s | flush_rows=%s | buckets=%s",
        uf,
        ", ".join(missing_years),
        batch_rows,
        flush_rows,
        RESULTADOS_HASH_BUCKETS,
    )
    write_ouro_event(
        progress_dir,
        label,
        "buckets_uf_ano_inicio",
        uf=uf,
        anos=missing_years,
        entrada=str(r_path),
        saida=str(root),
        batch_rows=batch_rows,
        flush_rows=flush_rows,
        buckets=RESULTADOS_HASH_BUCKETS,
    )

    rows_read = 0
    rows_kept = 0
    rows_flushed = 0
    files_written = 0
    batch_index = 0
    last_log_rows = 0
    started = time.perf_counter()
    part_counter: dict[tuple[str, int], int] = defaultdict(int)
    buffers: dict[tuple[str, int], list[pd.DataFrame]] = defaultdict(list)
    buffer_rows: dict[tuple[str, int], int] = defaultdict(int)
    rows_by_year: dict[str, int] = defaultdict(int)

    for batch in dataset.to_batches(columns=columns, batch_size=batch_rows):
        batch_index += 1
        rows_read += int(batch.num_rows)
        if batch.num_rows <= 0:
            continue
        df = batch.to_pandas()
        df["_bucket_ano"] = df["ano"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).replace({"": "SEM_ANO"})
        df = df.loc[df["_bucket_ano"].isin(missing_set)]
        if df.empty:
            if rows_read - last_log_rows >= FILE_PROGRESS_EVERY_ROWS:
                last_log_rows = rows_read
                elapsed = max(0.001, time.perf_counter() - started)
                logging.info("Cache UF/ano/bucket %s | batch=%s | lidas=%s | aproveitadas=%s | arquivos=%s | linhas/s=%.0f", uf, batch_index, rows_read, rows_kept, files_written, rows_read / elapsed)
            continue
        votos = pd.to_numeric(df.get("votos"), errors="coerce").fillna(0)
        df = df.loc[votos.gt(0)].copy()
        if df.empty:
            continue

        municipio = df["cd_municipio"].fillna("SEM_VALOR").astype(str).replace({"": "SEM_VALOR"})
        df["bucket_municipio"] = (pd.util.hash_pandas_object(municipio, index=False).astype("uint64") % RESULTADOS_HASH_BUCKETS).astype("int64").to_numpy()
        rows_kept += int(len(df))

        for (year_value, bucket), group in df.groupby(["_bucket_ano", "bucket_municipio"], sort=False, dropna=False):
            year_text = safe_text(year_value, "SEM_ANO") or "SEM_ANO"
            bucket_id = int(bucket)
            key = (year_text, bucket_id)
            group = group.drop(columns=["_bucket_ano"])
            buffers[key].append(group)
            buffer_rows[key] += int(len(group))
            rows_by_year[year_text] += int(len(group))
            if buffer_rows[key] >= flush_rows:
                flushed = flush_resultados_bucket_buffer(buffers, buffer_rows, part_counter, root, key)
                rows_flushed += flushed
                files_written += 1 if flushed else 0

        if rows_read - last_log_rows >= FILE_PROGRESS_EVERY_ROWS or batch_index == 1:
            last_log_rows = rows_read
            elapsed = max(0.001, time.perf_counter() - started)
            logging.info(
                "Cache UF/ano/bucket %s | batch=%s | lidas=%s | aproveitadas=%s | gravadas=%s | arquivos=%s | linhas/s=%.0f",
                uf,
                batch_index,
                rows_read,
                rows_kept,
                rows_flushed,
                files_written,
                rows_read / elapsed,
            )
            write_ouro_event(
                progress_dir,
                label,
                "buckets_uf_ano_progresso",
                uf=uf,
                anos=missing_years,
                batch=batch_index,
                linhas_lidas=rows_read,
                linhas_aproveitadas=rows_kept,
                linhas_gravadas=rows_flushed,
                arquivos_gravados=files_written,
                linhas_por_segundo=round(rows_read / elapsed, 2),
            )
        if batch_index % 25 == 0:
            clean_memory()

    for key in list(buffers.keys()):
        flushed = flush_resultados_bucket_buffer(buffers, buffer_rows, part_counter, root, key)
        rows_flushed += flushed
        files_written += 1 if flushed else 0

    duration = time.perf_counter() - started
    logging.info(
        "Cache UF/ano/bucket finalizado | uf=%s | anos=%s | duracao=%.1fs | lidas=%s | aproveitadas=%s | gravadas=%s | arquivos=%s",
        uf,
        ", ".join(missing_years),
        duration,
        rows_read,
        rows_kept,
        rows_flushed,
        files_written,
    )
    payload = {
        "uf": uf,
        "anos": missing_years,
        "entrada": str(r_path),
        "saida": str(root),
        "linhas_lidas": rows_read,
        "linhas_aproveitadas": rows_kept,
        "linhas_gravadas": rows_flushed,
        "arquivos_gravados": files_written,
        "linhas_por_ano": dict(rows_by_year),
        "duracao_segundos": round(duration, 3),
    }
    save_json(payload, progress_dir / f"{safe_name('resultados_buckets_uf_' + uf, 80)}.done.json")
    write_ouro_event(progress_dir, label, "buckets_uf_ano_fim", **payload)
    return root if rows_flushed > 0 else None


def materialize_resultados_hash_buckets_streaming(
    cfg: CleanDatabaseConfig,
    progress_dir: Path,
    r_path: Path,
    year: str,
    bucket_root: Path,
    slice_key: str,
) -> Path | None:
    import pyarrow.dataset as ds

    label = "ouro_resultados"
    if not parquet_dataset_exists(r_path):
        return None

    remove_path_if_exists(bucket_root)
    bucket_root.mkdir(parents=True, exist_ok=True)

    dataset = ds.dataset(str(r_path), format="parquet", partitioning="hive")
    available = set(dataset.schema.names)
    columns = [c for c in RESULTADOS_COLS if c in available]
    required = {"cd_municipio", "votos"}
    if year and "ano" in available:
        required.add("ano")
    if not required.issubset(set(columns)):
        missing = sorted(required - set(columns))
        raise RuntimeError(f"Colunas obrigatorias ausentes para buckets streaming: {missing}")

    batch_rows = max(10_000, min(max(10_000, int(cfg.chunk_rows or 100_000)), 200_000))
    logging.info(
        "Materializando cestas streaming %s | entrada=%s | batch_rows=%s | buckets=%s",
        slice_key,
        r_path,
        batch_rows,
        RESULTADOS_HASH_BUCKETS,
    )
    write_ouro_event(
        progress_dir,
        label,
        "buckets_streaming_inicio",
        slice=slice_key,
        entrada=str(r_path),
        saida=str(bucket_root),
        batch_rows=batch_rows,
        buckets=RESULTADOS_HASH_BUCKETS,
    )

    rows_read = 0
    rows_written = 0
    batch_index = 0
    part_counter: dict[int, int] = defaultdict(int)
    last_log_rows = 0
    started = time.perf_counter()

    for batch in dataset.to_batches(columns=columns, batch_size=batch_rows):
        batch_index += 1
        rows_read += int(batch.num_rows)
        if batch.num_rows <= 0:
            continue
        df = batch.to_pandas()
        if year and "ano" in df.columns:
            df = df.loc[df["ano"].astype(str).eq(str(year))]
        if df.empty:
            continue
        votos = pd.to_numeric(df.get("votos"), errors="coerce").fillna(0)
        df = df.loc[votos.gt(0)].copy()
        if df.empty:
            continue

        municipio = df["cd_municipio"].fillna("SEM_VALOR").astype(str).replace({"": "SEM_VALOR"})
        buckets = (pd.util.hash_pandas_object(municipio, index=False).astype("uint64") % RESULTADOS_HASH_BUCKETS).astype("int64")
        df["bucket_municipio"] = buckets.to_numpy()

        for bucket, group in df.groupby("bucket_municipio", sort=False, dropna=False):
            bucket_id = int(bucket)
            out_dir = bucket_root / f"bucket_municipio={bucket_id}"
            out_dir.mkdir(parents=True, exist_ok=True)
            part_id = part_counter[bucket_id]
            part_counter[bucket_id] += 1
            out_file = out_dir / f"part-{part_id:06d}.parquet"
            group.to_parquet(out_file, index=False, compression="snappy")
            rows_written += int(len(group))

        if rows_read - last_log_rows >= FILE_PROGRESS_EVERY_ROWS or batch_index == 1:
            last_log_rows = rows_read
            elapsed = max(0.001, time.perf_counter() - started)
            logging.info(
                "Buckets streaming %s | batch=%s | lidas=%s | gravadas=%s | linhas/s=%.0f",
                slice_key,
                batch_index,
                rows_read,
                rows_written,
                rows_read / elapsed,
            )
            write_ouro_event(
                progress_dir,
                label,
                "buckets_streaming_progresso",
                slice=slice_key,
                batch=batch_index,
                linhas_lidas=rows_read,
                linhas_gravadas=rows_written,
                buckets_com_dados=len(part_counter),
                linhas_por_segundo=round(rows_read / elapsed, 2),
            )
        clean_memory()

    duration = time.perf_counter() - started
    logging.info(
        "Buckets streaming finalizado %s em %.1fs | lidas=%s | gravadas=%s | buckets_com_dados=%s",
        slice_key,
        duration,
        rows_read,
        rows_written,
        len(part_counter),
    )
    write_ouro_event(
        progress_dir,
        label,
        "buckets_streaming_fim",
        slice=slice_key,
        linhas_lidas=rows_read,
        linhas_gravadas=rows_written,
        buckets_com_dados=len(part_counter),
        duracao_segundos=round(duration, 3),
        saida=str(bucket_root),
    )
    if rows_written <= 0:
        return None
    save_json(
        {
            "slice": slice_key,
            "entrada": str(r_path),
            "saida": str(bucket_root),
            "linhas_lidas": rows_read,
            "linhas_gravadas": rows_written,
            "buckets_com_dados": len(part_counter),
            "partes_por_bucket": {str(k): int(v) for k, v in sorted(part_counter.items())},
            "duracao_segundos": round(duration, 3),
        },
        progress_dir / f"{safe_name('resultados_buckets_' + slice_key, 80)}.done.json",
    )
    return bucket_root


def list_resultados_bucket_dirs(bucket_root: Path) -> list[tuple[str, Path]]:
    if not parquet_dataset_exists(bucket_root):
        return []
    buckets: list[tuple[str, Path]] = []
    for path in sorted(bucket_root.iterdir(), key=lambda p: p.name):
        if not path.is_dir() or not path.name.startswith("bucket_municipio="):
            continue
        bucket = safe_text(path.name.split("=", 1)[1], "0") or "0"
        if parquet_dataset_exists(path):
            buckets.append((bucket, path))
    if buckets:
        return buckets
    return [("0", bucket_root)]


def process_resultados_bucketed(
    cfg: CleanDatabaseConfig,
    progress_dir: Path,
    bucket_root: Path,
    out_root: Path,
    root_task_name: str,
    slice_key: str,
    outputs: dict[str, str],
) -> None:
    label = "ouro_resultados"
    buckets = list_resultados_bucket_dirs(bucket_root)
    if not buckets:
        logging.warning("Nenhum bucket fisico encontrado em %s para %s.", bucket_root, root_task_name)
        return

    pending: list[tuple[int, str, Path]] = []
    for index, (bucket, bucket_path) in enumerate(buckets, start=1):
        bucket_key = safe_name(f"bucket_{bucket}", 40)
        task_name = f"{root_task_name}_{bucket_key}"
        out = out_root / f"split={bucket_key}"
        marker = progress_dir / f"{safe_name(task_name, 80)}.done.json"
        if cfg.resume and marker.exists() and out.exists():
            logging.info("Pulando bucket ja concluido %s/%s: %s", index, len(buckets), task_name)
            outputs[task_name] = str(out)
            continue
        pending.append((index, bucket, bucket_path))

    logging.info(
        "Processando cestas fisicas de %s: total=%s, pendentes=%s, workers=%s.",
        root_task_name,
        len(buckets),
        len(pending),
        max(1, int(cfg.ouro_workers or 1)),
    )
    write_ouro_event(
        progress_dir,
        label,
        "buckets_processamento_inicio",
        tarefa=root_task_name,
        slice=slice_key,
        total_buckets=len(buckets),
        pendentes=len(pending),
        workers=max(1, int(cfg.ouro_workers or 1)),
    )
    if not pending:
        return

    requested_workers = max(1, int(cfg.ouro_workers or 1))
    workers = min(requested_workers, len(pending))
    threads_per_task = max(1, int(cfg.duckdb_threads or 1) // workers)
    worker_cfg = replace(cfg, duckdb_threads=threads_per_task)
    if workers <= 1:
        for index, bucket, bucket_path in pending:
            outputs.update(process_resultados_bucket(worker_cfg, progress_dir, bucket, bucket_path, out_root, root_task_name, slice_key, index, len(buckets)))
            clean_memory()
        return

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {}
        for index, bucket, bucket_path in pending:
            logging.info("Enfileirando bucket %s/%s de %s: %s", index, len(buckets), root_task_name, bucket)
            write_ouro_event(
                progress_dir,
                label,
                "bucket_enfileirado",
                tarefa=root_task_name,
                slice=slice_key,
                bucket=bucket,
                indice_bucket=index,
                total_buckets=len(buckets),
                entrada=str(bucket_path),
            )
            future = pool.submit(
                process_resultados_bucket,
                worker_cfg,
                progress_dir,
                bucket,
                bucket_path,
                out_root,
                root_task_name,
                slice_key,
                index,
                len(buckets),
            )
            future_map[future] = bucket
        completed = 0
        for future in as_completed(future_map):
            bucket = future_map[future]
            completed += 1
            try:
                result_map = future.result()
                outputs.update(result_map)
                logging.info("Bucket concluido %s/%s de %s: %s", completed, len(pending), root_task_name, bucket)
                write_ouro_event(
                    progress_dir,
                    label,
                    "bucket_concluido",
                    tarefa=root_task_name,
                    slice=slice_key,
                    bucket=bucket,
                    concluidos=completed,
                    pendentes=len(pending),
                )
            except Exception as exc:
                logging.exception("Erro no bucket %s de %s: %s", bucket, root_task_name, exc)
                write_ouro_event(progress_dir, label, "bucket_erro", tarefa=root_task_name, slice=slice_key, bucket=bucket, erro=str(exc))
            clean_memory()
    write_ouro_event(
        progress_dir,
        label,
        "buckets_processamento_fim",
        tarefa=root_task_name,
        slice=slice_key,
        total_buckets=len(buckets),
        pendentes_processadas=len(pending),
    )


def process_resultados_bucket(
    cfg: CleanDatabaseConfig,
    progress_dir: Path,
    bucket: str,
    bucket_path: Path,
    out_root: Path,
    root_task_name: str,
    slice_key: str,
    index: int,
    total: int,
) -> dict[str, str]:
    label = "ouro_resultados"
    bucket_key = safe_name(f"bucket_{bucket}", 40)
    task_name = f"{root_task_name}_{bucket_key}"
    out = out_root / f"split={bucket_key}"
    marker = progress_dir / f"{safe_name(task_name, 80)}.done.json"
    if cfg.resume and marker.exists() and out.exists():
        logging.info("Pulando bucket ja concluido %s/%s: %s", index, total, task_name)
        return {task_name: str(out)}

    bucket_expr = dataset_expr(bucket_path)
    logging.info("Processando bucket fisico %s/%s de %s: bucket=%s | entrada=%s", index, total, root_task_name, bucket, bucket_path)
    write_ouro_event(
        progress_dir,
        label,
        "bucket_processando",
        tarefa=task_name,
        tarefa_raiz=root_task_name,
        slice=slice_key,
        bucket=bucket,
        indice_bucket=index,
        total_buckets=total,
        entrada=str(bucket_path),
        saida=str(out),
        duckdb_threads=cfg.duckdb_threads,
    )
    task = copy_task(task_name, vencedores_secao_sql(bucket_expr), out, partition_by=SECTION_PARTITION_COLS)
    result = execute_ouro_task(task, cfg, label, progress_dir)
    local_outputs = {task_name: result}
    if str(result).startswith("ERRO:"):
        logging.warning("DuckDB falhou no bucket %s; tentando fallback PyArrow/Pandas.", task_name)
        remove_path_if_exists(out)
        fallback = write_vencedores_secao_pyarrow(bucket_path, out, task_name, progress_dir, label)
        if fallback:
            local_outputs[task_name] = str(out)
            clean_memory()
            return local_outputs
        logging.warning("Bucket %s ainda ficou pesado; subdividindo por cargo/turno/zona.", task_name)
        remove_path_if_exists(out)
        process_resultados_split_level(
            replace(cfg, ouro_workers=1),
            progress_dir,
            bucket_expr,
            out,
            task_name,
            slice_key,
            1,
            local_outputs,
        )
    clean_memory()
    return local_outputs


def write_vencedores_secao_pyarrow(
    source_path: Path,
    out: Path,
    task_name: str,
    progress_dir: Path,
    label: str,
) -> bool:
    try:
        import pyarrow.dataset as ds
    except Exception as exc:
        logging.warning("Fallback PyArrow indisponivel para %s: %s", task_name, exc)
        return False

    started = time.perf_counter()
    key_cols = ["ano", "uf", "cd_municipio", "nm_municipio", "zona", "secao", "cargo", "turno"]
    entity_cols = ["partido", "candidato", "nr_votavel"]
    needed = key_cols + entity_cols + ["votos"]
    try:
        dataset = ds.dataset(str(source_path), format="parquet", partitioning="hive")
        available = set(dataset.schema.names)
        columns = [c for c in needed if c in available]
        if "votos" not in columns:
            raise RuntimeError("coluna votos ausente")
        for col in needed:
            if col not in columns and col != "votos":
                columns.append(col)

        frames: list[pd.DataFrame] = []
        rows_read = 0
        rows_kept = 0
        for batch in dataset.to_batches(columns=[c for c in columns if c in available], batch_size=100_000):
            rows_read += int(batch.num_rows)
            if batch.num_rows <= 0:
                continue
            df = batch.to_pandas()
            for col in needed:
                if col not in df.columns:
                    df[col] = ""
            df["votos"] = pd.to_numeric(df["votos"], errors="coerce").fillna(0)
            df = df.loc[df["votos"].gt(0), needed].copy()
            if df.empty:
                continue
            for col in key_cols + entity_cols:
                df[col] = df[col].map(lambda x: safe_text(x, "SEM_VALOR") or "SEM_VALOR")
            grouped = df.groupby(key_cols + entity_cols, dropna=False, as_index=False)["votos"].sum()
            frames.append(grouped)
            rows_kept += int(len(df))
            if len(frames) >= 32:
                frames = [pd.concat(frames, ignore_index=True).groupby(key_cols + entity_cols, dropna=False, as_index=False)["votos"].sum()]
            clean_memory()

        if not frames:
            logging.warning("Fallback PyArrow sem votos validos para %s.", task_name)
            return False
        agg = pd.concat(frames, ignore_index=True).groupby(key_cols + entity_cols, dropna=False, as_index=False)["votos"].sum()
        if agg.empty:
            return False
        agg = agg.sort_values(key_cols + ["votos"], ascending=[True] * len(key_cols) + [False])
        agg["votos_total_secao"] = agg.groupby(key_cols, dropna=False)["votos"].transform("sum")
        winners = agg.drop_duplicates(subset=key_cols, keep="first").copy()
        winners = winners.rename(columns={
            "partido": "partido_vencedor",
            "candidato": "candidato_vencedor",
            "votos": "votos_vencedor",
        })
        winners["share_vencedor"] = winners["votos_vencedor"] / winners["votos_total_secao"].replace({0: pd.NA})
        out_cols = key_cols + ["partido_vencedor", "candidato_vencedor", "nr_votavel", "votos_vencedor", "votos_total_secao", "share_vencedor"]
        winners = winners[out_cols]

        remove_path_if_exists(out)
        out.mkdir(parents=True, exist_ok=True)
        out_file = out / "part-000000.parquet"
        winners.to_parquet(out_file, index=False, compression="snappy")
        duration = time.perf_counter() - started
        logging.info(
            "Fallback PyArrow finalizado: %s em %.1fs | lidas=%s | votos_validos=%s | vencedores=%s -> %s",
            task_name,
            duration,
            rows_read,
            rows_kept,
            len(winners),
            out_file,
        )
        write_ouro_event(
            progress_dir,
            label,
            "fallback_pyarrow_vencedores_fim",
            tarefa=task_name,
            entrada=str(source_path),
            saida=str(out),
            linhas_lidas=rows_read,
            votos_validos=rows_kept,
            vencedores=len(winners),
            duracao_segundos=round(duration, 3),
        )
        remove_path_if_exists(progress_dir / f"{safe_name(task_name, 80)}.error.json")
        save_json(
            {
                "label": label,
                "tarefa": task_name,
                "status": "ok",
                "modo": "fallback_pyarrow",
                "saida": str(out),
                "duracao_segundos": round(duration, 3),
            },
            progress_dir / f"{safe_name(task_name, 80)}.done.json",
        )
        return True
    except Exception as exc:
        logging.exception("Fallback PyArrow falhou em %s: %s", task_name, exc)
        write_ouro_event(progress_dir, label, "fallback_pyarrow_vencedores_erro", tarefa=task_name, entrada=str(source_path), erro=str(exc))
        return False


def process_resultados_split_level(
    cfg: CleanDatabaseConfig,
    progress_dir: Path,
    parent_expr: str,
    out_root: Path,
    root_task_name: str,
    slice_key: str,
    level_index: int,
    outputs: dict[str, str],
) -> None:
    if level_index >= len(RESULTADOS_SPLIT_LEVELS):
        task = copy_task(
            root_task_name,
            vencedores_secao_sql(parent_expr),
            out_root / f"split={safe_name(slice_key, 60)}",
            partition_by=SECTION_PARTITION_COLS,
        )
        logging.info("Executando fallback final sem novas subpartes: %s -> %s", task.get("name"), task.get("out"))
        write_ouro_event(
            progress_dir,
            "ouro_resultados",
            "executando_fallback_final",
            tarefa=root_task_name,
            slice=slice_key,
            saida=str(task.get("out", "")),
        )
        outputs[task["name"]] = execute_ouro_task(task, cfg, "ouro_resultados", progress_dir)
        return

    level_name, columns = RESULTADOS_SPLIT_LEVELS[level_index]
    logging.info(
        "Descobrindo subpartes de %s | nivel=%s | colunas=%s",
        root_task_name,
        level_name,
        ", ".join(columns),
    )
    write_ouro_event(
        progress_dir,
        "ouro_resultados",
        "descobrindo_subpartes",
        tarefa=root_task_name,
        slice=slice_key,
        nivel=level_name,
        colunas=columns,
        saida=str(out_root),
    )
    started = time.perf_counter()
    combos = list_split_combinations(level_name, parent_expr, columns, cfg)
    duration = time.perf_counter() - started
    logging.info(
        "Subpartes descobertas de %s | nivel=%s | qtd=%s | duracao=%.1fs",
        root_task_name,
        level_name,
        len(combos),
        duration,
    )
    write_ouro_event(
        progress_dir,
        "ouro_resultados",
        "subpartes_descobertas",
        tarefa=root_task_name,
        slice=slice_key,
        nivel=level_name,
        qtd_subpartes=len(combos),
        duracao_segundos=round(duration, 3),
    )
    if not combos:
        logging.warning("Sem combinacoes para fatiar %s no nivel %s; tentando query direta.", slice_key, level_name)
        process_resultados_split_level(cfg, progress_dir, parent_expr, out_root, root_task_name, slice_key, level_index + 1, outputs)
        return

    logging.info(
        "Fatiando %s no nivel %s: %s subparte(s).",
        slice_key,
        level_name,
        len(combos),
    )
    pending_combos: list[tuple[int, dict[str, str]]] = []
    for index, combo in enumerate(combos, start=1):
        combo_key = split_combo_key(combo)
        task_name = f"{root_task_name}_{level_name}_{combo_key}"
        out = out_root / f"split={safe_name(level_name + '_' + combo_key, 90)}"
        marker = progress_dir / f"{safe_name(task_name, 80)}.done.json"
        if cfg.resume and marker.exists() and out.exists():
            logging.info("Pulando subparte ja concluida %s/%s: %s", index, len(combos), task_name)
            outputs[task_name] = str(out)
            continue
        pending_combos.append((index, combo))
    if not pending_combos:
        logging.info("Todas as subpartes de %s ja estavam concluidas.", root_task_name)
        return

    requested_workers = max(1, int(cfg.ouro_workers or 1))
    workers = min(requested_workers, len(pending_combos))
    if workers <= 1:
        for index, combo in pending_combos:
            outputs.update(process_resultados_split_combo(
                cfg,
                progress_dir,
                parent_expr,
                out_root,
                root_task_name,
                slice_key,
                level_index,
                level_name,
                combo,
                index,
                len(combos),
            ))
            clean_memory()
        return

    threads_per_task = max(1, int(cfg.duckdb_threads or 1) // workers)
    worker_cfg = replace(cfg, duckdb_threads=threads_per_task)
    logging.info(
        "Paralelizando subpartes de %s | nivel=%s | workers=%s | duckdb_threads_por_worker=%s",
        root_task_name,
        level_name,
        workers,
        threads_per_task,
    )
    write_ouro_event(
        progress_dir,
        "ouro_resultados",
        "subpartes_paralelas_inicio",
        tarefa=root_task_name,
        slice=slice_key,
        nivel=level_name,
        total_subpartes=len(combos),
        pendentes=len(pending_combos),
        workers=workers,
        duckdb_threads_por_worker=threads_per_task,
    )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {}
        for index, combo in pending_combos:
            write_ouro_event(
                progress_dir,
                "ouro_resultados",
                "subparte_enfileirada",
                tarefa=root_task_name,
                slice=slice_key,
                nivel=level_name,
                indice_subparte=index,
                total_subpartes=len(combos),
                filtro=combo,
            )
            future = pool.submit(
                process_resultados_split_combo,
                worker_cfg,
                progress_dir,
                parent_expr,
                out_root,
                root_task_name,
                slice_key,
                level_index,
                level_name,
                combo,
                index,
                len(combos),
            )
            future_map[future] = combo
        completed = 0
        for future in as_completed(future_map):
            completed += 1
            try:
                result_map = future.result()
                outputs.update(result_map)
                logging.info("Subparte paralela concluida %s/%s pendentes de %s", completed, len(pending_combos), root_task_name)
                write_ouro_event(progress_dir, "ouro_resultados", "subparte_paralela_concluida", tarefa=root_task_name, concluidas=completed, pendentes=len(pending_combos), total_subpartes=len(combos))
            except Exception as exc:
                combo = future_map[future]
                logging.exception("Erro em subparte paralela %s de %s: %s", combo, root_task_name, exc)
                write_ouro_event(progress_dir, "ouro_resultados", "subparte_paralela_erro", tarefa=root_task_name, filtro=combo, erro=str(exc))
            clean_memory()
    write_ouro_event(progress_dir, "ouro_resultados", "subpartes_paralelas_fim", tarefa=root_task_name, slice=slice_key, nivel=level_name, total_subpartes=len(combos), pendentes_processadas=len(pending_combos))


def process_resultados_split_combo(
    cfg: CleanDatabaseConfig,
    progress_dir: Path,
    parent_expr: str,
    out_root: Path,
    root_task_name: str,
    slice_key: str,
    level_index: int,
    level_name: str,
    combo: dict[str, str],
    index: int,
    total: int,
) -> dict[str, str]:
    local_outputs: dict[str, str] = {}
    combo_expr = filter_expr_by_combo(parent_expr, combo)
    combo_key = split_combo_key(combo)
    task_name = f"{root_task_name}_{level_name}_{combo_key}"
    out = out_root / f"split={safe_name(level_name + '_' + combo_key, 90)}"
    marker = progress_dir / f"{safe_name(task_name, 80)}.done.json"
    if cfg.resume and marker.exists() and out.exists():
        logging.info("Pulando subparte ja concluida %s/%s: %s", index, total, task_name)
        return {task_name: str(out)}

    logging.info("Processando subparte %s/%s de %s: %s", index, total, slice_key, task_name)
    write_ouro_event(
        progress_dir,
        "ouro_resultados",
        "processando_subparte",
        tarefa=task_name,
        tarefa_raiz=root_task_name,
        slice=slice_key,
        nivel=level_name,
        indice_subparte=index,
        total_subpartes=total,
        filtro=combo,
        saida=str(out),
        duckdb_threads=cfg.duckdb_threads,
    )
    task = copy_task(task_name, vencedores_secao_sql(combo_expr), out, partition_by=SECTION_PARTITION_COLS)
    result = execute_ouro_task(task, cfg, "ouro_resultados", progress_dir)
    local_outputs[task_name] = result
    if str(result).startswith("ERRO:"):
        logging.warning(
            "Subparte %s falhou; dividindo mais uma vez a partir do nivel %s.",
            task_name,
            RESULTADOS_SPLIT_LEVELS[min(level_index + 1, len(RESULTADOS_SPLIT_LEVELS) - 1)][0],
        )
        remove_path_if_exists(out)
        process_resultados_split_level(
            replace(cfg, ouro_workers=1),
            progress_dir,
            combo_expr,
            out_root / f"split={safe_name(level_name + '_' + combo_key, 90)}",
            task_name,
            slice_key,
            level_index + 1,
            local_outputs,
        )
    clean_memory()
    return local_outputs


def run_prata_minima_correlacoes_stage(
    cfg: CleanDatabaseConfig,
    prata: Path,
    analyses: Path,
    plan: list[dict[str, Any]] | None = None,
    uf_parts: list[tuple[str, Path]] | None = None,
) -> dict[str, str]:
    label = "prata_minima"
    progress_dir = prepare_ouro_progress(cfg, label)
    write_ouro_event(progress_dir, label, "inicio_etapa", etapa=label)
    if uf_parts is None:
        uf_parts = [(uf, e_path) for uf, e_path in list_uf_partition_dirs(prata / "eleitorado") if parquet_dataset_exists(prata / "resultados_votos" / f"uf={uf}")]
    logging.info("Construindo prata_minima antes de todas as analises: %s UFs com eleitorado+resultados.", len(uf_parts))
    if plan is None:
        plan = build_correlacao_uf_year_plan(cfg, prata, uf_parts)
    outputs = materialize_prata_minima_correlacoes(cfg, analyses, plan, progress_dir, label)
    write_ouro_event(progress_dir, label, "fim_etapa", etapa=label, outputs=outputs)
    save_json({"label": label, "status": "ok", "outputs": outputs}, progress_dir / f"{label}_progresso.json")
    return outputs


def run_ouro_nivelado_analyses(
    cfg: CleanDatabaseConfig,
    analyses: Path,
    plan: list[dict[str, Any]],
) -> dict[str, str]:
    """Build ouro from small Lego-like pieces.

    Order is intentional:
    1. municipal outputs from prata_minima;
    2. state outputs from municipal outputs;
    3. Brazil outputs from state outputs;
    4. compatibility views from the already reduced outputs.
    """
    label = "ouro_nivelado"
    progress_dir = prepare_ouro_progress(cfg, label)
    write_ouro_event(progress_dir, label, "inicio_etapa", etapa=label)
    reset_ouro_targets(
        [
            analyses / "municipal",
            analyses / "estadual",
            analyses / "brasil",
            analyses / "retrato_municipal",
            analyses / "timeline_municipal",
            analyses / "timeline_uf",
            analyses / "timeline_nacional.parquet",
            analyses / "perfil_eleitor_por_ano",
            analyses / "perfil_eleitor_por_partido",
            analyses / "perfil_eleitor_por_candidato",
            analyses / "comparativo_anual_perfil_partido",
            analyses / "comparativo_anual_perfil_candidato",
            analyses / "top10_perfis_federacao_estado_municipio",
            analyses / "resultado_eleitorado_por_secao",
        ],
        resume=cfg.resume,
    )

    outputs: dict[str, str] = {}
    plan_by_uf: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in sort_uf_year_plan(plan):
        plan_by_uf[safe_text(item.get("uf"), "SEM_UF") or "SEM_UF"].append(item)

    ordered_ufs = sorted(plan_by_uf, key=uf_sort_key)
    logging.info("Ouro nivelado: %s UFs em ordem alfabetica. Primeiro municipal, depois estadual, depois Brasil.", len(ordered_ufs))
    save_json(
        {
            "label": label,
            "status": "planejado",
            "ordem": "prata_minima -> municipal -> estadual -> brasil",
            "ufs": ordered_ufs,
            "fatias_uf_ano": len(plan),
        },
        progress_dir / "ouro_nivelado_plano.json",
    )

    for uf_index, uf in enumerate(ordered_ufs, start=1):
        uf_plan = sorted(plan_by_uf[uf], key=lambda item: year_sort_key(item.get("ano", "")))
        logging.info("Ouro nivelado UF %s/%s: %s | anos=%s", uf_index, len(ordered_ufs), uf, ", ".join(safe_text(x.get("ano", "")) for x in uf_plan))
        write_ouro_event(
            progress_dir,
            label,
            "uf_inicio",
            uf=uf,
            indice_uf=uf_index,
            total_ufs=len(ordered_ufs),
            anos=[safe_text(x.get("ano", "")) for x in uf_plan],
        )
        for year_index, item in enumerate(uf_plan, start=1):
            outputs.update(run_ouro_municipal_slice(cfg, analyses, progress_dir, label, item, uf_index, len(ordered_ufs), year_index, len(uf_plan)))
            clean_memory()
        outputs.update(run_ouro_estadual_uf(cfg, analyses, progress_dir, label, uf, uf_index, len(ordered_ufs)))
        write_ouro_event(progress_dir, label, "uf_fim", uf=uf, indice_uf=uf_index, total_ufs=len(ordered_ufs))
        clean_memory()

    outputs.update(run_ouro_brasil_final(cfg, analyses, progress_dir, label))
    outputs.update(run_ouro_compatibilidade_final(cfg, analyses, progress_dir, label))
    write_ouro_event(progress_dir, label, "fim_etapa", etapa=label, outputs=outputs)
    save_json({"label": label, "status": "ok", "outputs": outputs}, progress_dir / f"{label}_progresso.json")
    return outputs


def run_ouro_estados_brasil_analyses(
    cfg: CleanDatabaseConfig,
    analyses: Path,
    plan: list[dict[str, Any]],
) -> dict[str, str]:
    """Short dashboard path: state outputs first, then Brazil, no municipal detail."""
    label = "ouro_estados_brasil"
    progress_dir = prepare_ouro_progress(cfg, label)
    write_ouro_event(progress_dir, label, "inicio_etapa", etapa=label)
    reset_ouro_targets(
        [
            analyses / "estadual",
            analyses / "brasil",
            analyses / "timeline_uf",
            analyses / "timeline_nacional.parquet",
            analyses / "perfil_eleitor_por_ano",
            analyses / "perfil_eleitor_por_partido",
            analyses / "comparativo_anual_perfil_partido",
            analyses / "top10_perfis_federacao_estado_municipio",
        ],
        resume=cfg.resume,
    )

    outputs: dict[str, str] = {}
    plan_by_uf: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in sort_uf_year_plan(plan):
        plan_by_uf[safe_text(item.get("uf"), "SEM_UF") or "SEM_UF"].append(item)

    ordered_ufs = sorted(plan_by_uf, key=uf_sort_key)
    logging.info(
        "Ouro estados+Brasil: %s UFs em ordem alfabetica; municipal detalhado sera ignorado neste modo.",
        len(ordered_ufs),
    )
    save_json(
        {
            "label": label,
            "status": "planejado",
            "ordem": "prata_minima -> estadual -> brasil",
            "ufs": ordered_ufs,
            "fatias_uf_ano": len(plan),
            "clusters": "pulados" if cfg.skip_clusters else "ativos",
        },
        progress_dir / "ouro_estados_brasil_plano.json",
    )

    for uf_index, uf in enumerate(ordered_ufs, start=1):
        uf_plan = sorted(plan_by_uf[uf], key=lambda item: year_sort_key(item.get("ano", "")))
        logging.info(
            "Ouro estados+Brasil UF %s/%s: %s | anos=%s",
            uf_index,
            len(ordered_ufs),
            uf,
            ", ".join(safe_text(x.get("ano", "")) for x in uf_plan),
        )
        write_ouro_event(
            progress_dir,
            label,
            "uf_inicio",
            uf=uf,
            indice_uf=uf_index,
            total_ufs=len(ordered_ufs),
            anos=[safe_text(x.get("ano", "")) for x in uf_plan],
        )
        for year_index, item in enumerate(uf_plan, start=1):
            outputs.update(run_ouro_estadual_slice_from_prata_minima(
                cfg,
                analyses,
                progress_dir,
                label,
                item,
                uf_index,
                len(ordered_ufs),
                year_index,
                len(uf_plan),
            ))
            clean_memory()
        write_ouro_event(progress_dir, label, "uf_fim", uf=uf, indice_uf=uf_index, total_ufs=len(ordered_ufs))
        clean_memory()

    outputs.update(run_ouro_brasil_final(cfg, analyses, progress_dir, label))
    outputs.update(run_ouro_compatibilidade_final(cfg, analyses, progress_dir, label))
    write_ouro_event(progress_dir, label, "fim_etapa", etapa=label, outputs=outputs)
    save_json({"label": label, "status": "ok", "outputs": outputs}, progress_dir / f"{label}_progresso.json")
    return outputs


def run_ouro_estadual_slice_from_prata_minima(
    cfg: CleanDatabaseConfig,
    analyses: Path,
    progress_dir: Path,
    label: str,
    item: dict[str, Any],
    uf_index: int,
    total_ufs: int,
    year_index: int,
    total_years: int,
) -> dict[str, str]:
    uf = safe_text(item.get("uf"), "SEM_UF") or "SEM_UF"
    year = safe_text(item.get("ano"), "") or ""
    slice_key = safe_text(item.get("slice_key"), slice_name(uf, year)) or slice_name(uf, year)
    profile_path = prata_minima_chunk(cfg, "perfil_secao", slice_key)
    result_path = prata_minima_chunk(cfg, "resultado_secao", slice_key)
    if not prata_minima_table_exists(cfg, "perfil_secao", slice_key) or not prata_minima_table_exists(cfg, "resultado_secao", slice_key):
        logging.warning("Ouro estadual curto pulado: prata_minima ausente para %s | perfil=%s resultado=%s", slice_key, profile_path, result_path)
        write_ouro_event(progress_dir, label, "estadual_prata_minima_ausente", uf=uf, ano=year, fatia=slice_key)
        return {}

    profile_expr = prata_minima_table_expr(profile_path, "perfil_secao", slice_key)
    result_expr = prata_minima_table_expr(result_path, "resultado_secao", slice_key)
    logging.info(
        "Ouro estados+Brasil fatia UF %s/%s ano %s/%s: %s %s | direto para estadual",
        uf_index,
        total_ufs,
        year_index,
        total_years,
        uf,
        year,
    )
    write_ouro_event(progress_dir, label, "estadual_fatia_inicio", uf=uf, ano=year, fatia=slice_key)

    tasks: list[dict[str, Any]] = []
    if ouro_task_enabled(cfg, "resumo"):
        tasks.append(copy_task(
            f"estadual_resumo_{slice_key}",
            aggregate_resumo_level_sql(sql_subquery(municipal_resumo_from_prata_minima_sql(profile_expr, result_expr)), "estado"),
            chunk_output(analyses / "estadual" / "resumo", slice_key),
            partition_by=YEAR_UF_PARTITION_COLS,
        ))
    if ouro_task_enabled(cfg, "perfil_eleitor"):
        tasks.append(copy_task(
            f"estadual_perfil_eleitor_{slice_key}",
            aggregate_perfil_eleitor_level_sql(sql_subquery(municipal_perfil_eleitor_from_prata_minima_sql(profile_expr)), "estado"),
            chunk_output(analyses / "estadual" / "perfil_eleitor", slice_key),
            partition_by=YEAR_UF_PARTITION_COLS,
        ))
    if ouro_task_enabled(cfg, "resultado_partido"):
        tasks.append(copy_task(
            f"estadual_resultado_partido_{slice_key}",
            aggregate_resultado_entidade_level_sql(sql_subquery(municipal_resultado_entidade_from_resultados_sql(result_expr, "partido")), "estado"),
            chunk_output(analyses / "estadual" / "resultado_partido", slice_key),
            partition_by=YEAR_UF_PARTITION_COLS,
        ))
    if ouro_task_enabled(cfg, "perfil_partido"):
        tasks.append(copy_task(
            f"estadual_perfil_partido_{slice_key}",
            aggregate_perfil_entidade_level_sql(sql_subquery(municipal_perfil_entidade_nivel_sql(profile_expr, result_expr, "partido")), "estado", "partido"),
            chunk_output(analyses / "estadual" / "perfil_partido", slice_key),
            partition_by=YEAR_UF_PARTITION_COLS,
        ))
    if ouro_task_enabled(cfg, "clusters_eleitores") or ouro_task_enabled(cfg, "clusters_eleitores_resultado"):
        tasks.extend([
            copy_task(
                f"estadual_clusters_eleitores_{slice_key}",
                aggregate_clusters_level_sql(sql_subquery(municipal_clusters_eleitores_sql(profile_expr, cfg)), "estado"),
                chunk_output(analyses / "estadual" / "clusters_eleitores", slice_key),
                partition_by=YEAR_UF_PARTITION_COLS,
            ),
            copy_task(
                f"estadual_clusters_eleitores_resultado_{slice_key}",
                aggregate_clusters_level_sql(sql_subquery(municipal_clusters_eleitores_resultado_sql(profile_expr, result_expr, cfg)), "estado"),
                chunk_output(analyses / "estadual" / "clusters_eleitores_resultado", slice_key),
                partition_by=YEAR_UF_PARTITION_COLS,
            ),
        ])
    if ouro_task_enabled(cfg, "resultado_candidato"):
        tasks.append(
            copy_task(
                f"estadual_resultado_candidato_{slice_key}",
                aggregate_resultado_entidade_level_sql(sql_subquery(municipal_resultado_entidade_from_resultados_sql(result_expr, "candidato")), "estado"),
                chunk_output(analyses / "estadual" / "resultado_candidato", slice_key),
                partition_by=YEAR_UF_PARTITION_COLS,
            )
        )
    if ouro_task_enabled(cfg, "perfil_candidato"):
        tasks.append(
            copy_task(
                f"estadual_perfil_candidato_{slice_key}",
                aggregate_perfil_entidade_level_sql(sql_subquery(municipal_perfil_entidade_nivel_sql(profile_expr, result_expr, "candidato")), "estado", "candidato"),
                chunk_output(analyses / "estadual" / "perfil_candidato", slice_key),
                partition_by=YEAR_UF_PARTITION_COLS,
            )
        )
    outputs = execute_copy_tasks(tasks, cfg, f"{label}_estadual_{slice_key}")
    write_ouro_event(progress_dir, label, "estadual_fatia_fim", uf=uf, ano=year, fatia=slice_key, tarefas=len(tasks))
    return outputs


def run_ouro_municipal_slice(
    cfg: CleanDatabaseConfig,
    analyses: Path,
    progress_dir: Path,
    label: str,
    item: dict[str, Any],
    uf_index: int,
    total_ufs: int,
    year_index: int,
    total_years: int,
) -> dict[str, str]:
    uf = safe_text(item.get("uf"), "SEM_UF") or "SEM_UF"
    year = safe_text(item.get("ano"), "") or ""
    slice_key = safe_text(item.get("slice_key"), slice_name(uf, year)) or slice_name(uf, year)
    profile_path = prata_minima_chunk(cfg, "perfil_secao", slice_key)
    result_path = prata_minima_chunk(cfg, "resultado_secao", slice_key)
    if not prata_minima_table_exists(cfg, "perfil_secao", slice_key) or not prata_minima_table_exists(cfg, "resultado_secao", slice_key):
        logging.error("Ouro municipal pulado: prata_minima ausente para %s | perfil=%s resultado=%s", slice_key, profile_path, result_path)
        write_ouro_event(progress_dir, label, "municipal_prata_minima_ausente", uf=uf, ano=year, fatia=slice_key)
        return {}

    profile_expr = prata_minima_table_expr(profile_path, "perfil_secao", slice_key)
    result_expr = prata_minima_table_expr(result_path, "resultado_secao", slice_key)
    logging.info(
        "Ouro municipal fatia UF %s/%s ano %s/%s: %s %s | fonte=prata_minima",
        uf_index,
        total_ufs,
        year_index,
        total_years,
        uf,
        year,
    )
    write_ouro_event(
        progress_dir,
        label,
        "municipal_fatia_inicio",
        uf=uf,
        ano=year,
        fatia=slice_key,
        perfil=str(profile_path),
        resultados=str(result_path),
    )
    municipios = list_municipios_for_municipal_slice(profile_expr, result_expr, cfg)
    if municipios and cfg.max_municipios_por_uf > 0:
        original_count = len(municipios)
        municipios = municipios[: max(1, int(cfg.max_municipios_por_uf))]
        logging.info(
            "Limite municipal ativo em %s: processando %s de %s municipio(s).",
            slice_key,
            len(municipios),
            original_count,
        )
        write_ouro_event(
            progress_dir,
            label,
            "municipal_limite_ativo",
            uf=uf,
            ano=year,
            fatia=slice_key,
            municipios_processados=len(municipios),
            municipios_total=original_count,
        )
    if municipios:
        logging.info(
            "Ouro municipal %s: %s municipio(s) serao processados um por vez para reaproveitar memoria.",
            slice_key,
            len(municipios),
        )
        write_ouro_event(
            progress_dir,
            label,
            "municipal_municipios_descobertos",
            uf=uf,
            ano=year,
            fatia=slice_key,
            municipios=len(municipios),
        )
        remove_legacy_municipal_slice_outputs(analyses, slice_key)
        outputs = run_ouro_municipios_queue(
            cfg,
            analyses,
            progress_dir,
            label,
            slice_key,
            uf,
            year,
            profile_expr,
            result_expr,
            municipios,
        )
        write_ouro_event(progress_dir, label, "municipal_fatia_fim", uf=uf, ano=year, fatia=slice_key, municipios=len(municipios))
        return outputs

    logging.warning("Ouro municipal %s: nenhum municipio encontrado; usando fallback por UF.", slice_key)
    tasks: list[dict[str, Any]] = []
    if ouro_task_enabled(cfg, "resumo"):
        tasks.append(copy_task(
            f"municipal_resumo_{slice_key}",
            municipal_resumo_from_prata_minima_sql(profile_expr, result_expr),
            chunk_output(analyses / "municipal" / "resumo", slice_key),
            partition_by=MUNICIPIO_PARTITION_COLS,
        ))
    if ouro_task_enabled(cfg, "perfil_eleitor"):
        tasks.append(copy_task(
            f"municipal_perfil_eleitor_{slice_key}",
            municipal_perfil_eleitor_from_prata_minima_sql(profile_expr),
            chunk_output(analyses / "municipal" / "perfil_eleitor", slice_key),
            partition_by=MUNICIPIO_PARTITION_COLS,
        ))
    if ouro_task_enabled(cfg, "resultado_partido"):
        tasks.append(copy_task(
            f"municipal_resultado_partido_{slice_key}",
            municipal_resultado_entidade_from_resultados_sql(result_expr, "partido"),
            chunk_output(analyses / "municipal" / "resultado_partido", slice_key),
            partition_by=MUNICIPIO_PARTITION_COLS,
        ))
    if ouro_task_enabled(cfg, "perfil_partido"):
        tasks.append(copy_task(
            f"municipal_perfil_partido_{slice_key}",
            municipal_perfil_entidade_nivel_sql(profile_expr, result_expr, "partido"),
            chunk_output(analyses / "municipal" / "perfil_partido", slice_key),
            partition_by=MUNICIPIO_PARTITION_COLS,
        ))
    if ouro_task_enabled(cfg, "clusters_eleitores"):
        tasks.append(copy_task(
            f"municipal_clusters_eleitores_{slice_key}",
            municipal_clusters_eleitores_sql(profile_expr, cfg),
            chunk_output(analyses / "municipal" / "clusters_eleitores", slice_key),
            partition_by=MUNICIPIO_PARTITION_COLS,
        ))
    if ouro_task_enabled(cfg, "clusters_eleitores_resultado"):
        tasks.append(copy_task(
            f"municipal_clusters_eleitores_resultado_{slice_key}",
            municipal_clusters_eleitores_resultado_sql(profile_expr, result_expr, cfg),
            chunk_output(analyses / "municipal" / "clusters_eleitores_resultado", slice_key),
            partition_by=MUNICIPIO_PARTITION_COLS,
        ))
    if ouro_task_enabled(cfg, "resultado_candidato"):
        tasks.append(
            copy_task(
                f"municipal_resultado_candidato_{slice_key}",
                municipal_resultado_entidade_from_resultados_sql(result_expr, "candidato"),
                chunk_output(analyses / "municipal" / "resultado_candidato", slice_key),
                partition_by=MUNICIPIO_PARTITION_COLS,
            )
        )
    if ouro_task_enabled(cfg, "perfil_candidato"):
        tasks.append(
            copy_task(
                f"municipal_perfil_candidato_{slice_key}",
                municipal_perfil_entidade_nivel_sql(profile_expr, result_expr, "candidato"),
                chunk_output(analyses / "municipal" / "perfil_candidato", slice_key),
                partition_by=MUNICIPIO_PARTITION_COLS,
            )
        )
    outputs = execute_copy_tasks(tasks, cfg, f"{label}_municipal_{slice_key}")
    write_ouro_event(progress_dir, label, "municipal_fatia_fim", uf=uf, ano=year, fatia=slice_key, tarefas=len(tasks))
    return outputs


def run_ouro_municipios_queue(
    cfg: CleanDatabaseConfig,
    analyses: Path,
    progress_dir: Path,
    label: str,
    slice_key: str,
    uf: str,
    year: str,
    profile_expr: str,
    result_expr: str,
    municipios: list[dict[str, str]],
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    max_workers = max(1, min(OURO_MUNICIPIO_PARALLEL_MAX, int(cfg.ouro_workers or 1)))
    batch: list[tuple[int, dict[str, str]]] = []

    def flush_batch() -> None:
        nonlocal outputs, batch
        if not batch:
            return
        if len(batch) == 1 or max_workers <= 1:
            for municipio_index, municipio in batch:
                outputs.update(run_ouro_municipio_unit(
                    cfg,
                    analyses,
                    progress_dir,
                    label,
                    slice_key,
                    uf,
                    year,
                    profile_expr,
                    result_expr,
                    municipio,
                    municipio_index,
                    len(municipios),
                ))
                clean_memory()
            batch = []
            return
        logging.info(
            "Processando lote paralelo de municipios %s: %s municipio(s), max_workers=%s.",
            slice_key,
            len(batch),
            min(max_workers, len(batch)),
        )
        write_ouro_event(
            progress_dir,
            label,
            "municipios_lote_paralelo_inicio",
            uf=uf,
            ano=year,
            fatia=slice_key,
            municipios=len(batch),
            workers=min(max_workers, len(batch)),
        )
        with ThreadPoolExecutor(max_workers=min(max_workers, len(batch))) as pool:
            future_map = {
                pool.submit(
                    run_ouro_municipio_unit,
                    cfg,
                    analyses,
                    progress_dir,
                    label,
                    slice_key,
                    uf,
                    year,
                    profile_expr,
                    result_expr,
                    municipio,
                    municipio_index,
                    len(municipios),
                ): (municipio_index, municipio)
                for municipio_index, municipio in batch
            }
            for future in as_completed(future_map):
                municipio_index, municipio = future_map[future]
                try:
                    outputs.update(future.result())
                    logging.info(
                        "Municipio concluido em paralelo %s/%s [%s]: %s",
                        municipio_index,
                        len(municipios),
                        slice_key,
                        municipio.get("cd_municipio", ""),
                    )
                except Exception as exc:
                    logging.exception(
                        "Erro processando municipio em paralelo %s/%s [%s] %s: %s",
                        municipio_index,
                        len(municipios),
                        slice_key,
                        municipio.get("cd_municipio", ""),
                        exc,
                    )
                clean_memory()
        write_ouro_event(progress_dir, label, "municipios_lote_paralelo_fim", uf=uf, ano=year, fatia=slice_key, municipios=len(batch))
        batch = []

    for municipio_index, municipio in enumerate(municipios, start=1):
        if municipio_is_large(municipio):
            flush_batch()
            logging.info(
                "Municipio grande isolado %s/%s [%s]: %s %s | eleitorado=%s linhas=%s",
                municipio_index,
                len(municipios),
                slice_key,
                municipio.get("cd_municipio", ""),
                municipio.get("nm_municipio", ""),
                municipio.get("eleitorado_estimado", "0"),
                municipio.get("linhas_total", "0"),
            )
            outputs.update(run_ouro_municipio_unit(
                cfg,
                analyses,
                progress_dir,
                label,
                slice_key,
                uf,
                year,
                profile_expr,
                result_expr,
                municipio,
                municipio_index,
                len(municipios),
            ))
            clean_memory()
            continue
        batch.append((municipio_index, municipio))
        if len(batch) >= max_workers:
            flush_batch()
    flush_batch()
    return outputs


def municipio_is_large(municipio: dict[str, str]) -> bool:
    eleitorado = parse_number(municipio.get("eleitorado_estimado"))
    linhas = parse_number(municipio.get("linhas_total"))
    eleitorado_value = float(eleitorado) if pd.notna(eleitorado) else 0.0
    linhas_value = float(linhas) if pd.notna(linhas) else 0.0
    return (
        eleitorado_value >= OURO_MUNICIPIO_LARGE_ELEITORADO_THRESHOLD
        or linhas_value >= OURO_MUNICIPIO_LARGE_ROWS_THRESHOLD
    )


def run_ouro_municipio_unit(
    cfg: CleanDatabaseConfig,
    analyses: Path,
    progress_dir: Path,
    label: str,
    slice_key: str,
    uf: str,
    year: str,
    profile_expr: str,
    result_expr: str,
    municipio: dict[str, str],
    municipio_index: int,
    total_municipios: int,
) -> dict[str, str]:
    cd_municipio = safe_text(municipio.get("cd_municipio"), "SEM_VALOR") or "SEM_VALOR"
    nm_municipio = safe_text(municipio.get("nm_municipio"), "") or ""
    municipio_key = municipio_slice_key(slice_key, cd_municipio)
    municipio_label = f"{label}_municipal_{municipio_key}"
    municipio_profile_expr = filter_municipio_expr(profile_expr, cd_municipio)
    municipio_result_expr = filter_municipio_expr(result_expr, cd_municipio)
    logging.info(
        "Ouro municipal %s/%s [%s]: %s %s municipio=%s %s",
        municipio_index,
        total_municipios,
        slice_key,
        uf,
        year,
        cd_municipio,
        nm_municipio,
    )
    write_ouro_event(
        progress_dir,
        label,
        "municipio_inicio",
        uf=uf,
        ano=year,
        fatia=slice_key,
        cd_municipio=cd_municipio,
        nm_municipio=nm_municipio,
        indice=municipio_index,
        total=total_municipios,
    )
    tasks: list[dict[str, Any]] = []
    if ouro_task_enabled(cfg, "resumo"):
        tasks.append(copy_task(
            f"municipal_resumo_{municipio_key}",
            municipal_resumo_from_prata_minima_sql(municipio_profile_expr, municipio_result_expr),
            municipio_chunk_output(analyses / "municipal" / "resumo", slice_key, cd_municipio),
            partition_by=MUNICIPIO_PARTITION_COLS,
        ))
    if ouro_task_enabled(cfg, "perfil_eleitor"):
        tasks.append(copy_task(
            f"municipal_perfil_eleitor_{municipio_key}",
            municipal_perfil_eleitor_from_prata_minima_sql(municipio_profile_expr),
            municipio_chunk_output(analyses / "municipal" / "perfil_eleitor", slice_key, cd_municipio),
            partition_by=MUNICIPIO_PARTITION_COLS,
        ))
    if ouro_task_enabled(cfg, "resultado_partido"):
        tasks.append(copy_task(
            f"municipal_resultado_partido_{municipio_key}",
            municipal_resultado_entidade_from_resultados_sql(municipio_result_expr, "partido"),
            municipio_chunk_output(analyses / "municipal" / "resultado_partido", slice_key, cd_municipio),
            partition_by=MUNICIPIO_PARTITION_COLS,
        ))
    if ouro_task_enabled(cfg, "perfil_partido"):
        tasks.append(copy_task(
            f"municipal_perfil_partido_{municipio_key}",
            municipal_perfil_entidade_nivel_sql(municipio_profile_expr, municipio_result_expr, "partido"),
            municipio_chunk_output(analyses / "municipal" / "perfil_partido", slice_key, cd_municipio),
            partition_by=MUNICIPIO_PARTITION_COLS,
        ))
    if ouro_task_enabled(cfg, "clusters_eleitores"):
        tasks.append(copy_task(
            f"municipal_clusters_eleitores_{municipio_key}",
            municipal_clusters_eleitores_sql(municipio_profile_expr, cfg),
            municipio_chunk_output(analyses / "municipal" / "clusters_eleitores", slice_key, cd_municipio),
            partition_by=MUNICIPIO_PARTITION_COLS,
        ))
    if ouro_task_enabled(cfg, "clusters_eleitores_resultado"):
        tasks.append(copy_task(
            f"municipal_clusters_eleitores_resultado_{municipio_key}",
            municipal_clusters_eleitores_resultado_sql(municipio_profile_expr, municipio_result_expr, cfg),
            municipio_chunk_output(analyses / "municipal" / "clusters_eleitores_resultado", slice_key, cd_municipio),
            partition_by=MUNICIPIO_PARTITION_COLS,
        ))
    if ouro_task_enabled(cfg, "resultado_candidato"):
        tasks.append(
            copy_task(
                f"municipal_resultado_candidato_{municipio_key}",
                municipal_resultado_entidade_from_resultados_sql(municipio_result_expr, "candidato"),
                municipio_chunk_output(analyses / "municipal" / "resultado_candidato", slice_key, cd_municipio),
                partition_by=MUNICIPIO_PARTITION_COLS,
            )
        )
    if ouro_task_enabled(cfg, "perfil_candidato"):
        tasks.append(
            copy_task(
                f"municipal_perfil_candidato_{municipio_key}",
                municipal_perfil_entidade_nivel_sql(municipio_profile_expr, municipio_result_expr, "candidato"),
                municipio_chunk_output(analyses / "municipal" / "perfil_candidato", slice_key, cd_municipio),
                partition_by=MUNICIPIO_PARTITION_COLS,
            )
        )
    outputs = execute_copy_tasks(tasks, cfg, municipio_label)
    write_ouro_event(
        progress_dir,
        label,
        "municipio_fim",
        uf=uf,
        ano=year,
        fatia=slice_key,
        cd_municipio=cd_municipio,
        tarefas=len(tasks),
    )
    return outputs


def remove_legacy_municipal_slice_outputs(analyses: Path, slice_key: str) -> None:
    roots = [
        analyses / "municipal" / "resumo",
        analyses / "municipal" / "perfil_eleitor",
        analyses / "municipal" / "resultado_partido",
        analyses / "municipal" / "perfil_partido",
        analyses / "municipal" / "clusters_eleitores",
        analyses / "municipal" / "clusters_eleitores_resultado",
        analyses / "municipal" / "resultado_candidato",
        analyses / "municipal" / "perfil_candidato",
    ]
    for root in roots:
        legacy = chunk_output(root, slice_key)
        if not legacy.exists():
            continue
        try:
            if legacy.is_dir():
                shutil.rmtree(legacy)
            else:
                legacy.unlink()
            logging.info("Removido chunk municipal legado por UF: %s", legacy)
        except Exception as exc:
            logging.warning("Nao consegui remover chunk municipal legado %s: %s", legacy, exc)


def run_ouro_estadual_uf(
    cfg: CleanDatabaseConfig,
    analyses: Path,
    progress_dir: Path,
    label: str,
    uf: str,
    uf_index: int,
    total_ufs: int,
) -> dict[str, str]:
    logging.info("Fechando ouro estadual UF %s/%s: %s a partir das analises municipais.", uf_index, total_ufs, uf)
    write_ouro_event(progress_dir, label, "estadual_inicio", uf=uf, indice_uf=uf_index, total_ufs=total_ufs)
    outputs: dict[str, str] = {}
    state_tasks: list[dict[str, Any]] = []
    resumo_root = analyses / "municipal" / "resumo"
    perfil_root = analyses / "municipal" / "perfil_eleitor"
    resultado_partido_root = analyses / "municipal" / "resultado_partido"
    perfil_partido_root = analyses / "municipal" / "perfil_partido"
    clusters_eleitores_root = analyses / "municipal" / "clusters_eleitores"
    clusters_resultado_root = analyses / "municipal" / "clusters_eleitores_resultado"
    if ouro_task_enabled(cfg, "resumo") and parquet_dataset_exists(resumo_root):
        state_tasks.append(copy_task(f"estadual_resumo_{uf}", aggregate_resumo_level_sql(filter_uf_expr(dataset_expr(resumo_root), uf), "estado"), chunk_output(analyses / "estadual" / "resumo", uf), partition_by=YEAR_UF_PARTITION_COLS))
    if ouro_task_enabled(cfg, "perfil_eleitor") and parquet_dataset_exists(perfil_root):
        state_tasks.append(copy_task(f"estadual_perfil_eleitor_{uf}", aggregate_perfil_eleitor_level_sql(filter_uf_expr(dataset_expr(perfil_root), uf), "estado"), chunk_output(analyses / "estadual" / "perfil_eleitor", uf), partition_by=YEAR_UF_PARTITION_COLS))
    if ouro_task_enabled(cfg, "resultado_partido") and parquet_dataset_exists(resultado_partido_root):
        state_tasks.append(copy_task(f"estadual_resultado_partido_{uf}", aggregate_resultado_entidade_level_sql(filter_uf_expr(dataset_expr(resultado_partido_root), uf), "estado"), chunk_output(analyses / "estadual" / "resultado_partido", uf), partition_by=YEAR_UF_PARTITION_COLS))
    if ouro_task_enabled(cfg, "perfil_partido") and parquet_dataset_exists(perfil_partido_root):
        state_tasks.append(copy_task(f"estadual_perfil_partido_{uf}", aggregate_perfil_entidade_level_sql(filter_uf_expr(dataset_expr(perfil_partido_root), uf), "estado", "partido"), chunk_output(analyses / "estadual" / "perfil_partido", uf), partition_by=YEAR_UF_PARTITION_COLS))
    if ouro_task_enabled(cfg, "clusters_eleitores") and parquet_dataset_exists(clusters_eleitores_root):
        state_tasks.append(copy_task(f"estadual_clusters_eleitores_{uf}", aggregate_clusters_level_sql(filter_uf_expr(dataset_expr(clusters_eleitores_root), uf), "estado"), chunk_output(analyses / "estadual" / "clusters_eleitores", uf), partition_by=YEAR_UF_PARTITION_COLS))
    if ouro_task_enabled(cfg, "clusters_eleitores_resultado") and parquet_dataset_exists(clusters_resultado_root):
        state_tasks.append(copy_task(f"estadual_clusters_eleitores_resultado_{uf}", aggregate_clusters_level_sql(filter_uf_expr(dataset_expr(clusters_resultado_root), uf), "estado"), chunk_output(analyses / "estadual" / "clusters_eleitores_resultado", uf), partition_by=YEAR_UF_PARTITION_COLS))
    resultado_candidato_root = analyses / "municipal" / "resultado_candidato"
    perfil_candidato_root = analyses / "municipal" / "perfil_candidato"
    if ouro_task_enabled(cfg, "resultado_candidato") and parquet_dataset_exists(resultado_candidato_root):
        state_tasks.append(copy_task(f"estadual_resultado_candidato_{uf}", aggregate_resultado_entidade_level_sql(filter_uf_expr(dataset_expr(resultado_candidato_root), uf), "estado"), chunk_output(analyses / "estadual" / "resultado_candidato", uf), partition_by=YEAR_UF_PARTITION_COLS))
    if ouro_task_enabled(cfg, "perfil_candidato") and parquet_dataset_exists(perfil_candidato_root):
        state_tasks.append(copy_task(f"estadual_perfil_candidato_{uf}", aggregate_perfil_entidade_level_sql(filter_uf_expr(dataset_expr(perfil_candidato_root), uf), "estado", "candidato"), chunk_output(analyses / "estadual" / "perfil_candidato", uf), partition_by=YEAR_UF_PARTITION_COLS))
    outputs.update(execute_copy_tasks(state_tasks, cfg, f"{label}_estadual_{uf}"))
    write_ouro_event(progress_dir, label, "estadual_fim", uf=uf, tarefas=len(state_tasks))
    return outputs


def run_ouro_brasil_final(
    cfg: CleanDatabaseConfig,
    analyses: Path,
    progress_dir: Path,
    label: str,
) -> dict[str, str]:
    logging.info("Fechando ouro Brasil a partir das analises estaduais.")
    write_ouro_event(progress_dir, label, "brasil_inicio")
    tasks: list[dict[str, Any]] = []
    roots = {
        "resumo": analyses / "estadual" / "resumo",
        "perfil_eleitor": analyses / "estadual" / "perfil_eleitor",
        "resultado_partido": analyses / "estadual" / "resultado_partido",
        "perfil_partido": analyses / "estadual" / "perfil_partido",
        "clusters_eleitores": analyses / "estadual" / "clusters_eleitores",
        "clusters_eleitores_resultado": analyses / "estadual" / "clusters_eleitores_resultado",
        "resultado_candidato": analyses / "estadual" / "resultado_candidato",
        "perfil_candidato": analyses / "estadual" / "perfil_candidato",
    }
    if ouro_task_enabled(cfg, "resumo") and parquet_dataset_exists(roots["resumo"]):
        tasks.append(copy_task("brasil_resumo", aggregate_resumo_level_sql(dataset_expr(roots["resumo"]), "brasil"), analyses / "brasil" / "resumo"))
    if ouro_task_enabled(cfg, "perfil_eleitor") and parquet_dataset_exists(roots["perfil_eleitor"]):
        tasks.append(copy_task("brasil_perfil_eleitor", aggregate_perfil_eleitor_level_sql(dataset_expr(roots["perfil_eleitor"]), "brasil"), analyses / "brasil" / "perfil_eleitor"))
    if ouro_task_enabled(cfg, "resultado_partido") and parquet_dataset_exists(roots["resultado_partido"]):
        tasks.append(copy_task("brasil_resultado_partido", aggregate_resultado_entidade_level_sql(dataset_expr(roots["resultado_partido"]), "brasil"), analyses / "brasil" / "resultado_partido"))
    if ouro_task_enabled(cfg, "perfil_partido") and parquet_dataset_exists(roots["perfil_partido"]):
        tasks.append(copy_task("brasil_perfil_partido", aggregate_perfil_entidade_level_sql(dataset_expr(roots["perfil_partido"]), "brasil", "partido"), analyses / "brasil" / "perfil_partido"))
    if ouro_task_enabled(cfg, "clusters_eleitores") and parquet_dataset_exists(roots["clusters_eleitores"]):
        tasks.append(copy_task("brasil_clusters_eleitores", aggregate_clusters_level_sql(dataset_expr(roots["clusters_eleitores"]), "brasil"), analyses / "brasil" / "clusters_eleitores"))
    if ouro_task_enabled(cfg, "clusters_eleitores_resultado") and parquet_dataset_exists(roots["clusters_eleitores_resultado"]):
        tasks.append(copy_task("brasil_clusters_eleitores_resultado", aggregate_clusters_level_sql(dataset_expr(roots["clusters_eleitores_resultado"]), "brasil"), analyses / "brasil" / "clusters_eleitores_resultado"))
    if ouro_task_enabled(cfg, "resultado_candidato") and parquet_dataset_exists(roots["resultado_candidato"]):
        tasks.append(copy_task("brasil_resultado_candidato", aggregate_resultado_entidade_level_sql(dataset_expr(roots["resultado_candidato"]), "brasil"), analyses / "brasil" / "resultado_candidato"))
    if ouro_task_enabled(cfg, "perfil_candidato") and parquet_dataset_exists(roots["perfil_candidato"]):
        tasks.append(copy_task("brasil_perfil_candidato", aggregate_perfil_entidade_level_sql(dataset_expr(roots["perfil_candidato"]), "brasil", "candidato"), analyses / "brasil" / "perfil_candidato"))
    outputs = execute_copy_tasks(tasks, cfg, f"{label}_brasil")
    write_ouro_event(progress_dir, label, "brasil_fim", tarefas=len(tasks))
    return outputs


def run_ouro_compatibilidade_final(
    cfg: CleanDatabaseConfig,
    analyses: Path,
    progress_dir: Path,
    label: str,
) -> dict[str, str]:
    logging.info("Gerando visoes de compatibilidade a partir do ouro nivelado reduzido.")
    write_ouro_event(progress_dir, label, "compatibilidade_inicio")
    tasks: list[dict[str, Any]] = []
    municipal_resumo = analyses / "municipal" / "resumo"
    estadual_resumo = analyses / "estadual" / "resumo"
    brasil_resumo = analyses / "brasil" / "resumo"
    municipal_perfil = analyses / "municipal" / "perfil_eleitor"
    estadual_perfil = analyses / "estadual" / "perfil_eleitor"
    brasil_perfil = analyses / "brasil" / "perfil_eleitor"
    municipal_resultado_eleitorado = analyses / "municipal" / "resultado_eleitorado_por_secao"
    municipal_partido = analyses / "municipal" / "perfil_partido"
    estadual_partido = analyses / "estadual" / "perfil_partido"
    brasil_partido = analyses / "brasil" / "perfil_partido"
    if parquet_dataset_exists(municipal_resumo):
        tasks.extend([
            copy_task("compat_retrato_municipal", compat_retrato_municipal_sql(dataset_expr(municipal_resumo)), analyses / "retrato_municipal", partition_by=MUNICIPIO_PARTITION_COLS),
            copy_task("compat_timeline_municipal", compat_timeline_municipal_sql(dataset_expr(municipal_resumo)), analyses / "timeline_municipal", partition_by=MUNICIPIO_PARTITION_COLS),
        ])
    if parquet_dataset_exists(estadual_resumo):
        tasks.append(copy_task("compat_timeline_uf", compat_timeline_uf_sql(dataset_expr(estadual_resumo)), analyses / "timeline_uf", partition_by=YEAR_UF_PARTITION_COLS))
    if parquet_dataset_exists(brasil_resumo):
        tasks.append(copy_task("compat_timeline_nacional", compat_timeline_nacional_sql(dataset_expr(brasil_resumo)), analyses / "timeline_nacional.parquet"))
    if parquet_dataset_exists(brasil_perfil):
        tasks.append(copy_task("compat_perfil_eleitor_por_ano", perfil_eleitor_por_ano_from_nivelado_sql(dataset_expr(brasil_perfil)), analyses / "perfil_eleitor_por_ano"))
    if parquet_dataset_exists(municipal_perfil) or parquet_dataset_exists(estadual_perfil) or parquet_dataset_exists(brasil_perfil):
        tasks.append(copy_task(
            "compat_top10_perfis",
            top10_perfis_nivelados_sql(
                dataset_expr(municipal_perfil) if parquet_dataset_exists(municipal_perfil) else "",
                dataset_expr(estadual_perfil) if parquet_dataset_exists(estadual_perfil) else "",
                dataset_expr(brasil_perfil) if parquet_dataset_exists(brasil_perfil) else "",
            ),
            analyses / "top10_perfis_federacao_estado_municipio",
            partition_by=ENTITY_PARTITION_COLS,
        ))
    if parquet_dataset_exists(municipal_resultado_eleitorado):
        tasks.append(copy_task("compat_resultado_eleitorado_por_secao", f"select * from {dataset_expr(municipal_resultado_eleitorado)}", analyses / "resultado_eleitorado_por_secao", partition_by=SECTION_PARTITION_COLS))
    if ouro_task_enabled(cfg, "perfil_partido") and (parquet_dataset_exists(municipal_partido) or parquet_dataset_exists(estadual_partido) or parquet_dataset_exists(brasil_partido)):
        partido_sql = perfil_entidade_union_nivelado_sql(
            dataset_expr(municipal_partido) if parquet_dataset_exists(municipal_partido) else "",
            dataset_expr(estadual_partido) if parquet_dataset_exists(estadual_partido) else "",
            dataset_expr(brasil_partido) if parquet_dataset_exists(brasil_partido) else "",
            "partido",
        )
        tasks.extend([
            copy_task("compat_perfil_eleitor_por_partido", partido_sql, analyses / "perfil_eleitor_por_partido", partition_by=ENTITY_PARTITION_COLS),
            copy_task("compat_comparativo_anual_perfil_partido", partido_sql, analyses / "comparativo_anual_perfil_partido", partition_by=ENTITY_PARTITION_COLS),
        ])
    municipal_candidato = analyses / "municipal" / "perfil_candidato"
    estadual_candidato = analyses / "estadual" / "perfil_candidato"
    brasil_candidato = analyses / "brasil" / "perfil_candidato"
    if ouro_task_enabled(cfg, "perfil_candidato") and (parquet_dataset_exists(municipal_candidato) or parquet_dataset_exists(estadual_candidato) or parquet_dataset_exists(brasil_candidato)):
        candidato_sql = perfil_entidade_union_nivelado_sql(
            dataset_expr(municipal_candidato) if parquet_dataset_exists(municipal_candidato) else "",
            dataset_expr(estadual_candidato) if parquet_dataset_exists(estadual_candidato) else "",
            dataset_expr(brasil_candidato) if parquet_dataset_exists(brasil_candidato) else "",
            "candidato",
        )
        tasks.extend([
            copy_task("compat_perfil_eleitor_por_candidato", candidato_sql, analyses / "perfil_eleitor_por_candidato", partition_by=ENTITY_PARTITION_COLS),
            copy_task("compat_comparativo_anual_perfil_candidato", candidato_sql, analyses / "comparativo_anual_perfil_candidato", partition_by=ENTITY_PARTITION_COLS),
        ])
    outputs = execute_copy_tasks(tasks, cfg, f"{label}_compatibilidade")
    write_ouro_event(progress_dir, label, "compatibilidade_fim", tarefas=len(tasks))
    return outputs


def run_correlacoes_ouro_partitioned(cfg: CleanDatabaseConfig, prata: Path, analyses: Path) -> dict[str, str]:
    label = "ouro_correlacoes"
    progress_dir = prepare_ouro_progress(cfg, label)
    write_ouro_event(progress_dir, label, "inicio_etapa", etapa=label)
    targets = [
        analyses / "base_gold_global",
        analyses / "resultado_eleitorado_por_secao",
        analyses / "perfil_eleitor_por_partido",
        analyses / "comparativo_anual_perfil_partido",
        analyses / "perfil_eleitor_por_candidato",
        analyses / "comparativo_anual_perfil_candidato",
        analyses / "_work" / "perfil_eleitor_por_partido_parts",
        analyses / "_work" / "perfil_eleitor_por_partido_estado_source_parts",
        analyses / "_work" / "perfil_eleitor_por_partido_top_local_parts",
        analyses / "_work" / "perfil_eleitor_por_candidato_parts",
        analyses / "_work" / "perfil_eleitor_por_candidato_estado_source_parts",
        analyses / "_work" / "perfil_eleitor_por_candidato_top_local_parts",
    ]
    reset_ouro_targets(targets, resume=cfg.resume)

    outputs: dict[str, str] = {}
    uf_parts = [(uf, e_path) for uf, e_path in list_uf_partition_dirs(prata / "eleitorado") if parquet_dataset_exists(prata / "resultados_votos" / f"uf={uf}")]
    logging.info("Gerando %s por UF: %s particoes com eleitorado+resultados.", label, len(uf_parts))
    correlacao_plan = build_correlacao_uf_year_plan(cfg, prata, uf_parts)
    if prata_minima_plan_ready(cfg, correlacao_plan):
        logging.info("Prata_minima ja pronta; %s vai apenas montar analises municipais/estaduais/Brasil.", label)
    else:
        logging.warning("Prata_minima incompleta ao iniciar %s; materializando pecas faltantes como fallback.", label)
        outputs.update(materialize_prata_minima_correlacoes(cfg, analyses, correlacao_plan, progress_dir, label))

    for index, (uf, e_path) in enumerate(uf_parts, start=1):
        write_ouro_event(progress_dir, label, "processando_uf", etapa=label, indice_uf=index, total_ufs=len(uf_parts), uf=uf, entrada=str(e_path))
        r_path = prata / "resultados_votos" / f"uf={uf}"
        years = sort_years(set(list_years_for_dataset(e_path, cfg)) & set(list_years_for_dataset(r_path, cfg)))
        logging.info("Ouro correlacoes UF %s/%s: %s | anos=%s", index, len(uf_parts), uf, ", ".join(years) or "sem_ano")
        uf_barrier_tasks: list[dict[str, Any]] = []
        for year_index, year in enumerate(years, start=1):
            slice_key = slice_name(uf, year)
            logging.info("Ouro correlacoes fatia %s/%s UF %s ano %s", year_index, len(years), uf, year)
            write_ouro_event(
                progress_dir,
                label,
                "processando_fatia",
                etapa=label,
                uf=uf,
                ano=year,
                indice_uf=index,
                total_ufs=len(uf_parts),
                indice_ano=year_index,
                total_anos=len(years),
            )
            save_json(
                {
                    "label": label,
                    "status": "processando_analises",
                    "uf_atual": uf,
                    "ano_atual": year,
                    "indice_uf": index,
                    "total_ufs": len(uf_parts),
                    "indice_ano": year_index,
                    "total_anos_uf": len(years),
                },
                progress_dir / f"{label}_progresso.json",
            )
            winners_path = chunk_output(analyses / "resultados_vencedores_secao", slice_key)
            profile_cache_path = prata_minima_chunk(cfg, "perfil_secao", slice_key)
            result_cache_path = prata_minima_chunk(cfg, "resultado_secao", slice_key)
            if not output_has_data(profile_cache_path) or not output_has_data(result_cache_path):
                logging.error("Prata minima ausente em %s; analises municipais/estaduais desta fatia serao puladas.", slice_key)
                write_ouro_event(
                    progress_dir,
                    label,
                    "prata_minima_ausente_na_analise",
                    etapa=label,
                    fatia=slice_key,
                    perfil_cache=str(profile_cache_path),
                    resultado_cache=str(result_cache_path),
                )
                continue
            profile_cache_expr = dataset_expr(profile_cache_path)
            result_cache_expr = dataset_expr(result_cache_path)
            partido_parts_path = chunk_output(analyses / "_work" / "perfil_eleitor_por_partido_parts", slice_key)
            partido_parts_expr = dataset_expr(partido_parts_path)
            candidato_parts_path = chunk_output(analyses / "_work" / "perfil_eleitor_por_candidato_parts", slice_key)
            candidato_parts_expr = dataset_expr(candidato_parts_path)

            tasks = [
                copy_task(
                    f"base_gold_global_{slice_key}",
                    base_gold_global_from_cache_sql(profile_cache_expr, result_cache_expr),
                    chunk_output(analyses / "base_gold_global", slice_key),
                    partition_by=ZONE_PARTITION_COLS,
                ),
                copy_task(
                    f"perfil_eleitor_por_partido_parts_{slice_key}",
                    perfil_entidade_parts_from_cache_sql(profile_cache_expr, result_cache_expr, "partido"),
                    partido_parts_path,
                    partition_by=MUNICIPIO_PARTITION_COLS,
                ),
            ]
            if parquet_dataset_exists(winners_path):
                tasks.insert(1, copy_task(
                    f"resultado_eleitorado_por_secao_{slice_key}",
                    resultado_eleitorado_secao_from_cache_sql(profile_cache_expr, winners_path),
                    chunk_output(analyses / "resultado_eleitorado_por_secao", slice_key),
                    partition_by=SECTION_PARTITION_COLS,
                ))
            else:
                logging.warning("Pulando resultado_eleitorado_por_secao %s: vencedores da fatia ainda nao existem.", slice_key)
            if not cfg.skip_heavy_analyses:
                tasks.append(copy_task(
                    f"perfil_eleitor_por_candidato_parts_{slice_key}",
                    perfil_entidade_parts_from_cache_sql(profile_cache_expr, result_cache_expr, "candidato"),
                    candidato_parts_path,
                    partition_by=MUNICIPIO_PARTITION_COLS,
                ))

            post_tasks = [
                copy_task(
                    f"perfil_eleitor_por_partido_estado_source_parts_{slice_key}",
                    perfil_entidade_estado_source_sql(partido_parts_expr, "partido"),
                    chunk_output(analyses / "_work" / "perfil_eleitor_por_partido_estado_source_parts", slice_key),
                    partition_by=["ano", "uf"],
                ),
                copy_task(
                    f"perfil_eleitor_por_partido_top_local_parts_{slice_key}",
                    perfil_entidade_top_local_sql(partido_parts_expr, "partido"),
                    chunk_output(analyses / "_work" / "perfil_eleitor_por_partido_top_local_parts", slice_key),
                    partition_by=ENTITY_PARTITION_COLS,
                ),
            ]
            if not cfg.skip_heavy_analyses:
                post_tasks.extend([
                    copy_task(
                        f"perfil_eleitor_por_candidato_estado_source_parts_{slice_key}",
                        perfil_entidade_estado_source_sql(candidato_parts_expr, "candidato"),
                        chunk_output(analyses / "_work" / "perfil_eleitor_por_candidato_estado_source_parts", slice_key),
                        partition_by=["ano", "uf"],
                    ),
                    copy_task(
                        f"perfil_eleitor_por_candidato_top_local_parts_{slice_key}",
                        perfil_entidade_top_local_sql(candidato_parts_expr, "candidato"),
                        chunk_output(analyses / "_work" / "perfil_eleitor_por_candidato_top_local_parts", slice_key),
                        partition_by=ENTITY_PARTITION_COLS,
                    ),
                ])

            all_slice_tasks = [*tasks, *post_tasks]
            uf_barrier_tasks.extend(all_slice_tasks)
            if cfg.resume and all(ouro_task_done(task, progress_dir) for task in all_slice_tasks):
                logging.info("Pulando fatia ouro_correlacoes ja concluida: %s", slice_key)
                outputs.update({task["name"]: str(task["out"]) for task in all_slice_tasks})
                continue

            base_tasks_done = cfg.resume and all(ouro_task_done(task, progress_dir) for task in tasks)
            if base_tasks_done:
                logging.info("Reaproveitando partes base ja concluidas em %s; gerando somente analises locais/estaduais faltantes.", slice_key)
                outputs.update({task["name"]: str(task["out"]) for task in tasks})
                ready_post_tasks = ready_correlacao_post_tasks(post_tasks, partido_parts_path, candidato_parts_path, cfg)
                outputs.update(execute_copy_tasks(ready_post_tasks, cfg, label))
                clean_memory()
                continue

            base_outputs = execute_copy_tasks(tasks, cfg, label)
            outputs.update(base_outputs)
            ready_post_tasks = ready_correlacao_post_tasks(post_tasks, partido_parts_path, candidato_parts_path, cfg)
            outputs.update(execute_copy_tasks(ready_post_tasks, cfg, label))
            clean_memory()

        if uf_barrier_tasks and all(ouro_task_done(task, progress_dir) for task in uf_barrier_tasks):
            marker = progress_dir / f"{safe_name(label + '_municipal_estadual_' + uf, 80)}.done.json"
            payload = {
                "label": label,
                "uf": uf,
                "status": "ok",
                "garantia": "Todas as analises municipais/estaduais desta UF foram concluidas antes de avancar.",
                "anos": years,
                "tarefas": [task.get("name", "") for task in uf_barrier_tasks],
            }
            save_json(payload, marker)
            logging.info(
                "UF %s/%s completa em %s: analises municipais e estaduais concluidas para %s.",
                index,
                len(uf_parts),
                label,
                uf,
            )
            write_ouro_event(progress_dir, label, "uf_municipal_estadual_completa", etapa=label, uf=uf, indice_uf=index, total_ufs=len(uf_parts), anos=years)
        else:
            pendentes = [task.get("name", "") for task in uf_barrier_tasks if not ouro_task_done(task, progress_dir)]
            logging.error("UF %s ainda tem analises municipais/estaduais pendentes: %s", uf, ", ".join(pendentes[:20]))
            write_ouro_event(progress_dir, label, "uf_municipal_estadual_pendente", etapa=label, uf=uf, pendentes=pendentes[:200])

    final_tasks = []
    partido_state_root = analyses / "_work" / "perfil_eleitor_por_partido_estado_source_parts"
    partido_local_root = analyses / "_work" / "perfil_eleitor_por_partido_top_local_parts"
    if parquet_dataset_exists(partido_state_root) and parquet_dataset_exists(partido_local_root):
        partido_state_expr = dataset_expr(partido_state_root)
        partido_local_expr = dataset_expr(partido_local_root)
        final_tasks.append(
            copy_task("perfil_eleitor_por_partido", perfil_entidade_final_from_state_sql(partido_state_expr, partido_local_expr, "partido"), analyses / "perfil_eleitor_por_partido", partition_by=ENTITY_PARTITION_COLS)
        )
    elif parquet_dataset_exists(analyses / "_work" / "perfil_eleitor_por_partido_parts"):
        partido_parts_expr = dataset_expr(analyses / "_work" / "perfil_eleitor_por_partido_parts")
        final_tasks.append(
            copy_task("perfil_eleitor_por_partido", perfil_entidade_final_sql(partido_parts_expr, "partido"), analyses / "perfil_eleitor_por_partido", partition_by=ENTITY_PARTITION_COLS)
        )
    candidato_state_root = analyses / "_work" / "perfil_eleitor_por_candidato_estado_source_parts"
    candidato_local_root = analyses / "_work" / "perfil_eleitor_por_candidato_top_local_parts"
    if not cfg.skip_heavy_analyses and parquet_dataset_exists(candidato_state_root) and parquet_dataset_exists(candidato_local_root):
        candidato_state_expr = dataset_expr(candidato_state_root)
        candidato_local_expr = dataset_expr(candidato_local_root)
        final_tasks.append(
            copy_task("perfil_eleitor_por_candidato", perfil_entidade_final_from_state_sql(candidato_state_expr, candidato_local_expr, "candidato"), analyses / "perfil_eleitor_por_candidato", partition_by=ENTITY_PARTITION_COLS)
        )
    elif not cfg.skip_heavy_analyses and parquet_dataset_exists(analyses / "_work" / "perfil_eleitor_por_candidato_parts"):
        candidato_parts_expr = dataset_expr(analyses / "_work" / "perfil_eleitor_por_candidato_parts")
        final_tasks.append(
            copy_task("perfil_eleitor_por_candidato", perfil_entidade_final_sql(candidato_parts_expr, "candidato"), analyses / "perfil_eleitor_por_candidato", partition_by=ENTITY_PARTITION_COLS)
        )
    for task in final_tasks:
        outputs[task["name"]] = execute_ouro_task(task, cfg, label, progress_dir)
        clean_memory()
    if "perfil_eleitor_por_partido" in outputs:
        outputs["comparativo_anual_perfil_partido"] = outputs["perfil_eleitor_por_partido"]
        logging.info("Comparativo anual por partido reaproveita perfil_eleitor_por_partido; sem segunda varredura.")
    if "perfil_eleitor_por_candidato" in outputs:
        outputs["comparativo_anual_perfil_candidato"] = outputs["perfil_eleitor_por_candidato"]
        logging.info("Comparativo anual por candidato reaproveita perfil_eleitor_por_candidato; sem segunda varredura.")

    save_json({"label": label, "status": "ok", "outputs": outputs}, progress_dir / f"{label}_progresso.json")
    write_ouro_event(progress_dir, label, "fim_etapa", etapa=label, outputs=outputs)
    return outputs


def run_candidatos_ouro_partitioned(cfg: CleanDatabaseConfig, prata: Path, analyses: Path) -> dict[str, str]:
    label = "ouro_candidatos"
    progress_dir = prepare_ouro_progress(cfg, label)
    write_ouro_event(progress_dir, label, "inicio_etapa", etapa=label)
    reset_ouro_targets([analyses / "perfil_candidatos"], resume=cfg.resume)
    outputs: dict[str, str] = {}
    uf_parts = list_uf_partition_dirs(prata / "candidatos")
    logging.info("Gerando %s por UF: %s particoes de candidatos.", label, len(uf_parts))
    for index, (uf, c_path) in enumerate(uf_parts, start=1):
        write_ouro_event(progress_dir, label, "processando_uf", etapa=label, indice_uf=index, total_ufs=len(uf_parts), uf=uf, entrada=str(c_path))
        years = list_years_for_dataset(c_path, cfg)
        logging.info("Ouro candidatos UF %s/%s: %s | anos=%s", index, len(uf_parts), uf, ", ".join(years) or "sem_ano")
        for year in years:
            slice_key = slice_name(uf, year)
            write_ouro_event(progress_dir, label, "processando_fatia", etapa=label, uf=uf, ano=year, indice_uf=index, total_ufs=len(uf_parts), tarefa=f"perfil_candidatos_{slice_key}")
            task = copy_task(
                f"perfil_candidatos_{slice_key}",
                perfil_candidato_sql(filtered_year_expr(dataset_expr(c_path), year)),
                chunk_output(analyses / "perfil_candidatos", slice_key),
                partition_by=MUNICIPIO_PARTITION_COLS,
            )
            outputs[task["name"]] = execute_ouro_task(task, cfg, label, progress_dir)
            clean_memory()
        marker = progress_dir / f"{safe_name(label + '_uf_' + uf, 80)}.done.json"
        save_json(
            {
                "label": label,
                "uf": uf,
                "status": "ok",
                "garantia": "Perfil de candidatos desta UF concluido antes de avancar para a proxima UF.",
                "anos": years,
            },
            marker,
        )
        logging.info("UF %s/%s completa em %s: %s.", index, len(uf_parts), label, uf)
    save_json({"label": label, "status": "ok", "outputs": outputs}, progress_dir / f"{label}_progresso.json")
    write_ouro_event(progress_dir, label, "fim_etapa", etapa=label, outputs=outputs)
    return outputs


def build_correlacao_uf_year_plan(
    cfg: CleanDatabaseConfig,
    prata: Path,
    uf_parts: list[tuple[str, Path]],
) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    logging.info("Preparando fila prata_minima: %s UFs ja organizadas na prata. Os anos serao separados durante a escrita, sem pre-scan.", len(uf_parts))
    if not uf_parts:
        return []
    write_pipeline_event(
        cfg.out,
        "prata_minima",
        "fila_uf_inicio",
        ufs=len(uf_parts),
        observacao="A prata ja esta particionada por UF; a prata_minima fisica fica particionada somente por UF. Ano, tipo_documento, municipio, zona e secao ficam como colunas.",
    )
    total = len(uf_parts)
    for uf_index, (uf, e_path) in enumerate(uf_parts, start=1):
        r_path = prata / "resultados_votos" / f"uf={uf}"
        plan.append(
            {
                "uf": uf,
                "ano": "",
                "slice_key": safe_name(uf, 20) or "SEM_UF",
                "eleitorado_path": str(e_path),
                "resultados_path": str(r_path),
                "indice_uf": uf_index,
                "total_ufs": total,
                "indice_ano": 1,
                "total_anos_uf": 1,
            }
        )
        logging.info("Fila prata_minima UF %s/%s: %s | anos serao descobertos pelas linhas durante COPY.", uf_index, total, uf)
    write_pipeline_event(
        cfg.out,
        "prata_minima",
        "fila_uf_fim",
        ufs=len(uf_parts),
        tarefas=len(plan),
    )
    return plan


def prata_minima_root(cfg: CleanDatabaseConfig) -> Path:
    return cfg.out / "ouro" / "prata_minima"


def parse_slice_key(slice_key: str) -> tuple[str, str]:
    text = safe_text(slice_key, "SEM_UF_SEM_ANO") or "SEM_UF_SEM_ANO"
    if "_" not in text:
        return text or "SEM_UF", "SEM_ANO"
    uf, year = text.rsplit("_", 1)
    return uf or "SEM_UF", year or "SEM_ANO"


def hive_part(name: str, value: str) -> str:
    cleaned = safe_name(value or "SEM_VALOR", 80) or "SEM_VALOR"
    return f"{name}={cleaned}"


def prata_minima_chunk(cfg: CleanDatabaseConfig, table: str, slice_key: str) -> Path:
    return prata_minima_root(cfg)


def prata_minima_legacy_chunk(cfg: CleanDatabaseConfig, table: str, slice_key: str) -> Path:
    return chunk_output(prata_minima_root(cfg) / table, slice_key)


def prata_minima_done_marker(cfg: CleanDatabaseConfig, table: str, slice_key: str) -> Path:
    return cfg.out / "logs" / "ouro" / f"prata_minima_{safe_name(table, 40)}_{safe_name(slice_key, 80)}.done.json"


def prata_minima_skipped_marker(cfg: CleanDatabaseConfig, table: str, slice_key: str) -> Path:
    return cfg.out / "logs" / "ouro" / f"prata_minima_{safe_name(table, 40)}_{safe_name(slice_key, 80)}.skipped.json"


def prata_minima_manifest_path(cfg: CleanDatabaseConfig) -> Path:
    return cfg.out / "logs" / "ouro" / "prata_minima_manifesto.json"


def prata_minima_public_manifest_path(cfg: CleanDatabaseConfig) -> Path:
    return prata_minima_root(cfg) / "_manifesto.json"


def prata_minima_current_progress_path(cfg: CleanDatabaseConfig) -> Path:
    return prata_minima_root(cfg) / "_progresso_atual.json"


def save_prata_minima_manifest(cfg: CleanDatabaseConfig, manifest: dict[str, dict[str, Any]]) -> None:
    save_json(manifest, prata_minima_manifest_path(cfg))
    try:
        save_json(manifest, prata_minima_public_manifest_path(cfg))
    except Exception as exc:
        logging.warning("Nao consegui espelhar manifesto dentro da prata_minima: %s", exc)


def load_prata_minima_manifest(cfg: CleanDatabaseConfig) -> dict[str, dict[str, Any]]:
    path = prata_minima_manifest_path(cfg)
    if not path.exists():
        return rebuild_prata_minima_manifest_from_done_markers(cfg)
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(data, dict):
            raw_manifest = {str(key): value for key, value in data.items() if isinstance(value, dict)}
            if not prata_minima_root(cfg).exists() and raw_manifest:
                logging.warning("Manifesto prata_minima existe, mas a pasta prata_minima nao existe; ignorando manifesto antigo.")
                return {}
            manifest = {key: value for key, value in raw_manifest.items() if prata_minima_payload_layout_current(value)}
            dropped = len(raw_manifest) - len(manifest)
            if dropped:
                logging.warning(
                    "Manifesto prata_minima tinha %s entrada(s) de layout antigo/invalido; elas nao serao reaproveitadas.",
                    dropped,
                )
                save_prata_minima_manifest(cfg, manifest)
            return manifest
    except Exception as exc:
        logging.warning("Nao foi possivel ler manifesto prata_minima; ele sera recriado. Erro: %s", exc)
    return rebuild_prata_minima_manifest_from_done_markers(cfg)


def rebuild_prata_minima_manifest_from_done_markers(cfg: CleanDatabaseConfig) -> dict[str, dict[str, Any]]:
    root = prata_minima_root(cfg)
    if not root.exists():
        return {}
    progress_dir = cfg.out / "logs" / "ouro"
    if not progress_dir.exists():
        return {}
    manifest: dict[str, dict[str, Any]] = {}
    for marker in progress_dir.glob("prata_minima_*.done.json"):
        try:
            payload = json.loads(marker.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        table = safe_text(payload.get("documento", ""))
        slice_key = safe_text(payload.get("fatia", ""))
        if not table or not slice_key or not prata_minima_payload_current(payload):
            continue
        manifest[prata_minima_manifest_key(table, slice_key)] = payload
    if manifest:
        save_prata_minima_manifest(cfg, manifest)
        logging.info("Manifesto prata_minima reconstruido a partir de .done.json: %s itens.", len(manifest))
    return manifest


def prata_minima_manifest_key(table: str, slice_key: str) -> str:
    return f"{safe_name(table, 40)}::{safe_name(slice_key, 80)}"


def prata_minima_payload_layout_current(payload: dict[str, Any]) -> bool:
    return (
        int(payload.get("layout_version", 0) or 0) == PRATA_MINIMA_LAYOUT_VERSION
        and list(payload.get("partition_cols", []) or []) == PRATA_MINIMA_SECTION_PARTITION_COLS
    )


def prata_minima_payload_current(payload: dict[str, Any]) -> bool:
    return (
        safe_text(payload.get("status", "")) == "ok"
        and int(payload.get("linhas_gravadas", 0) or 0) > 0
        and prata_minima_payload_layout_current(payload)
    )


def prata_minima_manifest_status(manifest: dict[str, dict[str, Any]], table: str, slice_key: str) -> str:
    item = manifest.get(prata_minima_manifest_key(table, slice_key)) or {}
    if not prata_minima_payload_layout_current(item):
        return ""
    return safe_text(item.get("status", ""))


def prata_minima_manifest_done(manifest: dict[str, dict[str, Any]], table: str, slice_key: str) -> bool:
    item = manifest.get(prata_minima_manifest_key(table, slice_key)) or {}
    return prata_minima_payload_current(item)


def prata_minima_manifest_skipped(manifest: dict[str, dict[str, Any]], table: str, slice_key: str) -> bool:
    return prata_minima_manifest_status(manifest, table, slice_key) == "skipped"


def update_prata_minima_manifest(
    cfg: CleanDatabaseConfig,
    manifest: dict[str, dict[str, Any]],
    table: str,
    slice_key: str,
    payload: dict[str, Any],
) -> None:
    manifest[prata_minima_manifest_key(table, slice_key)] = payload
    save_prata_minima_manifest(cfg, manifest)


def prata_minima_manifest_seed_payload(
    cfg: CleanDatabaseConfig,
    label: str,
    table: str,
    slice_key: str,
    uf: str,
    year: str,
    source_path: Path,
    item_index: int,
    total_items: int,
    status: str = "pending",
) -> dict[str, Any]:
    return {
        "label": label,
        "documento": table,
        "fatia": slice_key,
        "uf": uf,
        "ano": year,
        "status": status,
        "fonte": str(source_path),
        "saida": str(prata_minima_root(cfg)),
        "indice_fatia": item_index,
        "total_fatias": total_items,
        "linhas_lidas": 0,
        "linhas_gravadas": 0,
        "batches_pulados": 0,
        "layout_version": PRATA_MINIMA_LAYOUT_VERSION,
        "partition_cols": PRATA_MINIMA_SECTION_PARTITION_COLS,
        "colunas_localizacao": ["cd_municipio", "nm_municipio", "zona", "secao"],
        "atualizado_epoch": round(time.time(), 3),
    }


def seed_prata_minima_manifest_from_plan(
    cfg: CleanDatabaseConfig,
    manifest: dict[str, dict[str, Any]],
    plan: list[dict[str, Any]],
    label: str,
) -> dict[str, dict[str, Any]]:
    total_docs = 0
    added = 0
    recovered_done = 0
    recovered_skipped = 0
    for item_index, item in enumerate(plan, start=1):
        uf = safe_text(item.get("uf", "")) or "SEM_UF"
        year = safe_text(item.get("ano", ""))
        slice_key = safe_text(item.get("slice_key", "")) or slice_name(uf, year)
        docs: list[tuple[str, Path]] = [
            ("perfil_secao", Path(safe_text(item.get("eleitorado_path", "")))),
            ("resultado_secao", Path(safe_text(item.get("resultados_path", "")))),
        ]
        c_path = cfg.out / "prata" / "candidatos" / f"uf={uf}"
        if parquet_dataset_exists(c_path):
            docs.append(("candidatos_secao", c_path))
        for table, source_path in docs:
            total_docs += 1
            key = prata_minima_manifest_key(table, slice_key)
            existing = manifest.get(key)
            if existing and prata_minima_payload_layout_current(existing):
                continue
            marker = prata_minima_done_marker(cfg, table, slice_key)
            if marker.exists():
                try:
                    payload = json.loads(marker.read_text(encoding="utf-8-sig"))
                except Exception:
                    payload = {}
                if prata_minima_payload_current(payload):
                    manifest[key] = payload
                    recovered_done += 1
                    continue
            skipped_marker = prata_minima_skipped_marker(cfg, table, slice_key)
            if skipped_marker.exists():
                try:
                    payload = json.loads(skipped_marker.read_text(encoding="utf-8-sig"))
                except Exception:
                    payload = {}
                if isinstance(payload, dict):
                    payload.update(
                        {
                            "label": payload.get("label", label),
                            "documento": table,
                            "fatia": slice_key,
                            "uf": uf,
                            "ano": year,
                            "status": "skipped",
                            "fonte": str(source_path),
                            "saida": str(prata_minima_root(cfg)),
                            "layout_version": PRATA_MINIMA_LAYOUT_VERSION,
                            "partition_cols": PRATA_MINIMA_SECTION_PARTITION_COLS,
                            "atualizado_epoch": round(time.time(), 3),
                        }
                    )
                    manifest[key] = payload
                    recovered_skipped += 1
                    continue
            manifest[key] = prata_minima_manifest_seed_payload(
                cfg,
                label,
                table,
                slice_key,
                uf,
                year,
                source_path,
                item_index,
                len(plan),
                "pending",
            )
            added += 1
    save_prata_minima_manifest(cfg, manifest)
    logging.info(
        "Manifesto prata_minima preparado: documentos=%s | adicionados_pending=%s | recuperados_ok=%s | recuperados_skipped=%s",
        total_docs,
        added,
        recovered_done,
        recovered_skipped,
    )
    return manifest


def prata_minima_table_exists(cfg: CleanDatabaseConfig, table: str, slice_key: str) -> bool:
    marker = prata_minima_done_marker(cfg, table, slice_key)
    if not marker.exists():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8-sig"))
    except Exception:
        return False
    if not prata_minima_payload_current(payload):
        return False
    root = prata_minima_chunk(cfg, table, slice_key)
    return root.exists()


def prata_minima_table_expr(path: Path, table: str, slice_key: str = "") -> str:
    conditions = [f"{discrete_sql_value('tipo_documento')} = {sql_lit(table)}"]
    read_path = path
    if slice_key:
        uf, year = parse_slice_key(slice_key)
        if uf and uf != "SEM_UF":
            conditions.append(f"{discrete_sql_value('uf')} = {sql_lit(uf)}")
            uf_path = path / f"uf={safe_name(uf, 20)}"
            if uf_path.exists():
                read_path = uf_path
        if year and year != "SEM_ANO":
            conditions.append(f"{discrete_sql_value('ano')} = {sql_lit(year)}")
    return f"(select * from {dataset_expr(read_path)} where {' and '.join(conditions)})"


def prata_minima_document_sql(inner_sql: str, table: str) -> str:
    return f"""
    select *
    from (
      select *, {sql_lit(table)} as tipo_documento
      from ({inner_sql})
    )
    """


def prata_minima_stream_batch_rows(cfg: CleanDatabaseConfig) -> int:
    configured = int(cfg.chunk_rows or PRATA_MINIMA_STREAM_BATCH_ROWS)
    return max(1_000, min(configured, PRATA_MINIMA_STREAM_BATCH_ROWS))


def prata_minima_partition_text(value: Any, col: str) -> str:
    if col == "uf":
        return clean_uf(value)
    text = compact_code(value) if col in {"cd_municipio", "zona", "secao"} else clean_value(value)
    return safe_name(text, 80) or "SEM_VALOR"


def prata_minima_prepare_partitions(df: pd.DataFrame, tipo_documento: str) -> pd.DataFrame:
    for col in ["ano", "uf", "cd_municipio", "zona", "secao"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].map(lambda value, c=col: prata_minima_partition_text(value, c))
    df["tipo_documento"] = tipo_documento
    return df


def numeric_frame_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([0.0] * len(df), index=df.index)
    return pd.to_numeric(df[col].astype(str).str.replace(",", ".", regex=False), errors="coerce").fillna(0.0)


def text_frame_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([""] * len(df), index=df.index, dtype="object")
    return df[col].map(clean_value).astype("object")


def valid_profile_text(value: Any) -> bool:
    text = clean_value(value)
    lower = text.lower()
    return bool(text) and lower not in NULL_WORDS and "sem valor" not in lower


def profile_combo_frame(df: pd.DataFrame) -> pd.Series:
    labels = {
        "perfil_faixa_etaria": "faixa_etaria",
        "perfil_genero": "sexo_genero",
        "perfil_instrucao": "escolaridade",
        "perfil_estado_civil": "estado_civil",
        "perfil_raca_cor": "raca_cor",
    }
    pieces: list[pd.Series] = []
    for col, label in labels.items():
        values = text_frame_col(df, col)
        pieces.append(values.map(lambda value, l=label: f"{l}={value}" if valid_profile_text(value) else ""))
    if not pieces:
        return pd.Series([""] * len(df), index=df.index, dtype="object")
    combo = pieces[0]
    for part in pieces[1:]:
        combo = combo.str.cat(part, sep="; ")
    return combo.str.replace(r"(; )+", "; ", regex=True).str.strip("; ")


def transform_prata_minima_batch(df: pd.DataFrame, tipo_documento: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = pd.DataFrame(index=df.index)
    base_cols = ["ano", "uf", "cd_municipio", "nm_municipio", "zona", "secao", "cargo", "turno"]
    for col in base_cols:
        out[col] = text_frame_col(df, col)
    out["nm_municipio"] = text_frame_col(df, "nm_municipio")

    if tipo_documento == "perfil_secao":
        out["local_votacao"] = text_frame_col(df, "local_votacao")
        out["bairro"] = text_frame_col(df, "bairro")
        for col in PROFILE_COLS:
            out[col] = text_frame_col(df, col)
        out["eleitorado_perfil"] = numeric_frame_col(df, "eleitorado")
        out["eleitorado"] = numeric_frame_col(df, "eleitorado")
        out["comparecimento_estimado"] = numeric_frame_col(df, "comparecimento_estimado")
        out["abstencao_estimado"] = numeric_frame_col(df, "abstencao_estimado")
    elif tipo_documento == "resultado_secao":
        out["partido"] = text_frame_col(df, "partido")
        out["candidato"] = text_frame_col(df, "candidato")
        out["nr_votavel"] = text_frame_col(df, "nr_votavel")
        out["sq_candidato"] = text_frame_col(df, "sq_candidato")
        out["votos"] = numeric_frame_col(df, "votos")
        out["brancos"] = numeric_frame_col(df, "brancos")
        out["nulos"] = numeric_frame_col(df, "nulos")
        out["validos_estimados"] = numeric_frame_col(df, "validos_estimados")
    elif tipo_documento == "candidatos_secao":
        out["partido"] = text_frame_col(df, "partido")
        out["candidato"] = text_frame_col(df, "candidato")
        out["nr_candidato"] = text_frame_col(df, "nr_candidato")
        out["sq_candidato"] = text_frame_col(df, "sq_candidato")
        out["candidato_faixa_etaria"] = text_frame_col(df, "perfil_faixa_etaria")
        out["candidato_genero"] = text_frame_col(df, "perfil_genero")
        out["candidato_instrucao"] = text_frame_col(df, "perfil_instrucao")
        out["candidato_estado_civil"] = text_frame_col(df, "perfil_estado_civil")
        out["candidato_raca_cor"] = text_frame_col(df, "perfil_raca_cor")
        out["situacao_candidatura"] = text_frame_col(df, "situacao_candidatura")
        out["resultado_candidatura"] = text_frame_col(df, "resultado_candidatura")
        out["qtd_registros_candidato"] = numeric_frame_col(df, "qtd_registros")
    else:
        raise ValueError(f"Tipo de documento prata_minima desconhecido: {tipo_documento}")

    return prata_minima_prepare_partitions(out, tipo_documento)


def transform_prata_minima_arrow_table(table: Any, tipo_documento: str) -> Any:
    try:
        return transform_prata_minima_arrow_table_polars(table, tipo_documento)
    except Exception as exc:
        logging.warning("Polars indisponivel/falhou na prata_minima; usando fallback pandas. Erro: %s", exc)
        df = table.to_pandas(split_blocks=True, self_destruct=True)
        out = transform_prata_minima_batch(df, tipo_documento)
        del df
        try:
            import pyarrow as pa
            return pa.Table.from_pandas(out, preserve_index=False)
        finally:
            del out


def transform_prata_minima_arrow_table_polars(table: Any, tipo_documento: str) -> Any:
    import polars as pl

    frame = pl.from_arrow(table)

    def txt(col: str, alias: str | None = None) -> Any:
        name = alias or col
        if col not in frame.columns:
            return pl.lit("").alias(name)
        return pl.col(col).map_elements(clean_value, return_dtype=pl.Utf8).alias(name)

    def part(col: str) -> Any:
        if col not in frame.columns:
            return pl.lit("SEM_VALOR" if col != "uf" else "SEM_UF").alias(col)
        return pl.col(col).map_elements(lambda value, c=col: prata_minima_partition_text(value, c), return_dtype=pl.Utf8).alias(col)

    def num(col: str, alias: str | None = None) -> Any:
        name = alias or col
        if col not in frame.columns:
            return pl.lit(0.0).alias(name)
        return (
            pl.col(col)
            .cast(pl.Utf8, strict=False)
            .str.replace(",", ".", literal=True)
            .cast(pl.Float64, strict=False)
            .fill_null(0.0)
            .alias(name)
        )

    exprs: list[Any] = [
        part("ano"),
        part("uf"),
        part("cd_municipio"),
        txt("nm_municipio"),
        part("zona"),
        part("secao"),
        txt("cargo"),
        txt("turno"),
    ]
    if tipo_documento == "perfil_secao":
        exprs.extend([
            txt("local_votacao"),
            txt("bairro"),
            *(txt(col) for col in PROFILE_COLS),
            num("eleitorado", "eleitorado_perfil"),
            num("eleitorado"),
            num("comparecimento_estimado"),
            num("abstencao_estimado"),
        ])
    elif tipo_documento == "resultado_secao":
        exprs.extend([
            txt("partido"),
            txt("candidato"),
            txt("nr_votavel"),
            txt("sq_candidato"),
            num("votos"),
            num("brancos"),
            num("nulos"),
            num("validos_estimados"),
        ])
    elif tipo_documento == "candidatos_secao":
        exprs.extend([
            txt("partido"),
            txt("candidato"),
            txt("nr_candidato"),
            txt("sq_candidato"),
            txt("perfil_faixa_etaria", "candidato_faixa_etaria"),
            txt("perfil_genero", "candidato_genero"),
            txt("perfil_instrucao", "candidato_instrucao"),
            txt("perfil_estado_civil", "candidato_estado_civil"),
            txt("perfil_raca_cor", "candidato_raca_cor"),
            txt("situacao_candidatura"),
            txt("resultado_candidatura"),
            num("qtd_registros", "qtd_registros_candidato"),
        ])
    else:
        raise ValueError(f"Tipo de documento prata_minima desconhecido: {tipo_documento}")
    exprs.append(pl.lit(tipo_documento).alias("tipo_documento"))
    out = frame.select(exprs)
    return out.to_arrow()


def write_prata_minima_arrow_table(
    table: Any,
    tipo_documento: str,
    root: Path,
    basename_prefix: str,
    flush_index: int,
    max_partitions: int,
    depth: int = 0,
) -> int:
    import pyarrow.parquet as pq

    if table.num_rows == 0:
        return 0
    try:
        out_table = transform_prata_minima_arrow_table(table, tipo_documento)
        rows = out_table.num_rows
        if rows == 0:
            return 0
        file_name = f"{basename_prefix}_flush{flush_index:06d}_d{depth}_{{i}}.parquet"
        try:
            pq.write_to_dataset(
                out_table,
                root_path=root.as_posix(),
                partition_cols=PRATA_MINIMA_SECTION_PARTITION_COLS,
                compression="snappy",
                basename_template=file_name,
                max_partitions=max(1_024, min(max_partitions, max(rows * 2, 1_024))),
                use_threads=False,
            )
        except Exception as write_exc:
            msg = str(write_exc).lower()
            resource_error = (
                "cannot allocate memory" in msg
                or "input/output error" in msg
                or "error writing bytes" in msg
                or "cannot create directory" in msg
            )
            if not resource_error:
                raise
            fallback_dir = root / "_flat"
            fallback_dir.mkdir(parents=True, exist_ok=True)
            fallback_path = fallback_dir / f"{basename_prefix}_flush{flush_index:06d}_d{depth}.parquet"
            partial_pattern = f"{basename_prefix}_flush{flush_index:06d}_d{depth}_*.parquet"
            removed_partials = 0
            try:
                for partial in root.rglob(partial_pattern):
                    if partial.is_file():
                        partial.unlink()
                        removed_partials += 1
            except Exception as cleanup_exc:
                logging.warning("Prata minima nao conseguiu limpar parciais %s: %s", partial_pattern, cleanup_exc)
            logging.warning(
                "Prata minima fallback plano por erro de escrita particionada: documento=%s linhas=%s parciais_removidas=%s arquivo=%s erro=%s",
                tipo_documento,
                rows,
                removed_partials,
                fallback_path,
                write_exc,
            )
            pq.write_table(out_table, fallback_path.as_posix(), compression="snappy", use_dictionary=True)
        del out_table
        return rows
    except Exception as exc:
        msg = str(exc).lower()
        resource_error = (
            "cannot allocate memory" in msg
            or "input/output error" in msg
            or "error writing bytes" in msg
            or "cannot create directory" in msg
        )
        if not resource_error and table.num_rows > 1_000 and depth < 5:
            mid = table.num_rows // 2
            left = table.slice(0, mid)
            right = table.slice(mid)
            written = write_prata_minima_arrow_table(left, tipo_documento, root, basename_prefix, flush_index * 10 + 1, max_partitions, depth + 1)
            written += write_prata_minima_arrow_table(right, tipo_documento, root, basename_prefix, flush_index * 10 + 2, max_partitions, depth + 1)
            del left
            del right
            return written
        raise


def remove_prata_minima_partial(cfg: CleanDatabaseConfig, table: str, slice_key: str) -> None:
    root = prata_minima_root(cfg)
    if not root.exists():
        return
    uf, year = split_slice_key(slice_key)
    removed = 0
    for doc_dir in list(root.rglob(f"tipo_documento={table}")):
        if not doc_dir.is_dir():
            continue
        parts = {parent.name.split("=", 1)[0]: parent.name.split("=", 1)[1] for parent in doc_dir.parents if "=" in parent.name}
        if uf and parts.get("uf") != uf:
            continue
        if year and year != "SEM_ANO" and parts.get("ano") != year:
            continue
        shutil.rmtree(doc_dir)
        removed += 1
    file_prefix = safe_name(f"{table}_{slice_key}", 70)
    for parquet_file in list(root.rglob(f"{file_prefix}*.parquet")):
        if not parquet_file.is_file():
            continue
        try:
            parquet_file.unlink()
            removed += 1
        except Exception as exc:
            logging.warning("Prata minima nao conseguiu remover parcial %s: %s", parquet_file, exc)
    if removed:
        logging.info("Prata minima removeu %s arquivos/particoes parciais antigas: documento=%s fatia=%s", removed, table, slice_key)


def stream_prata_minima_document(
    cfg: CleanDatabaseConfig,
    source_path: Path,
    tipo_documento: str,
    slice_key: str,
    progress_dir: Path,
    label: str,
    manifest: dict[str, dict[str, Any]] | None = None,
) -> str:
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    if not parquet_dataset_exists(source_path):
        return ""

    root = prata_minima_root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    marker = prata_minima_done_marker(cfg, tipo_documento, slice_key)
    skipped_marker = prata_minima_skipped_marker(cfg, tipo_documento, slice_key)
    if manifest is not None and prata_minima_manifest_done(manifest, tipo_documento, slice_key):
        logging.info("Prata minima manifesto: pulando documento ja concluido: %s %s", tipo_documento, slice_key)
        return str(root)
    if manifest is not None and prata_minima_manifest_skipped(manifest, tipo_documento, slice_key):
        logging.info("Prata minima manifesto: pulando documento ja marcado como skipped: %s %s", tipo_documento, slice_key)
        return ""
    if marker.exists() and prata_minima_table_exists(cfg, tipo_documento, slice_key):
        logging.info("Prata minima pulando documento ja concluido: %s %s", tipo_documento, slice_key)
        if manifest is not None:
            try:
                payload = json.loads(marker.read_text(encoding="utf-8-sig"))
            except Exception:
                payload = {"status": "ok", "documento": tipo_documento, "fatia": slice_key, "saida": str(root)}
            update_prata_minima_manifest(cfg, manifest, tipo_documento, slice_key, payload)
        return str(root)

    remove_prata_minima_partial(cfg, tipo_documento, slice_key)
    remove_path_if_exists(marker)
    remove_path_if_exists(skipped_marker)

    batch_rows = prata_minima_stream_batch_rows(cfg)
    started = time.perf_counter()
    rows_read = 0
    rows_written = 0
    batches = 0
    if manifest is not None:
        update_prata_minima_manifest(
            cfg,
            manifest,
            tipo_documento,
            slice_key,
            {
                "label": label,
                "documento": tipo_documento,
                "fatia": slice_key,
                "status": "processing",
                "fonte": str(source_path),
                "saida": str(root),
                "batch_rows": batch_rows,
                "linhas_lidas": 0,
                "linhas_gravadas": 0,
                "batches_pulados": 0,
                "layout_version": PRATA_MINIMA_LAYOUT_VERSION,
                "partition_cols": PRATA_MINIMA_SECTION_PARTITION_COLS,
                "atualizado_epoch": round(time.time(), 3),
            },
        )
    logging.info(
        "Prata minima streaming iniciado: documento=%s fatia=%s batch_rows=%s fonte=%s",
        tipo_documento,
        slice_key,
        batch_rows,
        source_path,
    )
    write_ouro_event(
        progress_dir,
        label,
        "prata_minima_stream_inicio",
        documento=tipo_documento,
        fatia=slice_key,
        batch_rows=batch_rows,
        fonte=str(source_path),
        saida=str(root),
    )
    save_json(
        {
            "label": label,
            "status": "processando",
            "documento": tipo_documento,
            "fatia": slice_key,
            "batch_rows": batch_rows,
            "fonte": str(source_path),
            "saida": str(root),
            "layout_version": PRATA_MINIMA_LAYOUT_VERSION,
            "partition_cols": PRATA_MINIMA_SECTION_PARTITION_COLS,
            "inicio_epoch": round(started, 3),
        },
        prata_minima_current_progress_path(cfg),
    )
    try:
        dataset = ds.dataset(source_path.as_posix(), format="parquet", partitioning="hive")
        scanner = dataset.scanner(batch_size=batch_rows, use_threads=True)
    except Exception as exc:
        duration = time.perf_counter() - started
        logging.exception(
            "Prata minima pulando documento inteiro por erro de leitura: documento=%s fatia=%s fonte=%s erro=%s",
            tipo_documento,
            slice_key,
            source_path,
            exc,
        )
        save_json(
            skipped_payload := {
                "label": label,
                "documento": tipo_documento,
                "fatia": slice_key,
                "status": "skipped",
                "motivo": "erro_abrindo_dataset_parquet",
                "erro": str(exc),
                "fonte": str(source_path),
                "duracao_segundos": round(duration, 3),
                "layout_version": PRATA_MINIMA_LAYOUT_VERSION,
                "partition_cols": PRATA_MINIMA_SECTION_PARTITION_COLS,
                "atualizado_epoch": round(time.time(), 3),
            },
            skipped_marker,
        )
        if manifest is not None:
            update_prata_minima_manifest(cfg, manifest, tipo_documento, slice_key, skipped_payload)
        write_ouro_event(
            progress_dir,
            label,
            "prata_minima_documento_pulado",
            documento=tipo_documento,
            fatia=slice_key,
            fonte=str(source_path),
            erro=str(exc),
        )
        return ""
    basename_prefix = safe_name(f"{tipo_documento}_{slice_key}_{uuid.uuid4().hex[:8]}", 80)
    skipped_batches = 0
    flushes = 0
    buffer_batches: list[Any] = []
    buffer_rows = 0
    target_buffer_rows = min(max(batch_rows * 10, 50_000), 100_000)

    def flush_buffer() -> int:
        nonlocal buffer_batches, buffer_rows, flushes, rows_written, skipped_batches
        if not buffer_batches:
            return 0
        flushes += 1
        current_batches = buffer_batches
        current_rows = buffer_rows
        buffer_batches = []
        buffer_rows = 0
        try:
            combined = pa.Table.from_batches(current_batches)
            written = write_prata_minima_arrow_table(
                combined,
                tipo_documento,
                root,
                basename_prefix,
                flushes,
                max_partitions=max(100_000, current_rows * 4),
            )
            rows_written += written
            del combined
            for old_batch in current_batches:
                try:
                    del old_batch
                except Exception:
                    pass
            clean_memory()
            return written
        except Exception as exc:
            skipped_batches += len(current_batches)
            logging.exception(
                "Prata minima pulando cesta por erro de transformacao/escrita: documento=%s fatia=%s cesta=%s linhas=%s erro=%s",
                tipo_documento,
                slice_key,
                flushes,
                current_rows,
                exc,
            )
            write_ouro_event(
                progress_dir,
                label,
                "prata_minima_cesta_pulada",
                documento=tipo_documento,
                fatia=slice_key,
                cesta=flushes,
                linhas=current_rows,
                erro=str(exc),
            )
            for old_batch in current_batches:
                try:
                    del old_batch
                except Exception:
                    pass
            clean_memory()
            return 0

    try:
        batch_iter = scanner.to_batches()
    except Exception as exc:
        duration = time.perf_counter() - started
        logging.exception(
            "Prata minima pulando documento inteiro por erro criando batches: documento=%s fatia=%s erro=%s",
            tipo_documento,
            slice_key,
            exc,
        )
        save_json(
            skipped_payload := {
                "label": label,
                "documento": tipo_documento,
                "fatia": slice_key,
                "status": "skipped",
                "motivo": "erro_criando_batches",
                "erro": str(exc),
                "fonte": str(source_path),
                "duracao_segundos": round(duration, 3),
                "layout_version": PRATA_MINIMA_LAYOUT_VERSION,
                "partition_cols": PRATA_MINIMA_SECTION_PARTITION_COLS,
                "atualizado_epoch": round(time.time(), 3),
            },
            skipped_marker,
        )
        if manifest is not None:
            update_prata_minima_manifest(cfg, manifest, tipo_documento, slice_key, skipped_payload)
        write_ouro_event(
            progress_dir,
            label,
            "prata_minima_documento_pulado",
            documento=tipo_documento,
            fatia=slice_key,
            fonte=str(source_path),
            erro=str(exc),
        )
        return ""

    while True:
        try:
            batch = next(batch_iter)
        except StopIteration:
            break
        except Exception as exc:
            skipped_batches += 1
            logging.exception(
                "Prata minima pulando batch corrompido: documento=%s fatia=%s batch=%s erro=%s",
                tipo_documento,
                slice_key,
                batches + 1,
                exc,
            )
            write_ouro_event(
                progress_dir,
                label,
                "prata_minima_batch_pulado",
                documento=tipo_documento,
                fatia=slice_key,
                batch=batches + 1,
                erro=str(exc),
            )
            try:
                del batch
            except Exception:
                pass
            clean_memory()
            continue
        batches += 1
        rows_read += batch.num_rows
        if batch.num_rows == 0:
            try:
                del batch
            except Exception:
                pass
            clean_memory()
            continue
        buffer_batches.append(batch)
        buffer_rows += batch.num_rows
        if buffer_rows >= target_buffer_rows:
            flush_buffer()
        if batches == 1 or batches % 50 == 0:
            logging.info(
                "Prata minima streaming %s %s: batches=%s cestas=%s lidas=%s gravadas=%s buffer=%s pulados=%s",
                tipo_documento,
                slice_key,
                batches,
                flushes,
                rows_read,
                rows_written,
                buffer_rows,
                skipped_batches,
            )
            write_ouro_event(
                progress_dir,
                label,
                "prata_minima_stream_lote",
                documento=tipo_documento,
                fatia=slice_key,
                batches=batches,
                cestas=flushes,
                linhas_lidas=rows_read,
                linhas_gravadas=rows_written,
                buffer_linhas=buffer_rows,
                batches_pulados=skipped_batches,
            )
            save_json(
                {
                    "label": label,
                    "status": "processando",
                    "documento": tipo_documento,
                    "fatia": slice_key,
                    "batches": batches,
                    "cestas": flushes,
                    "linhas_lidas": rows_read,
                    "linhas_gravadas": rows_written,
                    "buffer_linhas": buffer_rows,
                    "batches_pulados": skipped_batches,
                    "fonte": str(source_path),
                    "saida": str(root),
                    "layout_version": PRATA_MINIMA_LAYOUT_VERSION,
                    "partition_cols": PRATA_MINIMA_SECTION_PARTITION_COLS,
                    "atualizado_epoch": round(time.time(), 3),
                },
                prata_minima_current_progress_path(cfg),
            )
        clean_memory()

    flush_buffer()
    duration = time.perf_counter() - started
    done_payload = {
            "label": label,
            "documento": tipo_documento,
            "fatia": slice_key,
            "status": "ok",
            "fonte": str(source_path),
            "saida": str(root),
            "batches": batches,
            "cestas_gravadas": flushes,
            "linhas_lidas": rows_read,
            "linhas_gravadas": rows_written,
            "batches_pulados": skipped_batches,
            "duracao_segundos": round(duration, 3),
            "modo": "polars_pyarrow_streaming_cestas",
            "layout_version": PRATA_MINIMA_LAYOUT_VERSION,
            "partition_cols": PRATA_MINIMA_SECTION_PARTITION_COLS,
            "colunas_localizacao": ["cd_municipio", "nm_municipio", "zona", "secao"],
    }
    save_json(done_payload, marker)
    if manifest is not None:
        update_prata_minima_manifest(cfg, manifest, tipo_documento, slice_key, done_payload)
    logging.info(
        "Prata minima streaming finalizado: documento=%s fatia=%s batches=%s cestas=%s lidas=%s gravadas=%s pulados=%s em %.1fs",
        tipo_documento,
        slice_key,
        batches,
        flushes,
        rows_read,
        rows_written,
        skipped_batches,
        duration,
    )
    write_ouro_event(
        progress_dir,
        label,
        "prata_minima_stream_fim",
        documento=tipo_documento,
        fatia=slice_key,
        batches=batches,
        cestas=flushes,
        linhas_lidas=rows_read,
        linhas_gravadas=rows_written,
        batches_pulados=skipped_batches,
        duracao_segundos=round(duration, 3),
    )
    save_json(
        {
            "label": label,
            "status": "finalizado",
            "documento": tipo_documento,
            "fatia": slice_key,
            "batches": batches,
            "cestas": flushes,
            "linhas_lidas": rows_read,
            "linhas_gravadas": rows_written,
            "batches_pulados": skipped_batches,
            "duracao_segundos": round(duration, 3),
            "layout_version": PRATA_MINIMA_LAYOUT_VERSION,
            "partition_cols": PRATA_MINIMA_SECTION_PARTITION_COLS,
            "atualizado_epoch": round(time.time(), 3),
        },
        prata_minima_current_progress_path(cfg),
    )
    try:
        del batch_iter
    except Exception:
        pass
    try:
        del scanner
    except Exception:
        pass
    try:
        del dataset
    except Exception:
        pass
    buffer_batches.clear()
    buffer_rows = 0
    clean_memory()
    return str(root)


def prata_minima_plan_ready(cfg: CleanDatabaseConfig, plan: list[dict[str, Any]]) -> bool:
    if not plan:
        return False
    for item in plan:
        uf = safe_text(item.get("uf", "")) or "SEM_UF"
        year = safe_text(item.get("ano", ""))
        slice_key = safe_text(item.get("slice_key", "")) or slice_name(uf, year)
        if not prata_minima_table_exists(cfg, "perfil_secao", slice_key) or not prata_minima_table_exists(cfg, "resultado_secao", slice_key):
            return False
    return True


def materialize_prata_minima_correlacoes(
    cfg: CleanDatabaseConfig,
    analyses: Path,
    plan: list[dict[str, Any]],
    progress_dir: Path,
    label: str,
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    manifest = load_prata_minima_manifest(cfg)
    manifest = seed_prata_minima_manifest_from_plan(cfg, manifest, plan, label)
    status_counts = dict(pd.Series([safe_text(item.get("status", "")) for item in manifest.values()]).value_counts()) if manifest else {}
    logging.info(
        "Manifesto prata_minima carregado/preparado: %s itens registrados | status=%s",
        len(manifest),
        status_counts,
    )
    logging.info("Etapa prata_minima: organizando %s UF/fatia(s) da prata em Parquet particionado somente por UF; ano/tipo_documento/municipio/zona/secao ficam como colunas.", len(plan))
    save_json(
        {
            "label": label,
            "status": "materializando_prata_minima",
            "fatias_total": len(plan),
            "particionamento": "uf",
            "observacao": "Ano, tipo_documento, municipio, zona e secao ficam como colunas dentro dos Parquets para reduzir diretorios e memoria.",
            "manifesto_logs": str(prata_minima_manifest_path(cfg)),
            "manifesto_visivel": str(prata_minima_public_manifest_path(cfg)),
            "ufs": [{"uf": item.get("uf"), "slice_key": item.get("slice_key")} for item in plan],
        },
        progress_dir / "prata_minima_plano.json",
    )
    for item_index, item in enumerate(plan, start=1):
        uf = safe_text(item.get("uf", "")) or "SEM_UF"
        year = safe_text(item.get("ano", ""))
        slice_key = safe_text(item.get("slice_key", "")) or slice_name(uf, year)
        e_path = Path(safe_text(item.get("eleitorado_path", "")))
        r_path = Path(safe_text(item.get("resultados_path", "")))
        c_path = cfg.out / "prata" / "candidatos" / f"uf={uf}"
        profile_cache_path = prata_minima_chunk(cfg, "perfil_secao", slice_key)
        result_cache_path = prata_minima_chunk(cfg, "resultado_secao", slice_key)
        docs_previstos = ["perfil_secao", "resultado_secao"]
        if parquet_dataset_exists(c_path):
            docs_previstos.append("candidatos_secao")
        logging.info(
            "Prata minima fatia %s/%s: uf=%s ano=%s slice=%s documentos=%s",
            item_index,
            len(plan),
            uf,
            year or "descoberto_por_linha",
            slice_key,
            ", ".join(docs_previstos),
        )
        save_json(
            {
                "label": label,
                "status": "entrando_fatia",
                "indice_fatia": item_index,
                "total_fatias": len(plan),
                "uf": uf,
                "ano": year,
                "slice_key": slice_key,
                "documentos_previstos": docs_previstos,
                "eleitorado_path": str(e_path),
                "resultados_path": str(r_path),
                "candidatos_path": str(c_path) if "candidatos_secao" in docs_previstos else "",
                "layout_version": PRATA_MINIMA_LAYOUT_VERSION,
                "partition_cols": PRATA_MINIMA_SECTION_PARTITION_COLS,
                "atualizado_epoch": round(time.time(), 3),
            },
            prata_minima_current_progress_path(cfg),
        )

        if prata_minima_manifest_skipped(manifest, "perfil_secao", slice_key):
            logging.warning("Prata minima pulando perfil_secao %s: status skipped no manifesto.", slice_key)
            outputs[f"prata_minima_perfil_secao_{slice_key}"] = ""
        elif prata_minima_manifest_done(manifest, "perfil_secao", slice_key) or prata_minima_table_exists(cfg, "perfil_secao", slice_key):
            outputs[f"prata_minima_perfil_secao_{slice_key}"] = str(profile_cache_path)
        else:
            outputs[f"prata_minima_perfil_secao_{slice_key}"] = stream_prata_minima_document(
                cfg,
                e_path,
                "perfil_secao",
                slice_key,
                progress_dir,
                label,
                manifest,
            )

        if prata_minima_manifest_skipped(manifest, "resultado_secao", slice_key):
            logging.warning("Prata minima pulando resultado_secao %s: status skipped no manifesto.", slice_key)
            outputs[f"prata_minima_resultado_secao_{slice_key}"] = ""
        elif prata_minima_manifest_done(manifest, "resultado_secao", slice_key) or prata_minima_table_exists(cfg, "resultado_secao", slice_key):
            outputs[f"prata_minima_resultado_secao_{slice_key}"] = str(result_cache_path)
        else:
            outputs[f"prata_minima_resultado_secao_{slice_key}"] = stream_prata_minima_document(
                cfg,
                r_path,
                "resultado_secao",
                slice_key,
                progress_dir,
                label,
                manifest,
            )

        if parquet_dataset_exists(c_path):
            if prata_minima_manifest_skipped(manifest, "candidatos_secao", slice_key):
                logging.warning("Prata minima pulando candidatos_secao %s: status skipped no manifesto.", slice_key)
                outputs[f"prata_minima_candidatos_secao_{slice_key}"] = ""
            elif prata_minima_manifest_done(manifest, "candidatos_secao", slice_key) or prata_minima_table_exists(cfg, "candidatos_secao", slice_key):
                outputs[f"prata_minima_candidatos_secao_{slice_key}"] = str(profile_cache_path)
            else:
                outputs[f"prata_minima_candidatos_secao_{slice_key}"] = stream_prata_minima_document(
                    cfg,
                    c_path,
                    "candidatos_secao",
                    slice_key,
                    progress_dir,
                    label,
                    manifest,
                )

    logging.info("Etapa prata_minima: streaming concluido ou reaproveitado; seguindo para verificacao.")

    missing: list[dict[str, str]] = []
    for item in plan:
        uf = safe_text(item.get("uf", "")) or "SEM_UF"
        year = safe_text(item.get("ano", ""))
        slice_key = safe_text(item.get("slice_key", "")) or slice_name(uf, year)
        profile_cache_path = prata_minima_chunk(cfg, "perfil_secao", slice_key)
        result_cache_path = prata_minima_chunk(cfg, "resultado_secao", slice_key)
        profile_ok = prata_minima_manifest_done(manifest, "perfil_secao", slice_key) or prata_minima_table_exists(cfg, "perfil_secao", slice_key)
        result_ok = prata_minima_manifest_done(manifest, "resultado_secao", slice_key) or prata_minima_table_exists(cfg, "resultado_secao", slice_key)
        if not profile_ok or not result_ok:
            missing.append(
                {
                    "uf": uf,
                    "ano": year,
                    "slice_key": slice_key,
                    "perfil_secao": str(profile_cache_path),
                    "resultado_secao": str(result_cache_path),
                    "perfil_secao_ok": profile_ok,
                    "resultado_secao_ok": result_ok,
                }
            )
    status = "ok" if not missing else "incompleta"
    save_json(
        {
            "label": label,
            "status": status,
            "ufs_total": len(plan),
            "faltantes": missing,
            "outputs": outputs,
        },
        progress_dir / "prata_minima_status.json",
    )
    if missing:
        logging.error("Etapa prata_minima incompleta: %s fatias faltantes. As analises dessas fatias serao puladas.", len(missing))
    else:
        logging.info("Etapa prata_minima concluida: todas as pecas necessarias existem.")
    return outputs


def ready_correlacao_post_tasks(
    post_tasks: list[dict[str, Any]],
    partido_parts_path: Path,
    candidato_parts_path: Path,
    cfg: CleanDatabaseConfig,
) -> list[dict[str, Any]]:
    ready: list[dict[str, Any]] = []
    if output_has_data(partido_parts_path):
        ready.extend(post_tasks[:2])
    else:
        logging.error("Partes por partido ausentes em %s; resumo estadual/local por partido sera pulado.", partido_parts_path)
    if not cfg.skip_heavy_analyses:
        if output_has_data(candidato_parts_path):
            ready.extend(post_tasks[2:])
        else:
            logging.error("Partes por candidato ausentes em %s; resumo estadual/local por candidato sera pulado.", candidato_parts_path)
    return ready


def prepare_ouro_progress(cfg: CleanDatabaseConfig, label: str) -> Path:
    progress_dir = cfg.out / "logs" / "ouro"
    progress_dir.mkdir(parents=True, exist_ok=True)
    save_json({"label": label, "status": "iniciando"}, progress_dir / f"{label}_progresso.json")
    write_ouro_event(progress_dir, label, "preparando_etapa", etapa=label)
    return progress_dir


def reset_ouro_targets(targets: list[Path], resume: bool = False) -> None:
    if resume:
        return
    for target in targets:
        remove_path_if_exists(target)
        if target.suffix == "":
            remove_path_if_exists(target.with_suffix(".parquet"))


def remove_path_if_exists(path: Path) -> None:
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def list_uf_partition_dirs(dataset_root: Path) -> list[tuple[str, Path]]:
    if not dataset_root.exists():
        return []
    out: list[tuple[str, Path]] = []
    for path in sorted(dataset_root.iterdir(), key=lambda p: uf_sort_key(p.name.split("=", 1)[1] if p.name.startswith("uf=") else p.name)):
        if not path.is_dir() or not path.name.startswith("uf="):
            continue
        uf = path.name.split("=", 1)[1] or "SEM_UF"
        if parquet_dataset_exists(path):
            out.append((uf, path))
    if out:
        return sorted(out, key=lambda item: uf_sort_key(item[0]))
    return [("SEM_UF", dataset_root)] if parquet_dataset_exists(dataset_root) else []


def uf_sort_key(uf: Any) -> tuple[int, str]:
    text = safe_text(uf, "SEM_UF").upper()
    if text in {"SEM_UF", "ZZ", ""}:
        return (1, text or "SEM_UF")
    return (0, text)


def year_sort_key(year: Any) -> tuple[int, int | str]:
    text = safe_text(year, "")
    parsed = parse_number(text)
    if pd.notna(parsed):
        return (0, int(parsed))
    return (1, text)


def sort_years(years: Iterable[Any]) -> list[str]:
    values = [safe_text(year) for year in years if safe_text(year)]
    return sorted(values, key=year_sort_key) or [""]


def sort_uf_year_plan(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        plan,
        key=lambda item: (
            uf_sort_key(item.get("uf", "SEM_UF")),
            year_sort_key(item.get("ano", "")),
            safe_text(item.get("tarefa", "")),
        ),
    )


def parquet_dataset_size_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0
    for parquet_path in path.rglob("*.parquet"):
        try:
            total += parquet_path.stat().st_size
        except OSError:
            continue
    return total / (1024 ** 3)


def chunk_output(root: Path, uf: str) -> Path:
    return root / f"chunk={safe_name(uf, 20) or 'SEM_UF'}"


def municipio_slice_key(slice_key: str, cd_municipio: str) -> str:
    municipio = safe_name(cd_municipio, 40) or "SEM_MUNICIPIO"
    return f"{safe_name(slice_key, 60) or 'SEM_FATIA'}_mun_{municipio}"


def municipio_chunk_output(root: Path, slice_key: str, cd_municipio: str) -> Path:
    chunk = f"{safe_name(slice_key, 50) or 'SEM_FATIA'}_mun_{safe_name(cd_municipio, 40) or 'SEM_MUNICIPIO'}"
    return root / f"chunk={chunk}"


def slice_name(uf: str, year: str) -> str:
    year_text = safe_name(year, 20) if year else "SEM_ANO"
    return f"{safe_name(uf, 20) or 'SEM_UF'}_{year_text}"


def split_slice_key(slice_key: str) -> tuple[str, str]:
    text = safe_text(slice_key, "")
    if "_" not in text:
        return text or "SEM_UF", ""
    uf, year = text.rsplit("_", 1)
    return uf or "SEM_UF", "" if year == "SEM_ANO" else year


def list_years_for_dataset(path: Path, cfg: CleanDatabaseConfig) -> list[str]:
    if not parquet_dataset_exists(path):
        return []
    import duckdb

    try:
        df = read_years_with_duckdb(path, cfg)
    except Exception as exc:
        if "No magic bytes" not in str(exc) and "Parquet" not in str(exc):
            raise
        logging.warning("Parquet invalido detectado em %s; movendo arquivos corrompidos para quarentena e tentando de novo.", path)
        quarantine_corrupt_parquets(path, cfg.out)
        if not parquet_dataset_exists(path):
            return []
        df = read_years_with_duckdb(path, cfg)
    years = [safe_text(x) for x in df.get("ano", []) if safe_text(x)]
    return sort_years(years)


def read_years_with_duckdb(path: Path, cfg: CleanDatabaseConfig) -> pd.DataFrame:
    import duckdb

    con = duckdb.connect(database=":memory:")
    configure_duckdb_connection(con, max(1, int(cfg.duckdb_threads or 1)))
    try:
        return con.execute(
            f"""
            select distinct cast(ano as varchar) as ano
            from {dataset_expr(path)}
            where nullif(trim(cast(ano as varchar)), '') is not null
            order by 1
            """
        ).fetchdf()
    finally:
        con.close()


def list_distinct_combinations(table_expr: str, columns: list[str], cfg: CleanDatabaseConfig) -> list[dict[str, str]]:
    if not columns:
        return []
    import duckdb

    select_cols = ", ".join(f"{discrete_sql_value(col)} as {col}" for col in columns)
    order_cols = ", ".join(columns)
    write_pipeline_event(cfg.out, "duckdb_auxiliar", "distinct_inicio", colunas=columns)
    started = time.perf_counter()
    con = duckdb.connect(database=":memory:")
    configure_duckdb_connection(con, max(1, int(cfg.duckdb_threads or 1)))
    try:
        df = con.execute(
            f"""
            select distinct {select_cols}
            from {table_expr}
            where {metric_sql('votos')} > 0
            order by {order_cols}
            """
        ).fetchdf()
    finally:
        con.close()
    combos: list[dict[str, str]] = []
    for row in df.to_dict(orient="records"):
        combo = {col: safe_text(row.get(col, "SEM_VALOR")) or "SEM_VALOR" for col in columns}
        combos.append(combo)
    write_pipeline_event(cfg.out, "duckdb_auxiliar", "distinct_fim", colunas=columns, qtd=len(combos), duracao_segundos=round(time.perf_counter() - started, 3))
    return combos


def list_split_combinations(level_name: str, table_expr: str, columns: list[str], cfg: CleanDatabaseConfig) -> list[dict[str, Any]]:
    if level_name == "municipio_bucket":
        col = columns[0] if columns else "cd_municipio"
        combos = [
            {"__hash_col": col, "__hash_mod": RESULTADOS_HASH_BUCKETS, "__hash_bucket": bucket}
            for bucket in range(RESULTADOS_HASH_BUCKETS)
        ]
        logging.info(
            "Usando fatiamento fixo por hash de %s: %s buckets. Sem query DISTINCT previa.",
            col,
            RESULTADOS_HASH_BUCKETS,
        )
        write_pipeline_event(
            cfg.out,
            "duckdb_auxiliar",
            "hash_buckets_criados",
            coluna=col,
            buckets=RESULTADOS_HASH_BUCKETS,
        )
        return combos
    return list_distinct_combinations(table_expr, columns, cfg)


def quarantine_corrupt_parquets(path: Path, banco_out: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        logging.warning("Nao foi possivel validar Parquets corrompidos sem pyarrow.parquet: %s", exc)
        return []

    root = path if path.is_dir() else path.parent
    if not root.exists():
        return []
    quarantine_root = banco_out / "metadados" / "parquets_corrompidos"
    rows: list[dict[str, Any]] = []
    for parquet_path in sorted(root.rglob("*.parquet")):
        try:
            if parquet_path.stat().st_size < 8:
                raise ValueError("arquivo parquet menor que o footer minimo")
            pq.ParquetFile(parquet_path).metadata
        except Exception as exc:
            rel = safe_rel(parquet_path, banco_out)
            digest = hashlib.sha1(rel.encode("utf-8", errors="ignore")).hexdigest()[:12]
            target_dir = quarantine_root / safe_name(str(parquet_path.parent.name), 80)
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"{safe_name(parquet_path.stem, 80)}_{digest}_{uuid.uuid4().hex[:8]}.parquet.corrupt"
            logging.warning("Movendo Parquet corrompido para quarentena: %s -> %s | erro=%s", parquet_path, target, exc)
            try:
                shutil.move(str(parquet_path), str(target))
                rows.append({
                    "arquivo_origem": str(parquet_path),
                    "arquivo_quarentena": str(target),
                    "erro": str(exc),
                })
            except Exception as move_exc:
                logging.error("Falha ao mover Parquet corrompido %s: %s", parquet_path, move_exc)
                rows.append({
                    "arquivo_origem": str(parquet_path),
                    "arquivo_quarentena": "",
                    "erro": f"{exc}; falha_move={move_exc}",
                })
    if rows:
        save_json(rows, quarantine_root / f"quarentena_{int(time.time())}.json")
    return rows


def filtered_year_expr(table_expr: str, year: str) -> str:
    text = safe_text(year)
    if not text:
        return table_expr
    parsed = parse_number(text)
    if pd.notna(parsed):
        return f"(select * from {table_expr} where {metric_sql('ano')} = {float(parsed)})"
    return f"(select * from {table_expr} where cast(ano as varchar) = {sql_lit(text)})"


def execute_ouro_task(task: dict[str, Any], cfg: CleanDatabaseConfig, label: str, progress_dir: Path) -> str:
    out = Path(task["out"])
    if cfg.resume and ouro_task_done(task, progress_dir):
        logging.info("Pulando tarefa ouro ja concluida [%s]: %s", label, task.get("name"))
        return str(out)
    task_name = safe_name(task.get("name", "tarefa"), 80)
    requested_threads = max(1, int(cfg.duckdb_threads or 1))
    retry_threads = []
    for value in [requested_threads, requested_threads // 2, 1]:
        value = max(1, int(value or 1))
        if value not in retry_threads:
            retry_threads.append(value)

    last_exc: Exception | None = None
    for attempt, threads in enumerate(retry_threads, start=1):
        try:
            if attempt > 1:
                logging.warning(
                    "Retry DuckDB [%s] %s com %s thread(s) apos erro: %s",
                    label,
                    task.get("name"),
                    threads,
                    last_exc,
                )
            return execute_copy_task(task, threads, progress_dir=progress_dir, label=label)
        except Exception as exc:
            last_exc = exc
            if not is_retryable_duckdb_error(exc) or attempt == len(retry_threads):
                break
            clean_memory()

    logging.error("Erro na tarefa ouro [%s] %s: %s", label, task.get("name"), last_exc)
    write_ouro_event(
        progress_dir,
        label,
        "tarefa_erro",
        tarefa=str(task.get("name", "")),
        saida=str(task.get("out", "")),
        erro=str(last_exc),
        tentativas_threads=retry_threads,
    )
    save_json(
        {
            "label": label,
            "tarefa": str(task.get("name", "")),
            "status": "erro",
            "saida": str(task.get("out", "")),
            "erro": str(last_exc),
            "tentativas_threads": retry_threads,
        },
        progress_dir / f"{task_name}.error.json",
    )
    return f"ERRO: {last_exc}"


def output_has_data(path: Path) -> bool:
    if path.is_dir():
        return parquet_dataset_exists(path)
    return path.exists() and path.stat().st_size > 0


def reusable_dataset_path(primary: Path, legacy: Path | None = None) -> Path:
    if output_has_data(primary):
        return primary
    if legacy is not None and output_has_data(legacy):
        return legacy
    return primary


def ouro_task_done(task: dict[str, Any], progress_dir: Path) -> bool:
    marker = progress_dir / f"{safe_name(task.get('name', 'tarefa'), 80)}.done.json"
    out = Path(task["out"])
    return marker.exists() and output_has_data(out)


def copy_task(name: str, sql: str, out: Path, partition_by: list[str] | None = None) -> dict[str, Any]:
    return {"name": name, "sql": sql, "out": out, "partition_by": partition_by or []}


def should_force_memory_safe_ouro(label: str, cfg: CleanDatabaseConfig) -> bool:
    if cfg.ouro_parallel_aggressive:
        return False
    safe_labels = (
        "ouro_nivelado_municipal_",
        "ouro_nivelado_estadual_",
        "ouro_nivelado_brasil",
        "ouro_nivelado_compat",
        "ouro_estados_brasil_estadual_",
        "ouro_estados_brasil_brasil",
        "ouro_estados_brasil_compat",
    )
    return any(marker in label for marker in safe_labels)


def execute_copy_tasks(tasks: list[dict[str, Any]], cfg: CleanDatabaseConfig, label: str) -> dict[str, str]:
    if not tasks:
        return {}
    progress_dir = cfg.out / "logs" / "ouro"
    progress_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "label": label,
            "status": "iniciando",
            "tarefas_total": len(tasks),
            "tarefas": [task.get("name", "") for task in tasks],
        },
        progress_dir / f"{safe_name(label, 80)}_progresso.json",
    )
    workers = min(max(1, int(cfg.ouro_workers or 1)), len(tasks))
    memory_safe = should_force_memory_safe_ouro(label, cfg)
    if workers > 1 and memory_safe:
        logging.info(
            "Modo memoria segura em %s: executando tarefas em fila para evitar OOM. "
            "Use --banco-ouro-paralelo-agressivo para forcar paralelismo.",
            label,
        )
        workers = 1
    threads_per_task = max(1, int(cfg.duckdb_threads or 1) // workers)
    if memory_safe:
        threads_per_task = 1
    worker_cfg = replace(cfg, duckdb_threads=threads_per_task)
    logging.info(
        "Gerando %s: %s tarefas com %s worker(s) DuckDB x %s threads/tarefa.",
        label,
        len(tasks),
        workers,
        threads_per_task,
    )
    if workers <= 1:
        outputs: dict[str, str] = {}
        for index, task in enumerate(tasks, start=1):
            logging.info("Iniciando tarefa ouro %s/%s [%s]: %s", index, len(tasks), label, task.get("name"))
            save_json(
                {
                    "label": label,
                    "status": "processando",
                    "tarefa_atual": task.get("name", ""),
                    "indice": index,
                    "total": len(tasks),
                    "concluidas": list(outputs),
                },
                progress_dir / f"{safe_name(label, 80)}_progresso.json",
            )
            outputs[task["name"]] = execute_ouro_task(task, worker_cfg, label, progress_dir)
            clean_memory()
        save_json({"label": label, "status": "ok", "outputs": outputs}, progress_dir / f"{safe_name(label, 80)}_progresso.json")
        return outputs
    outputs: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {}
        for index, task in enumerate(tasks, start=1):
            logging.info("Enfileirando tarefa ouro %s/%s [%s]: %s", index, len(tasks), label, task.get("name"))
            future_map[pool.submit(execute_ouro_task, task, worker_cfg, label, progress_dir)] = task
        for future in as_completed(future_map):
            task = future_map[future]
            try:
                outputs[task["name"]] = future.result()
                logging.info("Tarefa ouro concluida [%s]: %s", label, task.get("name"))
            except Exception as exc:
                logging.exception("Erro gerando ouro %s: %s", task.get("name"), exc)
                outputs[f"{task['name']}_erro"] = str(exc)
            save_json(
                {
                    "label": label,
                    "status": "processando",
                    "tarefas_total": len(tasks),
                    "concluidas": list(outputs),
                },
                progress_dir / f"{safe_name(label, 80)}_progresso.json",
            )
            clean_memory()
    save_json({"label": label, "status": "ok", "outputs": outputs}, progress_dir / f"{safe_name(label, 80)}_progresso.json")
    return outputs


def execute_copy_task(task: dict[str, Any], duckdb_threads: int, progress_dir: Path | None = None, label: str = "") -> str:
    import duckdb

    started = time.perf_counter()
    task_name = str(task.get("name", "tarefa"))
    temp_dir = duckdb_temp_dir()
    if progress_dir is not None:
        save_json(
            {
                "label": label,
                "tarefa": task_name,
                "status": "em_execucao",
                "saida": str(task.get("out", "")),
                "duckdb_threads": int(duckdb_threads or 1),
                "duckdb_temp_dir": str(temp_dir),
            },
            progress_dir / f"{safe_name(task_name, 80)}.started.json",
        )
    logging.info(
        "DuckDB COPY iniciado: %s -> %s | threads=%s | temp=%s",
        task_name,
        task.get("out", ""),
        max(1, int(duckdb_threads or 1)),
        temp_dir,
    )
    if progress_dir is not None:
        write_ouro_event(
            progress_dir,
            label,
            "duckdb_copy_inicio",
            tarefa=task_name,
            saida=str(task.get("out", "")),
            duckdb_threads=max(1, int(duckdb_threads or 1)),
            duckdb_temp_dir=str(temp_dir),
            partition_by=list(task.get("partition_by") or []),
        )
    con = duckdb.connect(database=":memory:")
    configure_duckdb_connection(con, max(1, int(duckdb_threads or 1)), temp_dir=temp_dir)
    out = Path(task["out"])
    try:
        copy_query(con, str(task["sql"]), out, partition_by=list(task.get("partition_by") or []))
    finally:
        con.close()
    duration = time.perf_counter() - started
    logging.info("DuckDB COPY finalizado: %s em %.1fs -> %s", task_name, duration, out)
    if progress_dir is not None:
        write_ouro_event(
            progress_dir,
            label,
            "duckdb_copy_fim",
            tarefa=task_name,
            saida=str(out),
            duracao_segundos=round(duration, 3),
        )
        remove_path_if_exists(progress_dir / f"{safe_name(task_name, 80)}.error.json")
        save_json(
            {
                "label": label,
                "tarefa": task_name,
                "status": "ok",
                "saida": str(out),
                "duracao_segundos": round(duration, 3),
            },
            progress_dir / f"{safe_name(task_name, 80)}.done.json",
        )
    return str(out)


def duckdb_temp_dir() -> Path:
    root = Path(tempfile.gettempdir()) / "analise_eleitoral_duckdb"
    root.mkdir(parents=True, exist_ok=True)
    return root


def configure_duckdb_connection(con: Any, duckdb_threads: int, temp_dir: Path | None = None) -> None:
    threads = max(1, int(duckdb_threads or 1))
    temp_root = temp_dir or duckdb_temp_dir()
    con.execute(f"SET threads={threads}")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory={sql_lit(temp_root.as_posix())}")


def is_retryable_duckdb_error(exc: Exception) -> bool:
    text = str(exc).lower()
    needles = [
        "out of memory",
        "could not allocate",
        "input/output error",
        "could not write file",
        "duckdb_temp_storage",
    ]
    return any(needle in text for needle in needles)


def ensure_parquet_engine() -> None:
    if PARQUET_ENGINE_OK:
        return
    raise RuntimeError(
        "O modo banco precisa de pyarrow para gravar Parquet. "
        "Instale com: python3 -m pip install -r scripts/pipeline_eleitoral_json/requirements.txt"
    )


def copy_query(con: Any, sql: str, out: Path, partition_by: list[str] | None = None) -> None:
    if not sql.strip():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        if out.is_dir():
            shutil.rmtree(out)
        else:
            out.unlink()
    if partition_by and out.suffix == "":
        legacy_file = out.with_suffix(".parquet")
        if legacy_file.exists():
            if legacy_file.is_dir():
                shutil.rmtree(legacy_file)
            else:
                legacy_file.unlink()
    target = sql_lit(out.as_posix())
    if partition_by:
        parts = ", ".join(partition_by)
        con.execute(f"COPY ({sql}) TO {target} (FORMAT PARQUET, PARTITION_BY ({parts}))")
    else:
        con.execute(f"COPY ({sql}) TO {target} (FORMAT PARQUET)")


def dataset_expr(path: Path) -> str:
    return f"read_parquet({sql_lit((path / '**' / '*.parquet').as_posix())}, union_by_name=true, hive_partitioning=true)"


def sql_subquery(sql: str) -> str:
    return f"({sql.strip()}) as src"


def parquet_dataset_exists(path: Path) -> bool:
    return path.exists() and any(path.rglob("*.parquet"))


def metric_sql(col: str) -> str:
    return f"coalesce(try_cast(replace(cast({col} as varchar), ',', '.') as double), 0)"


def valid_sql(col: str) -> str:
    nulls = ", ".join(sql_lit(x) for x in sorted(NULL_WORDS | {"sem valor"}))
    expr = f"lower(trim(cast({col} as varchar)))"
    return f"{expr} not in ({nulls}) and {expr} not like '%sem valor%'"


def discrete_sql_value(col: str) -> str:
    return f"coalesce(nullif(trim(cast({col} as varchar)), ''), 'SEM_VALOR')"


def filter_expr_by_combo(table_expr: str, combo: dict[str, str]) -> str:
    if "__hash_col" in combo:
        col = safe_text(combo.get("__hash_col", "cd_municipio")) or "cd_municipio"
        mod = int(combo.get("__hash_mod", RESULTADOS_HASH_BUCKETS) or RESULTADOS_HASH_BUCKETS)
        bucket = int(combo.get("__hash_bucket", 0) or 0)
        condition = f"(hash({discrete_sql_value(col)}) % {mod}) = {bucket}"
        return f"(select * from {table_expr} where {condition})"
    conditions = [
        f"{discrete_sql_value(col)} = {sql_lit(value or 'SEM_VALOR')}"
        for col, value in combo.items()
    ]
    if not conditions:
        return table_expr
    return f"(select * from {table_expr} where {' and '.join(conditions)})"


def split_combo_key(combo: dict[str, str]) -> str:
    if "__hash_col" in combo:
        col = safe_name(combo.get("__hash_col", "cd_municipio"), 24)
        mod = int(combo.get("__hash_mod", RESULTADOS_HASH_BUCKETS) or RESULTADOS_HASH_BUCKETS)
        bucket = int(combo.get("__hash_bucket", 0) or 0)
        return f"{col}_bucket_{bucket:03d}_de_{mod:03d}"
    pieces = []
    for col, value in combo.items():
        pieces.append(f"{safe_name(col, 20)}-{safe_name(value or 'SEM_VALOR', 35)}")
    return safe_name("__".join(pieces), 110) or "subparte"


def profile_combo_sql() -> str:
    parts = []
    labels = {
        "perfil_faixa_etaria": "faixa_etaria",
        "perfil_genero": "sexo_genero",
        "perfil_instrucao": "escolaridade",
        "perfil_estado_civil": "estado_civil",
        "perfil_raca_cor": "raca_cor",
    }
    for col, label in labels.items():
        parts.append(
            f"case when {valid_sql(col)} then '{label}=' || cast({col} as varchar) || '; ' else '' end"
        )
    return f"trim(trailing '; ' from concat({', '.join(parts)}))"


def section_keys(alias: str = "") -> list[str]:
    prefix = f"{alias}." if alias else ""
    return [f"{prefix}{c}" for c in ["ano", "uf", "cd_municipio", "zona", "secao", "cargo", "turno"]]


def timeline_sql(eleitorado_expr: str, scope_cols: str, label: str) -> str:
    scope = [c.strip() for c in scope_cols.split(",") if c.strip()]
    group_cols = ["ano", *scope]
    select_scope = ", ".join(group_cols)
    section_cols = []
    for col in [*group_cols, "uf", "cd_municipio", "nm_municipio", "zona", "secao"]:
        if col not in section_cols:
            section_cols.append(col)
    select_section = ", ".join(section_cols)
    return f"""
    with secao as (
      select {select_section},
             max({metric_sql('eleitorado')}) as eleitorado,
             max({metric_sql('comparecimento_estimado')}) as comparecimento_estimado,
             max({metric_sql('abstencao_estimado')}) as abstencao_estimado
      from {eleitorado_expr}
      group by all
    )
    select {select_scope},
           sum(eleitorado) as eleitorado,
           sum(comparecimento_estimado) as comparecimento_estimado,
           sum(abstencao_estimado) as abstencao_estimado,
           '{label}' as tipo_timeline
    from secao
    group by {select_scope}
    order by {select_scope}
    """


def retrato_municipal_sql(eleitorado_expr: str, resultados_expr: str) -> str:
    votos_join = ""
    votos_select = "0.0 as votos"
    if resultados_expr:
        votos_join = f"""
        left join (
          select ano, uf, cd_municipio, nm_municipio, sum({metric_sql('votos')}) as votos
          from {resultados_expr}
          group by all
        ) r using (ano, uf, cd_municipio, nm_municipio)
        """
        votos_select = "coalesce(r.votos, 0) as votos"
    return f"""
    with e as (
      select ano, uf, cd_municipio, nm_municipio,
             sum(eleitorado) as eleitorado,
             sum(comparecimento_estimado) as comparecimento_estimado,
             sum(abstencao_estimado) as abstencao_estimado
      from (
        select ano, uf, cd_municipio, nm_municipio, zona, secao,
               max({metric_sql('eleitorado')}) as eleitorado,
               max({metric_sql('comparecimento_estimado')}) as comparecimento_estimado,
               max({metric_sql('abstencao_estimado')}) as abstencao_estimado
        from {eleitorado_expr}
        group by all
      )
      group by all
    )
    select e.*, {votos_select}
    from e
    {votos_join}
    order by uf, nm_municipio, ano
    """


def perfil_eleitor_por_ano_sql(eleitorado_expr: str) -> str:
    profile_union = profile_union_sql(eleitorado_expr, include_location=False)
    return f"""
    with perfis as ({profile_union}),
    agg as (
      select ano, dimensao_perfil, valor_perfil, sum(eleitorado) as eleitorado
      from perfis
      where eleitorado > 0
      group by all
    ),
    ranked as (
      select *,
             eleitorado / nullif(sum(eleitorado) over(partition by ano, dimensao_perfil), 0) as share_eleitorado_ano,
             row_number() over(partition by ano, dimensao_perfil order by eleitorado desc) as rank_dimensao_ano
      from agg
    )
    select * from ranked
    where {valid_sql('valor_perfil')}
    order by ano, dimensao_perfil, rank_dimensao_ano
    """


def perfil_eleitor_por_ano_parts_sql(eleitorado_expr: str) -> str:
    profile_union = profile_union_sql(eleitorado_expr, include_location=False)
    return f"""
    with perfis as ({profile_union})
    select ano, dimensao_perfil, valor_perfil, sum(eleitorado) as eleitorado
    from perfis
    where eleitorado > 0 and {valid_sql('valor_perfil')}
    group by all
    """


def perfil_eleitor_por_ano_final_sql(parts_expr: str) -> str:
    return f"""
    with agg as (
      select ano, dimensao_perfil, valor_perfil, sum({metric_sql('eleitorado')}) as eleitorado
      from {parts_expr}
      where {valid_sql('valor_perfil')}
      group by all
    ),
    ranked as (
      select *,
             eleitorado / nullif(sum(eleitorado) over(partition by ano, dimensao_perfil), 0) as share_eleitorado_ano,
             row_number() over(partition by ano, dimensao_perfil order by eleitorado desc) as rank_dimensao_ano
      from agg
    )
    select * from ranked
    order by ano, dimensao_perfil, rank_dimensao_ano
    """


def profile_union_sql(eleitorado_expr: str, include_location: bool = True) -> str:
    base_cols = "ano, uf, cd_municipio, nm_municipio, zona, secao, cargo, turno" if include_location else "ano"
    frames = []
    for col in PROFILE_COLS:
        dim = col.replace("perfil_", "")
        frames.append(f"""
        select {base_cols},
               '{dim}' as dimensao_perfil,
               cast({col} as varchar) as valor_perfil,
               max({metric_sql('eleitorado')}) as eleitorado
        from {eleitorado_expr}
        where {valid_sql(col)}
        group by all
        """)
    return "\nunion all\n".join(frames)


def top10_perfis_sql(eleitorado_expr: str) -> str:
    combo = profile_combo_sql()
    return f"""
    with base as (
      select ano, uf, cd_municipio, nm_municipio, {combo} as perfil_combinado,
             sum({metric_sql('eleitorado')}) as eleitorado
      from {eleitorado_expr}
      group by all
    ),
    valid as (
      select * from base where {valid_sql('perfil_combinado')} and eleitorado > 0
    ),
    levels as (
      select 'brasil' as nivel, ano, '' as uf, '' as cd_municipio, '' as nm_municipio, perfil_combinado, sum(eleitorado) as eleitorado
      from valid group by all
      union all
      select 'estado' as nivel, ano, uf, '' as cd_municipio, '' as nm_municipio, perfil_combinado, sum(eleitorado) as eleitorado
      from valid where {valid_sql('uf')} group by all
      union all
      select 'municipio' as nivel, ano, uf, cd_municipio, nm_municipio, perfil_combinado, sum(eleitorado) as eleitorado
      from valid where {valid_sql('cd_municipio')} group by all
    ),
    ranked as (
      select *,
             eleitorado / nullif(sum(eleitorado) over(partition by nivel, ano, uf, cd_municipio), 0) as share_perfil,
             row_number() over(partition by nivel, ano, uf, cd_municipio order by eleitorado desc) as rank_perfil_ano
      from levels
    )
    select *,
           'Perfil ' || perfil_combinado || ' representa ' || round(share_perfil * 100, 2)::varchar || '% do eleitorado.' as descricao
    from ranked
    where rank_perfil_ano <= 10
    order by nivel, uf, nm_municipio, ano, rank_perfil_ano
    """


def top10_perfis_parts_sql(eleitorado_expr: str) -> str:
    combo = profile_combo_sql()
    return f"""
    with base as (
      select ano, uf, cd_municipio, nm_municipio, {combo} as perfil_combinado,
             sum({metric_sql('eleitorado')}) as eleitorado
      from {eleitorado_expr}
      group by all
    )
    select *
    from base
    where {valid_sql('perfil_combinado')} and eleitorado > 0
    """


def top10_perfis_from_parts_sql(parts_expr: str) -> str:
    return f"""
    with valid as (
      select ano, uf, cd_municipio, nm_municipio, perfil_combinado,
             sum({metric_sql('eleitorado')}) as eleitorado
      from {parts_expr}
      where {valid_sql('perfil_combinado')} and {metric_sql('eleitorado')} > 0
      group by all
    ),
    levels as (
      select 'brasil' as nivel, ano, '' as uf, '' as cd_municipio, '' as nm_municipio, perfil_combinado, sum(eleitorado) as eleitorado
      from valid group by all
      union all
      select 'estado' as nivel, ano, uf, '' as cd_municipio, '' as nm_municipio, perfil_combinado, sum(eleitorado) as eleitorado
      from valid where {valid_sql('uf')} group by all
      union all
      select 'municipio' as nivel, ano, uf, cd_municipio, nm_municipio, perfil_combinado, sum(eleitorado) as eleitorado
      from valid where {valid_sql('cd_municipio')} group by all
    ),
    ranked as (
      select *,
             eleitorado / nullif(sum(eleitorado) over(partition by nivel, ano, uf, cd_municipio), 0) as share_perfil,
             row_number() over(partition by nivel, ano, uf, cd_municipio order by eleitorado desc) as rank_perfil_ano
      from levels
    )
    select *,
           'Perfil ' || perfil_combinado || ' representa ' || round(share_perfil * 100, 2)::varchar || '% do eleitorado.' as descricao
    from ranked
    where rank_perfil_ano <= 10
    order by nivel, uf, nm_municipio, ano, rank_perfil_ano
    """


def timeline_nacional_from_timeline_uf_sql(timeline_uf_expr: str) -> str:
    return f"""
    select ano,
           sum({metric_sql('eleitorado')}) as eleitorado,
           sum({metric_sql('comparecimento_estimado')}) as comparecimento_estimado,
           sum({metric_sql('abstencao_estimado')}) as abstencao_estimado,
           'timeline_nacional' as tipo_timeline
    from {timeline_uf_expr}
    group by ano
    order by ano
    """


def vencedores_secao_sql(resultados_expr: str) -> str:
    key_cols = ["ano", "uf", "cd_municipio", "nm_municipio", "zona", "secao", "cargo", "turno"]
    keys = ", ".join(key_cols)
    key_select = ", ".join(f"{discrete_sql_value(col)} as {col}" for col in key_cols)
    return f"""
    with votos as (
      select {key_select},
             {discrete_sql_value('partido')} as partido,
             {discrete_sql_value('candidato')} as candidato,
             {discrete_sql_value('nr_votavel')} as nr_votavel,
             sum({metric_sql('votos')}) as votos
      from {resultados_expr}
      where {metric_sql('votos')} > 0
      group by all
    ),
    ranked as (
      select *,
             sum(votos) over(partition by {keys}) as votos_total_secao,
             row_number() over(partition by {keys} order by votos desc) as rank_secao
      from votos
    )
    select {keys},
           partido as partido_vencedor,
           candidato as candidato_vencedor,
           nr_votavel,
           votos as votos_vencedor,
           votos_total_secao,
           votos / nullif(votos_total_secao, 0) as share_vencedor
    from ranked
    where rank_secao = 1
    """


def resultado_eleitorado_secao_sql(eleitorado_expr: str, vencedores_path: Path) -> str:
    winners = dataset_expr(vencedores_path)
    combo = profile_combo_sql()
    join_keys = "ano, uf, cd_municipio, zona, secao, cargo, turno"
    return f"""
    with perfil as (
      select {join_keys}, nm_municipio, {combo} as perfil_predominante_secao,
             sum({metric_sql('eleitorado')}) as eleitorado_secao
      from {eleitorado_expr}
      group by all
    )
    select v.*, p.nm_municipio, p.perfil_predominante_secao, p.eleitorado_secao
    from {winners} v
    left join perfil p using ({join_keys})
    """


def correlacao_perfil_secao_cache_sql(eleitorado_expr: str) -> str:
    keys = ["ano", "uf", "cd_municipio", "zona", "secao", "cargo", "turno"]
    keys_sql = ", ".join(keys)
    combo = profile_combo_sql()
    profile_or = " or ".join(valid_sql(col) for col in PROFILE_COLS)
    return f"""
    with secao as (
      select {keys_sql},
             any_value(nm_municipio) as nm_municipio,
             any_value(local_votacao) as local_votacao,
             any_value(bairro) as bairro,
             max({metric_sql('eleitorado')}) as eleitorado,
             max({metric_sql('comparecimento_estimado')}) as comparecimento_estimado,
             max({metric_sql('abstencao_estimado')}) as abstencao_estimado
      from {eleitorado_expr}
      group by all
    ),
    perfil as (
      select {keys_sql},
             perfil_faixa_etaria,
             perfil_genero,
             perfil_instrucao,
             perfil_estado_civil,
             perfil_raca_cor,
             {combo} as perfil_combinado,
             sum({metric_sql('eleitorado')}) as eleitorado_perfil
      from {eleitorado_expr}
      where ({profile_or}) and {metric_sql('eleitorado')} > 0
      group by all
    ),
    ranked as (
      select p.*,
             s.nm_municipio,
             s.local_votacao,
             s.bairro,
             s.eleitorado,
             s.comparecimento_estimado,
             s.abstencao_estimado,
             row_number() over(partition by {", ".join("p." + key for key in keys)} order by p.eleitorado_perfil desc) as rn_perfil_secao
      from perfil p
      left join secao s using ({keys_sql})
      where {valid_sql('perfil_combinado')}
    )
    select *
    from ranked
    """


def correlacao_resultado_secao_cache_sql(resultados_expr: str) -> str:
    keys = ["ano", "uf", "cd_municipio", "zona", "secao", "cargo", "turno"]
    keys_sql = ", ".join(keys)
    return f"""
    select {keys_sql},
           any_value(nm_municipio) as nm_municipio,
           {discrete_sql_value('partido')} as partido,
           {discrete_sql_value('candidato')} as candidato,
           {discrete_sql_value('nr_votavel')} as nr_votavel,
           sum({metric_sql('votos')}) as votos,
           sum({metric_sql('brancos')}) as brancos,
           sum({metric_sql('nulos')}) as nulos,
           sum({metric_sql('validos_estimados')}) as validos_estimados
    from {resultados_expr}
    where {metric_sql('votos')} > 0
    group by all
    """


def candidatos_secao_prata_minima_sql(candidatos_expr: str) -> str:
    return f"""
    select ano,
           uf,
           cd_municipio,
           any_value(nm_municipio) as nm_municipio,
           zona,
           secao,
           cargo,
           turno,
           {discrete_sql_value('partido')} as partido,
           {discrete_sql_value('candidato')} as candidato,
           {discrete_sql_value('nr_candidato')} as nr_candidato,
           {discrete_sql_value('sq_candidato')} as sq_candidato,
           {discrete_sql_value('perfil_faixa_etaria')} as candidato_faixa_etaria,
           {discrete_sql_value('perfil_genero')} as candidato_genero,
           {discrete_sql_value('perfil_instrucao')} as candidato_instrucao,
           {discrete_sql_value('perfil_estado_civil')} as candidato_estado_civil,
           {discrete_sql_value('perfil_raca_cor')} as candidato_raca_cor,
           {discrete_sql_value('situacao_candidatura')} as situacao_candidatura,
           {discrete_sql_value('resultado_candidatura')} as resultado_candidatura,
           count(*) as qtd_registros_candidato
    from {candidatos_expr}
    where {valid_sql('candidato')} or {valid_sql('nr_candidato')} or {valid_sql('partido')}
    group by ano, uf, cd_municipio, zona, secao, cargo, turno, partido, candidato, nr_candidato, sq_candidato,
             candidato_faixa_etaria, candidato_genero, candidato_instrucao, candidato_estado_civil, candidato_raca_cor,
             situacao_candidatura, resultado_candidatura
    """


def base_gold_global_from_cache_sql(perfil_expr: str, resultados_expr: str) -> str:
    keys = ["ano", "uf", "cd_municipio", "zona", "secao", "cargo", "turno"]
    keys_sql = ", ".join(keys)
    combo = profile_combo_sql()
    profile_or = " or ".join(valid_sql(col) for col in PROFILE_COLS)
    return f"""
    with p_secao as (
      select {keys_sql},
             any_value(nm_municipio) as nm_municipio,
             any_value(local_votacao) as local_votacao,
             any_value(bairro) as bairro,
             max({metric_sql('eleitorado')}) as eleitorado,
             max({metric_sql('comparecimento_estimado')}) as comparecimento_estimado,
             max({metric_sql('abstencao_estimado')}) as abstencao_estimado
      from {perfil_expr}
      group by all
    ),
    p_agg as (
      select {keys_sql},
             perfil_faixa_etaria,
             perfil_genero,
             perfil_instrucao,
             perfil_estado_civil,
             perfil_raca_cor,
             {combo} as perfil_combinado,
             sum({metric_sql('eleitorado_perfil')}) as eleitorado_perfil
      from {perfil_expr}
      where ({profile_or}) and {metric_sql('eleitorado_perfil')} > 0
      group by all
    ),
    p_ranked as (
      select *,
             row_number() over(partition by {keys_sql} order by eleitorado_perfil desc) as rn
      from p_agg
    ),
    p as (
      select * exclude(rn)
      from p_ranked
      where rn = 1
    ),
    r as (
      select {keys_sql},
             any_value(nm_municipio) as nm_municipio,
             partido,
             candidato,
             sum({metric_sql('votos')}) as votos,
             sum({metric_sql('brancos')}) as brancos,
             sum({metric_sql('nulos')}) as nulos,
             sum({metric_sql('validos_estimados')}) as validos_estimados
      from {resultados_expr}
      where {metric_sql('votos')} > 0
      group by all
    )
    select r.ano,
           r.uf,
           r.cd_municipio,
           coalesce(nullif(r.nm_municipio, ''), p_secao.nm_municipio) as nm_municipio,
           r.zona,
           r.secao,
           p_secao.local_votacao,
           p_secao.bairro,
           r.turno,
           r.cargo,
           r.partido,
           r.candidato,
           r.partido as entidade,
           p.perfil_faixa_etaria,
           p.perfil_genero,
           p.perfil_instrucao,
           p.perfil_estado_civil,
           p.perfil_raca_cor,
           r.votos,
           p_secao.eleitorado,
           p_secao.comparecimento_estimado,
           p_secao.abstencao_estimado,
           r.brancos,
           r.nulos,
           r.validos_estimados,
           'banco_eleitoral_ouro_cache' as aggregation_mode
    from r
    left join p_secao using ({keys_sql})
    left join p using ({keys_sql})
    """


def resultado_eleitorado_secao_from_cache_sql(perfil_expr: str, vencedores_path: Path) -> str:
    winners = dataset_expr(vencedores_path)
    join_keys = "ano, uf, cd_municipio, zona, secao, cargo, turno"
    combo = profile_combo_sql()
    profile_or = " or ".join(valid_sql(col) for col in PROFILE_COLS)
    return f"""
    with secao as (
      select {join_keys},
             any_value(nm_municipio) as nm_municipio_cache,
             max({metric_sql('eleitorado')}) as eleitorado_secao
      from {perfil_expr}
      group by all
    ),
    perfil_agg as (
      select {join_keys},
             {combo} as perfil_predominante_secao,
             sum({metric_sql('eleitorado_perfil')}) as eleitorado_perfil
      from {perfil_expr}
      where ({profile_or}) and {metric_sql('eleitorado_perfil')} > 0
      group by all
    ),
    perfil_ranked as (
      select *,
             row_number() over(partition by {join_keys} order by eleitorado_perfil desc) as rn
      from perfil_agg
    ),
    perfil as (
      select s.{join_keys.replace(', ', ', s.')},
             s.nm_municipio_cache,
             p.perfil_predominante_secao,
             s.eleitorado_secao
      from secao s
      left join perfil_ranked p using ({join_keys})
      where coalesce(p.rn, 1) = 1
    )
    select v.ano,
           v.uf,
           v.cd_municipio,
           coalesce(nullif(v.nm_municipio, ''), p.nm_municipio_cache) as nm_municipio,
           v.zona,
           v.secao,
           v.cargo,
           v.turno,
           v.partido_vencedor,
           v.candidato_vencedor,
           v.nr_votavel,
           v.votos_vencedor,
           v.votos_total_secao,
           v.share_vencedor,
           p.perfil_predominante_secao,
           p.eleitorado_secao
    from {winners} v
    left join perfil p using ({join_keys})
    """


def perfil_entidade_parts_from_cache_sql(perfil_expr: str, resultados_expr: str, entity_col: str) -> str:
    keys = ["ano", "uf", "cd_municipio", "zona", "secao", "cargo", "turno"]
    keys_sql = ", ".join(keys)
    combo = profile_combo_sql()
    return f"""
    with r as (
      select {keys_sql},
             {entity_col} as entidade,
             sum({metric_sql('votos')}) as votos
      from {resultados_expr}
      where {valid_sql(entity_col)} and {metric_sql('votos')} > 0
      group by all
    ),
    shares as (
      select *,
             votos / nullif(sum(votos) over(partition by {keys_sql}), 0) as share_secao
      from r
    ),
    p as (
      select *,
             {combo} as perfil_combinado
      from {perfil_expr}
    ),
    joined as (
      select p.ano,
             p.uf,
             p.cd_municipio,
             p.nm_municipio,
             s.cargo,
             s.turno,
             s.entidade,
             p.perfil_combinado,
             p.eleitorado_perfil * s.share_secao as votos_proxy
      from p
      inner join shares s using ({keys_sql})
      where {valid_sql('perfil_combinado')} and {valid_sql('entidade')}
    )
    select ano, uf, cd_municipio, nm_municipio, cargo, turno, entidade, perfil_combinado,
           sum(votos_proxy) as votos_proxy
    from joined
    group by all
    """


def perfil_entidade_estado_source_sql(parts_expr: str, entity_col: str) -> str:
    return f"""
    select ano,
           uf,
           cargo,
           turno,
           entidade,
           perfil_combinado,
           sum({metric_sql('votos_proxy')}) as votos_proxy,
           '{entity_col}' as tipo_entidade
    from {parts_expr}
    where {valid_sql('perfil_combinado')} and {valid_sql('entidade')} and {metric_sql('votos_proxy')} > 0
    group by all
    """


def perfil_entidade_top_local_sql(parts_expr: str, entity_col: str) -> str:
    return f"""
    with joined as (
      select ano,
             uf,
             cd_municipio,
             nm_municipio,
             cargo,
             turno,
             entidade,
             perfil_combinado,
             sum({metric_sql('votos_proxy')}) as votos_proxy
      from {parts_expr}
      where {valid_sql('perfil_combinado')} and {valid_sql('entidade')} and {metric_sql('votos_proxy')} > 0
      group by all
    ),
    levels as (
      select 'estado' as nivel,
             ano,
             uf,
             '' as cd_municipio,
             '' as nm_municipio,
             cargo,
             turno,
             entidade,
             perfil_combinado,
             sum(votos_proxy) as votos
      from joined
      group by all
      union all
      select 'municipio' as nivel,
             ano,
             uf,
             cd_municipio,
             nm_municipio,
             cargo,
             turno,
             entidade,
             perfil_combinado,
             sum(votos_proxy) as votos
      from joined
      group by all
    ),
    ranked as (
      select *,
             votos / nullif(sum(votos) over(partition by nivel, ano, uf, cd_municipio, cargo, turno, entidade), 0) as share_perfil_na_entidade,
             row_number() over(partition by nivel, ano, uf, cd_municipio, cargo, turno, entidade order by votos desc) as rank_perfil_entidade_ano
      from levels
    )
    select nivel,
           ano,
           uf,
           cd_municipio,
           nm_municipio,
           cargo,
           turno,
           entidade,
           perfil_combinado,
           votos,
           share_perfil_na_entidade,
           rank_perfil_entidade_ano,
           '{entity_col}' as tipo_entidade
    from ranked
    where rank_perfil_entidade_ano <= 10
    """


def perfil_entidade_final_from_state_sql(state_expr: str, local_top_expr: str, entity_col: str) -> str:
    return f"""
    with estado_source as (
      select ano,
             cargo,
             turno,
             entidade,
             perfil_combinado,
             sum({metric_sql('votos_proxy')}) as votos_proxy
      from {state_expr}
      where {valid_sql('perfil_combinado')} and {valid_sql('entidade')} and {metric_sql('votos_proxy')} > 0
      group by all
    ),
    brasil as (
      select 'brasil' as nivel,
             ano,
             '' as uf,
             '' as cd_municipio,
             '' as nm_municipio,
             cargo,
             turno,
             entidade,
             perfil_combinado,
             sum(votos_proxy) as votos
      from estado_source
      group by all
    ),
    brasil_ranked as (
      select *,
             votos / nullif(sum(votos) over(partition by nivel, ano, cargo, turno, entidade), 0) as share_perfil_na_entidade,
             row_number() over(partition by nivel, ano, cargo, turno, entidade order by votos desc) as rank_perfil_entidade_ano
      from brasil
    ),
    local as (
      select nivel,
             ano,
             uf,
             cd_municipio,
             nm_municipio,
             cargo,
             turno,
             entidade,
             perfil_combinado,
             {metric_sql('votos')} as votos,
             {metric_sql('share_perfil_na_entidade')} as share_perfil_na_entidade,
             cast({metric_sql('rank_perfil_entidade_ano')} as bigint) as rank_perfil_entidade_ano,
             '{entity_col}' as tipo_entidade
      from {local_top_expr}
      where nivel in ('estado', 'municipio')
    )
    select nivel,
           ano,
           uf,
           cd_municipio,
           nm_municipio,
           cargo,
           turno,
           entidade,
           perfil_combinado,
           votos,
           share_perfil_na_entidade,
           rank_perfil_entidade_ano,
           '{entity_col}' as tipo_entidade
    from brasil_ranked
    where rank_perfil_entidade_ano <= 10
    union all
    select nivel,
           ano,
           uf,
           cd_municipio,
           nm_municipio,
           cargo,
           turno,
           entidade,
           perfil_combinado,
           votos,
           share_perfil_na_entidade,
           rank_perfil_entidade_ano,
           tipo_entidade
    from local
    order by nivel, uf, nm_municipio, ano, entidade, rank_perfil_entidade_ano
    """


def base_gold_global_sql(eleitorado_expr: str, resultados_expr: str) -> str:
    join_keys = ["ano", "uf", "cd_municipio", "zona", "secao", "cargo", "turno"]
    keys_sql = ", ".join(join_keys)
    profile_or = " or ".join(valid_sql(col) for col in PROFILE_COLS)
    return f"""
    with e_setor as (
      select {keys_sql},
             any_value(nm_municipio) as nm_municipio,
             any_value(local_votacao) as local_votacao,
             any_value(bairro) as bairro,
             max({metric_sql('eleitorado')}) as eleitorado,
             max({metric_sql('comparecimento_estimado')}) as comparecimento_estimado,
             max({metric_sql('abstencao_estimado')}) as abstencao_estimado
      from {eleitorado_expr}
      group by all
    ),
    e_perfil_ranked as (
      select {keys_sql},
             perfil_faixa_etaria,
             perfil_genero,
             perfil_instrucao,
             perfil_estado_civil,
             perfil_raca_cor,
             {metric_sql('eleitorado')} as eleitorado_perfil,
             row_number() over(partition by {keys_sql} order by {metric_sql('eleitorado')} desc) as rn
      from {eleitorado_expr}
      where ({profile_or}) and {metric_sql('eleitorado')} > 0
    ),
    e_perfil as (
      select * exclude(rn) from e_perfil_ranked where rn = 1
    ),
    r as (
      select {keys_sql},
             any_value(nm_municipio) as nm_municipio,
             partido,
             candidato,
             sum({metric_sql('votos')}) as votos,
             sum({metric_sql('brancos')}) as brancos,
             sum({metric_sql('nulos')}) as nulos,
             sum({metric_sql('validos_estimados')}) as validos_estimados
      from {resultados_expr}
      where {metric_sql('votos')} > 0
      group by all
    )
    select r.ano,
           r.uf,
           r.cd_municipio,
           coalesce(nullif(r.nm_municipio, ''), e_setor.nm_municipio) as nm_municipio,
           r.zona,
           r.secao,
           e_setor.local_votacao,
           e_setor.bairro,
           r.turno,
           r.cargo,
           r.partido,
           r.candidato,
           r.partido as entidade,
           e_perfil.perfil_faixa_etaria,
           e_perfil.perfil_genero,
           e_perfil.perfil_instrucao,
           e_perfil.perfil_estado_civil,
           e_perfil.perfil_raca_cor,
           r.votos,
           e_setor.eleitorado,
           e_setor.comparecimento_estimado,
           e_setor.abstencao_estimado,
           r.brancos,
           r.nulos,
           r.validos_estimados,
           'banco_eleitoral_ouro' as aggregation_mode
    from r
    left join e_setor using ({keys_sql})
    left join e_perfil using ({keys_sql})
    """


def perfil_entidade_sql(eleitorado_expr: str, resultados_expr: str, entity_col: str) -> str:
    combo = profile_combo_sql()
    keys = ", ".join(["ano", "uf", "cd_municipio", "zona", "secao", "cargo", "turno"])
    return f"""
    with r as (
      select {keys}, {entity_col} as entidade, sum({metric_sql('votos')}) as votos
      from {resultados_expr}
      where {valid_sql(entity_col)} and {metric_sql('votos')} > 0
      group by all
    ),
    shares as (
      select *, votos / nullif(sum(votos) over(partition by {keys}), 0) as share_secao
      from r
    ),
    p as (
      select {keys}, nm_municipio, {combo} as perfil_combinado,
             sum({metric_sql('eleitorado')}) as eleitorado
      from {eleitorado_expr}
      group by all
    ),
    joined as (
      select p.ano, p.uf, p.cd_municipio, p.nm_municipio, p.perfil_combinado,
             s.cargo, s.turno, s.entidade,
             p.eleitorado * s.share_secao as votos_proxy
      from p
      inner join shares s using ({keys})
      where {valid_sql('perfil_combinado')} and {valid_sql('entidade')}
    ),
    levels as (
      select 'brasil' as nivel, ano, '' as uf, '' as cd_municipio, '' as nm_municipio,
             cargo, turno, entidade, perfil_combinado, sum(votos_proxy) as votos
      from joined group by all
      union all
      select 'estado' as nivel, ano, uf, '' as cd_municipio, '' as nm_municipio,
             cargo, turno, entidade, perfil_combinado, sum(votos_proxy) as votos
      from joined group by all
      union all
      select 'municipio' as nivel, ano, uf, cd_municipio, nm_municipio,
             cargo, turno, entidade, perfil_combinado, sum(votos_proxy) as votos
      from joined group by all
    ),
    ranked as (
      select *,
             votos / nullif(sum(votos) over(partition by nivel, ano, uf, cd_municipio, cargo, turno, entidade), 0) as share_perfil_na_entidade,
             row_number() over(partition by nivel, ano, uf, cd_municipio, cargo, turno, entidade order by votos desc) as rank_perfil_entidade_ano
      from levels
    )
    select *, '{entity_col}' as tipo_entidade
    from ranked
    where rank_perfil_entidade_ano <= 10
    order by nivel, uf, nm_municipio, ano, entidade, rank_perfil_entidade_ano
    """


def filter_uf_expr(table_expr: str, uf: str) -> str:
    value = safe_text(uf, "SEM_UF") or "SEM_UF"
    return f"(select * from {table_expr} where {discrete_sql_value('uf')} = {sql_lit(value)})"


def filter_municipio_expr(table_expr: str, cd_municipio: str) -> str:
    value = safe_text(cd_municipio, "SEM_VALOR") or "SEM_VALOR"
    return f"(select * from {table_expr} where {discrete_sql_value('cd_municipio')} = {sql_lit(value)})"


def list_municipios_for_municipal_slice(perfil_expr: str, resultados_expr: str, cfg: CleanDatabaseConfig) -> list[dict[str, str]]:
    import duckdb

    sql = f"""
    with perfil_secao as (
      select {discrete_sql_value('cd_municipio')} as cd_municipio,
             max({discrete_sql_value('nm_municipio')}) as nm_municipio,
             zona,
             secao,
             cargo,
             turno,
             max({metric_sql('eleitorado')}) as eleitorado_secao,
             count(*) as linhas
      from {perfil_expr}
      where {valid_sql('cd_municipio')}
      group by 1, zona, secao, cargo, turno
    ),
    municipios as (
      select cd_municipio,
             max(nm_municipio) as nm_municipio,
             sum(linhas) as linhas,
             sum(eleitorado_secao) as eleitorado_estimado
      from perfil_secao
      group by 1
      union all
      select {discrete_sql_value('cd_municipio')} as cd_municipio,
             max({discrete_sql_value('nm_municipio')}) as nm_municipio,
             count(*) as linhas,
             0.0 as eleitorado_estimado
      from {resultados_expr}
      where {valid_sql('cd_municipio')}
      group by 1
    )
    select cd_municipio,
           max(nullif(nm_municipio, 'SEM_VALOR')) as nm_municipio,
           sum(linhas) as linhas_total,
           sum(eleitorado_estimado) as eleitorado_estimado
    from municipios
    where cd_municipio <> 'SEM_VALOR'
    group by cd_municipio
    order by cd_municipio
    """
    started = time.perf_counter()
    con = duckdb.connect(database=":memory:")
    configure_duckdb_connection(con, 1)
    try:
        df = con.execute(sql).fetchdf()
    except Exception as exc:
        logging.warning("Nao consegui listar municipios da fatia municipal; usando fallback por UF. Erro: %s", exc)
        return []
    finally:
        con.close()
        clean_memory()
    municipios = [
        {
            "cd_municipio": safe_text(row.get("cd_municipio"), "SEM_VALOR") or "SEM_VALOR",
            "nm_municipio": safe_text(row.get("nm_municipio"), "") or "",
            "linhas_total": str(int(parse_number(row.get("linhas_total")) or 0)),
            "eleitorado_estimado": str(int(parse_number(row.get("eleitorado_estimado")) or 0)),
        }
        for row in df.to_dict(orient="records")
        if safe_text(row.get("cd_municipio"), "")
    ]
    logging.info("Municipios descobertos para processamento municipal: %s em %.1fs.", len(municipios), time.perf_counter() - started)
    return municipios


def municipal_resumo_nivel_sql(perfil_expr: str, resultados_expr: str) -> str:
    keys = ["ano", "uf", "cd_municipio", "zona", "secao", "cargo", "turno"]
    keys_sql = ", ".join(keys)
    return f"""
    with p as (
      select {keys_sql},
             any_value(nm_municipio) as nm_municipio,
             max({metric_sql('eleitorado')}) as eleitorado,
             max({metric_sql('comparecimento_estimado')}) as comparecimento_estimado,
             max({metric_sql('abstencao_estimado')}) as abstencao_estimado
      from {perfil_expr}
      group by all
    ),
    r as (
      select {keys_sql},
             any_value(nm_municipio) as nm_municipio_resultado,
             sum({metric_sql('votos')}) as votos,
             sum({metric_sql('brancos')}) as brancos,
             sum({metric_sql('nulos')}) as nulos,
             sum({metric_sql('validos_estimados')}) as validos_estimados
      from {resultados_expr}
      group by all
    ),
    joined as (
      select coalesce(p.ano, r.ano) as ano,
             coalesce(p.uf, r.uf) as uf,
             coalesce(p.cd_municipio, r.cd_municipio) as cd_municipio,
             coalesce(p.nm_municipio, r.nm_municipio_resultado) as nm_municipio,
             coalesce(p.zona, r.zona) as zona,
             coalesce(p.secao, r.secao) as secao,
             coalesce(p.cargo, r.cargo) as cargo,
             coalesce(p.turno, r.turno) as turno,
             coalesce(p.eleitorado, 0) as eleitorado,
             coalesce(p.comparecimento_estimado, 0) as comparecimento_estimado,
             coalesce(p.abstencao_estimado, 0) as abstencao_estimado,
             coalesce(r.votos, 0) as votos,
             coalesce(r.brancos, 0) as brancos,
             coalesce(r.nulos, 0) as nulos,
             coalesce(r.validos_estimados, 0) as validos_estimados
      from p
      full outer join r using ({keys_sql})
    )
    select 'municipio' as nivel,
           ano,
           uf,
           cd_municipio,
           any_value(nm_municipio) as nm_municipio,
           cargo,
           turno,
           1 as qtd_municipios,
           count(distinct zona || '|' || secao) as qtd_secoes,
           sum(eleitorado) as eleitorado,
           sum(comparecimento_estimado) as comparecimento_estimado,
           sum(abstencao_estimado) as abstencao_estimado,
           sum(votos) as votos,
           sum(brancos) as brancos,
           sum(nulos) as nulos,
           sum(validos_estimados) as validos_estimados,
           sum(abstencao_estimado) / nullif(sum(eleitorado), 0) as abstencao_media,
           sum(comparecimento_estimado) / nullif(sum(eleitorado), 0) as comparecimento_medio
    from joined
    where {valid_sql('cd_municipio')}
    group by ano, uf, cd_municipio, cargo, turno
    """


def municipal_perfil_eleitor_nivel_sql(perfil_expr: str) -> str:
    combo = profile_combo_sql()
    return f"""
    with agg as (
      select 'municipio' as nivel,
             ano,
             uf,
             cd_municipio,
             any_value(nm_municipio) as nm_municipio,
             perfil_faixa_etaria,
             perfil_genero,
             perfil_instrucao,
             perfil_estado_civil,
             perfil_raca_cor,
             {combo} as perfil_combinado,
             sum({metric_sql('eleitorado_perfil')}) as eleitorado
      from {perfil_expr}
      where ({' or '.join(valid_sql(col) for col in PROFILE_COLS)}) and {metric_sql('eleitorado_perfil')} > 0
      group by ano, uf, cd_municipio, perfil_faixa_etaria, perfil_genero, perfil_instrucao, perfil_estado_civil, perfil_raca_cor
    ),
    ranked as (
      select *,
             eleitorado / nullif(sum(eleitorado) over(partition by nivel, ano, uf, cd_municipio), 0) as share_perfil,
             row_number() over(partition by nivel, ano, uf, cd_municipio order by eleitorado desc) as rank_perfil_ano
      from agg
    )
    select *,
           'Eleitor predominante: ' || perfil_combinado || ' (' || round(share_perfil * 100, 2)::varchar || '%).' as descricao
    from ranked
    """


def municipal_resultado_entidade_nivel_sql(resultados_expr: str, entity_col: str) -> str:
    return f"""
    with agg as (
      select 'municipio' as nivel,
             ano,
             uf,
             cd_municipio,
             any_value(nm_municipio) as nm_municipio,
             cargo,
             turno,
             {discrete_sql_value(entity_col)} as entidade,
             sum({metric_sql('votos')}) as votos,
             sum({metric_sql('brancos')}) as brancos,
             sum({metric_sql('nulos')}) as nulos,
             sum({metric_sql('validos_estimados')}) as validos_estimados,
             '{entity_col}' as tipo_entidade
      from {resultados_expr}
      where {valid_sql(entity_col)} and {metric_sql('votos')} > 0
      group by ano, uf, cd_municipio, cargo, turno, entidade
    ),
    ranked as (
      select *,
             votos / nullif(sum(votos) over(partition by nivel, ano, uf, cd_municipio, cargo, turno), 0) as share_votos,
             row_number() over(partition by nivel, ano, uf, cd_municipio, cargo, turno order by votos desc) as rank_entidade
      from agg
    )
    select *
    from ranked
    where rank_entidade <= 50
    """


def municipal_perfil_secao_sql(perfil_expr: str) -> str:
    combo = profile_combo_sql()
    profile_or = " or ".join(valid_sql(col) for col in PROFILE_COLS)
    return f"""
    select 'secao' as nivel,
           ano,
           uf,
           cd_municipio,
           any_value(nm_municipio) as nm_municipio,
           zona,
           secao,
           cargo,
           turno,
           any_value(local_votacao) as local_votacao,
           any_value(bairro) as bairro,
           perfil_faixa_etaria,
           perfil_genero,
           perfil_instrucao,
           perfil_estado_civil,
           perfil_raca_cor,
           {combo} as perfil_combinado,
           sum({metric_sql('eleitorado_perfil')}) as eleitorado_perfil,
           max({metric_sql('eleitorado')}) as eleitorado_secao,
           max({metric_sql('comparecimento_estimado')}) as comparecimento_estimado,
           max({metric_sql('abstencao_estimado')}) as abstencao_estimado
    from {perfil_expr}
    where ({profile_or}) and {metric_sql('eleitorado_perfil')} > 0
    group by ano, uf, cd_municipio, zona, secao, cargo, turno,
             perfil_faixa_etaria, perfil_genero, perfil_instrucao, perfil_estado_civil, perfil_raca_cor
    """


def municipal_resumo_from_section_sql(resultado_eleitorado_expr: str) -> str:
    return f"""
    select 'municipio' as nivel,
           ano,
           uf,
           cd_municipio,
           any_value(nm_municipio) as nm_municipio,
           cargo,
           turno,
           1 as qtd_municipios,
           count(distinct zona || '|' || secao) as qtd_secoes,
           sum({metric_sql('eleitorado_secao')}) as eleitorado,
           sum({metric_sql('votos_total_secao')}) as votos,
           sum({metric_sql('votos_vencedor')}) as votos_vencedor,
           sum({metric_sql('eleitorado_secao')}) - sum({metric_sql('votos_total_secao')}) as abstencao_estimado,
           sum({metric_sql('votos_total_secao')}) as comparecimento_estimado,
           0.0 as brancos,
           0.0 as nulos,
           sum({metric_sql('votos_total_secao')}) as validos_estimados,
           (sum({metric_sql('eleitorado_secao')}) - sum({metric_sql('votos_total_secao')})) / nullif(sum({metric_sql('eleitorado_secao')}), 0) as abstencao_media,
           sum({metric_sql('votos_total_secao')}) / nullif(sum({metric_sql('eleitorado_secao')}), 0) as comparecimento_medio
    from {resultado_eleitorado_expr}
    where {valid_sql('cd_municipio')}
    group by ano, uf, cd_municipio, cargo, turno
    """


def municipal_resumo_from_prata_minima_sql(perfil_expr: str, resultados_expr: str) -> str:
    return f"""
    with e as (
      select ano,
             uf,
             cd_municipio,
             any_value(nm_municipio) as nm_municipio,
             cargo,
             turno,
             count(distinct zona || '|' || secao) as qtd_secoes,
             sum(eleitorado_secao) as eleitorado,
             sum(comparecimento_secao) as comparecimento_estimado,
             sum(abstencao_secao) as abstencao_estimado
      from (
        select ano,
               uf,
               cd_municipio,
               any_value(nm_municipio) as nm_municipio,
               zona,
               secao,
               cargo,
               turno,
               max({metric_sql('eleitorado')}) as eleitorado_secao,
               max({metric_sql('comparecimento_estimado')}) as comparecimento_secao,
               max({metric_sql('abstencao_estimado')}) as abstencao_secao
        from {perfil_expr}
        where {valid_sql('cd_municipio')}
        group by ano, uf, cd_municipio, zona, secao, cargo, turno
      )
      group by ano, uf, cd_municipio, cargo, turno
    ),
    r as (
      select ano,
             uf,
             cd_municipio,
             cargo,
             turno,
             sum({metric_sql('votos')}) as votos,
             sum({metric_sql('brancos')}) as brancos,
             sum({metric_sql('nulos')}) as nulos,
             sum({metric_sql('validos_estimados')}) as validos_estimados
      from {resultados_expr}
      where {valid_sql('cd_municipio')}
      group by ano, uf, cd_municipio, cargo, turno
    )
    select 'municipio' as nivel,
           e.ano,
           e.uf,
           e.cd_municipio,
           e.nm_municipio,
           e.cargo,
           e.turno,
           1 as qtd_municipios,
           e.qtd_secoes,
           e.eleitorado,
           coalesce(r.votos, 0.0) as votos,
           0.0 as votos_vencedor,
           coalesce(e.abstencao_estimado, e.eleitorado - coalesce(r.votos, 0.0)) as abstencao_estimado,
           coalesce(e.comparecimento_estimado, coalesce(r.votos, 0.0)) as comparecimento_estimado,
           coalesce(r.brancos, 0.0) as brancos,
           coalesce(r.nulos, 0.0) as nulos,
           coalesce(r.validos_estimados, r.votos, 0.0) as validos_estimados,
           coalesce(e.abstencao_estimado, e.eleitorado - coalesce(r.votos, 0.0)) / nullif(e.eleitorado, 0) as abstencao_media,
           coalesce(e.comparecimento_estimado, coalesce(r.votos, 0.0)) / nullif(e.eleitorado, 0) as comparecimento_medio
    from e
    left join r using (ano, uf, cd_municipio, cargo, turno)
    """


def municipal_perfil_eleitor_from_section_sql(perfil_secao_expr: str) -> str:
    return f"""
    with agg as (
      select 'municipio' as nivel,
             ano,
             uf,
             cd_municipio,
             any_value(nm_municipio) as nm_municipio,
             perfil_faixa_etaria,
             perfil_genero,
             perfil_instrucao,
             perfil_estado_civil,
             perfil_raca_cor,
             perfil_combinado,
             sum({metric_sql('eleitorado_perfil')}) as eleitorado
      from {perfil_secao_expr}
      where {valid_sql('perfil_combinado')} and {metric_sql('eleitorado_perfil')} > 0
      group by ano, uf, cd_municipio, perfil_faixa_etaria, perfil_genero, perfil_instrucao, perfil_estado_civil, perfil_raca_cor, perfil_combinado
    ),
    ranked as (
      select *,
             eleitorado / nullif(sum(eleitorado) over(partition by nivel, ano, uf, cd_municipio), 0) as share_perfil,
             row_number() over(partition by nivel, ano, uf, cd_municipio order by eleitorado desc) as rank_perfil_ano
      from agg
    )
    select *,
           'Eleitor predominante: ' || perfil_combinado || ' (' || round(share_perfil * 100, 2)::varchar || '%).' as descricao
    from ranked
    """


def municipal_perfil_eleitor_from_prata_minima_sql(perfil_expr: str) -> str:
    combo = profile_combo_sql()
    profile_or = " or ".join(valid_sql(col) for col in PROFILE_COLS)
    return f"""
    with agg as (
      select 'municipio' as nivel,
             ano,
             uf,
             cd_municipio,
             any_value(nm_municipio) as nm_municipio,
             perfil_faixa_etaria,
             perfil_genero,
             perfil_instrucao,
             perfil_estado_civil,
             perfil_raca_cor,
             {combo} as perfil_combinado,
             sum({metric_sql('eleitorado_perfil')}) as eleitorado
      from {perfil_expr}
      where ({profile_or}) and {metric_sql('eleitorado_perfil')} > 0 and {valid_sql('cd_municipio')}
      group by ano, uf, cd_municipio, perfil_faixa_etaria, perfil_genero, perfil_instrucao, perfil_estado_civil, perfil_raca_cor
    ),
    ranked as (
      select *,
             eleitorado / nullif(sum(eleitorado) over(partition by nivel, ano, uf, cd_municipio), 0) as share_perfil,
             row_number() over(partition by nivel, ano, uf, cd_municipio order by eleitorado desc) as rank_perfil_ano
      from agg
      where {valid_sql('perfil_combinado')}
    )
    select *,
           'Eleitor predominante: ' || perfil_combinado || ' (' || round(share_perfil * 100, 2)::varchar || '%).' as descricao
    from ranked
    """


def municipal_resultado_entidade_from_base_sql(base_secao_expr: str, entity_col: str) -> str:
    return f"""
    with agg as (
      select 'municipio' as nivel,
             ano,
             uf,
             cd_municipio,
             any_value(nm_municipio) as nm_municipio,
             cargo,
             turno,
             {discrete_sql_value(entity_col)} as entidade,
             sum({metric_sql('votos')}) as votos,
             sum({metric_sql('brancos')}) as brancos,
             sum({metric_sql('nulos')}) as nulos,
             sum({metric_sql('validos_estimados')}) as validos_estimados,
             '{entity_col}' as tipo_entidade
      from {base_secao_expr}
      where {valid_sql(entity_col)} and {metric_sql('votos')} > 0
      group by ano, uf, cd_municipio, cargo, turno, entidade
    ),
    ranked as (
      select *,
             votos / nullif(sum(votos) over(partition by nivel, ano, uf, cd_municipio, cargo, turno), 0) as share_votos,
             row_number() over(partition by nivel, ano, uf, cd_municipio, cargo, turno order by votos desc) as rank_entidade
      from agg
    )
    select *
    from ranked
    where rank_entidade <= 50
    """


def municipal_resultado_entidade_from_resultados_sql(resultados_expr: str, entity_col: str) -> str:
    return f"""
    with agg as (
      select 'municipio' as nivel,
             ano,
             uf,
             cd_municipio,
             any_value(nm_municipio) as nm_municipio,
             cargo,
             turno,
             {discrete_sql_value(entity_col)} as entidade,
             sum({metric_sql('votos')}) as votos,
             sum({metric_sql('brancos')}) as brancos,
             sum({metric_sql('nulos')}) as nulos,
             sum({metric_sql('validos_estimados')}) as validos_estimados,
             '{entity_col}' as tipo_entidade
      from {resultados_expr}
      where {valid_sql(entity_col)} and {metric_sql('votos')} > 0 and {valid_sql('cd_municipio')}
      group by ano, uf, cd_municipio, cargo, turno, entidade
    ),
    ranked as (
      select *,
             votos / nullif(sum(votos) over(partition by nivel, ano, uf, cd_municipio, cargo, turno), 0) as share_votos,
             row_number() over(partition by nivel, ano, uf, cd_municipio, cargo, turno order by votos desc) as rank_entidade
      from agg
    )
    select *
    from ranked
    where rank_entidade <= 50
    """


def municipal_perfil_entidade_from_base_sql(base_secao_expr: str, entity_col: str) -> str:
    return f"""
    with agg as (
      select 'municipio' as nivel,
             ano,
             uf,
             cd_municipio,
             any_value(nm_municipio) as nm_municipio,
             cargo,
             turno,
             {discrete_sql_value(entity_col)} as entidade,
             perfil_faixa_etaria,
             perfil_genero,
             perfil_instrucao,
             perfil_estado_civil,
             perfil_raca_cor,
             {profile_combo_sql()} as perfil_combinado,
             sum({metric_sql('votos')}) as votos,
             '{entity_col}' as tipo_entidade
      from {base_secao_expr}
      where {valid_sql(entity_col)} and {metric_sql('votos')} > 0
      group by ano, uf, cd_municipio, cargo, turno, entidade,
               perfil_faixa_etaria, perfil_genero, perfil_instrucao, perfil_estado_civil, perfil_raca_cor
    ),
    ranked as (
      select *,
             votos / nullif(sum(votos) over(partition by nivel, ano, uf, cd_municipio, cargo, turno, entidade), 0) as share_perfil_na_entidade,
             row_number() over(partition by nivel, ano, uf, cd_municipio, cargo, turno, entidade order by votos desc) as rank_perfil_entidade_ano
      from agg
      where {valid_sql('perfil_combinado')}
    )
    select *,
           'Perfil que mais aparece em ' || tipo_entidade || ' ' || entidade || ': ' || perfil_combinado || ' (' || round(share_perfil_na_entidade * 100, 2)::varchar || '%).' as descricao
    from ranked
    where rank_perfil_entidade_ano <= 10
    """


def municipal_perfil_entidade_nivel_sql(perfil_expr: str, resultados_expr: str, entity_col: str) -> str:
    parts_sql = perfil_entidade_parts_from_cache_sql(perfil_expr, resultados_expr, entity_col)
    return f"""
    with parts as ({parts_sql}),
    ranked as (
      select 'municipio' as nivel,
             ano,
             uf,
             cd_municipio,
             any_value(nm_municipio) as nm_municipio,
             cargo,
             turno,
             entidade,
             perfil_combinado,
             sum({metric_sql('votos_proxy')}) as votos,
             '{entity_col}' as tipo_entidade
      from parts
      where {valid_sql('perfil_combinado')} and {valid_sql('entidade')} and {metric_sql('votos_proxy')} > 0
      group by ano, uf, cd_municipio, cargo, turno, entidade, perfil_combinado
    ),
    final as (
      select *,
             votos / nullif(sum(votos) over(partition by nivel, ano, uf, cd_municipio, cargo, turno, entidade), 0) as share_perfil_na_entidade,
             row_number() over(partition by nivel, ano, uf, cd_municipio, cargo, turno, entidade order by votos desc) as rank_perfil_entidade_ano
      from ranked
    )
    select *,
           'Perfil que mais aparece em ' || tipo_entidade || ' ' || entidade || ': ' || perfil_combinado || ' (' || round(share_perfil_na_entidade * 100, 2)::varchar || '%).' as descricao
    from final
    where rank_perfil_entidade_ano <= 10
    """


def municipal_base_secao_sql(perfil_expr: str, resultados_expr: str) -> str:
    base_sql = base_gold_global_from_cache_sql(perfil_expr, resultados_expr)
    return f"""
    select 'municipio' as nivel, *
    from ({base_sql})
    """


def municipal_resultado_eleitorado_secao_sql(perfil_expr: str, resultados_expr: str) -> str:
    winners_sql = vencedores_secao_sql(resultados_expr)
    join_keys = "ano, uf, cd_municipio, zona, secao, cargo, turno"
    combo = profile_combo_sql()
    profile_or = " or ".join(valid_sql(col) for col in PROFILE_COLS)
    return f"""
    with vencedores as ({winners_sql}),
    secao as (
      select {join_keys},
             any_value(nm_municipio) as nm_municipio_cache,
             max({metric_sql('eleitorado')}) as eleitorado_secao
      from {perfil_expr}
      group by all
    ),
    perfil_agg as (
      select {join_keys},
             {combo} as perfil_predominante_secao,
             sum({metric_sql('eleitorado_perfil')}) as eleitorado_perfil
      from {perfil_expr}
      where ({profile_or}) and {metric_sql('eleitorado_perfil')} > 0
      group by all
    ),
    perfil_ranked as (
      select *,
             row_number() over(partition by {join_keys} order by eleitorado_perfil desc) as rn
      from perfil_agg
    ),
    perfil as (
      select s.ano,
             s.uf,
             s.cd_municipio,
             s.zona,
             s.secao,
             s.cargo,
             s.turno,
             s.nm_municipio_cache,
             p.perfil_predominante_secao,
             s.eleitorado_secao
      from secao s
      left join perfil_ranked p using ({join_keys})
      where coalesce(p.rn, 1) = 1
    )
    select 'municipio' as nivel,
           v.ano,
           v.uf,
           v.cd_municipio,
           coalesce(nullif(v.nm_municipio, ''), p.nm_municipio_cache) as nm_municipio,
           v.zona,
           v.secao,
           v.cargo,
           v.turno,
           v.partido_vencedor,
           v.candidato_vencedor,
           v.nr_votavel,
           v.votos_vencedor,
           v.votos_total_secao,
           v.share_vencedor,
           p.perfil_predominante_secao,
           p.eleitorado_secao
    from vencedores v
    left join perfil p using ({join_keys})
    """


def cluster_count_sql(cfg: CleanDatabaseConfig) -> int:
    min_k = max(2, int(getattr(cfg, "cluster_min_k", 2) or 2))
    max_k = max(min_k, int(getattr(cfg, "cluster_max_k", min_k) or min_k))
    return max(min_k, min(max_k, 12))


def municipal_clusters_eleitores_sql(perfil_expr: str, cfg: CleanDatabaseConfig) -> str:
    k = cluster_count_sql(cfg)
    combo = profile_combo_sql()
    return f"""
    with src as (
      select *,
             {combo} as perfil_combinado
      from {perfil_expr}
    ),
    agg as (
      select 'municipio' as nivel,
             ano,
             uf,
             cd_municipio,
             any_value(nm_municipio) as nm_municipio,
             cast(hash(perfil_combinado) % {k} as integer) as cluster_id,
             'eleitores_discretos' as cluster_tipo,
             perfil_faixa_etaria,
             perfil_genero,
             perfil_instrucao,
             perfil_estado_civil,
             perfil_raca_cor,
             perfil_combinado,
             '' as partido,
             sum({metric_sql('eleitorado_perfil')}) as eleitorado,
             0.0 as votos_proxy,
             'DiscreteProfileHash' as algoritmo_cluster,
             'sexo_genero, escolaridade, estado_civil, faixa_etaria, raca_cor' as tipo_features_cluster
      from src
      where {valid_sql('perfil_combinado')} and {metric_sql('eleitorado_perfil')} > 0
      group by ano, uf, cd_municipio, cluster_id, perfil_faixa_etaria, perfil_genero, perfil_instrucao, perfil_estado_civil, perfil_raca_cor, perfil_combinado
    ),
    final as (
      select *,
             eleitorado / nullif(sum(eleitorado) over(partition by nivel, ano, uf, cd_municipio), 0) as share_cluster,
             row_number() over(partition by nivel, ano, uf, cd_municipio, cluster_id order by eleitorado desc) as rank_persona_cluster
      from agg
    )
    select *,
           'Pessoa do cluster: ' || perfil_combinado || '.' as descricao
    from final
    where rank_persona_cluster <= 5
    """


def municipal_clusters_eleitores_resultado_sql(perfil_expr: str, resultados_expr: str, cfg: CleanDatabaseConfig) -> str:
    k = cluster_count_sql(cfg)
    parts_sql = perfil_entidade_parts_from_cache_sql(perfil_expr, resultados_expr, "partido")
    return f"""
    with parts as ({parts_sql}),
    agg as (
      select 'municipio' as nivel,
             ano,
             uf,
             cd_municipio,
             any_value(nm_municipio) as nm_municipio,
             cast(hash(perfil_combinado || '|' || entidade) % {k} as integer) as cluster_id,
             'eleitores_resultado_discreto' as cluster_tipo,
             '' as perfil_faixa_etaria,
             '' as perfil_genero,
             '' as perfil_instrucao,
             '' as perfil_estado_civil,
             '' as perfil_raca_cor,
             perfil_combinado,
             entidade as partido,
             0.0 as eleitorado,
             sum({metric_sql('votos_proxy')}) as votos_proxy,
             'DiscreteProfilePartyHash' as algoritmo_cluster,
             'perfil_eleitor_discreto + partido' as tipo_features_cluster
      from parts
      where {valid_sql('perfil_combinado')} and {valid_sql('entidade')} and {metric_sql('votos_proxy')} > 0
      group by ano, uf, cd_municipio, cluster_id, perfil_combinado, entidade
    ),
    final as (
      select *,
             votos_proxy / nullif(sum(votos_proxy) over(partition by nivel, ano, uf, cd_municipio), 0) as share_cluster,
             row_number() over(partition by nivel, ano, uf, cd_municipio, cluster_id order by votos_proxy desc) as rank_persona_cluster
      from agg
    )
    select *,
           'Pessoa do cluster: ' || perfil_combinado || '. Tendencia partidaria: ' || partido || '.' as descricao
    from final
    where rank_persona_cluster <= 5
    """


def aggregate_resumo_level_sql(source_expr: str, level: str) -> str:
    if level == "estado":
        group_cols = "ano, uf, cargo, turno"
        select_loc = "uf, '' as cd_municipio, '' as nm_municipio"
        partition = "nivel, ano, uf"
    else:
        group_cols = "ano, cargo, turno"
        select_loc = "'' as uf, '' as cd_municipio, '' as nm_municipio"
        partition = "nivel, ano"
    return f"""
    select '{level}' as nivel,
           ano,
           {select_loc},
           cargo,
           turno,
           sum({metric_sql('qtd_municipios')}) as qtd_municipios,
           sum({metric_sql('qtd_secoes')}) as qtd_secoes,
           sum({metric_sql('eleitorado')}) as eleitorado,
           sum({metric_sql('comparecimento_estimado')}) as comparecimento_estimado,
           sum({metric_sql('abstencao_estimado')}) as abstencao_estimado,
           sum({metric_sql('votos')}) as votos,
           sum({metric_sql('brancos')}) as brancos,
           sum({metric_sql('nulos')}) as nulos,
           sum({metric_sql('validos_estimados')}) as validos_estimados,
           sum({metric_sql('abstencao_estimado')}) / nullif(sum({metric_sql('eleitorado')}), 0) as abstencao_media,
           sum({metric_sql('comparecimento_estimado')}) / nullif(sum({metric_sql('eleitorado')}), 0) as comparecimento_medio
    from {source_expr}
    group by {group_cols}
    order by {partition}, cargo, turno
    """


def aggregate_perfil_eleitor_level_sql(source_expr: str, level: str) -> str:
    if level == "estado":
        group_cols = "ano, uf, perfil_faixa_etaria, perfil_genero, perfil_instrucao, perfil_estado_civil, perfil_raca_cor, perfil_combinado"
        select_loc = "uf, '' as cd_municipio, '' as nm_municipio"
        partition = "nivel, ano, uf"
    else:
        group_cols = "ano, perfil_faixa_etaria, perfil_genero, perfil_instrucao, perfil_estado_civil, perfil_raca_cor, perfil_combinado"
        select_loc = "'' as uf, '' as cd_municipio, '' as nm_municipio"
        partition = "nivel, ano"
    return f"""
    with agg as (
      select '{level}' as nivel,
             ano,
             {select_loc},
             perfil_faixa_etaria,
             perfil_genero,
             perfil_instrucao,
             perfil_estado_civil,
             perfil_raca_cor,
             perfil_combinado,
             sum({metric_sql('eleitorado')}) as eleitorado
      from {source_expr}
      where {valid_sql('perfil_combinado')} and {metric_sql('eleitorado')} > 0
      group by {group_cols}
    ),
    ranked as (
      select *,
             eleitorado / nullif(sum(eleitorado) over(partition by {partition}), 0) as share_perfil,
             row_number() over(partition by {partition} order by eleitorado desc) as rank_perfil_ano
      from agg
    )
    select *,
           'Eleitor predominante: ' || perfil_combinado || ' (' || round(share_perfil * 100, 2)::varchar || '%).' as descricao
    from ranked
    """


def aggregate_resultado_entidade_level_sql(source_expr: str, level: str) -> str:
    if level == "estado":
        group_cols = "ano, uf, cargo, turno, entidade, tipo_entidade"
        select_loc = "uf, '' as cd_municipio, '' as nm_municipio"
        partition = "nivel, ano, uf, cargo, turno, tipo_entidade"
    else:
        group_cols = "ano, cargo, turno, entidade, tipo_entidade"
        select_loc = "'' as uf, '' as cd_municipio, '' as nm_municipio"
        partition = "nivel, ano, cargo, turno, tipo_entidade"
    return f"""
    with agg as (
      select '{level}' as nivel,
             ano,
             {select_loc},
             cargo,
             turno,
             entidade,
             tipo_entidade,
             sum({metric_sql('votos')}) as votos,
             sum({metric_sql('brancos')}) as brancos,
             sum({metric_sql('nulos')}) as nulos,
             sum({metric_sql('validos_estimados')}) as validos_estimados
      from {source_expr}
      where {valid_sql('entidade')} and {metric_sql('votos')} > 0
      group by {group_cols}
    ),
    ranked as (
      select *,
             votos / nullif(sum(votos) over(partition by {partition}), 0) as share_votos,
             row_number() over(partition by {partition} order by votos desc) as rank_entidade
      from agg
    )
    select *
    from ranked
    where rank_entidade <= 50
    """


def aggregate_perfil_entidade_level_sql(source_expr: str, level: str, entity_col: str) -> str:
    if level == "estado":
        group_cols = "ano, uf, cargo, turno, entidade, perfil_combinado"
        select_loc = "uf, '' as cd_municipio, '' as nm_municipio"
        partition = "nivel, ano, uf, cargo, turno, entidade"
    else:
        group_cols = "ano, cargo, turno, entidade, perfil_combinado"
        select_loc = "'' as uf, '' as cd_municipio, '' as nm_municipio"
        partition = "nivel, ano, cargo, turno, entidade"
    return f"""
    with agg as (
      select '{level}' as nivel,
             ano,
             {select_loc},
             cargo,
             turno,
             entidade,
             perfil_combinado,
             sum({metric_sql('votos')}) as votos,
             '{entity_col}' as tipo_entidade
      from {source_expr}
      where {valid_sql('perfil_combinado')} and {valid_sql('entidade')} and {metric_sql('votos')} > 0
      group by {group_cols}
    ),
    ranked as (
      select *,
             votos / nullif(sum(votos) over(partition by {partition}), 0) as share_perfil_na_entidade,
             row_number() over(partition by {partition} order by votos desc) as rank_perfil_entidade_ano
      from agg
    )
    select *,
           'Perfil que mais aparece em ' || tipo_entidade || ' ' || entidade || ': ' || perfil_combinado || ' (' || round(share_perfil_na_entidade * 100, 2)::varchar || '%).' as descricao
    from ranked
    where rank_perfil_entidade_ano <= 10
    """


def aggregate_clusters_level_sql(source_expr: str, level: str) -> str:
    if level == "estado":
        group_cols = "ano, uf, cluster_id, cluster_tipo, perfil_faixa_etaria, perfil_genero, perfil_instrucao, perfil_estado_civil, perfil_raca_cor, perfil_combinado, partido, algoritmo_cluster, tipo_features_cluster"
        select_loc = "uf, '' as cd_municipio, '' as nm_municipio"
        partition = "nivel, ano, uf, cluster_tipo"
    else:
        group_cols = "ano, cluster_id, cluster_tipo, perfil_faixa_etaria, perfil_genero, perfil_instrucao, perfil_estado_civil, perfil_raca_cor, perfil_combinado, partido, algoritmo_cluster, tipo_features_cluster"
        select_loc = "'' as uf, '' as cd_municipio, '' as nm_municipio"
        partition = "nivel, ano, cluster_tipo"
    return f"""
    with agg as (
      select '{level}' as nivel,
             ano,
             {select_loc},
             cluster_id,
             cluster_tipo,
             perfil_faixa_etaria,
             perfil_genero,
             perfil_instrucao,
             perfil_estado_civil,
             perfil_raca_cor,
             perfil_combinado,
             partido,
             sum({metric_sql('eleitorado')}) as eleitorado,
             sum({metric_sql('votos_proxy')}) as votos_proxy,
             algoritmo_cluster,
             tipo_features_cluster
      from {source_expr}
      where {valid_sql('perfil_combinado')}
      group by {group_cols}
    ),
    scored as (
      select *,
             greatest(eleitorado, votos_proxy) as peso_cluster
      from agg
    ),
    ranked as (
      select *,
             peso_cluster / nullif(sum(peso_cluster) over(partition by {partition}), 0) as share_cluster,
             row_number() over(partition by {partition}, cluster_id order by peso_cluster desc) as rank_persona_cluster
      from scored
    )
    select *,
           case when {valid_sql('partido')}
                then 'Pessoa do cluster: ' || perfil_combinado || '. Tendencia partidaria: ' || partido || '.'
                else 'Pessoa do cluster: ' || perfil_combinado || '.'
           end as descricao
    from ranked
    where rank_persona_cluster <= 5
    """


def compat_retrato_municipal_sql(source_expr: str) -> str:
    return f"""
    select ano, uf, cd_municipio, nm_municipio,
           sum({metric_sql('eleitorado')}) as eleitorado,
           sum({metric_sql('comparecimento_estimado')}) as comparecimento_estimado,
           sum({metric_sql('abstencao_estimado')}) as abstencao_estimado,
           sum({metric_sql('votos')}) as votos,
           sum({metric_sql('brancos')}) as brancos,
           sum({metric_sql('nulos')}) as nulos,
           sum({metric_sql('validos_estimados')}) as validos_estimados,
           sum({metric_sql('qtd_secoes')}) as qtd_secoes,
           sum({metric_sql('abstencao_estimado')}) / nullif(sum({metric_sql('eleitorado')}), 0) as abstencao_media,
           sum({metric_sql('comparecimento_estimado')}) / nullif(sum({metric_sql('eleitorado')}), 0) as comparecimento_medio
    from {source_expr}
    group by ano, uf, cd_municipio, nm_municipio
    """


def compat_timeline_municipal_sql(source_expr: str) -> str:
    return f"""
    select ano, uf, cd_municipio, nm_municipio,
           sum({metric_sql('eleitorado')}) as eleitorado,
           sum({metric_sql('comparecimento_estimado')}) as comparecimento_estimado,
           sum({metric_sql('abstencao_estimado')}) as abstencao_estimado,
           'timeline_municipal' as tipo_timeline
    from {source_expr}
    group by ano, uf, cd_municipio, nm_municipio
    """


def compat_timeline_uf_sql(source_expr: str) -> str:
    return f"""
    select ano, uf,
           sum({metric_sql('eleitorado')}) as eleitorado,
           sum({metric_sql('comparecimento_estimado')}) as comparecimento_estimado,
           sum({metric_sql('abstencao_estimado')}) as abstencao_estimado,
           'timeline_uf' as tipo_timeline
    from {source_expr}
    group by ano, uf
    """


def compat_timeline_nacional_sql(source_expr: str) -> str:
    return f"""
    select ano,
           sum({metric_sql('eleitorado')}) as eleitorado,
           sum({metric_sql('comparecimento_estimado')}) as comparecimento_estimado,
           sum({metric_sql('abstencao_estimado')}) as abstencao_estimado,
           'timeline_nacional' as tipo_timeline
    from {source_expr}
    group by ano
    """


def perfil_eleitor_por_ano_from_nivelado_sql(brasil_perfil_expr: str) -> str:
    frames = []
    mapping = {
        "perfil_faixa_etaria": "faixa_etaria",
        "perfil_genero": "genero",
        "perfil_instrucao": "instrucao",
        "perfil_estado_civil": "estado_civil",
        "perfil_raca_cor": "raca_cor",
    }
    for col, dim in mapping.items():
        frames.append(f"""
        select ano,
               '{dim}' as dimensao_perfil,
               cast({col} as varchar) as valor_perfil,
               sum({metric_sql('eleitorado')}) as eleitorado
        from {brasil_perfil_expr}
        where {valid_sql(col)}
        group by ano, valor_perfil
        """)
    union_sql = "\nunion all\n".join(frames)
    return f"""
    with agg as ({union_sql}),
    ranked as (
      select *,
             eleitorado / nullif(sum(eleitorado) over(partition by ano, dimensao_perfil), 0) as share_eleitorado_ano,
             row_number() over(partition by ano, dimensao_perfil order by eleitorado desc) as rank_dimensao_ano
      from agg
      where {valid_sql('valor_perfil')} and eleitorado > 0
    )
    select *
    from ranked
    order by ano, dimensao_perfil, rank_dimensao_ano
    """


def top10_perfis_nivelados_sql(municipal_expr: str, estadual_expr: str, brasil_expr: str) -> str:
    parts = []
    for expr in [municipal_expr, estadual_expr, brasil_expr]:
        if expr:
            parts.append(f"""
            select nivel,
                   ano,
                   uf,
                   cd_municipio,
                   nm_municipio,
                   perfil_combinado,
                   eleitorado,
                   share_perfil,
                   rank_perfil_ano,
                   descricao
            from {expr}
            where rank_perfil_ano <= 10
            """)
    if not parts:
        return "select 'sem_dados' as nivel, '' as ano, '' as uf, '' as cd_municipio, '' as nm_municipio, '' as perfil_combinado, 0.0 as eleitorado, 0.0 as share_perfil, 0 as rank_perfil_ano, 'Sem dados' as descricao where false"
    return "\nunion all\n".join(parts)


def perfil_entidade_union_nivelado_sql(municipal_expr: str, estadual_expr: str, brasil_expr: str, entity_col: str) -> str:
    parts = []
    for expr in [municipal_expr, estadual_expr, brasil_expr]:
        if expr:
            parts.append(f"""
            select nivel,
                   ano,
                   uf,
                   cd_municipio,
                   nm_municipio,
                   cargo,
                   turno,
                   entidade,
                   perfil_combinado,
                   votos,
                   share_perfil_na_entidade,
                   rank_perfil_entidade_ano,
                   '{entity_col}' as tipo_entidade,
                   descricao
            from {expr}
            where rank_perfil_entidade_ano <= 10
            """)
    if not parts:
        return "select 'sem_dados' as nivel, '' as ano, '' as uf, '' as cd_municipio, '' as nm_municipio, '' as cargo, '' as turno, '' as entidade, '' as perfil_combinado, 0.0 as votos, 0.0 as share_perfil_na_entidade, 0 as rank_perfil_entidade_ano, '' as tipo_entidade, 'Sem dados' as descricao where false"
    return "\nunion all\n".join(parts)


def perfil_entidade_parts_sql(eleitorado_expr: str, resultados_expr: str, entity_col: str) -> str:
    combo = profile_combo_sql()
    keys = ", ".join(["ano", "uf", "cd_municipio", "zona", "secao", "cargo", "turno"])
    return f"""
    with r as (
      select {keys}, {entity_col} as entidade, sum({metric_sql('votos')}) as votos
      from {resultados_expr}
      where {valid_sql(entity_col)} and {metric_sql('votos')} > 0
      group by all
    ),
    shares as (
      select *, votos / nullif(sum(votos) over(partition by {keys}), 0) as share_secao
      from r
    ),
    p as (
      select {keys}, nm_municipio, {combo} as perfil_combinado,
             sum({metric_sql('eleitorado')}) as eleitorado
      from {eleitorado_expr}
      group by all
    ),
    joined as (
      select p.ano, p.uf, p.cd_municipio, p.nm_municipio,
             s.cargo, s.turno, s.entidade, p.perfil_combinado,
             p.eleitorado * s.share_secao as votos_proxy
      from p
      inner join shares s using ({keys})
      where {valid_sql('perfil_combinado')} and {valid_sql('entidade')}
    )
    select ano, uf, cd_municipio, nm_municipio, cargo, turno, entidade, perfil_combinado,
           sum(votos_proxy) as votos_proxy
    from joined
    group by all
    """


def perfil_entidade_final_sql(parts_expr: str, entity_col: str) -> str:
    return f"""
    with joined as (
      select ano, uf, cd_municipio, nm_municipio, cargo, turno, entidade, perfil_combinado,
             sum({metric_sql('votos_proxy')}) as votos_proxy
      from {parts_expr}
      where {valid_sql('perfil_combinado')} and {valid_sql('entidade')}
      group by all
    ),
    levels as (
      select 'brasil' as nivel, ano, '' as uf, '' as cd_municipio, '' as nm_municipio,
             cargo, turno, entidade, perfil_combinado, sum(votos_proxy) as votos
      from joined group by all
      union all
      select 'estado' as nivel, ano, uf, '' as cd_municipio, '' as nm_municipio,
             cargo, turno, entidade, perfil_combinado, sum(votos_proxy) as votos
      from joined group by all
      union all
      select 'municipio' as nivel, ano, uf, cd_municipio, nm_municipio,
             cargo, turno, entidade, perfil_combinado, sum(votos_proxy) as votos
      from joined group by all
    ),
    ranked as (
      select *,
             votos / nullif(sum(votos) over(partition by nivel, ano, uf, cd_municipio, cargo, turno, entidade), 0) as share_perfil_na_entidade,
             row_number() over(partition by nivel, ano, uf, cd_municipio, cargo, turno, entidade order by votos desc) as rank_perfil_entidade_ano
      from levels
    )
    select *, '{entity_col}' as tipo_entidade
    from ranked
    where rank_perfil_entidade_ano <= 10
    order by nivel, uf, nm_municipio, ano, entidade, rank_perfil_entidade_ano
    """


def perfil_candidato_sql(candidatos_expr: str) -> str:
    combo = profile_combo_sql()
    return f"""
    select ano, uf, cd_municipio, nm_municipio, cargo, partido, candidato,
           {combo} as perfil_candidato,
           count(*) as qtd_registros_candidato
    from {candidatos_expr}
    where {valid_sql('candidato')}
    group by all
    order by ano, uf, nm_municipio, partido, candidato
    """


def write_small_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df is None or df.empty:
        df = pd.DataFrame(columns=["status"])
    df.to_parquet(path, index=False, compression="snappy")


def direct_value(record: dict[str, Any], names: Iterable[str]) -> str:
    normalized_names = [str(n).upper() for n in names]
    for name in normalized_names:
        if name in record and safe_text(record[name]):
            return safe_text(record[name])
    for col, value in record.items():
        c = str(col).upper()
        if any(name in c for name in normalized_names) and safe_text(value):
            return safe_text(value)
    return ""


def candidate_name(record: dict[str, Any], gold: dict[str, Any]) -> str:
    for name in ["NM_URNA_CANDIDATO", "NM_CANDIDATO", "NM_VOTAVEL", "DS_CANDIDATO", "CANDIDATO"]:
        value = direct_value(record, [name])
        if value and not looks_numeric(value):
            return clean_value(value)
    value = clean_value(gold.get("candidato", ""))
    return "" if looks_numeric(value) else value


def age_band(value: Any) -> str:
    num = parse_number(value)
    if pd.isna(num):
        return ""
    age = int(num)
    if age < 16 or age > 120:
        return label_category_value(value, col="perfil_faixa_etaria", role="perfil_faixa_etaria")
    bands = [
        (16, 17, "16 a 17 anos"),
        (18, 20, "18 a 20 anos"),
        (21, 24, "21 a 24 anos"),
        (25, 29, "25 a 29 anos"),
        (30, 34, "30 a 34 anos"),
        (35, 39, "35 a 39 anos"),
        (40, 44, "40 a 44 anos"),
        (45, 49, "45 a 49 anos"),
        (50, 54, "50 a 54 anos"),
        (55, 59, "55 a 59 anos"),
        (60, 64, "60 a 64 anos"),
        (65, 69, "65 a 69 anos"),
        (70, 74, "70 a 74 anos"),
        (75, 79, "75 a 79 anos"),
        (80, 120, "80 anos ou mais"),
    ]
    for lo, hi, label in bands:
        if lo <= age <= hi:
            return label
    return ""


def metric_value(value: Any) -> float:
    num = parse_number(value)
    return 0.0 if pd.isna(num) else float(num)


def clean_value(value: Any) -> str:
    text = safe_text(value, "").strip()
    lower = text.lower()
    if lower in NULL_WORDS:
        return ""
    if lower.endswith("_sem_valor") or lower.endswith(" sem valor"):
        return ""
    return text


def meaningful_profile(value: Any) -> str:
    text = clean_value(value)
    if text.lower().startswith("codigo "):
        return ""
    if text.lower() in {"nao informado", "não informado"}:
        return ""
    return text


def clean_uf(value: Any) -> str:
    text = clean_value(value).upper()
    return text if len(text) == 2 and text.isalpha() else "SEM_UF"


def partition_value(value: Any) -> str:
    text = clean_uf(value)
    return safe_name(text, limit=20) or "SEM_UF"


def year_from_path(rel: str) -> str:
    years = extract_years_from_value(rel)
    return str(years[0]) if years else ""


def schema_hash(keys: Iterable[Any], domain: str) -> str:
    joined = domain + "|" + "|".join(sorted(str(k) for k in keys))
    return hashlib.sha1(joined.encode("utf-8", errors="ignore")).hexdigest()[:16]


def looks_numeric(value: Any) -> bool:
    text = clean_value(value)
    return bool(text) and pd.notna(parse_number(text))


def safe_rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except Exception:
        return path.as_posix()


def sql_lit(value: Any) -> str:
    return "'" + str(value).replace("'", "''") + "'"
