from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
from collections import defaultdict
import hashlib
import json
import logging
import shutil
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
YEAR_UF_PARTITION_COLS = ["ano", "uf"]
MUNICIPIO_PARTITION_COLS = ["ano", "uf", "cd_municipio"]
ZONE_PARTITION_COLS = ["ano", "uf", "cd_municipio", "zona"]
SECTION_PARTITION_COLS = ["ano", "uf", "cd_municipio", "zona", "secao"]
ENTITY_PARTITION_COLS = ["nivel", "ano", "uf", "cd_municipio"]


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
    delete_source_after_success: bool = False
    ouro_parallel_aggressive: bool = False
    auto_tune_info: dict[str, Any] | None = None
    log_level: str = "INFO"


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
        df.to_parquet(out_dir / f"part-{part_id:06d}.parquet", index=False, compression="snappy")
        self.rows_written += len(df)


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
    last_exc: Exception | None = None
    for workers in worker_retry_ladder(requested_workers):
        try:
            logging.info("Tentativa de processamento com %s worker(s): %s", workers, item.get("relativo", ""))
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
        for idx, item in enumerate(items, start=1):
            logging.info(
                "Iniciando documento %s/%s [%s]: %s",
                idx,
                len(items),
                label,
                describe_work_item(item),
            )
            yield process_json_file_item(item, cfg)
        return

    logging.info("Processando %s arquivos %s em paralelo com %s workers.", len(items), label, workers)
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
                yield result
            except Exception as exc:
                logging.exception("Erro processando banco %s: %s", item.get("relativo"), exc)
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
            read_rows += int(manifest.get("linhas_lidas", 0) or 0)
            written_rows += int(manifest.get("linhas_gravadas", 0) or 0)
            for schema_id, row in (result.get("schemas") or {}).items():
                schemas.setdefault(schema_id, row)
            for table, count in (result.get("linhas_por_tabela") or {}).items():
                rows_written[table] += int(count or 0)
            bronze_rows_written += int(result.get("linhas_bronze", 0) or 0)

    if errors:
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

    def drain_completed(done_futures: set[Any]) -> None:
        for future in done_futures:
            chunk = future_map.pop(future, {})
            try:
                merge_batch_result(future.result(), chunk)
            except Exception as exc:
                logging.exception("Erro processando lote %s de %s: %s", chunk.get("batch_index"), rel, exc)
                errors.append(f"lote {chunk.get('batch_index')}: {exc}")

    with ProcessPoolExecutor(max_workers=workers) as pool:
        batch: list[dict[str, Any]] = []
        for rec in iter_json_records(path):
            batch.append(rec)
            if len(batch) < batch_rows:
                continue

            chunk = build_batch_item(item, batch, base_shard, batch_index)
            logging.info("Enfileirando lote %s de %s | linhas=%s", batch_index + 1, rel, len(batch))
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
            future = pool.submit(process_record_batch_item, chunk, cfg)
            pending.add(future)
            future_map[future] = chunk
            submitted_batches += 1

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            drain_completed(done)

    if errors:
        raise RuntimeError(f"Falha em lotes de {rel}: {' | '.join(errors[:5])}")

    size_mb = float(item.get("tamanho_gb", 0) or 0) * 1024
    duration = time.perf_counter() - started
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
    try:
        import duckdb
    except Exception as exc:
        logging.warning("DuckDB indisponivel; analises derivadas nao foram geradas: %s", exc)
        return {"status": "duckdb_indisponivel", "erro": str(exc)}

    analyses = cfg.out / "ouro"
    analyses.mkdir(parents=True, exist_ok=True)

    prata = cfg.out / "prata"
    e = dataset_expr(prata / "eleitorado")
    r = dataset_expr(prata / "resultados_votos")
    c = dataset_expr(prata / "candidatos")
    outputs: dict[str, str] = {}

    if parquet_dataset_exists(prata / "eleitorado"):
        outputs.update(run_eleitorado_ouro_partitioned(cfg, prata, analyses))

    if parquet_dataset_exists(prata / "resultados_votos"):
        outputs.update(run_resultados_ouro_partitioned(cfg, prata, analyses))

    if parquet_dataset_exists(prata / "eleitorado") and parquet_dataset_exists(prata / "resultados_votos"):
        outputs.update(run_correlacoes_ouro_partitioned(cfg, prata, analyses))

    if parquet_dataset_exists(prata / "candidatos"):
        outputs.update(run_candidatos_ouro_partitioned(cfg, prata, analyses))

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
        years = list_years_for_dataset(e_path, cfg)
        logging.info("Ouro eleitorado UF %s/%s: %s | anos=%s", index, len(uf_parts), uf, ", ".join(years) or "sem_ano")
        for year_index, year in enumerate(years, start=1):
            slice_key = slice_name(uf, year)
            logging.info("Ouro eleitorado fatia %s/%s UF %s ano %s", year_index, len(years), uf, year)
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
            for task in tasks:
                outputs[task["name"]] = execute_ouro_task(task, cfg, label, progress_dir)
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
    for task in final_tasks:
        outputs[task["name"]] = execute_ouro_task(task, cfg, label, progress_dir)
        clean_memory()

    save_json({"label": label, "status": "ok", "outputs": outputs}, progress_dir / f"{label}_progresso.json")
    return outputs


def run_resultados_ouro_partitioned(cfg: CleanDatabaseConfig, prata: Path, analyses: Path) -> dict[str, str]:
    label = "ouro_resultados"
    progress_dir = prepare_ouro_progress(cfg, label)
    reset_ouro_targets([analyses / "resultados_vencedores_secao"], resume=cfg.resume)
    outputs: dict[str, str] = {}
    uf_parts = list_uf_partition_dirs(prata / "resultados_votos")
    logging.info("Gerando %s por UF: %s particoes de resultados.", label, len(uf_parts))
    for index, (uf, r_path) in enumerate(uf_parts, start=1):
        years = list_years_for_dataset(r_path, cfg)
        logging.info("Ouro resultados UF %s/%s: %s | anos=%s", index, len(uf_parts), uf, ", ".join(years) or "sem_ano")
        for year in years:
            slice_key = slice_name(uf, year)
            task = copy_task(
                f"resultados_vencedores_secao_{slice_key}",
                vencedores_secao_sql(filtered_year_expr(dataset_expr(r_path), year)),
                chunk_output(analyses / "resultados_vencedores_secao", slice_key),
                partition_by=SECTION_PARTITION_COLS,
            )
            outputs[task["name"]] = execute_ouro_task(task, cfg, label, progress_dir)
            clean_memory()
    save_json({"label": label, "status": "ok", "outputs": outputs}, progress_dir / f"{label}_progresso.json")
    return outputs


def run_correlacoes_ouro_partitioned(cfg: CleanDatabaseConfig, prata: Path, analyses: Path) -> dict[str, str]:
    label = "ouro_correlacoes"
    progress_dir = prepare_ouro_progress(cfg, label)
    targets = [
        analyses / "base_gold_global",
        analyses / "resultado_eleitorado_por_secao",
        analyses / "perfil_eleitor_por_partido",
        analyses / "comparativo_anual_perfil_partido",
        analyses / "perfil_eleitor_por_candidato",
        analyses / "comparativo_anual_perfil_candidato",
        analyses / "_work" / "perfil_eleitor_por_partido_parts",
        analyses / "_work" / "perfil_eleitor_por_candidato_parts",
    ]
    reset_ouro_targets(targets, resume=cfg.resume)

    outputs: dict[str, str] = {}
    uf_parts = [(uf, e_path) for uf, e_path in list_uf_partition_dirs(prata / "eleitorado") if parquet_dataset_exists(prata / "resultados_votos" / f"uf={uf}")]
    logging.info("Gerando %s por UF: %s particoes com eleitorado+resultados.", label, len(uf_parts))

    for index, (uf, e_path) in enumerate(uf_parts, start=1):
        r_path = prata / "resultados_votos" / f"uf={uf}"
        years = sorted(set(list_years_for_dataset(e_path, cfg)) & set(list_years_for_dataset(r_path, cfg)))
        logging.info("Ouro correlacoes UF %s/%s: %s | anos=%s", index, len(uf_parts), uf, ", ".join(years) or "sem_ano")
        for year_index, year in enumerate(years, start=1):
            slice_key = slice_name(uf, year)
            logging.info("Ouro correlacoes fatia %s/%s UF %s ano %s", year_index, len(years), uf, year)
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
            winners_path = chunk_output(analyses / "resultados_vencedores_secao", slice_key)
            e_slice = filtered_year_expr(dataset_expr(e_path), year)
            r_slice = filtered_year_expr(dataset_expr(r_path), year)
            tasks = [
                copy_task(f"base_gold_global_{slice_key}", base_gold_global_sql(e_slice, r_slice), chunk_output(analyses / "base_gold_global", slice_key), partition_by=ZONE_PARTITION_COLS),
                copy_task(f"perfil_eleitor_por_partido_parts_{slice_key}", perfil_entidade_parts_sql(e_slice, r_slice, "partido"), chunk_output(analyses / "_work" / "perfil_eleitor_por_partido_parts", slice_key), partition_by=MUNICIPIO_PARTITION_COLS),
            ]
            if parquet_dataset_exists(winners_path):
                tasks.insert(1, copy_task(
                    f"resultado_eleitorado_por_secao_{slice_key}",
                    resultado_eleitorado_secao_sql(e_slice, winners_path),
                    chunk_output(analyses / "resultado_eleitorado_por_secao", slice_key),
                    partition_by=SECTION_PARTITION_COLS,
                ))
            else:
                logging.warning("Pulando resultado_eleitorado_por_secao %s: vencedores da fatia ainda nao existem.", slice_key)
            if not cfg.skip_heavy_analyses:
                tasks.append(copy_task(
                    f"perfil_eleitor_por_candidato_parts_{slice_key}",
                    perfil_entidade_parts_sql(e_slice, r_slice, "candidato"),
                    chunk_output(analyses / "_work" / "perfil_eleitor_por_candidato_parts", slice_key),
                    partition_by=MUNICIPIO_PARTITION_COLS,
                ))
            for task in tasks:
                outputs[task["name"]] = execute_ouro_task(task, cfg, label, progress_dir)
                clean_memory()

    final_tasks = []
    if parquet_dataset_exists(analyses / "_work" / "perfil_eleitor_por_partido_parts"):
        partido_parts_expr = dataset_expr(analyses / "_work" / "perfil_eleitor_por_partido_parts")
        final_tasks.extend([
            copy_task("perfil_eleitor_por_partido", perfil_entidade_final_sql(partido_parts_expr, "partido"), analyses / "perfil_eleitor_por_partido", partition_by=ENTITY_PARTITION_COLS),
            copy_task("comparativo_anual_perfil_partido", perfil_entidade_final_sql(partido_parts_expr, "partido"), analyses / "comparativo_anual_perfil_partido", partition_by=ENTITY_PARTITION_COLS),
        ])
    if not cfg.skip_heavy_analyses and parquet_dataset_exists(analyses / "_work" / "perfil_eleitor_por_candidato_parts"):
        candidato_parts_expr = dataset_expr(analyses / "_work" / "perfil_eleitor_por_candidato_parts")
        final_tasks.extend([
            copy_task("perfil_eleitor_por_candidato", perfil_entidade_final_sql(candidato_parts_expr, "candidato"), analyses / "perfil_eleitor_por_candidato", partition_by=ENTITY_PARTITION_COLS),
            copy_task("comparativo_anual_perfil_candidato", perfil_entidade_final_sql(candidato_parts_expr, "candidato"), analyses / "comparativo_anual_perfil_candidato", partition_by=ENTITY_PARTITION_COLS),
        ])
    for task in final_tasks:
        outputs[task["name"]] = execute_ouro_task(task, cfg, label, progress_dir)
        clean_memory()

    save_json({"label": label, "status": "ok", "outputs": outputs}, progress_dir / f"{label}_progresso.json")
    return outputs


def run_candidatos_ouro_partitioned(cfg: CleanDatabaseConfig, prata: Path, analyses: Path) -> dict[str, str]:
    label = "ouro_candidatos"
    progress_dir = prepare_ouro_progress(cfg, label)
    reset_ouro_targets([analyses / "perfil_candidatos"], resume=cfg.resume)
    outputs: dict[str, str] = {}
    uf_parts = list_uf_partition_dirs(prata / "candidatos")
    logging.info("Gerando %s por UF: %s particoes de candidatos.", label, len(uf_parts))
    for index, (uf, c_path) in enumerate(uf_parts, start=1):
        years = list_years_for_dataset(c_path, cfg)
        logging.info("Ouro candidatos UF %s/%s: %s | anos=%s", index, len(uf_parts), uf, ", ".join(years) or "sem_ano")
        for year in years:
            slice_key = slice_name(uf, year)
            task = copy_task(
                f"perfil_candidatos_{slice_key}",
                perfil_candidato_sql(filtered_year_expr(dataset_expr(c_path), year)),
                chunk_output(analyses / "perfil_candidatos", slice_key),
                partition_by=MUNICIPIO_PARTITION_COLS,
            )
            outputs[task["name"]] = execute_ouro_task(task, cfg, label, progress_dir)
            clean_memory()
    save_json({"label": label, "status": "ok", "outputs": outputs}, progress_dir / f"{label}_progresso.json")
    return outputs


def prepare_ouro_progress(cfg: CleanDatabaseConfig, label: str) -> Path:
    progress_dir = cfg.out / "logs" / "ouro"
    progress_dir.mkdir(parents=True, exist_ok=True)
    save_json({"label": label, "status": "iniciando"}, progress_dir / f"{label}_progresso.json")
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
    for path in sorted(dataset_root.iterdir(), key=lambda p: p.name):
        if not path.is_dir() or not path.name.startswith("uf="):
            continue
        uf = path.name.split("=", 1)[1] or "SEM_UF"
        if parquet_dataset_exists(path):
            out.append((uf, path))
    if out:
        return out
    return [("SEM_UF", dataset_root)] if parquet_dataset_exists(dataset_root) else []


def chunk_output(root: Path, uf: str) -> Path:
    return root / f"chunk={safe_name(uf, 20) or 'SEM_UF'}"


def slice_name(uf: str, year: str) -> str:
    year_text = safe_name(year, 20) if year else "SEM_ANO"
    return f"{safe_name(uf, 20) or 'SEM_UF'}_{year_text}"


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
    return years or [""]


def read_years_with_duckdb(path: Path, cfg: CleanDatabaseConfig) -> pd.DataFrame:
    import duckdb

    con = duckdb.connect(database=":memory:")
    con.execute(f"PRAGMA threads={max(1, int(cfg.duckdb_threads or 1))}")
    con.execute("PRAGMA preserve_insertion_order=false")
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
    marker = progress_dir / f"{safe_name(task.get('name', 'tarefa'), 80)}.done.json"
    if cfg.resume and marker.exists() and output_has_data(out):
        logging.info("Pulando tarefa ouro ja concluida [%s]: %s", label, task.get("name"))
        return str(out)
    try:
        return execute_copy_task(task, max(1, int(cfg.duckdb_threads or 1)), progress_dir=progress_dir, label=label)
    except Exception as exc:
        task_name = safe_name(task.get("name", "tarefa"), 80)
        logging.exception("Erro na tarefa ouro [%s] %s: %s", label, task.get("name"), exc)
        save_json(
            {
                "label": label,
                "tarefa": str(task.get("name", "")),
                "status": "erro",
                "saida": str(task.get("out", "")),
                "erro": str(exc),
            },
            progress_dir / f"{task_name}.error.json",
        )
        return f"ERRO: {exc}"


def output_has_data(path: Path) -> bool:
    if path.is_dir():
        return parquet_dataset_exists(path)
    return path.exists() and path.stat().st_size > 0


def copy_task(name: str, sql: str, out: Path, partition_by: list[str] | None = None) -> dict[str, Any]:
    return {"name": name, "sql": sql, "out": out, "partition_by": partition_by or []}


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
    requested_workers = min(max(1, int(cfg.ouro_workers or 1)), len(tasks))
    if label in HEAVY_OURO_LABELS and not cfg.ouro_parallel_aggressive:
        workers = 1
        logging.info(
            "Modo seguro ouro: %s roda uma tarefa por vez para evitar OOM. Workers solicitados=%s, efetivos=1.",
            label,
            requested_workers,
        )
    else:
        workers = requested_workers
    threads_per_task = max(1, int(cfg.duckdb_threads or 1) // workers)
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
            try:
                outputs[task["name"]] = execute_copy_task(task, threads_per_task, progress_dir=progress_dir, label=label)
            except Exception as exc:
                logging.exception("Erro gerando ouro %s: %s", task.get("name"), exc)
                outputs[f"{task['name']}_erro"] = str(exc)
        save_json({"label": label, "status": "ok", "outputs": outputs}, progress_dir / f"{safe_name(label, 80)}_progresso.json")
        return outputs
    outputs: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {}
        for index, task in enumerate(tasks, start=1):
            logging.info("Enfileirando tarefa ouro %s/%s [%s]: %s", index, len(tasks), label, task.get("name"))
            future_map[pool.submit(execute_copy_task, task, threads_per_task, progress_dir, label)] = task
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
    save_json({"label": label, "status": "ok", "outputs": outputs}, progress_dir / f"{safe_name(label, 80)}_progresso.json")
    return outputs


def execute_copy_task(task: dict[str, Any], duckdb_threads: int, progress_dir: Path | None = None, label: str = "") -> str:
    import duckdb

    started = time.perf_counter()
    task_name = str(task.get("name", "tarefa"))
    if progress_dir is not None:
        save_json(
            {
                "label": label,
                "tarefa": task_name,
                "status": "em_execucao",
                "saida": str(task.get("out", "")),
                "duckdb_threads": int(duckdb_threads or 1),
            },
            progress_dir / f"{safe_name(task_name, 80)}.started.json",
        )
    logging.info("DuckDB COPY iniciado: %s -> %s", task_name, task.get("out", ""))
    con = duckdb.connect(database=":memory:")
    con.execute(f"PRAGMA threads={max(1, int(duckdb_threads or 1))}")
    con.execute("PRAGMA preserve_insertion_order=false")
    out = Path(task["out"])
    try:
        copy_query(con, str(task["sql"]), out, partition_by=list(task.get("partition_by") or []))
    finally:
        con.close()
    duration = time.perf_counter() - started
    logging.info("DuckDB COPY finalizado: %s em %.1fs -> %s", task_name, duration, out)
    if progress_dir is not None:
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


def parquet_dataset_exists(path: Path) -> bool:
    return path.exists() and any(path.rglob("*.parquet"))


def metric_sql(col: str) -> str:
    return f"coalesce(try_cast(replace(cast({col} as varchar), ',', '.') as double), 0)"


def valid_sql(col: str) -> str:
    nulls = ", ".join(sql_lit(x) for x in sorted(NULL_WORDS | {"sem valor"}))
    expr = f"lower(trim(cast({col} as varchar)))"
    return f"{expr} not in ({nulls}) and {expr} not like '%sem valor%'"


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
    keys = ", ".join(["ano", "uf", "cd_municipio", "nm_municipio", "zona", "secao", "cargo", "turno"])
    return f"""
    with votos as (
      select {keys},
             partido,
             candidato,
             nr_votavel,
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
