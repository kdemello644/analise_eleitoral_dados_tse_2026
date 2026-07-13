from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any
import logging

from .individual import process_file
from .utils import clean_memory, save_json


def _file_size_gb(path: Path) -> float:
    try:
        return path.stat().st_size / (1024 ** 3)
    except Exception:
        return 0.0


def _process_file_worker(file: Path, dados: Path, cfg) -> dict[str, Any]:
    try:
        return process_file(file, dados, cfg)
    finally:
        clean_memory()


def run_individual_stage(json_files: list[Path], cfg) -> list[dict[str, Any]]:
    """Run the individual-analysis block.

    Contract:
    - reads only source JSON/JSONL files;
    - writes one folder per original file, grouped by detected year;
    - writes per-year gold/electoral outputs inside each individual folder;
    - returns a manifest consumed by the global block.
    """
    results: list[dict[str, Any]] = []
    threshold = float(getattr(cfg, "large_file_threshold_gb", 10.0) or 10.0)
    small_files = [p for p in json_files if _file_size_gb(p) < threshold]
    large_files = [p for p in json_files if _file_size_gb(p) >= threshold]

    def _append_result(result: dict[str, Any], file: Path, done: int) -> None:
        results.append(result)
        save_json(results, cfg.out / "logs" / "resultados_individuais_parciais.json")
        clean_memory()
        logging.info("Memoria limpa apos arquivo concluido %s/%s: %s", done, len(json_files), file)

    done = 0
    if small_files and int(getattr(cfg, "workers_individual", 1) or 1) > 1:
        workers = min(int(getattr(cfg, "workers_individual", 1) or 1), len(small_files))
        logging.info("Processando %s arquivos pequenos/medios em paralelo com %s workers.", len(small_files), workers)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_map = {pool.submit(_process_file_worker, file, cfg.dados, cfg): file for file in small_files}
            for future in as_completed(future_map):
                file = future_map[future]
                done += 1
                logging.info("=" * 100)
                logging.info("(%s/%s) Bloco individual concluido: %s", done, len(json_files), file)
                try:
                    _append_result(future.result(), file, done)
                except Exception as exc:
                    logging.exception("Erro em worker individual %s: %s", file, exc)
                    _append_result({"status": "erro", "arquivo": str(file), "relativo": str(file), "erro": str(exc), "html": ""}, file, done)
    else:
        for file in small_files:
            done += 1
            logging.info("=" * 100)
            logging.info("(%s/%s) Bloco individual: %s", done, len(json_files), file)
            try:
                result = process_file(file, cfg.dados, cfg)
                _append_result(result, file, done)
            finally:
                clean_memory()

    if large_files:
        workers_large = min(int(getattr(cfg, "workers_large_files", 1) or 1), len(large_files))
        if workers_large > 1:
            logging.info("Processando %s arquivos grandes em paralelo com %s workers.", len(large_files), workers_large)
            with ProcessPoolExecutor(max_workers=workers_large) as pool:
                future_map = {pool.submit(_process_file_worker, file, cfg.dados, cfg): file for file in large_files}
                for future in as_completed(future_map):
                    file = future_map[future]
                    done += 1
                    logging.info("=" * 100)
                    logging.info("(%s/%s) Bloco individual grande concluido: %s", done, len(json_files), file)
                    try:
                        _append_result(future.result(), file, done)
                    except Exception as exc:
                        logging.exception("Erro em worker individual grande %s: %s", file, exc)
                        _append_result({"status": "erro", "arquivo": str(file), "relativo": str(file), "erro": str(exc), "html": ""}, file, done)
        else:
            for file in large_files:
                done += 1
                logging.info("=" * 100)
                logging.info("(%s/%s) Bloco individual grande isolado: %s", done, len(json_files), file)
                try:
                    result = process_file(file, cfg.dados, cfg)
                    _append_result(result, file, done)
                finally:
                    clean_memory()

    save_json(results, cfg.out / "logs" / "resultados_individuais.json")
    return results
