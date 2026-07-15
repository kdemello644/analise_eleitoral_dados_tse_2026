from __future__ import annotations

from typing import Any
import logging

from .clean_database import write_pipeline_event
from .global_analysis import build_global
from .utils import clean_memory, save_json


def run_global_stage(results: list[dict[str, Any]], cfg) -> dict[str, Any]:
    """Run the global-analysis block.

    Contract:
    - consumes only the individual-stage manifest and files pointed by it;
    - materializes consolidated CSV/Parquet tables;
    - builds correlations, global electoral analysis, clusters and graphs;
    - returns a global manifest consumed by the simulation block.
    """
    logging.info("=" * 100)
    logging.info("Bloco global: consolidando saidas individuais.")
    write_pipeline_event(cfg.out, "global", "inicio", resultados=len(results))
    try:
        global_info = build_global(results, cfg)
        save_json(global_info, cfg.out / "logs" / "global_info.json")
        write_pipeline_event(cfg.out, "global", "fim", outputs=global_info)
        return global_info
    except Exception as exc:
        write_pipeline_event(cfg.out, "global", "erro", erro=str(exc))
        raise
    finally:
        clean_memory()
        logging.info("Memoria limpa apos bloco global.")
