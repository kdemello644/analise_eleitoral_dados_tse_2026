from __future__ import annotations

from typing import Any
import logging

from .clean_database import write_pipeline_event
from .simulation import run_prediction
from .utils import clean_memory, save_json


def run_simulation_stage(global_info: dict[str, Any], cfg) -> dict[str, Any]:
    """Run the simulation block.

    Contract:
    - consumes the global-stage manifest;
    - loads the consolidated global gold table;
    - simulates vote share by section/municipality/entity for 2026;
    - writes scenario, Monte Carlo, decisive-section and national summary tables.
    """
    logging.info("=" * 100)
    logging.info("Bloco simulacao: usando base global consolidada.")
    write_pipeline_event(cfg.out, "simulacao", "inicio", global_info=global_info)
    try:
        pred_info = run_prediction(global_info, cfg)
        save_json(pred_info, cfg.out / "logs" / "pred_info.json")
        write_pipeline_event(cfg.out, "simulacao", "fim", outputs=pred_info)
        return pred_info
    except Exception as exc:
        write_pipeline_event(cfg.out, "simulacao", "erro", erro=str(exc))
        raise
    finally:
        clean_memory()
        logging.info("Memoria limpa apos bloco de simulacao.")
