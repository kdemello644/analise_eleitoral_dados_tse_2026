from __future__ import annotations

from pathlib import Path
from typing import Any
import logging

import pandas as pd

from .aggregation import GOLD_KEYS, GOLD_METRICS
from .profiler import role_by_name
from .utils import save_csv, save_parquet


def aggregate_json_with_pyspark(
    path: Path,
    arquivo_origem: str,
    parts_dir: Path,
    cfg,
) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    """Optional Spark path for very large JSON/JSONL files.

    This is intentionally conservative: it only uses columns whose role can be
    inferred from names, writes a compact Parquet dataset, and returns a small
    preview. If Spark is not installed/configured, the caller falls back to the
    current pandas/streaming path.
    """
    try:
        from pyspark.sql import SparkSession, functions as F
    except Exception as exc:
        logging.warning("PySpark indisponivel; usando engine pandas. Detalhe: %s", exc)
        return None

    try:
        spark = (
            SparkSession.builder
            .appName("analise-eleitoral-json")
            .master(getattr(cfg, "spark_master", "local[*]") or "local[*]")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("WARN")
        raw = spark.read.option("multiLine", "false").json(str(path))
        if not raw.columns:
            raw = spark.read.option("multiLine", "true").json(str(path))
        if not raw.columns:
            return None

        role_cols: dict[str, list[str]] = {}
        for col in raw.columns:
            role = role_by_name(col)
            if role:
                role_cols.setdefault(role, []).append(col)

        def first_expr(role: str):
            cols = role_cols.get(role, [])
            if not cols:
                return F.lit("").alias(role)
            return F.coalesce(*[F.col(c).cast("string") for c in cols]).alias(role)

        selected = [first_expr(role) for role in GOLD_KEYS]
        for metric in GOLD_METRICS:
            cols = role_cols.get(metric, [])
            if cols:
                expr = sum([F.coalesce(F.col(c).cast("double"), F.lit(0.0)) for c in cols], F.lit(0.0)).alias(metric)
            else:
                expr = F.lit(0.0).alias(metric)
            selected.append(expr)

        gold = raw.select(*selected)
        grouped = gold.groupBy(*GOLD_KEYS).agg(
            *[F.sum(metric).alias(metric) for metric in GOLD_METRICS],
            F.count(F.lit(1)).alias("linhas_origem"),
        )
        grouped = grouped.withColumn("arquivo_origem", F.lit(arquivo_origem))
        grouped = grouped.withColumn("aggregation_mode", F.lit("pyspark_json_parquet"))
        grouped = grouped.withColumn("validos_estimados", F.when(F.col("validos") > 0, F.col("validos")).otherwise(F.col("votos")))
        grouped = grouped.withColumn(
            "comparecimento_estimado",
            F.when(F.col("comparecimento") > 0, F.col("comparecimento")).otherwise(F.col("validos_estimados") + F.col("brancos") + F.col("nulos")),
        )
        grouped = grouped.withColumn(
            "abstencao_estimado",
            F.when(F.col("abstencao") > 0, F.col("abstencao")).otherwise(F.greatest(F.col("eleitorado") - F.col("comparecimento_estimado"), F.lit(0.0))),
        )
        grouped = grouped.withColumn("pct_comparecimento", F.when(F.col("eleitorado") > 0, F.col("comparecimento_estimado") / F.col("eleitorado")))
        grouped = grouped.withColumn("pct_abstencao", F.when(F.col("eleitorado") > 0, F.col("abstencao_estimado") / F.col("eleitorado")))

        out_dir = parts_dir / "spark_gold_dataset"
        out_dir.mkdir(parents=True, exist_ok=True)
        grouped.write.mode("overwrite").partitionBy("uf").parquet(str(out_dir))

        preview_rows = max(0, int(getattr(cfg, "analysis_max_rows", 200000) or 0))
        preview = grouped.limit(preview_rows).toPandas() if preview_rows else pd.DataFrame()
        manifest = pd.DataFrame([{
            "parte": 1,
            "linhas": int(grouped.count()),
            "parquet": str(out_dir),
            "csv": "",
            "engine": "pyspark",
        }])
        save_csv(manifest, parts_dir / "manifesto_partes_gold.csv")
        save_parquet(manifest, parts_dir / "manifesto_partes_gold.parquet")
        info = {
            "modo_gold": "pyspark_parquet_dataset",
            "linhas_json_lidas": int(manifest["linhas"].iloc[0]),
            "partes_gold": 1,
            "gold_parts_dir": str(out_dir),
            "gold_parts_manifest": str(parts_dir / "manifesto_partes_gold.csv"),
            "gold_parts_manifest_parquet": str(parts_dir / "manifesto_partes_gold.parquet"),
            "linhas_preview_gold": int(len(preview)),
            "engine": "pyspark",
        }
        return preview, info
    except Exception as exc:
        logging.warning("Falha no engine PySpark para %s; usando engine pandas. Detalhe: %s", path, exc)
        return None
