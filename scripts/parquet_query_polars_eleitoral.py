from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import polars as pl
except ModuleNotFoundError as exc:  # pragma: no cover - depende do ambiente do usuario
    pl = None  # type: ignore[assignment]
    POLARS_IMPORT_ERROR = exc
else:
    POLARS_IMPORT_ERROR = None


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
    "municipal": ["ouro/municipal/resumo", "ouro/retrato_municipal", "ouro/retrato_municipal.parquet", "global/parquet/retrato_municipal_global.parquet"],
    "timeline_nacional": ["ouro/brasil/resumo", "ouro/timeline_nacional.parquet", "global/parquet/timeline_nacional.parquet"],
    "timeline_uf": ["ouro/estadual/resumo", "ouro/timeline_uf", "ouro/timeline_uf.parquet", "global/parquet/timeline_uf.parquet"],
    "timeline_municipal": ["ouro/municipal/resumo", "ouro/timeline_municipal", "ouro/timeline_municipal.parquet", "global/parquet/timeline_municipal.parquet"],
    "perfil_eleitor_brasil": ["ouro/brasil/perfil_eleitor"],
    "perfil_eleitor_estado": ["ouro/estadual/perfil_eleitor"],
    "perfil_eleitor_municipio": ["ouro/municipal/perfil_eleitor"],
    "contagem_colunas_perfil_eleitor_brasil": ["ouro/brasil/contagem_colunas_perfil_eleitor"],
    "contagem_colunas_perfil_eleitor_estado": ["ouro/estadual/contagem_colunas_perfil_eleitor"],
    "contagem_colunas_perfil_eleitor_municipio": ["ouro/municipal/contagem_colunas_perfil_eleitor"],
    "perfil_ano": ["ouro/perfil_eleitor_por_ano", "ouro/brasil/perfil_eleitor", "ouro/estadual/perfil_eleitor", "ouro/municipal/perfil_eleitor", "ouro/perfil_eleitor_por_ano.parquet"],
    "perfil_partido_brasil": ["ouro/brasil/perfil_partido"],
    "perfil_partido_estado": ["ouro/estadual/perfil_partido"],
    "perfil_partido_municipio": ["ouro/municipal/perfil_partido"],
    "contagem_colunas_perfil_partido_brasil": ["ouro/brasil/contagem_colunas_perfil_partido"],
    "contagem_colunas_perfil_partido_estado": ["ouro/estadual/contagem_colunas_perfil_partido"],
    "contagem_colunas_perfil_partido_municipio": ["ouro/municipal/contagem_colunas_perfil_partido"],
    "perfil_partido": ["ouro/brasil/perfil_partido", "ouro/estadual/perfil_partido", "ouro/municipal/perfil_partido", "ouro/perfil_eleitor_por_partido", "ouro/perfil_eleitor_por_partido.parquet"],
    "perfil_candidato_brasil": ["ouro/brasil/perfil_candidato"],
    "perfil_candidato_estado": ["ouro/estadual/perfil_candidato"],
    "perfil_candidato_municipio": ["ouro/municipal/perfil_candidato"],
    "contagem_colunas_perfil_candidato_brasil": ["ouro/brasil/contagem_colunas_perfil_candidato"],
    "contagem_colunas_perfil_candidato_estado": ["ouro/estadual/contagem_colunas_perfil_candidato"],
    "contagem_colunas_perfil_candidato_municipio": ["ouro/municipal/contagem_colunas_perfil_candidato"],
    "perfil_candidato": ["ouro/brasil/perfil_candidato", "ouro/estadual/perfil_candidato", "ouro/municipal/perfil_candidato", "ouro/perfil_eleitor_por_candidato", "ouro/perfil_eleitor_por_candidato.parquet"],
    "resultado_partido_brasil": ["ouro/brasil/resultado_partido"],
    "resultado_partido_estado": ["ouro/estadual/resultado_partido"],
    "resultado_partido_municipio": ["ouro/municipal/resultado_partido"],
    "resultado_partido": ["ouro/brasil/resultado_partido", "ouro/estadual/resultado_partido", "ouro/municipal/resultado_partido"],
    "resultado_candidato_brasil": ["ouro/brasil/resultado_candidato"],
    "resultado_candidato_estado": ["ouro/estadual/resultado_candidato"],
    "resultado_candidato_municipio": ["ouro/municipal/resultado_candidato"],
    "resultado_candidato": ["ouro/brasil/resultado_candidato", "ouro/estadual/resultado_candidato", "ouro/municipal/resultado_candidato"],
    "top10_perfis": ["ouro/top10_perfis_federacao_estado_municipio", "ouro/brasil/perfil_eleitor", "ouro/estadual/perfil_eleitor", "ouro/municipal/perfil_eleitor", "ouro/top10_perfis_federacao_estado_municipio.parquet"],
    "vencedor_secao": ["ouro/resultados_vencedores_secao", "ouro/resultados_vencedores_secao.parquet"],
    "resultado_eleitorado": ["ouro/resultado_eleitorado_por_secao", "ouro/resultado_eleitorado_por_secao.parquet"],
    "perfil_candidatos": ["ouro/perfil_candidatos", "ouro/perfil_candidatos.parquet"],
    "cluster_voter_brasil": ["ouro/brasil/clusters_eleitores"],
    "cluster_voter_estado": ["ouro/estadual/clusters_eleitores"],
    "cluster_voter_municipio": ["ouro/municipal/clusters_eleitores"],
    "contagem_colunas_clusters_eleitores_brasil": ["ouro/brasil/contagem_colunas_clusters_eleitores"],
    "contagem_colunas_clusters_eleitores_estado": ["ouro/estadual/contagem_colunas_clusters_eleitores"],
    "contagem_colunas_clusters_eleitores_municipio": ["ouro/municipal/contagem_colunas_clusters_eleitores"],
    "cluster_voter_personas": ["ouro/brasil/clusters_eleitores", "ouro/estadual/clusters_eleitores", "ouro/municipal/clusters_eleitores"],
    "cluster_result_brasil": ["ouro/brasil/clusters_eleitores_resultado"],
    "cluster_result_estado": ["ouro/estadual/clusters_eleitores_resultado"],
    "cluster_result_municipio": ["ouro/municipal/clusters_eleitores_resultado"],
    "contagem_colunas_clusters_eleitores_resultado_brasil": ["ouro/brasil/contagem_colunas_clusters_eleitores_resultado"],
    "contagem_colunas_clusters_eleitores_resultado_estado": ["ouro/estadual/contagem_colunas_clusters_eleitores_resultado"],
    "contagem_colunas_clusters_eleitores_resultado_municipio": ["ouro/municipal/contagem_colunas_clusters_eleitores_resultado"],
    "cluster_result_personas": ["ouro/brasil/clusters_eleitores_resultado", "ouro/estadual/clusters_eleitores_resultado", "ouro/municipal/clusters_eleitores_resultado"],
    "banco_prata_eleitorado": ["prata/eleitorado"],
    "banco_prata_candidatos": ["prata/candidatos"],
    "banco_prata_resultados": ["prata/resultados_votos"],
    "sim_partidos_brasil": ["preditivo_2026/parquet/partidos_2026_brasil.parquet", "preditivo_2026/tabelas/partidos_2026_brasil.csv"],
    "sim_partidos_estados": ["preditivo_2026/parquet/partidos_2026_estados.parquet", "preditivo_2026/tabelas/partidos_2026_estados.csv"],
    "sim_partidos_municipios": ["preditivo_2026/parquet/partidos_2026_municipios.parquet", "preditivo_2026/tabelas/partidos_2026_municipios.csv"],
    "sim_partidos_correlacao": ["preditivo_2026/parquet/partidos_2026_correlacao_historica.parquet", "preditivo_2026/tabelas/partidos_2026_correlacao_historica.csv"],
}

TABLE_LABELS: dict[str, str] = {
    "catalogo": "Metadados - catalogo dos arquivos",
    "municipal": "Ouro - retrato municipal",
    "timeline_nacional": "Ouro - timeline nacional",
    "timeline_uf": "Ouro - timeline por UF",
    "timeline_municipal": "Ouro - timeline municipal",
    "perfil_eleitor_brasil": "Ouro - perfil do eleitor Brasil",
    "perfil_eleitor_estado": "Ouro - perfil do eleitor por estado",
    "perfil_eleitor_municipio": "Ouro - perfil do eleitor por municipio",
    "contagem_colunas_perfil_eleitor_brasil": "Ouro - histogramas do eleitor Brasil",
    "contagem_colunas_perfil_eleitor_estado": "Ouro - histogramas do eleitor por estado",
    "contagem_colunas_perfil_eleitor_municipio": "Ouro - histogramas do eleitor por municipio",
    "perfil_ano": "Ouro - perfil do eleitor por ano",
    "perfil_partido_brasil": "Ouro - perfil por partido Brasil",
    "perfil_partido_estado": "Ouro - perfil por partido estado",
    "perfil_partido_municipio": "Ouro - perfil por partido municipio",
    "contagem_colunas_perfil_partido_brasil": "Ouro - histogramas do eleitor por partido Brasil",
    "contagem_colunas_perfil_partido_estado": "Ouro - histogramas do eleitor por partido estado",
    "contagem_colunas_perfil_partido_municipio": "Ouro - histogramas do eleitor por partido municipio",
    "perfil_partido": "Ouro - perfil do eleitor por partido",
    "perfil_candidato_brasil": "Ouro - perfil por candidato Brasil",
    "perfil_candidato_estado": "Ouro - perfil por candidato estado",
    "perfil_candidato_municipio": "Ouro - perfil por candidato municipio",
    "contagem_colunas_perfil_candidato_brasil": "Ouro - histogramas do eleitor por candidato Brasil",
    "contagem_colunas_perfil_candidato_estado": "Ouro - histogramas do eleitor por candidato estado",
    "contagem_colunas_perfil_candidato_municipio": "Ouro - histogramas do eleitor por candidato municipio",
    "perfil_candidato": "Ouro - perfil do eleitor por candidato",
    "resultado_partido_brasil": "Ouro - votos por partido Brasil",
    "resultado_partido_estado": "Ouro - votos por partido estado",
    "resultado_partido_municipio": "Ouro - votos por partido municipio",
    "resultado_partido": "Ouro - votos por partido nivelado",
    "resultado_candidato_brasil": "Ouro - votos por candidato Brasil",
    "resultado_candidato_estado": "Ouro - votos por candidato estado",
    "resultado_candidato_municipio": "Ouro - votos por candidato municipio",
    "resultado_candidato": "Ouro - votos por candidato nivelado",
    "top10_perfis": "Ouro - top 10 perfis Brasil/UF/municipio",
    "vencedor_secao": "Ouro - vencedores por secao ja concluidos",
    "resultado_eleitorado": "Ouro - resultado + eleitorado por secao",
    "perfil_candidatos": "Ouro - perfil dos candidatos",
    "cluster_voter_brasil": "Clusters - eleitores Brasil",
    "cluster_voter_estado": "Clusters - eleitores por estado",
    "cluster_voter_municipio": "Clusters - eleitores por municipio",
    "contagem_colunas_clusters_eleitores_brasil": "Histogramas - clusters de eleitores Brasil",
    "contagem_colunas_clusters_eleitores_estado": "Histogramas - clusters de eleitores por estado",
    "contagem_colunas_clusters_eleitores_municipio": "Histogramas - clusters de eleitores por municipio",
    "cluster_voter_personas": "Clusters - eleitorado discreto",
    "cluster_result_brasil": "Clusters - eleitores + resultado Brasil",
    "cluster_result_estado": "Clusters - eleitores + resultado por estado",
    "cluster_result_municipio": "Clusters - eleitores + resultado por municipio",
    "contagem_colunas_clusters_eleitores_resultado_brasil": "Histogramas - clusters eleitor + resultado Brasil",
    "contagem_colunas_clusters_eleitores_resultado_estado": "Histogramas - clusters eleitor + resultado por estado",
    "contagem_colunas_clusters_eleitores_resultado_municipio": "Histogramas - clusters eleitor + resultado por municipio",
    "cluster_result_personas": "Clusters - eleitorado + resultado",
    "banco_prata_eleitorado": "Prata - eleitorado limpo",
    "banco_prata_candidatos": "Prata - candidatos limpos",
    "banco_prata_resultados": "Prata - resultados/votos limpos",
    "sim_partidos_brasil": "Simulacao 2026 - partidos Brasil",
    "sim_partidos_estados": "Simulacao 2026 - partidos Estados",
    "sim_partidos_municipios": "Simulacao 2026 - partidos Municipios",
    "sim_partidos_correlacao": "Simulacao 2026 - correlacao historica",
}

ANALYSIS_MODES = [
    "completa",
    "estados_brasil",
    "eleitor",
    "candidato",
    "eleitor_partido",
    "eleitor_candidato_partido",
]

MODE_LABELS = {
    "completa": "Completa",
    "estados_brasil": "Estados + Brasil",
    "eleitor": "Somente eleitor",
    "candidato": "Somente candidato",
    "eleitor_partido": "Eleitor + partido",
    "eleitor_candidato_partido": "Eleitor + candidato + partido",
}

MODE_FEATURES = {
    "completa": {"brasil", "estado", "municipio", "perfil", "partido", "candidato", "cluster", "simulacao", "secao"},
    "estados_brasil": {"brasil", "estado", "perfil", "partido", "simulacao"},
    "eleitor": {"brasil", "estado", "municipio", "perfil"},
    "candidato": {"brasil", "estado", "municipio", "candidato"},
    "eleitor_partido": {"brasil", "estado", "municipio", "perfil", "partido", "simulacao"},
    "eleitor_candidato_partido": {"brasil", "estado", "municipio", "perfil", "partido", "candidato", "simulacao"},
}


def normalize_modalidade(value: Any) -> str:
    text = str(value or "completa").strip().lower()
    return text if text in MODE_FEATURES else "completa"


def modalidade_allows(value: Any, feature: str) -> bool:
    mode = normalize_modalidade(value)
    return str(feature).strip().lower() in MODE_FEATURES.get(mode, set())


def modalidade_info(value: Any) -> dict[str, Any]:
    mode = normalize_modalidade(value)
    features = sorted(MODE_FEATURES.get(mode, set()))
    return {
        "modalidade": mode,
        "label": MODE_LABELS.get(mode, mode),
        "features": features,
        "permite_brasil": "brasil" in features,
        "permite_estado": "estado" in features,
        "permite_municipio": "municipio" in features,
        "permite_perfil": "perfil" in features,
        "permite_partido": "partido" in features,
        "permite_candidato": "candidato" in features,
        "permite_cluster": "cluster" in features,
        "permite_simulacao": "simulacao" in features,
    }


def polars_available() -> bool:
    return pl is not None


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


def collect_lazy(lf: Any) -> Any:
    try:
        return lf.collect(engine="streaming")
    except TypeError:
        return lf.collect(streaming=True)


def records(df: Any) -> list[dict[str, Any]]:
    if df is None:
        return []
    if hasattr(df, "to_dicts"):
        return df.to_dicts()
    if hasattr(df, "to_dict"):
        return df.to_dict(orient="records")
    return []


class PolarsStore:
    def __init__(self, run_path: Path):
        if pl is None:
            raise RuntimeError(f"Polars nao instalado: {POLARS_IMPORT_ERROR}")
        self.run_path = run_path

    def __enter__(self) -> "PolarsStore":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    def path_for(self, key: str) -> Path | None:
        for rel in TABLE_CANDIDATES.get(key, []):
            path = self.run_path / rel
            if path.is_dir():
                if next(path.rglob("*.parquet"), None) is not None:
                    return path
                continue
            if path.exists():
                return path
        return None

    def available_tables(self) -> list[str]:
        return [key for key in TABLE_CANDIDATES if self.path_for(key) is not None]

    def scan(self, key: str) -> Any | None:
        path = self.path_for(key)
        if path is None:
            return None
        if path.is_dir():
            return pl.scan_parquet(str(path / "**" / "*.parquet"), hive_partitioning=True)
        suffix = path.suffix.lower()
        if suffix in {".csv", ".txt"}:
            return pl.scan_csv(str(path), separator=";", ignore_errors=True, infer_schema_length=5000)
        # A camada ouro nivelada grava alguns Parquets como arquivos sem extensao
        # (ex.: ouro/brasil/resumo). Esses arquivos nao podem cair no leitor CSV.
        if suffix in {"", ".parquet"}:
            return pl.scan_parquet(str(path), hive_partitioning=False)
        return pl.scan_csv(str(path), separator=";", ignore_errors=True, infer_schema_length=5000)

    def columns(self, key: str) -> list[str]:
        lf = self.scan(key)
        if lf is None:
            return []
        try:
            return list(lf.collect_schema().names())
        except Exception:
            try:
                return list(lf.schema.keys())
            except Exception:
                return []

    def ouro_resultados_status(self) -> dict[str, Any]:
        path = self.run_path / "logs" / "ouro" / "ouro_resultados_status_fatias.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            return {"erro": str(exc), "status": []}

    def ouro_resultados_summary(self) -> dict[str, Any]:
        status = self.ouro_resultados_status()
        rows = status.get("status") if isinstance(status, dict) else []
        rows = rows if isinstance(rows, list) else []
        concluidas = [r for r in rows if isinstance(r, dict) and str(r.get("status", "")).lower() == "concluido"]
        pendentes = [r for r in rows if isinstance(r, dict) and str(r.get("status", "")).lower() != "concluido"]
        return {
            "total": int(status.get("total") or len(rows) or 0) if isinstance(status, dict) else len(rows),
            "concluidas": int(status.get("concluidas") or len(concluidas)) if isinstance(status, dict) else len(concluidas),
            "pendentes": int(status.get("pendentes") or len(pendentes)) if isinstance(status, dict) else len(pendentes),
            "ufs_concluidas": sorted({str(r.get("uf", "")) for r in concluidas if r.get("uf")}),
            "ufs_pendentes": sorted({str(r.get("uf", "")) for r in pendentes if r.get("uf")}),
            "erro": status.get("erro", "") if isinstance(status, dict) else "",
        }

    def grouped_resultados_status(self, wanted_status: str) -> dict[str, list[str]]:
        rows = self.ouro_resultados_status().get("status") or []
        grouped: dict[str, list[str]] = {}
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            if str(row.get("status", "")).lower() != wanted_status:
                continue
            uf = clean_text(row.get("uf")) or "SEM_UF"
            ano = clean_text(row.get("ano")) or "sem ano"
            grouped.setdefault(uf, []).append(ano)
        return {uf: sorted(set(anos)) for uf, anos in sorted(grouped.items())}

    def _meaningful_expr(self, column: str) -> Any:
        value = pl.col(column).cast(pl.Utf8).str.strip_chars()
        lower = value.str.to_lowercase()
        return value.is_not_null() & (~lower.is_in(sorted(NULL_WORDS))) & (~lower.str.contains("sem valor", literal=True))

    def _first_col(self, cols: list[str], names: list[str]) -> str | None:
        for name in names:
            if name in cols:
                return name
        return None

    def _status_col(self, cols: list[str]) -> str | None:
        return self._first_col(
            cols,
            [
                "resultado_eleitoral",
                "situacao_eleitoral",
                "situacao_total",
                "situacao",
                "ds_sit_tot_turno",
                "situacao_turno",
                "status_eleicao",
                "turno",
            ],
        )

    def _with_winner_priority(self, lf: Any, cols: list[str]) -> tuple[Any, list[str]]:
        status_col = self._status_col(cols)
        if status_col:
            status = pl.col(status_col).cast(pl.Utf8, strict=False).fill_null("").str.strip_chars()
            lower = status.str.to_lowercase()
            negative = (
                lower.str.contains("nao eleito", literal=True)
                | lower.str.contains("não eleito", literal=True)
                | lower.str.contains("suplente", literal=True)
            )
            positive = lower.str.contains("eleito", literal=True) & (~negative)
            lf = lf.with_columns(
                pl.when(positive)
                .then(pl.lit("eleito"))
                .when(negative)
                .then(pl.lit("nao_eleito"))
                .otherwise(pl.lit("outro"))
                .alias("resultado_eleitoral"),
                pl.when(positive).then(pl.lit(1)).otherwise(pl.lit(0)).alias("_prioridade_eleito"),
            )
            return lf, list(dict.fromkeys([*cols, "resultado_eleitoral", "_prioridade_eleito"]))
        rank_col = self._first_col(cols, ["rank_entidade", "rank_partido", "rank_candidato", "rank"])
        if rank_col:
            rank = pl.col(rank_col).cast(pl.Float64, strict=False)
            elected = rank.eq(1)
            lf = lf.with_columns(
                pl.when(elected).then(pl.lit("vencedor_ranking")).otherwise(pl.lit("nao_vencedor_ranking")).alias("resultado_eleitoral"),
                pl.when(elected).then(pl.lit(1)).otherwise(pl.lit(0)).alias("_prioridade_eleito"),
            )
            return lf, list(dict.fromkeys([*cols, "resultado_eleitoral", "_prioridade_eleito"]))
        return lf, cols

    def _key_for_level(self, stem: str, nivel: str | None = None, uf: str | None = None, municipio: str | None = None) -> str:
        if nivel:
            normalized = str(nivel).strip().lower()
        elif municipio:
            normalized = "municipio"
        elif uf:
            normalized = "estado"
        else:
            normalized = "brasil"
        suffix = "municipio" if normalized == "municipio" else ("estado" if normalized == "estado" else "brasil")
        specific = f"{stem}_{suffix}"
        return specific if self.path_for(specific) is not None else stem

    def _profile_label_expr(self, cols: list[str]) -> Any | None:
        if "perfil_combinado" in cols:
            return pl.col("perfil_combinado").cast(pl.Utf8).alias("perfil_combinado")
        if "descricao" in cols:
            return pl.col("descricao").cast(pl.Utf8).alias("perfil_combinado")
        pieces: list[Any] = []
        for label, col in [
            ("Faixa", "perfil_faixa_etaria"),
            ("Sexo", "perfil_genero"),
            ("Escolaridade", "perfil_instrucao"),
            ("Estado civil", "perfil_estado_civil"),
            ("Raca/cor", "perfil_raca_cor"),
        ]:
            if col in cols:
                pieces.append(pl.concat_str([pl.lit(f"{label}: "), pl.col(col).cast(pl.Utf8).fill_null("")]))
        if not pieces:
            return None
        return pl.concat_str(pieces, separator=" | ").alias("perfil_combinado")

    def _apply_filters(self, lf: Any, cols: list[str], uf: str | None = None, municipio: str | None = None, ano: str | None = None) -> Any:
        if uf and "uf" in cols:
            lf = lf.filter(pl.col("uf").cast(pl.Utf8) == str(uf))
        if municipio:
            parts = str(municipio).split("|", 1)
            if len(parts) == 2 and "cd_municipio" in cols:
                lf = lf.filter(pl.col("cd_municipio").cast(pl.Utf8) == parts[0])
            elif "nm_municipio" in cols:
                lf = lf.filter(pl.col("nm_municipio").cast(pl.Utf8) == str(municipio))
        if ano:
            for col in ["ano", "ano_correlacao", "ano_num"]:
                if col in cols:
                    lf = lf.filter(pl.col(col).cast(pl.Utf8) == str(ano))
                    break
        return lf

    def municipios(self, uf: str | None) -> list[dict[str, str]]:
        if not uf:
            return []
        key = "municipal" if self.path_for("municipal") else "sim_partidos_municipios"
        lf = self.scan(key)
        cols = self.columns(key)
        if lf is None or "nm_municipio" not in cols:
            return []
        lf = lf.filter(self._meaningful_expr("nm_municipio"))
        if "uf" in cols:
            lf = lf.filter(pl.col("uf").cast(pl.Utf8) == str(uf))
        value_expr = (
            (pl.col("cd_municipio").cast(pl.Utf8) + pl.lit("|") + pl.col("nm_municipio").cast(pl.Utf8)).alias("value")
            if "cd_municipio" in cols
            else pl.col("nm_municipio").cast(pl.Utf8).alias("value")
        )
        df = collect_lazy(lf.select(value_expr, pl.col("nm_municipio").cast(pl.Utf8).alias("label")).unique().sort("label").limit(8000))
        return [{"label": clean_text(r.get("label")), "value": clean_text(r.get("value"))} for r in df.to_dicts() if clean_text(r.get("label"))]

    def top_profiles(self, nivel: str, uf: str | None = None, municipio: str | None = None, ano: str | None = None, limit: int = 10) -> Any:
        key = self._key_for_level("perfil_eleitor", nivel=nivel, uf=uf, municipio=municipio)
        lf = self.scan(key)
        cols = self.columns(key)
        if lf is None:
            key = "top10_perfis"
            lf = self.scan(key)
            cols = self.columns(key)
        if lf is None:
            return pl.DataFrame()
        if "nivel" in cols:
            lf = lf.filter(pl.col("nivel").cast(pl.Utf8).str.to_lowercase() == str(nivel).lower())
        else:
            lf = lf.with_columns(pl.lit(str(nivel).lower()).alias("nivel"))
            cols = cols + ["nivel"]
        lf = self._apply_filters(lf, cols, uf=uf, municipio=municipio, ano=ano)
        label_expr = self._profile_label_expr(cols)
        if label_expr is not None and "perfil_combinado" not in cols:
            lf = lf.with_columns(label_expr)
            cols = cols + ["perfil_combinado"]
        if "perfil_combinado" in cols:
            lf = lf.filter(self._meaningful_expr("perfil_combinado"))
        metric_col = self._first_col(cols, ["share_perfil", "share", "eleitorado", "votos", "peso"])
        if "share_perfil" not in cols and metric_col:
            total_expr = pl.col(metric_col).cast(pl.Float64, strict=False).sum()
            lf = lf.with_columns((pl.col(metric_col).cast(pl.Float64, strict=False) / total_expr).alias("share_perfil"))
            cols = cols + ["share_perfil"]
        sort_cols = [c for c in ["rank_perfil_ano", "share_perfil", "eleitorado"] if c in cols]
        if sort_cols:
            descending = [False if c == "rank_perfil_ano" else True for c in sort_cols]
            lf = lf.sort(sort_cols, descending=descending)
        selected = [
            c
            for c in [
                "nivel",
                "ano",
                "uf",
                "cd_municipio",
                "nm_municipio",
                "perfil_faixa_etaria",
                "perfil_genero",
                "perfil_instrucao",
                "perfil_estado_civil",
                "perfil_raca_cor",
                "perfil_combinado",
                "share_perfil",
                "eleitorado",
                "qtd_eleitores_perfil",
                "histograma_qtd_pessoas",
                "rank_perfil_ano",
                "padrao_temporal",
                "descricao",
            ]
            if c in cols
        ]
        if not selected:
            return pl.DataFrame()
        return collect_lazy(lf.select(selected).limit(int(limit)))

    def profile_distribution(self, ano: str | None = None, limit: int = 24) -> Any:
        key = "perfil_eleitor_brasil" if self.path_for("perfil_eleitor_brasil") is not None else "perfil_ano"
        lf = self.scan(key)
        cols = self.columns(key)
        if lf is None:
            return pl.DataFrame()
        if not {"dimensao_perfil", "valor_perfil"}.issubset(cols):
            return self._profile_distribution_from_profile_rows(lf, cols, ano=ano, limit=limit)
        lf = lf.filter(self._meaningful_expr("valor_perfil"))
        dim = pl.col("dimensao_perfil").cast(pl.Utf8).str.to_lowercase()
        lf = lf.filter(~dim.str.contains("biometria|data|hora"))
        lf = self._apply_filters(lf, cols, ano=ano)
        group_cols = [c for c in ["ano", "dimensao_perfil", "valor_perfil"] if c in cols]
        metric = pl.col("eleitorado").cast(pl.Float64, strict=False).sum().alias("peso") if "eleitorado" in cols else pl.len().alias("peso")
        aggs = [metric]
        if "share_eleitorado_ano" in cols:
            aggs.append(pl.col("share_eleitorado_ano").cast(pl.Float64, strict=False).mean().alias("share"))
        return collect_lazy(lf.group_by(group_cols).agg(aggs).sort("peso", descending=True).limit(int(limit)))

    def _profile_distribution_from_profile_rows(self, lf: Any, cols: list[str], ano: str | None = None, limit: int = 24) -> Any:
        profile_cols = [
            ("faixa_etaria", "perfil_faixa_etaria"),
            ("sexo_genero", "perfil_genero"),
            ("escolaridade", "perfil_instrucao"),
            ("estado_civil", "perfil_estado_civil"),
            ("raca_cor", "perfil_raca_cor"),
        ]
        available = [(label, col) for label, col in profile_cols if col in cols]
        if not available:
            return pl.DataFrame()
        lf = self._apply_filters(lf, cols, ano=ano)
        metric_col = "eleitorado" if "eleitorado" in cols else ("votos" if "votos" in cols else "")
        metric_expr = pl.col(metric_col).cast(pl.Float64, strict=False) if metric_col else pl.lit(1.0)
        frames = []
        for label, col in available:
            frames.append(
                lf.filter(self._meaningful_expr(col))
                .select(
                    *([pl.col("ano").cast(pl.Utf8).alias("ano")] if "ano" in cols else [pl.lit("").alias("ano")]),
                    pl.lit(label).alias("dimensao_perfil"),
                    pl.col(col).cast(pl.Utf8).alias("valor_perfil"),
                    metric_expr.alias("peso_base"),
                )
            )
        stacked = pl.concat(frames, how="vertical_relaxed")
        grouped = (
            stacked
            .group_by(["ano", "dimensao_perfil", "valor_perfil"])
            .agg(pl.col("peso_base").sum().alias("peso"))
            .with_columns((pl.col("peso") / pl.col("peso").sum().over(["ano", "dimensao_perfil"])).alias("share"))
            .sort("peso", descending=True)
            .limit(int(limit))
        )
        return collect_lazy(grouped)

    def party_prediction(self, key: str, uf: str | None = None, municipio: str | None = None, cenario: str | None = "base", limit: int = 20) -> Any:
        lf = self.scan(key)
        cols = self.columns(key)
        if lf is None or "partido" not in cols:
            return pl.DataFrame()
        lf = lf.filter(self._meaningful_expr("partido"))
        lf = self._apply_filters(lf, cols, uf=uf, municipio=municipio)
        if cenario and "cenario" in cols:
            lf = lf.filter(pl.col("cenario").cast(pl.Utf8) == str(cenario))
        selected = [c for c in ["cenario", "nivel", "uf", "cd_municipio", "nm_municipio", "cargo", "turno", "partido", "share_pred_2026", "votos_pred_2026", "perfil_eleitor_2026", "tendencia_partido", "forca_correlacao_historica", "justificativa_previsao_partido_2026"] if c in cols]
        order_col = "share_pred_2026" if "share_pred_2026" in cols else ("votos_pred_2026" if "votos_pred_2026" in cols else "partido")
        return collect_lazy(lf.select(selected).sort(order_col, descending=True).limit(int(limit)))

    def state_party_map(self, ano: str | None = None, cenario: str | None = "base", limit: int = 80) -> Any:
        sim = self._state_party_map_from_prediction(ano=ano, cenario=cenario, limit=limit)
        if sim is not None and sim.height:
            return sim
        return self._state_party_map_from_history(ano=ano, limit=limit)

    def _state_party_map_from_prediction(self, ano: str | None = None, cenario: str | None = "base", limit: int = 80) -> Any:
        key = "sim_partidos_estados"
        lf = self.scan(key)
        cols = self.columns(key)
        if lf is None or not {"uf", "partido"}.issubset(cols):
            return pl.DataFrame()
        lf = lf.filter(self._meaningful_expr("uf") & self._meaningful_expr("partido"))
        if cenario and "cenario" in cols:
            lf = lf.filter(pl.col("cenario").cast(pl.Utf8) == str(cenario))
        if ano:
            lf = self._apply_filters(lf, cols, ano=ano)
        aggs: list[Any] = []
        has_share = "share_pred_2026" in cols
        if has_share:
            aggs.append(pl.col("share_pred_2026").cast(pl.Float64, strict=False).max().alias("share_pred_2026"))
        if "votos_pred_2026" in cols:
            aggs.append(pl.col("votos_pred_2026").cast(pl.Float64, strict=False).sum().alias("votos_pred_2026"))
        if "perfil_eleitor_2026" in cols:
            aggs.append(pl.col("perfil_eleitor_2026").cast(pl.Utf8).drop_nulls().first().alias("perfil_eleitor_2026"))
        if not aggs:
            aggs.append(pl.len().alias("votos_pred_2026"))
        grouped = lf.group_by("uf", "partido").agg(aggs)
        if not has_share:
            grouped = grouped.with_columns((pl.col("votos_pred_2026") / pl.col("votos_pred_2026").sum().over("uf")).alias("share_pred_2026"))
        metric = "share_pred_2026" if "share_pred_2026" in grouped.collect_schema().names() else "votos_pred_2026"
        selected = [c for c in ["uf", "partido", "share_pred_2026", "votos_pred_2026", "perfil_eleitor_2026"] if c in grouped.collect_schema().names()]
        return collect_lazy(
            grouped
            .sort(["uf", metric], descending=[False, True])
            .unique(subset=["uf"], keep="first", maintain_order=True)
            .with_columns(pl.lit("simulacao_2026").alias("fonte"))
            .select(selected + ["fonte"])
            .sort("uf")
            .limit(int(limit))
        )

    def _state_party_map_from_history(self, ano: str | None = None, limit: int = 80) -> Any:
        for key in ["resultado_partido_estado", "resultado_partido", "vencedor_secao", "banco_prata_resultados"]:
            lf = self.scan(key)
            cols = self.columns(key)
            if lf is None or "uf" not in cols:
                continue
            party_col = self._first_col(cols, ["partido", "entidade", "partido_vencedor", "sg_partido", "nm_partido"])
            metric_col = self._first_col(cols, ["votos", "votos_pred_2026", "votos_vencedor", "qt_votos", "votos_total_secao", "qt_votos_nominais"])
            if party_col is None:
                continue
            lf = lf.filter(self._meaningful_expr("uf") & self._meaningful_expr(party_col))
            lf = self._apply_filters(lf, cols, ano=ano)
            metric = pl.col(metric_col).cast(pl.Float64, strict=False).sum().alias("votos_pred_2026") if metric_col else pl.len().alias("votos_pred_2026")
            grouped = (
                lf.group_by(pl.col("uf").cast(pl.Utf8).alias("uf"), pl.col(party_col).cast(pl.Utf8).alias("partido"))
                .agg(metric)
                .filter(pl.col("votos_pred_2026") > 0)
                .with_columns((pl.col("votos_pred_2026") / pl.col("votos_pred_2026").sum().over("uf")).alias("share_pred_2026"))
                .sort(["uf", "share_pred_2026"], descending=[False, True])
                .unique(subset=["uf"], keep="first", maintain_order=True)
                .with_columns(
                    pl.lit("historico_processado").alias("fonte"),
                    pl.lit("Resultado historico ja processado na camada ouro/prata.").alias("perfil_eleitor_2026"),
                )
                .select(["uf", "partido", "share_pred_2026", "votos_pred_2026", "perfil_eleitor_2026", "fonte"])
                .sort("uf")
                .limit(int(limit))
            )
            df = collect_lazy(grouped)
            if df.height:
                return df
        return pl.DataFrame()

    def historical_party_results(self, uf: str | None = None, municipio: str | None = None, ano: str | None = None, limit: int = 20) -> Any:
        first_key = self._key_for_level("resultado_partido", uf=uf, municipio=municipio)
        keys = [first_key, "resultado_partido", "vencedor_secao", "banco_prata_resultados"]
        for key in dict.fromkeys(keys):
            lf = self.scan(key)
            cols = self.columns(key)
            if lf is None:
                continue
            party_col = self._first_col(cols, ["partido", "entidade", "partido_vencedor", "sg_partido", "nm_partido"])
            metric_col = self._first_col(cols, ["votos", "votos_pred_2026", "votos_vencedor", "qt_votos", "votos_total_secao", "qt_votos_nominais"])
            if party_col is None:
                continue
            lf = lf.filter(self._meaningful_expr(party_col))
            lf = self._apply_filters(lf, cols, uf=uf, municipio=municipio, ano=ano)
            lf, cols = self._with_winner_priority(lf, cols)
            metric = pl.col(metric_col).cast(pl.Float64, strict=False).sum().alias("votos_pred_2026") if metric_col else pl.len().alias("votos_pred_2026")
            group_exprs = [pl.col(party_col).cast(pl.Utf8).alias("partido")]
            if "resultado_eleitoral" in cols:
                group_exprs.append(pl.col("resultado_eleitoral"))
            aggs = [metric]
            if "_prioridade_eleito" in cols:
                aggs.append(pl.col("_prioridade_eleito").max().alias("_prioridade_eleito"))
            agg = (
                lf.group_by(*group_exprs)
                .agg(aggs)
                .filter(pl.col("votos_pred_2026") > 0)
                .with_columns((pl.col("votos_pred_2026") / pl.col("votos_pred_2026").sum()).alias("share_pred_2026"))
                .with_columns(
                    pl.lit("historico_processado").alias("tendencia_partido"),
                    pl.lit("dados reais ja processados").alias("forca_correlacao_historica"),
                    pl.lit("Resultado historico parcial/total conforme fatias concluidas da camada ouro ou prata.").alias("perfil_eleitor_2026"),
                )
                .sort([c for c in ["_prioridade_eleito", "share_pred_2026"] if c in cols or c == "share_pred_2026"], descending=True)
                .limit(int(limit))
            )
            df = collect_lazy(agg)
            if df.height:
                return df
        return pl.DataFrame()

    def quick_party_results(self, nivel: str = "brasil", uf: str | None = None, municipio: str | None = None, ano: str | None = None, limit: int = 20) -> Any:
        key = self._key_for_level("resultado_partido", nivel=nivel, uf=uf, municipio=municipio)
        lf = self.scan(key)
        cols = self.columns(key)
        if lf is None:
            return pl.DataFrame()
        party_col = self._first_col(cols, ["partido", "entidade", "sg_partido", "nm_partido"])
        metric_col = self._first_col(cols, ["share_votos", "votos", "qt_votos", "votos_pred_2026"])
        if party_col is None:
            return pl.DataFrame()
        lf = self._apply_filters(lf, cols, uf=uf, municipio=municipio, ano=ano)
        lf = lf.filter(self._meaningful_expr(party_col))
        lf, cols = self._with_winner_priority(lf, cols)
        if party_col != "partido":
            lf = lf.with_columns(pl.col(party_col).cast(pl.Utf8).alias("partido"))
            cols = cols + ["partido"]
        if "share_votos" not in cols and metric_col:
            lf = lf.with_columns(
                (pl.col(metric_col).cast(pl.Float64, strict=False) / pl.col(metric_col).cast(pl.Float64, strict=False).sum()).alias("share_votos")
            )
            cols = cols + ["share_votos"]
        selected = [c for c in ["nivel", "ano", "uf", "cd_municipio", "nm_municipio", "cargo", "turno", "resultado_eleitoral", "partido", "share_votos", "votos", "rank_entidade"] if c in cols]
        sort_col = "share_votos" if "share_votos" in cols else (metric_col or "partido")
        sort_cols = [c for c in ["_prioridade_eleito", sort_col] if c in cols or c == sort_col]
        return collect_lazy(lf.select(selected + [c for c in ["_prioridade_eleito"] if c in cols]).sort(sort_cols, descending=True).drop([c for c in ["_prioridade_eleito"] if c in cols]).limit(int(limit)))

    def entity_results(
        self,
        entity: str = "partido",
        nivel: str = "brasil",
        uf: str | None = None,
        municipio: str | None = None,
        ano: str | None = None,
        limit: int = 20,
    ) -> Any:
        stem = "resultado_candidato" if str(entity).strip().lower() == "candidato" else "resultado_partido"
        key = self._key_for_level(stem, nivel=nivel, uf=uf, municipio=municipio)
        lf = self.scan(key)
        cols = self.columns(key)
        if lf is None:
            lf = self.scan(stem)
            cols = self.columns(stem)
        if lf is None:
            return pl.DataFrame()
        entity_col = self._first_col(cols, ["entidade", "candidato", "partido", "nm_candidato", "nm_votavel", "sg_partido"])
        metric_col = self._first_col(cols, ["share_votos", "votos", "votos_pred_2026", "qt_votos"])
        if entity_col is None:
            return pl.DataFrame()
        if "nivel" in cols:
            lf = lf.filter(pl.col("nivel").cast(pl.Utf8).str.to_lowercase() == str(nivel).lower())
        else:
            lf = lf.with_columns(pl.lit(str(nivel).lower()).alias("nivel"))
            cols = cols + ["nivel"]
        lf = self._apply_filters(lf, cols, uf=uf, municipio=municipio, ano=ano)
        lf = lf.filter(self._meaningful_expr(entity_col))
        lf, cols = self._with_winner_priority(lf, cols)
        if entity_col != "entidade":
            lf = lf.with_columns(pl.col(entity_col).cast(pl.Utf8).alias("entidade"))
            cols = cols + ["entidade"]
        order_col = metric_col or entity_col
        selected = [c for c in ["nivel", "ano", "uf", "cd_municipio", "nm_municipio", "cargo", "turno", "resultado_eleitoral", "entidade", "tipo_entidade", "votos", "share_votos", "rank_entidade"] if c in cols or c == "entidade"]
        sort_cols = [c for c in ["_prioridade_eleito", metric_col or entity_col] if c and (c in cols or c == entity_col)]
        return collect_lazy(lf.select(selected + [c for c in ["_prioridade_eleito"] if c in cols]).sort(sort_cols, descending=True).drop([c for c in ["_prioridade_eleito"] if c in cols]).limit(int(limit)))

    def entity_profiles(
        self,
        entity: str = "partido",
        nivel: str = "brasil",
        uf: str | None = None,
        municipio: str | None = None,
        ano: str | None = None,
        limit: int = 20,
    ) -> Any:
        stem = "perfil_candidato" if str(entity).strip().lower() == "candidato" else "perfil_partido"
        key = self._key_for_level(stem, nivel=nivel, uf=uf, municipio=municipio)
        lf = self.scan(key)
        cols = self.columns(key)
        if lf is None:
            lf = self.scan(stem)
            cols = self.columns(stem)
        if lf is None:
            return pl.DataFrame()
        entity_col = self._first_col(cols, ["entidade", "candidato", "partido", "nm_candidato", "sg_partido"])
        if entity_col is None:
            return pl.DataFrame()
        if "nivel" in cols:
            lf = lf.filter(pl.col("nivel").cast(pl.Utf8).str.to_lowercase() == str(nivel).lower())
        else:
            lf = lf.with_columns(pl.lit(str(nivel).lower()).alias("nivel"))
            cols = cols + ["nivel"]
        lf = self._apply_filters(lf, cols, uf=uf, municipio=municipio, ano=ano)
        lf = lf.filter(self._meaningful_expr(entity_col))
        if "perfil_combinado" in cols:
            lf = lf.filter(self._meaningful_expr("perfil_combinado"))
        if entity_col != "entidade":
            lf = lf.with_columns(pl.col(entity_col).cast(pl.Utf8).alias("entidade"))
            cols = cols + ["entidade"]
        sort_col = self._first_col(cols, ["share_perfil_na_entidade", "votos", "rank_perfil_entidade_ano"])
        if sort_col:
            lf = lf.sort(sort_col, descending=sort_col != "rank_perfil_entidade_ano")
        selected = [c for c in ["nivel", "ano", "uf", "cd_municipio", "nm_municipio", "cargo", "turno", "entidade", "perfil_combinado", "votos", "share_perfil_na_entidade", "rank_perfil_entidade_ano", "tipo_entidade", "descricao"] if c in cols or c == "entidade"]
        return collect_lazy(lf.select(selected).limit(int(limit)))

    def cluster_personas(
        self,
        tipo: str = "eleitores",
        nivel: str = "brasil",
        uf: str | None = None,
        municipio: str | None = None,
        ano: str | None = None,
        limit: int = 20,
    ) -> Any:
        stem = "cluster_result" if str(tipo).strip().lower() in {"resultado", "eleitores_resultado", "result"} else "cluster_voter"
        key = self._key_for_level(stem, nivel=nivel, uf=uf, municipio=municipio)
        lf = self.scan(key)
        cols = self.columns(key)
        if lf is None:
            fallback = "cluster_result_personas" if stem == "cluster_result" else "cluster_voter_personas"
            lf = self.scan(fallback)
            cols = self.columns(fallback)
        if lf is None:
            return pl.DataFrame()
        if "nivel" in cols:
            lf = lf.filter(pl.col("nivel").cast(pl.Utf8).str.to_lowercase() == str(nivel).lower())
        else:
            lf = lf.with_columns(pl.lit(str(nivel).lower()).alias("nivel"))
            cols = cols + ["nivel"]
        lf = self._apply_filters(lf, cols, uf=uf, municipio=municipio, ano=ano)
        if "perfil_combinado" in cols:
            lf = lf.filter(self._meaningful_expr("perfil_combinado"))
        if "descricao" not in cols and "perfil_combinado" in cols:
            if "partido" in cols:
                lf = lf.with_columns(
                    pl.concat_str(
                        [
                            pl.lit("Pessoa do cluster: "),
                            pl.col("perfil_combinado").cast(pl.Utf8),
                            pl.lit(". Tendencia partidaria: "),
                            pl.col("partido").cast(pl.Utf8).fill_null(""),
                            pl.lit("."),
                        ]
                    ).alias("descricao")
                )
            else:
                lf = lf.with_columns(pl.concat_str([pl.lit("Pessoa do cluster: "), pl.col("perfil_combinado").cast(pl.Utf8), pl.lit(".")]).alias("descricao"))
            cols = cols + ["descricao"]
        sort_col = self._first_col(cols, ["share_cluster", "peso_cluster", "eleitorado", "votos_proxy"])
        if sort_col:
            lf = lf.sort(sort_col, descending=True)
        selected = [
            c
            for c in [
                "nivel",
                "ano",
                "uf",
                "cd_municipio",
                "nm_municipio",
                "cluster_id",
                "cluster_tipo",
                "perfil_faixa_etaria",
                "perfil_genero",
                "perfil_instrucao",
                "perfil_estado_civil",
                "perfil_raca_cor",
                "perfil_combinado",
                "partido",
                "eleitorado",
                "votos_proxy",
                "share_cluster",
                "rank_persona_cluster",
                "algoritmo_cluster",
                "tipo_features_cluster",
                "descricao",
            ]
            if c in cols
        ]
        if not selected:
            return pl.DataFrame()
        return collect_lazy(lf.select(selected).limit(int(limit)))

    def metrics_by_year(self, key: str, uf: str | None = None, municipio: str | None = None) -> Any:
        lf = self.scan(key)
        cols = self.columns(key)
        if lf is None:
            return pl.DataFrame()
        year_col = self._first_col(cols, ["ano", "ano_correlacao", "ano_num"])
        if year_col is None:
            return pl.DataFrame()
        lf = self._apply_filters(lf, cols, uf=uf, municipio=municipio)
        metrics = [c for c in ["votos", "eleitorado", "comparecimento_estimado", "abstencao_estimado", "votos_total", "eleitorado_total", "comparecimento_medio", "abstencao_media"] if c in cols]
        if not metrics:
            return pl.DataFrame()
        aggs = [pl.col(c).cast(pl.Float64, strict=False).sum().alias(c) for c in metrics]
        return collect_lazy(lf.group_by(pl.col(year_col).cast(pl.Utf8).alias("ano")).agg(aggs).sort("ano"))

    def table(self, key: str, limit: int = 100) -> Any:
        lf = self.scan(key)
        if lf is None:
            return pl.DataFrame()
        return collect_lazy(lf.limit(int(limit)))
