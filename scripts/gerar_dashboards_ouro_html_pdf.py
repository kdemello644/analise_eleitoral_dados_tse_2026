from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
except ModuleNotFoundError as exc:  # pragma: no cover - depende do ambiente do usuario
    raise SystemExit("Instale plotly para gerar os dashboards HTML: python3 -m pip install plotly") from exc

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.pdfgen import canvas
except ModuleNotFoundError:  # pragma: no cover - PDF fica opcional
    colors = None  # type: ignore[assignment]
    A4 = None  # type: ignore[assignment]
    landscape = None  # type: ignore[assignment]
    canvas = None  # type: ignore[assignment]

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from parquet_query_polars_eleitoral import PolarsStore, collect_lazy, polars_available, records as polars_records  # noqa: E402


UF_CENTROIDS: dict[str, tuple[float, float]] = {
    "AC": (-8.77, -70.55),
    "AL": (-9.62, -36.82),
    "AM": (-3.47, -65.1),
    "AP": (1.41, -51.77),
    "BA": (-12.97, -38.51),
    "CE": (-5.2, -39.53),
    "DF": (-15.83, -47.86),
    "ES": (-19.19, -40.34),
    "GO": (-15.98, -49.86),
    "MA": (-5.42, -45.44),
    "MG": (-18.1, -44.38),
    "MS": (-20.51, -54.54),
    "MT": (-12.64, -55.42),
    "PA": (-3.79, -52.48),
    "PB": (-7.28, -36.72),
    "PE": (-8.38, -37.86),
    "PI": (-6.6, -42.28),
    "PR": (-24.89, -51.55),
    "RJ": (-22.25, -42.66),
    "RN": (-5.81, -36.59),
    "RO": (-10.83, -63.34),
    "RR": (1.99, -61.33),
    "RS": (-30.17, -53.5),
    "SC": (-27.45, -50.95),
    "SE": (-10.57, -37.45),
    "SP": (-22.19, -48.79),
    "TO": (-10.25, -48.25),
}


THEME = {
    "bg": "#f4f7fb",
    "paper": "#ffffff",
    "ink": "#0d172a",
    "muted": "#617089",
    "line": "#d9e2ef",
    "accent": "#0f766e",
    "accent2": "#2563eb",
    "hot": "#f97316",
    "dark": "#0b1424",
}


LOG_TEXT_PATH: Path | None = None
LOG_JSONL_PATH: Path | None = None


def set_log_outputs(text_path: Path, jsonl_path: Path) -> None:
    global LOG_TEXT_PATH, LOG_JSONL_PATH
    LOG_TEXT_PATH = text_path
    LOG_JSONL_PATH = jsonl_path
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text("", encoding="utf-8")
    jsonl_path.write_text("", encoding="utf-8")


def _log_value(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def log(message: str, **fields: Any) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    suffix = ""
    if fields:
        suffix = " | " + " | ".join(f"{key}={_log_value(value)}" for key, value in fields.items())
    line = f"{timestamp} | INFO | {message}{suffix}"
    print(line, flush=True)
    if LOG_TEXT_PATH is not None:
        with LOG_TEXT_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    if LOG_JSONL_PATH is not None:
        payload = {"timestamp": timestamp, "mensagem": message, **{key: _log_value(value) for key, value in fields.items()}}
        with LOG_JSONL_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def df_brief(df: pd.DataFrame, max_cols: int = 10) -> dict[str, Any]:
    if not isinstance(df, pd.DataFrame):
        return {"tipo": type(df).__name__}
    cols = list(df.columns)
    return {
        "linhas": int(len(df)),
        "colunas": int(len(cols)),
        "campos": ",".join(cols[:max_cols]),
    }


def log_df(label: str, df: pd.DataFrame) -> None:
    log("dados prontos", etapa=label, **df_brief(df))


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path.cwd() / path).resolve()


def slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value).strip())
    return text.strip("_") or "sem_nome"


def to_df(data: Any) -> pd.DataFrame:
    if data is None:
        return pd.DataFrame()
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return pd.DataFrame(polars_records(data))


def as_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0)


def first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def format_num(value: Any) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except Exception:
        return str(value)
    if math.isnan(number):
        return "-"
    if 0 < abs(number) < 1:
        return f"{number:.1%}"
    return f"{number:,.0f}".replace(",", ".")


def safe_sum(df: pd.DataFrame, candidates: list[str]) -> float:
    col = first_col(df, candidates)
    if not col:
        return 0.0
    return float(as_number(df[col]).sum())


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        return {"erro": str(exc)}


@dataclass
class OutputPaths:
    root: Path
    estados: Path
    pdf: Path
    assets: Path


@dataclass
class GeneratorConfig:
    banco: Path
    out: Path
    top_n: int
    max_municipios_por_estado: int
    ufs: list[str]
    ano: str
    cenario: str
    gerar_pdf: bool
    self_contained: bool


class StaticDashboardGenerator:
    def __init__(self, cfg: GeneratorConfig):
        self.cfg = cfg
        self.store = PolarsStore(cfg.banco)
        self.paths = OutputPaths(
            root=cfg.out,
            estados=cfg.out / "estados",
            pdf=cfg.out / "pdf",
            assets=cfg.out / "assets",
        )
        for path in [self.paths.root, self.paths.estados, self.paths.pdf, self.paths.assets]:
            path.mkdir(parents=True, exist_ok=True)
        set_log_outputs(self.paths.assets / "geracao_dashboards.log", self.paths.assets / "geracao_dashboards_eventos.jsonl")
        log(
            "gerador inicializado",
            banco=cfg.banco,
            saida=cfg.out,
            top_n=cfg.top_n,
            max_municipios_por_estado=cfg.max_municipios_por_estado,
            ano=cfg.ano or "todos",
            cenario=cfg.cenario,
            gerar_pdf=cfg.gerar_pdf,
        )
        self.include_plotlyjs = True if cfg.self_contained else "cdn"
        self.manifest: dict[str, Any] = {
            "banco": str(cfg.banco),
            "saida": str(cfg.out),
            "gerado_em": datetime.now().isoformat(timespec="seconds"),
            "modo_processamento": "polars_lazy_streaming_agregado_antes_do_pandas",
            "arquivos": [],
            "observacoes": [],
        }
        self.ufs_processadas: list[str] = []

    def run(self) -> Path:
        started = time.perf_counter()
        if not polars_available():
            raise SystemExit("Polars nao esta instalado neste ambiente.")
        log("lendo camada ouro", banco=self.cfg.banco)
        ufs = self.selected_ufs()
        self.ufs_processadas = list(ufs)
        log("ufs selecionadas", quantidade=len(ufs), ufs=", ".join(ufs) if ufs else "nenhuma")
        log("bloco global inicio")
        global_data = self.build_global_data(ufs)
        log("bloco global dados coletados")
        self.write_global_html(global_data, ufs)
        if self.cfg.gerar_pdf:
            self.write_global_pdf(global_data, ufs)
        log("bloco global fim")
        for idx, uf in enumerate(ufs, start=1):
            uf_started = time.perf_counter()
            log("uf inicio", indice=idx, total=len(ufs), uf=uf)
            state_data = self.build_state_data(uf)
            log("uf dados coletados", uf=uf)
            self.write_state_html(uf, state_data)
            if self.cfg.gerar_pdf:
                self.write_state_pdf(uf, state_data)
            log("uf fim", uf=uf, duracao_segundos=round(time.perf_counter() - uf_started, 3))
        manifest_path = self.paths.assets / "manifesto_dashboards.json"
        manifest_path.write_text(json.dumps(self.manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        log("manifesto gerado", arquivo=manifest_path)
        log("geracao finalizada", saida=self.paths.root, duracao_segundos=round(time.perf_counter() - started, 3))
        return self.paths.root

    def selected_ufs(self) -> list[str]:
        if self.cfg.ufs:
            ufs = sorted({uf.strip().upper() for uf in self.cfg.ufs if uf.strip()})
            log("selecionando ufs por parametro", quantidade=len(ufs), ufs=", ".join(ufs))
            return ufs
        resumo = self.store.ouro_resultados_summary()
        ufs = sorted(set(resumo.get("ufs_concluidas") or []) | set(resumo.get("ufs_pendentes") or []))
        if ufs:
            log(
                "selecionando ufs pelo manifesto ouro_resultados",
                total=resumo.get("total", 0),
                concluidas=resumo.get("concluidas", 0),
                pendentes=resumo.get("pendentes", 0),
                ufs=len(ufs),
            )
            return ufs
        log("manifesto sem ufs; tentando timeline_uf")
        df = self.safe_call(lambda: to_df(self.store.metrics_by_year("timeline_uf")), "timeline_uf_ufs")
        if "uf" in df.columns:
            ufs = sorted(df["uf"].dropna().astype(str).str.upper().unique().tolist())
            log("ufs detectadas por timeline_uf", quantidade=len(ufs))
            return ufs
        log("timeline_uf sem ufs; tentando retrato municipal")
        df = self.safe_collect_table("municipal", limit=100000)
        if "uf" in df.columns:
            ufs = sorted(df["uf"].dropna().astype(str).str.upper().unique().tolist())
            log("ufs detectadas por retrato municipal", quantidade=len(ufs))
            return ufs
        log("nenhuma uf detectada")
        return []

    def safe_call(self, fn: Any, label: str) -> pd.DataFrame:
        started = time.perf_counter()
        log("coleta inicio", etapa=label)
        try:
            df = fn()
            if isinstance(df, pd.DataFrame):
                log("coleta fim", etapa=label, duracao_segundos=round(time.perf_counter() - started, 3), **df_brief(df))
            else:
                log("coleta fim", etapa=label, duracao_segundos=round(time.perf_counter() - started, 3), tipo=type(df).__name__)
            return df
        except Exception as exc:
            msg = f"{label}: {exc}"
            log("coleta erro", etapa=label, erro=str(exc), duracao_segundos=round(time.perf_counter() - started, 3))
            self.manifest["observacoes"].append(msg)
            return pd.DataFrame()

    def safe_collect_table(self, key: str, limit: int = 1000, uf: str | None = None) -> pd.DataFrame:
        def collect() -> pd.DataFrame:
            lf = self.store.scan(key)
            cols = self.store.columns(key)
            log("scan tabela", tabela=key, caminho=self.store.path_for(key) or "nao_encontrado", limite=limit, uf=uf or "todos", colunas=len(cols))
            if lf is None:
                log("scan tabela vazio", tabela=key)
                return pd.DataFrame()
            import polars as pl

            if uf and "uf" in cols:
                log("aplicando filtro uf", tabela=key, uf=uf)
                lf = lf.filter(pl.col("uf").cast(pl.Utf8) == uf)
            return to_df(collect_lazy(lf.limit(limit)))

        return self.safe_call(collect, f"coleta_{key}")

    def lazy_columns(self, lf: Any) -> list[str]:
        try:
            return list(lf.collect_schema().names())
        except Exception:
            try:
                return list(lf.schema.keys())
            except Exception:
                return []

    def collect_streaming_df(self, label: str, lf: Any) -> pd.DataFrame:
        started = time.perf_counter()
        log("streaming collect inicio", etapa=label)
        df = to_df(collect_lazy(lf))
        log("streaming collect fim", etapa=label, duracao_segundos=round(time.perf_counter() - started, 3), **df_brief(df))
        return df

    def scan_vencedores_uf(self, uf: str) -> tuple[Any | None, list[str], str]:
        import polars as pl

        base = self.store.path_for("vencedor_secao")
        if base is None:
            log("scan vencedor uf indisponivel", uf=uf, motivo="sem_tabela_vencedor_secao")
            return None, [], ""
        if base.is_dir():
            files = sorted(base.glob(f"chunk={uf}_*/**/*.parquet"))
            if files:
                log("scan vencedor uf por chunks", uf=uf, arquivos=len(files), base=base)
                lf = pl.scan_parquet([str(path) for path in files], hive_partitioning=True)
                return lf, self.lazy_columns(lf), f"{base}/chunk={uf}_*/**/*.parquet"
            log("scan vencedor uf sem chunks especificos", uf=uf, base=base)
        lf = self.store.scan("vencedor_secao")
        cols = self.store.columns("vencedor_secao")
        if lf is None:
            return None, [], str(base)
        if "uf" in cols:
            log("scan vencedor uf fallback com filtro", uf=uf, base=base)
            lf = lf.filter(pl.col("uf").cast(pl.Utf8) == uf)
        return lf, cols, str(base)

    def historical_party_results_streaming(self, uf: str | None = None, limit: int | None = None) -> pd.DataFrame:
        direct = to_df(self.store.historical_party_results(uf=uf, limit=limit or self.cfg.top_n))
        if not direct.empty:
            log("partidos usando camada ouro nivelada", uf=uf or "BR", **df_brief(direct))
            return direct
        if uf:
            return self.party_aggregate_for_uf(uf, limit=limit)
        frames = []
        for idx, state in enumerate(self.ufs_processadas, start=1):
            log("partidos brasil agregando uf", indice=idx, total=len(self.ufs_processadas), uf=state)
            frame = self.party_aggregate_for_uf(state, limit=None)
            if not frame.empty:
                frames.append(frame)
        if not frames:
            return pd.DataFrame()
        work = pd.concat(frames, ignore_index=True)
        if "partido" not in work.columns or "votos_pred_2026" not in work.columns:
            return pd.DataFrame()
        work["votos_pred_2026"] = as_number(work["votos_pred_2026"])
        out = work.groupby("partido", as_index=False)["votos_pred_2026"].sum()
        total = float(out["votos_pred_2026"].sum() or 0)
        out["share_pred_2026"] = out["votos_pred_2026"] / total if total else 0
        out["fonte"] = "historico_streaming_por_uf"
        out["perfil_eleitor_2026"] = "Resultado historico agregado por UF em streaming; dados individuais nao foram carregados em memoria."
        out = out.sort_values("share_pred_2026", ascending=False)
        if limit:
            out = out.head(int(limit))
        log("partidos brasil agregado streaming", **df_brief(out))
        return out

    def party_aggregate_for_uf(self, uf: str, limit: int | None = None) -> pd.DataFrame:
        import polars as pl

        direct = to_df(self.store.historical_party_results(uf=uf, limit=limit or self.cfg.top_n))
        if not direct.empty:
            log("partidos uf usando resultado_partido nivelado", uf=uf, **df_brief(direct))
            return direct
        lf, cols, origem = self.scan_vencedores_uf(uf)
        if lf is None:
            return pd.DataFrame()
        party_col = first_existing(cols, ["partido_vencedor", "partido", "sg_partido", "nm_partido"])
        metric_col = first_existing(cols, ["votos_vencedor", "votos", "qt_votos", "votos_total_secao", "qt_votos_nominais"])
        if not party_col:
            log("partidos uf indisponivel", uf=uf, origem=origem, motivo="sem_coluna_partido")
            return pd.DataFrame()
        log("partidos uf agregando streaming", uf=uf, origem=origem, partido=party_col, metrica=metric_col or "contagem", limite=limit or "todos")
        if self.cfg.ano:
            for year_col in ["ano", "ano_correlacao", "ano_num"]:
                if year_col in cols:
                    lf = lf.filter(pl.col(year_col).cast(pl.Utf8) == str(self.cfg.ano))
                    log("partidos uf filtro ano", uf=uf, coluna=year_col, ano=self.cfg.ano)
                    break
        metric = pl.col(metric_col).cast(pl.Float64, strict=False).sum().alias("votos_pred_2026") if metric_col else pl.len().alias("votos_pred_2026")
        query = (
            lf.filter(self.store._meaningful_expr(party_col))
            .group_by(pl.col(party_col).cast(pl.Utf8).alias("partido"))
            .agg(metric)
            .filter(pl.col("votos_pred_2026") > 0)
            .sort("votos_pred_2026", descending=True)
        )
        if limit:
            query = query.limit(int(limit))
        out = self.collect_streaming_df(f"partidos_{uf}", query)
        if out.empty:
            return out
        total = float(as_number(out["votos_pred_2026"]).sum() or 0)
        out["share_pred_2026"] = as_number(out["votos_pred_2026"]) / total if total else 0
        out["uf"] = uf
        out["fonte"] = "historico_streaming_uf"
        out["perfil_eleitor_2026"] = "Resultado historico da UF agregado em streaming."
        return out

    def state_party_map_streaming(self, ufs: list[str]) -> pd.DataFrame:
        sim = to_df(self.store.party_prediction("sim_partidos_estados", cenario=self.cfg.cenario, limit=5000))
        if not sim.empty and {"uf", "partido"}.issubset(sim.columns):
            log("mapa estados usando simulacao", **df_brief(sim))
            value_col = first_existing(list(sim.columns), ["share_pred_2026", "votos_pred_2026"])
            if value_col:
                sim[value_col] = as_number(sim[value_col])
                return sim.sort_values(["uf", value_col], ascending=[True, False]).drop_duplicates("uf").head(100)
        frames = []
        for idx, uf in enumerate(ufs, start=1):
            log("mapa estados historico uf", indice=idx, total=len(ufs), uf=uf)
            df = self.party_aggregate_for_uf(uf, limit=1)
            if not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        log("mapa estados historico pronto", **df_brief(out))
        return out

    def build_global_data(self, ufs: list[str]) -> dict[str, pd.DataFrame | dict[str, Any]]:
        started = time.perf_counter()
        log("montagem global inicio", ufs=len(ufs))
        data: dict[str, pd.DataFrame | dict[str, Any]] = {}
        data["timeline"] = self.safe_call(lambda: to_df(self.store.metrics_by_year("timeline_nacional")), "timeline_nacional")
        data["partidos"] = self.safe_call(lambda: self.best_parties("brasil"), "partidos_brasil")
        data["perfis"] = self.safe_call(lambda: to_df(self.store.top_profiles("brasil", limit=self.cfg.top_n)), "perfis_brasil")
        data["perfil_discreto"] = self.safe_call(lambda: to_df(self.store.profile_distribution(ano=self.cfg.ano or None, limit=self.cfg.top_n * 4)), "perfil_discreto_brasil")
        data["mapa_estados"] = self.safe_call(lambda: self.state_party_map_streaming(ufs), "mapa_estados_streaming")
        data["timeline_uf"] = self.safe_call(lambda: self.collect_timeline_uf(), "timeline_uf")
        data["status"] = self.store.ouro_resultados_summary()
        log("status ouro resultados carregado", status=data["status"])
        data["clusters"] = self.find_cluster_like_data()
        log("montagem global fim", duracao_segundos=round(time.perf_counter() - started, 3))
        return data

    def build_state_data(self, uf: str) -> dict[str, pd.DataFrame | dict[str, Any]]:
        started = time.perf_counter()
        log("montagem estado inicio", uf=uf)
        data: dict[str, pd.DataFrame | dict[str, Any]] = {}
        data["timeline"] = self.safe_call(lambda: to_df(self.store.metrics_by_year("timeline_uf", uf=uf)), f"timeline_{uf}")
        data["partidos"] = self.safe_call(lambda: self.best_parties("estado", uf=uf), f"partidos_{uf}")
        data["perfis"] = self.safe_call(lambda: to_df(self.store.top_profiles("estado", uf=uf, limit=self.cfg.top_n)), f"perfis_{uf}")
        data["municipios"] = self.safe_call(lambda: self.collect_municipal_snapshot(uf), f"municipios_{uf}")
        data["municipios_partidos"] = self.safe_call(lambda: self.collect_municipal_party_snapshot(uf), f"municipios_partidos_{uf}")
        data["clusters"] = self.find_cluster_like_data(uf=uf)
        log("montagem estado fim", uf=uf, duracao_segundos=round(time.perf_counter() - started, 3))
        return data

    def best_parties(self, escopo: str, uf: str | None = None, municipio: str | None = None) -> pd.DataFrame:
        log("partidos consulta inicio", escopo=escopo, uf=uf or "", municipio=municipio or "", cenario=self.cfg.cenario, ano=self.cfg.ano or "")
        if escopo == "brasil":
            data = self.store.party_prediction("sim_partidos_brasil", cenario=self.cfg.cenario, limit=self.cfg.top_n)
            df = to_df(data)
            if not df.empty:
                log("partidos usando simulacao", escopo=escopo, linhas=len(df))
                return df
            hist = self.historical_party_results_streaming(limit=self.cfg.top_n)
            log("partidos usando historico", escopo=escopo, linhas=len(hist))
            return hist
        if escopo == "estado":
            data = self.store.party_prediction("sim_partidos_estados", uf=uf, cenario=self.cfg.cenario, limit=self.cfg.top_n)
            df = to_df(data)
            if not df.empty:
                log("partidos usando simulacao", escopo=escopo, uf=uf or "", linhas=len(df))
                return df
            hist = self.historical_party_results_streaming(uf=uf, limit=self.cfg.top_n)
            log("partidos usando historico", escopo=escopo, uf=uf or "", linhas=len(hist))
            return hist
        data = self.store.party_prediction("sim_partidos_municipios", uf=uf, municipio=municipio, cenario=self.cfg.cenario, limit=self.cfg.top_n)
        df = to_df(data)
        if not df.empty:
            log("partidos usando simulacao", escopo=escopo, uf=uf or "", municipio=municipio or "", linhas=len(df))
            return df
        hist = self.historical_party_results_streaming(uf=uf, limit=self.cfg.top_n)
        log("partidos usando historico", escopo=escopo, uf=uf or "", municipio=municipio or "", linhas=len(hist))
        return hist

    def collect_timeline_uf(self) -> pd.DataFrame:
        started = time.perf_counter()
        lf = self.store.scan("timeline_uf")
        cols = self.store.columns("timeline_uf")
        log("timeline_uf scan", caminho=self.store.path_for("timeline_uf") or "nao_encontrado", colunas=len(cols))
        if lf is None or "uf" not in cols:
            log("timeline_uf indisponivel", motivo="sem_lazyframe_ou_sem_uf")
            return pd.DataFrame()
        metric = first_existing(cols, ["eleitorado", "eleitorado_total", "votos", "votos_total", "comparecimento_estimado"])
        if not metric:
            log("timeline_uf indisponivel", motivo="sem_metrica")
            return pd.DataFrame()
        import polars as pl

        aggs = [pl.col(metric).cast(pl.Float64, strict=False).sum().alias(metric)]
        for col in ["comparecimento_estimado", "abstencao_estimado", "votos", "votos_total"]:
            if col in cols and col != metric:
                aggs.append(pl.col(col).cast(pl.Float64, strict=False).sum().alias(col))
        log("timeline_uf agregando", metrica_principal=metric, agregacoes=len(aggs))
        df = collect_lazy(lf.group_by("uf").agg(aggs).sort(metric, descending=True).limit(80))
        out = to_df(df)
        log("timeline_uf agregado", duracao_segundos=round(time.perf_counter() - started, 3), **df_brief(out))
        return out

    def collect_municipal_snapshot(self, uf: str) -> pd.DataFrame:
        started = time.perf_counter()
        lf = self.store.scan("municipal")
        cols = self.store.columns("municipal")
        log("municipios scan", uf=uf, caminho=self.store.path_for("municipal") or "nao_encontrado", colunas=len(cols))
        if lf is None or "uf" not in cols:
            log("municipios indisponivel", uf=uf, motivo="sem_lazyframe_ou_sem_uf")
            return pd.DataFrame()
        import polars as pl

        name_col = first_existing(cols, ["nm_municipio", "municipio"])
        code_col = first_existing(cols, ["cd_municipio", "codigo_municipio"])
        metric_cols = [
            col
            for col in ["eleitorado", "eleitorado_total", "votos", "votos_total", "comparecimento_estimado", "abstencao_estimado", "comparecimento_medio", "abstencao_media"]
            if col in cols
        ]
        if not name_col or not metric_cols:
            log("municipios indisponivel", uf=uf, motivo="sem_nome_ou_metricas", nome=name_col or "", metricas=",".join(metric_cols))
            return pd.DataFrame()
        group_cols = [c for c in ["uf", code_col, name_col, "ano"] if c and c in cols]
        aggs = [pl.col(col).cast(pl.Float64, strict=False).sum().alias(col) for col in metric_cols]
        sort_col = first_existing(metric_cols, ["eleitorado", "eleitorado_total", "votos_total", "votos", "comparecimento_estimado"]) or metric_cols[0]
        log("municipios agregando", uf=uf, grupos=",".join(group_cols), metricas=",".join(metric_cols), ordenacao=sort_col, limite=self.cfg.max_municipios_por_estado)
        df = (
            lf.filter(pl.col("uf").cast(pl.Utf8) == uf)
            .group_by(group_cols)
            .agg(aggs)
            .sort(sort_col, descending=True)
            .limit(int(self.cfg.max_municipios_por_estado))
        )
        df = collect_lazy(df)
        out = to_df(df)
        log("municipios agregado", uf=uf, duracao_segundos=round(time.perf_counter() - started, 3), **df_brief(out))
        return out

    def collect_municipal_party_snapshot(self, uf: str) -> pd.DataFrame:
        started = time.perf_counter()
        lf, cols, origem = self.scan_vencedores_uf(uf)
        log("municipios_partidos scan", uf=uf, caminho=origem or "nao_encontrado", colunas=len(cols), modo="streaming_chunk_uf")
        if lf is None:
            log("municipios_partidos indisponivel", uf=uf, motivo="sem_lazyframe_ou_sem_uf")
            return pd.DataFrame()
        import polars as pl

        party_col = first_existing(cols, ["partido_vencedor", "partido", "sg_partido", "nm_partido"])
        metric_col = first_existing(cols, ["votos_vencedor", "votos", "qt_votos", "votos_total_secao", "qt_votos_nominais"])
        name_col = first_existing(cols, ["nm_municipio", "municipio"])
        code_col = first_existing(cols, ["cd_municipio", "codigo_municipio"])
        if not party_col or not name_col:
            log("municipios_partidos indisponivel", uf=uf, motivo="sem_partido_ou_municipio", partido=party_col or "", municipio=name_col or "")
            return pd.DataFrame()
        group_cols = [c for c in [code_col, name_col, party_col] if c]
        metric = pl.col(metric_col).cast(pl.Float64, strict=False).sum().alias("votos") if metric_col else pl.len().alias("votos")
        log("municipios_partidos agregando", uf=uf, grupos=",".join(group_cols), metrica=metric_col or "contagem", limite=int(self.cfg.max_municipios_por_estado * 3))
        query = lf
        if "uf" in cols:
            query = query.filter(pl.col("uf").cast(pl.Utf8) == uf)
        df = (
            query.filter(self.store._meaningful_expr(party_col))
            .group_by(group_cols)
            .agg(metric)
            .sort("votos", descending=True)
            .limit(int(self.cfg.max_municipios_por_estado * 3))
        )
        df = collect_lazy(df)
        out = to_df(df)
        if party_col in out.columns:
            out = out.rename(columns={party_col: "partido"})
        if name_col in out.columns:
            out = out.rename(columns={name_col: "nm_municipio"})
        log("municipios_partidos agregado", uf=uf, duracao_segundos=round(time.perf_counter() - started, 3), **df_brief(out))
        return out

    def find_cluster_like_data(self, uf: str | None = None) -> pd.DataFrame:
        started = time.perf_counter()
        log("clusters busca inicio", uf=uf or "BR")
        nivel = "estado" if uf else "brasil"
        frames = []
        for tipo in ["eleitores", "resultado"]:
            frame = to_df(self.store.cluster_personas(tipo=tipo, nivel=nivel, uf=uf, limit=self.cfg.top_n))
            if not frame.empty:
                frame = frame.copy()
                frame["tipo_cluster_consulta"] = tipo
                frames.append(frame)
                log("clusters tabela encontrada", uf=uf or "BR", tipo=tipo, **df_brief(frame))
        if frames:
            out = pd.concat(frames, ignore_index=True)
            log("clusters carregados da ouro", uf=uf or "BR", duracao_segundos=round(time.perf_counter() - started, 3), **df_brief(out))
            return out
        paths: list[Path] = []
        for root in [self.cfg.banco / "ouro", self.cfg.banco / "global", self.cfg.banco / "preditivo_2026"]:
            if root.exists():
                log("clusters varrendo pasta", uf=uf or "BR", pasta=root)
                paths.extend([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".parquet", ".csv"} and "cluster" in p.name.lower()])
        rows: list[dict[str, Any]] = []
        for path in paths[:30]:
            rows.append({"arquivo": str(path), "tipo": "cluster", "uf": uf or "BR"})
        if rows:
            out = pd.DataFrame(rows)
            log("clusters encontrados", uf=uf or "BR", arquivos=len(rows), duracao_segundos=round(time.perf_counter() - started, 3))
            return out
        log("clusters nao encontrados; usando fallback de perfis", uf=uf or "BR")
        top = self.safe_call(lambda: to_df(self.store.top_profiles("estado" if uf else "brasil", uf=uf, limit=self.cfg.top_n)), f"cluster_fallback_{uf or 'br'}")
        if not top.empty:
            top = top.copy()
            top["tipo"] = "perfil_discreto_cluster_fallback"
            log("clusters fallback pronto", uf=uf or "BR", duracao_segundos=round(time.perf_counter() - started, 3), **df_brief(top))
            return top
        log("clusters vazio", uf=uf or "BR", duracao_segundos=round(time.perf_counter() - started, 3))
        return pd.DataFrame()

    def write_global_html(self, data: dict[str, Any], ufs: list[str]) -> None:
        started = time.perf_counter()
        log("html global inicio", ufs=len(ufs))
        self.include_plotlyjs = True if self.cfg.self_contained else "cdn"
        html = self.page_shell(
            title="Dashboard Brasil",
            subtitle="Analise global do Brasil a partir dos Parquets ja processados na camada ouro.",
            body=self.global_body(data, ufs),
            active="brasil",
        )
        path = self.paths.root / "index.html"
        path.write_text(html, encoding="utf-8")
        self.manifest["arquivos"].append(str(path))
        log("html global fim", arquivo=path, bytes=path.stat().st_size, duracao_segundos=round(time.perf_counter() - started, 3))

    def write_state_html(self, uf: str, data: dict[str, Any]) -> None:
        started = time.perf_counter()
        log("html estado inicio", uf=uf)
        self.include_plotlyjs = True if self.cfg.self_contained else "cdn"
        html = self.page_shell(
            title=f"Dashboard {uf}",
            subtitle=f"Analise estadual e municipal de {uf} com os dados ja existentes na camada ouro.",
            body=self.state_body(uf, data),
            active=uf,
        )
        path = self.paths.estados / f"{slug(uf)}.html"
        path.write_text(html, encoding="utf-8")
        self.manifest["arquivos"].append(str(path))
        log("html estado fim", uf=uf, arquivo=path, bytes=path.stat().st_size, duracao_segundos=round(time.perf_counter() - started, 3))

    def global_body(self, data: dict[str, Any], ufs: list[str]) -> str:
        log("html global corpo inicio")
        timeline = data_df(data, "timeline")
        parties = data_df(data, "partidos")
        profiles = data_df(data, "perfis")
        profile_dist = data_df(data, "perfil_discreto")
        state_map = data_df(data, "mapa_estados")
        timeline_uf = data_df(data, "timeline_uf")
        status = data.get("status") if isinstance(data.get("status"), dict) else {}
        cards = [
            ("UFs", len(ufs), "Estados detectados no banco"),
            ("Fatias resultado", f"{status.get('concluidas', 0)}/{status.get('total', 0)}", "Processamento ouro"),
            ("Eleitorado", format_num(safe_sum(timeline, ["eleitorado", "eleitorado_total"])), "Soma nacional disponivel"),
            ("Partidos", len(parties), "Ranking carregado"),
        ]
        links = "".join(f"<a class='state-link' href='estados/{slug(uf)}.html'>{escape(uf)}</a>" for uf in ufs)
        sections = [
            self.cards_html(cards),
            f"<section class='panel'><h2>Estados processados</h2><div class='state-links'>{links}</div></section>",
            self.chart_section(
                "Linha nacional",
                [
                    self.timeline_chart(timeline, "Evolucao nacional"),
                    self.metric_area_chart(timeline, "Comparecimento e abstencao"),
                ],
            ),
            self.chart_section(
                "Voto e partidos",
                [
                    self.party_bar(parties, "Ranking de partidos"),
                    self.party_pie(parties, "Distribuicao por partido"),
                ],
            ),
            self.chart_section(
                "Perfil do eleitor",
                [
                    self.profile_bar(profile_dist, "Perfil discreto"),
                    self.profile_treemap(profiles, "Top perfis Brasil"),
                ],
            ),
            self.chart_section(
                "Estados no mapa",
                [
                    self.brazil_map(state_map, "Partido dominante por UF"),
                    self.state_ranking_chart(timeline_uf, "Estados por volume eleitoral"),
                ],
            ),
            self.cluster_section(data_df(data, "clusters")),
        ]
        log("html global corpo fim", secoes=len(sections))
        return "\n".join(sections)

    def state_body(self, uf: str, data: dict[str, Any]) -> str:
        log("html estado corpo inicio", uf=uf)
        timeline = data_df(data, "timeline")
        parties = data_df(data, "partidos")
        profiles = data_df(data, "perfis")
        municipios = data_df(data, "municipios")
        municipios_partidos = data_df(data, "municipios_partidos")
        cards = [
            ("UF", uf, "Estado analisado"),
            ("Municipios", municipios["nm_municipio"].nunique() if "nm_municipio" in municipios else len(municipios), "Municipios carregados"),
            ("Eleitorado", format_num(safe_sum(municipios, ["eleitorado", "eleitorado_total"])), "Soma municipal disponivel"),
            ("Partidos", len(parties), "Ranking carregado"),
        ]
        sections = [
                "<section class='panel'><a class='back-link' href='../index.html'>Voltar ao Brasil</a></section>",
                self.cards_html(cards),
                self.chart_section(
                    f"Estado {uf}",
                    [
                        self.timeline_chart(timeline, f"Evolucao de {uf}"),
                        self.metric_area_chart(timeline, f"Comparecimento e abstencao - {uf}"),
                    ],
                ),
                self.chart_section(
                    "Partidos e voto",
                    [
                        self.party_bar(parties, f"{uf} por partido"),
                        self.party_pie(parties, f"Participacao partidaria - {uf}"),
                    ],
                ),
                self.chart_section(
                    "Municipios",
                    [
                        self.municipality_treemap(municipios, f"Municipios por peso eleitoral - {uf}"),
                        self.municipality_scatter(municipios, f"Distribuicao municipal - {uf}"),
                        self.municipality_party_heatmap(municipios_partidos, f"Partidos por municipio - {uf}"),
                    ],
                ),
                self.chart_section(
                    "Perfil do eleitor",
                    [
                        self.profile_treemap(profiles, f"Top perfis - {uf}"),
                        self.profile_cards_section(profiles),
                    ],
                ),
                self.cluster_section(data_df(data, "clusters")),
            ]
        log("html estado corpo fim", uf=uf, secoes=len(sections))
        return "\n".join(sections)

    def page_shell(self, title: str, subtitle: str, body: str, active: str) -> str:
        generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"""<!doctype html>
<html lang="pt-br">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>{self.css()}</style>
</head>
<body>
  <header class="hero">
    <div>
      <span class="eyebrow">Camada ouro + dashboards estaticos</span>
      <h1>{escape(title)}</h1>
      <p>{escape(subtitle)}</p>
      <div class="meta-row">
        <span>Banco: {escape(str(self.cfg.banco))}</span>
        <span>Gerado em: {escape(generated)}</span>
        <span>Pagina: {escape(active)}</span>
      </div>
    </div>
  </header>
  <main class="wrap">
    {body}
  </main>
</body>
</html>
"""

    def css(self) -> str:
        return """
:root{--bg:#f4f7fb;--paper:#fff;--ink:#0d172a;--muted:#617089;--line:#d9e2ef;--accent:#0f766e;--blue:#2563eb;--hot:#f97316;--dark:#0b1424}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 Inter,Segoe UI,Roboto,Arial,sans-serif}
.hero{background:radial-gradient(circle at 10% 20%,rgba(45,212,191,.35),transparent 32%),linear-gradient(135deg,#0b1424 0%,#10345c 48%,#0f766e 100%);color:white;padding:48px 5vw 40px}
.hero h1{font-size:clamp(34px,5vw,68px);line-height:.95;margin:10px 0 12px;letter-spacing:0;font-weight:850;max-width:1060px}.hero p{font-size:18px;color:#e7fbf7;max-width:980px;margin:0 0 18px}.eyebrow{color:#8cf5df;font-weight:800;text-transform:uppercase;letter-spacing:.08em;font-size:12px}.meta-row{display:flex;flex-wrap:wrap;gap:10px}.meta-row span,.state-link,.back-link{background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.18);border-radius:999px;padding:8px 12px;color:white;text-decoration:none;font-size:13px}
.wrap{max-width:1480px;margin:0 auto;padding:24px}.panel{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:20px;margin:0 0 20px;box-shadow:0 18px 50px rgba(15,23,42,.07)}.panel h2{font-size:22px;margin:0 0 14px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px;margin:0 0 20px}.card{background:white;border:1px solid var(--line);border-radius:12px;padding:18px;box-shadow:0 12px 32px rgba(15,23,42,.06)}.card .label{font-size:12px;color:var(--muted);font-weight:800;text-transform:uppercase}.card .value{font-size:30px;font-weight:850;margin-top:4px}.card .hint{font-size:13px;color:var(--muted)}
.chart-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:16px}.chart-tile{background:white;border:1px solid var(--line);border-radius:12px;padding:10px;min-height:360px;transition:transform .16s ease,box-shadow .16s ease}.chart-tile:hover{transform:scale(1.012);box-shadow:0 24px 56px rgba(15,23,42,.14);z-index:2}.chart-tile.empty{display:flex;align-items:center;justify-content:center;color:var(--muted);min-height:220px;text-align:center}
.state-links{display:flex;flex-wrap:wrap;gap:8px}.state-link{background:#e8f6f4;color:#0f766e;border-color:#bce5dd;font-weight:800}.back-link{display:inline-block;background:#10243d;color:#fff;border-color:#10243d}
.profile-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}.profile{border:1px solid var(--line);border-radius:12px;padding:14px;background:#fff}.profile b{display:block;margin-bottom:8px}.pill{display:inline-block;border-radius:999px;background:#eff6ff;color:#1d4ed8;padding:4px 8px;font-size:12px;margin:2px 4px 2px 0}.note{color:var(--muted);font-size:14px}
@media(max-width:760px){.wrap{padding:14px}.chart-grid{grid-template-columns:1fr}.hero{padding:34px 18px}.chart-tile{min-height:280px}}
"""

    def cards_html(self, cards: list[tuple[str, Any, str]]) -> str:
        return "<section class='cards'>" + "".join(
            f"<div class='card'><div class='label'>{escape(str(label))}</div><div class='value'>{escape(str(value))}</div><div class='hint'>{escape(str(hint))}</div></div>"
            for label, value, hint in cards
        ) + "</section>"

    def chart_section(self, title: str, charts: list[str]) -> str:
        log("secao visual montada", titulo=title, graficos=len(charts))
        return f"<section class='panel'><h2>{escape(title)}</h2><div class='chart-grid'>{''.join(charts)}</div></section>"

    def chart_tile(self, fig: Any | None, empty: str = "Dados ainda nao disponiveis nesta camada.") -> str:
        if fig is None:
            log("grafico vazio", motivo=empty)
            return f"<div class='chart-tile empty'>{escape(empty)}</div>"
        title = ""
        try:
            title = str(fig.layout.title.text or "")
        except Exception:
            title = ""
        log("grafico renderizando", titulo=title, traces=len(getattr(fig, "data", []) or []), include_plotlyjs=self.include_plotlyjs)
        fig.update_layout(template="plotly_white", margin=dict(l=28, r=18, t=54, b=30), font=dict(family="Inter, Segoe UI, Arial"))
        html = pio.to_html(fig, include_plotlyjs=self.include_plotlyjs, full_html=False, config={"displayModeBar": True, "responsive": True})
        self.include_plotlyjs = False
        log("grafico renderizado", titulo=title, html_bytes=len(html))
        return f"<div class='chart-tile'>{html}</div>"

    def log_chart_input(self, tipo: str, title: str, df: pd.DataFrame) -> None:
        log("grafico preparando", tipo=tipo, titulo=title, **df_brief(df))

    def timeline_chart(self, df: pd.DataFrame, title: str) -> str:
        self.log_chart_input("linha", title, df)
        if df.empty or "ano" not in df.columns:
            return self.chart_tile(None)
        metrics = [col for col in ["eleitorado", "eleitorado_total", "votos", "votos_total", "comparecimento_estimado", "abstencao_estimado"] if col in df.columns]
        if not metrics:
            return self.chart_tile(None)
        work = df.copy()
        for col in metrics:
            work[col] = as_number(work[col])
        fig = px.line(work.sort_values("ano"), x="ano", y=metrics[:4], markers=True, title=title)
        return self.chart_tile(fig)

    def metric_area_chart(self, df: pd.DataFrame, title: str) -> str:
        self.log_chart_input("area", title, df)
        cols = [col for col in ["comparecimento_estimado", "abstencao_estimado"] if col in df.columns]
        if df.empty or "ano" not in df.columns or not cols:
            return self.chart_tile(None)
        work = df.copy()
        for col in cols:
            work[col] = as_number(work[col])
        fig = px.area(work.sort_values("ano"), x="ano", y=cols, title=title)
        return self.chart_tile(fig)

    def party_bar(self, df: pd.DataFrame, title: str) -> str:
        self.log_chart_input("barra_partido", title, df)
        if df.empty or "partido" not in df.columns:
            return self.chart_tile(None)
        value_col = first_existing(list(df.columns), ["share_pred_2026", "votos_pred_2026", "votos"])
        if not value_col:
            return self.chart_tile(None)
        work = df.copy()
        work[value_col] = as_number(work[value_col])
        work = work.sort_values(value_col, ascending=False).head(self.cfg.top_n)
        fig = px.bar(work.sort_values(value_col), x=value_col, y="partido", orientation="h", color="partido", title=title, text=value_col)
        if value_col.startswith("share"):
            fig.update_traces(texttemplate="%{text:.1%}")
            fig.update_xaxes(tickformat=".0%")
        return self.chart_tile(fig)

    def party_pie(self, df: pd.DataFrame, title: str) -> str:
        self.log_chart_input("pizza_partido", title, df)
        if df.empty or "partido" not in df.columns:
            return self.chart_tile(None)
        value_col = first_existing(list(df.columns), ["share_pred_2026", "votos_pred_2026", "votos"])
        if not value_col:
            return self.chart_tile(None)
        work = df.copy()
        work[value_col] = as_number(work[value_col])
        work = work.sort_values(value_col, ascending=False).head(min(self.cfg.top_n, 12))
        fig = px.pie(work, names="partido", values=value_col, title=title, hole=.44)
        return self.chart_tile(fig)

    def profile_bar(self, df: pd.DataFrame, title: str) -> str:
        self.log_chart_input("barra_perfil", title, df)
        if df.empty or not {"dimensao_perfil", "valor_perfil"}.issubset(df.columns):
            return self.chart_tile(None)
        value_col = first_existing(list(df.columns), ["peso", "eleitorado", "share"])
        if not value_col:
            return self.chart_tile(None)
        work = df.copy()
        work[value_col] = as_number(work[value_col])
        work = work.sort_values(value_col, ascending=False).head(self.cfg.top_n * 2)
        fig = px.bar(work.sort_values(value_col), x=value_col, y="valor_perfil", color="dimensao_perfil", orientation="h", title=title)
        return self.chart_tile(fig)

    def profile_treemap(self, df: pd.DataFrame, title: str) -> str:
        self.log_chart_input("treemap_perfil", title, df)
        if df.empty:
            return self.chart_tile(None)
        label_col = first_existing(list(df.columns), ["perfil_combinado", "descricao", "valor_perfil"])
        value_col = first_existing(list(df.columns), ["share_perfil", "eleitorado", "peso"])
        if not label_col:
            return self.chart_tile(None)
        work = df.copy().head(self.cfg.top_n)
        if value_col:
            work[value_col] = as_number(work[value_col]).clip(lower=1e-9)
        else:
            value_col = "_peso"
            work[value_col] = 1
        fig = px.treemap(work, path=[label_col], values=value_col, title=title)
        return self.chart_tile(fig)

    def brazil_map(self, df: pd.DataFrame, title: str) -> str:
        self.log_chart_input("mapa_brasil", title, df)
        if df.empty or "uf" not in df.columns:
            return self.chart_tile(None)
        work = df.copy()
        work["uf"] = work["uf"].astype(str).str.upper()
        work["lat"] = work["uf"].map(lambda uf: UF_CENTROIDS.get(uf, (None, None))[0])
        work["lon"] = work["uf"].map(lambda uf: UF_CENTROIDS.get(uf, (None, None))[1])
        work = work.dropna(subset=["lat", "lon"])
        if work.empty:
            return self.chart_tile(None)
        if "share_pred_2026" in work.columns:
            work["share_pred_2026"] = as_number(work["share_pred_2026"])
        else:
            work["share_pred_2026"] = 0.2
        work["tamanho"] = (work["share_pred_2026"] * 100).clip(lower=8)
        fig = px.scatter_geo(
            work,
            lat="lat",
            lon="lon",
            size="tamanho",
            color="partido" if "partido" in work.columns else None,
            hover_name="uf",
            hover_data=[c for c in ["partido", "share_pred_2026", "votos_pred_2026", "fonte"] if c in work.columns],
            title=title,
            projection="mercator",
        )
        fig.update_geos(lataxis_range=[-35, 6], lonaxis_range=[-75, -32], showland=True, landcolor="#eef3f7", showocean=True, oceancolor="#eaf6ff")
        fig.update_layout(height=560)
        return self.chart_tile(fig)

    def state_ranking_chart(self, df: pd.DataFrame, title: str) -> str:
        self.log_chart_input("ranking_estado", title, df)
        if df.empty or "uf" not in df.columns:
            return self.chart_tile(None)
        value_col = first_existing(list(df.columns), ["eleitorado", "eleitorado_total", "votos_total", "votos", "comparecimento_estimado"])
        if not value_col:
            return self.chart_tile(None)
        work = df.copy()
        work[value_col] = as_number(work[value_col])
        work = work.sort_values(value_col, ascending=False).head(30)
        fig = px.bar(work, x="uf", y=value_col, color="uf", title=title)
        return self.chart_tile(fig)

    def municipality_treemap(self, df: pd.DataFrame, title: str) -> str:
        self.log_chart_input("treemap_municipio", title, df)
        if df.empty or "nm_municipio" not in df.columns:
            return self.chart_tile(None)
        value_col = first_existing(list(df.columns), ["eleitorado", "eleitorado_total", "votos_total", "votos", "comparecimento_estimado"])
        if not value_col:
            return self.chart_tile(None)
        work = df.copy()
        work[value_col] = as_number(work[value_col]).clip(lower=1e-9)
        work = work.sort_values(value_col, ascending=False).head(self.cfg.max_municipios_por_estado)
        fig = px.treemap(work, path=["nm_municipio"], values=value_col, title=title)
        return self.chart_tile(fig)

    def municipality_scatter(self, df: pd.DataFrame, title: str) -> str:
        self.log_chart_input("dispersao_municipio", title, df)
        if df.empty or "nm_municipio" not in df.columns:
            return self.chart_tile(None)
        x_col = first_existing(list(df.columns), ["comparecimento_estimado", "comparecimento_medio", "votos_total", "votos"])
        y_col = first_existing(list(df.columns), ["abstencao_estimado", "abstencao_media", "eleitorado", "eleitorado_total"])
        size_col = first_existing(list(df.columns), ["eleitorado", "eleitorado_total", "votos_total", "votos"])
        if not x_col or not y_col:
            return self.chart_tile(None)
        work = df.copy()
        for col in {x_col, y_col, size_col} - {None}:
            work[col] = as_number(work[col])
        fig = px.scatter(work, x=x_col, y=y_col, size=size_col, hover_name="nm_municipio", color="ano" if "ano" in work.columns else None, title=title)
        return self.chart_tile(fig)

    def municipality_party_heatmap(self, df: pd.DataFrame, title: str) -> str:
        self.log_chart_input("heatmap_municipio_partido", title, df)
        if df.empty or not {"nm_municipio", "partido", "votos"}.issubset(df.columns):
            return self.chart_tile(None)
        work = df.copy()
        work["votos"] = as_number(work["votos"])
        top_muns = work.groupby("nm_municipio")["votos"].sum().sort_values(ascending=False).head(24).index
        top_parties = work.groupby("partido")["votos"].sum().sort_values(ascending=False).head(12).index
        work = work[work["nm_municipio"].isin(top_muns) & work["partido"].isin(top_parties)]
        pivot = work.pivot_table(index="nm_municipio", columns="partido", values="votos", aggfunc="sum", fill_value=0)
        if pivot.empty:
            return self.chart_tile(None)
        fig = px.imshow(pivot, aspect="auto", title=title, labels=dict(x="Partido", y="Municipio", color="Votos"))
        return self.chart_tile(fig)

    def profile_cards_section(self, df: pd.DataFrame) -> str:
        log("cards perfil preparando", **df_brief(df))
        if df.empty:
            return "<div class='chart-tile empty'>Perfis ainda nao disponiveis.</div>"
        cards = []
        for _, row in df.head(self.cfg.top_n).iterrows():
            profile = row.get("perfil_combinado") or row.get("descricao") or row.get("valor_perfil") or "Perfil"
            pills = []
            for col in ["ano", "uf", "nm_municipio", "share_perfil", "eleitorado", "tipo"]:
                if col in row and pd.notna(row[col]) and str(row[col]).strip():
                    pills.append(f"<span class='pill'>{escape(col)}: {escape(format_num(row[col]))}</span>")
            cards.append(f"<div class='profile'><b>{escape(str(profile))}</b>{''.join(pills)}</div>")
        return f"<div class='chart-tile'><div class='profile-grid'>{''.join(cards)}</div></div>"

    def cluster_section(self, df: pd.DataFrame) -> str:
        log("secao clusters preparando", **df_brief(df))
        if df.empty:
            content = "<div class='chart-tile empty'>Clusters ainda nao encontrados na camada ouro; quando existirem, entram aqui automaticamente.</div>"
        elif "perfil_combinado" in df.columns:
            content = self.profile_cards_section(df)
        else:
            cards = []
            for _, row in df.head(self.cfg.top_n).iterrows():
                arquivo = row.get("arquivo", "cluster")
                cards.append(f"<div class='profile'><b>{escape(str(row.get('tipo', 'cluster')))}</b><span class='pill'>{escape(str(arquivo))}</span></div>")
            content = f"<div class='chart-tile'><div class='profile-grid'>{''.join(cards)}</div></div>"
        return f"<section class='panel'><h2>Clusters e perfis discriminados</h2><div class='chart-grid'>{content}</div></section>"

    def write_global_pdf(self, data: dict[str, Any], ufs: list[str]) -> None:
        started = time.perf_counter()
        if canvas is None:
            log("pdf global ignorado", motivo="reportlab_nao_instalado")
            return
        path = self.paths.pdf / "brasil.pdf"
        log("pdf global inicio", arquivo=path)
        pdf = SimplePdf(path, "Dashboard Brasil")
        timeline = data_df(data, "timeline")
        parties = data_df(data, "partidos")
        profiles = data_df(data, "perfis")
        status = data.get("status") if isinstance(data.get("status"), dict) else {}
        pdf.cover("Dashboard Brasil", "Camada ouro eleitoral")
        pdf.cards([("UFs", len(ufs)), ("Resultados", f"{status.get('concluidas', 0)}/{status.get('total', 0)}"), ("Partidos", len(parties))])
        pdf.hbar("Partidos", parties, "partido", first_existing(list(parties.columns), ["share_pred_2026", "votos_pred_2026", "votos"]), self.cfg.top_n)
        pdf.hbar("Timeline nacional", timeline, "ano", first_existing(list(timeline.columns), ["eleitorado", "eleitorado_total", "votos_total", "votos"]), self.cfg.top_n)
        pdf.profile_cards("Perfis Brasil", profiles, self.cfg.top_n)
        pdf.close()
        self.manifest["arquivos"].append(str(path))
        log("pdf global fim", arquivo=path, bytes=path.stat().st_size, duracao_segundos=round(time.perf_counter() - started, 3))

    def write_state_pdf(self, uf: str, data: dict[str, Any]) -> None:
        started = time.perf_counter()
        if canvas is None:
            log("pdf estado ignorado", uf=uf, motivo="reportlab_nao_instalado")
            return
        path = self.paths.pdf / f"estado_{slug(uf)}.pdf"
        log("pdf estado inicio", uf=uf, arquivo=path)
        pdf = SimplePdf(path, f"Dashboard {uf}")
        timeline = data_df(data, "timeline")
        parties = data_df(data, "partidos")
        profiles = data_df(data, "perfis")
        municipios = data_df(data, "municipios")
        pdf.cover(f"Dashboard {uf}", "Estado e municipios")
        pdf.cards([("Municipios", municipios["nm_municipio"].nunique() if "nm_municipio" in municipios else len(municipios)), ("Partidos", len(parties)), ("Eleitorado", format_num(safe_sum(municipios, ["eleitorado", "eleitorado_total"])))])
        pdf.hbar(f"Partidos {uf}", parties, "partido", first_existing(list(parties.columns), ["share_pred_2026", "votos_pred_2026", "votos"]), self.cfg.top_n)
        pdf.hbar(f"Municipios {uf}", municipios, "nm_municipio", first_existing(list(municipios.columns), ["eleitorado", "eleitorado_total", "votos_total", "votos"]), self.cfg.top_n)
        pdf.hbar(f"Timeline {uf}", timeline, "ano", first_existing(list(timeline.columns), ["eleitorado", "eleitorado_total", "votos_total", "votos"]), self.cfg.top_n)
        pdf.profile_cards(f"Perfis {uf}", profiles, self.cfg.top_n)
        pdf.close()
        self.manifest["arquivos"].append(str(path))
        log("pdf estado fim", uf=uf, arquivo=path, bytes=path.stat().st_size, duracao_segundos=round(time.perf_counter() - started, 3))


class SimplePdf:
    def __init__(self, path: Path, title: str):
        log("pdf arquivo criando", arquivo=path, titulo=title)
        self.path = path
        self.c = canvas.Canvas(str(path), pagesize=landscape(A4))  # type: ignore[union-attr,operator]
        self.width, self.height = landscape(A4)  # type: ignore[operator]
        self.title = title
        self.margin = 42
        self.y = self.height - self.margin

    def cover(self, title: str, subtitle: str) -> None:
        log("pdf capa", arquivo=self.path, titulo=title)
        self.c.setFillColor(colors.HexColor("#0b1424"))  # type: ignore[union-attr]
        self.c.rect(0, 0, self.width, self.height, fill=1, stroke=0)
        self.c.setFillColor(colors.white)  # type: ignore[union-attr]
        self.c.setFont("Helvetica-Bold", 30)
        self.c.drawString(self.margin, self.height - 150, title)
        self.c.setFont("Helvetica", 14)
        self.c.drawString(self.margin, self.height - 178, subtitle)
        self.c.drawString(self.margin, self.height - 204, f"Gerado em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.c.showPage()
        self.y = self.height - self.margin

    def ensure_space(self, amount: float) -> None:
        if self.y - amount < self.margin:
            log("pdf nova pagina", arquivo=self.path)
            self.c.showPage()
            self.y = self.height - self.margin

    def heading(self, text: str) -> None:
        log("pdf titulo secao", arquivo=self.path, titulo=text)
        self.ensure_space(42)
        self.c.setFillColor(colors.HexColor("#0d172a"))  # type: ignore[union-attr]
        self.c.setFont("Helvetica-Bold", 18)
        self.c.drawString(self.margin, self.y, text[:100])
        self.y -= 30

    def cards(self, items: list[tuple[str, Any]]) -> None:
        log("pdf cards", arquivo=self.path, quantidade=len(items))
        self.ensure_space(92)
        card_w = (self.width - self.margin * 2 - 24) / max(1, len(items))
        for idx, (label, value) in enumerate(items):
            x = self.margin + idx * (card_w + 12)
            self.c.setFillColor(colors.HexColor("#f8fafc"))  # type: ignore[union-attr]
            self.c.setStrokeColor(colors.HexColor("#d9e2ef"))  # type: ignore[union-attr]
            self.c.roundRect(x, self.y - 74, card_w, 70, 8, fill=1, stroke=1)
            self.c.setFillColor(colors.HexColor("#617089"))  # type: ignore[union-attr]
            self.c.setFont("Helvetica-Bold", 8)
            self.c.drawString(x + 12, self.y - 24, str(label).upper()[:28])
            self.c.setFillColor(colors.HexColor("#0d172a"))  # type: ignore[union-attr]
            self.c.setFont("Helvetica-Bold", 18)
            self.c.drawString(x + 12, self.y - 52, str(value)[:28])
        self.y -= 92

    def hbar(self, title: str, df: pd.DataFrame, label_col: str | None, value_col: str | None, limit: int) -> None:
        if df.empty or not label_col or not value_col or label_col not in df.columns or value_col not in df.columns:
            log("pdf barra ignorada", arquivo=self.path, titulo=title, linhas=len(df), label=label_col or "", valor=value_col or "")
            return
        log("pdf barra inicio", arquivo=self.path, titulo=title, linhas=len(df), label=label_col, valor=value_col, limite=limit)
        self.heading(title)
        work = df.copy()
        work[value_col] = as_number(work[value_col])
        work = work.sort_values(value_col, ascending=False).head(limit)
        max_value = float(work[value_col].max() or 1)
        bar_w = self.width - self.margin * 2 - 220
        for _, row in work.iterrows():
            self.ensure_space(24)
            label = str(row[label_col])[:28]
            value = float(row[value_col] or 0)
            width = 0 if max_value <= 0 else bar_w * value / max_value
            self.c.setFillColor(colors.HexColor("#0d172a"))  # type: ignore[union-attr]
            self.c.setFont("Helvetica", 8)
            self.c.drawString(self.margin, self.y - 9, label)
            self.c.setFillColor(colors.HexColor("#0f766e"))  # type: ignore[union-attr]
            self.c.rect(self.margin + 160, self.y - 15, width, 12, fill=1, stroke=0)
            self.c.setFillColor(colors.HexColor("#617089"))  # type: ignore[union-attr]
            self.c.drawString(self.margin + 166 + width, self.y - 9, format_num(value))
            self.y -= 20
        self.y -= 10
        log("pdf barra fim", arquivo=self.path, titulo=title, linhas_desenhadas=len(work))

    def profile_cards(self, title: str, df: pd.DataFrame, limit: int) -> None:
        if df.empty:
            log("pdf perfis ignorado", arquivo=self.path, titulo=title, motivo="sem_linhas")
            return
        self.heading(title)
        label_col = first_existing(list(df.columns), ["perfil_combinado", "descricao", "valor_perfil"])
        if not label_col:
            log("pdf perfis ignorado", arquivo=self.path, titulo=title, motivo="sem_coluna_label")
            return
        log("pdf perfis inicio", arquivo=self.path, titulo=title, linhas=len(df), label=label_col, limite=limit)
        for _, row in df.head(limit).iterrows():
            self.ensure_space(30)
            self.c.setFillColor(colors.HexColor("#0d172a"))  # type: ignore[union-attr]
            self.c.setFont("Helvetica", 8)
            self.c.drawString(self.margin, self.y, str(row.get(label_col, ""))[:135])
            self.y -= 18
        log("pdf perfis fim", arquivo=self.path, titulo=title, linhas_desenhadas=min(len(df), limit))

    def close(self) -> None:
        log("pdf salvando", arquivo=self.path)
        self.c.save()
        log("pdf salvo", arquivo=self.path)


def first_existing(cols: list[str], candidates: list[str]) -> str | None:
    for col in candidates:
        if col in cols:
            return col
    return None


def data_df(data: dict[str, Any], key: str) -> pd.DataFrame:
    value = data.get(key)
    return value if isinstance(value, pd.DataFrame) else pd.DataFrame()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera dashboards HTML estaticos e PDFs a partir da camada ouro ja processada.")
    parser.add_argument("banco", nargs="?", default="dados/banco_eleitoral", help="Pasta do banco eleitoral com a camada ouro.")
    parser.add_argument("--out", default="", help="Pasta de saida. Padrao: resultados/dashboards_ouro_<timestamp>.")
    parser.add_argument("--top-n", type=int, default=20, help="Quantidade de itens por ranking/grafico.")
    parser.add_argument("--max-municipios-por-estado", type=int, default=350, help="Maximo de municipios agregados em cada HTML estadual.")
    parser.add_argument("--ufs", default="", help="Lista de UFs separadas por virgula. Vazio gera todas detectadas.")
    parser.add_argument("--ano", default="", help="Ano preferencial para alguns graficos, ex: 2024.")
    parser.add_argument("--cenario", default="base", help="Cenario da simulacao 2026 quando existir.")
    parser.add_argument("--sem-pdf", action="store_true", help="Gera apenas HTML.")
    parser.add_argument("--self-contained", action="store_true", help="Embute plotly.js em cada HTML; arquivos ficam maiores.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    banco = resolve_path(args.banco)
    if not banco.exists():
        raise SystemExit(f"Banco nao encontrado: {banco}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = resolve_path(args.out) if args.out else resolve_path(f"resultados/dashboards_ouro_{timestamp}")
    cfg = GeneratorConfig(
        banco=banco,
        out=out,
        top_n=max(3, int(args.top_n)),
        max_municipios_por_estado=max(10, int(args.max_municipios_por_estado)),
        ufs=[uf.strip().upper() for uf in args.ufs.split(",") if uf.strip()],
        ano=str(args.ano or ""),
        cenario=str(args.cenario or "base"),
        gerar_pdf=not bool(args.sem_pdf),
        self_contained=bool(args.self_contained),
    )
    generator = StaticDashboardGenerator(cfg)
    root = generator.run()
    print("\n" + "=" * 100)
    print(f"Dashboards gerados em: {root}")
    print(f"Abra: {root / 'index.html'}")
    print(f"PDFs: {root / 'pdf'}")
    print("=" * 100)


if __name__ == "__main__":
    main()
