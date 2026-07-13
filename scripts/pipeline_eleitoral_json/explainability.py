from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import extract_years_from_value, safe_text, save_csv


def read_csv_if_exists(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or p.stat().st_size <= 5:
        return pd.DataFrame()
    try:
        return pd.read_csv(p, sep=";", dtype=str)
    except Exception:
        return pd.DataFrame()


def load_global_context(global_info: dict[str, Any]) -> dict[str, pd.DataFrame]:
    keys = [
        "inventario_temporal_csv",
        "matriz_arquivo_ano_csv",
        "timeline_nacional_csv",
        "timeline_uf_csv",
        "timeline_municipal_csv",
        "timeline_entidades_csv",
        "evolucao_municipal_csv",
        "correlacoes_temporais_csv",
        "correlacoes_entidades_csv",
        "similaridade_tabelas_csv",
        "similaridade_campos_csv",
        "mapa_canonico_csv",
    ]
    return {k.replace("_csv", ""): read_csv_if_exists(global_info.get(k, "")) for k in keys}


def traceability_from_gold(global_gold: pd.DataFrame) -> pd.DataFrame:
    if global_gold is None or global_gold.empty:
        return pd.DataFrame()

    df = global_gold.copy()
    if "arquivo_origem" not in df.columns:
        df["arquivo_origem"] = "SEM_ARQUIVO_ORIGEM"

    for c in ["ano", "uf", "cd_municipio", "cargo", "turno"]:
        if c not in df.columns:
            df[c] = ""

    for c in ["votos", "eleitorado", "comparecimento_estimado", "abstencao_estimado"]:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    rows = []
    for file, g in df.groupby("arquivo_origem", dropna=False):
        years = sorted(pd.to_numeric(g["ano"], errors="coerce").dropna().astype(int).unique().tolist())
        rows.append({
            "arquivo_origem": file,
            "anos_usados": ", ".join(map(str, years)),
            "ano_min": min(years) if years else np.nan,
            "ano_max": max(years) if years else np.nan,
            "ufs": int(g["uf"].replace("", np.nan).dropna().nunique()),
            "municipios": int(g["cd_municipio"].replace("", np.nan).dropna().nunique()),
            "cargos": ", ".join(sorted([safe_text(x) for x in g["cargo"].dropna().unique() if safe_text(x)])[:40]),
            "linhas_gold": int(len(g)),
            "votos_soma": float(g["votos"].sum()),
            "uso_na_simulacao": "alimenta base_modelagem, swing histórico, timeline e Monte Carlo",
        })

    out = pd.DataFrame(rows)
    return out.sort_values(["ano_min", "votos_soma"], ascending=[True, False], na_position="last") if not out.empty else out


def evidence_summary(context: dict[str, pd.DataFrame], backtest: pd.DataFrame, model_df: pd.DataFrame, electoral_outputs: dict[str, str] | None = None) -> pd.DataFrame:
    rows = []

    inventory = context.get("inventario_temporal", pd.DataFrame())
    if inventory is not None and not inventory.empty:
        years = sorted(set(
            y
            for col in ["anos_detectados_conteudo", "anos_detectados_nome"]
            for value in inventory.get(col, pd.Series(dtype=str)).fillna("").astype(str)
            for y in extract_years_from_value(value)
        ))
        rows.append({
            "tipo": "cobertura_temporal",
            "evidencia": "anos_detectados",
            "valor": ", ".join(map(str, years)),
            "impacto_na_simulacao": "A simulação usa os anos efetivamente detectados nos JSONs; não há ano fixo no código.",
        })

    corr = context.get("correlacoes_temporais", pd.DataFrame())
    if corr is not None and not corr.empty:
        for _, r in corr.head(20).iterrows():
            rows.append({
                "tipo": "correlacao_temporal",
                "evidencia": f"{r.get('metrica','')} {r.get('ano_1','')}→{r.get('ano_2','')}",
                "valor": r.get("pearson", ""),
                "impacto_na_simulacao": "Correlação maior indica estabilidade histórica e reduz a cautela; correlação baixa aumenta incerteza.",
            })

    entities = context.get("correlacoes_entidades", pd.DataFrame())
    if entities is not None and not entities.empty:
        for _, r in entities.head(15).iterrows():
            rows.append({
                "tipo": "correlacao_entidades",
                "evidencia": f"{r.get('ano_1','')}→{r.get('ano_2','')}",
                "valor": r.get("pearson_share", ""),
                "impacto_na_simulacao": "Mede persistência de força das entidades entre anos.",
            })

    sim_tables = context.get("similaridade_tabelas", pd.DataFrame())
    if sim_tables is not None and not sim_tables.empty:
        for _, r in sim_tables.head(10).iterrows():
            rows.append({
                "tipo": "similaridade_tabelas",
                "evidencia": f"{Path(str(r.get('arquivo_1',''))).name} ↔ {Path(str(r.get('arquivo_2',''))).name}",
                "valor": r.get("score_similaridade", ""),
                "impacto_na_simulacao": "Tabelas estruturalmente próximas tornam comparações históricas mais defensáveis.",
            })

    field_map = context.get("mapa_canonico", pd.DataFrame())
    if field_map is not None and not field_map.empty:
        rows.append({
            "tipo": "mapa_canonico",
            "evidencia": "campos_aprendidos",
            "valor": f"{len(field_map)} grupos canônicos",
            "impacto_na_simulacao": "Reduz dependência de nome exato de coluna; usa padrões observados nos JSONs.",
        })

    if model_df is not None and not model_df.empty and "ano_num" in model_df.columns:
        years = sorted(pd.to_numeric(model_df["ano_num"], errors="coerce").dropna().astype(int).unique().tolist())
        rows.append({
            "tipo": "base_modelagem",
            "evidencia": "anos_efetivamente_usados",
            "valor": ", ".join(map(str, years)),
            "impacto_na_simulacao": "Esses são os anos que entraram no cálculo de share, swing e Monte Carlo.",
        })

    if electoral_outputs:
        for key, path in electoral_outputs.items():
            rows.append({
                "tipo": "analise_eleitoral",
                "evidencia": key,
                "valor": path,
                "impacto_na_simulacao": "Esta análise eleitoral global é usada para explicar contexto: perfil do eleitorado, seção, tendência, abstenção, cristalização, transferência proxy ou voto útil proxy.",
            })

    if backtest is not None and not backtest.empty and "mae_share" in backtest.columns:
        vals = pd.to_numeric(backtest["mae_share"], errors="coerce").dropna()
        if not vals.empty:
            rows.append({
                "tipo": "backtesting",
                "evidencia": "mae_share_medio",
                "valor": float(vals.mean()),
                "impacto_na_simulacao": "Erro histórico ajuda a interpretar a confiança da simulação.",
            })

    return pd.DataFrame(rows)


def entity_contribution(model_df: pd.DataFrame, scenarios: pd.DataFrame, mc: pd.DataFrame) -> pd.DataFrame:
    if model_df is None or model_df.empty or scenarios is None or scenarios.empty:
        return pd.DataFrame()

    df = model_df.copy()
    df["ano_num"] = pd.to_numeric(df["ano_num"], errors="coerce")
    base_year = int(df["ano_num"].max()) if df["ano_num"].notna().any() else None
    base = df.loc[df["ano_num"].eq(base_year)].copy() if base_year is not None else df.copy()

    for c in ["votos", "share", "eleitorado"]:
        base[c] = pd.to_numeric(base.get(c, 0), errors="coerce").fillna(0)

    b = base.groupby(["cargo", "turno", "entidade"], dropna=False).agg({
        "votos": "sum",
        "share": "mean",
        "eleitorado": "sum",
    }).reset_index().rename(columns={
        "votos": "votos_base",
        "share": "share_medio_base",
        "eleitorado": "eleitorado_base",
    })
    b["ano_base"] = base_year

    pred = scenarios.loc[scenarios["cenario"].astype(str).eq("base")].copy()
    if pred.empty:
        pred = scenarios.copy()

    for c in ["votos_pred_2026", "share_pred_2026", "swing_anual_estimado", "sigma_estimado"]:
        pred[c] = pd.to_numeric(pred.get(c, 0), errors="coerce").fillna(0)

    p = pred.groupby(["cargo", "turno", "entidade"], dropna=False).agg({
        "votos_pred_2026": "sum",
        "share_pred_2026": "mean",
        "swing_anual_estimado": "mean",
        "sigma_estimado": "mean",
    }).reset_index()

    out = b.merge(p, on=["cargo", "turno", "entidade"], how="outer")
    out["delta_share_medio"] = pd.to_numeric(out["share_pred_2026"], errors="coerce") - pd.to_numeric(out["share_medio_base"], errors="coerce")
    out["explicacao"] = out.apply(
        lambda r: (
            f"Parte de share médio {float(r.get('share_medio_base', 0) or 0):.4f} no ano-base {safe_text(r.get('ano_base', ''))}; "
            f"aplica swing anual {float(r.get('swing_anual_estimado', 0) or 0):+.4f}; "
            f"chega a share médio {float(r.get('share_pred_2026', 0) or 0):.4f}; "
            f"sigma usado {float(r.get('sigma_estimado', 0) or 0):.4f}."
        ),
        axis=1,
    )

    if mc is not None and not mc.empty:
        keep = [c for c in ["cargo", "turno", "entidade", "share_medio", "share_p05", "share_p50", "share_p95", "share_desvio"] if c in mc.columns]
        mcb = mc.loc[mc["cenario"].astype(str).eq("base"), keep].copy() if "cenario" in mc.columns else mc[keep].copy()
        out = out.merge(mcb, on=["cargo", "turno", "entidade"], how="left")

    return out.sort_values("votos_pred_2026", ascending=False) if "votos_pred_2026" in out.columns else out


def scenario_justification(scenarios: pd.DataFrame) -> pd.DataFrame:
    if scenarios is None or scenarios.empty:
        return pd.DataFrame()

    premises = {
        "base": "tendência histórica ponderada sem choque externo forte",
        "continuidade_historica": "menor peso para mudança brusca; maior estabilidade",
        "impulso_territorial": "maior peso para swing territorial observado",
        "alta_abstencao": "reduz comparecimento esperado",
        "baixa_abstencao": "aumenta comparecimento esperado",
        "maior_volatilidade": "aumenta ruído e dispersão do Monte Carlo",
    }

    rows = []
    for scenario, g in scenarios.groupby("cenario", dropna=False):
        s = safe_text(scenario, "SEM_CENARIO")
        swing = pd.to_numeric(g.get("swing_anual_estimado", pd.Series(dtype=float)), errors="coerce").mean()
        sigma = pd.to_numeric(g.get("sigma_estimado", pd.Series(dtype=float)), errors="coerce").mean()
        votes = pd.to_numeric(g.get("votos_pred_2026", pd.Series(dtype=float)), errors="coerce").sum()
        rows.append({
            "cenario": s,
            "premissa": premises.get(s, "cenário parametrizado pelo modelo"),
            "swing_medio": swing,
            "sigma_medio": sigma,
            "votos_preditos": votes,
            "porque_chegou_nisso": (
                f"O cenário '{s}' usa a premissa '{premises.get(s, 'cenário parametrizado')}', "
                f"com swing médio {swing:+.4f} e sigma médio {sigma:.4f}."
            ),
        })

    return pd.DataFrame(rows)


def markdown_explanation(
    trace: pd.DataFrame,
    evidence: pd.DataFrame,
    contribution: pd.DataFrame,
    justification: pd.DataFrame,
    nacional: pd.DataFrame,
) -> str:
    lines = []
    lines.append("# Explicação detalhada da simulação")
    lines.append("")
    lines.append("A simulação não é executada isoladamente. Ela parte dos resultados individuais, da análise global, da timeline, das correlações entre anos e da estabilidade observada no próprio dado.")
    lines.append("O código não assume anos fixos. Ele usa os anos detectados nos JSONs.")

    if trace is not None and not trace.empty:
        lines.append("\n## Arquivos individuais que alimentaram a simulação")
        for _, r in trace.head(30).iterrows():
            lines.append(f"- `{r.get('arquivo_origem','')}`: anos `{r.get('anos_usados','')}`, municípios `{r.get('municipios','')}`, votos `{r.get('votos_soma','')}`.")

    if evidence is not None and not evidence.empty:
        lines.append("\n## Evidências globais usadas")
        for _, r in evidence.head(60).iterrows():
            lines.append(f"- **{r.get('tipo','')} / {r.get('evidencia','')}**: {r.get('valor','')} — {r.get('impacto_na_simulacao','')}")

    if nacional is not None and not nacional.empty:
        lines.append("\n## Resultado nacional no cenário base")
        base = nacional.loc[nacional["cenario"].astype(str).eq("base")].copy()
        if base.empty:
            base = nacional.copy()
        for _, r in base.sort_values("share_nacional_pred_2026", ascending=False).head(10).iterrows():
            try:
                lines.append(f"- `{r.get('entidade','SEM_ENTIDADE')}`: {float(r.get('share_nacional_pred_2026', 0))*100:.2f}%")
            except Exception:
                pass

    if contribution is not None and not contribution.empty:
        lines.append("\n## Por que cada entidade chegou naquele resultado")
        for _, r in contribution.head(20).iterrows():
            lines.append(f"- `{r.get('entidade','SEM_ENTIDADE')}`: {r.get('explicacao','')}")

    if justification is not None and not justification.empty:
        lines.append("\n## Justificativa dos cenários")
        for _, r in justification.iterrows():
            lines.append(f"- **{r.get('cenario','')}**: {r.get('porque_chegou_nisso','')}")

    lines.append("\n## Leitura correta")
    lines.append("A saída é uma simulação explicada por evidências, não uma previsão oficial. Quando a correlação histórica é fraca ou há pouca cobertura temporal, o intervalo deve ser interpretado com mais cautela.")
    return "\n".join(lines)


def build_explainability(global_info: dict[str, Any], global_gold: pd.DataFrame, model_df: pd.DataFrame, backtest: pd.DataFrame, scenarios: pd.DataFrame, nacional: pd.DataFrame, mc: pd.DataFrame, pred_dir: Path) -> dict[str, Any]:
    out_dir = pred_dir / "explicabilidade"
    out_dir.mkdir(parents=True, exist_ok=True)

    context = load_global_context(global_info)
    trace = traceability_from_gold(global_gold)
    evidence = evidence_summary(context, backtest, model_df, global_info.get("analise_eleitoral_outputs", {}))
    contribution = entity_contribution(model_df, scenarios, mc)
    justification = scenario_justification(scenarios)
    md = markdown_explanation(trace, evidence, contribution, justification, nacional)

    save_csv(trace, out_dir / "01_rastreabilidade_arquivos_individuais.csv")
    save_csv(evidence, out_dir / "02_evidencias_globais_usadas.csv")
    save_csv(contribution, out_dir / "03_contribuicao_entidades.csv")
    save_csv(justification, out_dir / "04_justificativa_cenarios.csv")

    md_path = out_dir / "explicacao_detalhada_simulacao.md"
    md_path.write_text(md, encoding="utf-8")

    return {
        "context": context,
        "trace": trace,
        "evidence": evidence,
        "contribution": contribution,
        "justification": justification,
        "markdown": md,
        "markdown_path": str(md_path),
    }
