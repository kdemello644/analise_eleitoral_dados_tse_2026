from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from .utils import MATPLOTLIB_OK, clean_memory, parse_number, safe_name
from .discrete import is_useful_discrete_series, label_category_series, readable_field_label

ALLOWED_PLOT_METRIC_ROLES = {"votos", "abstencao", "comparecimento"}
SKIP_PLOT_ROLES = {"perfil_biometria", "data", "hora", "datetime"}

if MATPLOTLIB_OK:
    import matplotlib.pyplot as plt
else:
    plt = None


def plot_numeric_distributions(df: pd.DataFrame, profile: pd.DataFrame, plots_dir: Path, cfg) -> List[Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    images = []
    if not MATPLOTLIB_OK:
        return images
    if df is None or df.empty or profile is None or profile.empty:
        return images

    cols = profile.loc[
        profile["usar_como_metrica"].astype(str).isin(["True", "true", "1"]),
        "coluna"
    ].astype(str).tolist()[: cfg.top_n_plots]

    for col in cols:
        if col not in df.columns:
            continue
        role = ""
        if "role_sugerido" in profile.columns:
            match = profile.loc[profile["coluna"].astype(str).eq(str(col)), "role_sugerido"]
            role = str(match.iloc[0]) if not match.empty else ""
        if role not in ALLOWED_PLOT_METRIC_ROLES:
            continue
        s = df[col].map(parse_number).dropna().astype(float)
        if s.empty or s.nunique(dropna=True) <= 1:
            continue
        plt.figure(figsize=(9, 4.8))
        plt.hist(s, bins=40)
        plt.title(f"Distribuição - {col}")
        plt.tight_layout()
        path = plots_dir / f"hist_{safe_name(col)}.png"
        plt.savefig(path, dpi=140)
        plt.close()
        images.append(path)
        clean_memory()

    return images


def plot_categorical_frequencies(df: pd.DataFrame, profile: pd.DataFrame, plots_dir: Path, cfg) -> List[Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    images = []
    if not MATPLOTLIB_OK:
        return images
    if df is None or df.empty or profile is None or profile.empty:
        return images

    if "usar_como_discreto" in profile.columns:
        mask = profile["usar_como_discreto"].astype(str).isin(["True", "true", "1"])
    else:
        mask = profile["tipo_inferido"].astype(str).isin(["dimensao", "categorico", "categorico_discreto", "ano_detectado_por_valor"])
    cols = profile.loc[mask, "coluna"].astype(str).tolist()[: cfg.top_n_plots]

    for col in cols:
        if col not in df.columns:
            continue
        role = ""
        if "role_sugerido" in profile.columns:
            match = profile.loc[profile["coluna"].astype(str).eq(str(col)), "role_sugerido"]
            role = str(match.iloc[0]) if not match.empty else ""
        if role in SKIP_PLOT_ROLES or "biometr" in str(col).lower():
            continue
        if not is_useful_discrete_series(df[col], col=col, role=role):
            continue
        labeled = label_category_series(df[col], col=col, role=role)
        freq = labeled.astype(str).replace({"nan": "Sem valor", "": "Sem valor"}).value_counts().head(20)
        freq = freq.loc[~freq.index.astype(str).str.lower().isin(["sem valor", "nan", "none", "null"])]
        if freq.empty or len(freq) <= 1:
            continue
        labels = [str(x).strip() if str(x).strip().lower() not in {"", "nan", "none"} else "Sem valor" for x in freq.index]
        values = pd.to_numeric(pd.Series(freq.values), errors="coerce").fillna(0).to_numpy(float)
        plt.figure(figsize=(10, max(4, len(labels) * 0.35)))
        plt.barh(labels, values)
        plt.title(f"Frequencia - {readable_field_label(col)}")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        path = plots_dir / f"freq_{safe_name(col)}.png"
        plt.savefig(path, dpi=140)
        plt.close()
        images.append(path)
        clean_memory()

    return images


def plot_gold_complete_summary(summary_path: str | Path, plots_dir: Path, cfg) -> List[Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    images = []
    if not MATPLOTLIB_OK:
        return images
    path = Path(summary_path)
    if not path.exists():
        return images
    try:
        if path.suffix.lower() == ".parquet":
            summary = pd.read_parquet(path)
        else:
            summary = pd.read_csv(path, sep=";", dtype=str)
    except Exception:
        return images
    if summary.empty:
        return images
    summary["total"] = pd.to_numeric(summary.get("total", np.nan), errors="coerce").fillna(0)
    summary["votos"] = pd.to_numeric(summary.get("votos", np.nan), errors="coerce").fillna(0)
    summary["eleitorado"] = pd.to_numeric(summary.get("eleitorado", np.nan), errors="coerce").fillna(0)

    cat_fields = [
        "perfil_faixa_etaria",
        "perfil_genero",
        "perfil_instrucao",
        "perfil_estado_civil",
        "perfil_raca_cor",
        "partido",
        "cargo",
        "uf",
    ][: max(1, int(getattr(cfg, "top_n_plots", 15) or 15))]
    for field in cat_fields:
        d = summary.loc[
            summary["tipo_resumo"].astype(str).eq("categoria")
            & summary["campo"].astype(str).eq(field)
            & summary["metrica"].astype(str).eq("linhas_origem")
        ].copy()
        if d.empty:
            continue
        value_col = "votos" if d["votos"].sum() > 0 else "eleitorado" if d["eleitorado"].sum() > 0 else "total"
        d = d.sort_values(value_col, ascending=False).head(20)
        d = d.loc[d["valor"].astype(str).str.strip().ne("")]
        if d[value_col].sum() <= 0 or d["valor"].nunique(dropna=True) <= 1:
            continue
        plt.figure(figsize=(10, max(4, len(d) * 0.35)))
        plt.barh(d["valor"].astype(str), d[value_col].to_numpy(float))
        plt.title(f"Resumo completo - {readable_field_label(field)}")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        out = plots_dir / f"completo_{safe_name(field)}.png"
        plt.savefig(out, dpi=140)
        plt.close()
        images.append(out)
        clean_memory()

    for tipo, campo, label in [
        ("metrica_por_ano", "ano", "por ano"),
        ("metrica_por_uf", "uf", "por UF"),
    ]:
        for metric in ["votos", "eleitorado", "comparecimento_estimado", "abstencao_estimado"]:
            d = summary.loc[
                summary["tipo_resumo"].astype(str).eq(tipo)
                & summary["campo"].astype(str).eq(campo)
                & summary["metrica"].astype(str).eq(metric)
            ].copy()
            if d.empty or d["total"].sum() <= 0 or d["valor"].nunique(dropna=True) <= 1:
                continue
            if campo == "ano":
                d["_ord"] = pd.to_numeric(d["valor"], errors="coerce")
                d = d.sort_values("_ord")
                plt.figure(figsize=(9, 4.8))
                plt.plot(d["valor"].astype(str), d["total"].to_numpy(float), marker="o")
            else:
                d = d.sort_values("total", ascending=False).head(30)
                plt.figure(figsize=(10, max(5, len(d) * 0.25)))
                plt.barh(d["valor"].astype(str), d["total"].to_numpy(float))
                plt.gca().invert_yaxis()
            plt.title(f"Resumo completo - {metric} {label}")
            plt.tight_layout()
            out = plots_dir / f"completo_{safe_name(metric)}_{safe_name(campo)}.png"
            plt.savefig(out, dpi=140)
            plt.close()
            images.append(out)
            clean_memory()

    return images


def plot_gold_datasets(manifest_path: str | Path, plots_dir: Path, cfg) -> List[Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    images = []
    if not MATPLOTLIB_OK:
        return images
    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        return images
    try:
        manifest = pd.read_parquet(manifest_file) if manifest_file.suffix.lower() == ".parquet" else pd.read_csv(manifest_file, sep=";", dtype=str)
    except Exception:
        return images
    if manifest.empty:
        return images

    limit = max(1, int(getattr(cfg, "top_n_plots", 15) or 15))
    for _, row in manifest.head(limit * 3).iterrows():
        data_path = Path(str(row.get("parquet", "") or row.get("csv", "")))
        if not data_path.exists():
            continue
        try:
            data = pd.read_parquet(data_path) if data_path.is_dir() or data_path.suffix.lower() == ".parquet" else pd.read_csv(data_path, sep=";", dtype=str)
        except Exception:
            continue
        if data.empty or "valor" not in data.columns:
            continue
        chart_type = str(row.get("tipo_grafico", ""))
        field = str(row.get("campo", ""))
        if field == "perfil_biometria" or "biometr" in field.lower():
            continue
        metric = str(row.get("metrica", "total"))
        value_col = metric if metric in data.columns else "total"
        data[value_col] = pd.to_numeric(data[value_col], errors="coerce").fillna(0)
        data = data.loc[data["valor"].astype(str).str.strip().ne("")]
        if data[value_col].sum() <= 0 or data["valor"].nunique(dropna=True) <= 1:
            continue

        if chart_type == "por_ano":
            d = data.copy()
            d["_ord"] = pd.to_numeric(d["valor"], errors="coerce")
            d = d.sort_values("_ord")
            plt.figure(figsize=(9, 4.8))
            plt.plot(d["valor"].astype(str), d[value_col].to_numpy(float), marker="o")
            plt.title(f"Completo - {metric} por ano")
        else:
            d = data.sort_values(value_col, ascending=False).head(30)
            plt.figure(figsize=(10, max(4, len(d) * 0.30)))
            plt.barh(d["valor"].astype(str), d[value_col].to_numpy(float))
            title = f"Completo - {readable_field_label(field)}" if chart_type == "categoria" else f"Completo - {metric} por UF"
            plt.title(title)
            plt.gca().invert_yaxis()
        plt.tight_layout()
        out = plots_dir / f"parquet_{safe_name(row.get('chart_id', field or metric))}.png"
        plt.savefig(out, dpi=140)
        plt.close()
        images.append(out)
        clean_memory()
        if len(images) >= limit:
            break

    return images


def plot_global(municipal: pd.DataFrame, timelines: dict[str, pd.DataFrame], corr: pd.DataFrame, plots_dir: Path, cfg) -> List[Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    images = []
    if not MATPLOTLIB_OK:
        return images

    if municipal is not None and not municipal.empty and "uf" in municipal.columns and "votos" in municipal.columns:
        tmp = municipal.copy()
        tmp["uf_plot"] = tmp["uf"].where(tmp["uf"].notna(), "SEM_UF").astype(str).str.strip()
        tmp["uf_plot"] = tmp["uf_plot"].replace({"": "SEM_UF", "nan": "SEM_UF", "None": "SEM_UF"})
        tmp["votos_plot"] = pd.to_numeric(tmp["votos"], errors="coerce").fillna(0)
        agg = tmp.groupby("uf_plot")["votos_plot"].sum().sort_values(ascending=False).head(30)
        if len(agg) > 1 and agg.sum() > 0:
            plt.figure(figsize=(10, max(5, len(agg) * 0.25)))
            plt.barh([str(x) for x in agg.index], agg.values)
            plt.title("Votos agregados por UF")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            path = plots_dir / "votos_por_uf.png"
            plt.savefig(path, dpi=140)
            plt.close()
            images.append(path)

        for metric, title, ylabel in [
            ("pct_abstencao", "Abstencao media por UF", "abstencao media (%)"),
            ("pct_comparecimento", "Comparecimento medio por UF", "comparecimento medio (%)"),
        ]:
            if metric not in tmp.columns:
                continue
            tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
            agg_metric = tmp.groupby("uf_plot")[metric].mean().dropna().sort_values(ascending=False).head(30)
            if len(agg_metric) > 1:
                plt.figure(figsize=(10, max(5, len(agg_metric) * 0.25)))
                plt.barh([str(x) for x in agg_metric.index], agg_metric.values * 100)
                plt.title(title)
                plt.xlabel(ylabel)
                plt.gca().invert_yaxis()
                plt.tight_layout()
                path = plots_dir / f"{safe_name(metric)}_por_uf.png"
                plt.savefig(path, dpi=140)
                plt.close()
                images.append(path)

        if {"nm_municipio", "cd_municipio", "pct_abstencao"}.issubset(tmp.columns):
            tmp["pct_abstencao"] = pd.to_numeric(tmp["pct_abstencao"], errors="coerce")
            mun = tmp.dropna(subset=["pct_abstencao"]).copy()
            mun["municipio_plot"] = mun["uf_plot"] + " | " + mun["nm_municipio"].astype(str).where(mun["nm_municipio"].astype(str).str.strip().ne(""), mun["cd_municipio"].astype(str))
            top = mun.groupby("municipio_plot")["pct_abstencao"].mean().sort_values(ascending=False).head(25)
            if len(top) > 1:
                plt.figure(figsize=(11, max(5, len(top) * 0.30)))
                plt.barh(top.index.astype(str), top.values * 100)
                plt.title("Municipios com maior abstencao media")
                plt.xlabel("abstencao media (%)")
                plt.gca().invert_yaxis()
                plt.tight_layout()
                path = plots_dir / "municipios_maior_abstencao.png"
                plt.savefig(path, dpi=140)
                plt.close()
                images.append(path)

    tn = timelines.get("timeline_nacional", pd.DataFrame()) if timelines else pd.DataFrame()
    if tn is not None and not tn.empty:
        tn = tn.copy()
        tn["ano_num"] = pd.to_numeric(tn.get("ano", np.nan), errors="coerce")
        tn = tn.loc[tn["ano_num"].notna()]
        for col in ["votos", "comparecimento_estimado", "abstencao_estimado"]:
            if col not in tn.columns:
                continue
            yearly = tn.groupby("ano_num")[col].sum().reset_index().sort_values("ano_num")
            if yearly[col].sum() <= 0 or yearly["ano_num"].nunique(dropna=True) <= 1:
                continue
            plt.figure(figsize=(9, 4.8))
            plt.plot(yearly["ano_num"], yearly[col], marker="o")
            plt.title(f"Timeline nacional - {col}")
            plt.xlabel("ano")
            plt.ylabel(col)
            plt.tight_layout()
            path = plots_dir / f"timeline_nacional_{safe_name(col)}.png"
            plt.savefig(path, dpi=140)
            plt.close()
            images.append(path)

    te = timelines.get("timeline_entidades", pd.DataFrame()) if timelines else pd.DataFrame()
    if te is not None and not te.empty and "entidade" in te.columns:
        ent = te.copy()
        ent["votos"] = pd.to_numeric(ent.get("votos", np.nan), errors="coerce").fillna(0)
        ent["share"] = pd.to_numeric(ent.get("share", np.nan), errors="coerce")
        ent["ano_num"] = pd.to_numeric(ent.get("ano", np.nan), errors="coerce")
        latest = ent["ano_num"].dropna().max()
        if pd.notna(latest):
            latest_ent = ent.loc[ent["ano_num"].eq(latest)].copy()
        else:
            latest_ent = ent
        top_votes = latest_ent.groupby("entidade", dropna=False)["votos"].sum().sort_values(ascending=False).head(25)
        if len(top_votes) > 1 and top_votes.sum() > 0:
            plt.figure(figsize=(11, max(5, len(top_votes) * 0.30)))
            plt.barh([str(x) for x in top_votes.index], top_votes.values)
            plt.title("Entidades com mais votos no ano mais recente")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            path = plots_dir / "entidades_top_votos_ano_recente.png"
            plt.savefig(path, dpi=140)
            plt.close()
            images.append(path)

        top_share = pd.Series(dtype=float)
        if not top_share.empty:
            plt.figure(figsize=(11, max(5, len(top_share) * 0.30)))
            plt.barh([str(x) for x in top_share.index], top_share.values * 100)
            plt.title("Entidades com maior share medio no ano mais recente")
            plt.xlabel("share medio (%)")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            path = plots_dir / "entidades_top_share_ano_recente.png"
            plt.savefig(path, dpi=140)
            plt.close()
            images.append(path)

    if False and corr is not None and not corr.empty:
        top = corr.head(25)
        labels = (top["metrica"].astype(str) + " " + top["ano_1"].astype(str) + "→" + top["ano_2"].astype(str)).tolist()
        vals = pd.to_numeric(top["pearson"], errors="coerce").fillna(0).to_numpy(float)
        if len(labels):
            plt.figure(figsize=(10, max(5, len(labels) * 0.28)))
            plt.barh(labels, vals)
            plt.title("Correlações temporais mais fortes")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            path = plots_dir / "correlacoes_temporais_top.png"
            plt.savefig(path, dpi=140)
            plt.close()
            images.append(path)

    clean_memory()
    return images


def plot_behavior_clusters(clustered: pd.DataFrame, interpretation: pd.DataFrame, plots_dir: Path, cfg) -> List[Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    images = []
    if not MATPLOTLIB_OK:
        return images

    if clustered is not None and not clustered.empty and "cluster_comportamento_eleitoral" in clustered.columns:
        tmp = clustered.copy()
        tmp["cluster_plot"] = tmp["cluster_comportamento_eleitoral"].astype(str)

        counts = tmp["cluster_plot"].value_counts().sort_index()
        if len(counts) > 1:
            plt.figure(figsize=(10, 4.8))
            plt.bar(counts.index.astype(str), counts.values)
            plt.title("Clusters comportamentais - quantidade de recortes")
            plt.xlabel("cluster")
            plt.ylabel("recortes")
            plt.tight_layout()
            path = plots_dir / "clusters_comportamento_tamanho.png"
            plt.savefig(path, dpi=140)
            plt.close()
            images.append(path)

        if "taxa_abstencao" in tmp.columns:
            abst = tmp.copy()
            abst["taxa_abstencao"] = pd.to_numeric(abst["taxa_abstencao"], errors="coerce")
            abst = abst.groupby("cluster_plot")["taxa_abstencao"].mean().dropna().sort_values(ascending=False)
            if len(abst) > 1:
                plt.figure(figsize=(10, 4.8))
                plt.bar(abst.index.astype(str), abst.values * 100)
                plt.title("Clusters comportamentais - abstencao media")
                plt.xlabel("cluster")
                plt.ylabel("abstencao media (%)")
                plt.tight_layout()
                path = plots_dir / "clusters_comportamento_abstencao.png"
                plt.savefig(path, dpi=140)
                plt.close()
                images.append(path)

        if "uf" in tmp.columns:
            uf = tmp.groupby(["cluster_plot", "uf"], dropna=False).size().reset_index(name="qtd")
            uf = uf.sort_values(["cluster_plot", "qtd"], ascending=[True, False]).groupby("cluster_plot").head(5)
            labels = (uf["cluster_plot"].astype(str) + " | " + uf["uf"].astype(str)).tolist()
            vals = pd.to_numeric(uf["qtd"], errors="coerce").fillna(0).to_numpy(float)
            if len(set(labels)) > 1:
                plt.figure(figsize=(11, max(5, len(labels) * 0.28)))
                plt.barh(labels, vals)
                plt.title("Clusters comportamentais - principais UFs por cluster")
                plt.gca().invert_yaxis()
                plt.tight_layout()
                path = plots_dir / "clusters_comportamento_ufs.png"
                plt.savefig(path, dpi=140)
                plt.close()
                images.append(path)

    if interpretation is not None and not interpretation.empty and "entidade_mais_associada" in interpretation.columns:
        d = interpretation.copy()
        if "total_votos_cluster" in d.columns:
            d["total_votos_cluster"] = pd.to_numeric(d["total_votos_cluster"], errors="coerce").fillna(0)
            d = d.sort_values("total_votos_cluster", ascending=False).head(getattr(cfg, "top_n_plots", 15))
            labels = (d.get("cluster", pd.Series(dtype=str)).astype(str) + " | " + d["entidade_mais_associada"].astype(str)).tolist()
            vals = d["total_votos_cluster"].to_numpy(float)
            if len(set(labels)) > 1 and vals.sum() > 0:
                plt.figure(figsize=(11, max(5, len(labels) * 0.35)))
                plt.barh(labels, vals)
                plt.title("Clusters comportamentais - entidade dominante e votos")
                plt.gca().invert_yaxis()
                plt.tight_layout()
                path = plots_dir / "clusters_comportamento_entidade_dominante.png"
                plt.savefig(path, dpi=140)
                plt.close()
                images.append(path)

    clean_memory()
    return images


def plot_prediction(nacional: pd.DataFrame, mc: pd.DataFrame, plots_dir: Path, cfg) -> List[Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    images = []
    if not MATPLOTLIB_OK:
        return images

    if nacional is not None and not nacional.empty:
        scen_list = nacional["cenario"].dropna().astype(str).unique().tolist()[:6]
        for scen in scen_list:
            d = nacional.loc[nacional["cenario"].astype(str).eq(scen)].sort_values("share_nacional_pred_2026", ascending=False).head(12)
            labels = [str(x).strip() if str(x).strip().lower() not in {"", "nan", "none"} else "SEM_ENTIDADE" for x in d["entidade"].tolist()]
            values = pd.to_numeric(d["share_nacional_pred_2026"], errors="coerce").fillna(0).to_numpy(float) * 100
            if len(set(labels)) > 1 and values.sum() > 0:
                plt.figure(figsize=(10, max(4, len(labels) * 0.35)))
                plt.barh(labels, values)
                plt.title(f"Cenário nacional 2026 - {scen}")
                plt.xlabel("share estimado (%)")
                plt.gca().invert_yaxis()
                plt.tight_layout()
                path = plots_dir / f"cenario_nacional_{safe_name(scen)}.png"
                plt.savefig(path, dpi=140)
                plt.close()
                images.append(path)

    if mc is not None and not mc.empty:
        d = mc.sort_values("share_medio", ascending=False).head(15)
        labels = [f"{c} | {e}" for c, e in zip(d["cenario"].astype(str), d["entidade"].astype(str))]
        values = pd.to_numeric(d["share_medio"], errors="coerce").fillna(0).to_numpy(float) * 100
        if len(set(labels)) > 1:
            plt.figure(figsize=(10, max(5, len(labels) * 0.35)))
            plt.barh(labels, values)
            plt.title("Monte Carlo - principais médias")
            plt.xlabel("share médio (%)")
            plt.gca().invert_yaxis()
            plt.tight_layout()
            path = plots_dir / "monte_carlo_top_medias.png"
            plt.savefig(path, dpi=140)
            plt.close()
            images.append(path)

    clean_memory()
    return images
