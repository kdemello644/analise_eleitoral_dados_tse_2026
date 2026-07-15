from __future__ import annotations

import argparse
import sys
from html import escape
from typing import Any

import pandas as pd
import plotly.express as px
import requests
from requests import ReadTimeout
import streamlit as st


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--api", default="http://localhost:8055", help="URL base da API FastAPI.")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout das consultas API em segundos; o front limita em 30s.")
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


def api_get(base_url: str, path: str, params: dict[str, Any] | None = None, timeout: int = 30) -> dict[str, Any]:
    url = base_url.rstrip("/") + path
    response = requests.get(url, params=params or {}, timeout=timeout)
    response.raise_for_status()
    return response.json()


def api_post(base_url: str, path: str, payload: dict[str, Any], timeout: int = 30) -> dict[str, Any]:
    url = base_url.rstrip("/") + path
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def safe_api_get(base_url: str, path: str, params: dict[str, Any] | None = None, timeout: int = 30) -> dict[str, Any]:
    try:
        return api_get(base_url, path, params=params, timeout=timeout)
    except ReadTimeout:
        st.warning(f"Consulta demorou mais que {timeout}s: {path}. Tente reduzir Top N, escolher uma modalidade mais leve ou aguardar a camada ouro terminar.")
        return {"status": "timeout", "dados": []}
    except Exception as exc:
        st.error(f"Erro consultando {path}: {exc}")
        return {"status": "error", "erro": str(exc), "dados": []}


def df_from_payload(payload: dict[str, Any], key: str = "dados") -> pd.DataFrame:
    data = payload.get(key) or []
    return pd.DataFrame(data)


def format_value(value: Any) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, float):
        if abs(value) < 1 and value != 0:
            return f"{value:.1%}"
        return f"{value:,.0f}".replace(",", ".")
    if isinstance(value, int):
        return f"{value:,}".replace(",", ".")
    return str(value)


def clean_profile_value(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>", "sem valor", "geral"}:
        return ""
    if "=" in text:
        text = text.split("=", 1)[1].strip()
    return text


def display_label(col: str) -> str:
    labels = {
        "perfil_faixa_etaria": "faixa etaria",
        "perfil_genero": "sexo",
        "perfil_instrucao": "escolaridade",
        "perfil_estado_civil": "estado civil",
        "perfil_raca_cor": "raca/cor",
        "share_perfil": "participacao",
        "eleitorado": "base",
        "pessoas_perfil_estimado": "pessoas",
        "nm_municipio": "municipio",
    }
    return labels.get(str(col), str(col).replace("_", " "))


def build_profile_label(row: dict[str, Any]) -> str:
    parts = []
    for label, col in [
        ("Faixa etaria", "perfil_faixa_etaria"),
        ("Sexo", "perfil_genero"),
        ("Escolaridade", "perfil_instrucao"),
        ("Estado civil", "perfil_estado_civil"),
        ("Raca/cor", "perfil_raca_cor"),
    ]:
        value = clean_profile_value(row.get(col))
        if value:
            parts.append(f"{label}: {value}")
    if parts:
        return " | ".join(parts)
    return str(row.get("perfil_combinado") or row.get("descricao") or row.get("valor_perfil") or "Perfil")


def add_profile_people_estimate(perfil: pd.DataFrame, resumo: pd.DataFrame) -> pd.DataFrame:
    if perfil.empty or "share_perfil" not in perfil.columns or resumo.empty:
        return perfil
    total_col = next((c for c in ["eleitorado", "eleitorado_total", "qtd_eleitores"] if c in resumo.columns), None)
    if not total_col:
        return perfil
    total = pd.to_numeric(resumo[total_col], errors="coerce").dropna()
    if total.empty:
        return perfil
    # Usa o maior total nacional disponivel como base de leitura; evita expor somas duplicadas vindas de agregacoes intermediarias.
    base = float(total.max())
    out = perfil.copy()
    share = pd.to_numeric(out["share_perfil"], errors="coerce").fillna(0).clip(lower=0)
    out["pessoas_perfil_estimado"] = share * base
    return out


def profile_weight_col(df: pd.DataFrame) -> str | None:
    for col in ["histograma_qtd_pessoas", "qtd_eleitores_perfil", "pessoas_perfil_estimado", "eleitorado", "share_perfil"]:
        if col in df.columns:
            return col
    return None


def profile_total_histogram(df: pd.DataFrame, title: str = "Quantidade de pessoas por perfil") -> None:
    if df.empty:
        return
    weight_col = profile_weight_col(df)
    if not weight_col:
        return
    work = df.copy()
    work["perfil_label"] = work.apply(lambda row: build_profile_label(row.to_dict()), axis=1)
    work[weight_col] = pd.to_numeric(work[weight_col], errors="coerce").fillna(0)
    work = work[(work["perfil_label"].astype(str).str.strip() != "") & (work[weight_col] > 0)]
    if work.empty:
        return
    if "ano" in work.columns:
        group_cols = ["perfil_label"]
        color_col = "perfil_instrucao" if "perfil_instrucao" in work.columns else ("perfil_genero" if "perfil_genero" in work.columns else None)
        if color_col:
            group_cols.append(color_col)
        work = work.groupby(group_cols, as_index=False)[weight_col].sum()
    work = work.sort_values(weight_col, ascending=False).head(25)
    color = "perfil_instrucao" if "perfil_instrucao" in work.columns else ("perfil_genero" if "perfil_genero" in work.columns else None)
    fig = px.bar(
        work.sort_values(weight_col),
        x=weight_col,
        y="perfil_label",
        orientation="h",
        color=color,
        title=title,
        text=weight_col,
    )
    fig.update_traces(texttemplate="%{text:,.0f}", hovertemplate="%{y}<br>%{x:,.0f} pessoas<extra></extra>")
    fig.update_layout(
        height=max(540, min(920, 34 * len(work) + 150)),
        template="plotly_white",
        margin=dict(l=8, r=8, t=58, b=24),
        yaxis_title="Perfil",
        xaxis_title="Quantidade de pessoas",
        legend_title_text=display_label(color) if color else "",
    )
    st.plotly_chart(fig, use_container_width=True)


def profile_dimension_charts(df: pd.DataFrame) -> None:
    if df.empty:
        return
    weight_col = profile_weight_col(df)
    if not weight_col:
        return
    dims = [
        ("perfil_faixa_etaria", "Faixa etaria"),
        ("perfil_genero", "Sexo"),
        ("perfil_instrucao", "Escolaridade"),
        ("perfil_estado_civil", "Estado civil"),
        ("perfil_raca_cor", "Raca/cor"),
    ]
    available = [(col, label) for col, label in dims if col in df.columns]
    if not available:
        return
    st.markdown("### Distribuicao por dimensao do perfil")
    chart_cols = st.columns(2)
    for idx, (col, label) in enumerate(available):
        work = df[[col, weight_col]].copy()
        work[col] = work[col].map(clean_profile_value)
        work[weight_col] = pd.to_numeric(work[weight_col], errors="coerce").fillna(0)
        work = work[(work[col] != "") & (work[weight_col] > 0)]
        if work.empty:
            continue
        agg = work.groupby(col, as_index=False)[weight_col].sum().sort_values(weight_col, ascending=False).head(18)
        if len(agg) <= 8:
            fig = px.pie(agg, names=col, values=weight_col, hole=0.42, title=label)
            fig.update_traces(textposition="inside", textinfo="percent+label")
        else:
            fig = px.bar(agg.sort_values(weight_col), x=weight_col, y=col, orientation="h", color=col, title=label)
        fig.update_layout(height=420, template="plotly_white", showlegend=False, margin=dict(l=8, r=8, t=48, b=20))
        with chart_cols[idx % 2]:
            st.plotly_chart(fig, use_container_width=True)


def histogram_dimension_cards(df: pd.DataFrame, title: str, value_col: str = "qtd_pessoas") -> None:
    if df.empty or "dimensao_perfil" not in df.columns or "valor_perfil" not in df.columns:
        return
    work = df.copy()
    if value_col not in work.columns:
        value_col = "qtd_votos" if "qtd_votos" in work.columns else ""
    if not value_col:
        return
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce").fillna(0)
    work = work[(work[value_col] > 0) & (work["valor_perfil"].astype(str).str.strip() != "")]
    if work.empty:
        return
    st.markdown(f"### {title}")
    wanted = ["perfil_combinado", "faixa_etaria", "sexo_genero", "escolaridade", "estado_civil", "raca_cor"]
    dims = [d for d in wanted if d in set(work["dimensao_perfil"].astype(str))]
    cols = st.columns(2)
    for idx, dim in enumerate(dims[:6]):
        sub = work[work["dimensao_perfil"].astype(str) == dim].copy()
        if "share_histograma" in sub.columns:
            sub["share_histograma"] = pd.to_numeric(sub["share_histograma"], errors="coerce").fillna(0)
        sub = sub.groupby("valor_perfil", as_index=False)[value_col].sum().sort_values(value_col, ascending=False).head(18)
        if sub.empty:
            continue
        if dim != "perfil_combinado" and len(sub) <= 10:
            fig = px.pie(sub, names="valor_perfil", values=value_col, hole=0.42, title=display_label(dim))
            fig.update_traces(textinfo="percent+label")
        else:
            fig = px.bar(sub.sort_values(value_col), x=value_col, y="valor_perfil", orientation="h", color="valor_perfil", title=display_label(dim))
        fig.update_layout(height=430, template="plotly_white", showlegend=False, margin=dict(l=8, r=8, t=48, b=18))
        with cols[idx % 2]:
            st.plotly_chart(fig, use_container_width=True)


def split_winner_status(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df, df
    work = df.copy()
    if "partido" not in work.columns:
        for col in ["entidade", "sg_partido", "partido_vencedor", "nm_partido", "nr_partido"]:
            if col in work.columns:
                work["partido"] = work[col].astype(str)
                break
    if "resultado_eleitoral" in work.columns:
        status = work["resultado_eleitoral"].astype(str).str.lower()
        winners = work[status.str.contains("vencedor|eleito|ganhou", regex=True, na=False)].copy()
        losers = work[~work.index.isin(winners.index)].copy()
        return winners, losers
    if "rank_entidade" in work.columns:
        rank = pd.to_numeric(work["rank_entidade"], errors="coerce")
        return work[rank == 1].copy(), work[rank != 1].copy()
    return work.head(0).copy(), work


def inject_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.15rem; max-width: 1500px;}
        section[data-testid="stSidebar"] {width: 230px !important;}
        section[data-testid="stSidebar"] * {font-size: .92rem;}
        .hero-box {
            padding: 1.3rem 1.45rem;
            border-radius: 10px;
            background:
                linear-gradient(120deg, rgba(15, 23, 42, .96) 0%, rgba(17, 94, 89, .96) 62%, rgba(30, 64, 175, .92) 100%);
            color: #fff;
            margin-bottom: 1rem;
            box-shadow: 0 18px 42px rgba(8, 20, 45, .22);
        }
        .hero-kicker {font-size: .82rem; letter-spacing: .06em; text-transform: uppercase; color: #9df3df; font-weight: 700;}
        .hero-title {font-size: 2.1rem; line-height: 1.05; font-weight: 800; margin: .35rem 0 .45rem;}
        .hero-subtitle {max-width: 900px; color: #e7f5f3; font-size: 1rem;}
        .card-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: .75rem;
            margin: .75rem 0 1rem;
        }
        .metric-card {
            border: 1px solid #d7e0ee;
            border-radius: 8px;
            background: #fff;
            padding: .9rem 1rem;
            box-shadow: 0 10px 26px rgba(21, 35, 59, .06);
        }
        .metric-label {font-size: .78rem; color: #56657a; font-weight: 700; text-transform: uppercase;}
        .metric-value {font-size: 1.75rem; font-weight: 800; color: #0b1830; margin-top: .15rem;}
        .metric-help {font-size: .82rem; color: #68758a; margin-top: .15rem;}
        .card-grid .metric-card:nth-child(1) {border-top: 4px solid #2563eb;}
        .card-grid .metric-card:nth-child(2) {border-top: 4px solid #059669;}
        .card-grid .metric-card:nth-child(3) {border-top: 4px solid #dc2626;}
        .card-grid .metric-card:nth-child(4) {border-top: 4px solid #7c3aed;}
        .control-panel {
            border: 1px solid #d7e0ee;
            border-radius: 8px;
            background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
            padding: 1rem;
            margin: .85rem 0 1rem;
            box-shadow: 0 10px 28px rgba(21, 35, 59, .07);
        }
        .control-title {font-size: 1rem; font-weight: 850; color: #10243d; margin-bottom: .15rem;}
        .control-subtitle {font-size: .86rem; color: #64748b; margin-bottom: .6rem;}
        .command-box {
            background: #0b1220;
            color: #dbeafe;
            border-radius: 8px;
            padding: .75rem;
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            font-size: .75rem;
            white-space: pre-wrap;
            border: 1px solid #1e293b;
        }
        .quick-actions {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: .7rem;
            margin: .75rem 0 1rem;
        }
        .action-card {
            border: 1px solid #d7e0ee;
            border-radius: 8px;
            background: linear-gradient(180deg, #ffffff 0%, #f7fbff 100%);
            padding: .85rem 1rem;
            box-shadow: 0 8px 22px rgba(21, 35, 59, .05);
        }
        .quick-actions .action-card:nth-child(1) {border-left: 5px solid #2563eb;}
        .quick-actions .action-card:nth-child(2) {border-left: 5px solid #059669;}
        .quick-actions .action-card:nth-child(3) {border-left: 5px solid #d97706;}
        .action-title {font-size: .95rem; font-weight: 800; color: #0b1830;}
        .action-subtitle {font-size: .82rem; color: #5f6f86; margin-top: .15rem;}
        .profile-card {
            border: 1px solid #d7e0ee;
            border-radius: 8px;
            background: #fff;
            padding: .85rem;
            min-height: 120px;
            box-shadow: 0 8px 20px rgba(21, 35, 59, .05);
        }
        .profile-title {font-weight: 800; color: #101827; margin-bottom: .35rem;}
        .pill {
            display: inline-block;
            border-radius: 999px;
            padding: .18rem .5rem;
            background: #e8f5ff;
            color: #075985;
            font-size: .78rem;
            margin: .12rem .18rem .12rem 0;
        }
        .log-box {
            background: #08111f;
            color: #dcecff;
            border-radius: 8px;
            padding: .85rem;
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            font-size: .78rem;
            max-height: 430px;
            overflow: auto;
            white-space: pre-wrap;
        }
        div[data-testid="stPlotlyChart"] {
            transition: transform .15s ease, box-shadow .15s ease;
            transform-origin: center top;
        }
        div[data-testid="stPlotlyChart"]:hover {
            transform: scale(1.015);
            box-shadow: 0 18px 40px rgba(15, 23, 42, .12);
            z-index: 5;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=120, show_spinner=False)
def cached_api_get(base_url: str, path: str, params_tuple: tuple[tuple[str, Any], ...], timeout: int) -> dict[str, Any]:
    return api_get(base_url, path, params=dict(params_tuple), timeout=timeout)


def render_quick_actions() -> None:
    st.markdown(
        """
        <div class="quick-actions">
          <div class="action-card"><div class="action-title">1. Ver Brasil</div><div class="action-subtitle">Resumo nacional carregado direto da ouro.</div></div>
          <div class="action-card"><div class="action-title">2. Escolher estado</div><div class="action-subtitle">Mapa e ranking por UF quando precisar.</div></div>
          <div class="action-card"><div class="action-title">3. Gerar PDF</div><div class="action-subtitle">Brasil primeiro, depois estados e municipios.</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_brasil_payload(payload: dict[str, Any]) -> None:
    partidos_payload = {"dados": payload.get("partidos") or [], "fonte": payload.get("fonte_partidos", "-")}
    perfis_payload = {"dados": payload.get("perfis") or []}
    metricas_payload = {"dados": payload.get("metricas") or []}
    render_stat_grid(
        [
            ("Fonte partidos", partidos_payload.get("fonte", "-"), "ouro Brasil processado"),
            ("Perfis", len(perfis_payload.get("dados") or []), "cards de eleitor medio"),
            ("Partidos", len(partidos_payload.get("dados") or []), "linhas retornadas"),
            ("Metricas", len(metricas_payload.get("dados") or []), "anos retornados"),
        ]
    )
    party_chart(df_from_payload(partidos_payload), "Brasil por partido")
    st.markdown("### Eleitor medio no Brasil")
    profile_cards(df_from_payload(perfis_payload))


def render_brasil_tabelas(payload: dict[str, Any]) -> None:
    tabelas = payload.get("tabelas") or {}
    resumo = pd.DataFrame(tabelas.get("resumo") or [])
    perfil = pd.DataFrame(tabelas.get("perfil_eleitor") or [])
    resultado = pd.DataFrame(tabelas.get("resultado_partido") or [])
    hist_perfil = pd.DataFrame(tabelas.get("contagem_colunas_perfil_eleitor") or [])
    hist_partido = pd.DataFrame(tabelas.get("contagem_colunas_perfil_partido") or [])
    hist_candidato = pd.DataFrame(tabelas.get("contagem_colunas_perfil_candidato") or [])
    hist_cluster = pd.DataFrame(tabelas.get("contagem_colunas_clusters_eleitores") or [])
    perfil = add_profile_people_estimate(perfil, resumo)
    render_stat_grid(
        [
            ("Resumo", len(resumo), "linhas de ouro/brasil/resumo"),
            ("Perfil eleitor", len(perfil), "linhas de ouro/brasil/perfil_eleitor"),
            ("Resultado partido", len(resultado), "linhas de ouro/brasil/resultado_partido"),
            ("Histogramas", len(hist_perfil) + len(hist_partido) + len(hist_candidato), "contagens auxiliares sem nulos"),
        ]
    )
    if not hist_perfil.empty:
        histogram_dimension_cards(hist_perfil, "Histogramas do eleitor no Brasil", "qtd_pessoas")
    if not perfil.empty:
        st.markdown("### Quantidade de pessoas por perfil")
        if hist_perfil.empty:
            profile_total_histogram(perfil, "Brasil - quantas pessoas existem em cada perfil")
        st.markdown("### Perfil predominante do eleitor")
        profile_cards(perfil, max_cards=8)
        if hist_perfil.empty:
            profile_dimension_charts(perfil)
        cols = [
            c
            for c in [
                "ano",
                "perfil_faixa_etaria",
                "perfil_genero",
                "perfil_instrucao",
                "perfil_estado_civil",
                "perfil_raca_cor",
                "share_perfil",
                "qtd_eleitores_perfil",
                "histograma_qtd_pessoas",
                "pessoas_perfil_estimado",
            ]
            if c in perfil.columns
        ]
        with st.expander("Dados usados nos graficos de perfil"):
            st.dataframe(perfil[cols] if cols else perfil, use_container_width=True, height=260)
    if not resultado.empty:
        vencedores, perdedores = split_winner_status(resultado)
        st.markdown("### Quem ganhou")
        if vencedores.empty:
            st.warning("A camada ouro ainda nao marcou vencedores neste recorte. Reprocesse estados+Brasil para gerar resultado_eleitoral.")
        else:
            party_chart(vencedores.rename(columns={"share_votos": "share_pred_2026"}), "Brasil - vencedores por partido")
            interp_col = "interpretacao_resultado" if "interpretacao_resultado" in vencedores.columns else ""
            if interp_col:
                for text in vencedores[interp_col].dropna().astype(str).head(3).tolist():
                    st.info(text)
        if not perdedores.empty:
            st.markdown("### Quem nao ganhou")
            party_chart(perdedores.rename(columns={"share_votos": "share_pred_2026"}), "Brasil - partidos/candidaturas que nao lideraram")
        with st.expander("Dados usados nos graficos de resultado"):
            st.dataframe(resultado, use_container_width=True, height=260)
    if not hist_partido.empty:
        histogram_dimension_cards(hist_partido, "Quem vota por partido", "qtd_votos")
    if not hist_candidato.empty:
        histogram_dimension_cards(hist_candidato, "Perfil do voto por candidato", "qtd_votos")
    if not hist_cluster.empty:
        histogram_dimension_cards(hist_cluster, "Histogramas dos clusters", "qtd_pessoas")
    if not resumo.empty:
        with st.expander("Resumo bruto Brasil"):
            st.dataframe(resumo, use_container_width=True, height=220)


def render_stat_grid(cards: list[tuple[str, Any, str]]) -> None:
    html_cards = []
    for label, value, help_text in cards:
        html_cards.append(
            "<div class='metric-card'>"
            f"<div class='metric-label'>{escape(str(label))}</div>"
            f"<div class='metric-value'>{escape(format_value(value))}</div>"
            f"<div class='metric-help'>{escape(str(help_text))}</div>"
            "</div>"
        )
    st.markdown("<div class='card-grid'>" + "".join(html_cards) + "</div>", unsafe_allow_html=True)


def party_chart(df: pd.DataFrame, title: str) -> None:
    if df.empty:
        st.info("Sem dados de partido para graficar nesta consulta.")
        return
    work = df.copy()
    if "partido" not in work.columns:
        for col in ["entidade", "sg_partido", "partido_vencedor", "nm_partido", "nr_partido"]:
            if col in work.columns:
                work["partido"] = work[col].astype(str)
                break
    if "partido" not in work.columns:
        st.info("Sem coluna de partido/entidade para graficar nesta consulta.")
        return
    share_col = "share_pred_2026" if "share_pred_2026" in work.columns else None
    if share_col:
        work[share_col] = pd.to_numeric(work[share_col], errors="coerce").fillna(0)
        work = work.sort_values(share_col, ascending=False).head(20)
        fig = px.bar(
            work.sort_values(share_col),
            x=share_col,
            y="partido",
            orientation="h",
            title=title,
            text=share_col,
            color="partido",
        )
        fig.update_traces(texttemplate="%{text:.1%}", hovertemplate="%{y}<br>%{x:.2%}<extra></extra>")
        fig.update_xaxes(tickformat=".0%")
    else:
        value_col = "votos_pred_2026" if "votos_pred_2026" in work.columns else work.columns[-1]
        work[value_col] = pd.to_numeric(work[value_col], errors="coerce").fillna(0)
        work = work.sort_values(value_col, ascending=False).head(20)
        fig = px.bar(work.sort_values(value_col), x=value_col, y="partido", orientation="h", title=title, color="partido")
    fig.update_layout(height=520, template="plotly_white", showlegend=False, margin=dict(l=10, r=10, t=56, b=20))
    st.plotly_chart(fig, use_container_width=True)


def state_map_chart(df: pd.DataFrame) -> None:
    if df.empty or "uf" not in df.columns:
        st.info("Sem dados de mapa para esta consulta.")
        return
    work = df.copy()
    work["uf"] = work["uf"].astype(str).str.upper()
    work["lat"] = work["uf"].map(lambda uf: UF_CENTROIDS.get(uf, (None, None))[0])
    work["lon"] = work["uf"].map(lambda uf: UF_CENTROIDS.get(uf, (None, None))[1])
    work = work.dropna(subset=["lat", "lon"])
    if work.empty:
        st.info("Os estados retornados nao possuem coordenadas conhecidas no dashboard.")
        return
    if "share_pred_2026" in work.columns:
        work["share_pred_2026"] = pd.to_numeric(work["share_pred_2026"], errors="coerce").fillna(0)
    else:
        work["share_pred_2026"] = 0
    work["tamanho"] = (work["share_pred_2026"] * 100).clip(lower=8)
    hover_cols = [c for c in ["partido", "share_pred_2026", "votos_pred_2026", "perfil_eleitor_2026", "fonte"] if c in work.columns]
    fig = px.scatter_geo(
        work,
        lat="lat",
        lon="lon",
        size="tamanho",
        color="partido" if "partido" in work.columns else None,
        hover_name="uf",
        hover_data=hover_cols,
        title="Mapa do Brasil por estado",
        projection="mercator",
    )
    fig.update_geos(
        lataxis_range=[-35, 6],
        lonaxis_range=[-75, -32],
        showland=True,
        landcolor="#eef3f7",
        showcountries=True,
        countrycolor="#aab5c4",
        showocean=True,
        oceancolor="#eaf6ff",
        fitbounds=False,
    )
    fig.update_layout(height=610, template="plotly_white", margin=dict(l=0, r=0, t=55, b=0), legend_title_text="Partido")
    st.plotly_chart(fig, use_container_width=True)


def profile_cards(df: pd.DataFrame, max_cards: int = 10) -> None:
    if df.empty:
        st.info("Sem perfis para mostrar nesta consulta.")
        return
    rows = df.head(max_cards).to_dict(orient="records")
    columns = st.columns(2)
    for idx, row in enumerate(rows):
        profile = build_profile_label(row)
        tags = []
        for col in ["ano", "uf", "nm_municipio", "share_perfil", "pessoas_perfil_estimado", "eleitorado"]:
            if col in row and pd.notna(row[col]) and row[col] != "":
                tags.append((display_label(col), row[col]))
        with columns[idx % 2]:
            pills = "".join(f"<span class='pill'>{escape(str(k))}: {escape(format_value(v))}</span>" for k, v in tags[:5])
            st.markdown(
                "<div class='profile-card'>"
                f"<div class='profile-title'>{escape(str(profile))}</div>"
                f"{pills}"
                "</div>",
                unsafe_allow_html=True,
            )


def cluster_cards(df: pd.DataFrame, title: str, max_cards: int = 12) -> None:
    st.markdown(f"### {escape(title)}", unsafe_allow_html=True)
    if df.empty:
        st.info("Sem clusters para mostrar nesta consulta.")
        return
    rows = df.head(max_cards).to_dict(orient="records")
    columns = st.columns(2)
    for idx, row in enumerate(rows):
        description = row.get("descricao") or row.get("perfil_combinado") or "Cluster"
        cluster_id = row.get("cluster_id", "")
        heading = f"Cluster {cluster_id}" if str(cluster_id).strip() else "Cluster"
        tags = []
        for col in ["ano", "uf", "nm_municipio", "partido", "share_cluster", "eleitorado", "votos_proxy"]:
            value = row.get(col)
            if pd.notna(value) and str(value).strip():
                tags.append((col, value))
        with columns[idx % 2]:
            pills = "".join(f"<span class='pill'>{escape(str(k))}: {escape(format_value(v))}</span>" for k, v in tags[:7])
            st.markdown(
                "<div class='profile-card'>"
                f"<div class='profile-title'>{escape(str(heading))}</div>"
                f"<div>{escape(str(description))}</div>"
                f"{pills}"
                "</div>",
                unsafe_allow_html=True,
            )


def cluster_chart(df: pd.DataFrame, title: str) -> None:
    if df.empty or "cluster_id" not in df.columns:
        return
    value_col = None
    for candidate in ["share_cluster", "votos_proxy", "eleitorado"]:
        if candidate in df.columns:
            value_col = candidate
            break
    if not value_col:
        return
    work = df.copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce").fillna(0)
    work["cluster_label"] = "Cluster " + work["cluster_id"].astype(str)
    color_col = "partido" if "partido" in work.columns else "cluster_label"
    fig = px.bar(
        work.sort_values(value_col).tail(30),
        x=value_col,
        y="cluster_label",
        color=color_col,
        orientation="h",
        title=title,
        hover_data=[c for c in ["perfil_combinado", "descricao", "partido", "ano", "uf", "nm_municipio"] if c in work.columns],
    )
    if value_col.startswith("share"):
        fig.update_xaxes(tickformat=".0%")
    fig.update_layout(height=560, template="plotly_white", showlegend=False, margin=dict(l=10, r=10, t=56, b=20))
    st.plotly_chart(fig, use_container_width=True)


def log_panel(payload: dict[str, Any]) -> None:
    eventos = payload.get("eventos") or {}
    if eventos:
        st.markdown("#### Evento atual")
        cols = st.columns(min(3, max(1, len(eventos))))
        for idx, (name, data) in enumerate(eventos.items()):
            with cols[idx % len(cols)]:
                if isinstance(data, dict):
                    render_stat_grid(
                        [
                            ("Arquivo", name, str(data.get("evento") or data.get("label") or "")),
                            ("UF", data.get("uf") or "-", "processo atual"),
                            ("Linhas/s", data.get("linhas_por_segundo") or "-", "velocidade reportada"),
                        ]
                    )
                else:
                    st.json(data)
    logs = payload.get("logs_recentes") or payload.get("arquivos") or []
    if not logs:
        st.info("Nenhum log encontrado.")
        return
    st.markdown("#### Logs recentes")
    for item in logs[:6]:
        label = f"{item.get('relativo', item.get('nome'))} | {item.get('modificado_em', '')}"
        with st.expander(label, expanded=False):
            if "json" in item:
                st.json(item["json"])
            else:
                lines = item.get("linhas") or []
                st.markdown("<div class='log-box'>" + escape("\n".join(lines[-120:])) + "</div>", unsafe_allow_html=True)


def main() -> None:
    args = parse_args()
    api_timeout = min(30, max(5, int(args.timeout or 30)))
    st.set_page_config(page_title="Dashboard Eleitoral API", layout="wide")
    inject_style()

    st.markdown(
        """
        <div class="hero-box">
          <div class="hero-kicker">FastAPI + Polars + Streamlit</div>
          <div class="hero-title">Dashboard Eleitoral consultando Parquet direto</div>
          <div class="hero-subtitle">O motor consulta os Parquets pela API; o front mostra cards, graficos, mapa e logs sem embutir tabela gigante no HTML.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("API")
        api_url = st.text_input("URL", value=args.api, label_visibility="collapsed")
        try:
            health = api_get(api_url, "/api/health", timeout=api_timeout)
            progresso = api_get(api_url, "/api/progresso", timeout=api_timeout)
            api_online = True
        except Exception as exc:
            health = {"erro": str(exc)}
            progresso = {}
            api_online = False
        if api_online:
            st.success("Conectada")
        else:
            st.error("Desconectada")
            st.caption(str(health.get("erro", ""))[:180])
        st.caption("Comando para iniciar API")
        st.markdown(
            "<div class='command-box'>python3 scripts/dashboard_api_eleitoral.py --run dados/banco_eleitoral --host 0.0.0.0 --port 8055 --engine polars</div>",
            unsafe_allow_html=True,
        )

    modalidade_labels = {
        "completa": "Completa",
        "estados_brasil": "Estados + Brasil",
        "eleitor": "Somente eleitor",
        "candidato": "Somente candidato",
        "eleitor_partido": "Eleitor + partido",
        "eleitor_candidato_partido": "Eleitor + candidato + partido",
    }
    resumo = progresso.get("ouro_resultados") or {}
    uf_options = sorted(set((resumo.get("ufs_concluidas") or []) + (resumo.get("ufs_pendentes") or [])))
    st.markdown(
        "<div class='control-panel'><div class='control-title'>Central de consulta</div>"
        "<div class='control-subtitle'>Escolha o recorte e o painel carrega direto dos Parquets processados. As consultas do front param em 30 segundos.</div></div>",
        unsafe_allow_html=True,
    )
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1.4, 1, .85, .75])
    modalidade = ctrl1.selectbox("Analise", list(modalidade_labels.keys()), format_func=lambda key: modalidade_labels.get(key, key), index=1)
    uf = ctrl2.selectbox("Estado", [""] + uf_options, index=0, format_func=lambda value: "Brasil" if value == "" else value)
    ano = ctrl3.selectbox("Ano", ["", "2014", "2018", "2022", "2024"], index=0, format_func=lambda value: "Todos" if value == "" else value)
    limit = ctrl4.slider("Itens", 5, 50, 20)
    cenario = "base"
    municipios = []
    if uf:
        municipios = safe_api_get(api_url, "/api/municipios", {"uf": uf, "modalidade": modalidade}, timeout=api_timeout).get("municipios", [])
    municipio_labels = [""] + [m["label"] for m in municipios]
    municipio_values = {m["label"]: m["value"] for m in municipios}
    municipio_label = ""
    municipio = ""
    if municipios:
        municipio_label = st.selectbox("Municipio", municipio_labels, index=0, format_func=lambda value: "Escolha um municipio" if value == "" else value)
        municipio = municipio_values.get(municipio_label, "")

    resumo = progresso.get("ouro_resultados") or {}
    render_stat_grid(
        [
            ("Tabelas", len(health.get("tabelas") or []), "camadas detectadas"),
            ("Resultados", f"{resumo.get('concluidas', 0)}/{resumo.get('total', 0)}", "fatias UF/ano concluidas"),
            ("Pendentes", resumo.get("pendentes", 0), "ainda processando"),
            ("Engine", health.get("engine", "-"), "consulta pela API"),
        ]
    )

    render_quick_actions()

    tabs = st.tabs(["Brasil", "Estados", "Municipios", "Rodar analise", "Clusters/Perfis", "Progresso e logs", "Consulta", "PDF"])

    with tabs[0]:
        st.subheader("Brasil")
        status_box = st.empty()
        if api_online:
            status_box.info("Carregando as 3 tabelas Brasil da camada ouro...")
            params = tuple(sorted({"limit": min(limit, 50)}.items()))
            try:
                payload = cached_api_get(api_url, "/api/brasil/tabelas", params, api_timeout)
                status_box.success("Brasil carregado direto de ouro/brasil.")
                render_brasil_tabelas(payload)
            except Exception as exc:
                status_box.warning(f"Nao consegui carregar automaticamente: {exc}")
                if st.button("Tentar carregar Brasil novamente", key="buscar_brasil"):
                    payload = safe_api_get(api_url, "/api/brasil/tabelas", {"limit": min(limit, 50)}, timeout=api_timeout)
                    render_brasil_tabelas(payload)
        else:
            st.info("Inicie a API para carregar a visao nacional.")

    with tabs[1]:
        st.subheader("Estados")
        col_map, col_state = st.columns([1.15, 1])
        with col_map:
            if st.button("Atualizar mapa dos estados", key="buscar_mapa"):
                with st.spinner("Consultando mapa via API..."):
                    mapa = api_get(api_url, "/api/mapa/estados", {"ano": ano, "cenario": cenario, "modalidade": modalidade, "limit": 80})
                state_map_chart(df_from_payload(mapa))
            else:
                st.info("Clique para carregar o mapa com os dados ja processados.")
        with col_state:
            if st.button("Buscar estado selecionado", key="buscar_estado"):
                if not uf:
                    st.warning("Selecione um estado.")
                else:
                    payload = api_get(api_url, "/api/partidos", {"escopo": "estado", "uf": uf, "ano": ano, "cenario": cenario, "modalidade": modalidade, "limit": limit})
                    render_stat_grid(
                        [
                            ("Estado", uf, "UF selecionada"),
                            ("Fonte", payload.get("fonte", "-"), "origem do dado"),
                            ("Partidos", len(payload.get("dados") or []), "retorno do ranking"),
                        ]
                    )
                    party_chart(df_from_payload(payload), f"{uf} por partido")
                    perf = api_get(api_url, "/api/perfis", {"nivel": "estado", "uf": uf, "ano": ano, "modalidade": modalidade, "limit": limit})
                    profile_cards(df_from_payload(perf), max_cards=6)
                    if modalidade in {"candidato", "eleitor_candidato_partido", "completa"}:
                        cand = api_get(api_url, "/api/candidatos", {"escopo": "estado", "tipo": "perfil", "uf": uf, "ano": ano, "modalidade": modalidade, "limit": limit})
                        st.markdown("### Candidatos")
                        st.dataframe(df_from_payload(cand), use_container_width=True, height=260)
            else:
                st.info("Selecione um estado na lateral. Os municipios aparecem automaticamente no filtro.")

    with tabs[2]:
        st.subheader("Municipios")
        if st.button("Buscar municipio selecionado", key="buscar_municipio"):
            if not uf or not municipio:
                st.warning("Selecione UF e municipio.")
            else:
                payload = api_get(api_url, "/api/partidos", {"escopo": "municipio", "uf": uf, "municipio": municipio, "ano": ano, "cenario": cenario, "modalidade": modalidade, "limit": limit})
                render_stat_grid(
                    [
                        ("UF", uf, "estado"),
                        ("Municipio", municipio_label, "municipio selecionado"),
                        ("Fonte", payload.get("fonte", "-"), "origem do dado"),
                    ]
                )
                party_chart(df_from_payload(payload), f"{municipio_label} por partido")
                perf = api_get(api_url, "/api/perfis", {"nivel": "municipio", "uf": uf, "municipio": municipio, "ano": ano, "modalidade": modalidade, "limit": limit})
                st.markdown("### Perfil do eleitor no municipio")
                profile_cards(df_from_payload(perf), max_cards=8)
                if modalidade in {"candidato", "eleitor_candidato_partido", "completa"}:
                    cand = api_get(api_url, "/api/candidatos", {"escopo": "municipio", "tipo": "perfil", "uf": uf, "municipio": municipio, "ano": ano, "modalidade": modalidade, "limit": limit})
                    st.markdown("### Candidatos")
                    st.dataframe(df_from_payload(cand), use_container_width=True, height=260)
        else:
            st.info("Escolha UF e municipio na lateral e clique para consultar.")

    with tabs[3]:
        st.subheader("Rodar analise do banco ouro")
        st.caption("A API dispara o mesmo pipeline em background, grava log proprio e reaproveita o que ja existe com --resume.")
        analysis_preset = st.selectbox(
            "Preset",
            [
                "estados_brasil_rapido",
                "eleitor_partido_rapido",
                "candidato_rapido",
                "eleitor_candidato_partido",
                "completa_segura",
                "personalizado",
            ],
            format_func=lambda value: {
                "estados_brasil_rapido": "Estados + Brasil rapido",
                "eleitor_partido_rapido": "Eleitor + partido rapido",
                "candidato_rapido": "Candidato rapido",
                "eleitor_candidato_partido": "Eleitor + candidato + partido",
                "completa_segura": "Completa segura",
                "personalizado": "Personalizado",
            }.get(value, value),
        )
        preset_defaults = {
            "estados_brasil_rapido": {
                "modalidade": "estados_brasil",
                "somente_estados_brasil": True,
                "max_municipios": 0,
                "cenarios": 100,
                "skip_heavy": True,
                "skip_clusters": True,
                "workers": 1,
                "threads": 1,
            },
            "eleitor_partido_rapido": {
                "modalidade": "eleitor_partido",
                "somente_estados_brasil": False,
                "max_municipios": 20,
                "cenarios": 50,
                "skip_heavy": True,
                "skip_clusters": True,
                "workers": 1,
                "threads": 1,
            },
            "candidato_rapido": {
                "modalidade": "candidato",
                "somente_estados_brasil": False,
                "max_municipios": 20,
                "cenarios": 50,
                "skip_heavy": True,
                "skip_clusters": True,
                "workers": 1,
                "threads": 1,
            },
            "eleitor_candidato_partido": {
                "modalidade": "eleitor_candidato_partido",
                "somente_estados_brasil": False,
                "max_municipios": 20,
                "cenarios": 100,
                "skip_heavy": True,
                "skip_clusters": True,
                "workers": 1,
                "threads": 1,
            },
            "completa_segura": {
                "modalidade": "completa",
                "somente_estados_brasil": False,
                "max_municipios": 0,
                "cenarios": 3000,
                "skip_heavy": False,
                "skip_clusters": False,
                "workers": 1,
                "threads": 1,
            },
            "personalizado": {
                "modalidade": modalidade,
                "somente_estados_brasil": False,
                "max_municipios": 20,
                "cenarios": 100,
                "skip_heavy": True,
                "skip_clusters": True,
                "workers": 1,
                "threads": 1,
            },
        }
        defaults = preset_defaults[analysis_preset]
        modalidade_keys = list(modalidade_labels.keys())
        selected_mode = st.selectbox(
            "Modalidade da analise",
            modalidade_keys,
            index=modalidade_keys.index(defaults["modalidade"]) if defaults["modalidade"] in modalidade_keys else 0,
            format_func=lambda key: modalidade_labels.get(key, key),
            key="analysis_mode_select",
        )
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            ufs_job = st.text_input("UFs, separadas por virgula", value=uf, key="analysis_ufs")
            somente_estados = st.checkbox("Somente estados + Brasil", value=bool(defaults["somente_estados_brasil"]), key="analysis_only_states")
            skip_heavy = st.checkbox("Pular analises pesadas", value=bool(defaults["skip_heavy"]), key="analysis_skip_heavy")
        with col_b:
            max_municipios_job = st.number_input("Max municipios/UF (0 = todos)", min_value=0, max_value=10000, value=int(defaults["max_municipios"]), step=5)
            cenarios_job = st.number_input("Cenarios 2026", min_value=0, max_value=100000, value=int(defaults["cenarios"]), step=50)
            predict_2026 = st.checkbox("Gerar simulacao 2026", value=True, key="analysis_predict")
        with col_c:
            skip_clusters = st.checkbox("Pular clusters", value=bool(defaults["skip_clusters"]), key="analysis_skip_clusters")
            workers_job = st.number_input("Workers ouro", min_value=1, max_value=64, value=int(defaults["workers"]), step=1)
            threads_job = st.number_input("DuckDB threads", min_value=1, max_value=64, value=int(defaults["threads"]), step=1)
        col_k1, col_k2, col_ag = st.columns(3)
        cluster_min_k = col_k1.number_input("Cluster min K", min_value=2, max_value=50, value=2, step=1)
        cluster_max_k = col_k2.number_input("Cluster max K", min_value=2, max_value=80, value=10, step=1)
        paralelo_agressivo = col_ag.checkbox("Paralelismo agressivo", value=False)
        if st.button("Iniciar analise pela API"):
            payload = {
                "modalidade_analise": selected_mode,
                "ufs": ufs_job,
                "somente_estados_brasil": bool(somente_estados),
                "max_municipios_por_uf": int(max_municipios_job),
                "cenarios": int(cenarios_job),
                "cluster_min_k": int(cluster_min_k),
                "cluster_max_k": int(cluster_max_k),
                "banco_ouro_workers": int(workers_job),
                "banco_duckdb_threads": int(threads_job),
                "skip_heavy_analyses": bool(skip_heavy),
                "skip_clusters": bool(skip_clusters),
                "predict_2026": bool(predict_2026),
                "paralelo_agressivo": bool(paralelo_agressivo),
            }
            job = api_post(api_url, "/api/analises/jobs", payload).get("job", {})
            st.success(f"Analise iniciada: {job.get('id')}")
            st.json(job)
        col_jobs, col_log = st.columns([1, 1])
        with col_jobs:
            if st.button("Listar jobs de analise"):
                st.json(api_get(api_url, "/api/analises/jobs"))
        with col_log:
            job_id = st.text_input("Job ID da analise", value="", key="analysis_job_id")
            if st.button("Ver log da analise"):
                if not job_id.strip():
                    st.warning("Informe o Job ID.")
                else:
                    payload = api_get(api_url, f"/api/analises/jobs/{job_id.strip()}/logs", {"max_lines": 220})
                    st.json(payload.get("job") or {})
                    st.markdown("<div class='log-box'>" + escape("\n".join(payload.get("linhas") or [])) + "</div>", unsafe_allow_html=True)

    with tabs[4]:
        st.subheader("Clusters e perfis discretos")
        nivel = st.radio("Nivel", ["brasil", "estado", "municipio"], horizontal=True)
        if st.button("Buscar perfis/clusters", key="buscar_perfis"):
            params: dict[str, Any] = {"nivel": nivel, "ano": ano, "modalidade": modalidade, "limit": limit}
            if nivel in {"estado", "municipio"}:
                params["uf"] = uf
            if nivel == "municipio":
                params["municipio"] = municipio
            payload = api_get(api_url, "/api/perfis", params)
            profile_cards(df_from_payload(payload), max_cards=12)
            cluster_params = dict(params)
            voter_clusters = api_get(api_url, "/api/clusters", {**cluster_params, "tipo": "eleitores"})
            result_clusters = api_get(api_url, "/api/clusters", {**cluster_params, "tipo": "resultado"})
            voter_df = df_from_payload(voter_clusters)
            result_df = df_from_payload(result_clusters)
            cluster_cards(voter_df, "Clusters somente eleitorado discreto", max_cards=8)
            cluster_chart(voter_df, "Peso dos clusters de eleitorado")
            cluster_cards(result_df, "Clusters eleitorado + partido", max_cards=8)
            cluster_chart(result_df, "Peso dos clusters com resultado")
            dist = api_get(api_url, "/api/brasil", {"ano": ano, "cenario": cenario, "modalidade": modalidade, "limit": 30})
            dist_df = df_from_payload(dist, "perfil_discreto")
            if not dist_df.empty and {"dimensao_perfil", "valor_perfil", "peso"}.issubset(dist_df.columns):
                fig = px.bar(
                    dist_df.head(30),
                    x="peso",
                    y="valor_perfil",
                    color="dimensao_perfil",
                    orientation="h",
                    title="Distribuicao discreta dos perfis",
                )
                fig.update_layout(height=560, template="plotly_white", margin=dict(l=10, r=10, t=56, b=20))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Clique para consultar os perfis discretos ja consolidados.")

    with tabs[5]:
        st.subheader("Progresso e logs")
        if st.button("Atualizar logs", key="buscar_logs"):
            payload = api_get(api_url, "/api/processamento", {"max_files": 10, "max_lines": 120})
        else:
            payload = {"resumo": {"ouro_resultados": resumo}, "eventos": {}, "logs_recentes": []}
        resumo_logs = (payload.get("resumo") or {}).get("ouro_resultados") or resumo
        render_stat_grid(
            [
                ("Total", resumo_logs.get("total", 0), "fatias planejadas"),
                ("Concluidas", resumo_logs.get("concluidas", 0), "ja gravadas"),
                ("Pendentes", resumo_logs.get("pendentes", 0), "restantes"),
                ("UFs pendentes", len(resumo_logs.get("ufs_pendentes") or []), "estados ainda abertos"),
            ]
        )
        log_panel(payload)

    with tabs[6]:
        st.subheader("Consulta de tabelas tratadas")
        try:
            tables_payload = api_get(api_url, "/api/tabelas")
            tables = tables_payload.get("tabelas") or []
        except Exception as exc:
            st.error(f"Erro lendo tabelas: {exc}")
            tables = []
        table_label_to_key = {t["label"]: t["key"] for t in tables}
        table_label = st.selectbox("Tabela", list(table_label_to_key.keys()) or [""])
        table_limit = st.number_input("Linhas", min_value=10, max_value=1000, value=100, step=10)
        if st.button("Buscar tabela"):
            key = table_label_to_key.get(table_label, "")
            payload = api_get(api_url, "/api/tabela", {"key": key, "limit": int(table_limit)})
            st.dataframe(df_from_payload(payload), use_container_width=True, height=520)

    with tabs[7]:
        st.subheader("Gerar PDF pela API")
        out = st.text_input("Arquivo de saida", value="")
        log_dir = st.text_input("Pasta de logs", value="")
        pdf_cols = st.columns(4)
        max_pages = pdf_cols[0].number_input("Max paginas", min_value=1, max_value=5000, value=300)
        top_n = pdf_cols[1].number_input("Top N", min_value=1, max_value=200, value=15)
        municipios_por_uf = pdf_cols[2].number_input("Municipios/UF", min_value=0, max_value=200, value=5)
        duckdb_threads = pdf_cols[3].number_input("DuckDB threads", min_value=1, max_value=32, value=2)
        incluir_secoes = st.checkbox("Incluir secoes eleitorais", value=False)
        separado_por_nivel = st.checkbox("Gerar PDFs separados: Brasil, estados e municipios", value=True)
        ufs_pdf = st.text_input("UFs do PDF, separadas por virgula", value=uf)
        if st.button("Disparar PDF"):
            payload = {
                "out": out,
                "log_dir": log_dir,
                "modalidade_analise": modalidade,
                "max_pages": int(max_pages),
                "top_n": int(top_n),
                "ufs": ufs_pdf,
                "municipios_por_uf": int(municipios_por_uf),
                "incluir_secoes": bool(incluir_secoes),
                "duckdb_threads": int(duckdb_threads),
                "engine": "polars",
                "separado_por_nivel": bool(separado_por_nivel),
            }
            job = api_post(api_url, "/api/pdf/jobs", payload).get("job", {})
            st.success(f"Job criado: {job.get('id')}")
            st.json(job)
        if st.button("Listar jobs PDF"):
            st.json(api_get(api_url, "/api/pdf/jobs"))
        job_id_logs = st.text_input("Job ID para ver logs", value="")
        if st.button("Ver logs do PDF"):
            if not job_id_logs.strip():
                st.warning("Informe o Job ID.")
            else:
                pdf_logs = api_get(api_url, f"/api/pdf/jobs/{job_id_logs.strip()}/logs", {"max_files": 10, "max_lines": 120})
                log_panel(pdf_logs)


if __name__ == "__main__":
    main()
