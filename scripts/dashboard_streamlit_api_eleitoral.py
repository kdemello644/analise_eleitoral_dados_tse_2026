from __future__ import annotations

import argparse
import sys
from html import escape
from typing import Any

import pandas as pd
import plotly.express as px
import requests
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
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


def api_get(base_url: str, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    url = base_url.rstrip("/") + path
    response = requests.get(url, params=params or {}, timeout=240)
    response.raise_for_status()
    return response.json()


def api_post(base_url: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = base_url.rstrip("/") + path
    response = requests.post(url, json=payload, timeout=240)
    response.raise_for_status()
    return response.json()


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


def inject_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.25rem; max-width: 1480px;}
        .hero-box {
            padding: 1.35rem 1.5rem;
            border-radius: 10px;
            background: linear-gradient(120deg, #10243d 0%, #116c69 100%);
            color: #fff;
            margin-bottom: 1rem;
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
    if df.empty or "partido" not in df.columns:
        st.info("Sem dados de partido para graficar nesta consulta.")
        return
    work = df.copy()
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
        profile = row.get("perfil_combinado") or row.get("descricao") or row.get("valor_perfil") or "Perfil"
        tags = []
        for col in ["ano", "uf", "nm_municipio", "share_perfil", "eleitorado"]:
            if col in row and pd.notna(row[col]) and row[col] != "":
                tags.append((col, row[col]))
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
        api_url = st.text_input("URL da API", value=args.api)
        try:
            health = api_get(api_url, "/api/health")
            progresso = api_get(api_url, "/api/progresso")
            api_online = True
        except Exception as exc:
            health = {"erro": str(exc)}
            progresso = {}
            api_online = False
        if api_online:
            st.success("API online")
        else:
            st.error(f"API indisponivel: {health.get('erro')}")
        if st.button("Testar API"):
            st.json(health)

        st.header("Filtros")
        modalidade_labels = {
            "completa": "Completa",
            "estados_brasil": "Estados + Brasil",
            "eleitor": "Somente eleitor",
            "candidato": "Somente candidato",
            "eleitor_partido": "Eleitor + partido",
            "eleitor_candidato_partido": "Eleitor + candidato + partido",
        }
        modalidade = st.selectbox(
            "Modalidade",
            list(modalidade_labels.keys()),
            format_func=lambda key: modalidade_labels.get(key, key),
            index=0,
        )
        resumo = progresso.get("ouro_resultados") or {}
        uf_options = sorted(set((resumo.get("ufs_concluidas") or []) + (resumo.get("ufs_pendentes") or [])))
        uf = st.selectbox("Estado", [""] + uf_options, index=0)
        municipios = []
        if uf:
            try:
                municipios = api_get(api_url, "/api/municipios", {"uf": uf, "modalidade": modalidade}).get("municipios", [])
            except Exception as exc:
                st.warning(f"Nao consegui listar municipios: {exc}")
        municipio_labels = [""] + [m["label"] for m in municipios]
        municipio_values = {m["label"]: m["value"] for m in municipios}
        municipio_label = st.selectbox("Municipio", municipio_labels, index=0)
        municipio = municipio_values.get(municipio_label, "")
        ano = st.selectbox("Ano", ["", "2014", "2018", "2022", "2024"], index=0)
        cenario = st.text_input("Cenario", value="base")
        limit = st.slider("Top N", 5, 100, 20)

    resumo = progresso.get("ouro_resultados") or {}
    render_stat_grid(
        [
            ("Tabelas", len(health.get("tabelas") or []), "camadas detectadas"),
            ("Resultados", f"{resumo.get('concluidas', 0)}/{resumo.get('total', 0)}", "fatias UF/ano concluidas"),
            ("Pendentes", resumo.get("pendentes", 0), "ainda processando"),
            ("Engine", health.get("engine", "-"), "consulta pela API"),
        ]
    )

    tabs = st.tabs(["Brasil", "Estados", "Municipios", "Rodar analise", "Clusters/Perfis", "Progresso e logs", "Consulta", "PDF"])

    with tabs[0]:
        st.subheader("Brasil")
        if st.button("Buscar Brasil", key="buscar_brasil"):
            with st.spinner("Consultando Brasil na API..."):
                payload = api_get(api_url, "/api/brasil", {"ano": ano, "cenario": cenario, "modalidade": modalidade, "limit": limit})
            render_stat_grid(
                [
                    ("Fonte partidos", payload.get("fonte_partidos", "-"), "simulacao ou historico"),
                    ("Perfis", len(payload.get("perfis") or []), "cards de eleitor medio"),
                    ("Partidos", len(payload.get("partidos") or []), "linhas retornadas"),
                ]
            )
            party_chart(df_from_payload(payload, "partidos"), "Brasil por partido")
            st.markdown("### Eleitor medio no Brasil")
            profile_cards(df_from_payload(payload, "perfis"))
        else:
            st.info("Clique em Buscar Brasil para carregar os graficos nacionais.")

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
