from __future__ import annotations

from pathlib import Path
from typing import Any
import html

import pandas as pd

from .utils import save_html, safe_text


def read_csv_if_exists(path_value: str) -> pd.DataFrame:
    if not path_value:
        return pd.DataFrame()
    path = Path(path_value)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep=";", dtype=str)
    except Exception:
        return pd.DataFrame()


def update_global_dashboard_with_simulation(cfg, pred_info: dict[str, Any]) -> str:
    html_path = Path(cfg.out) / "global" / "relatorio_global.html"
    if not html_path.exists() or not pred_info or pred_info.get("status") != "ok":
        return ""

    content = html_path.read_text(encoding="utf-8")
    section = build_simulation_dashboard_section(pred_info)
    if 'data-panel="simulacao"' not in content:
        content = content.replace(
            '<button data-panel="consulta" onclick="showPanel(\'consulta\', this)">Tabelas</button>',
            '<button data-panel="simulacao" onclick="showPanel(\'simulacao\', this)">Simulacao</button>\n'
            '      <button data-panel="consulta" onclick="showPanel(\'consulta\', this)">Tabelas</button>',
        )
    if 'id="simulacao"' not in content:
        content = content.replace('<section id="consulta" class="panel">', section + '\n\n    <section id="consulta" class="panel">')
    else:
        start = content.find('<section id="simulacao"')
        end = content.find('<section id="consulta"', start)
        if start >= 0 and end > start:
            content = content[:start] + section + "\n\n    " + content[end:]

    html_path.write_text(content, encoding="utf-8")
    return str(html_path)


def build_simulation_dashboard_section(pred_info: dict[str, Any]) -> str:
    nacional = read_csv_if_exists(pred_info.get("cenarios_nacionais_csv", ""))
    mc = read_csv_if_exists(pred_info.get("monte_carlo_csv", ""))
    decisive = read_csv_if_exists(pred_info.get("secoes_municipios_decisivos_csv", ""))
    party_br = read_csv_if_exists(pred_info.get("partidos_2026_brasil_csv", ""))
    party_uf = read_csv_if_exists(pred_info.get("partidos_2026_estados_csv", ""))
    party_mun = read_csv_if_exists(pred_info.get("partidos_2026_municipios_csv", ""))

    top_cards = simulation_candidate_cards(nacional)
    party_br_cards = party_prediction_cards(party_br, "Brasil")
    party_uf_cards = party_prediction_cards(party_uf, "Estados")
    party_mun_cards = party_prediction_cards(party_mun, "Municipios")
    mc_cards = monte_carlo_cards(mc)
    decisive_cards = decisive_area_cards(decisive)
    sim_html = html.escape(str(pred_info.get("html", "")))
    return f"""
    <section id="simulacao" class="panel">
      <div class="grid">
        <div class="card wide">
          <h3>Simulacao 2026 por partido - Brasil</h3>
          <div class="persona-list">{party_br_cards}</div>
        </div>
        <div class="card wide">
          <h3>Simulacao 2026 por partido - estados</h3>
          <div class="persona-list">{party_uf_cards}</div>
        </div>
        <div class="card wide">
          <h3>Simulacao 2026 por partido - municipios</h3>
          <div class="persona-list">{party_mun_cards}</div>
        </div>
        <div class="card wide">
          <h3>Simulacao 2026 - cenario nacional</h3>
          <div class="persona-list">{top_cards}</div>
          <p class="smallnote">Relatorio completo da simulacao: {sim_html}</p>
        </div>
        <div class="card wide">
          <h3>Monte Carlo</h3>
          <div class="persona-list">{mc_cards}</div>
        </div>
        <div class="card wide">
          <h3>Municipios e secoes decisivos</h3>
          <div class="persona-list">{decisive_cards}</div>
        </div>
      </div>
    </section>"""


def simulation_candidate_cards(nacional: pd.DataFrame, limit: int = 18) -> str:
    if nacional is None or nacional.empty:
        return "<p class='smallnote'>Simulacao nacional sem dados.</p>"
    df = nacional.copy()
    if "cenario" in df.columns and df["cenario"].astype(str).eq("base").any():
        df = df.loc[df["cenario"].astype(str).eq("base")]
    df["share_nacional_pred_2026"] = pd.to_numeric(df.get("share_nacional_pred_2026"), errors="coerce")
    df["votos_pred_2026"] = pd.to_numeric(df.get("votos_pred_2026"), errors="coerce")
    df = df.loc[df["entidade"].map(_meaningful).ne("")]
    df = df.sort_values("share_nacional_pred_2026", ascending=False).head(limit)
    cards = []
    for _, r in df.iterrows():
        cards.append(
            "<div class='persona'>"
            f"<h4>{html.escape(safe_text(r.get('entidade', '')))}</h4>"
            f"<p><span class='pill'>{_pct(r.get('share_nacional_pred_2026'))}</span>"
            f"<span class='pill'>Votos: {_int(r.get('votos_pred_2026'))}</span></p>"
            f"<p class='smallnote'>{html.escape(safe_text(r.get('cargo', '')))} | turno {html.escape(safe_text(r.get('turno', '')))}</p>"
            "</div>"
        )
    return "".join(cards) or "<p class='smallnote'>Sem candidato/entidade valido para cards.</p>"


def party_prediction_cards(df: pd.DataFrame, level_label: str, limit: int = 18) -> str:
    if df is None or df.empty:
        return f"<p class='smallnote'>Sem simulacao partidaria para {html.escape(level_label)}.</p>"
    work = df.copy()
    if "cenario" in work.columns and work["cenario"].astype(str).eq("base").any():
        work = work.loc[work["cenario"].astype(str).eq("base")].copy()
    work["share_pred_2026"] = pd.to_numeric(work.get("share_pred_2026"), errors="coerce")
    if "partido" not in work.columns:
        return f"<p class='smallnote'>Tabela partidaria sem coluna partido para {html.escape(level_label)}.</p>"
    work = work.loc[work["partido"].map(_meaningful).ne("")]
    work = work.sort_values("share_pred_2026", ascending=False).head(limit)
    cards = []
    for _, r in work.iterrows():
        local = " / ".join(x for x in [
            _meaningful(r.get("uf", "")),
            _meaningful(r.get("nm_municipio", "")) or _meaningful(r.get("cd_municipio", "")),
        ] if x)
        cards.append(
            "<details class='persona'>"
            f"<summary>{html.escape(_meaningful(r.get('partido', '')))} - {_pct(r.get('share_pred_2026'))}</summary>"
            f"<p>{html.escape(_meaningful(r.get('perfil_eleitor_2026', '')) or 'Perfil de eleitor ainda insuficiente neste recorte.')}</p>"
            f"<p><span class='pill'>{html.escape(local or level_label)}</span><span class='pill'>{html.escape(_meaningful(r.get('tendencia_partido', '')))}</span><span class='pill'>{html.escape(_meaningful(r.get('forca_correlacao_historica', '')))}</span></p>"
            f"<p class='smallnote'>{html.escape(_meaningful(r.get('justificativa_previsao_partido_2026', '')) or _meaningful(r.get('justificativa_correlacao', '')))}</p>"
            "</details>"
        )
    return "".join(cards) or f"<p class='smallnote'>Sem partidos validos para {html.escape(level_label)}.</p>"


def monte_carlo_cards(mc: pd.DataFrame, limit: int = 12) -> str:
    if mc is None or mc.empty:
        return "<p class='smallnote'>Monte Carlo sem dados.</p>"
    df = mc.copy()
    if "cenario" in df.columns and df["cenario"].astype(str).eq("base").any():
        df = df.loc[df["cenario"].astype(str).eq("base")]
    df["share_medio"] = pd.to_numeric(df.get("share_medio"), errors="coerce")
    df = df.loc[df["entidade"].map(_meaningful).ne("")]
    df = df.sort_values("share_medio", ascending=False).head(limit)
    cards = []
    for _, r in df.iterrows():
        cards.append(
            "<details class='persona'>"
            f"<summary>{html.escape(safe_text(r.get('entidade', '')))} - {_pct(r.get('share_medio'))}</summary>"
            f"<p><span class='pill'>P05 {_pct(r.get('share_p05'))}</span><span class='pill'>P50 {_pct(r.get('share_p50'))}</span><span class='pill'>P95 {_pct(r.get('share_p95'))}</span></p>"
            f"<p class='smallnote'>Simulacoes: {_int(r.get('n_simulacoes'))}</p>"
            "</details>"
        )
    return "".join(cards) or "<p class='smallnote'>Sem Monte Carlo valido para cards.</p>"


def decisive_area_cards(decisive: pd.DataFrame, limit: int = 18) -> str:
    if decisive is None or decisive.empty:
        return "<p class='smallnote'>Sem municipios decisivos calculados.</p>"
    df = decisive.copy()
    df["indice_decisivo"] = pd.to_numeric(df.get("indice_decisivo"), errors="coerce")
    df = df.sort_values("indice_decisivo", ascending=False).head(limit)
    cards = []
    for _, r in df.iterrows():
        indice = pd.to_numeric(r.get("indice_decisivo"), errors="coerce")
        indice_val = float(indice) if pd.notna(indice) else 0.0
        local = " / ".join(x for x in [
            safe_text(r.get("uf", "")),
            safe_text(r.get("nm_municipio", "")) or safe_text(r.get("cd_municipio", "")),
            safe_text(r.get("zona", "")),
            safe_text(r.get("secao", "")),
        ] if x)
        cards.append(
            "<details class='persona'>"
            f"<summary>{html.escape(local or 'Local sem nome')}</summary>"
            f"<p>Lider: {html.escape(safe_text(r.get('lider_pred', '')))}</p>"
            f"<p>Segundo: {html.escape(safe_text(r.get('segundo_pred', '')))}</p>"
            f"<p class='smallnote'>Indice decisivo: {indice_val:.3f}</p>"
            "</details>"
        )
    return "".join(cards)


def generate_executive_report(cfg, global_info: dict[str, Any], pred_info: dict[str, Any]) -> dict[str, str]:
    out_dir = Path(cfg.out) / "relatorio_executivo"
    out_dir.mkdir(parents=True, exist_ok=True)
    nacional = read_csv_if_exists(pred_info.get("cenarios_nacionais_csv", ""))
    party_br = read_csv_if_exists(pred_info.get("partidos_2026_brasil_csv", ""))
    municipal = read_csv_if_exists(global_info.get("municipal_csv", ""))
    profile_party = read_csv_if_exists((global_info.get("analise_eleitoral_outputs") or {}).get("perfil_eleitor_por_partido", ""))

    md = executive_markdown(nacional, municipal, profile_party, party_br)
    md_path = out_dir / "relatorio_eleicoes_simulacao.md"
    html_path = out_dir / "relatorio_eleicoes_simulacao.html"
    pdf_path = out_dir / "relatorio_eleicoes_simulacao.pdf"
    md_path.write_text(md, encoding="utf-8")
    save_html(html_path, "Relatorio executivo - eleicoes e simulacao", "<pre>" + html.escape(md) + "</pre>")
    pdf_ok = write_pdf_if_possible(pdf_path, md)
    return {
        "md": str(md_path),
        "html": str(html_path),
        "pdf": str(pdf_path) if pdf_ok else "",
    }


def executive_markdown(nacional: pd.DataFrame, municipal: pd.DataFrame, profile_party: pd.DataFrame, party_br: pd.DataFrame | None = None) -> str:
    lines = ["# Relatorio executivo eleitoral", ""]
    if municipal is not None and not municipal.empty:
        lines.append(f"- Municipios analisados: {municipal.get('cd_municipio', pd.Series(dtype=str)).replace('', pd.NA).dropna().nunique()}")
        lines.append(f"- Votos agregados: {_int(pd.to_numeric(municipal.get('votos'), errors='coerce').sum())}")
    if profile_party is not None and not profile_party.empty and "status" not in profile_party.columns:
        lines.append("")
        lines.append("## Quem vota por partido")
        profile_party = profile_party.copy()
        profile_party["votos_partido_num"] = pd.to_numeric(profile_party.get("votos_partido"), errors="coerce")
        for _, r in profile_party.sort_values("votos_partido_num", ascending=False, na_position="last").head(10).iterrows():
            lines.append(f"- {r.get('partido')}: {r.get('pessoa_do_partido')} ({_int(r.get('votos_partido'))} votos)")
    if nacional is not None and not nacional.empty:
        lines.append("")
        lines.append("## Simulacao 2026 - cenario base")
        df = nacional.loc[nacional["cenario"].astype(str).eq("base")].copy() if "cenario" in nacional.columns else nacional.copy()
        df["share_nacional_pred_2026"] = pd.to_numeric(df.get("share_nacional_pred_2026"), errors="coerce")
        for _, r in df.loc[df["entidade"].map(_meaningful).ne("")].sort_values("share_nacional_pred_2026", ascending=False).head(10).iterrows():
            lines.append(f"- {r.get('entidade')}: {_pct(r.get('share_nacional_pred_2026'))}")
    if party_br is not None and not party_br.empty and "partido" in party_br.columns:
        lines.append("")
        lines.append("## Simulacao 2026 por partido - Brasil")
        df = party_br.loc[party_br["cenario"].astype(str).eq("base")].copy() if "cenario" in party_br.columns else party_br.copy()
        df["share_pred_2026"] = pd.to_numeric(df.get("share_pred_2026"), errors="coerce")
        for _, r in df.loc[df["partido"].map(_meaningful).ne("")].sort_values("share_pred_2026", ascending=False).head(10).iterrows():
            lines.append(f"- {r.get('partido')}: {_pct(r.get('share_pred_2026'))}; perfil: {r.get('perfil_eleitor_2026', '')}; historico: {r.get('forca_correlacao_historica', '')}")
    lines.append("")
    lines.append("Nota: perfis sao agregados/ecologicos; nao sao voto individual declarado.")
    return "\n".join(lines)


def write_pdf_if_possible(path: Path, markdown: str) -> bool:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
    except Exception:
        return False
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(path), pagesize=A4)
    story = []
    for line in markdown.splitlines():
        text = html.escape(line if line else " ")
        style = styles["Heading1"] if line.startswith("# ") else styles["Heading2"] if line.startswith("## ") else styles["BodyText"]
        story.append(Paragraph(text.lstrip("# "), style))
        story.append(Spacer(1, 6))
    doc.build(story)
    return True


def _meaningful(value: Any) -> str:
    text = safe_text(value, "").strip()
    lower = text.lower()
    code_value = lower.replace("codigo ", "", 1).replace("código ", "", 1).replace(".", "", 1).lstrip("-+")
    if lower in {"", "sem valor", "sem_valor", "nan", "none", "null", "<na>", "#nulo#", "geral", "nao informado", "não informado", "sem_entidade"}:
        return ""
    if lower.endswith("_sem_valor") or lower.endswith(" sem valor"):
        return ""
    if (lower.startswith("codigo ") or lower.startswith("código ")) and code_value.isdigit():
        return ""
    return text


def _pct(value: Any) -> str:
    num = pd.to_numeric(value, errors="coerce")
    if pd.isna(num):
        return "sem dado"
    return f"{float(num) * 100:.1f}%"


def _int(value: Any) -> str:
    num = pd.to_numeric(value, errors="coerce")
    if pd.isna(num):
        return "0"
    return f"{int(float(num)):,}".replace(",", ".")
