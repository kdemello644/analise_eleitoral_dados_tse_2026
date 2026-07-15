from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
import re

import numpy as np
import pandas as pd

from .utils import (
    compact_code,
    extract_years_from_value,
    hash_short,
    normalize_col,
    parse_number,
    safe_text,
)
from .discrete import is_useful_discrete_series


ALLOWED_METRIC_ROLES = {"votos", "comparecimento", "abstencao"}
IGNORED_CLUSTER_ROLES = {"ano", "data", "hora", "datetime"}


def role_by_name(col: str) -> str:
    c = normalize_col(col)
    if "DATA" in c or c.startswith("DT_") or c in {"DT", "DATE"}:
        return "data"
    if "HORA" in c or c.startswith("HH_") or c in {"HR", "HOUR"}:
        return "hora"

    # Heurísticas fracas e data-driven: servem para sugerir papéis,
    # mas os relatórios sempre mostram o inventário real de campos.
    if "ANO" in c and ("ELEICAO" in c or c == "ANO" or c.startswith("AA_")):
        return "ano"
    if c in {"SG_UF", "UF", "SIGLA_UF"} or c.endswith("_UF"):
        return "uf"
    if "MUNICIPIO" in c or "MUN" in c or "IBGE" in c:
        if c.startswith(("CD_", "COD", "CODIGO")) or "IBGE" in c:
            return "cd_municipio"
        return "nm_municipio"
    if "ZONA" in c or c in {"NR_ZONA", "CD_ZONA"}:
        return "zona"
    if "SECAO" in c or "SEÇÃO" in c or c in {"NR_SECAO", "NR_SEÇÃO"}:
        return "secao"
    if "LOCAL_VOT" in c or "LOCAL_DE_VOT" in c or "LOCALIDADE" in c:
        return "local_votacao"
    if "BAIRRO" in c:
        return "bairro"
    if "TURNO" in c:
        return "turno"
    if "CARGO" in c:
        return "cargo"
    if "PARTIDO" in c or c == "SG_PARTIDO":
        return "partido"
    if "IDEOLOG" in c or "ESPECTRO" in c or "POSICAO_POLITICA" in c or "POSIÇÃO_POLÍTICA" in c:
        return "ideologia"
    if "COLIG" in c or "FEDERACAO" in c or "FEDERAÇÃO" in c:
        return "coalizao"
    if "GRAU" in c or "INSTRU" in c or "ESCOLAR" in c:
        return "perfil_instrucao"
    if (
        "CANDIDATO" in c
        or "URNA" in c
        or "VOTAVEL" in c
        or "VOTÁVEL" in c
        or "VOTÃVEL" in c
    ):
        return "candidato"

    # Perfil do eleitorado
    if "FAIXA_ETARIA" in c or "FAIXA_ETÁRIA" in c or "IDADE" in c:
        return "perfil_faixa_etaria"
    if c in {"SEXO", "GENERO", "GÊNERO", "DS_GENERO", "DS_GÊNERO"} or "GENERO" in c or "GÊNERO" in c:
        return "perfil_genero"
    if "GRAU_INSTRUCAO" in c or "GRAU_INSTRUÇÃO" in c or "ESCOLARIDADE" in c or "INSTRUCAO" in c or "INSTRUÇÃO" in c:
        return "perfil_instrucao"
    if "ESTADO_CIVIL" in c:
        return "perfil_estado_civil"
    if "RACA" in c or "RAÇA" in c or "COR_" in c:
        return "perfil_raca_cor"
    if "BIOMETRIA" in c or "BIOMETRICO" in c or "BIOMÉTRICO" in c:
        return "perfil_biometria"

    # Campos de texto/engajamento só entram se existirem nos JSONs.
    if "SENTIMENT" in c or "SENTIMENTO" in c:
        return "sentimento"
    if "ENGAJ" in c or "LIKE" in c or "COMENT" in c or "SHARE" in c or "COMPART" in c or "REACAO" in c or "REAÇÃO" in c:
        return "engajamento"
    if "TEXTO" in c or "COMENTARIO" in c or "COMENTÁRIO" in c or "MENSAGEM" in c:
        return "texto_politico"

    if "VOTO" in c and "VALID" not in c and "BRANC" not in c and "NUL" not in c:
        return "votos"
    if "ELEITORADO" in c or "ELEITOR" in c or "APTOS" in c:
        return "eleitorado"
    if "COMPAREC" in c:
        return "comparecimento"
    if "ABST" in c:
        return "abstencao"
    if "BRANC" in c:
        return "brancos"
    if "NUL" in c:
        return "nulos"
    if "VALID" in c or "VÁLID" in c:
        return "validos"

    return ""


def is_code_like(col: str) -> bool:
    c = normalize_col(col)
    return (
        c.startswith(("CD_", "NR_", "SQ_", "ID_", "TP_"))
        or c.endswith("_ID")
        or c in {"ID", "CODIGO", "CÓDIGO"}
    )


def infer_column_profile(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    if df is None or df.empty:
        return pd.DataFrame()

    n_rows = len(df)
    for col in df.columns:
        s = df[col]
        ss = s.astype(str).str.strip()
        null_mask = s.isna() | ss.eq("") | ss.str.lower().isin({"nan", "none", "null", "<na>"})
        non_null = s[~null_mask]
        n_non = int(non_null.shape[0])
        nunique = int(non_null.astype(str).nunique(dropna=True)) if n_non else 0

        sample = non_null.head(min(10000, n_non))
        nums = sample.map(parse_number)
        numeric_ratio = float(nums.notna().mean()) if n_non else 0.0

        year_hits = []
        for value in sample.head(2000):
            year_hits.extend(extract_years_from_value(value))
        year_ratio = len(year_hits) / max(len(sample), 1)

        role = role_by_name(col)
        code_like = is_code_like(col)
        discrete_ok = is_useful_discrete_series(s, col=col, role=role)
        if role in IGNORED_CLUSTER_ROLES:
            discrete_ok = False

        if role in {"ano", "uf", "cd_municipio", "nm_municipio", "zona", "secao", "local_votacao", "bairro", "turno", "cargo", "partido", "candidato", "ideologia", "coalizao", "perfil_faixa_etaria", "perfil_genero", "perfil_instrucao", "perfil_estado_civil", "perfil_raca_cor", "perfil_biometria"}:
            kind = "dimensao"
        elif role in {"votos", "eleitorado", "comparecimento", "abstencao", "brancos", "nulos", "validos"} and numeric_ratio >= 0.5:
            kind = "metrica"
        elif year_ratio >= 0.5 and nunique <= 80:
            kind = "ano_detectado_por_valor"
            if not role:
                role = "ano_candidato"
        elif numeric_ratio >= 0.95 and discrete_ok:
            kind = "categorico_discreto"
        elif numeric_ratio >= 0.95 and not code_like:
            kind = "numerico"
        elif code_like:
            kind = "codigo"
        elif nunique <= min(250, max(30, int(n_non * 0.08))):
            kind = "categorico"
        else:
            kind = "texto"

        records.append({
            "coluna": col,
            "role_sugerido": role,
            "tipo_inferido": kind,
            "linhas_amostra": n_rows,
            "nao_nulos": n_non,
            "nulos": int(null_mask.sum()),
            "pct_nulos": float(null_mask.sum() / max(n_rows, 1)),
            "unicos": nunique,
            "ratio_unicos": float(nunique / max(n_non, 1)),
            "numeric_ratio": numeric_ratio,
            "year_ratio": year_ratio,
            "anos_detectados_no_valor": ", ".join(map(str, sorted(set(year_hits))[:30])),
            "parece_codigo": bool(code_like),
            "usar_como_metrica": role in ALLOWED_METRIC_ROLES and kind in {"metrica", "numerico"} and not code_like,
            "usar_como_discreto": bool(discrete_ok),
            "usar_em_clustering": bool(discrete_ok),
        })

    return pd.DataFrame(records)


def canonical_field_name(col: Any) -> str:
    s = normalize_col(col)
    s = re.sub(r"^(QT_|QTD_|CD_|NR_|SQ_|NM_|DS_|SG_|TP_|ID_)", "", s)
    s = s.replace("VOTOS", "VOTO").replace("ELEITORES", "ELEITORADO").replace("ELEITOR", "ELEITORADO")
    s = s.replace("MUNICIPIO", "MUN").replace("MUNICÍPIO", "MUN")
    s = re.sub(r"[^A-Z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def collect_profiles(results: list[dict[str, Any]]) -> pd.DataFrame:
    frames = []
    for r in results:
        if r.get("status") != "ok" or not r.get("perfil_csv"):
            continue
        p = Path(r["perfil_csv"])
        if not p.exists():
            continue
        df = pd.read_csv(p, sep=";", dtype=str)
        df["arquivo_relativo"] = r.get("relativo", "")
        df["arquivo_id"] = hash_short(r.get("relativo", ""))
        df["campo_canonico"] = df["coluna"].map(canonical_field_name)
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def table_similarity(profiles: pd.DataFrame) -> pd.DataFrame:
    if profiles is None or profiles.empty:
        return pd.DataFrame()

    items = []
    for file, g in profiles.groupby("arquivo_relativo", dropna=False):
        fields = set(g["campo_canonico"].dropna().astype(str))
        roles = set(g.get("role_sugerido", pd.Series(dtype=str)).fillna("").astype(str))
        metrics = set(g.loc[g.get("usar_como_metrica", "False").astype(str).isin(["True", "true", "1"]), "campo_canonico"].astype(str))
        items.append({"arquivo": file, "fields": fields, "roles": roles, "metrics": metrics})

    rows = []
    for i, a in enumerate(items):
        for b in items[i+1:]:
            f_inter = a["fields"] & b["fields"]
            f_union = a["fields"] | b["fields"]
            r_inter = a["roles"] & b["roles"]
            r_union = a["roles"] | b["roles"]
            m_inter = a["metrics"] & b["metrics"]
            m_union = a["metrics"] | b["metrics"]
            score = (
                0.55 * len(f_inter) / max(len(f_union), 1)
                + 0.25 * len(r_inter) / max(len(r_union), 1)
                + 0.20 * len(m_inter) / max(len(m_union), 1)
            )
            rows.append({
                "arquivo_1": a["arquivo"],
                "arquivo_2": b["arquivo"],
                "score_similaridade": score,
                "campos_em_comum": len(f_inter),
                "roles_em_comum": len(r_inter),
                "metricas_em_comum": len(m_inter),
                "exemplos_campos_comuns": ", ".join(sorted(f_inter)[:30]),
            })

    out = pd.DataFrame(rows)
    return out.sort_values("score_similaridade", ascending=False) if not out.empty else out


def field_similarity(profiles: pd.DataFrame, max_pairs: int = 70000) -> pd.DataFrame:
    if profiles is None or profiles.empty:
        return pd.DataFrame()

    cols = profiles[["arquivo_relativo", "coluna", "campo_canonico", "role_sugerido", "tipo_inferido"]].drop_duplicates()
    records = []
    rows = cols.to_dict(orient="records")

    for i, a in enumerate(rows):
        for b in rows[i+1:]:
            if a["arquivo_relativo"] == b["arquivo_relativo"] and a["coluna"] == b["coluna"]:
                continue

            same_role = safe_text(a.get("role_sugerido")) and a.get("role_sugerido") == b.get("role_sugerido")
            same_type = safe_text(a.get("tipo_inferido")) and a.get("tipo_inferido") == b.get("tipo_inferido")
            name_ratio = SequenceMatcher(None, str(a.get("campo_canonico", "")), str(b.get("campo_canonico", ""))).ratio()

            if not same_role and not same_type and name_ratio < 0.84:
                continue

            score = 0.65 * name_ratio + 0.25 * float(bool(same_role)) + 0.10 * float(bool(same_type))
            if score >= 0.80:
                records.append({
                    "arquivo_1": a["arquivo_relativo"],
                    "campo_1": a["coluna"],
                    "arquivo_2": b["arquivo_relativo"],
                    "campo_2": b["coluna"],
                    "campo_1_canonico": a["campo_canonico"],
                    "campo_2_canonico": b["campo_canonico"],
                    "role_1": a.get("role_sugerido", ""),
                    "role_2": b.get("role_sugerido", ""),
                    "tipo_1": a.get("tipo_inferido", ""),
                    "tipo_2": b.get("tipo_inferido", ""),
                    "score_similaridade": score,
                    "similaridade_nome": name_ratio,
                    "mesmo_role": bool(same_role),
                    "mesmo_tipo": bool(same_type),
                })
                if len(records) >= max_pairs:
                    break
        if len(records) >= max_pairs:
            break

    out = pd.DataFrame(records)
    return out.sort_values("score_similaridade", ascending=False) if not out.empty else out


def learned_canonical_map(profiles: pd.DataFrame, fields: pd.DataFrame) -> pd.DataFrame:
    if profiles is None or profiles.empty:
        return pd.DataFrame()

    nodes = set(profiles["campo_canonico"].fillna("").astype(str))
    parent = {n: n for n in nodes}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        if not a or not b:
            return
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    if fields is not None and not fields.empty:
        for _, r in fields.iterrows():
            try:
                score = float(r.get("score_similaridade", 0))
            except Exception:
                score = 0.0
            if score >= 0.88:
                union(str(r.get("campo_1_canonico", "")), str(r.get("campo_2_canonico", "")))

    tmp = profiles.copy()
    tmp["grupo_canonico_aprendido"] = tmp["campo_canonico"].map(find)
    out = tmp.groupby("grupo_canonico_aprendido", dropna=False).agg({
        "coluna": lambda s: ", ".join(sorted(set(map(str, s)))[:50]),
        "arquivo_relativo": "nunique",
        "role_sugerido": lambda s: ", ".join(sorted(set([x for x in map(str, s) if safe_text(x)]))[:15]),
        "tipo_inferido": lambda s: ", ".join(sorted(set([x for x in map(str, s) if safe_text(x)]))[:15]),
    }).reset_index().rename(columns={
        "coluna": "campos_originais",
        "arquivo_relativo": "qtd_arquivos",
        "role_sugerido": "roles_detectados",
        "tipo_inferido": "tipos_detectados",
    })
    out["observacao"] = "Mapa aprendido a partir dos JSONs encontrados; não assume schema fechado."
    return out.sort_values(["qtd_arquivos", "grupo_canonico_aprendido"], ascending=[False, True])
