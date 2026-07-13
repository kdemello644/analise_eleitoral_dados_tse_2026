from __future__ import annotations

from typing import Any
import re
import unicodedata

import pandas as pd

from .utils import normalize_col, parse_number, safe_text


NULL_TOKENS = {"", "nan", "none", "null", "<na>", "na", "n/a", "sem_valor", "nao informado", "não informado"}

CATEGORY_LABELS = {
    "ano": {},
    "uf": {
        "AC": "Acre",
        "AL": "Alagoas",
        "AP": "Amapa",
        "AM": "Amazonas",
        "BA": "Bahia",
        "CE": "Ceara",
        "DF": "Distrito Federal",
        "ES": "Espirito Santo",
        "GO": "Goias",
        "MA": "Maranhao",
        "MT": "Mato Grosso",
        "MS": "Mato Grosso do Sul",
        "MG": "Minas Gerais",
        "PA": "Para",
        "PB": "Paraiba",
        "PR": "Parana",
        "PE": "Pernambuco",
        "PI": "Piaui",
        "RJ": "Rio de Janeiro",
        "RN": "Rio Grande do Norte",
        "RS": "Rio Grande do Sul",
        "RO": "Rondonia",
        "RR": "Roraima",
        "SC": "Santa Catarina",
        "SP": "Sao Paulo",
        "SE": "Sergipe",
        "TO": "Tocantins",
    },
    "perfil_genero": {
        "1": "Masculino",
        "2": "Masculino",
        "3": "Feminino",
        "4": "Feminino",
        "M": "Masculino",
        "F": "Feminino",
        "MASC": "Masculino",
        "FEM": "Feminino",
        "0": "Nao informado",
        "-1": "Nao informado",
        "-3": "Nao informado",
        "-4": "Nao informado",
    },
    "perfil_faixa_etaria": {
        "1600": "16 anos",
        "1700": "17 anos",
        "1800": "18 anos",
        "1900": "19 anos",
        "2000": "20 anos",
        "2124": "21 a 24 anos",
        "2529": "25 a 29 anos",
        "3034": "30 a 34 anos",
        "3539": "35 a 39 anos",
        "4044": "40 a 44 anos",
        "4549": "45 a 49 anos",
        "5054": "50 a 54 anos",
        "5559": "55 a 59 anos",
        "6064": "60 a 64 anos",
        "6569": "65 a 69 anos",
        "7074": "70 a 74 anos",
        "7579": "75 a 79 anos",
        "8084": "80 a 84 anos",
        "8589": "85 a 89 anos",
        "9094": "90 a 94 anos",
        "9599": "95 a 99 anos",
        "9999": "100 anos ou mais",
        "0": "Nao informado",
        "-1": "Nao informado",
        "-3": "Nao informado",
        "-4": "Nao informado",
    },
    "turno": {
        "1": "Primeiro turno",
        "2": "Segundo turno",
        "0": "Turno nao informado",
    },
    "cargo": {
        "1": "Presidente",
        "3": "Governador",
        "5": "Senador",
        "6": "Deputado federal",
        "7": "Deputado estadual",
        "8": "Deputado distrital",
        "9": "Deputado estadual",
        "11": "Prefeito",
        "12": "Vice-prefeito",
        "13": "Vereador",
    },
    "perfil_instrucao": {
        "1": "Analfabeto",
        "2": "Le e escreve",
        "3": "Fundamental incompleto",
        "4": "Fundamental completo",
        "5": "Medio incompleto",
        "6": "Medio completo",
        "7": "Superior incompleto",
        "8": "Superior completo",
        "0": "Nao informado",
        "-1": "Nao informado",
        "-3": "Nao informado",
        "-4": "Nao informado",
    },
    "perfil_estado_civil": {
        "1": "Solteiro",
        "3": "Casado",
        "5": "Viuvo",
        "7": "Separado judicialmente",
        "9": "Divorciado",
        "0": "Nao informado",
        "-1": "Nao informado",
        "-3": "Nao informado",
        "-4": "Nao informado",
    },
    "perfil_raca_cor": {
        "1": "Branca",
        "2": "Preta",
        "3": "Parda",
        "4": "Amarela",
        "5": "Indigena",
        "6": "Nao informado",
        "0": "Nao informado",
        "-1": "Nao informado",
        "-3": "Nao informado",
        "-4": "Nao informado",
    },
    "perfil_biometria": {
        "S": "Com biometria",
        "SIM": "Com biometria",
        "1": "Com biometria",
        "N": "Sem biometria",
        "NAO": "Sem biometria",
        "NÃO": "Sem biometria",
        "0": "Sem biometria",
        "-1": "Nao informado",
        "-3": "Nao informado",
        "-4": "Nao informado",
    },
}

TEXT_DISCRETE_ROLES = {
    "uf",
    "regiao",
    "turno",
    "cargo",
    "partido",
    "candidato",
    "ideologia",
    "coalizao",
    "perfil_faixa_etaria",
    "perfil_genero",
    "perfil_instrucao",
    "perfil_estado_civil",
    "perfil_raca_cor",
    "perfil_biometria",
}

IGNORED_DISCRETE_ROLES = {"ano", "data", "hora", "datetime"}

FIELD_LABELS = {
    "ano": "Ano",
    "uf": "UF",
    "regiao": "Regiao",
    "cargo": "Cargo",
    "turno": "Turno",
    "partido": "Partido",
    "candidato": "Candidato",
    "perfil_faixa_etaria": "Faixa etaria",
    "perfil_genero": "Genero",
    "perfil_instrucao": "Instrucao",
    "perfil_estado_civil": "Estado civil",
    "perfil_raca_cor": "Raca/cor",
    "perfil_biometria": "Biometria",
    "perfil_predominante_setor": "Perfil predominante",
    "partido_vencedor_setor": "Partido vencedor",
    "candidato_vencedor_setor": "Candidato vencedor",
    "vencedor_setor": "Vencedor",
    "faixa_votos_setor": "Faixa de votos",
    "faixa_abstencao_setor": "Faixa de abstencao",
    "faixa_comparecimento_setor": "Faixa de comparecimento",
    "municipio_grupo": "Municipio",
}


def role_from_column(col: Any, role: str = "") -> str:
    if safe_text(role):
        return safe_text(role)
    c = normalize_col(col)
    if "DATA" in c or c.startswith("DT_") or c in {"DT", "DATE"}:
        return "data"
    if "HORA" in c or c.startswith("HH_") or c in {"HR", "HOUR"}:
        return "hora"
    if "ANO" in c:
        return "ano"
    if c in {"SG_UF", "UF", "SIGLA_UF"} or c.endswith("_UF"):
        return "uf"
    if "REGIAO" in c or "REGIÃO" in c:
        return "regiao"
    if "GENERO" in c or c in {"SEXO", "DS_SEXO", "CD_GENERO"}:
        return "perfil_genero"
    if "FAIXA_ETARIA" in c or "IDADE" in c:
        return "perfil_faixa_etaria"
    if "TURNO" in c:
        return "turno"
    if "CARGO" in c:
        return "cargo"
    if "PARTIDO" in c or c == "SG_PARTIDO":
        return "partido"
    if "CANDIDATO" in c or "URNA" in c or "VOTAVEL" in c:
        return "candidato"
    if "INSTRU" in c or "ESCOLAR" in c or "GRAU" in c:
        return "perfil_instrucao"
    if "ESTADO_CIVIL" in c:
        return "perfil_estado_civil"
    if "RACA" in c or "RAÇA" in c or "COR" in c:
        return "perfil_raca_cor"
    if "BIOMET" in c:
        return "perfil_biometria"
    return ""


def label_category_value(value: Any, col: Any = "", role: str = "") -> str:
    text = safe_text(value, "SEM_VALOR")
    if text.lower() in NULL_TOKENS:
        return "Sem valor"

    role_key = role_from_column(col, role)
    mapping = CATEGORY_LABELS.get(role_key, {})
    key = _canonical_value_key(text)
    if key in mapping:
        return mapping[key]

    if role_key == "ano" and re.fullmatch(r"[-+]?\d+(?:\.0)?", text):
        return f"Ano {text[:-2] if text.endswith('.0') else text}"

    if role_key.startswith("perfil_") and re.fullmatch(r"[-+]?\d+(?:\.0)?", text):
        return f"Codigo {text[:-2] if text.endswith('.0') else text}"

    return _humanize_text(text)


def label_category_series(series: pd.Series, col: Any = "", role: str = "") -> pd.Series:
    return series.map(lambda x: label_category_value(x, col=col, role=role))


def readable_field_label(col: Any) -> str:
    text = safe_text(col, "")
    if text in FIELD_LABELS:
        return FIELD_LABELS[text]
    return _humanize_text(text.replace("perfil_", ""))


def is_useful_discrete_series(series: pd.Series, col: Any = "", role: str = "", min_categories: int = 2, max_categories: int = 500) -> bool:
    if series is None:
        return False
    role_key = role_from_column(col, role)
    if role_key in IGNORED_DISCRETE_ROLES:
        return False
    raw = series.map(lambda x: safe_text(x, ""))
    raw_clean = raw.loc[
        lambda s: s.str.lower().notna() & ~s.str.lower().isin(NULL_TOKENS | {"sem valor"})
    ]
    if raw_clean.empty:
        return False

    has_known_role = role_key in TEXT_DISCRETE_ROLES or role_key in CATEGORY_LABELS
    numeric_ratio = raw_clean.map(lambda x: pd.notna(parse_number(x))).mean()
    if numeric_ratio >= 0.80 and role_key not in CATEGORY_LABELS and role_key != "ano":
        return False
    if numeric_ratio >= 0.80 and not has_known_role:
        return False

    labeled = label_category_series(series, col=col, role=role)
    cleaned = labeled.map(lambda x: safe_text(x, ""))
    cleaned = cleaned.loc[
        lambda s: s.str.lower().notna()
        & ~s.str.lower().isin(NULL_TOKENS | {"sem valor"})
        & ~s.map(_is_code_label)
    ]
    nunique = cleaned.nunique(dropna=True)
    if nunique < min_categories:
        return False
    if nunique > max_categories:
        return False
    if role_key in CATEGORY_LABELS:
        code_like_share = cleaned.str.startswith("Codigo ").mean() if len(cleaned) else 0
        if code_like_share >= 0.80:
            return False
    return True


def discrete_summary(df: pd.DataFrame, profile: pd.DataFrame | None = None, max_values: int = 20) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    rows = []
    profile_map = {}
    if profile is not None and not profile.empty and "coluna" in profile.columns:
        for _, r in profile.iterrows():
            profile_map[str(r.get("coluna", ""))] = str(r.get("role_sugerido", ""))

    for col in df.columns:
        role = profile_map.get(str(col), "")
        if not is_useful_discrete_series(df[col], col=col, role=role):
            continue
        labeled = label_category_series(df[col], col=col, role=role)
        cleaned = labeled.map(lambda x: safe_text(x, "Sem valor"))
        cleaned = cleaned.loc[
            lambda s: s.str.lower().notna()
            & ~s.str.lower().isin(NULL_TOKENS | {"sem valor"})
            & ~s.map(_is_code_label)
        ]
        if cleaned.empty:
            continue
        counts = cleaned.value_counts(dropna=False).head(max_values)
        total = max(len(cleaned), 1)
        for rank, (value, qtd) in enumerate(counts.items(), start=1):
            rows.append({
                "campo": col,
                "campo_legivel": readable_field_label(col),
                "valor": value,
                "qtd": int(qtd),
                "share": float(qtd / total),
                "rank_valor": rank,
                "qtd_categorias": int(cleaned.nunique(dropna=True)),
                "observacao": "Campo discreto com mais de uma categoria real.",
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["campo_legivel", "rank_valor"])
    return out


def _canonical_value_key(text: str) -> str:
    value = _strip_accents(safe_text(text, ""))
    if value.endswith(".0") and value[:-2].lstrip("-+").isdigit():
        value = value[:-2]
    return normalize_col(value).replace("_", " ").strip().replace(" ", "_") if len(value) > 1 and not value.lstrip("-+").isdigit() else value


def _humanize_text(text: str) -> str:
    value = safe_text(text, "Sem valor")
    value = value.replace("_", " ").strip()
    if value.isupper() and len(value) > 3:
        return value.title()
    return value


def _is_code_label(value: Any) -> bool:
    text = safe_text(value, "").strip().lower()
    if not (text.startswith("codigo ") or text.startswith("código ")):
        return False
    code = text.replace("codigo ", "", 1).replace("código ", "", 1).replace(".", "", 1).lstrip("-+")
    return code.isdigit()


def _strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(ch)
    )
