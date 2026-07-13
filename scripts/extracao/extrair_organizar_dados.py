# organizar_eleicoes.py
# Extrai PDFs e agrupa CSVs com o mesmo conjunto de campos, pasta por pasta.
#
# Uso:
#   python organizar_eleicoes.py "C:\CAMINHO\PARA\Analise_Eleitoral\dados"
#
# Exemplo, estando dentro da pasta Analise_Eleitoral:
#   python organizar_eleicoes.py ".\dados"
#
# Saída criada dentro de cada pasta que contém ZIPs:
#   CSV\   -> CSVs agrupados por campos iguais
#   PDFs\  -> PDFs encontrados dentro dos ZIPs
#   manifesto_processamento.csv -> controle do que foi agrupado

from __future__ import annotations

import argparse
import csv
import io
import os
import re
import shutil
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def ajustar_limite_csv() -> None:
    """Evita erro com linhas muito grandes."""
    limite = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limite)
            break
        except OverflowError:
            limite = int(limite / 10)


ajustar_limite_csv()


UFS = {
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG",
    "MS", "MT", "PA", "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR",
    "RS", "SC", "SE", "SP", "TO", "BR", "ZZ"
}


@dataclass
class GrupoCSV:
    group_id: int
    campos_originais: List[str]
    campos_normalizados: Tuple[str, ...]
    arquivo_tmp: Path
    fontes: List[str] = field(default_factory=list)
    linhas: int = 0
    arquivo_final: str = ""


def normalizar_campo(campo: str) -> str:
    """Normaliza nome de coluna só para comparar campos iguais."""
    return campo.strip().replace("\ufeff", "").upper()


def chave_campos(campos: List[str]) -> Tuple[str, ...]:
    """
    Agrupa CSVs que têm os mesmos campos, mesmo se a ordem mudar.

    Se você quiser agrupar somente quando os campos estão na mesma ordem,
    substitua esta função por:
        return tuple(normalizar_campo(c) for c in campos)
    """
    return tuple(sorted(normalizar_campo(c) for c in campos))


def detectar_encoding(sample: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            sample.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "latin1"


def detectar_delimitador(texto_amostra: str) -> str:
    # Dados do TSE normalmente vêm separados por ponto e vírgula.
    try:
        dialect = csv.Sniffer().sniff(texto_amostra, delimiters=";,|\t")
        return dialect.delimiter
    except csv.Error:
        return ";"


def nome_seguro(nome: str) -> str:
    nome = Path(nome).name
    nome = re.sub(r'[<>:"/\\|?*]', "_", nome)
    nome = re.sub(r"\s+", "_", nome).strip("_")
    return nome or "arquivo"


def caminho_unico(path: Path) -> Path:
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    i = 2

    while True:
        candidato = parent / f"{stem}_{i}{suffix}"
        if not candidato.exists():
            return candidato
        i += 1


def limpar_pasta_saida(csv_dir: Path, pdf_dir: Path, limpar_saida: bool) -> None:
    if not limpar_saida:
        return

    if csv_dir.exists():
        for arquivo in csv_dir.glob("*.csv"):
            arquivo.unlink()

    if pdf_dir.exists():
        for arquivo in pdf_dir.glob("*.pdf"):
            arquivo.unlink()


def encontrar_pastas_com_zip(root: Path) -> List[Path]:
    """
    Encontra pastas que têm ZIPs diretamente dentro delas.
    Assim o processamento fica separado:
      candidatos_2014, eleitorado_2014, resultados_2014, etc.
    """
    pastas: List[Path] = []

    todas = [root]
    todas.extend(p for p in root.rglob("*") if p.is_dir())

    for pasta in todas:
        if pasta.name.upper() in {"CSV", "PDFS"}:
            continue

        try:
            tem_zip = any(
                item.is_file() and item.suffix.lower() == ".zip"
                for item in pasta.iterdir()
            )
        except PermissionError:
            continue

        if tem_zip:
            pastas.append(pasta)

    return sorted(set(pastas))


def membros_zip(zip_ref: zipfile.ZipFile) -> Iterable[zipfile.ZipInfo]:
    for info in zip_ref.infolist():
        if info.is_dir():
            continue
        yield info


def extrair_pdf(zip_ref: zipfile.ZipFile, info: zipfile.ZipInfo, zip_path: Path, pdf_dir: Path) -> Path:
    pdf_dir.mkdir(exist_ok=True)

    pdf_nome = nome_seguro(Path(info.filename).name)
    destino_base = pdf_dir / f"{nome_seguro(zip_path.stem)}__{pdf_nome}"
    destino = caminho_unico(destino_base)

    with zip_ref.open(info, "r") as origem, open(destino, "wb") as saida:
        shutil.copyfileobj(origem, saida)

    return destino


def abrir_csv_do_zip(zip_ref: zipfile.ZipFile, info: zipfile.ZipInfo):
    """
    Abre CSV de dentro do ZIP em modo texto, detectando encoding/delimitador.
    Retorna: text_stream, delimitador
    """
    with zip_ref.open(info, "r") as raw:
        sample = raw.read(65536)

    encoding = detectar_encoding(sample)
    texto_amostra = sample.decode(encoding, errors="replace")
    delimitador = detectar_delimitador(texto_amostra)

    raw_stream = zip_ref.open(info, "r")
    text_stream = io.TextIOWrapper(
        raw_stream,
        encoding=encoding,
        errors="replace",
        newline=""
    )
    return text_stream, delimitador


def montar_nome_final(grupo: GrupoCSV, pasta: Path, usados: set[str]) -> Path:
    """
    Cria nome compreensível para o CSV agrupado com base nos arquivos de origem.
    Ex.: votacao_secao_2014_AC + votacao_secao_2014_SP -> votacao_secao_2014_agrupado.csv
    """
    stems = []

    for fonte in grupo.fontes:
        # fonte vem como zip::arquivo.csv
        arquivo = fonte.split("::")[-1]
        stem = Path(arquivo).stem

        # remove UF no final do nome, quando existir
        stem = re.sub(
            r"_(AC|AL|AM|AP|BA|CE|DF|ES|GO|MA|MG|MS|MT|PA|PB|PE|PI|PR|RJ|RN|RO|RR|RS|SC|SE|SP|TO|BR|ZZ)$",
            "",
            stem,
            flags=re.I
        )

        stems.append(stem)

    if not stems:
        base = f"{pasta.name}_grupo_{grupo.group_id:03d}"
    elif len(stems) == 1:
        base = stems[0]
    else:
        base = os.path.commonprefix(stems).strip("_-. ")
        if len(base) < 5:
            base = stems[0]

    base = nome_seguro(base)
    nome = f"{base}_agrupado.csv"

    # evita conflito caso dois grupos gerem o mesmo nome
    if nome.lower() in usados:
        nome = f"{base}_grupo_{grupo.group_id:03d}_agrupado.csv"

    usados.add(nome.lower())
    return grupo.arquivo_tmp.parent / nome


def processar_csv(
    zip_ref: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    zip_path: Path,
    csv_dir: Path,
    grupos: Dict[Tuple[str, ...], GrupoCSV],
) -> None:
    csv_dir.mkdir(exist_ok=True)

    fonte_nome = f"{zip_path.name}::{info.filename}"

    text_stream, delimitador = abrir_csv_do_zip(zip_ref, info)

    try:
        reader = csv.reader(text_stream, delimiter=delimitador)

        campos = next(reader, None)
        if not campos:
            print(f"      [AVISO] CSV vazio ignorado: {fonte_nome}")
            return

        campos = [campo.strip().replace("\ufeff", "") for campo in campos]
        chave = chave_campos(campos)

        novo_grupo = False

        if chave not in grupos:
            group_id = len(grupos) + 1
            arquivo_tmp = csv_dir / f"_grupo_{group_id:03d}.csv"

            grupos[chave] = GrupoCSV(
                group_id=group_id,
                campos_originais=campos,
                campos_normalizados=chave,
                arquivo_tmp=arquivo_tmp,
            )

            novo_grupo = True

        grupo = grupos[chave]
        grupo.fontes.append(fonte_nome)

        # Mapeia a ordem das colunas do CSV atual para a ordem canônica do grupo.
        mapa_atual = {
            normalizar_campo(campo): idx
            for idx, campo in enumerate(campos)
        }

        idx_map = [
            mapa_atual[normalizar_campo(campo_canonico)]
            for campo_canonico in grupo.campos_originais
        ]

        modo = "w" if novo_grupo else "a"
        encoding_saida = "utf-8-sig" if novo_grupo else "utf-8"

        with open(grupo.arquivo_tmp, modo, encoding=encoding_saida, newline="") as saida:
            writer = csv.writer(saida, delimiter=";", lineterminator="\n", quoting=csv.QUOTE_MINIMAL)

            if novo_grupo:
                writer.writerow(grupo.campos_originais)

            linhas_csv = 0

            for row in reader:
                if not row or all(not str(valor).strip() for valor in row):
                    continue

                linha_reordenada = [
                    row[i] if i < len(row) else ""
                    for i in idx_map
                ]

                writer.writerow(linha_reordenada)
                linhas_csv += 1

        grupo.linhas += linhas_csv
        print(f"      CSV agrupado: {info.filename} -> grupo {grupo.group_id:03d} | {linhas_csv:,} linhas")

    finally:
        text_stream.close()


def gravar_manifesto(pasta: Path, grupos: Dict[Tuple[str, ...], GrupoCSV], pdfs_extraidos: List[str]) -> None:
    manifesto = pasta / "manifesto_processamento.csv"

    with open(manifesto, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter=";", lineterminator="\n")

        writer.writerow([
            "pasta",
            "grupo_id",
            "arquivo_saida",
            "qtd_linhas",
            "qtd_fontes",
            "campos",
            "fontes"
        ])

        for grupo in sorted(grupos.values(), key=lambda g: g.group_id):
            writer.writerow([
                str(pasta),
                grupo.group_id,
                grupo.arquivo_final,
                grupo.linhas,
                len(grupo.fontes),
                " | ".join(grupo.campos_originais),
                " | ".join(grupo.fontes)
            ])

    if pdfs_extraidos:
        manifesto_pdfs = pasta / "manifesto_pdfs.csv"
        with open(manifesto_pdfs, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f, delimiter=";", lineterminator="\n")
            writer.writerow(["pasta", "arquivo_pdf"])
            for pdf in pdfs_extraidos:
                writer.writerow([str(pasta), pdf])


def finalizar_nomes_csv(pasta: Path, grupos: Dict[Tuple[str, ...], GrupoCSV]) -> None:
    usados: set[str] = set()

    for grupo in sorted(grupos.values(), key=lambda g: g.group_id):
        destino = montar_nome_final(grupo, pasta, usados)

        if destino.exists():
            destino.unlink()

        grupo.arquivo_tmp.rename(destino)
        grupo.arquivo_final = str(destino.name)


def processar_pasta(pasta: Path, limpar_saida: bool = True) -> Tuple[int, int, int]:
    zips = sorted([p for p in pasta.iterdir() if p.is_file() and p.suffix.lower() == ".zip"])

    if not zips:
        return 0, 0, 0

    print("\n" + "=" * 100)
    print(f"Processando pasta: {pasta}")
    print(f"ZIPs encontrados: {len(zips)}")

    csv_dir = pasta / "CSV"
    pdf_dir = pasta / "PDFs"

    csv_dir.mkdir(exist_ok=True)
    pdf_dir.mkdir(exist_ok=True)

    limpar_pasta_saida(csv_dir, pdf_dir, limpar_saida=limpar_saida)

    grupos: Dict[Tuple[str, ...], GrupoCSV] = {}
    pdfs_extraidos: List[str] = []
    total_csvs_lidos = 0
    total_pdfs_extraidos = 0

    for zip_path in zips:
        print(f"\n  Abrindo ZIP: {zip_path.name}")

        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                for info in membros_zip(zip_ref):
                    suffix = Path(info.filename).suffix.lower()

                    if suffix == ".csv":
                        processar_csv(zip_ref, info, zip_path, csv_dir, grupos)
                        total_csvs_lidos += 1

                    elif suffix == ".pdf":
                        destino_pdf = extrair_pdf(zip_ref, info, zip_path, pdf_dir)
                        pdfs_extraidos.append(str(destino_pdf.name))
                        total_pdfs_extraidos += 1
                        print(f"      PDF extraído: {info.filename} -> PDFs\\{destino_pdf.name}")

        except zipfile.BadZipFile:
            print(f"  [ERRO] ZIP corrompido ou inválido: {zip_path}")
        except PermissionError:
            print(f"  [ERRO] Sem permissão para abrir: {zip_path}")
        except Exception as exc:
            print(f"  [ERRO] Falha ao processar {zip_path.name}: {exc}")

    finalizar_nomes_csv(pasta, grupos)
    gravar_manifesto(pasta, grupos, pdfs_extraidos)

    print(f"\nResumo da pasta: {pasta.name}")
    print(f"  CSVs lidos de dentro dos ZIPs: {total_csvs_lidos}")
    print(f"  Grupos CSV gerados: {len(grupos)}")
    print(f"  PDFs extraídos: {total_pdfs_extraidos}")
    print(f"  Pasta CSV: {csv_dir}")
    print(f"  Pasta PDFs: {pdf_dir}")

    return total_csvs_lidos, len(grupos), total_pdfs_extraidos


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extrai PDFs e agrupa CSVs com campos iguais dentro de ZIPs, pasta por pasta."
    )

    parser.add_argument(
        "dados",
        help="Caminho da pasta dados ou de uma pasta específica, por exemplo: .\\dados"
    )

    parser.add_argument(
        "--nao-limpar",
        action="store_true",
        help="Não limpa os CSVs/PDFs gerados anteriormente antes de reprocessar."
    )

    args = parser.parse_args()

    root = Path(args.dados).expanduser().resolve()

    if not root.exists():
        print(f"[ERRO] Caminho não encontrado: {root}")
        sys.exit(1)

    if not root.is_dir():
        print(f"[ERRO] O caminho informado não é uma pasta: {root}")
        sys.exit(1)

    pastas = encontrar_pastas_com_zip(root)

    if not pastas:
        print(f"Nenhuma pasta com arquivos .zip foi encontrada em: {root}")
        sys.exit(0)

    print(f"Raiz informada: {root}")
    print(f"Pastas com ZIP encontradas: {len(pastas)}")

    total_csvs = 0
    total_grupos = 0
    total_pdfs = 0

    for pasta in pastas:
        csvs, grupos, pdfs = processar_pasta(pasta, limpar_saida=not args.nao_limpar)
        total_csvs += csvs
        total_grupos += grupos
        total_pdfs += pdfs

    print("\n" + "=" * 100)
    print("PROCESSAMENTO FINALIZADO")
    print(f"Total de CSVs lidos: {total_csvs}")
    print(f"Total de grupos CSV gerados: {total_grupos}")
    print(f"Total de PDFs extraídos: {total_pdfs}")


if __name__ == "__main__":
    main()
