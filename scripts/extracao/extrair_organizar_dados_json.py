# organizar_eleicoes_json_rapido.py
# Extrai PDFs e agrupa CSVs com os mesmos campos, pasta por pasta, gerando JSON/JSONL.
#
# Uso recomendado no WSL, dentro da pasta Analise_Eleitoral:
#   python3 organizar_eleicoes_json_rapido.py ./dados --formato jsonl --workers 4
#
# Para gerar JSON tradicional, com array:
#   python3 organizar_eleicoes_json_rapido.py ./dados --formato json --workers 4
#
# Saída criada dentro de cada pasta que contém ZIPs:
#   JSON/  -> arquivos agrupados por campos iguais
#   PDFs/  -> PDFs encontrados dentro dos ZIPs
#   manifesto_processamento.json -> controle do que foi agrupado
# python3 scripts/extracao/extrair_organizar_dados_json.py ./dados --formato jsonl --workers 4
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import shutil
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


try:
    import orjson  # type: ignore

    def dumps_json_bytes(obj: dict) -> bytes:
        return orjson.dumps(obj)

    JSON_ENGINE = "orjson"

except Exception:
    def dumps_json_bytes(obj: dict) -> bytes:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    JSON_ENGINE = "json"


def ajustar_limite_csv() -> None:
    limite = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limite)
            break
        except OverflowError:
            limite = int(limite / 10)


ajustar_limite_csv()


@dataclass
class GrupoJSON:
    group_id: int
    campos_originais: List[str]
    campos_normalizados: Tuple[str, ...]
    arquivo_tmp: Path
    formato: str
    fontes: List[str] = field(default_factory=list)
    linhas: int = 0
    primeiro_registro: bool = True
    arquivo_final: str = ""


def normalizar_campo(campo: str) -> str:
    return campo.strip().replace("\ufeff", "").upper()


def chave_campos(campos: List[str]) -> Tuple[str, ...]:
    """
    Agrupa arquivos que possuem os mesmos campos, mesmo se a ordem for diferente.

    Se quiser exigir a mesma ordem de colunas também, troque por:
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


def limpar_pasta_saida(json_dir: Path, pdf_dir: Path, limpar_saida: bool) -> None:
    if not limpar_saida:
        return

    if json_dir.exists():
        for arquivo in json_dir.glob("*.json"):
            arquivo.unlink()
        for arquivo in json_dir.glob("*.jsonl"):
            arquivo.unlink()

    if pdf_dir.exists():
        for arquivo in pdf_dir.glob("*.pdf"):
            arquivo.unlink()


def encontrar_pastas_com_zip(root: Path) -> List[Path]:
    pastas: List[Path] = []

    todas = [root]
    todas.extend(p for p in root.rglob("*") if p.is_dir())

    ignorar = {"CSV", "JSON", "PDFS", "__MACOSX"}

    for pasta in todas:
        if pasta.name.upper() in ignorar:
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
        if "__MACOSX" in info.filename:
            continue
        yield info


def abrir_csv_do_zip(zip_ref: zipfile.ZipFile, info: zipfile.ZipInfo):
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


def extrair_pdf(zip_ref: zipfile.ZipFile, info: zipfile.ZipInfo, zip_path: Path, pdf_dir: Path) -> Path:
    pdf_dir.mkdir(exist_ok=True)

    pdf_nome = nome_seguro(Path(info.filename).name)
    destino_base = pdf_dir / f"{nome_seguro(zip_path.stem)}__{pdf_nome}"
    destino = caminho_unico(destino_base)

    with zip_ref.open(info, "r") as origem, open(destino, "wb") as saida:
        shutil.copyfileobj(origem, saida, length=1024 * 1024)

    return destino


def iniciar_arquivo_grupo(grupo: GrupoJSON) -> None:
    if grupo.formato == "json":
        with open(grupo.arquivo_tmp, "wb") as f:
            f.write(b"[\n")
    else:
        grupo.arquivo_tmp.touch()


def finalizar_arquivo_grupo(grupo: GrupoJSON) -> None:
    if grupo.formato == "json":
        with open(grupo.arquivo_tmp, "ab") as f:
            f.write(b"\n]\n")


def escrever_objeto_json(grupo: GrupoJSON, objeto: dict, fh) -> None:
    if grupo.formato == "jsonl":
        fh.write(dumps_json_bytes(objeto))
        fh.write(b"\n")
        return

    if not grupo.primeiro_registro:
        fh.write(b",\n")

    fh.write(dumps_json_bytes(objeto))
    grupo.primeiro_registro = False


def montar_nome_final(grupo: GrupoJSON, pasta: Path, usados: set[str]) -> Path:
    stems = []

    for fonte in grupo.fontes:
        arquivo = fonte.split("::")[-1]
        stem = Path(arquivo).stem

        # Remove UF no final: votacao_secao_2014_SP -> votacao_secao_2014
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
    ext = ".jsonl" if grupo.formato == "jsonl" else ".json"
    nome = f"{base}_agrupado{ext}"

    if nome.lower() in usados:
        nome = f"{base}_grupo_{grupo.group_id:03d}_agrupado{ext}"

    usados.add(nome.lower())
    return grupo.arquivo_tmp.parent / nome


def processar_csv(
    zip_ref: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    zip_path: Path,
    json_dir: Path,
    grupos: Dict[Tuple[str, ...], GrupoJSON],
    formato: str,
    adicionar_fonte: bool,
) -> int:
    json_dir.mkdir(exist_ok=True)

    fonte_nome = f"{zip_path.name}::{info.filename}"
    text_stream, delimitador = abrir_csv_do_zip(zip_ref, info)

    try:
        reader = csv.reader(text_stream, delimiter=delimitador)

        campos = next(reader, None)
        if not campos:
            print(f"      [AVISO] CSV vazio ignorado: {fonte_nome}", flush=True)
            return 0

        campos = [campo.strip().replace("\ufeff", "") for campo in campos]
        chave = chave_campos(campos)

        novo_grupo = False

        if chave not in grupos:
            group_id = len(grupos) + 1
            ext = "jsonl" if formato == "jsonl" else "json"
            arquivo_tmp = json_dir / f"_grupo_{group_id:03d}.{ext}"

            grupos[chave] = GrupoJSON(
                group_id=group_id,
                campos_originais=campos,
                campos_normalizados=chave,
                arquivo_tmp=arquivo_tmp,
                formato=formato,
            )

            iniciar_arquivo_grupo(grupos[chave])
            novo_grupo = True

        grupo = grupos[chave]
        grupo.fontes.append(fonte_nome)

        mapa_atual = {
            normalizar_campo(campo): idx
            for idx, campo in enumerate(campos)
        }

        idx_map = [
            mapa_atual[normalizar_campo(campo_canonico)]
            for campo_canonico in grupo.campos_originais
        ]

        linhas_csv = 0

        with open(grupo.arquivo_tmp, "ab", buffering=1024 * 1024) as saida:
            for row in reader:
                if not row or all(not str(valor).strip() for valor in row):
                    continue

                valores = [
                    row[i] if i < len(row) else ""
                    for i in idx_map
                ]

                objeto = dict(zip(grupo.campos_originais, valores))

                if adicionar_fonte:
                    objeto["_fonte_zip"] = zip_path.name
                    objeto["_fonte_arquivo"] = info.filename

                escrever_objeto_json(grupo, objeto, saida)
                linhas_csv += 1

        grupo.linhas += linhas_csv

        if novo_grupo:
            print(f"      Novo grupo {grupo.group_id:03d}: {len(grupo.campos_originais)} campos", flush=True)

        print(f"      JSON agrupado: {info.filename} -> grupo {grupo.group_id:03d} | {linhas_csv:,} linhas", flush=True)

        return linhas_csv

    finally:
        text_stream.close()


def finalizar_nomes_json(pasta: Path, grupos: Dict[Tuple[str, ...], GrupoJSON]) -> None:
    usados: set[str] = set()

    for grupo in sorted(grupos.values(), key=lambda g: g.group_id):
        finalizar_arquivo_grupo(grupo)

        destino = montar_nome_final(grupo, pasta, usados)

        if destino.exists():
            destino.unlink()

        grupo.arquivo_tmp.rename(destino)
        grupo.arquivo_final = str(destino.name)


def gravar_manifesto(pasta: Path, grupos: Dict[Tuple[str, ...], GrupoJSON], pdfs_extraidos: List[str], formato: str) -> None:
    manifesto = pasta / "manifesto_processamento.json"

    dados = {
        "pasta": str(pasta),
        "formato_saida": formato,
        "grupos": [
            {
                "grupo_id": grupo.group_id,
                "arquivo_saida": grupo.arquivo_final,
                "qtd_linhas": grupo.linhas,
                "qtd_fontes": len(grupo.fontes),
                "campos": grupo.campos_originais,
                "fontes": grupo.fontes,
            }
            for grupo in sorted(grupos.values(), key=lambda g: g.group_id)
        ],
        "pdfs_extraidos": pdfs_extraidos,
    }

    with open(manifesto, "wb") as f:
        f.write(json.dumps(dados, ensure_ascii=False, indent=2).encode("utf-8"))


def processar_pasta(
    pasta_str: str,
    limpar_saida: bool = True,
    formato: str = "jsonl",
    adicionar_fonte: bool = False,
    extrair_pdfs: bool = True,
) -> dict:
    pasta = Path(pasta_str)
    zips = sorted([p for p in pasta.iterdir() if p.is_file() and p.suffix.lower() == ".zip"])

    if not zips:
        return {
            "pasta": str(pasta),
            "zips": 0,
            "csvs_lidos": 0,
            "grupos": 0,
            "pdfs": 0,
            "linhas": 0,
            "status": "sem_zip",
        }

    print("\n" + "=" * 100, flush=True)
    print(f"Processando pasta: {pasta}", flush=True)
    print(f"ZIPs encontrados: {len(zips)}", flush=True)

    json_dir = pasta / "JSON"
    pdf_dir = pasta / "PDFs"

    json_dir.mkdir(exist_ok=True)
    pdf_dir.mkdir(exist_ok=True)

    limpar_pasta_saida(json_dir, pdf_dir, limpar_saida=limpar_saida)

    grupos: Dict[Tuple[str, ...], GrupoJSON] = {}
    pdfs_extraidos: List[str] = []

    total_csvs_lidos = 0
    total_linhas = 0
    total_pdfs_extraidos = 0
    erros: List[str] = []

    for zip_path in zips:
        print(f"\n  Abrindo ZIP: {zip_path.name}", flush=True)

        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                for info in membros_zip(zip_ref):
                    suffix = Path(info.filename).suffix.lower()

                    if suffix == ".csv":
                        linhas = processar_csv(
                            zip_ref=zip_ref,
                            info=info,
                            zip_path=zip_path,
                            json_dir=json_dir,
                            grupos=grupos,
                            formato=formato,
                            adicionar_fonte=adicionar_fonte,
                        )
                        total_csvs_lidos += 1
                        total_linhas += linhas

                    elif suffix == ".pdf" and extrair_pdfs:
                        destino_pdf = extrair_pdf(zip_ref, info, zip_path, pdf_dir)
                        pdfs_extraidos.append(str(destino_pdf.name))
                        total_pdfs_extraidos += 1
                        print(f"      PDF extraído: {info.filename} -> PDFs/{destino_pdf.name}", flush=True)

        except zipfile.BadZipFile:
            msg = f"ZIP corrompido ou inválido: {zip_path}"
            print(f"  [ERRO] {msg}", flush=True)
            erros.append(msg)
        except PermissionError:
            msg = f"Sem permissão para abrir: {zip_path}"
            print(f"  [ERRO] {msg}", flush=True)
            erros.append(msg)
        except Exception as exc:
            msg = f"Falha ao processar {zip_path.name}: {exc}"
            print(f"  [ERRO] {msg}", flush=True)
            erros.append(msg)

    finalizar_nomes_json(pasta, grupos)
    gravar_manifesto(pasta, grupos, pdfs_extraidos, formato=formato)

    print(f"\nResumo da pasta: {pasta.name}", flush=True)
    print(f"  CSVs lidos de dentro dos ZIPs: {total_csvs_lidos}", flush=True)
    print(f"  Linhas convertidas para JSON: {total_linhas:,}", flush=True)
    print(f"  Grupos JSON gerados: {len(grupos)}", flush=True)
    print(f"  PDFs extraídos: {total_pdfs_extraidos}", flush=True)
    print(f"  Pasta JSON: {json_dir}", flush=True)
    print(f"  Pasta PDFs: {pdf_dir}", flush=True)

    return {
        "pasta": str(pasta),
        "zips": len(zips),
        "csvs_lidos": total_csvs_lidos,
        "grupos": len(grupos),
        "pdfs": total_pdfs_extraidos,
        "linhas": total_linhas,
        "status": "ok" if not erros else "com_erros",
        "erros": erros,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extrai PDFs e agrupa CSVs com campos iguais dentro de ZIPs, pasta por pasta, gerando JSON/JSONL."
    )

    parser.add_argument(
        "dados",
        help="Caminho da pasta dados ou de uma pasta específica. Ex.: ./dados"
    )

    parser.add_argument(
        "--formato",
        choices=["jsonl", "json"],
        default="jsonl",
        help="jsonl é mais rápido e recomendado para dados grandes. json gera array JSON tradicional."
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Quantidade de pastas processadas em paralelo. Ex.: --workers 4"
    )

    parser.add_argument(
        "--nao-limpar",
        action="store_true",
        help="Não limpa os JSONs/PDFs gerados anteriormente antes de reprocessar."
    )

    parser.add_argument(
        "--adicionar-fonte",
        action="store_true",
        help="Adiciona _fonte_zip e _fonte_arquivo em cada registro JSON."
    )

    parser.add_argument(
        "--sem-pdfs",
        action="store_true",
        help="Não extrai PDFs. Use para ganhar tempo quando só precisa dos JSONs."
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

    workers = max(1, int(args.workers))

    print(f"Raiz informada: {root}")
    print(f"Pastas com ZIP encontradas: {len(pastas)}")
    print(f"Formato de saída: {args.formato}")
    print(f"Engine JSON: {JSON_ENGINE}")
    print(f"Workers: {workers}")

    resultados: List[dict] = []

    if workers == 1:
        for pasta in pastas:
            resultados.append(
                processar_pasta(
                    str(pasta),
                    limpar_saida=not args.nao_limpar,
                    formato=args.formato,
                    adicionar_fonte=args.adicionar_fonte,
                    extrair_pdfs=not args.sem_pdfs,
                )
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    processar_pasta,
                    str(pasta),
                    not args.nao_limpar,
                    args.formato,
                    args.adicionar_fonte,
                    not args.sem_pdfs,
                )
                for pasta in pastas
            ]

            for future in as_completed(futures):
                resultados.append(future.result())

    total_csvs = sum(r.get("csvs_lidos", 0) for r in resultados)
    total_grupos = sum(r.get("grupos", 0) for r in resultados)
    total_pdfs = sum(r.get("pdfs", 0) for r in resultados)
    total_linhas = sum(r.get("linhas", 0) for r in resultados)

    resumo = {
        "raiz": str(root),
        "formato_saida": args.formato,
        "engine_json": JSON_ENGINE,
        "workers": workers,
        "total_pastas": len(pastas),
        "total_csvs_lidos": total_csvs,
        "total_grupos_json_gerados": total_grupos,
        "total_pdfs_extraidos": total_pdfs,
        "total_linhas_convertidas": total_linhas,
        "resultados_por_pasta": resultados,
    }

    resumo_path = root / "resumo_processamento_json.json"
    with open(resumo_path, "wb") as f:
        f.write(json.dumps(resumo, ensure_ascii=False, indent=2).encode("utf-8"))

    print("\n" + "=" * 100)
    print("PROCESSAMENTO FINALIZADO")
    print(f"Total de CSVs lidos: {total_csvs}")
    print(f"Total de linhas convertidas: {total_linhas:,}")
    print(f"Total de grupos JSON gerados: {total_grupos}")
    print(f"Total de PDFs extraídos: {total_pdfs}")
    print(f"Resumo geral: {resumo_path}")


if __name__ == "__main__":
    main()
