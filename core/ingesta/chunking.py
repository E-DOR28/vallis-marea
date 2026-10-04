"""Troceado del corpus consciente de la estructura del documento.

El chunking por ventana fija corta a mitad de frase y pierde el encabezado, con
lo que el fragmento recuperado llega al LLM sin contexto y la cita queda
inservible. Aqui se trocea por seccion (encabezados `##`) y solo se subdivide
cuando una seccion excede el tamano objetivo, respetando limites de parrafo.

Cada fragmento arrastra sus metadatos (fuente, titulo, seccion, version,
fecha). Esos metadatos son lo que permite citar: sin ellos el agente puede
decir algo correcto, pero no puede probarlo.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from core import config


def _parsear_frontmatter(texto: str) -> tuple[dict[str, str], str]:
    """Extrae el bloque YAML simple del inicio del documento."""
    if not texto.startswith("---"):
        return {}, texto
    partes = texto.split("---", 2)
    if len(partes) < 3:
        return {}, texto
    meta: dict[str, str] = {}
    for linea in partes[1].strip().splitlines():
        if ":" in linea:
            clave, valor = linea.split(":", 1)
            meta[clave.strip()] = valor.strip()
    return meta, partes[2].strip()


def _dividir_por_secciones(cuerpo: str) -> list[tuple[str, str]]:
    """Parte el cuerpo en (titulo_seccion, contenido) usando encabezados `##`."""
    secciones: list[tuple[str, str]] = []
    actual_titulo = "Introduccion"
    actual: list[str] = []
    for linea in cuerpo.splitlines():
        m = re.match(r"^##\s+(.*)", linea)
        if m:
            if actual and "".join(actual).strip():
                secciones.append((actual_titulo, "\n".join(actual).strip()))
            actual_titulo = m.group(1).strip()
            actual = []
        else:
            actual.append(linea)
    if actual and "".join(actual).strip():
        secciones.append((actual_titulo, "\n".join(actual).strip()))
    return secciones


def _subdividir(texto: str, objetivo: int, solape: int) -> list[str]:
    """Parte una seccion larga por parrafos, sin cortar frases."""
    if len(texto) <= objetivo:
        return [texto]

    parrafos = [p.strip() for p in re.split(r"\n\s*\n", texto) if p.strip()]
    piezas: list[str] = []
    actual = ""
    for p in parrafos:
        if not actual:
            actual = p
        elif len(actual) + len(p) + 2 <= objetivo:
            actual = f"{actual}\n\n{p}"
        else:
            piezas.append(actual)
            # Solape: se arrastra la cola del fragmento anterior para no perder
            # el hilo entre piezas contiguas.
            cola = actual[-solape:] if solape else ""
            corte = cola.find(" ")
            actual = (cola[corte + 1 :] + "\n\n" + p) if corte > 0 else p
    if actual:
        piezas.append(actual)

    # Un parrafo unico mas largo que el objetivo se parte por frases.
    finales: list[str] = []
    for pieza in piezas:
        if len(pieza) <= objetivo * 1.6:
            finales.append(pieza)
            continue
        frases = re.split(r"(?<=[.!?])\s+", pieza)
        acc = ""
        for f in frases:
            if len(acc) + len(f) + 1 <= objetivo:
                acc = f"{acc} {f}".strip()
            else:
                if acc:
                    finales.append(acc)
                acc = f
        if acc:
            finales.append(acc)
    return finales


def trocear_documento(ruta: Path) -> list[dict[str, Any]]:
    """Convierte un .md del corpus en una lista de fragmentos con metadatos."""
    crudo = ruta.read_text(encoding="utf-8")
    meta, cuerpo = _parsear_frontmatter(crudo)

    fuente = meta.get("fuente", ruta.stem)
    titulo = meta.get("titulo", ruta.stem)
    categoria = meta.get("categoria", "general")
    version = meta.get("version", "s/v")
    fecha = meta.get("fecha", "")

    fragmentos: list[dict[str, Any]] = []
    for seccion_titulo, contenido in _dividir_por_secciones(cuerpo):
        piezas = _subdividir(
            contenido,
            config.CHUNK_OBJETIVO_CARACTERES,
            config.CHUNK_SOLAPE_CARACTERES,
        )
        for i, pieza in enumerate(piezas):
            # El texto que se indexa lleva el encabezado incorporado: mejora la
            # recuperacion densa y hace el fragmento legible por si solo.
            texto_indexable = f"{titulo} > {seccion_titulo}\n\n{pieza}"
            cid = hashlib.sha1(
                f"{fuente}|{seccion_titulo}|{i}|{pieza[:80]}".encode("utf-8")
            ).hexdigest()[:16]
            fragmentos.append(
                {
                    "id": cid,
                    "fuente": fuente,
                    "titulo": titulo,
                    "seccion": seccion_titulo,
                    "categoria": categoria,
                    "version": version,
                    "fecha": fecha,
                    "texto": texto_indexable,
                    "texto_crudo": pieza,
                    "archivo": ruta.name,
                    "orden": i,
                }
            )
    return fragmentos


def trocear_corpus(ruta_corpus: Path | None = None) -> list[dict[str, Any]]:
    ruta_corpus = ruta_corpus or config.RUTA_CORPUS
    fragmentos: list[dict[str, Any]] = []
    for archivo in sorted(ruta_corpus.glob("*.md")):
        fragmentos.extend(trocear_documento(archivo))
    return fragmentos
