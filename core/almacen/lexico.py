"""Busqueda lexica (BM25) sobre los fragmentos del corpus.

Produccion usa `tsvector` de Postgres; aqui se usa FTS5 de SQLite, que trae
BM25 incorporado. Es la mitad lexica de la busqueda hibrida: recupera lo que el
embedding pierde, tipicamente numeros, codigos y terminos exactos ("72 horas",
"VM-04", "Bocachica").
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any

from core import config

_TABLA = config.TABLA_FTS


def conexion() -> sqlite3.Connection:
    con = sqlite3.connect(config.RUTA_SQLITE, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def inicializar(recrear: bool = False) -> None:
    con = conexion()
    try:
        if recrear:
            con.execute(f"DROP TABLE IF EXISTS {_TABLA}")
        # remove_diacritics 2 hace que "politica" encuentre "política".
        con.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {_TABLA} USING fts5("
            " chunk_id UNINDEXED, fuente, titulo, seccion, texto,"
            " tokenize = 'unicode61 remove_diacritics 2')"
        )
        con.commit()
    finally:
        con.close()


def indexar(fragmentos: list[dict[str, Any]]) -> int:
    con = conexion()
    try:
        con.execute(f"DELETE FROM {_TABLA}")
        con.executemany(
            f"INSERT INTO {_TABLA} (chunk_id, fuente, titulo, seccion, texto) "
            "VALUES (?,?,?,?,?)",
            [(f["id"], f["fuente"], f["titulo"], f["seccion"], f["texto"])
             for f in fragmentos],
        )
        con.commit()
        return len(fragmentos)
    finally:
        con.close()


_PALABRA = re.compile(r"\w+", re.UNICODE)


def _consulta_fts(texto: str) -> str:
    """Convierte lenguaje natural en una consulta FTS5 segura.

    Se entrecomilla cada token para que caracteres como '-' o '?' no se
    interpreten como operadores de FTS5, y se unen con OR: queremos recuperar
    candidatos amplios y dejar que BM25 y la fusion decidan el orden.
    """
    tokens = [t for t in _PALABRA.findall(texto.lower()) if len(t) > 2]
    if not tokens:
        tokens = _PALABRA.findall(texto.lower()) or ["*"]
    return " OR ".join(f'"{t}"' for t in tokens[:24])


def buscar(consulta: str, k: int = config.TOP_K_LEXICO) -> list[dict[str, Any]]:
    """Devuelve los k fragmentos con mejor BM25. Puntaje mayor = mejor."""
    con = conexion()
    try:
        try:
            filas = con.execute(
                f"SELECT chunk_id, fuente, titulo, seccion, texto, "
                f"       bm25({_TABLA}) AS puntaje "
                f"FROM {_TABLA} WHERE {_TABLA} MATCH ? "
                f"ORDER BY puntaje LIMIT ?",
                (_consulta_fts(consulta), k),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        # bm25() de SQLite devuelve valores negativos donde mas negativo es
        # mejor; se invierte para que "mayor es mejor" en todo el proyecto.
        return [
            {
                "chunk_id": f["chunk_id"],
                "fuente": f["fuente"],
                "titulo": f["titulo"],
                "seccion": f["seccion"],
                "texto": f["texto"],
                "puntaje_lexico": -float(f["puntaje"]),
            }
            for f in filas
        ]
    finally:
        con.close()


def contar() -> int:
    con = conexion()
    try:
        try:
            return con.execute(f"SELECT COUNT(*) FROM {_TABLA}").fetchone()[0]
        except sqlite3.OperationalError:
            return 0
    finally:
        con.close()
