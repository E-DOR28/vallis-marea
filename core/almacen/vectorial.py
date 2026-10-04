"""Busqueda densa sobre embeddings.

Produccion usa pgvector dentro del mismo Postgres. El demo usa Chroma
persistente en disco. La interfaz publica de este modulo (`inicializar`,
`indexar`, `buscar`, `contar`) es la unica superficie que el resto del proyecto
conoce, asi que migrar a pgvector es reescribir este archivo y nada mas.

Los embeddings se calculan siempre en `core.llm.gemini` y se pasan explicitos:
no se delega en la funcion de embedding por defecto de Chroma, porque el modelo
debe ser el mismo en indexacion y en consulta, y debe quedar registrado.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core import config

_COLECCION = config.COLECCION_CONOCIMIENTO
_cliente = None


def cliente():
    global _cliente
    if _cliente is None:
        import chromadb

        _cliente = chromadb.PersistentClient(path=str(config.RUTA_CHROMA))
    return _cliente


def _coleccion(crear: bool = True):
    c = cliente()
    if crear:
        return c.get_or_create_collection(
            name=_COLECCION,
            metadata={"hnsw:space": "cosine", "modelo": config.MODELO_EMBEDDING},
        )
    return c.get_collection(name=_COLECCION)


def inicializar(recrear: bool = False) -> None:
    if recrear:
        try:
            cliente().delete_collection(_COLECCION)
        except Exception:
            pass  # no existia
    _coleccion()


def indexar(fragmentos: list[dict[str, Any]], vectores: np.ndarray) -> int:
    """Carga fragmentos con sus vectores ya calculados."""
    if len(fragmentos) != len(vectores):
        raise ValueError(
            f"fragmentos ({len(fragmentos)}) y vectores ({len(vectores)}) no coinciden"
        )
    col = _coleccion()
    TAM_LOTE = 200
    for i in range(0, len(fragmentos), TAM_LOTE):
        lote = fragmentos[i : i + TAM_LOTE]
        col.upsert(
            ids=[f["id"] for f in lote],
            embeddings=[v.tolist() for v in vectores[i : i + TAM_LOTE]],
            documents=[f["texto"] for f in lote],
            metadatas=[
                {
                    "fuente": f["fuente"],
                    "titulo": f["titulo"],
                    "seccion": f["seccion"],
                    "categoria": f["categoria"],
                    "version": f["version"],
                    "fecha": f["fecha"],
                }
                for f in lote
            ],
        )
    return len(fragmentos)


def buscar(vector_consulta: np.ndarray, k: int = config.TOP_K_DENSO) -> list[dict[str, Any]]:
    """Devuelve los k fragmentos mas similares. Puntaje mayor = mejor."""
    try:
        col = _coleccion(crear=False)
    except Exception:
        return []
    if col.count() == 0:
        return []

    r = col.query(
        query_embeddings=[vector_consulta.tolist()],
        n_results=min(k, col.count()),
        include=["documents", "metadatas", "distances"],
    )
    salida: list[dict[str, Any]] = []
    for cid, doc, meta, dist in zip(
        r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0]
    ):
        salida.append(
            {
                "chunk_id": cid,
                "texto": doc,
                "fuente": meta.get("fuente", ""),
                "titulo": meta.get("titulo", ""),
                "seccion": meta.get("seccion", ""),
                "categoria": meta.get("categoria", ""),
                "version": meta.get("version", ""),
                "fecha": meta.get("fecha", ""),
                # Chroma devuelve distancia coseno (0 = identico).
                "puntaje_denso": 1.0 - float(dist),
            }
        )
    return salida


def contar() -> int:
    try:
        return _coleccion(crear=False).count()
    except Exception:
        return 0
