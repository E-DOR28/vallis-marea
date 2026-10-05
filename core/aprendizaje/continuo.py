"""Aprendizaje continuo: aprender de la interaccion sin reentrenar desde cero.

En Vallis Marea ya existe la senal de aprendizaje mas valiosa y nadie la esta
usando: cada vez que un operador interviene en Chatwoot para corregir o
completar lo que dijo el bot, esta produciendo un ejemplo etiquetado por un
experto del negocio, gratis.

Este modulo convierte esa senal en dos mecanismos:

1. **Correcciones de intencion.** Se acumulan en un JSONL y se integran al
   dataset del router. Reentrenar es entonces incremental sobre datos nuevos,
   no una reconstruccion desde cero.

2. **Memoria episodica.** Cada turno resuelto se indexa con su embedding. Ante
   un mensaje nuevo, el orquestador puede recuperar como se resolvieron casos
   parecidos. Es aprendizaje sin gradientes: el sistema mejora porque acumula
   experiencia recuperable, no porque cambien sus pesos.

Limite honesto, para el informe: en esta entrega el ciclo esta implementado y
se puede ejecutar, pero no se ha alimentado con volumen real de produccion. Lo
que se demuestra es el mecanismo, no una curva de mejora.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

import numpy as np

from core import config
from core.llm import gemini

RUTA_CORRECCIONES = config.RUTA_APRENDIZAJE / "correcciones.jsonl"
RUTA_EPISODIOS = config.RUTA_APRENDIZAJE / "memoria_episodica.jsonl"
_COLECCION_EPISODICA = "vallis_episodios"


# ---------------------------------------------------------------------------
# 1. Correcciones de intencion
# ---------------------------------------------------------------------------
def registrar_correccion(
    texto: str,
    intencion_predicha: str,
    intencion_correcta: str,
    *,
    confianza_predicha: float = 0.0,
    operador: str = "chatwoot",
    nota: str = "",
) -> dict[str, Any]:
    """Guarda una correccion hecha por un humano."""
    if intencion_correcta not in config.INTENCIONES:
        return {"ok": False,
                "error": f"Intencion invalida. Validas: {sorted(config.INTENCIONES)}"}

    registro = {
        "id": str(uuid.uuid4())[:8],
        "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
        "texto": texto.strip(),
        "intencion_predicha": intencion_predicha,
        "intencion_correcta": intencion_correcta,
        "confianza_predicha": round(confianza_predicha, 3),
        "acierto": intencion_predicha == intencion_correcta,
        "operador": operador,
        "nota": nota,
    }
    with RUTA_CORRECCIONES.open("a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")
    return {"ok": True, "registro": registro}


def leer_correcciones() -> list[dict[str, Any]]:
    if not RUTA_CORRECCIONES.exists():
        return []
    filas = []
    for linea in RUTA_CORRECCIONES.read_text(encoding="utf-8").splitlines():
        if linea.strip():
            try:
                filas.append(json.loads(linea))
            except json.JSONDecodeError:
                continue
    return filas


def estadisticas_correcciones() -> dict[str, Any]:
    """Salud del router segun el juicio humano acumulado."""
    filas = leer_correcciones()
    if not filas:
        return {"total": 0, "errores": 0, "tasa_acierto": None}
    errores = [f for f in filas if not f["acierto"]]
    confusiones: dict[str, int] = {}
    for f in errores:
        clave = f"{f['intencion_predicha']} -> {f['intencion_correcta']}"
        confusiones[clave] = confusiones.get(clave, 0) + 1
    return {
        "total": len(filas),
        "errores": len(errores),
        "tasa_acierto": round(1 - len(errores) / len(filas), 4),
        "confusiones_frecuentes": dict(
            sorted(confusiones.items(), key=lambda kv: -kv[1])[:5]
        ),
    }


def reentrenar_con_correcciones(minimo: int = 10) -> dict[str, Any]:
    """Integra las correcciones al dataset y reentrena el router.

    Solo las correcciones donde el humano discrepo del modelo: los aciertos ya
    estan representados y agregarlos solo refuerza el sesgo existente.
    """
    from core.router import datos, entrenar

    errores = [f for f in leer_correcciones() if not f["acierto"]]
    if len(errores) < minimo:
        return {
            "ok": False,
            "motivo": f"Hay {len(errores)} correcciones utiles; se requieren {minimo}.",
            "correcciones_disponibles": len(errores),
        }

    d = datos.construir_dataset()
    existentes = {e["texto"].strip().lower() for e in d["ejemplos"]}
    nuevos = [
        {"texto": f["texto"], "intencion": f["intencion_correcta"],
         "origen": "correccion_humana"}
        for f in errores
        if f["texto"].strip().lower() not in existentes
    ]
    if not nuevos:
        return {"ok": False, "motivo": "Las correcciones ya estaban en el dataset."}

    d["ejemplos"].extend(nuevos)
    d["conteo_por_clase"] = {
        c: sum(1 for e in d["ejemplos"] if e["intencion"] == c) for c in d["clases"]
    }
    datos.RUTA_DATASET.write_text(
        json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    metricas = entrenar.entrenar(verbose=False)
    return {
        "ok": True,
        "ejemplos_incorporados": len(nuevos),
        "dataset_total": len(d["ejemplos"]),
        "f1_macro": metricas["f1_macro"],
        "exactitud": metricas["exactitud"],
    }


# ---------------------------------------------------------------------------
# 2. Memoria episodica
# ---------------------------------------------------------------------------
def _coleccion_episodica():
    from core.almacen.vectorial import cliente

    return cliente().get_or_create_collection(
        name=_COLECCION_EPISODICA,
        metadata={"hnsw:space": "cosine", "modelo": config.MODELO_EMBEDDING},
    )


def registrar_episodio(
    texto_cliente: str,
    respuesta: str,
    *,
    intencion: str = "",
    context_id: str = "",
    exito: bool = True,
    escalado: bool = False,
) -> dict[str, Any]:
    """Indexa un turno resuelto como experiencia recuperable.

    Con VM_APRENDIZAJE=0 no hace nada: en el despliegue publico la memoria
    episodica no se escribe, para que lo que un visitante teclea no pueda
    influir en las respuestas de otro.
    """
    if not config.APRENDIZAJE_ACTIVO:
        return {"ok": True, "omitido": True}
    episodio = {
        "id": str(uuid.uuid4())[:12],
        "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
        "texto_cliente": texto_cliente.strip(),
        "respuesta": respuesta.strip(),
        "intencion": intencion,
        "context_id": context_id,
        "exito": exito,
        "escalado": escalado,
    }
    with RUTA_EPISODIOS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(episodio, ensure_ascii=False) + "\n")

    try:
        vector = gemini.embed_consulta(texto_cliente)
        _coleccion_episodica().upsert(
            ids=[episodio["id"]],
            embeddings=[vector.tolist()],
            documents=[texto_cliente],
            metadatas=[{
                "respuesta": respuesta[:1500],
                "intencion": intencion,
                "exito": exito,
                "escalado": escalado,
                "fecha": episodio["fecha"],
            }],
        )
        episodio["indexado"] = True
    except Exception as exc:
        episodio["indexado"] = False
        episodio["error_indexado"] = f"{exc.__class__.__name__}: {exc}"

    return {"ok": True, "episodio": episodio}


def recuperar_similares(texto: str, k: int = 3, solo_exitosos: bool = True) -> list[dict]:
    """Turnos pasados parecidos, para que el orquestador se apoye en ellos."""
    try:
        col = _coleccion_episodica()
        if col.count() == 0:
            return []
        filtro = {"exito": True} if solo_exitosos else None
        r = col.query(
            query_embeddings=[gemini.embed_consulta(texto).tolist()],
            n_results=min(k, col.count()),
            where=filtro,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        return []

    salida = []
    for doc, meta, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0]):
        salida.append({
            "texto_cliente": doc,
            "respuesta": meta.get("respuesta", ""),
            "intencion": meta.get("intencion", ""),
            "similitud": round(1.0 - float(dist), 4),
        })
    return salida


def estadisticas_memoria() -> dict[str, Any]:
    try:
        total_indexados = _coleccion_episodica().count()
    except Exception:
        total_indexados = 0
    episodios = []
    if RUTA_EPISODIOS.exists():
        for linea in RUTA_EPISODIOS.read_text(encoding="utf-8").splitlines():
            if linea.strip():
                try:
                    episodios.append(json.loads(linea))
                except json.JSONDecodeError:
                    continue
    return {
        "episodios_registrados": len(episodios),
        "episodios_indexados": total_indexados,
        "escalados": sum(1 for e in episodios if e.get("escalado")),
        "tasa_resolucion_autonoma": (
            round(1 - sum(1 for e in episodios if e.get("escalado")) / len(episodios), 4)
            if episodios else None
        ),
    }


def resumen() -> dict[str, Any]:
    """Estado del ciclo de aprendizaje. Alimenta la interfaz y el informe."""
    return {
        "correcciones": estadisticas_correcciones(),
        "memoria_episodica": estadisticas_memoria(),
    }
