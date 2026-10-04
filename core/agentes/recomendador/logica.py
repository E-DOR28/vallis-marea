"""Agente Recomendador: empareja preferencias del cliente con el catalogo.

Usa embeddings de forma distinta al RAG. El RAG recupera *texto para citar*;
aqui se compara la descripcion en lenguaje natural de lo que el cliente quiere
contra una representacion textual de cada ruta y cada embarcacion. Es
recuperacion semantica sobre datos estructurados, no sobre documentos.

La ventaja frente a filtrar por etiquetas: el cliente no dice "rumba", dice
"algo para celebrar con mis amigos y que suene musica". El embedding cruza esa
distancia; un filtro por tags no.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.almacen import estructurado
from core.llm import gemini

ESQUEMA_PERFIL = {
    "type": "object",
    "properties": {
        "resumen_preferencias": {
            "type": "string",
            "description": "Que busca el cliente, en una frase, con sus propias palabras.",
        },
        "pasajeros": {"type": "integer", "description": "0 si no se menciona."},
        "ocasion": {
            "type": "string",
            "description": "familia, pareja, amigos, despedida, corporativo, pesca u otro.",
        },
        "prioriza_precio": {"type": "boolean"},
        "requiere_bano": {"type": "boolean"},
    },
    "required": ["resumen_preferencias", "pasajeros", "ocasion", "prioriza_precio",
                 "requiere_bano"],
}


def extraer_perfil(texto: str) -> dict[str, Any]:
    """Convierte el mensaje del cliente en un perfil de preferencias."""
    if gemini.modo_degradado():
        t = texto.lower()
        return {
            "resumen_preferencias": texto,
            "pasajeros": 0,
            "ocasion": ("despedida" if any(p in t for p in ("despedida", "rumba", "fiesta"))
                        else "familia" if any(p in t for p in ("familia", "ninos", "niños"))
                        else "pareja" if any(p in t for p in ("pareja", "aniversario", "romantic"))
                        else "pesca" if "pesca" in t else "otro"),
            "prioriza_precio": any(p in t for p in ("economic", "barat", "presupuesto")),
            "requiere_bano": "bano" in t or "baño" in t,
            "_degradado": True,
        }
    datos, _ = gemini.generar_json(
        f"Mensaje del cliente:\n{texto}\n\nExtrae sus preferencias de paseo en lancha.",
        ESQUEMA_PERFIL,
        sistema=("Extraes preferencias de clientes de un negocio de alquiler de lanchas "
                 "en Cartagena. No inventes: si un dato no esta, usa 0, false o 'otro'."),
    )
    return datos or {"resumen_preferencias": texto, "pasajeros": 0, "ocasion": "otro",
                     "prioriza_precio": False, "requiere_bano": False}


def _texto_ruta(r: dict[str, Any]) -> str:
    return (f"Paseo {r['nombre']}. {r['resumen']} "
            f"Ideal para: {', '.join(r['tags'])}. "
            f"Duracion {r['duracion'].replace('_', ' ')}, "
            f"zarpe {r['hora_zarpe']}, regreso {r['hora_regreso']}.")


def _texto_embarcacion(e: dict[str, Any]) -> str:
    return (f"Embarcacion {e['nombre']}, {e['tipo']} de {e['eslora_pies']} pies para "
            f"{e['capacidad']} pasajeros. "
            f"{'Con bano a bordo. ' if e['bano'] else 'Sin bano a bordo. '}"
            f"Sombra: {e['sombra']}. Sonido: {e['sonido']}. "
            f"Caracteristicas: {', '.join(e['caracteristicas'])}. {e['notas'] or ''}")


def _similitudes(consulta: str, textos: list[str]) -> np.ndarray:
    """Coseno entre la consulta y cada item. Los vectores ya vienen normalizados."""
    v_consulta = gemini.embed_consulta(consulta)
    matriz = gemini.embed_documentos(textos)
    return matriz @ v_consulta


def recomendar(
    texto_cliente: str,
    pasajeros: int | None = None,
    fecha: str | None = None,
    top_n: int = 3,
) -> dict[str, Any]:
    """Recomienda rutas y embarcaciones a partir de lo que el cliente describe."""
    perfil = extraer_perfil(texto_cliente)
    pasajeros = pasajeros or (perfil.get("pasajeros") or 0) or None
    consulta = perfil.get("resumen_preferencias") or texto_cliente

    rutas = estructurado.listar_rutas()
    sim_rutas = _similitudes(consulta, [_texto_ruta(r) for r in rutas])
    orden_rutas = np.argsort(-sim_rutas)

    rutas_reco = []
    for i in orden_rutas[:top_n]:
        r = rutas[int(i)]
        rutas_reco.append({
            "codigo": r["codigo"],
            "nombre": r["nombre"],
            "resumen": r["resumen"],
            "duracion": r["duracion"],
            "hora_zarpe": r["hora_zarpe"],
            "afinidad": round(float(sim_rutas[int(i)]), 4),
            "tags": r["tags"],
        })

    # Las embarcaciones se filtran por restricciones duras antes de ordenar por
    # afinidad: recomendar una lancha que no cabe el grupo no es una sugerencia,
    # es un error.
    embarcaciones = estructurado.listar_embarcaciones()
    ocupadas = estructurado.embarcaciones_ocupadas(fecha) if fecha else set()
    candidatas = [
        e for e in embarcaciones
        if (not pasajeros or e["capacidad"] >= pasajeros)
        and e["id"] not in ocupadas
        and (not perfil.get("requiere_bano") or e["bano"])
    ]
    descartadas = len(embarcaciones) - len(candidatas)

    embarcaciones_reco = []
    if candidatas:
        sim_emb = _similitudes(consulta, [_texto_embarcacion(e) for e in candidatas])
        if perfil.get("prioriza_precio"):
            # Penaliza tarifa alta cuando el cliente dijo que le importa el precio.
            maximo = max(e["tarifa_dia_completo"] for e in candidatas)
            penalizacion = np.array(
                [e["tarifa_dia_completo"] / maximo for e in candidatas], dtype=np.float32
            )
            sim_emb = sim_emb - 0.15 * penalizacion
        for i in np.argsort(-sim_emb)[:top_n]:
            e = candidatas[int(i)]
            embarcaciones_reco.append({
                "id": e["id"],
                "nombre": e["nombre"],
                "capacidad": e["capacidad"],
                "bano": e["bano"],
                "tipo": e["tipo"],
                "tarifa_dia_completo": e["tarifa_dia_completo"],
                "tarifa_medio_dia": e["tarifa_medio_dia"],
                "afinidad": round(float(sim_emb[int(i)]), 4),
                "por_que": e["notas"] or ", ".join(e["caracteristicas"]),
            })

    return {
        "ok": True,
        "perfil_detectado": perfil,
        "rutas_recomendadas": rutas_reco,
        "embarcaciones_recomendadas": embarcaciones_reco,
        "embarcaciones_descartadas": descartadas,
        "nota": (
            "Recomendacion por afinidad semantica. No confirma disponibilidad: "
            "eso lo resuelve el Agente de Disponibilidad."
        ),
    }
