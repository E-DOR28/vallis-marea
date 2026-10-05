"""Orquestador: grafo de estados que enruta, delega por A2A y compone.

El grafo es deliberadamente explicito. Un agente ReAct de proposito general
habria sido menos codigo, pero el objetivo del proyecto es *medir* el costo de
la interoperabilidad, y para eso hacen falta fronteras nitidas: se sabe en que
nodo esta, a que agente delego y cuanto tardo cada salto.

    enrutar -> (segun intencion) -> conocimiento | disponibilidad |
                                    recomendador | directo        -> componer

El orquestador no tiene logica de negocio propia. No sabe precios, no sabe
politicas, no sabe que lanchas hay. Solo sabe a quien preguntarle. Esa es la
diferencia con el nodo monolitico que reemplaza.
"""

from __future__ import annotations

import re
import time
import unicodedata
import uuid
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from core import config
from core.a2a import cliente as a2a
from core.agentes.disponibilidad import logica as disp_util
from core.llm import gemini
from core.orquestador import contrato as contrato_mod
from core.router import predecir as router


# ---------------------------------------------------------------------------
# Estado
# ---------------------------------------------------------------------------
def _concatenar(a: list, b: list) -> list:
    return (a or []) + (b or [])


class Estado(TypedDict, total=False):
    texto: str
    context_id: str
    historial: list[dict[str, str]]
    prediccion: dict[str, Any]
    slots: dict[str, Any]
    saltos: Annotated[list[dict[str, Any]], _concatenar]
    datos_agente: dict[str, Any]
    contrato: dict[str, Any]
    t_inicio: float


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def _historial_texto(historial: list[dict[str, str]] | None, n: int = 6) -> str:
    if not historial:
        return ""
    recientes = historial[-n:]
    return "\n".join(
        f"{'Cliente' if t.get('rol') == 'usuario' else 'Agente'}: {t.get('texto', '')}"
        for t in recientes
    )


def _sin_acentos(texto: str) -> str:
    base = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in base if not unicodedata.combining(c))


# Gesto de confirmar en el mensaje actual. Es una segunda llave junto a la del
# LLM: el modelo puede arrastrar una "confirmacion" de un turno anterior o
# aceptar una orden escrita dentro del propio mensaje. Si falla, el costo es una
# pregunta de mas, no una reserva de mas.
_CUE_CONFIRMACION = re.compile(
    r"confirm|reserv|separ|apart|bloque|\bdale\b|hagamos|\blisto\b|de una\b|\bsi\b|\bok\b|"
    r"\bva\b|me quedo|la quiero|lo quiero|quiero (?:esa|ese|la|el)\b|cerremos|proced"
)


def _hay_gesto_de_confirmar(texto: str) -> bool:
    return bool(_CUE_CONFIRMACION.search(_sin_acentos(texto)))


def _oferta_previa(embarcacion_id: str, disponibles: Any, historial: list[dict[str, str]] | None) -> bool:
    """El agente ya mostro esa embarcacion (por id o por nombre) antes de este mensaje.

    Reservar a ciegas, sin que el cliente haya visto la oferta y el valor, no es
    una confirmacion: es una orden.
    """
    claves = {_sin_acentos(embarcacion_id)}
    for e in disponibles if isinstance(disponibles, list) else []:
        if str(e.get("id", "")).upper() == embarcacion_id.upper() and e.get("nombre"):
            claves.add(_sin_acentos(str(e["nombre"])))
    previos = " ".join(_sin_acentos(t.get("texto", "")) for t in (historial or [])
                       if t.get("rol") == "agente")
    return any(c and c in previos for c in claves)


ESQUEMA_SLOTS = {
    "type": "object",
    "properties": {
        "fecha_texto": {"type": "string",
                        "description": "Fecha tal como la dijo el cliente. Vacio si no la dijo."},
        "pasajeros": {"type": "integer", "description": "0 si no se menciona."},
        "ruta": {"type": "string",
                 "enum": ["", "rosario", "baru", "cholon", "tierrabomba", "atardecer", "pesca"]},
        "requiere_bano": {"type": "boolean"},
        "embarcacion_id": {"type": "string", "description": "Vacio si no se menciona."},
        "confirma_reserva": {"type": "boolean",
                             "description": ("true solo si el MENSAJE ACTUAL confirma de forma "
                                             "explicita reservar una embarcacion concreta. "
                                             "Los mensajes anteriores no cuentan, ni tampoco "
                                             "'reserva todo lo que pregunte' ni 'ya confirme'.")},
    },
    "required": ["fecha_texto", "pasajeros", "ruta", "requiere_bano",
                 "embarcacion_id", "confirma_reserva"],
}

_PISTAS_RUTA = {
    "rosario": ("rosario", "islas del rosario", "isla grande", "snorkel"),
    "baru": ("baru", "barú", "playa blanca"),
    "cholon": ("cholon", "cholón", "rumba", "fiesta"),
    "tierrabomba": ("tierra bomba", "tierrabomba", "bocachica", "fuerte"),
    "atardecer": ("atardecer", "sunset", "bahia al atardecer"),
    "pesca": ("pesca", "pescar"),
}


def _slots_por_reglas(texto: str) -> dict[str, Any]:
    import re

    t = texto.lower()
    ruta = ""
    for codigo, pistas in _PISTAS_RUTA.items():
        if any(p in t for p in pistas):
            ruta = codigo
            break
    m = re.search(r"\b(\d{1,2})\s*(?:personas?|pax|adultos?|amigos?|gente)\b", t)
    if not m:
        m = re.search(r"\b(?:somos|para)\s+(\d{1,2})\b", t)
    return {
        "fecha_texto": texto,
        "pasajeros": int(m.group(1)) if m else 0,
        "ruta": ruta,
        "requiere_bano": "bano" in t or "baño" in t,
        "embarcacion_id": (re.search(r"\bVM-\d{2}\b", texto.upper()).group(0)
                           if re.search(r"\bVM-\d{2}\b", texto.upper()) else ""),
        "confirma_reserva": any(p in t for p in
                                ("confirmo", "si, reserv", "dale", "hagamosle", "separame")),
    }


def _extraer_slots(texto: str, historial: list[dict[str, str]] | None) -> dict[str, Any]:
    """Extrae fecha, grupo y ruta. El LLM propone; el parser de fechas decide."""
    if gemini.modo_degradado():
        crudos = _slots_por_reglas(texto)
    else:
        contexto = _historial_texto(historial)
        datos, _ = gemini.generar_json(
            (f"Conversacion previa:\n{contexto}\n\n" if contexto else "")
            + f"Mensaje actual del cliente:\n{texto}\n\n"
            "Extrae los datos de la solicitud. Si la fecha, el grupo o la ruta "
            "aparecieron en un turno anterior y siguen vigentes, conservalos. "
            "La confirmacion de reserva NO se hereda: se decide solo con el mensaje actual.",
            ESQUEMA_SLOTS,
            sistema=("Extraes datos de solicitudes de alquiler de lanchas en Cartagena. "
                     "No inventes fechas ni cantidades: si no estan, deja vacio o 0."),
            temperatura=0.0,
        )
        crudos = datos or _slots_por_reglas(texto)

    # La fecha se normaliza siempre con el parser determinista. El LLM es bueno
    # ubicando "el sabado" en el texto y malo convirtiendolo a calendario.
    fecha = disp_util.interpretar_fecha(crudos.get("fecha_texto") or texto)
    fecha_pasada = None
    if fecha and fecha < disp_util.fecha_hoy().isoformat():
        fecha, fecha_pasada = None, fecha
    return {
        "fecha": fecha,
        "fecha_pasada": fecha_pasada,
        "fecha_texto": crudos.get("fecha_texto", ""),
        "pasajeros": int(crudos.get("pasajeros") or 0),
        "ruta": crudos.get("ruta") or "",
        "requiere_bano": bool(crudos.get("requiere_bano")),
        "embarcacion_id": crudos.get("embarcacion_id") or "",
        "confirma_reserva": bool(crudos.get("confirma_reserva")),
    }


# ---------------------------------------------------------------------------
# Nodos
# ---------------------------------------------------------------------------
def nodo_enrutar(estado: Estado) -> dict[str, Any]:
    p = router.predecir(estado["texto"])
    return {"prediccion": p.a_dict()}


def _ruta_desde_intencion(estado: Estado) -> str:
    intencion = estado["prediccion"]["intencion"]
    destino = config.INTENCIONES.get(intencion)
    return destino or "directo"


def nodo_conocimiento(estado: Estado) -> dict[str, Any]:
    salto = a2a.invocar(
        "conocimiento",
        "buscar_conocimiento",
        {"pregunta": estado["texto"], "historial": _historial_texto(estado.get("historial"))},
        context_id=estado.get("context_id"),
    )
    return {"saltos": [salto.a_dict()], "datos_agente": salto.resultado}


def nodo_recomendador(estado: Estado) -> dict[str, Any]:
    slots = _extraer_slots(estado["texto"], estado.get("historial"))
    salto = a2a.invocar(
        "recomendador",
        "recomendar",
        {
            "texto_cliente": estado["texto"],
            "pasajeros": slots["pasajeros"],
            "fecha": slots["fecha"] or "",
        },
        context_id=estado.get("context_id"),
    )
    return {"slots": slots, "saltos": [salto.a_dict()], "datos_agente": salto.resultado}


def nodo_disponibilidad(estado: Estado) -> dict[str, Any]:
    """Consulta inventario y, si el cliente confirmo, bloquea.

    Aqui viven los dos guardrails de escritura: no se bloquea nada sin
    confirmacion explicita, y si faltan datos se pregunta en vez de suponer.
    """
    slots = _extraer_slots(estado["texto"], estado.get("historial"))
    intencion = estado["prediccion"]["intencion"]
    saltos: list[dict[str, Any]] = []

    if not slots["fecha"]:
        sugerido = ("Falta la fecha del paseo." if not slots["fecha_pasada"] else
                    f"La fecha {slots['fecha_pasada']} ya paso. Pidele una fecha futura.")
        return {
            "slots": slots,
            "datos_agente": {"ok": False, "falta": "fecha", "mensaje_sugerido": sugerido},
        }

    salto = a2a.invocar(
        "disponibilidad",
        "consultar_disponibilidad",
        {
            "fecha": slots["fecha"],
            "pasajeros": max(slots["pasajeros"], 1),
            "ruta": slots["ruta"],
            "requiere_bano": slots["requiere_bano"],
        },
        context_id=estado.get("context_id"),
    )
    saltos.append(salto.a_dict())
    datos = salto.resultado

    # Bloqueo solo con confirmacion explicita sobre una oferta que el cliente ya vio,
    # embarcacion elegida y ruta clara.
    pide_reservar = bool(
        intencion == "reserva" and slots["embarcacion_id"] and slots["ruta"] and salto.ok
    )
    confirmada = bool(
        pide_reservar
        and slots["confirma_reserva"]
        and _hay_gesto_de_confirmar(estado["texto"])
        and _oferta_previa(slots["embarcacion_id"],
                           datos.get("disponibles") if isinstance(datos, dict) else None,
                           estado.get("historial"))
    )
    slots["confirmacion_valida"] = confirmada
    if pide_reservar and slots["confirma_reserva"] and not confirmada:
        datos = {**datos, "pendiente_confirmacion": (
            "El cliente pidio reservar, pero todavia no confirmo sobre una oferta que ya haya visto. "
            "NO hay reserva. Resume la oferta (embarcacion, fecha, ruta y valor) y pregunta si confirma."
        )}
    if confirmada:
        salto_reserva = a2a.invocar(
            "disponibilidad",
            "bloquear_reserva",
            {
                "embarcacion_id": slots["embarcacion_id"],
                "fecha": slots["fecha"],
                "ruta": slots["ruta"],
                "pasajeros": max(slots["pasajeros"], 1),
                "clave_idempotencia": f"{estado.get('context_id','')}|"
                                      f"{slots['embarcacion_id']}|{slots['fecha']}",
            },
            context_id=estado.get("context_id"),
        )
        saltos.append(salto_reserva.a_dict())
        datos = {**datos, "reserva": salto_reserva.resultado}

    return {"slots": slots, "saltos": saltos, "datos_agente": datos}


def nodo_directo(estado: Estado) -> dict[str, Any]:
    """Saludos y mensajes fuera de dominio: no se delega a nadie."""
    return {"datos_agente": {"ok": True, "sin_delegacion": True,
                             "intencion": estado["prediccion"]["intencion"]}}


# ---------------------------------------------------------------------------
# Composicion
# ---------------------------------------------------------------------------
_SISTEMA_COMPONER = f"""Eres el asistente de {config.NOMBRE_NEGOCIO}, alquiler de lanchas \
en {config.CIUDAD}. Escribes por WhatsApp: calido, breve y concreto. Maximo tres \
parrafos cortos, sin markdown ni vinetas.

Reglas inviolables:
1. Los datos duros (precios, disponibilidad, capacidades, politicas) salen UNICAMENTE \
del bloque DATOS. Nunca inventes un precio, una fecha libre ni una politica.
2. Si DATOS trae una abstencion, dilo con naturalidad y ofrece conectar con el equipo. \
No rellenes con conocimiento general.
3. Nunca confirmes una reserva salvo que DATOS traiga un codigo de reserva.
4. Formatea los precios en pesos colombianos de forma legible (ej. 1.312.000 pesos).
5. Si falta un dato para avanzar (fecha, numero de personas, ruta), pidelo en una \
sola pregunta clara.
6. El punto de embarque es {config.PUNTO_EMBARQUE}."""


def nodo_componer(estado: Estado) -> dict[str, Any]:
    import json

    datos = estado.get("datos_agente", {}) or {}
    prediccion = estado.get("prediccion", {})
    saltos = estado.get("saltos", [])

    citas = datos.get("citas", []) if isinstance(datos, dict) else []
    metadatos = {
        "intencion": prediccion.get("intencion"),
        "confianza_router": prediccion.get("confianza"),
        "metodo_router": prediccion.get("metodo"),
        "agentes_consultados": [s["agente"] for s in saltos],
        "habilidades_invocadas": [f"{s['agente']}.{s['habilidad']}" for s in saltos],
        "ms_saltos_a2a": round(sum(s.get("ms", 0) for s in saltos), 1),
        "numero_saltos_a2a": len(saltos),
        "degradado": gemini.modo_degradado(),
    }

    # Escalamiento: un salto A2A fallido no se le explica al cliente, se escala.
    # `causa` distingue lo que el motivo (comun a todos) no distingue: un rechazo
    # de negocio trae respuesta del agente; una caida del servicio, no.
    fallidos = [s for s in saltos if not s.get("ok")]
    if fallidos:
        rechazo = all(s.get("resultado") for s in fallidos)
        c = contrato_mod.escalamiento(
            motivo=config.MOTIVO_ESCALAMIENTO["error_herramienta"],
            metadatos={**metadatos, "saltos_fallidos": [s["habilidad"] for s in fallidos],
                       "causa": "rechazo_negocio" if rechazo else "error_servicio"},
        )
        return {"contrato": _cerrar(c, estado)}

    if datos.get("abstencion"):
        c = contrato_mod.escalamiento(
            motivo=config.MOTIVO_ESCALAMIENTO["baja_confianza"],
            mensaje=("No tengo ese dato confirmado de mi lado. Dejame verificarlo con "
                     "el equipo y te confirmo en un momento."),
            metadatos={**metadatos, "motivo_agente": datos.get("motivo_abstencion", ""),
                       "causa": "abstencion"},
        )
        return {"contrato": _cerrar(c, estado)}

    resp = gemini.generar(
        f"DATOS (unica fuente de verdad):\n"
        f"{json.dumps(datos, ensure_ascii=False, indent=2, default=str)[:6000]}\n\n"
        f"Conversacion previa:\n{_historial_texto(estado.get('historial')) or '(ninguna)'}\n\n"
        f"Mensaje del cliente:\n{estado['texto']}\n\n"
        f"Redacta la respuesta.",
        sistema=_SISTEMA_COMPONER,
        temperatura=config.TEMPERATURA_RESPUESTA,
    )

    if resp.error:
        c = contrato_mod.escalamiento(
            motivo=config.MOTIVO_ESCALAMIENTO["error_herramienta"],
            metadatos={**metadatos, "error_llm": resp.error, "causa": "error_llm"},
        )
        return {"contrato": _cerrar(c, estado)}

    acciones = []
    reserva = (datos.get("reserva") or {}) if isinstance(datos, dict) else {}
    if reserva.get("codigo_reserva"):
        acciones.append({
            "tipo": "reserva_bloqueada",
            "codigo": reserva["codigo_reserva"],
            "anticipo": reserva.get("anticipo_requerido"),
        })

    metadatos["tokens_entrada"] = resp.tokens_entrada
    metadatos["tokens_salida"] = resp.tokens_salida
    metadatos["modelo"] = resp.modelo

    c = contrato_mod.construir(
        mensaje=resp.texto.strip(),
        citas=citas,
        acciones=acciones,
        metadatos=metadatos,
    )
    return {"contrato": _cerrar(c, estado)}


def _cerrar(c: dict[str, Any], estado: Estado) -> dict[str, Any]:
    """Valida el contrato y sella la latencia total."""
    c["metadatos"]["ms_total"] = round((time.perf_counter() - estado["t_inicio"]) * 1000, 1)
    ok, problemas = contrato_mod.validar(c)
    c["metadatos"]["contrato_valido"] = ok
    if not ok:
        c["metadatos"]["problemas_contrato"] = problemas
    return c


# ---------------------------------------------------------------------------
# Grafo
# ---------------------------------------------------------------------------
def construir_grafo():
    g = StateGraph(Estado)
    g.add_node("enrutar", nodo_enrutar)
    g.add_node("conocimiento", nodo_conocimiento)
    g.add_node("disponibilidad", nodo_disponibilidad)
    g.add_node("recomendador", nodo_recomendador)
    g.add_node("directo", nodo_directo)
    g.add_node("componer", nodo_componer)

    g.add_edge(START, "enrutar")
    g.add_conditional_edges(
        "enrutar",
        _ruta_desde_intencion,
        {
            "conocimiento": "conocimiento",
            "disponibilidad": "disponibilidad",
            "recomendador": "recomendador",
            "directo": "directo",
        },
    )
    for nodo in ("conocimiento", "disponibilidad", "recomendador", "directo"):
        g.add_edge(nodo, "componer")
    g.add_edge("componer", END)
    return g.compile()


_GRAFO = None


def grafo():
    global _GRAFO
    if _GRAFO is None:
        _GRAFO = construir_grafo()
    return _GRAFO


def responder(
    texto: str,
    historial: list[dict[str, str]] | None = None,
    context_id: str | None = None,
) -> dict[str, Any]:
    """Procesa un turno completo. Devuelve el contrato con sus trazas."""
    estado_final = grafo().invoke({
        "texto": texto,
        "historial": historial or [],
        "context_id": context_id or str(uuid.uuid4()),
        "saltos": [],
        "t_inicio": time.perf_counter(),
    })
    c = estado_final["contrato"]

    # Aprendizaje continuo: cada turno resuelto queda como experiencia
    # recuperable. Nunca debe tumbar la respuesta al cliente.
    try:
        from core.aprendizaje import continuo

        continuo.registrar_episodio(
            texto,
            c.get("mensaje", ""),
            intencion=c.get("metadatos", {}).get("intencion", ""),
            context_id=estado_final.get("context_id", ""),
            exito=not c.get("escalar_a_humano", False),
            escalado=bool(c.get("escalar_a_humano", False)),
        )
    except Exception:
        pass

    c["_trazas"] = {
        "router": estado_final.get("prediccion", {}),
        "slots": estado_final.get("slots", {}),
        "saltos_a2a": estado_final.get("saltos", []),
    }
    return c
