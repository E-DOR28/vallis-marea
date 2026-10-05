"""Validacion de arranque: los Agent Cards publican lo que el orquestador invoca.

El orquestador pide cuatro habilidades por su nombre, escrito en `grafo.py`. Los
Agent Cards no deciden a quien llamar, asi que si un agente renombra o retira una
habilidad el fallo apareceria en medio de una conversacion, disfrazado de
escalamiento a un humano. Esta comprobacion lo hace visible al arrancar.

La lista `HABILIDADES_REQUERIDAS` debe coincidir con las llamadas `a2a.invocar`
de `grafo.py`; `documentacion/diagramas/verificar_trazas.py` lo comprueba.
"""

from __future__ import annotations

import logging
from typing import Any

from core.a2a import cliente as a2a

log = logging.getLogger("vallis.validacion")

HABILIDADES_REQUERIDAS: dict[str, tuple[str, ...]] = {
    "conocimiento": ("buscar_conocimiento",),
    "disponibilidad": ("consultar_disponibilidad", "bloquear_reserva"),
    "recomendador": ("recomendar",),
}


def validar_agent_cards(usar_cache: bool = False) -> dict[str, Any]:
    """Lee el Agent Card de cada agente y compara con las habilidades requeridas.

    Devuelve `estado` "listo" si todo coincide y "degradado" si falta un agente o
    una habilidad. Los faltantes se registran con `log.error`, que es lo que
    `/api/salud` mostrara.
    """
    agentes: dict[str, dict[str, Any]] = {}
    faltantes: list[str] = []
    sin_respuesta: list[str] = []

    for clave, requeridas in HABILIDADES_REQUERIDAS.items():
        d = a2a.descubrir(clave, timeout=2.0, usar_cache=usar_cache)
        if not d.get("ok"):
            sin_respuesta.append(clave)
            agentes[clave] = {"en_linea": False, "faltantes": list(requeridas),
                              "error": d.get("error")}
            log.error("Agent Card de '%s' no disponible: %s", clave, d.get("error"))
            continue

        publicadas = {s.get("id") for s in d["tarjeta"].get("skills", [])}
        ausentes = [h for h in requeridas if h not in publicadas]
        agentes[clave] = {"en_linea": True, "faltantes": ausentes, "error": None}
        for h in ausentes:
            faltantes.append(f"{clave}.{h}")
            log.error("El Agent Card de '%s' no publica la habilidad '%s' que "
                      "el orquestador invoca", clave, h)

    ok = not faltantes and not sin_respuesta
    return {
        "ok": ok,
        "estado": "listo" if ok else "degradado",
        "agentes": agentes,
        "faltantes": faltantes,
        "sin_respuesta": sin_respuesta,
    }
