"""Inferencia del router de intencion, con cascada de tres niveles.

    1. Clasificador entrenado sobre embeddings   (barato, ~ms)
    2. LLM como desempate si la confianza es baja (caro, ~cientos de ms)
    3. Reglas por palabras clave                  (ultimo recurso, sin red)

La cascada es el punto: no es que el clasificador reemplace al LLM, es que lo
reserva para los casos dudosos. En el informe se reporta cuantos mensajes
resuelve cada nivel, porque ahi esta el ahorro medible.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from core import config
from core.llm import gemini
from core.router.entrenar import RUTA_MODELO

_MODELO: dict[str, Any] | None = None
_AFINADO_ERROR: str | None = None  # si el afinado no carga, no se reintenta en cada mensaje


@dataclass
class Prediccion:
    intencion: str
    confianza: float
    metodo: str  # clasificador | llm | reglas
    ms: float = 0.0
    alternativas: list[tuple[str, float]] = None  # type: ignore[assignment]
    # Umbral propio del modelo que decidio; None -> config.UMBRAL_CONFIANZA_ROUTER.
    umbral: float | None = None

    def a_dict(self) -> dict[str, Any]:
        return {
            "intencion": self.intencion,
            "confianza": round(self.confianza, 3),
            "metodo": self.metodo,
            "ms": round(self.ms, 1),
            "alternativas": [(c, round(p, 3)) for c, p in (self.alternativas or [])[:3]],
            "agente_destino": config.INTENCIONES.get(self.intencion),
        }


def _cargar_modelo() -> dict[str, Any] | None:
    global _MODELO
    if _MODELO is not None:
        return _MODELO
    if not RUTA_MODELO.exists():
        return None
    d = np.load(RUTA_MODELO, allow_pickle=False)
    _MODELO = {
        "coef": d["coef"],
        "intercept": d["intercept"],
        "clases": d["clases"],
        "dim": int(d["dim"][0]),
        "degradado": bool(d["degradado"][0]),
    }
    return _MODELO


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _por_afinado(texto: str) -> Prediccion | None:
    """Nivel 1 con el encoder afinado (VM_ROUTER=afinado). None si no esta disponible."""
    global _AFINADO_ERROR
    if _AFINADO_ERROR is not None or not config.ROUTER_MODELO:
        return None
    try:
        from core.router import afinado

        m = afinado.modelo_actual(config.ROUTER_MODELO, config.HF_TOKEN)
        probas = m.probas([texto])[0]
    except Exception as exc:  # sin torch, sin red, revision invalida...
        _AFINADO_ERROR = f"{exc.__class__.__name__}: {exc}"
        print(f"[router] El modelo afinado no esta disponible ({_AFINADO_ERROR}); "
              "se usa el clasificador sobre embeddings.")
        return None
    orden = np.argsort(-probas)
    return Prediccion(
        intencion=m.clases[orden[0]],
        confianza=float(probas[orden[0]]),
        metodo="clasificador",
        alternativas=[(m.clases[i], float(probas[i])) for i in orden],
        umbral=m.umbral,
    )


def _por_clasificador(texto: str) -> Prediccion | None:
    if config.ROUTER_NIVEL1 == "afinado":
        p = _por_afinado(texto)
        if p is not None:
            return p
    modelo = _cargar_modelo()
    if modelo is None:
        return None
    # Un modelo entrenado con embeddings degradados y una inferencia con
    # embeddings de Gemini (o al reves) darian resultados sin sentido.
    if modelo["degradado"] != gemini.modo_degradado():
        return None

    v = gemini.embed_clasificacion([texto])[0]
    if v.shape[0] != modelo["dim"]:
        return None

    logits = modelo["coef"] @ v + modelo["intercept"]
    probas = _softmax(logits)
    orden = np.argsort(-probas)
    return Prediccion(
        intencion=str(modelo["clases"][orden[0]]),
        confianza=float(probas[orden[0]]),
        metodo="clasificador",
        alternativas=[(str(modelo["clases"][i]), float(probas[i])) for i in orden],
    )


ESQUEMA_INTENCION = {
    "type": "object",
    "properties": {
        "intencion": {"type": "string", "enum": sorted(config.INTENCIONES)},
        "confianza": {"type": "number"},
    },
    "required": ["intencion", "confianza"],
}

_SISTEMA_ROUTER = f"""Clasificas el mensaje de un cliente de un negocio de alquiler de \
lanchas en Cartagena en UNA de estas intenciones:

- saludo: saludos, despedidas, agradecimientos.
- disponibilidad: pregunta si hay lancha libre en una fecha.
- precio: pregunta cuanto cuesta o pide cotizacion.
- reserva: quiere reservar, confirmar, cancelar o reprogramar una reserva.
- politica: cancelaciones, reembolsos, requisitos de abordaje, seguridad, normas.
- faq: que incluye, formas de pago, que llevar, logistica general.
- ruta_turistica: pregunta como es un destino o que paseos existen.
- recomendacion: pide que le sugieran un plan o una embarcacion.
- otro: cualquier cosa ajena al negocio.

Responde solo la intencion y tu confianza de 0 a 1."""


def _por_llm(texto: str) -> Prediccion | None:
    if gemini.modo_degradado():
        return None
    datos, resp = gemini.generar_json(
        f"Mensaje del cliente: {texto}",
        ESQUEMA_INTENCION,
        sistema=_SISTEMA_ROUTER,
        temperatura=config.TEMPERATURA_ROUTER,
        modelo=config.MODELO_GENERACION_RAPIDO,
    )
    if not datos or datos.get("intencion") not in config.INTENCIONES:
        return None
    return Prediccion(
        intencion=datos["intencion"],
        confianza=float(datos.get("confianza", 0.7)),
        metodo="llm",
    )


_REGLAS: list[tuple[str, str]] = [
    (r"\b(hola|buenas|buen d|saludos|gracias|chao|hasta luego)\b", "saludo"),
    (r"\b(reserv|apart|separ|confirm|anticipo|cancelar mi|reprogram)\w*", "reserva"),
    (r"\b(cuanto|precio|tarifa|vale|cuesta|cotiz|cobran|descuento)\w*", "precio"),
    (r"\b(disponib|libre|cupo|queda|ocupad)\w*", "disponibilidad"),
    (r"\b(politic|reembols|devuelv|cancelaci|document|chalec|embarazad|mascota)\w*",
     "politica"),
    (r"\b(recomend|sugier|sugerenc|mejor opcion|que me conviene)\w*", "recomendacion"),
    (r"\b(rosario|baru|cholon|playa blanca|tierra bomba|atardecer|pesca|ruta|paseo|"
     r"destino)\w*", "ruta_turistica"),
    (r"\b(incluye|pago|pagar|tarjeta|nequi|llevar|bano|muelle|almuerzo|dura)\w*", "faq"),
]


def _por_reglas(texto: str) -> Prediccion:
    t = texto.lower()
    for patron, intencion in _REGLAS:
        if re.search(patron, t):
            return Prediccion(intencion=intencion, confianza=0.35, metodo="reglas")
    return Prediccion(intencion="otro", confianza=0.2, metodo="reglas")


def predecir(texto: str) -> Prediccion:
    """Clasifica la intencion con la cascada completa."""
    t0 = time.perf_counter()

    p = _por_clasificador(texto)
    if p is not None and p.confianza >= (
        p.umbral if p.umbral is not None else config.UMBRAL_CONFIANZA_ROUTER
    ):
        p.ms = (time.perf_counter() - t0) * 1000
        return p

    p_llm = _por_llm(texto)
    if p_llm is not None:
        p_llm.ms = (time.perf_counter() - t0) * 1000
        # Se conserva el ranking del clasificador como evidencia en la traza.
        p_llm.alternativas = p.alternativas if p else []
        return p_llm

    if p is not None:
        p.metodo = "clasificador_baja_confianza"
        p.ms = (time.perf_counter() - t0) * 1000
        return p

    p_reglas = _por_reglas(texto)
    p_reglas.ms = (time.perf_counter() - t0) * 1000
    return p_reglas


def agente_para(intencion: str) -> str | None:
    return config.INTENCIONES.get(intencion)
