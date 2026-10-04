"""Cliente A2A del orquestador, con trazas por salto.

Cada delegacion queda registrada con su latencia. Esa medicion es el dato que
sostiene la parte critica del proyecto: cuanto cuesta, en milisegundos reales,
sustituir una llamada a funcion por un salto de protocolo entre agentes.

NOTA DE MEDICION -- importa para no publicar una cifra equivocada.
Una implementacion ingenua abre una conexion HTTP nueva y vuelve a descargar el
Agent Card en cada invocacion. Eso mide el costo del *descubrimiento repetido*,
no el de la delegacion, e inflaba el sobrecosto medido en dos ordenes de
magnitud (~1100 ms por salto contra ~10 ms reales).

Aqui se hace lo que haria un sistema en produccion:

  - un unico event loop en un hilo de fondo, vivo durante toda la sesion;
  - un cliente HTTP con conexiones persistentes por agente;
  - el Agent Card se descubre UNA vez y se cachea.

El descubrimiento se mide aparte, porque es un costo real pero de arranque, no
de cada mensaje. Confundir ambos es la diferencia entre "A2A cuesta 8000% mas"
y "A2A cuesta unos milisegundos por salto".
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx
from a2a.client import ClientConfig, create_client
from a2a.types import Message, Part, Role, SendMessageRequest

from core import config


@dataclass
class SaltoA2A:
    """Registro de una delegacion del orquestador a un agente par."""

    agente: str
    habilidad: str
    parametros: dict[str, Any]
    resultado: dict[str, Any] = field(default_factory=dict)
    ms: float = 0.0
    ok: bool = True
    error: str | None = None

    def a_dict(self) -> dict[str, Any]:
        return {
            "agente": self.agente,
            "habilidad": self.habilidad,
            "parametros": self.parametros,
            "ms": round(self.ms, 1),
            "ok": self.ok,
            "error": self.error,
            "resultado": self.resultado,
        }


# ---------------------------------------------------------------------------
# Event loop persistente
# ---------------------------------------------------------------------------
# Los clientes httpx y a2a quedan ligados al loop donde se crearon. Con
# asyncio.run() por llamada, el loop muere y la conexion cacheada queda
# inservible. Por eso se mantiene un unico loop vivo en un hilo de fondo.
_LOOP: asyncio.AbstractEventLoop | None = None
_LOCK = threading.Lock()


def _loop() -> asyncio.AbstractEventLoop:
    global _LOOP
    with _LOCK:
        if _LOOP is None or _LOOP.is_closed():
            _LOOP = asyncio.new_event_loop()
            threading.Thread(
                target=_LOOP.run_forever, daemon=True, name="a2a-loop"
            ).start()
        return _LOOP


def _correr(corrutina):
    """Ejecuta una corrutina en el loop persistente, desde codigo sincrono."""
    return asyncio.run_coroutine_threadsafe(corrutina, _loop()).result()


# ---------------------------------------------------------------------------
# Conexiones y descubrimiento, cacheados
# ---------------------------------------------------------------------------
_HTTP: dict[str, httpx.AsyncClient] = {}
_CLIENTES: dict[str, Any] = {}
_TARJETAS: dict[str, dict[str, Any]] = {}
_MS_DESCUBRIMIENTO: dict[str, float] = {}


async def _http(clave: str) -> httpx.AsyncClient:
    if clave not in _HTTP:
        _HTTP[clave] = httpx.AsyncClient(
            timeout=config.TIMEOUT_A2A_SEGUNDOS,
            limits=httpx.Limits(max_keepalive_connections=8, max_connections=16),
        )
    return _HTTP[clave]


async def _cliente_a2a(clave: str):
    """Cliente A2A del agente, creado una sola vez por sesion."""
    if clave not in _CLIENTES:
        t0 = time.perf_counter()
        _CLIENTES[clave] = await create_client(
            config.url_agente(clave),
            client_config=ClientConfig(
                streaming=False, httpx_client=await _http(clave)
            ),
        )
        _MS_DESCUBRIMIENTO[clave] = (time.perf_counter() - t0) * 1000
    return _CLIENTES[clave]


def reiniciar_conexiones() -> None:
    """Olvida clientes y tarjetas. Util si se reinician los agentes."""
    _CLIENTES.clear()
    _TARJETAS.clear()
    _MS_DESCUBRIMIENTO.clear()


def ms_descubrimiento() -> dict[str, float]:
    """Costo de arranque del descubrimiento, por agente. Se reporta aparte."""
    return {k: round(v, 1) for k, v in _MS_DESCUBRIMIENTO.items()}


# ---------------------------------------------------------------------------
# Descubrimiento
# ---------------------------------------------------------------------------
async def descubrir_async(
    clave: str, timeout: float = 5.0, usar_cache: bool = True
) -> dict[str, Any]:
    """Lee el Agent Card publicado por un agente par.

    Es el paso que hace real la interoperabilidad: el orquestador no tiene
    cableadas las capacidades del par, las lee de su tarjeta.
    """
    if usar_cache and clave in _TARJETAS:
        return {"ok": True, "url": config.url_agente(clave),
                "tarjeta": _TARJETAS[clave], "cacheada": True}

    url = f"{config.url_agente(clave)}{config.RUTA_AGENT_CARD}"
    try:
        async with httpx.AsyncClient(timeout=timeout) as http:
            r = await http.get(url)
            r.raise_for_status()
            tarjeta = r.json()
        _TARJETAS[clave] = tarjeta
        return {"ok": True, "url": url, "tarjeta": tarjeta, "cacheada": False}
    except Exception as exc:
        return {"ok": False, "url": url, "error": f"{exc.__class__.__name__}: {exc}"}


def descubrir(clave: str, timeout: float = 5.0, usar_cache: bool = True) -> dict[str, Any]:
    return _correr(descubrir_async(clave, timeout, usar_cache))


def habilidades_publicadas(clave: str) -> list[str]:
    """IDs de habilidad que el agente declara en su Agent Card."""
    d = descubrir(clave)
    if not d.get("ok"):
        return []
    return [s.get("id", "") for s in d["tarjeta"].get("skills", [])]


# ---------------------------------------------------------------------------
# Invocacion
# ---------------------------------------------------------------------------
async def invocar_async(
    clave: str,
    habilidad: str,
    parametros: dict[str, Any] | None = None,
    *,
    context_id: str | None = None,
) -> SaltoA2A:
    """Delega una habilidad en un agente par por A2A."""
    parametros = parametros or {}
    salto = SaltoA2A(agente=clave, habilidad=habilidad, parametros=parametros)

    sobre = json.dumps(
        {"habilidad": habilidad, "parametros": parametros}, ensure_ascii=False
    )

    try:
        # El cliente se resuelve ANTES de arrancar el cronometro: la primera
        # llamada paga el descubrimiento, y ese costo se reporta por separado.
        cliente = await _cliente_a2a(clave)

        t0 = time.perf_counter()
        peticion = SendMessageRequest(
            message=Message(
                message_id=str(uuid.uuid4()),
                context_id=context_id or str(uuid.uuid4()),
                role=Role.ROLE_USER,
                parts=[Part(text=sobre)],
            )
        )
        texto = ""
        async for respuesta in cliente.send_message(peticion):
            texto = _texto_de_respuesta(respuesta) or texto
            if texto:
                break
        salto.ms = (time.perf_counter() - t0) * 1000

        if not texto:
            salto.ok = False
            salto.error = "El agente no devolvio contenido."
            return salto

        try:
            salto.resultado = json.loads(texto)
        except json.JSONDecodeError:
            salto.resultado = {"ok": True, "texto": texto}
        salto.ok = bool(salto.resultado.get("ok", True))
        if not salto.ok:
            salto.error = salto.resultado.get("error")
        return salto

    except Exception as exc:
        # Una conexion cacheada puede haber muerto (agente reiniciado).
        _CLIENTES.pop(clave, None)
        salto.ok = False
        salto.error = f"{exc.__class__.__name__}: {exc}"
        return salto


def _texto_de_respuesta(respuesta: Any) -> str:
    """Extrae el texto de un StreamResponse, venga como Message o como Task."""
    mensaje = getattr(respuesta, "message", None)
    if mensaje is not None and getattr(mensaje, "parts", None):
        for parte in mensaje.parts:
            if getattr(parte, "text", ""):
                return parte.text

    tarea = getattr(respuesta, "task", None)
    if tarea is not None:
        estado = getattr(tarea, "status", None)
        msg_estado = getattr(estado, "update", None) or getattr(estado, "message", None)
        if msg_estado is not None and getattr(msg_estado, "parts", None):
            for parte in msg_estado.parts:
                if getattr(parte, "text", ""):
                    return parte.text
        for artefacto in getattr(tarea, "artifacts", []) or []:
            for parte in getattr(artefacto, "parts", []) or []:
                if getattr(parte, "text", ""):
                    return parte.text
    return ""


def invocar(
    clave: str,
    habilidad: str,
    parametros: dict[str, Any] | None = None,
    *,
    context_id: str | None = None,
) -> SaltoA2A:
    """Version sincrona, para LangGraph y Streamlit."""
    return _correr(invocar_async(clave, habilidad, parametros, context_id=context_id))


def precalentar() -> dict[str, float]:
    """Fuerza el descubrimiento de los tres agentes.

    Se llama al arrancar la interfaz o antes de medir, para que el costo de
    descubrimiento no contamine la latencia del primer mensaje del cliente.
    """
    for clave in config.AGENTES:
        try:
            _correr(_cliente_a2a(clave))
        except Exception:
            pass
    return ms_descubrimiento()


def estado_agentes(usar_cache: bool = False) -> dict[str, dict[str, Any]]:
    """Salud y habilidades de los tres agentes pares. Alimenta la UI."""
    salida: dict[str, dict[str, Any]] = {}
    for clave in config.AGENTES:
        d = descubrir(clave, timeout=2.0, usar_cache=usar_cache)
        salida[clave] = {
            "en_linea": d.get("ok", False),
            "url": config.url_agente(clave),
            "nombre": config.AGENTES[clave]["nombre"],
            "habilidades": (
                [s.get("id") for s in d["tarjeta"].get("skills", [])] if d.get("ok") else []
            ),
            "error": d.get("error"),
        }
    return salida
