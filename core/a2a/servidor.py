"""Capa A2A: cada agente par expuesto como servicio interoperable.

El punto central del diseno esta en `EjecutorMCP`: el ejecutor A2A no
reimplementa nada, sino que despacha sobre el **mismo** servidor MCP del agente.
De ahi que cada capacidad quede publicada dos veces sin duplicar logica:

    MCP  -> cualquier cliente compatible (Claude Desktop, un IDE, otro agente)
    A2A  -> el orquestador de Vallis Marea, u otro agente de otro proveedor

Eso es exactamente lo que la propuesta prometia demostrar, y es la razon por la
que el desacoplamiento se puede medir: la capacidad es una sola, los protocolos
de acceso son dos.

El sobre A2A que viaja es un Message con una parte de texto que contiene JSON:

    {"habilidad": "consultar_disponibilidad",
     "parametros": {"fecha": "2026-09-19", "pasajeros": 12}}

La respuesta viaja igual, con el resultado de la herramienta MCP.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import add_a2a_routes_to_fastapi, create_agent_card_routes, \
    create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    Message,
    Part,
    Role,
)
from a2a.utils import DEFAULT_RPC_URL, TransportProtocol
from fastapi import FastAPI
from mcp.server.mcpserver import MCPServer

from core import config
from core.agentes import servidores_mcp


# ---------------------------------------------------------------------------
# Agent Card
# ---------------------------------------------------------------------------
# Descripciones por habilidad. Son lo que otro agente lee para decidir a quien
# delegar, asi que se escriben pensando en ese lector, no en un humano.
_HABILIDADES: dict[str, dict[str, dict[str, Any]]] = {
    "conocimiento": {
        "buscar_conocimiento": {
            "nombre": "Responder con el corpus citando la fuente",
            "descripcion": (
                "Responde preguntas sobre politicas de cancelacion, requisitos de "
                "abordaje, FAQs, rutas y flota, usando RAG hibrido sobre el corpus "
                "indexado. Siempre cita la fuente o se abstiene explicitamente."
            ),
            "tags": ["rag", "politicas", "faq", "rutas", "citacion"],
            "ejemplos": [
                "Cuanto me devuelven si cancelo con un dia de anticipacion",
                "Puedo llevar mi mascota a bordo",
                "Que incluye el alquiler",
            ],
        },
        "obtener_fragmentos": {
            "nombre": "Recuperar evidencia cruda del corpus",
            "descripcion": (
                "Devuelve los fragmentos mas relevantes sin generar respuesta, "
                "para que otro agente razone sobre la evidencia."
            ),
            "tags": ["rag", "evidencia", "recuperacion"],
            "ejemplos": ["Fragmentos sobre impuesto de muelle"],
        },
    },
    "disponibilidad": {
        "consultar_disponibilidad": {
            "nombre": "Consultar inventario libre",
            "descripcion": (
                "Lista embarcaciones libres en una fecha que cubran al grupo, con "
                "precio calculado. Unica fuente valida de disponibilidad."
            ),
            "tags": ["inventario", "disponibilidad", "precios"],
            "ejemplos": ["Hay lancha para 12 personas el 19 de septiembre"],
        },
        "cotizar": {
            "nombre": "Cotizar una embarcacion",
            "descripcion": "Desglosa tarifa, temporada, descuento y anticipo.",
            "tags": ["precio", "cotizacion"],
            "ejemplos": ["Cuanto vale la VM-05 para Baru el 10 de octubre"],
        },
        "bloquear_reserva": {
            "nombre": "Bloquear una reserva",
            "descripcion": (
                "Bloquea inventario de forma idempotente. Operacion con efecto "
                "real; requiere confirmacion explicita del cliente."
            ),
            "tags": ["reserva", "escritura", "idempotente"],
            "ejemplos": ["Confirma la reserva de la Manglar para el sabado"],
        },
        "interpretar_fecha": {
            "nombre": "Normalizar fechas en espanol",
            "descripcion": "Convierte 'el sabado' o '15 de octubre' a YYYY-MM-DD.",
            "tags": ["fechas", "nlp"],
            "ejemplos": ["el proximo sabado"],
        },
        "listar_rutas": {
            "nombre": "Listar rutas operadas",
            "descripcion": "Catalogo de rutas con horarios y duracion.",
            "tags": ["rutas", "catalogo"],
            "ejemplos": ["Que paseos ofrecen"],
        },
    },
    "recomendador": {
        "recomendar": {
            "nombre": "Recomendar ruta y embarcacion",
            "descripcion": (
                "Sugiere paseo y embarcacion por afinidad semantica con lo que el "
                "cliente describe, filtrando capacidad y bano. No confirma "
                "disponibilidad."
            ),
            "tags": ["recomendacion", "embeddings", "preferencias"],
            "ejemplos": [
                "Somos 8 amigos y queremos algo con musica para celebrar",
                "Busco un plan tranquilo con ninos pequenos",
            ],
        },
        "extraer_perfil": {
            "nombre": "Extraer perfil de preferencias",
            "descripcion": "Estructura ocasion, grupo, sensibilidad al precio y bano.",
            "tags": ["perfil", "extraccion"],
            "ejemplos": ["Vamos mi esposa y yo por el aniversario"],
        },
    },
}


def construir_agent_card(clave: str) -> AgentCard:
    """Agent Card del agente, publicado en /.well-known/agent-card.json."""
    meta = config.AGENTES[clave]
    url = config.url_agente(clave)

    habilidades = [
        AgentSkill(
            id=hid,
            name=h["nombre"],
            description=h["descripcion"],
            tags=h["tags"],
            examples=h["ejemplos"],
            input_modes=["text/plain", "application/json"],
            output_modes=["application/json"],
        )
        for hid, h in _HABILIDADES[clave].items()
    ]

    return AgentCard(
        name=meta["nombre"],
        description=meta["descripcion"],
        version=config.VERSION_AGENTES,
        supported_interfaces=[
            AgentInterface(url=url, protocol_binding=TransportProtocol.JSONRPC.value)
        ],
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        default_input_modes=["text/plain", "application/json"],
        default_output_modes=["application/json"],
        skills=habilidades,
    )


# ---------------------------------------------------------------------------
# Ejecutor: puente A2A -> MCP
# ---------------------------------------------------------------------------
class EjecutorMCP(AgentExecutor):
    """Traduce una peticion A2A en una invocacion de herramienta MCP."""

    def __init__(self, clave: str, servidor_mcp: MCPServer) -> None:
        self.clave = clave
        self.mcp = servidor_mcp

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        entrada = context.get_user_input() or ""
        try:
            payload = json.loads(entrada)
            habilidad = payload["habilidad"]
            parametros = payload.get("parametros", {}) or {}
        except (json.JSONDecodeError, KeyError, TypeError):
            # Un agente par externo podria mandar texto plano. Se responde con
            # un error explicito en vez de adivinar la intencion.
            await self._responder(
                context,
                event_queue,
                {
                    "ok": False,
                    "error": (
                        "Se esperaba JSON con la forma "
                        '{"habilidad": "...", "parametros": {...}}'
                    ),
                    "habilidades_disponibles": list(_HABILIDADES[self.clave]),
                },
            )
            return

        if habilidad not in _HABILIDADES[self.clave]:
            await self._responder(
                context,
                event_queue,
                {
                    "ok": False,
                    "error": f"El agente '{self.clave}' no expone '{habilidad}'.",
                    "habilidades_disponibles": list(_HABILIDADES[self.clave]),
                },
            )
            return

        try:
            resultado = await self.mcp.call_tool(habilidad, parametros)
            await self._responder(context, event_queue, _extraer_resultado(resultado))
        except Exception as exc:
            await self._responder(
                context,
                event_queue,
                {"ok": False, "error": f"{exc.__class__.__name__}: {exc}",
                 "habilidad": habilidad},
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        await self._responder(
            context, event_queue,
            {"ok": False, "error": "Cancelacion no soportada por este agente."},
        )

    async def _responder(
        self, context: RequestContext, event_queue: EventQueue, datos: dict[str, Any]
    ) -> None:
        mensaje = Message(
            message_id=str(uuid.uuid4()),
            context_id=context.context_id or "",
            task_id=context.task_id or "",
            role=Role.ROLE_AGENT,
            parts=[Part(text=json.dumps(datos, ensure_ascii=False, default=str))],
        )
        await _encolar(event_queue, mensaje)


async def _encolar(event_queue: EventQueue, mensaje: Message) -> None:
    """enqueue_event puede ser sincrono o corrutina segun la version del SDK."""
    resultado = event_queue.enqueue_event(mensaje)
    if hasattr(resultado, "__await__"):
        await resultado


def _extraer_resultado(resultado: Any) -> dict[str, Any]:
    """Normaliza el CallToolResult de MCP a un dict plano."""
    # mcp 2.x devuelve CallToolResult; se prefiere el contenido estructurado.
    estructurado = getattr(resultado, "structured_content", None) or getattr(
        resultado, "structuredContent", None
    )
    if isinstance(estructurado, dict):
        # Las herramientas que devuelven un dict quedan envueltas en {"result": ...}.
        return estructurado.get("result", estructurado)

    contenido = getattr(resultado, "content", None)
    if contenido:
        texto = getattr(contenido[0], "text", None)
        if texto:
            try:
                return json.loads(texto)
            except json.JSONDecodeError:
                return {"ok": True, "texto": texto}

    if isinstance(resultado, dict):
        return resultado
    return {"ok": True, "resultado": str(resultado)}


# ---------------------------------------------------------------------------
# Aplicacion
# ---------------------------------------------------------------------------
def construir_app(clave: str) -> FastAPI:
    """FastAPI con las rutas A2A del agente montadas."""
    tarjeta = construir_agent_card(clave)
    servidor_mcp = servidores_mcp.construir(clave)

    manejador = DefaultRequestHandler(
        agent_executor=EjecutorMCP(clave, servidor_mcp),
        task_store=InMemoryTaskStore(),
        agent_card=tarjeta,
    )

    app = FastAPI(
        title=f"A2A - {config.AGENTES[clave]['nombre']}",
        version=config.VERSION_AGENTES,
    )
    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=create_agent_card_routes(agent_card=tarjeta),
        jsonrpc_routes=create_jsonrpc_routes(
            request_handler=manejador, rpc_url=DEFAULT_RPC_URL
        ),
    )

    @app.get("/salud")
    def salud() -> dict[str, Any]:
        return {
            "ok": True,
            "agente": clave,
            "nombre": tarjeta.name,
            "habilidades": list(_HABILIDADES[clave]),
        }

    return app


def main() -> None:
    import argparse

    import uvicorn

    p = argparse.ArgumentParser(description="Levanta un agente par A2A.")
    p.add_argument("agente", choices=sorted(config.AGENTES))
    args = p.parse_args()

    uvicorn.run(
        construir_app(args.agente),
        host=config.HOST_AGENTES,
        port=config.AGENTES[args.agente]["puerto"],
        log_level="warning",
    )


if __name__ == "__main__":
    main()
