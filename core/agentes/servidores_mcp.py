"""Servidores MCP de los tres agentes pares.

Cada agente publica sus capacidades como herramientas MCP tipadas. Esto es la
mitad del componente de innovacion del proyecto: la misma capacidad queda
disponible para cualquier cliente MCP (Claude Desktop, un IDE, otro agente) y
no solo para el orquestador de Vallis Marea.

Nota de version: en `mcp` 2.x la clase se llama `MCPServer` (era `FastMCP` en
1.x). Las firmas y docstrings de cada funcion se convierten automaticamente en
el esquema JSON de la herramienta, asi que los tipos y las descripciones no son
decoracion: son el contrato que ve el cliente.

Ejecucion como servidor MCP independiente (stdio):
    python -m core.agentes.servidores_mcp conocimiento
    python -m core.agentes.servidores_mcp disponibilidad
    python -m core.agentes.servidores_mcp recomendador
"""

from __future__ import annotations

import sys
from typing import Any

from mcp.server.mcpserver import MCPServer

from core import config
from core.agentes.conocimiento import rag
from core.agentes.disponibilidad import logica as disp
from core.agentes.recomendador import logica as reco


# ---------------------------------------------------------------------------
# Agente de Conocimiento
# ---------------------------------------------------------------------------
def servidor_conocimiento() -> MCPServer:
    srv = MCPServer(
        name="vallis-conocimiento",
        title=config.AGENTES["conocimiento"]["nombre"],
        instructions=config.AGENTES["conocimiento"]["descripcion"],
        version=config.VERSION_AGENTES,
    )

    @srv.tool(
        name="buscar_conocimiento",
        description=(
            "Responde una pregunta sobre politicas, FAQs, rutas o flota de Vallis "
            "Marea usando RAG sobre el corpus indexado. Devuelve la respuesta con "
            "las fuentes citadas, o una abstencion explicita si el corpus no "
            "contiene la informacion. Usar para cualquier pregunta de conocimiento; "
            "NO usar para disponibilidad de fechas concretas ni para reservar."
        ),
    )
    def buscar_conocimiento(pregunta: str, historial: str = "") -> dict[str, Any]:
        """Busca en el corpus y responde citando la fuente.

        Args:
            pregunta: La pregunta del cliente en lenguaje natural.
            historial: Contexto de turnos previos, opcional.
        """
        return rag.responder(pregunta, historial=historial).a_dict()

    @srv.tool(
        name="obtener_fragmentos",
        description=(
            "Devuelve los fragmentos del corpus mas relevantes para una consulta, "
            "sin generar respuesta. Util para inspeccionar que recupera el sistema "
            "o para que otro agente razone sobre la evidencia cruda."
        ),
    )
    def obtener_fragmentos(consulta: str, k: int = 4) -> dict[str, Any]:
        """Recupera fragmentos crudos del corpus.

        Args:
            consulta: Texto de busqueda.
            k: Cuantos fragmentos devolver.
        """
        fragmentos, diagnostico = rag.recuperar(consulta, k_final=k)
        return {
            "ok": True,
            "diagnostico": diagnostico,
            "fragmentos": [
                {
                    "chunk_id": f.chunk_id,
                    "texto": f.texto,
                    "cita": f.cita(),
                    "puntaje_fusion": round(f.puntaje_fusion, 5),
                    "puntaje_denso": round(f.puntaje_denso, 4),
                    "puntaje_lexico": round(f.puntaje_lexico, 4),
                }
                for f in fragmentos
            ],
        }

    return srv


# ---------------------------------------------------------------------------
# Agente de Disponibilidad y Reservas
# ---------------------------------------------------------------------------
def servidor_disponibilidad() -> MCPServer:
    srv = MCPServer(
        name="vallis-disponibilidad",
        title=config.AGENTES["disponibilidad"]["nombre"],
        instructions=config.AGENTES["disponibilidad"]["descripcion"],
        version=config.VERSION_AGENTES,
    )

    @srv.tool(
        name="interpretar_fecha",
        description=(
            "Convierte una expresion de fecha en espanol ('el sabado', 'manana', "
            "'15 de octubre') a formato YYYY-MM-DD. Devuelve fecha nula si no "
            "logra interpretarla: en ese caso hay que preguntarle al cliente, "
            "nunca adivinar."
        ),
    )
    def interpretar_fecha(texto: str) -> dict[str, Any]:
        """Normaliza una fecha escrita en lenguaje natural.

        Args:
            texto: Expresion de fecha tal como la escribio el cliente.
        """
        fecha = disp.interpretar_fecha(texto)
        return {"ok": fecha is not None, "fecha": fecha, "texto_original": texto}

    @srv.tool(
        name="consultar_disponibilidad",
        description=(
            "Lista las embarcaciones libres en una fecha que cubran al grupo, "
            "ordenadas de mas economica a mas costosa, con el precio ya calculado "
            "cuando se indica la ruta. Es la unica fuente valida de disponibilidad."
        ),
    )
    def consultar_disponibilidad(
        fecha: str,
        pasajeros: int,
        ruta: str = "",
        requiere_bano: bool = False,
    ) -> dict[str, Any]:
        """Consulta inventario libre.

        Args:
            fecha: Fecha del paseo en formato YYYY-MM-DD.
            pasajeros: Numero de personas del grupo.
            ruta: Codigo de ruta (rosario, baru, cholon, tierrabomba, atardecer, pesca).
            requiere_bano: True si el cliente exige bano a bordo.
        """
        return disp.consultar_disponibilidad(
            fecha=fecha,
            pasajeros=pasajeros,
            ruta=ruta or None,
            requiere_bano=requiere_bano or None,
        )

    @srv.tool(
        name="cotizar",
        description=(
            "Cotiza una embarcacion concreta para una ruta y fecha, con desglose "
            "de tarifa base, temporada, descuento y anticipo. Rechaza la cotizacion "
            "si el grupo excede la capacidad autorizada."
        ),
    )
    def cotizar(embarcacion_id: str, fecha: str, ruta: str, pasajeros: int) -> dict[str, Any]:
        """Cotiza un paseo especifico.

        Args:
            embarcacion_id: Codigo de la embarcacion, por ejemplo VM-05.
            fecha: Fecha en formato YYYY-MM-DD.
            ruta: Codigo de la ruta.
            pasajeros: Numero de personas.
        """
        return disp.cotizar(embarcacion_id, fecha, ruta, pasajeros)

    @srv.tool(
        name="bloquear_reserva",
        description=(
            "Bloquea una embarcacion para una fecha y ruta. Operacion con efecto "
            "real sobre el inventario. Es idempotente: repetir la llamada con la "
            "misma clave_idempotencia devuelve la reserva ya creada en lugar de "
            "duplicarla. Solo llamar con confirmacion explicita del cliente."
        ),
    )
    def bloquear_reserva(
        embarcacion_id: str,
        fecha: str,
        ruta: str,
        pasajeros: int,
        cliente: str = "",
        clave_idempotencia: str = "",
    ) -> dict[str, Any]:
        """Bloquea inventario de forma idempotente.

        Args:
            embarcacion_id: Codigo de la embarcacion.
            fecha: Fecha en formato YYYY-MM-DD.
            ruta: Codigo de la ruta.
            pasajeros: Numero de personas.
            cliente: Nombre del cliente, opcional.
            clave_idempotencia: Identificador estable de la solicitud.
        """
        return disp.bloquear_reserva(
            embarcacion_id=embarcacion_id,
            fecha=fecha,
            ruta=ruta,
            pasajeros=pasajeros,
            cliente=cliente or None,
            clave_idempotencia=clave_idempotencia or None,
        )

    @srv.tool(
        name="listar_rutas",
        description="Devuelve el catalogo de rutas operadas con horarios y duracion.",
    )
    def listar_rutas() -> dict[str, Any]:
        """Lista las rutas disponibles."""
        return disp.listar_rutas()

    return srv


# ---------------------------------------------------------------------------
# Agente Recomendador
# ---------------------------------------------------------------------------
def servidor_recomendador() -> MCPServer:
    srv = MCPServer(
        name="vallis-recomendador",
        title=config.AGENTES["recomendador"]["nombre"],
        instructions=config.AGENTES["recomendador"]["descripcion"],
        version=config.VERSION_AGENTES,
    )

    @srv.tool(
        name="recomendar",
        description=(
            "Sugiere rutas y embarcaciones por afinidad semantica con lo que el "
            "cliente describe (ocasion, grupo, expectativa), filtrando por "
            "restricciones duras de capacidad y de bano. No confirma disponibilidad."
        ),
    )
    def recomendar(
        texto_cliente: str,
        pasajeros: int = 0,
        fecha: str = "",
        top_n: int = 3,
    ) -> dict[str, Any]:
        """Recomienda paseo y embarcacion.

        Args:
            texto_cliente: Lo que el cliente dijo que busca.
            pasajeros: Tamano del grupo, 0 si no se sabe.
            fecha: Fecha YYYY-MM-DD para excluir embarcaciones ocupadas, opcional.
            top_n: Cuantas opciones devolver.
        """
        return reco.recomendar(
            texto_cliente=texto_cliente,
            pasajeros=pasajeros or None,
            fecha=fecha or None,
            top_n=top_n,
        )

    @srv.tool(
        name="extraer_perfil",
        description=(
            "Extrae el perfil de preferencias del cliente (ocasion, tamano del "
            "grupo, sensibilidad al precio, necesidad de bano) desde texto libre."
        ),
    )
    def extraer_perfil(texto_cliente: str) -> dict[str, Any]:
        """Extrae preferencias estructuradas del mensaje del cliente.

        Args:
            texto_cliente: Mensaje del cliente en lenguaje natural.
        """
        return {"ok": True, "perfil": reco.extraer_perfil(texto_cliente)}

    return srv


# ---------------------------------------------------------------------------
CONSTRUCTORES = {
    "conocimiento": servidor_conocimiento,
    "disponibilidad": servidor_disponibilidad,
    "recomendador": servidor_recomendador,
}


def construir(clave: str) -> MCPServer:
    if clave not in CONSTRUCTORES:
        raise ValueError(f"Agente '{clave}'. Validos: {sorted(CONSTRUCTORES)}")
    return CONSTRUCTORES[clave]()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Uso: python -m core.agentes.servidores_mcp <{'|'.join(CONSTRUCTORES)}>",
              file=sys.stderr)
        raise SystemExit(2)
    construir(sys.argv[1]).run(transport="stdio")
