"""Contrato JSON de salida del orquestador.

Es el mismo contrato que Vallis Marea ya tiene validado en produccion: n8n
publica esta estructura en Chatwoot. Conservarlo intacto es lo que permite
sustituir el nodo monolitico de IA por el ecosistema de agentes sin tocar la
capa de transporte ni el frontend de WhatsApp.

Por eso la migracion a MCP/A2A no obliga a reescribir la integracion: el
contrato es la frontera estable entre las dos mitades del sistema.
"""

from __future__ import annotations

from typing import Any

TIPOS_VALIDOS = {"texto", "texto_con_opciones", "escalamiento"}


def construir(
    *,
    mensaje: str,
    tipo: str = "texto",
    citas: list[dict[str, str]] | None = None,
    acciones: list[dict[str, Any]] | None = None,
    escalar_a_humano: bool = False,
    motivo_escalamiento: str = "",
    metadatos: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "tipo": tipo if tipo in TIPOS_VALIDOS else "texto",
        "mensaje": mensaje,
        "citas": citas or [],
        "acciones": acciones or [],
        "escalar_a_humano": escalar_a_humano,
        "motivo_escalamiento": motivo_escalamiento,
        "metadatos": metadatos or {},
    }


def validar(contrato: dict[str, Any]) -> tuple[bool, list[str]]:
    """Valida la estructura antes de publicarla. Devuelve (ok, problemas)."""
    problemas: list[str] = []

    for campo in ("tipo", "mensaje", "citas", "acciones", "escalar_a_humano"):
        if campo not in contrato:
            problemas.append(f"falta el campo '{campo}'")

    if contrato.get("tipo") not in TIPOS_VALIDOS:
        problemas.append(f"tipo invalido: {contrato.get('tipo')}")

    mensaje = contrato.get("mensaje", "")
    if not isinstance(mensaje, str):
        problemas.append("'mensaje' debe ser texto")
    elif not mensaje.strip() and not contrato.get("escalar_a_humano"):
        problemas.append("mensaje vacio sin escalamiento")

    if not isinstance(contrato.get("citas", []), list):
        problemas.append("'citas' debe ser lista")
    if not isinstance(contrato.get("acciones", []), list):
        problemas.append("'acciones' debe ser lista")

    if contrato.get("escalar_a_humano") and not contrato.get("motivo_escalamiento"):
        problemas.append("escalamiento sin motivo declarado")

    return (not problemas), problemas


def escalamiento(motivo: str, mensaje: str | None = None,
                 metadatos: dict[str, Any] | None = None) -> dict[str, Any]:
    """Contrato de escalamiento a un humano en Chatwoot."""
    return construir(
        tipo="escalamiento",
        mensaje=mensaje or (
            "Dejame confirmar ese dato con el equipo y te respondo en un momento."
        ),
        escalar_a_humano=True,
        motivo_escalamiento=motivo,
        metadatos=metadatos,
    )
