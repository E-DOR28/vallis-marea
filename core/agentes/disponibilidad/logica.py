"""Agente de Disponibilidad y Reservas: logica de negocio.

Es el agente que toca datos estructurados y que, a diferencia del de
conocimiento, produce efectos: bloquea inventario. Por eso tres invariantes:

- Nunca se excede la capacidad autorizada de una embarcacion.
- Nunca se entrega un precio que no salga de la tabla de tarifas.
- Reservar es idempotente: un mensaje de WhatsApp reenviado no reserva dos veces.
"""

from __future__ import annotations

import hashlib
import re
from datetime import date, datetime, timedelta
from typing import Any

from core.almacen import estructurado

_HOY_POR_DEFECTO = date(2026, 9, 13)  # fecha de referencia del demo


# ---------------------------------------------------------------------------
# Fechas
# ---------------------------------------------------------------------------
_DIAS = {
    "lunes": 0, "martes": 1, "miercoles": 2, "miércoles": 2, "jueves": 3,
    "viernes": 4, "sabado": 5, "sábado": 5, "domingo": 6,
}
_MESES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}


def interpretar_fecha(texto: str, hoy: date | None = None) -> str | None:
    """Convierte expresiones en espanol a YYYY-MM-DD.

    Cubre lo que de verdad escriben los clientes por WhatsApp: "el sabado",
    "manana", "15 de octubre", "2026-10-15". Lo que no reconoce devuelve None,
    y el agente pregunta en vez de adivinar: reservar en la fecha equivocada es
    peor que pedir una aclaracion.
    """
    hoy = hoy or _HOY_POR_DEFECTO
    t = texto.lower().strip()

    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", t)
    if m:
        try:
            return datetime.strptime(m.group(0), "%Y-%m-%d").date().isoformat()
        except ValueError:
            return None

    m = re.search(r"\b(\d{1,2})\s*/\s*(\d{1,2})(?:\s*/\s*(\d{2,4}))?", t)
    if m:
        dia, mes = int(m.group(1)), int(m.group(2))
        anio = int(m.group(3) or hoy.year)
        if anio < 100:
            anio += 2000
        try:
            return date(anio, mes, dia).isoformat()
        except ValueError:
            return None

    m = re.search(r"\b(\d{1,2})\s+de\s+(\w+)", t)
    if m and m.group(2) in _MESES:
        dia, mes = int(m.group(1)), _MESES[m.group(2)]
        anio = hoy.year
        candidata = None
        try:
            candidata = date(anio, mes, dia)
        except ValueError:
            return None
        if candidata < hoy:
            try:
                candidata = date(anio + 1, mes, dia)
            except ValueError:
                return None
        return candidata.isoformat()

    if "pasado manana" in t or "pasado mañana" in t:
        return (hoy + timedelta(days=2)).isoformat()
    if "manana" in t or "mañana" in t:
        return (hoy + timedelta(days=1)).isoformat()
    if "hoy" in t:
        return hoy.isoformat()

    for nombre, idx in _DIAS.items():
        if re.search(rf"\b{nombre}\b", t):
            delta = (idx - hoy.weekday()) % 7
            if delta == 0:
                delta = 7  # "el sabado" dicho un sabado = el proximo
            if "proximo" in t or "próximo" in t or "entrante" in t:
                delta += 7 if delta < 7 else 0
            return (hoy + timedelta(days=delta)).isoformat()

    return None


# ---------------------------------------------------------------------------
# Precios
# ---------------------------------------------------------------------------
def _tarifa_base(emb: dict[str, Any], duracion: str) -> int:
    return int(emb["tarifa_dia_completo"] if duracion == "dia_completo"
               else emb["tarifa_medio_dia"])


def calcular_precio(emb: dict[str, Any], ruta: dict[str, Any], fecha: str) -> dict[str, Any]:
    base = _tarifa_base(emb, ruta["duracion"])
    alta = estructurado.es_temporada_alta(fecha)
    descuento = 0.0 if alta else estructurado.descuento_temporada_baja()
    total = int(round(base * (1 - descuento)))
    return {
        "tarifa_base": base,
        "temporada": "alta" if alta else "baja",
        "descuento_aplicado": round(descuento, 4),
        "valor_alquiler": total,
        "anticipo_50": int(round(total * 0.5)),
        "moneda": "COP",
        "nota": (
            "El alquiler es por embarcacion, no por persona. No incluye almuerzos, "
            "entradas ni impuesto de muelle."
        ),
    }


# ---------------------------------------------------------------------------
# Herramientas del agente
# ---------------------------------------------------------------------------
def consultar_disponibilidad(
    fecha: str,
    pasajeros: int,
    ruta: str | None = None,
    requiere_bano: bool | None = None,
) -> dict[str, Any]:
    """Embarcaciones libres en una fecha que cubran al grupo."""
    try:
        datetime.strptime(fecha, "%Y-%m-%d")
    except ValueError:
        return {"ok": False, "error": f"Fecha invalida: '{fecha}'. Se espera YYYY-MM-DD."}
    if pasajeros < 1:
        return {"ok": False, "error": "El numero de pasajeros debe ser al menos 1."}

    datos_ruta = estructurado.obtener_ruta(ruta) if ruta else None
    if ruta and not datos_ruta:
        disponibles = [r["codigo"] for r in estructurado.listar_rutas()]
        return {"ok": False,
                "error": f"Ruta '{ruta}' no existe. Rutas validas: {disponibles}"}

    tope = (datos_ruta or {}).get("capacidad_maxima_forzada")
    if tope and pasajeros > tope:
        return {
            "ok": False,
            "error": (f"La ruta {datos_ruta['nombre']} admite maximo {tope} pasajeros "
                      f"y se solicitaron {pasajeros}."),
        }

    ocupadas = estructurado.embarcaciones_ocupadas(fecha)
    tipo_requerido = (datos_ruta or {}).get("tipo_embarcacion_requerido")

    libres, descartadas = [], []
    for emb in estructurado.listar_embarcaciones():
        if emb["id"] in ocupadas:
            descartadas.append({"id": emb["id"], "motivo": "reservada en esa fecha"})
            continue
        if emb["capacidad"] < pasajeros:
            descartadas.append({"id": emb["id"], "motivo": f"capacidad {emb['capacidad']}"})
            continue
        if tipo_requerido and emb["tipo"] != tipo_requerido:
            descartadas.append({"id": emb["id"], "motivo": f"requiere {tipo_requerido}"})
            continue
        if requiere_bano and not emb["bano"]:
            descartadas.append({"id": emb["id"], "motivo": "sin bano a bordo"})
            continue

        fila = {
            "id": emb["id"],
            "nombre": emb["nombre"],
            "tipo": emb["tipo"],
            "capacidad": emb["capacidad"],
            "bano": emb["bano"],
            "sombra": emb["sombra"],
            "caracteristicas": emb["caracteristicas"],
        }
        if datos_ruta:
            fila["precio"] = calcular_precio(emb, datos_ruta, fecha)
        libres.append(fila)

    # Mas economica primero: es el criterio de asignacion del negocio.
    if datos_ruta:
        libres.sort(key=lambda e: e["precio"]["valor_alquiler"])
    else:
        libres.sort(key=lambda e: e["capacidad"])

    return {
        "ok": True,
        "fecha": fecha,
        "pasajeros": pasajeros,
        "ruta": datos_ruta["codigo"] if datos_ruta else None,
        "nombre_ruta": datos_ruta["nombre"] if datos_ruta else None,
        "temporada": "alta" if estructurado.es_temporada_alta(fecha) else "baja",
        "disponibles": libres,
        "total_disponibles": len(libres),
        "descartadas": descartadas,
    }


def cotizar(embarcacion_id: str, fecha: str, ruta: str, pasajeros: int) -> dict[str, Any]:
    """Cotizacion detallada de una embarcacion concreta."""
    emb = next((e for e in estructurado.listar_embarcaciones() if e["id"] == embarcacion_id), None)
    if not emb:
        return {"ok": False, "error": f"Embarcacion '{embarcacion_id}' no existe."}
    datos_ruta = estructurado.obtener_ruta(ruta)
    if not datos_ruta:
        return {"ok": False, "error": f"Ruta '{ruta}' no existe."}
    if pasajeros > emb["capacidad"]:
        return {
            "ok": False,
            "error": (f"{emb['nombre']} admite {emb['capacidad']} pasajeros y se "
                      f"solicitaron {pasajeros}. Nunca se excede la capacidad autorizada."),
        }
    tope = datos_ruta.get("capacidad_maxima_forzada")
    if tope and pasajeros > tope:
        return {"ok": False,
                "error": f"La ruta {datos_ruta['nombre']} admite maximo {tope} pasajeros."}

    if embarcacion_id in estructurado.embarcaciones_ocupadas(fecha):
        return {"ok": False, "error": f"{emb['nombre']} ya esta reservada el {fecha}."}

    precio = calcular_precio(emb, datos_ruta, fecha)
    return {
        "ok": True,
        "embarcacion": {"id": emb["id"], "nombre": emb["nombre"],
                        "capacidad": emb["capacidad"], "bano": emb["bano"]},
        "ruta": {"codigo": datos_ruta["codigo"], "nombre": datos_ruta["nombre"],
                 "hora_zarpe": datos_ruta["hora_zarpe"],
                 "hora_regreso": datos_ruta["hora_regreso"]},
        "fecha": fecha,
        "pasajeros": pasajeros,
        **precio,
    }


def bloquear_reserva(
    embarcacion_id: str,
    fecha: str,
    ruta: str,
    pasajeros: int,
    cliente: str | None = None,
    clave_idempotencia: str | None = None,
) -> dict[str, Any]:
    """Bloquea la embarcacion. Idempotente por `clave_idempotencia`."""
    if not clave_idempotencia:
        clave_idempotencia = hashlib.sha1(
            f"{embarcacion_id}|{fecha}|{ruta}|{pasajeros}|{cliente or ''}".encode()
        ).hexdigest()[:20]

    # La idempotencia se resuelve ANTES de cotizar. Si no, el reintento de una
    # reserva ya creada falla al cotizar -- la embarcacion figura ocupada por la
    # reserva que el propio reintento creo.
    previa = estructurado.reserva_por_idempotencia(clave_idempotencia)
    if previa:
        emb_previa = next(
            (e for e in estructurado.listar_embarcaciones() if e["id"] == previa["embarcacion_id"]),
            {},
        )
        ruta_previa = estructurado.obtener_ruta(previa["ruta"]) or {}
        return {
            "ok": True,
            "codigo_reserva": previa["codigo"],
            "estado": previa["estado"],
            "reutilizada_por_idempotencia": True,
            "embarcacion": {"id": previa["embarcacion_id"],
                            "nombre": emb_previa.get("nombre", ""),
                            "capacidad": emb_previa.get("capacidad"),
                            "bano": emb_previa.get("bano")},
            "ruta": {"codigo": previa["ruta"], "nombre": ruta_previa.get("nombre", ""),
                     "hora_zarpe": ruta_previa.get("hora_zarpe", ""),
                     "hora_regreso": ruta_previa.get("hora_regreso", "")},
            "fecha": previa["fecha"],
            "pasajeros": previa["pasajeros"],
            "valor_alquiler": previa["valor_total"],
            "anticipo_requerido": int(round((previa["valor_total"] or 0) * 0.5)),
            "siguiente_paso": "Esta reserva ya existia; no se creo una nueva.",
        }

    cotizacion = cotizar(embarcacion_id, fecha, ruta, pasajeros)
    if not cotizacion.get("ok"):
        return cotizacion

    codigo = estructurado.siguiente_codigo_reserva()
    ok, resultado = estructurado.crear_reserva(
        codigo=codigo,
        embarcacion_id=embarcacion_id,
        fecha=fecha,
        ruta=ruta,
        pasajeros=pasajeros,
        cliente=cliente,
        valor_total=cotizacion["valor_alquiler"],
        clave_idempotencia=clave_idempotencia,
    )
    if not ok:
        return {"ok": False, "error": f"No se pudo bloquear: {resultado}"}

    reutilizada = resultado != codigo
    return {
        "ok": True,
        "codigo_reserva": resultado,
        "estado": "bloqueada",
        "reutilizada_por_idempotencia": reutilizada,
        "embarcacion": cotizacion["embarcacion"],
        "ruta": cotizacion["ruta"],
        "fecha": fecha,
        "pasajeros": pasajeros,
        "valor_alquiler": cotizacion["valor_alquiler"],
        "anticipo_requerido": cotizacion["anticipo_50"],
        "siguiente_paso": (
            "La reserva queda bloqueada. Se confirma al recibir el anticipo del 50%."
        ),
    }


def listar_rutas() -> dict[str, Any]:
    return {"ok": True, "rutas": estructurado.listar_rutas()}
