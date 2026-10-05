"""Datos estructurados: flota, rutas y reservas.

En produccion esto es Postgres. Aqui es SQLite con el mismo modelo relacional,
detras de la misma interfaz, para que migrar sea reemplazar la conexion.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime
from typing import Any

from core import config

ESQUEMA = """
CREATE TABLE IF NOT EXISTS embarcaciones (
    id                 TEXT PRIMARY KEY,
    nombre             TEXT NOT NULL,
    tipo               TEXT NOT NULL,
    eslora_pies        INTEGER NOT NULL,
    capacidad          INTEGER NOT NULL,
    motor              TEXT,
    bano               INTEGER NOT NULL DEFAULT 0,
    sombra             TEXT,
    nevera             INTEGER NOT NULL DEFAULT 1,
    sonido             TEXT,
    tarifa_dia_completo INTEGER NOT NULL,
    tarifa_medio_dia    INTEGER NOT NULL,
    caracteristicas     TEXT NOT NULL,
    notas               TEXT
);

CREATE TABLE IF NOT EXISTS rutas (
    codigo   TEXT PRIMARY KEY,
    nombre   TEXT NOT NULL,
    duracion TEXT NOT NULL,
    hora_zarpe TEXT NOT NULL,
    hora_regreso TEXT NOT NULL,
    tags TEXT NOT NULL,
    capacidad_minima_sugerida INTEGER,
    capacidad_maxima_forzada  INTEGER,
    tipo_embarcacion_requerido TEXT,
    resumen TEXT
);

CREATE TABLE IF NOT EXISTS reservas (
    codigo         TEXT PRIMARY KEY,
    embarcacion_id TEXT NOT NULL REFERENCES embarcaciones(id),
    fecha          TEXT NOT NULL,
    ruta           TEXT NOT NULL,
    estado         TEXT NOT NULL,
    pasajeros      INTEGER NOT NULL,
    cliente        TEXT,
    valor_total    INTEGER,
    creada_en      TEXT NOT NULL DEFAULT (datetime('now')),
    clave_idempotencia TEXT
);

-- Una embarcacion no puede tener dos reservas vivas el mismo dia.
-- Es la garantia que evita la doble reserva, el peor error posible aqui.
CREATE UNIQUE INDEX IF NOT EXISTS ux_reserva_embarcacion_fecha
    ON reservas(embarcacion_id, fecha)
    WHERE estado IN ('confirmada', 'bloqueada');

CREATE UNIQUE INDEX IF NOT EXISTS ux_reserva_idempotencia
    ON reservas(clave_idempotencia)
    WHERE clave_idempotencia IS NOT NULL;
"""


_ESPERA_BLOQUEO_SEGUNDOS = 10.0


def conexion() -> sqlite3.Connection:
    # Tres agentes y el orquestador comparten este archivo. Sin espera de
    # bloqueo, dos escrituras simultaneas fallan con "database is locked" en
    # vez de turnarse.
    con = sqlite3.connect(config.RUTA_SQLITE, check_same_thread=False,
                          timeout=_ESPERA_BLOQUEO_SEGUNDOS)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


def inicializar(recrear: bool = False) -> dict[str, int]:
    """Crea el esquema y carga los datos semilla desde data/*.json."""
    con = conexion()
    try:
        # WAL deja leer mientras otro proceso escribe; el modo persiste en el archivo.
        con.execute("PRAGMA journal_mode = WAL")
        if recrear:
            for t in ("reservas", "rutas", "embarcaciones"):
                con.execute(f"DROP TABLE IF EXISTS {t}")
        con.executescript(ESQUEMA)

        flota = json.loads((config.RUTA_DATA / "flota.json").read_text(encoding="utf-8"))
        rutas = json.loads((config.RUTA_DATA / "rutas.json").read_text(encoding="utf-8"))

        for e in flota["embarcaciones"]:
            con.execute(
                "INSERT OR REPLACE INTO embarcaciones VALUES "
                "(:id,:nombre,:tipo,:eslora_pies,:capacidad,:motor,:bano,:sombra,"
                " :nevera,:sonido,:tarifa_dia_completo,:tarifa_medio_dia,"
                " :caracteristicas,:notas)",
                {**e,
                 "bano": int(e["bano"]),
                 "nevera": int(e["nevera"]),
                 "caracteristicas": json.dumps(e["caracteristicas"], ensure_ascii=False)},
            )

        for r in rutas["rutas"]:
            con.execute(
                "INSERT OR REPLACE INTO rutas VALUES "
                "(:codigo,:nombre,:duracion,:hora_zarpe,:hora_regreso,:tags,"
                " :capacidad_minima_sugerida,:capacidad_maxima_forzada,"
                " :tipo_embarcacion_requerido,:resumen)",
                {
                    "capacidad_maxima_forzada": None,
                    **r,
                    "tags": json.dumps(r["tags"], ensure_ascii=False),
                },
            )

        for rv in flota["reservas_existentes"]:
            con.execute(
                "INSERT OR IGNORE INTO reservas "
                "(codigo, embarcacion_id, fecha, ruta, estado, pasajeros) "
                "VALUES (:codigo,:embarcacion_id,:fecha,:ruta,:estado,:pasajeros)",
                rv,
            )
        con.commit()

        return {
            "embarcaciones": con.execute("SELECT COUNT(*) FROM embarcaciones").fetchone()[0],
            "rutas": con.execute("SELECT COUNT(*) FROM rutas").fetchone()[0],
            "reservas": con.execute("SELECT COUNT(*) FROM reservas").fetchone()[0],
        }
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Consultas de negocio
# ---------------------------------------------------------------------------
def _fila_a_embarcacion(f: sqlite3.Row) -> dict[str, Any]:
    d = dict(f)
    d["bano"] = bool(d["bano"])
    d["nevera"] = bool(d["nevera"])
    d["caracteristicas"] = json.loads(d["caracteristicas"])
    return d


def listar_embarcaciones() -> list[dict[str, Any]]:
    con = conexion()
    try:
        return [_fila_a_embarcacion(f)
                for f in con.execute("SELECT * FROM embarcaciones ORDER BY capacidad")]
    finally:
        con.close()


def obtener_ruta(codigo: str) -> dict[str, Any] | None:
    con = conexion()
    try:
        f = con.execute("SELECT * FROM rutas WHERE codigo = ?", (codigo,)).fetchone()
        if not f:
            return None
        d = dict(f)
        d["tags"] = json.loads(d["tags"])
        return d
    finally:
        con.close()


def listar_rutas() -> list[dict[str, Any]]:
    con = conexion()
    try:
        salida = []
        for f in con.execute("SELECT * FROM rutas"):
            d = dict(f)
            d["tags"] = json.loads(d["tags"])
            salida.append(d)
        return salida
    finally:
        con.close()


def es_temporada_alta(fecha: str) -> bool:
    rutas = json.loads((config.RUTA_DATA / "rutas.json").read_text(encoding="utf-8"))
    d = datetime.strptime(fecha, "%Y-%m-%d").date()
    for tramo in rutas["temporada_alta"]:
        if (datetime.strptime(tramo["desde"], "%Y-%m-%d").date()
                <= d
                <= datetime.strptime(tramo["hasta"], "%Y-%m-%d").date()):
            return True
    return False


def descuento_temporada_baja() -> float:
    rutas = json.loads((config.RUTA_DATA / "rutas.json").read_text(encoding="utf-8"))
    return float(rutas["descuento_temporada_baja"])


def embarcaciones_ocupadas(fecha: str) -> set[str]:
    # Toda lectura de disponibilidad pasa por aqui, asi que es el unico punto
    # donde basta aplicar el vencimiento para que nadie vea un bloqueo viejo.
    liberar_vencidas()
    con = conexion()
    try:
        return {
            f["embarcacion_id"]
            for f in con.execute(
                "SELECT embarcacion_id FROM reservas "
                "WHERE fecha = ? AND estado IN ('confirmada','bloqueada')",
                (fecha,),
            )
        }
    finally:
        con.close()


def _siguiente_codigo(con: sqlite3.Connection) -> str:
    prefijo = f"VM-{date.today().year}-"
    n = con.execute("SELECT COUNT(*) FROM reservas").fetchone()[0]
    mayor = con.execute(
        "SELECT MAX(CAST(substr(codigo, ?) AS INTEGER)) FROM reservas "
        "WHERE substr(codigo, 1, ?) = ?",
        (len(prefijo) + 1, len(prefijo), prefijo),
    ).fetchone()[0] or 0
    return f"{prefijo}{max(1000 + n, mayor) + 1}"


def _sesion_de(clave: str) -> str:
    return clave.split("|", 1)[0] if "|" in clave else ""


def crear_reserva(
    *,
    embarcacion_id: str,
    fecha: str,
    ruta: str,
    pasajeros: int,
    cliente: str | None,
    valor_total: int,
    clave_idempotencia: str,
    limite_sesion: int = 0,
    limite_total: int = 0,
) -> tuple[bool, str, bool]:
    """Inserta una reserva. Devuelve (ok, codigo_o_mensaje, reutilizada).

    Todo ocurre en una transaccion con bloqueo de escritura: el codigo se asigna
    dentro de ella, asi que dos solicitudes simultaneas nunca reciben el mismo.

    Idempotente: repetir la misma clave devuelve la reserva ya creada en vez de
    duplicarla. Un mensaje de WhatsApp reenviado no puede reservar dos veces.

    `limite_sesion` y `limite_total` (0 = sin tope) cuentan reservas bloqueadas
    vivas; la sesion es el prefijo de la clave antes del primer "|".
    """
    con = conexion()
    try:
        con.execute("BEGIN IMMEDIATE")
        previa = con.execute(
            "SELECT codigo FROM reservas WHERE clave_idempotencia = ?",
            (clave_idempotencia,),
        ).fetchone()
        if previa:
            con.rollback()
            return True, previa["codigo"], True

        if limite_total:
            vivas = con.execute(
                "SELECT COUNT(*) FROM reservas WHERE estado = 'bloqueada'"
            ).fetchone()[0]
            if vivas >= limite_total:
                con.rollback()
                return False, "limite_total", False
        sesion = _sesion_de(clave_idempotencia)
        if limite_sesion and sesion:
            marca = sesion + "|"
            propias = con.execute(
                "SELECT COUNT(*) FROM reservas WHERE estado = 'bloqueada' "
                "AND substr(clave_idempotencia, 1, ?) = ?",
                (len(marca), marca),
            ).fetchone()[0]
            if propias >= limite_sesion:
                con.rollback()
                return False, "limite_sesion", False

        codigo = _siguiente_codigo(con)
        con.execute(
            "INSERT INTO reservas (codigo, embarcacion_id, fecha, ruta, estado, "
            "pasajeros, cliente, valor_total, clave_idempotencia) "
            "VALUES (?,?,?,?,'bloqueada',?,?,?,?)",
            (codigo, embarcacion_id, fecha, ruta, pasajeros, cliente,
             valor_total, clave_idempotencia),
        )
        con.commit()
        return True, codigo, False
    except sqlite3.IntegrityError as exc:
        # El indice unico parcial atrapo una doble reserva.
        con.rollback()
        return False, f"conflicto: {exc}", False
    finally:
        con.close()


def liberar_vencidas(minutos: int | None = None) -> int:
    """Libera reservas bloqueadas sin anticipo pasado el plazo. Devuelve cuantas.

    La clave de idempotencia se anula: sin eso, quien reintente la misma
    solicitud recibiria de vuelta una reserva que ya no existe.
    """
    minutos = config.TTL_RESERVA_MINUTOS if minutos is None else minutos
    if minutos <= 0:
        return 0
    con = conexion()
    try:
        cur = con.execute(
            "UPDATE reservas SET estado = 'liberada', clave_idempotencia = NULL "
            "WHERE estado = 'bloqueada' AND creada_en <= datetime('now', ?)",
            (f"-{int(minutos)} minutes",),
        )
        con.commit()
        return cur.rowcount
    finally:
        con.close()


def liberar_por_sesion(sesion: str) -> int:
    """Libera las reservas bloqueadas de una sesion (cierre o expiracion)."""
    if not sesion:
        return 0
    marca = sesion + "|"
    con = conexion()
    try:
        cur = con.execute(
            "UPDATE reservas SET estado = 'liberada', clave_idempotencia = NULL "
            "WHERE estado = 'bloqueada' AND substr(clave_idempotencia, 1, ?) = ?",
            (len(marca), marca),
        )
        con.commit()
        return cur.rowcount
    finally:
        con.close()


def contar_bloqueadas() -> int:
    con = conexion()
    try:
        return con.execute(
            "SELECT COUNT(*) FROM reservas WHERE estado = 'bloqueada'"
        ).fetchone()[0]
    finally:
        con.close()


def reserva_por_idempotencia(clave: str) -> dict[str, Any] | None:
    """Busca una reserva ya creada con esa clave de idempotencia."""
    con = conexion()
    try:
        f = con.execute(
            "SELECT * FROM reservas WHERE clave_idempotencia = ?", (clave,)
        ).fetchone()
        return dict(f) if f else None
    finally:
        con.close()


def obtener_reserva(codigo: str) -> dict[str, Any] | None:
    con = conexion()
    try:
        f = con.execute("SELECT * FROM reservas WHERE codigo = ?", (codigo,)).fetchone()
        return dict(f) if f else None
    finally:
        con.close()


def siguiente_codigo_reserva() -> str:
    """Codigo que se asignaria ahora. Solo informativo: no reserva nada."""
    con = conexion()
    try:
        return _siguiente_codigo(con)
    finally:
        con.close()
