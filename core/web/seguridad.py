"""Controles de la pagina publica: limites, encabezados, IP y token de administracion.

Todo vive en memoria del proceso. El servicio corre con una sola instancia, asi
que no hace falta un almacen compartido; el costo es que un reinicio borra los
contadores (el tope diario incluido). El respaldo real contra el gasto es la
cuota de la clave de Gemini y la alerta de presupuesto del hosting.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

_COLOMBIA = timezone(timedelta(hours=-5))
_SAL = os.getenv("VM_SAL_REGISTRO") or secrets.token_hex(8)


def _entero(nombre: str, defecto: int) -> int:
    try:
        return int(os.getenv(nombre, str(defecto)))
    except ValueError:
        return defecto


@dataclass(frozen=True)
class Limites:
    max_turnos_sesion: int
    tope_diario: int
    max_caracteres: int
    ttl_sesion_min: int
    max_sesiones: int
    sesiones_por_ip_hora: int
    turnos_por_minuto_sesion: int
    turnos_por_minuto_ip: int
    concurrencia: int
    cola: int
    timeout_turno_s: int
    saltos_proxy: int


def leer_limites() -> Limites:
    return Limites(
        max_turnos_sesion=_entero("VM_MAX_TURNOS_SESION", 30),
        tope_diario=_entero("VM_TOPE_DIARIO", 500),
        max_caracteres=_entero("VM_MAX_CARACTERES", 1000),
        ttl_sesion_min=_entero("VM_TTL_SESION_MIN", 120),
        max_sesiones=_entero("VM_MAX_SESIONES", 300),
        sesiones_por_ip_hora=_entero("VM_SESIONES_POR_IP_HORA", 20),
        turnos_por_minuto_sesion=_entero("VM_RITMO_SESION", 8),
        turnos_por_minuto_ip=_entero("VM_RITMO_IP", 30),
        concurrencia=_entero("VM_CONCURRENCIA", 6),
        cola=_entero("VM_COLA", 4),
        timeout_turno_s=_entero("VM_TIMEOUT_TURNO", 80),
        # Cloud Run agrega la IP real al final de X-Forwarded-For; lo anterior
        # lo escribe el cliente y se puede falsificar. 0 = no hay proxy.
        saltos_proxy=_entero("VM_SALTOS_PROXY", 0),
    )


# ---------------------------------------------------------------------------
# Identidad
# ---------------------------------------------------------------------------
def ip_cliente(peer: str | None, x_forwarded_for: str | None, saltos_proxy: int) -> str:
    """IP del cliente. Solo confia en X-Forwarded-For desde el extremo derecho."""
    if saltos_proxy > 0 and x_forwarded_for:
        partes = [p.strip() for p in x_forwarded_for.split(",") if p.strip()]
        if len(partes) >= saltos_proxy:
            return partes[-saltos_proxy][:64]
    return (peer or "desconocida")[:64]


def hash_corto(valor: str) -> str:
    """Identificador estable en el proceso que no revela el valor original."""
    return hashlib.sha256((_SAL + valor).encode("utf-8")).hexdigest()[:12]


def token_admin_valido(cabecera: str | None, esperado: str) -> int:
    """200 si coincide, 401 si falta o esta mal formado, 403 si no coincide."""
    if not cabecera:
        return 401
    esquema, _, valor = cabecera.partition(" ")
    if esquema.lower() != "bearer" or not valor.strip():
        return 401
    ok = hmac.compare_digest(valor.strip().encode("utf-8"), esperado.encode("utf-8"))
    return 200 if ok else 403


# ---------------------------------------------------------------------------
# Limites de uso
# ---------------------------------------------------------------------------
class VentanaDeslizante:
    """Cuenta eventos por clave en una ventana de tiempo."""

    def __init__(self, reloj=time.monotonic) -> None:
        self._eventos: dict[str, deque[float]] = {}
        self._lock = threading.Lock()
        self._reloj = reloj

    def intentar(self, clave: str, limite: int, ventana_s: float) -> tuple[bool, int]:
        """Registra un evento si cabe. Devuelve (ok, segundos_hasta_poder_reintentar)."""
        if limite <= 0:
            return True, 0
        ahora = self._reloj()
        with self._lock:
            q = self._eventos.setdefault(clave, deque())
            while q and ahora - q[0] >= ventana_s:
                q.popleft()
            if len(q) >= limite:
                return False, max(1, int(ventana_s - (ahora - q[0])) + 1)
            q.append(ahora)
            if len(self._eventos) > 5000:
                self._podar(ahora, ventana_s)
            return True, 0

    def _podar(self, ahora: float, ventana_s: float) -> None:
        for k in [k for k, q in self._eventos.items() if not q or ahora - q[-1] >= ventana_s]:
            del self._eventos[k]


class TopeDiario:
    """Turnos globales por dia calendario de Colombia."""

    def __init__(self, tope: int, hoy=None) -> None:
        self._tope = tope
        self._hoy = hoy or (lambda: datetime.now(_COLOMBIA).date())
        self._dia = self._hoy()
        self._n = 0
        self._lock = threading.Lock()

    def consumir(self) -> bool:
        if self._tope <= 0:
            return True
        with self._lock:
            hoy = self._hoy()
            if hoy != self._dia:
                self._dia, self._n = hoy, 0
            if self._n >= self._tope:
                return False
            self._n += 1
            return True

    @property
    def usados(self) -> int:
        return self._n


# ---------------------------------------------------------------------------
# Entrada y encabezados
# ---------------------------------------------------------------------------
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200f\u2028\u2029\u202a-\u202e\u2066-\u2069]")


def limpiar_texto(texto: str) -> str:
    """Quita caracteres de control y de direccion invisibles; conserva saltos de linea."""
    return _CONTROL.sub("", texto).strip()


def encabezados_seguros(es_api: bool, frame_ancestors: str = "'none'") -> dict[str, str]:
    csp = (
        "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        "connect-src 'self'; font-src 'self'; base-uri 'none'; form-action 'none'; "
        f"frame-ancestors {frame_ancestors}"
    )
    h = {
        "Content-Security-Policy": csp,
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "no-referrer",
        "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=()",
        "Cross-Origin-Opener-Policy": "same-origin",
        "Strict-Transport-Security": "max-age=31536000",
    }
    h["Cache-Control"] = "no-store" if es_api else "no-cache"
    return h
