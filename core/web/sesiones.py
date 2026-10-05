"""Sesiones del chat, emitidas y validadas por el servidor.

El `session_id` es un secreto de portador: quien lo tiene conversa en esa
sesion. El `context_id` que ve el grafo (y por tanto la clave de idempotencia de
las reservas) es otro valor, derivado en el servidor, para que ningun texto del
cliente pueda forjar ni reutilizar la clave de otra sesion.
"""

from __future__ import annotations

import re
import secrets
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

from core import config

ID_VALIDO = re.compile(r"^[A-Za-z0-9_-]{20,64}$")
REQUEST_ID_VALIDO = re.compile(r"^[A-Za-z0-9-]{8,64}$")
_MAX_RESPUESTAS_EN_CACHE = 20


class SinCupo(Exception):
    """El servidor ya tiene el maximo de sesiones abiertas."""


@dataclass
class Sesion:
    id: str
    context_id: str
    ip_hash: str
    creada: float
    ultimo_uso: float
    turnos: int = 0
    historial: list[dict[str, str]] = field(default_factory=list)
    respuestas: "OrderedDict[str, dict[str, Any]]" = field(default_factory=OrderedDict)
    # Lo que se necesita para el feedback pero nunca se devuelve al navegador.
    privado: "OrderedDict[str, dict[str, Any]]" = field(default_factory=OrderedDict)
    pendientes: dict[str, Future] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def recordar(self, usuario: str, agente: str) -> None:
        self.historial.append({"rol": "usuario", "texto": usuario})
        self.historial.append({"rol": "agente", "texto": agente})
        del self.historial[:-config.MAX_TURNOS_MEMORIA_CORTA]

    def guardar_respuesta(self, request_id: str, respuesta: dict[str, Any],
                          privado: dict[str, Any] | None = None) -> None:
        self.respuestas[request_id] = respuesta
        self.privado[request_id] = privado or {}
        while len(self.respuestas) > _MAX_RESPUESTAS_EN_CACHE:
            self.respuestas.popitem(last=False)
        while len(self.privado) > _MAX_RESPUESTAS_EN_CACHE:
            self.privado.popitem(last=False)


class Almacen:
    def __init__(self, ttl_min: int, max_sesiones: int, reloj=time.monotonic) -> None:
        self._sesiones: dict[str, Sesion] = {}
        self._lock = threading.Lock()
        self._ttl_s = ttl_min * 60
        self._max = max_sesiones
        self._reloj = reloj
        # Sesiones que `obtener` descarto por inactividad y cuyas reservas
        # todavia hay que liberar.
        self._por_liberar: list[str] = []

    def crear(self, ip_hash: str) -> Sesion:
        with self._lock:
            if len(self._sesiones) >= self._max:
                raise SinCupo()
            ahora = self._reloj()
            s = Sesion(
                id=secrets.token_urlsafe(24),
                context_id=uuid.uuid4().hex,
                ip_hash=ip_hash,
                creada=ahora,
                ultimo_uso=ahora,
            )
            self._sesiones[s.id] = s
            return s

    def obtener(self, session_id: str) -> Sesion | None:
        if not ID_VALIDO.match(session_id or ""):
            return None
        with self._lock:
            s = self._sesiones.get(session_id)
            if s is None:
                return None
            ahora = self._reloj()
            if self._ttl_s and ahora - s.ultimo_uso > self._ttl_s and not s.pendientes:
                del self._sesiones[session_id]
                self._por_liberar.append(s.context_id)
                return None
            s.ultimo_uso = ahora
            return s

    def expirar(self) -> list[str]:
        """Quita las sesiones inactivas y devuelve sus `context_id`.

        Quien llama libera las reservas de demo de esas sesiones.
        """
        ahora = self._reloj()
        with self._lock:
            muertas = [
                s for s in self._sesiones.values()
                if self._ttl_s and ahora - s.ultimo_uso > self._ttl_s and not s.pendientes
            ]
            for s in muertas:
                del self._sesiones[s.id]
            salida = [s.context_id for s in muertas] + self._por_liberar
            self._por_liberar = []
        return salida

    def cerrar(self, session_id: str) -> str | None:
        with self._lock:
            s = self._sesiones.pop(session_id, None)
        return s.context_id if s else None

    def __len__(self) -> int:
        return len(self._sesiones)
