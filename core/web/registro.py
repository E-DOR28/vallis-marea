"""Registro de turnos y feedback en JSONL (build-spec 6.3).

Cada linea se escribe en disco y tambien en la salida estandar. El disco del
contenedor es efimero; la salida estandar la conserva el hosting (Cloud
Logging), y de ahi se exporta para el analisis de la prueba con usuarios.

El texto del cliente y la respuesta solo se registran si la persona marco la
casilla de consentimiento. El resto son metricas sin contenido.
"""

from __future__ import annotations

import json
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from core import config

_LOCK = threading.Lock()
_MAX_COMENTARIO = 500


def _ahora() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Registro:
    def __init__(self, carpeta: Path | None = None, salida=None) -> None:
        self.carpeta = carpeta or (config.RUTA_TRAZAS / "web")
        self.carpeta.mkdir(parents=True, exist_ok=True)
        self._salida = salida if salida is not None else sys.stdout

    @property
    def ruta_turnos(self) -> Path:
        return self.carpeta / "turnos.jsonl"

    @property
    def ruta_feedback(self) -> Path:
        return self.carpeta / "feedback.jsonl"

    @property
    def ruta_candidatos(self) -> Path:
        # Candidatos a revision humana. Nada los integra solo al dataset del router.
        return self.carpeta / "candidatos_router.jsonl"

    def _escribir(self, ruta: Path, tipo: str, registro: dict[str, Any]) -> None:
        linea = json.dumps({"tipo": tipo, **registro}, ensure_ascii=False, default=str)
        with _LOCK:
            with ruta.open("a", encoding="utf-8") as f:
                f.write(linea + "\n")
            print(linea, file=self._salida, flush=True)

    def turno(
        self,
        *,
        sesion_hash: str,
        request_id: str,
        contrato: dict[str, Any],
        trazas: dict[str, Any],
        texto: str,
        consentimiento: bool,
    ) -> dict[str, Any]:
        meta = contrato.get("metadatos", {}) or {}
        router = trazas.get("router", {}) or {}
        causa = meta.get("causa")
        r: dict[str, Any] = {
            "ts": _ahora(),
            "sesion_hash": sesion_hash,
            "request_id": request_id,
            "intencion": router.get("intencion"),
            "metodo_router": router.get("metodo"),
            "confianza": router.get("confianza"),
            "saltos": [
                {"agente": s.get("agente"), "habilidad": s.get("habilidad"),
                 "ms": s.get("ms"), "ok": s.get("ok")}
                for s in trazas.get("saltos_a2a", [])
            ],
            "ms_total": meta.get("ms_total"),
            "escalado": bool(contrato.get("escalar_a_humano")),
            "abstencion": causa == "abstencion",
            "causa": causa,
            "reserva_bloqueada": any(a.get("tipo") == "reserva_bloqueada"
                                     for a in contrato.get("acciones", [])),
            "tokens_entrada": meta.get("tokens_entrada"),
            "tokens_salida": meta.get("tokens_salida"),
            "modelo": meta.get("modelo"),
            "degradado": bool(meta.get("degradado")),
            "longitud_texto": len(texto),
            "consentimiento": consentimiento,
        }
        if consentimiento:
            r["texto"] = texto
            r["respuesta"] = contrato.get("mensaje", "")
        self._escribir(self.ruta_turnos, "turno", r)
        return r

    def error(self, *, sesion_hash: str, request_id: str, codigo: str, longitud_texto: int) -> None:
        self._escribir(self.ruta_turnos, "turno_error", {
            "ts": _ahora(), "sesion_hash": sesion_hash, "request_id": request_id,
            "codigo": codigo, "longitud_texto": longitud_texto,
        })

    def feedback(
        self,
        *,
        sesion_hash: str,
        request_id: str,
        valor: str,
        comentario: str,
        candidato: dict[str, Any] | None,
    ) -> None:
        self._escribir(self.ruta_feedback, "feedback", {
            "ts": _ahora(), "sesion_hash": sesion_hash, "request_id": request_id,
            "valor": valor, "comentario": comentario[:_MAX_COMENTARIO],
        })
        if valor == "down" and candidato:
            self._escribir(self.ruta_candidatos, "candidato_router", {
                "ts": _ahora(), "sesion_hash": sesion_hash, "request_id": request_id,
                **candidato,
            })

    def exportar(self, max_lineas: int = 5000) -> Iterator[str]:
        """Lineas NDJSON de turnos y feedback, las mas recientes al final."""
        for ruta in (self.ruta_turnos, self.ruta_feedback):
            if not ruta.exists():
                continue
            with ruta.open(encoding="utf-8") as f:
                lineas = f.readlines()[-max_lineas:]
            for linea in lineas:
                yield linea if linea.endswith("\n") else linea + "\n"
