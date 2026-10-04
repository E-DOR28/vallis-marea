"""Arranque de los tres agentes pares como servicios A2A.

Dos modos:

- `levantar_en_hilos()`: los tres agentes corren como servidores uvicorn en
  hilos del proceso actual. Es lo que usa la interfaz Streamlit para que el
  demo arranque con un solo comando. Aunque compartan proceso, la comunicacion
  sigue siendo HTTP/JSON-RPC real: el protocolo A2A se ejercita de verdad, no
  se simula con llamadas a funcion.

- `python -m core.a2a.lanzador`: los tres como procesos separados, que es como
  correrian en produccion.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import httpx
import uvicorn

from core import config
from core.a2a.servidor import construir_app

_SERVIDORES: dict[str, uvicorn.Server] = {}
_HILOS: dict[str, threading.Thread] = {}


def _servidor(clave: str) -> uvicorn.Server:
    cfg = uvicorn.Config(
        construir_app(clave),
        host=config.HOST_AGENTES,
        port=config.AGENTES[clave]["puerto"],
        log_level="warning",
        access_log=False,
    )
    return uvicorn.Server(cfg)


def _responde(clave: str, timeout: float = 1.0) -> bool:
    try:
        r = httpx.get(
            f"{config.url_agente(clave)}{config.RUTA_AGENT_CARD}", timeout=timeout
        )
        return r.status_code == 200
    except Exception:
        return False


def levantar_en_hilos(espera_maxima: float = 25.0) -> dict[str, Any]:
    """Arranca los agentes que no esten ya respondiendo. Idempotente."""
    arrancados: list[str] = []
    for clave in config.AGENTES:
        if _responde(clave):
            continue
        if clave in _HILOS and _HILOS[clave].is_alive():
            continue
        srv = _servidor(clave)
        hilo = threading.Thread(target=srv.run, daemon=True, name=f"a2a-{clave}")
        hilo.start()
        _SERVIDORES[clave] = srv
        _HILOS[clave] = hilo
        arrancados.append(clave)

    limite = time.time() + espera_maxima
    pendientes = set(config.AGENTES)
    while pendientes and time.time() < limite:
        pendientes = {c for c in pendientes if not _responde(c, timeout=0.5)}
        if pendientes:
            time.sleep(0.3)

    return {
        "ok": not pendientes,
        "arrancados": arrancados,
        "no_responden": sorted(pendientes),
        "en_linea": [c for c in config.AGENTES if _responde(c)],
    }


def detener() -> None:
    for srv in _SERVIDORES.values():
        srv.should_exit = True


def main() -> int:
    import multiprocessing

    procesos = []
    for clave in config.AGENTES:
        p = multiprocessing.Process(
            target=_correr_proceso, args=(clave,), name=f"a2a-{clave}", daemon=False
        )
        p.start()
        procesos.append(p)
        print(f"  {config.AGENTES[clave]['nombre']:42} {config.url_agente(clave)}")

    print("\nAgentes A2A arriba. Ctrl+C para detener.")
    try:
        for p in procesos:
            p.join()
    except KeyboardInterrupt:
        for p in procesos:
            p.terminate()
    return 0


def _correr_proceso(clave: str) -> None:
    uvicorn.run(
        construir_app(clave),
        host=config.HOST_AGENTES,
        port=config.AGENTES[clave]["puerto"],
        log_level="warning",
    )


if __name__ == "__main__":
    raise SystemExit(main())
