"""Punto de entrada del contenedor:  python -m core.web"""

from __future__ import annotations

import os

import uvicorn

from core.web.app import app_por_defecto


def main() -> None:
    # Un solo worker: los agentes A2A viven en hilos de este proceso y el estado
    # de sesiones esta en memoria. proxy_headers=False porque la IP del cliente
    # se resuelve en core.web.seguridad, no aqui.
    uvicorn.run(
        app_por_defecto(),
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8080")),
        workers=1,
        proxy_headers=False,
        server_header=False,
        access_log=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
