"""Construye un almacen limpio para la imagen del contenedor.

    python scripts/preparar_almacen.py            # escribe en build/almacen

Parte de cero: indices del corpus (vectorial y lexico) y la base SQLite con la
flota, las rutas y solo las reservas de la semilla. No hereda reservas, turnos
ni memoria episodica de corridas de desarrollo. Necesita GOOGLE_API_KEY una
sola vez, para los embeddings del corpus; la cache de embeddings no se publica.

El servicio tambien reconstruye el indice al arrancar si no lo encuentra, asi
que la imagen funciona sin esta carpeta; incluirla evita ese costo en cada
arranque en frio.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DESTINO = RAIZ / "build" / "almacen"

if DESTINO.exists():
    shutil.rmtree(DESTINO)
DESTINO.mkdir(parents=True)
os.environ["VM_RUTA_ALMACEN"] = str(DESTINO)
sys.path.insert(0, str(RAIZ))

from core import config  # noqa: E402
from core.almacen import estructurado, lexico, vectorial  # noqa: E402
from core.ingesta import indexar  # noqa: E402
from core.llm import gemini  # noqa: E402


def main() -> int:
    if gemini.modo_degradado():
        print("Falta GOOGLE_API_KEY: un indice con embeddings locales no sirve en produccion.")
        return 1
    assert config.RUTA_ALMACEN == DESTINO, "el destino no se aplico"

    r = indexar.indexar_todo(recrear=True, verbose=False)
    if not r.get("ok", True):
        print("Fallo la indexacion:", r)
        return 1
    estructurado.inicializar(recrear=True)

    con = estructurado.conexion()
    try:
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        con.execute("PRAGMA journal_mode = DELETE")
        estados = dict(con.execute("SELECT estado, COUNT(*) FROM reservas GROUP BY estado").fetchall())
    finally:
        con.close()

    # La cache guarda cada texto consultado: no debe viajar en la imagen.
    (DESTINO / "cache_llm.db").unlink(missing_ok=True)

    n_vec, n_lex = vectorial.contar(), lexico.contar()
    print(f"Almacen en {DESTINO.relative_to(RAIZ)}: {n_vec} vectores, {n_lex} filas lexicas, reservas {estados}")
    if n_vec == 0 or n_lex == 0 or set(estados) - {"confirmada"}:
        print("El almacen no quedo como se esperaba.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
