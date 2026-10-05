"""Base de datos SQLite temporal, sembrada con flota.json y rutas.json."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core import config
from core.almacen import estructurado


class ConBaseTemporal(unittest.TestCase):
    """Cada prueba corre contra su propia base: nunca toca data/almacen/vallis.db."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._ruta_original = config.RUTA_SQLITE
        config.RUTA_SQLITE = Path(self._tmp.name) / "vallis.db"
        estructurado.inicializar()

    def tearDown(self) -> None:
        config.RUTA_SQLITE = self._ruta_original
        self._tmp.cleanup()

    @staticmethod
    def contar_reservas(estados: tuple[str, ...] | None = None) -> int:
        con = estructurado.conexion()
        try:
            if estados is None:
                return con.execute("SELECT COUNT(*) FROM reservas").fetchone()[0]
            marcas = ",".join("?" * len(estados))
            return con.execute(
                f"SELECT COUNT(*) FROM reservas WHERE estado IN ({marcas})", estados
            ).fetchone()[0]
        finally:
            con.close()
