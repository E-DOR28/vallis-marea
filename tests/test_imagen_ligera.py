"""La imagen del servicio no trae scikit-learn, torch, pandas ni streamlit.

En el entorno de desarrollo esos paquetes si estan instalados y ocultarian un
import accidental hasta que el contenedor fallara en produccion. Esta prueba
los bloquea en un subproceso e importa todo `core`.

    python -m unittest tests.test_imagen_ligera -v
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

# Solo se usan para entrenar el router (herramienta de desarrollo, no del servicio).
SOLO_ENTRENAMIENTO = {"core.router.entrenar"}

GUION = """
import importlib, pkgutil, sys
for nombre in ("sklearn", "torch", "pandas", "streamlit", "transformers",
               "sentence_transformers", "scipy", "matplotlib"):
    sys.modules[nombre] = None
import core
excluidos = set(sys.argv[1:])
fallan = []
for m in pkgutil.walk_packages(core.__path__, "core."):
    if m.name.endswith("__main__") or m.name in excluidos:
        continue
    try:
        importlib.import_module(m.name)
    except Exception as exc:
        fallan.append(f"{m.name}: {type(exc).__name__}: {exc}")
print("\\n".join(fallan))
sys.exit(1 if fallan else 0)
"""


class ImagenLigeraTest(unittest.TestCase):
    def test_core_se_importa_sin_dependencias_de_entrenamiento(self) -> None:
        r = subprocess.run(
            [sys.executable, "-c", GUION, *sorted(SOLO_ENTRENAMIENTO)],
            cwd=RAIZ, capture_output=True, text=True, timeout=120,
            env={"PYTHONPATH": str(RAIZ), "PATH": "/usr/bin:/bin"},
        )
        self.assertEqual(r.returncode, 0, f"imports que necesitan paquetes pesados:\n{r.stdout}{r.stderr}")

    def test_la_prueba_detecta_un_import_pesado(self) -> None:
        """Sin excluir `entrenar`, el bloqueo debe hacer fallar el import."""
        r = subprocess.run(
            [sys.executable, "-c", GUION],
            cwd=RAIZ, capture_output=True, text=True, timeout=120,
            env={"PYTHONPATH": str(RAIZ), "PATH": "/usr/bin:/bin"},
        )
        self.assertEqual(r.returncode, 1)
        self.assertIn("core.router.entrenar", r.stdout)


if __name__ == "__main__":
    unittest.main()
