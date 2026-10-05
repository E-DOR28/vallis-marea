"""Reintentos con espera exponencial ante fallos transitorios de Gemini.

    python -m unittest tests.test_llm_reintentos -v
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from core import config
from core.llm import gemini


class ErrorApi(Exception):
    def __init__(self, code: int, texto: str = "") -> None:
        super().__init__(texto or f"error {code}")
        self.code = code


def _respuesta(texto: str = "hola"):
    return SimpleNamespace(
        text=texto,
        usage_metadata=SimpleNamespace(prompt_token_count=3, candidates_token_count=2),
    )


def _cliente_con(*resultados):
    """Cliente falso: cada llamada devuelve o lanza el siguiente elemento."""
    cola = list(resultados)

    def generate_content(**_):
        r = cola.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    modelos = SimpleNamespace(generate_content=generate_content)
    return SimpleNamespace(models=modelos), cola


class ReintentosTest(unittest.TestCase):
    def setUp(self) -> None:
        p = [
            mock.patch.object(config, "GOOGLE_API_KEY", "clave-falsa"),
            mock.patch.object(config, "LLM_REINTENTOS", 2),
            mock.patch.object(gemini.time, "sleep"),
        ]
        self.dormir = None
        for patcher in p:
            m = patcher.start()
            self.addCleanup(patcher.stop)
            if patcher is p[2]:
                self.dormir = m

    def _generar(self, *resultados):
        cli, cola = _cliente_con(*resultados)
        with mock.patch.object(gemini, "cliente", return_value=cli):
            return gemini.generar("pregunta"), cola

    def test_un_503_transitorio_se_reintenta_y_se_recupera(self) -> None:
        r, cola = self._generar(ErrorApi(503), ErrorApi(429), _respuesta("listo"))
        self.assertIsNone(r.error)
        self.assertEqual(r.texto, "listo")
        self.assertEqual(cola, [])
        self.assertEqual(self.dormir.call_count, 2)

    def test_la_espera_crece_de_forma_exponencial(self) -> None:
        self._generar(ErrorApi(503), ErrorApi(503), _respuesta())
        esperas = [c.args[0] for c in self.dormir.call_args_list]
        self.assertLess(esperas[0], esperas[1])

    def test_un_error_del_cliente_no_se_reintenta(self) -> None:
        r, cola = self._generar(ErrorApi(400, "peticion invalida"), _respuesta())
        self.assertIn("ErrorApi", r.error)
        self.assertEqual(len(cola), 1, "no debio consumir un segundo intento")
        self.dormir.assert_not_called()

    def test_al_agotar_los_reintentos_devuelve_el_error_sin_lanzarlo(self) -> None:
        r, cola = self._generar(ErrorApi(503), ErrorApi(503), ErrorApi(503), _respuesta())
        self.assertTrue(r.error)
        self.assertEqual(r.texto, "")
        self.assertEqual(len(cola), 1)

    def test_con_reintentos_en_cero_hay_un_solo_intento(self) -> None:
        with mock.patch.object(config, "LLM_REINTENTOS", 0):
            r, cola = self._generar(ErrorApi(503), _respuesta())
        self.assertTrue(r.error)
        self.assertEqual(len(cola), 1)

    def test_clasificacion_de_errores_transitorios(self) -> None:
        self.assertTrue(gemini._es_transitorio(ErrorApi(500)))
        self.assertTrue(gemini._es_transitorio(ErrorApi(429)))
        self.assertTrue(gemini._es_transitorio(RuntimeError("503 UNAVAILABLE: overloaded")))
        self.assertTrue(gemini._es_transitorio(TimeoutError("tiempo")))
        self.assertFalse(gemini._es_transitorio(ErrorApi(400)))
        self.assertFalse(gemini._es_transitorio(ErrorApi(403)))
        self.assertFalse(gemini._es_transitorio(ValueError("x")))


if __name__ == "__main__":
    unittest.main()
