"""Pruebas de la integracion del router afinado en la cascada (R2).

Se ejecutan con:  python -m unittest tests.test_router_predecir -v
No usan red, clave de API ni torch: el modelo afinado se simula.
"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from core import config
from core.router import predecir
from core.router.entrenar import RUTA_MODELO
from core.router.predecir import Prediccion


class CargaSinPickleTest(unittest.TestCase):
    @unittest.skipUnless(RUTA_MODELO.exists(), "no hay router_intencion.npz entrenado")
    def test_el_modelo_actual_se_carga_con_allow_pickle_false(self) -> None:
        d = np.load(RUTA_MODELO, allow_pickle=False)
        self.assertEqual(set(d.files), {"coef", "intercept", "clases", "dim", "degradado"})
        self.assertEqual(d["clases"].dtype.kind, "U")


class AfinadoNoDisponibleTest(unittest.TestCase):
    def setUp(self) -> None:
        predecir._AFINADO_ERROR = None

    def tearDown(self) -> None:
        predecir._AFINADO_ERROR = None

    def test_sin_modelo_configurado_devuelve_none(self) -> None:
        with mock.patch.object(config, "ROUTER_MODELO", ""):
            self.assertIsNone(predecir._por_afinado("hola"))

    def test_si_falla_la_carga_cae_a_embeddings_y_no_reintenta(self) -> None:
        with mock.patch.object(config, "ROUTER_NIVEL1", "afinado"), \
                mock.patch.object(config, "ROUTER_MODELO", "usuario/repo@" + "a" * 40), \
                mock.patch("core.router.afinado.modelo_actual",
                           side_effect=RuntimeError("sin red")) as carga, \
                mock.patch.object(predecir, "_cargar_modelo", return_value=None):
            self.assertIsNone(predecir._por_clasificador("hola"))
            self.assertIn("sin red", predecir._AFINADO_ERROR or "")
            predecir._por_clasificador("hola otra vez")
            self.assertEqual(carga.call_count, 1)

    def test_usa_embeddings_por_defecto(self) -> None:
        self.assertEqual(config.ROUTER_NIVEL1, "embeddings")


class UmbralPropioTest(unittest.TestCase):
    def test_el_umbral_del_modelo_afinado_decide_si_se_consulta_al_llm(self) -> None:
        llm = Prediccion("precio", 0.9, "llm")
        dudoso = Prediccion("saludo", 0.6, "clasificador", umbral=0.8)
        with mock.patch.object(predecir, "_por_clasificador", return_value=dudoso), \
                mock.patch.object(predecir, "_por_llm", return_value=llm) as por_llm:
            self.assertEqual(predecir.predecir("x").metodo, "llm")
            por_llm.assert_called_once()

    def test_sobre_el_umbral_propio_no_se_consulta_al_llm(self) -> None:
        seguro = Prediccion("saludo", 0.85, "clasificador", umbral=0.8)
        with mock.patch.object(predecir, "_por_clasificador", return_value=seguro), \
                mock.patch.object(predecir, "_por_llm") as por_llm:
            self.assertEqual(predecir.predecir("x").metodo, "clasificador")
            por_llm.assert_not_called()

    def test_sin_umbral_propio_rige_el_de_config(self) -> None:
        justo_debajo = config.UMBRAL_CONFIANZA_ROUTER - 0.01
        p = Prediccion("saludo", justo_debajo, "clasificador")
        with mock.patch.object(predecir, "_por_clasificador", return_value=p), \
                mock.patch.object(predecir, "_por_llm", return_value=None) as por_llm:
            predecir.predecir("x")
            por_llm.assert_called_once()

    def test_el_contrato_del_diccionario_no_cambia(self) -> None:
        d = Prediccion("saludo", 0.8, "clasificador", umbral=0.7,
                       alternativas=[("saludo", 0.8)]).a_dict()
        self.assertEqual(set(d), {"intencion", "confianza", "metodo", "ms",
                                  "alternativas", "agente_destino"})


if __name__ == "__main__":
    unittest.main()
