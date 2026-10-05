"""Pruebas del protocolo de evaluacion del router (R2).

Se ejecutan con:  python -m unittest tests.test_router_protocolo -v
No necesitan red, clave ni torch.
"""

from __future__ import annotations

import json
import unittest

import numpy as np

from core import config
from core.router import datos
from core.router import protocolo as P

CASOS_DUPLICADOS = [
    "cuanto cuesta el paseo a baru",
    "Cuánto cuesta el paseo a Baru?",
    "cuanto cuesta el paseo a baru hoy",
    "hay lancha libre el sabado",
    "quiero reservar para el domingo",
    "quiero reservar para el domingo por favor",
]


def _dataset() -> tuple[list[str], np.ndarray]:
    d = json.loads(datos.RUTA_DATASET.read_text(encoding="utf-8"))
    return [e["texto"] for e in d["ejemplos"]], np.array([e["intencion"] for e in d["ejemplos"]])


class AgrupamientoTest(unittest.TestCase):
    def test_casi_duplicados_comparten_grupo(self) -> None:
        g = P.agrupar_casi_duplicados(CASOS_DUPLICADOS)
        self.assertEqual(g[0], g[1])
        self.assertEqual(g[0], g[2])
        self.assertEqual(g[4], g[5])
        self.assertNotEqual(g[0], g[3])
        self.assertNotEqual(g[3], g[4])

    def test_no_mira_la_etiqueta(self) -> None:
        # Dos frases casi iguales con etiquetas distintas deben quedar juntas:
        # el agrupamiento solo recibe textos.
        g = P.agrupar_casi_duplicados(["incluye almuerzo el paseo", "incluye almuerzo el paseo?"])
        self.assertEqual(g[0], g[1])


class ParticionesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.textos, cls.y = _dataset()
        cls.grupos = P.agrupar_casi_duplicados(cls.textos)
        cls.partes = P.particiones_cv(cls.y, cls.grupos)

    def test_cantidad(self) -> None:
        self.assertEqual(len(self.partes), P.PARTICIONES * len(P.SEMILLAS))

    def test_ningun_grupo_cruza_entrenamiento_y_prueba(self) -> None:
        for p in self.partes:
            g_tr = set(self.grupos[p["entrenamiento"]])
            g_te = set(self.grupos[p["prueba"]])
            self.assertFalse(g_tr & g_te, p)

    def test_entrenamiento_y_prueba_son_disjuntos_y_completos(self) -> None:
        for p in self.partes:
            self.assertFalse(set(p["entrenamiento"]) & set(p["prueba"]))
            self.assertEqual(len(p["entrenamiento"]) + len(p["prueba"]), len(self.y))

    def test_cada_ejemplo_se_evalua_una_vez_por_repeticion(self) -> None:
        for rep in range(len(P.SEMILLAS)):
            vistos = [i for p in self.partes if p["repeticion"] == rep for i in p["prueba"]]
            self.assertEqual(sorted(vistos), list(range(len(self.y))))

    def test_toda_clase_aparece_en_cada_prueba(self) -> None:
        for p in self.partes:
            self.assertEqual(set(self.y[p["prueba"]]), set(self.y))

    def test_son_deterministas(self) -> None:
        otra = P.particiones_cv(self.y, self.grupos)
        self.assertEqual(self.partes, otra)

    def test_la_validacion_interna_no_toca_la_prueba(self) -> None:
        for p in self.partes[:5]:
            tr, va = P.dividir_validacion(p["entrenamiento"], self.y, self.grupos, 7)
            self.assertFalse(set(tr) & set(va))
            self.assertFalse(set(va) & set(p["prueba"]))
            self.assertFalse(set(tr) & set(p["prueba"]))
            self.assertEqual(set(self.y[va]), set(self.y))

    def test_split_fijo_replica_la_linea_base(self) -> None:
        tr, te = P.split_fijo(self.y)
        self.assertEqual((len(tr), len(te)), (137, 46))

    def test_clases_del_dataset_son_las_del_catalogo(self) -> None:
        self.assertEqual(sorted(set(self.y)), sorted(config.INTENCIONES))


class FiltracionConGruposTest(unittest.TestCase):
    """Con casi-duplicados reales, el reparto por grupo evita la filtracion que un
    reparto al azar si produce. Es la razon de ser del agrupamiento."""

    def test_reparto_por_grupo_no_filtra_y_el_aleatorio_si(self) -> None:
        textos, etiquetas = [], []
        for c, plantilla in enumerate(["cuanto cuesta el paseo a {}", "hay lancha libre el {}",
                                       "quiero reservar para el {}"]):
            for lugar in ["baru", "cholon", "rosario", "sabado", "domingo", "lunes"]:
                for variante in ("", " por favor", " hoy"):
                    textos.append(plantilla.format(lugar) + variante)
                    etiquetas.append(f"clase{c}")
        g = P.agrupar_casi_duplicados(textos)
        y = np.array(etiquetas)
        for p in P.particiones_cv(y, g, k=3):
            self.assertFalse(P.filtracion_entre(g, p["entrenamiento"], p["prueba"]))

        rng = np.random.default_rng(0)
        idx = rng.permutation(len(textos))
        tr, te = idx[: len(idx) * 2 // 3], idx[len(idx) * 2 // 3 :]
        self.assertTrue(P.filtracion_entre(g, tr, te))


class UmbralTest(unittest.TestCase):
    def test_elige_el_menor_umbral_que_cumple(self) -> None:
        r = P.umbral_por_cobertura_exactitud(
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4], [1, 1, 1, 0, 1, 0], objetivo=0.95, cobertura_min=0.5)
        self.assertTrue(r["cumple"])
        self.assertAlmostEqual(r["umbral"], 0.7)

    def test_si_nada_cumple_todo_pasa_al_llm(self) -> None:
        r = P.umbral_por_cobertura_exactitud([0.9, 0.8], [0, 0])
        self.assertFalse(r["cumple"])
        self.assertGreater(r["umbral"], 0.9)

    def test_respeta_la_cobertura_minima(self) -> None:
        r = P.umbral_por_cobertura_exactitud([0.99, 0.5, 0.4, 0.3], [1, 0, 0, 0],
                                             objetivo=0.95, cobertura_min=0.5)
        self.assertFalse(r["cumple"])


class BootstrapTest(unittest.TestCase):
    def test_coincide_con_sklearn_y_el_intervalo_contiene_la_diferencia(self) -> None:
        rng = np.random.default_rng(0)
        clases = [f"c{i}" for i in range(5)]
        y = np.array([clases[i % 5] for i in range(100)])
        g = np.arange(100)

        def ruido(p: float) -> np.ndarray:
            return np.array([[y[i] if rng.random() < p else clases[rng.integers(5)]
                              for i in range(100)] for _ in range(3)])

        a, b = ruido(0.95), ruido(0.55)
        r = P.bootstrap_diferencia_f1(y, a, b, g, clases, replicas=500)
        ref = (np.mean([P.f1_macro(y, a[i], clases) for i in range(3)])
               - np.mean([P.f1_macro(y, b[i], clases) for i in range(3)]))
        self.assertAlmostEqual(r["diferencia"], ref, places=3)
        self.assertLessEqual(r["ic95_inferior"], r["diferencia"])
        self.assertGreaterEqual(r["ic95_superior"], r["diferencia"])
        self.assertGreater(r["ic95_inferior"], 0)

    def test_prediccion_invalida_cuenta_como_error(self) -> None:
        clases = ["a", "b"]
        y = np.array(["a", "b"] * 10)
        buena = np.array([y.tolist()])
        mala = np.array([["__error__"] * 20])
        r = P.bootstrap_diferencia_f1(y, buena, mala, np.arange(20), clases, replicas=100)
        self.assertAlmostEqual(r["diferencia"], 1.0, places=3)


if __name__ == "__main__":
    unittest.main()


class ProtocoloCongeladoTest(unittest.TestCase):
    """El codigo debe coincidir con lo que se congelo antes de entrenar."""

    @classmethod
    def setUpClass(cls) -> None:
        from core.router import afinar
        cls.A = afinar
        cls.v1 = json.loads(afinar.RUTA_PROTOCOLO.read_text(encoding="utf-8"))
        cls.v2 = json.loads(afinar.RUTA_PROTOCOLO_V2.read_text(encoding="utf-8"))

    def test_hiperparametros_congelados_son_los_del_codigo(self) -> None:
        self.assertEqual(self.v1["hiperparametros"], self.A.HIPER_V1)
        self.assertEqual(self.v2["hiperparametros"], self.A.HIPER_V2)

    def test_la_enmienda_solo_cambia_la_regla_de_parada(self) -> None:
        distintas = {k for k in self.A.HIPER_V1 if self.A.HIPER_V1[k] != self.A.HIPER_V2[k]}
        self.assertEqual(distintas, {"epocas_max", "paciencia"})

    def test_el_dataset_no_cambio_desde_que_se_congelo(self) -> None:
        sha = P.sha256_archivo(datos.RUTA_DATASET)
        self.assertEqual(self.v1["dataset"]["sha256"], sha)
        self.assertEqual(self.v2["dataset"]["sha256"], sha)

    def test_las_particiones_son_las_mismas_en_ambas_versiones(self) -> None:
        self.assertEqual(self.v1["validacion_cruzada"]["sha256_particiones"],
                         self.v2["validacion_cruzada"]["sha256_particiones"])
        textos, y = _dataset()
        g = P.agrupar_casi_duplicados(textos)
        self.assertEqual(self.A._hash_particiones(self.A._particiones(y, g)),
                         self.v1["validacion_cruzada"]["sha256_particiones"])

    def test_v2_referencia_el_protocolo_v1_sin_alterarlo(self) -> None:
        self.assertEqual(self.v2["protocolo_base"]["sha256"],
                         P.sha256_archivo(self.A.RUTA_PROTOCOLO))

    def test_xlmr_no_esta_entre_los_candidatos_de_la_v2(self) -> None:
        self.assertEqual(set(self.v2["candidatos"]), {"e5_small", "minilm"})
        self.assertIn("FacebookAI/xlm-roberta-base", self.v2["descartados"])
