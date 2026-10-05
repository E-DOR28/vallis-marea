"""Baselines del R2, evaluados con las mismas particiones que los candidatos.

    B0  regresion logistica sobre embeddings de Gemini (el router actual)
    B1  TF-IDF con regresion logistica
    B2  Gemini flash-lite sin entrenamiento, con el prompt de `predecir.py`
    B3  embeddings locales congelados (el encoder sin afinar) con regresion
        logistica: aisla el efecto de afinar frente al de cambiar de embedding

B0, B1 y B3 usan los mismos hiperparametros que el router actual (C=4,
class_weight balanceado) y no se ajustan: lo que se compara es la receta que
hoy esta en produccion.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from core import config
from core.llm import gemini
from core.router import protocolo as P

PARAMETROS_LR = {"C": 4.0, "class_weight": "balanced", "max_iter": 2000}
PARAMETROS_TFIDF = {"analyzer": "char_wb", "ngram_range": (2, 5), "sublinear_tf": True,
                    "strip_accents": "unicode", "lowercase": True}
UMBRAL_PRODUCCION_B0 = config.UMBRAL_CONFIANZA_ROUTER


def _lr(semilla: int) -> LogisticRegression:
    return LogisticRegression(random_state=semilla, **PARAMETROS_LR)


def fabrica_b0(X: np.ndarray, semilla: int) -> Callable[[], Any]:
    return lambda: _Ajuste(X, _lr(semilla))


def fabrica_b3(X: np.ndarray, semilla: int) -> Callable[[], Any]:
    return fabrica_b0(X, semilla)


def fabrica_b1(textos: Sequence[str], semilla: int) -> Callable[[], Any]:
    t = np.asarray(textos, dtype=object)
    return lambda: _Ajuste(
        t, make_pipeline(TfidfVectorizer(**PARAMETROS_TFIDF), _lr(semilla)))


class _Ajuste:
    """Ajusta y predice por indices sobre una matriz densa o un arreglo de textos."""

    def __init__(self, X: np.ndarray, modelo: Any) -> None:
        self.X, self.modelo = X, modelo

    def fit(self, idx: Sequence[int], y: np.ndarray) -> "_Ajuste":
        self.modelo.fit(self.X[idx], y[idx])
        return self

    def predict_proba(self, idx: Sequence[int]) -> np.ndarray:
        return self.modelo.predict_proba(self.X[idx])

    @property
    def clases(self) -> list[str]:
        return [str(c) for c in self.modelo.classes_]


def registro_particion(
    fabrica: Callable[[], Any],
    part: dict[str, Any],
    y: np.ndarray,
    grupos: np.ndarray,
    clases: Sequence[str],
) -> dict[str, Any]:
    """Un baseline en una particion: umbral desde validacion, prueba aparte.

    El umbral sale de un modelo ajustado solo con el entrenamiento menos la
    validacion; las predicciones de prueba, de uno ajustado con todo el
    entrenamiento (como hace el router actual).
    """
    t0 = time.perf_counter()
    tr, te = np.asarray(part["entrenamiento"]), np.asarray(part["prueba"])
    tr2, va = P.dividir_validacion(
        tr, y, grupos, P.semilla_validacion(part["semilla"], part["particion"]))

    m = fabrica().fit(tr2, y)
    assert m.clases == list(clases)
    pv = m.predict_proba(va)
    ok_val = np.asarray(clases)[pv.argmax(1)] == y[va]
    umbral = P.umbral_por_cobertura_exactitud(pv.max(1), ok_val)

    m = fabrica().fit(tr, y)
    pt = m.predict_proba(te)
    return {
        "id": part["id"], "repeticion": part["repeticion"], "semilla": part["semilla"],
        "particion": part["particion"], "prueba": te.tolist(),
        "pred": np.asarray(clases)[pt.argmax(1)].tolist(),
        "confianza": [round(float(v), 5) for v in pt.max(1)],
        "umbral": {k: v for k, v in umbral.items() if k != "curva"},
        "segundos": round(time.perf_counter() - t0, 2),
    }


# ---------------------------------------------------------------------------
# B2: LLM sin entrenamiento
# ---------------------------------------------------------------------------
def predicciones_b2(textos: Sequence[str]) -> dict[str, Any]:
    """Clasifica cada texto con el mismo prompt y esquema de `predecir.py`.

    Se cachea en SQLite (usar_cache=True) para que repetir el analisis no
    cambie las cifras. Una respuesta invalida se registra como `__error__` y
    cuenta como acierto nulo.
    """
    from core.router import predecir as pr

    pred: list[str] = []
    conf: list[float] = []
    errores = 0
    for t in textos:
        resp = gemini.generar(
            f"Mensaje del cliente: {t}",
            sistema=pr._SISTEMA_ROUTER,
            temperatura=config.TEMPERATURA_ROUTER,
            modelo=config.MODELO_GENERACION_RAPIDO,
            esquema_json=pr.ESQUEMA_INTENCION,
            usar_cache=True,
        )
        datos = None
        if not resp.error and resp.texto:
            try:
                datos = json.loads(resp.texto)
            except json.JSONDecodeError:
                datos = None
        if not datos or datos.get("intencion") not in config.INTENCIONES:
            pred.append("__error__")
            conf.append(0.0)
            errores += 1
        else:
            pred.append(datos["intencion"])
            conf.append(float(datos.get("confianza", 0.7)))
    return {"pred": pred, "confianza": conf, "errores": errores,
            "modelo": config.MODELO_GENERACION_RAPIDO}
