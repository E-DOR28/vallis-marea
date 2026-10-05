"""Protocolo de evaluacion del router (R2).

Todo lo que decide como se mide vive aqui y no depende de torch, para que el
protocolo se pueda congelar, probar y revisar sin entrenar nada.

Decisiones que importan para que las cifras sean creibles con 183 ejemplos:

* Los ejemplos del dominio son frases cortas escritas por el equipo y muchas
  son casi iguales ("cuanto cuesta el paseo a baru" / "cuanto vale el paseo a
  baru"). Un split al azar deja una en entrenamiento y la otra en prueba, y la
  metrica premia memorizar. Por eso se agrupan los casi-duplicados y las
  particiones nunca separan un grupo.
* Con 46 ejemplos de prueba una sola particion tiene un intervalo enorme. Se
  usa validacion cruzada por grupos repetida y un bootstrap por grupos sobre
  las diferencias pareadas entre modelos.
* El umbral de confianza se elige solo con datos de validacion tomados del
  entrenamiento, nunca con la prueba.
"""

from __future__ import annotations

import difflib
import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

SEMILLAS = (7, 11, 23)
PARTICIONES = 5
SEMILLA_SPLIT_FIJO = 7
TAMANO_PRUEBA_FIJO = 0.25
FRACCION_VALIDACION = 0.2

UMBRAL_SIMILITUD_CARACTERES = 0.85
UMBRAL_JACCARD_PALABRAS = 0.8
MIN_PALABRAS_JACCARD = 3

EXACTITUD_OBJETIVO_UMBRAL = 0.95
COBERTURA_MINIMA_UMBRAL = 0.5


def sha256_archivo(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def normalizar(texto: str) -> str:
    """Minusculas, sin tildes ni puntuacion, espacios colapsados."""
    t = unicodedata.normalize("NFKD", texto.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _son_casi_duplicados(a: str, b: str) -> bool:
    if a == b:
        return True
    if difflib.SequenceMatcher(None, a, b).ratio() >= UMBRAL_SIMILITUD_CARACTERES:
        return True
    pa, pb = set(a.split()), set(b.split())
    if min(len(pa), len(pb)) >= MIN_PALABRAS_JACCARD:
        return len(pa & pb) / len(pa | pb) >= UMBRAL_JACCARD_PALABRAS
    return False


def agrupar_casi_duplicados(textos: Sequence[str]) -> np.ndarray:
    """Asigna un id de grupo a cada texto; los casi-duplicados comparten id.

    Union-find sobre todos los pares. Con unos cientos de textos cuesta menos
    de un segundo. Se compara entre cualquier par, sin mirar la etiqueta: si
    dos frases casi iguales tuvieran etiquetas distintas tambien deben quedar
    juntas.
    """
    norm = [normalizar(t) for t in textos]
    padre = list(range(len(norm)))

    def raiz(i: int) -> int:
        while padre[i] != i:
            padre[i] = padre[padre[i]]
            i = padre[i]
        return i

    for i in range(len(norm)):
        for j in range(i + 1, len(norm)):
            if _son_casi_duplicados(norm[i], norm[j]):
                ri, rj = raiz(i), raiz(j)
                if ri != rj:
                    padre[max(ri, rj)] = min(ri, rj)

    ids: dict[int, int] = {}
    grupos = np.zeros(len(norm), dtype=int)
    for i in range(len(norm)):
        grupos[i] = ids.setdefault(raiz(i), len(ids))
    return grupos


def resumen_grupos(textos: Sequence[str], grupos: np.ndarray) -> dict[str, Any]:
    tamanos = np.bincount(grupos)
    multiples = [int(g) for g in np.where(tamanos > 1)[0]]
    return {
        "n_ejemplos": int(len(textos)),
        "n_grupos": int(len(tamanos)),
        "n_grupos_con_casi_duplicados": len(multiples),
        "n_ejemplos_en_grupos_multiples": int(tamanos[multiples].sum()) if multiples else 0,
        "tamano_maximo_grupo": int(tamanos.max()),
        "ejemplos_de_grupos": [
            [textos[i] for i in np.where(grupos == g)[0]] for g in multiples[:8]
        ],
    }


def particiones_cv(
    etiquetas: Sequence[str],
    grupos: np.ndarray,
    k: int = PARTICIONES,
    semillas: Iterable[int] = SEMILLAS,
) -> list[dict[str, Any]]:
    """Validacion cruzada estratificada por grupo, repetida una vez por semilla."""
    y = np.asarray(etiquetas)
    salida: list[dict[str, Any]] = []
    for rep, semilla in enumerate(semillas):
        cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=semilla)
        for fold, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y, grupos)):
            salida.append({
                "repeticion": rep, "semilla": int(semilla), "particion": fold,
                "entrenamiento": tr.tolist(), "prueba": te.tolist(),
            })
    return salida


def split_fijo(etiquetas: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Replica el split de `entrenar.py` (semilla 7, 25 %, estratificado).

    Sirve solo para comparar con la linea base. No respeta grupos; por eso se
    reporta aparte cuantos ejemplos de prueba tienen un casi-duplicado en
    entrenamiento (`filtracion_split_fijo`).
    """
    y = np.asarray(etiquetas)
    idx = np.arange(len(y))
    tr, te = train_test_split(
        idx, test_size=TAMANO_PRUEBA_FIJO, random_state=SEMILLA_SPLIT_FIJO, stratify=y
    )
    return np.sort(tr), np.sort(te)


def filtracion_entre(grupos: np.ndarray, entrenamiento: Sequence[int],
                     prueba: Sequence[int]) -> list[int]:
    """Indices de prueba cuyo grupo tambien aparece en entrenamiento."""
    g_tr = set(int(grupos[i]) for i in entrenamiento)
    return [int(i) for i in prueba if int(grupos[i]) in g_tr]


def semilla_validacion(semilla: int, particion: int) -> int:
    """Misma validacion interna para todos los modelos de una particion."""
    return semilla * 100 + particion


def dividir_validacion(
    entrenamiento: Sequence[int],
    etiquetas: Sequence[str],
    grupos: np.ndarray,
    semilla: int,
    fraccion: float = FRACCION_VALIDACION,
) -> tuple[np.ndarray, np.ndarray]:
    """Separa del entrenamiento una validacion estratificada por grupo."""
    entrenamiento = np.asarray(entrenamiento)
    y = np.asarray(etiquetas)[entrenamiento]
    g = grupos[entrenamiento]
    k = max(2, round(1 / fraccion))
    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=semilla)
    tr, va = next(cv.split(np.zeros(len(y)), y, g))
    return entrenamiento[tr], entrenamiento[va]


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------
def f1_macro(y_true: Sequence[str], y_pred: Sequence[str], clases: Sequence[str]) -> float:
    return float(f1_score(y_true, y_pred, labels=list(clases), average="macro",
                          zero_division=0))


def exactitud(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def umbral_por_cobertura_exactitud(
    confianza: Sequence[float],
    correcto: Sequence[bool],
    objetivo: float = EXACTITUD_OBJETIVO_UMBRAL,
    cobertura_min: float = COBERTURA_MINIMA_UMBRAL,
) -> dict[str, Any]:
    """Elige el umbral desde la curva cobertura-exactitud de la validacion.

    Devuelve el menor umbral cuya exactitud sobre lo aceptado (conf >= umbral)
    llega al objetivo sin bajar de la cobertura minima. Si ninguno cumple,
    `cumple` es False y el umbral queda por encima de toda la validacion: el
    clasificador no decide solo y todo pasa al LLM. Esa salida es deliberada.
    """
    c = np.asarray(confianza, dtype=float)
    ok = np.asarray(correcto, dtype=bool)
    candidatos = np.unique(c)
    curva = []
    elegido = None
    for u in candidatos:
        acepta = c >= u
        cobertura = float(acepta.mean())
        acc = float(ok[acepta].mean()) if acepta.any() else 0.0
        curva.append({"umbral": round(float(u), 4), "cobertura": round(cobertura, 4),
                      "exactitud": round(acc, 4)})
        if elegido is None and acc >= objetivo and cobertura >= cobertura_min:
            elegido = float(u)
    if elegido is None:
        return {"umbral": float(c.max()) + 1e-6 if len(c) else 1.0, "cumple": False,
                "curva": curva}
    return {"umbral": elegido, "cumple": True, "curva": curva}


# ---------------------------------------------------------------------------
# Bootstrap por grupos sobre diferencias pareadas
# ---------------------------------------------------------------------------
def bootstrap_diferencia_f1(
    y_true: Sequence[str],
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    grupos: np.ndarray,
    clases: Sequence[str],
    replicas: int = 5000,
    semilla: int = 2026,
) -> dict[str, float]:
    """IC del 95 % de F1macro(A) - F1macro(B).

    `pred_a` y `pred_b` tienen forma (repeticiones, n): la prediccion fuera de
    muestra de cada ejemplo en cada repeticion. En cada replica se remuestrean
    grupos completos con reposicion (no ejemplos sueltos, para no subestimar la
    varianza por casi-duplicados) y se promedia el F1 sobre las repeticiones.

    Mide la incertidumbre por la muestra de ejemplos; la varianza del
    entrenamiento queda reflejada en `sd_entre_repeticiones` del informe.
    """
    codigo = {c: i for i, c in enumerate(clases)}
    k = len(codigo)
    y = np.array([codigo[v] for v in y_true])
    # Una prediccion invalida (p. ej. el LLM no devolvio JSON) cae en el
    # indice k: cuenta como error y no pertenece a ninguna clase.
    pa = np.vectorize(lambda v: codigo.get(v, k))(np.asarray(pred_a))
    pb = np.vectorize(lambda v: codigo.get(v, k))(np.asarray(pred_b))
    rng = np.random.default_rng(semilla)
    ids = np.unique(grupos)
    miembros = {g: np.where(grupos == g)[0] for g in ids}

    def f1_rapido(yt: np.ndarray, yp: np.ndarray) -> float:
        # Misma definicion que sklearn con zero_division=0 y todas las clases.
        tp = np.bincount(yt[yt == yp], minlength=k + 1)[:k].astype(float)
        den = (np.bincount(yt, minlength=k + 1) + np.bincount(yp, minlength=k + 1))[:k]
        return float(np.mean(np.divide(2 * tp, den, out=np.zeros(k), where=den > 0)))

    def diff(idx: np.ndarray) -> float:
        fa = np.mean([f1_rapido(y[idx], pa[r][idx]) for r in range(pa.shape[0])])
        fb = np.mean([f1_rapido(y[idx], pb[r][idx]) for r in range(pb.shape[0])])
        return float(fa - fb)

    base = diff(np.arange(len(y)))
    muestras = np.empty(replicas)
    for b in range(replicas):
        elegidos = rng.choice(ids, size=len(ids), replace=True)
        idx = np.concatenate([miembros[g] for g in elegidos])
        muestras[b] = diff(idx)
    return {
        "diferencia": round(base, 4),
        "ic95_inferior": round(float(np.percentile(muestras, 2.5)), 4),
        "ic95_superior": round(float(np.percentile(muestras, 97.5)), 4),
        "replicas": replicas,
    }
