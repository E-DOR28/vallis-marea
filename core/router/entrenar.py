"""Entrenamiento del router de intencion.

QUE ES ESTO, CON PRECISION: se entrena una regresion logistica sobre
embeddings **congelados** de Gemini. Los pesos del modelo de embeddings no se
tocan.

Eso NO es fine-tuning de un LLM, y el informe no debe llamarlo asi. Es
`feature extraction` + clasificador lineal. Se eligio por la restriccion de
tiempo del proyecto: afinar un transformer pequeno (XLM-R, DistilBERT
multilingue) sobre este mismo dataset es el paso siguiente inmediato y esta
documentado como trabajo futuro.

Por que hay un router entrenado y no se le pregunta al LLM en cada mensaje:
un clasificador lineal sobre un embedding cuesta un orden de magnitud menos en
latencia y en dinero que una llamada de generacion. El LLM queda de desempate,
solo cuando la confianza del clasificador cae por debajo del umbral.

Uso:  python -m core.router.entrenar
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from core import config
from core.llm import gemini
from core.router import datos

RUTA_MODELO = config.RUTA_MODELOS / "router_intencion.npz"
RUTA_METRICAS = config.RUTA_EVALUACION / "metricas_router.json"


def entrenar(semilla: int = 7, verbose: bool = True) -> dict[str, Any]:
    log = print if verbose else (lambda *a, **k: None)

    d = datos.construir_dataset()
    ejemplos = d["ejemplos"]
    textos = [e["texto"] for e in ejemplos]
    etiquetas = np.array([e["intencion"] for e in ejemplos])

    log(f"Dataset: {len(ejemplos)} ejemplos, {len(d['clases'])} clases")
    log(f"  fuente publica (fuera de dominio): {d['fuera_de_dominio_ok']}")

    log("Calculando embeddings...")
    if gemini.modo_degradado():
        log("  AVISO: sin GOOGLE_API_KEY -> embeddings locales. "
            "Las metricas NO son representativas; reentrenar con la clave.")
    X = gemini.embed_clasificacion(textos)

    # Estratificado: con clases de 15-42 ejemplos, un split aleatorio puede
    # dejar una clase entera fuera del test y volver la metrica ilegible.
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, etiquetas, test_size=0.25, random_state=semilla, stratify=etiquetas
    )

    modelo = LogisticRegression(
        max_iter=2000,
        C=4.0,
        class_weight="balanced",  # `otro` tiene el doble de ejemplos
        random_state=semilla,
    )
    modelo.fit(X_tr, y_tr)

    y_pred = modelo.predict(X_te)
    f1_macro = float(f1_score(y_te, y_pred, average="macro"))
    exactitud = float((y_pred == y_te).mean())

    reporte = classification_report(y_te, y_pred, output_dict=True, zero_division=0)
    matriz = confusion_matrix(y_te, y_pred, labels=sorted(set(etiquetas)))

    log(f"\n  exactitud (held-out): {exactitud:.3f}")
    log(f"  F1 macro:             {f1_macro:.3f}")
    log("\n" + classification_report(y_te, y_pred, zero_division=0))

    np.savez(
        RUTA_MODELO,
        coef=modelo.coef_,
        intercept=modelo.intercept_,
        clases=modelo.classes_,
        dim=np.array([X.shape[1]]),
        degradado=np.array([gemini.modo_degradado()]),
    )

    metricas = {
        "exactitud": round(exactitud, 4),
        "f1_macro": round(f1_macro, 4),
        "n_entrenamiento": int(len(y_tr)),
        "n_prueba": int(len(y_te)),
        "clases": sorted(set(etiquetas)),
        "dimension_embedding": int(X.shape[1]),
        "modelo_embedding": "local-degradado" if gemini.modo_degradado()
                            else config.MODELO_EMBEDDING,
        "embeddings_degradados": gemini.modo_degradado(),
        "por_clase": {
            c: {k: round(v, 4) for k, v in m.items()}
            for c, m in reporte.items()
            if isinstance(m, dict) and c in set(etiquetas)
        },
        "matriz_confusion": {
            "etiquetas": sorted(set(etiquetas)),
            "matriz": matriz.tolist(),
        },
        "dataset": {
            "total": len(ejemplos),
            "conteo_por_clase": d["conteo_por_clase"],
            "fuente_publica": d["fuente_publica"],
        },
        "nota_metodologica": (
            "Regresion logistica sobre embeddings congelados. No es fine-tuning "
            "de un LLM: los pesos del modelo de embeddings no se modifican."
        ),
    }
    RUTA_METRICAS.write_text(
        json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"\nModelo -> {RUTA_MODELO.name} | metricas -> {RUTA_METRICAS.name}")
    return metricas


if __name__ == "__main__":
    entrenar()
