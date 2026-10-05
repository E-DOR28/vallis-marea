"""Fine-tuning de un encoder para el router de intencion (R2).

QUE ES: se afinan **todos los pesos** de un encoder multilingue (118 M de
parametros en el candidato principal) junto con una cabeza lineal, con entropia
cruzada ponderada por clase. NO es fine-tuning de un LLM generativo.

Ordenes (desde la raiz del proyecto):

    python -m core.router.afinar congelar      # escribe data/evaluacion/r2/protocolo.json
    python -m core.router.afinar baselines     # B0, B1, B2 y B3
    python -m core.router.afinar cv e5_small   # validacion cruzada de un candidato
    python -m core.router.afinar latencia
    python -m core.router.afinar analizar      # metricas, bootstrap y regla de adopcion
    python -m core.router.afinar final e5_small

El protocolo se congela y se hace commit ANTES de entrenar. `cv` y `baselines`
se niegan a correr si el protocolo o el codigo del router tienen cambios sin
commitear, para que cada resultado quede atado a un commit.
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import random
import statistics
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score

from core import config
from core.router import afinado, baselines, datos
from core.router import protocolo as P

RAIZ = config.RUTA_DATA.parent
DIR_R2 = config.RUTA_EVALUACION / "r2"
RUTA_PROTOCOLO = DIR_R2 / "protocolo.json"
RUTA_METRICAS = config.RUTA_EVALUACION / "metricas_router_afinado.json"
RUTA_ARTEFACTO = config.RUTA_MODELOS / "router_afinado"
ARCHIVOS_DE_CODIGO = [
    "core/router/protocolo.py", "core/router/baselines.py",
    "core/router/afinado.py", "core/router/afinar.py",
]

CLASES = sorted(config.INTENCIONES)

CANDIDATOS: dict[str, dict[str, str]] = {
    "e5_small": {
        "repo": "intfloat/multilingual-e5-small",
        "revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
        "prefijo": "query: ", "licencia": "MIT", "rol": "principal",
    },
    "minilm": {
        "repo": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "revision": "e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
        "prefijo": "", "licencia": "Apache-2.0", "rol": "contraste",
    },
    "xlmr": {
        "repo": "FacebookAI/xlm-roberta-base",
        "revision": "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089",
        "prefijo": "", "licencia": "MIT", "rol": "contraste",
    },
}
DESCARTADOS = {
    "dccuchile/bert-base-spanish-wwm-cased":
        "solo publica pytorch_model.bin (pickle); el protocolo exige safetensors",
}

HIPER: dict[str, Any] = {
    "max_len": 64, "lote": 16, "epocas_max": 15, "paciencia": 3,
    "lrs_encoder": [2e-5, 5e-5], "lr_cabeza": 1e-3, "weight_decay": 0.01,
    "warmup": 0.1, "dropout": 0.1, "clip_gradiente": 1.0,
    "perdida": "entropia cruzada con pesos de clase balanceados",
    "seleccion_epoca": "mayor F1 macro en validacion; ante empate, la epoca anterior",
    "seleccion_lr": "mayor F1 macro en validacion; ante empate, el lr menor",
    "dispositivo": "cpu",
}

REGLA_ADOPCION = {
    "candidato_principal": "e5_small",
    "contrastes": "se reportan; no sustituyen al principal",
    "condicion_1": "limite inferior del IC95 % (bootstrap por grupos) de "
                   "F1macro(afinado) - F1macro(B0) >= -0.02",
    "condicion_2": "latencia p50 local del afinado < latencia p50 de B0 "
                   "(embedding de Gemini sin cache + regresion logistica)",
    "condicion_3": "umbral calibrado sobre validacion (cobertura-exactitud) sin usar la prueba",
    "umbral": {
        "regla": "menor umbral con exactitud >= objetivo y cobertura >= minima en validacion",
        "exactitud_objetivo": P.EXACTITUD_OBJETIVO_UMBRAL,
        "cobertura_minima": P.COBERTURA_MINIMA_UMBRAL,
        "umbral_desplegado": "mediana de los umbrales por particion",
    },
    "lr_desplegado": "el lr elegido con mas frecuencia en la validacion de las particiones",
    "si_no_cumple": "queda disponible como VM_ROUTER=afinado; el informe lo reporta igual",
}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=RAIZ, capture_output=True, text=True,
                          check=False).stdout.strip()


def commit_actual() -> str:
    return _git("rev-parse", "HEAD")


def _cargar_dataset() -> tuple[list[str], np.ndarray, np.ndarray]:
    d = datos.construir_dataset()
    textos = [e["texto"] for e in d["ejemplos"]]
    y = np.array([e["intencion"] for e in d["ejemplos"]])
    assert sorted(set(y)) == CLASES, "las clases del dataset no coinciden con config.INTENCIONES"
    return textos, y, P.agrupar_casi_duplicados(textos)


def _particiones(y: np.ndarray, grupos: np.ndarray) -> list[dict[str, Any]]:
    partes = P.particiones_cv(y, grupos)
    for p in partes:
        p["id"] = f"r{p['repeticion']}p{p['particion']}"
    tr, te = P.split_fijo(y)
    partes.append({"id": "fijo", "repeticion": -1, "semilla": P.SEMILLA_SPLIT_FIJO,
                   "particion": 0, "entrenamiento": tr.tolist(), "prueba": te.tolist()})
    return partes


def _hash_particiones(partes: list[dict[str, Any]]) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(partes, sort_keys=True).encode()).hexdigest()


def _versiones() -> dict[str, str]:
    nombres = ["torch", "transformers", "safetensors", "tokenizers", "scikit-learn",
               "numpy", "huggingface_hub"]
    return {n: metadata.version(n) for n in nombres} | {
        "python": platform.python_version(), "plataforma": platform.platform()}


def _exigir_reproducibilidad() -> None:
    """El protocolo debe estar commiteado y el codigo del router sin cambios."""
    if _git("ls-files", "--error-unmatch", str(RUTA_PROTOCOLO.relative_to(RAIZ))) == "":
        sys.exit("El protocolo no esta en git: ejecuta `congelar` y haz commit antes de entrenar.")
    sucios = _git("status", "--porcelain", "--", str(RUTA_PROTOCOLO.relative_to(RAIZ)),
                  *ARCHIVOS_DE_CODIGO)
    if sucios:
        sys.exit("Hay cambios sin commitear en el protocolo o en el codigo del router:\n" + sucios)
    guardado = json.loads(RUTA_PROTOCOLO.read_text(encoding="utf-8"))
    ruta_ds = datos.RUTA_DATASET
    if P.sha256_archivo(ruta_ds) != guardado["dataset"]["sha256"]:
        sys.exit("El dataset cambio despues de congelar el protocolo.")


# ---------------------------------------------------------------------------
# Congelar el protocolo
# ---------------------------------------------------------------------------
def congelar() -> dict[str, Any]:
    textos, y, grupos = _cargar_dataset()
    ejemplos = json.loads(datos.RUTA_DATASET.read_text(encoding="utf-8"))["ejemplos"]
    partes = _particiones(y, grupos)

    fuga = []
    for p in partes[:-1]:
        solapan = set(grupos[p["entrenamiento"]]) & set(grupos[p["prueba"]])
        if solapan or set(p["entrenamiento"]) & set(p["prueba"]):
            fuga.append(p["id"])
    tr, te = P.split_fijo(y)
    fuga_fijo = P.filtracion_entre(grupos, tr, te)

    protocolo = {
        "congelado_en": time.strftime("%Y-%m-%d %H:%M:%S"),
        "commit_padre": commit_actual(),
        "dataset": {
            "ruta": str(datos.RUTA_DATASET.relative_to(RAIZ)),
            "sha256": P.sha256_archivo(datos.RUTA_DATASET),
            "n": len(textos),
            "conteo_por_clase": {c: int((y == c).sum()) for c in CLASES},
            "origen": {o: sum(1 for e in ejemplos if e["origen"] == o)
                       for o in ("dominio_sintetico", "massive_es")},
        },
        "casi_duplicados": {
            "umbral_similitud_caracteres": P.UMBRAL_SIMILITUD_CARACTERES,
            "umbral_jaccard_palabras": P.UMBRAL_JACCARD_PALABRAS,
            "resultado": P.resumen_grupos(textos, grupos),
            "nota": "Los umbrales se fijaron antes de medir. Con ellos no hay "
                    "casi-duplicados: las particiones por grupo equivalen a las "
                    "estratificadas y la prueba anti-fuga se conserva como guarda.",
        },
        "validacion_cruzada": {
            "particiones": P.PARTICIONES, "semillas": list(P.SEMILLAS),
            "tipo": "StratifiedGroupKFold con shuffle",
            "sha256_particiones": _hash_particiones(partes),
            "fugas_de_grupo": fuga,
            "min_ejemplos_por_clase_en_prueba": min(
                int(sum(1 for i in p["prueba"] if y[i] == c)) for p in partes[:-1]
                for c in CLASES),
        },
        "split_fijo": {"semilla": P.SEMILLA_SPLIT_FIJO, "n_entrenamiento": len(tr),
                       "n_prueba": len(te), "con_casi_duplicado_en_entrenamiento": len(fuga_fijo)},
        "validacion_interna": {"fraccion": P.FRACCION_VALIDACION,
                               "semilla": "semilla * 100 + particion"},
        "bootstrap": {"replicas": 5000, "semilla": 2026, "unidad": "grupo",
                      "estadistico": "F1 macro promediado sobre las 3 repeticiones"},
        "baselines": {
            "B0": "regresion logistica (C=4, balanced) sobre embeddings de Gemini "
                  f"{config.MODELO_EMBEDDING}, tarea CLASSIFICATION",
            "B1": f"TF-IDF {baselines.PARAMETROS_TFIDF} + regresion logistica (C=4, balanced)",
            "B2": f"{config.MODELO_GENERACION_RAPIDO} sin entrenamiento, prompt y esquema de predecir.py",
            "B3": "embeddings congelados del encoder principal (promedio de tokens) "
                  "+ regresion logistica (C=4, balanced)",
        },
        "candidatos": CANDIDATOS,
        "descartados": DESCARTADOS,
        "hiperparametros": HIPER,
        "regla_de_adopcion": REGLA_ADOPCION,
        "versiones": _versiones(),
    }
    DIR_R2.mkdir(parents=True, exist_ok=True)
    RUTA_PROTOCOLO.write_text(json.dumps(protocolo, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"Protocolo -> {RUTA_PROTOCOLO.relative_to(RAIZ)}")
    print(f"  fugas de grupo en particiones: {fuga or 'ninguna'}; "
          f"casi-duplicados: {protocolo['casi_duplicados']['resultado']['n_grupos_con_casi_duplicados']}")
    return protocolo


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------
def _fijar_semillas(semilla: int) -> None:
    import torch
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)


def entrenar_una(cand: dict[str, str], tok: Any, textos: list[str], y: np.ndarray,
                 tr: np.ndarray, va: np.ndarray, lr: float, semilla: int,
                 epocas_max: int | None = None) -> tuple[Any, dict[str, Any]]:
    """Ajusta encoder y cabeza con `tr`; la parada temprana usa `va`."""
    import torch
    from transformers import AutoModel, get_linear_schedule_with_warmup

    h = HIPER
    epocas_max = epocas_max or h["epocas_max"]
    _fijar_semillas(semilla)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    encoder = AutoModel.from_pretrained(cand["repo"], revision=cand["revision"],
                                        use_safetensors=True, trust_remote_code=False)
    modelo = afinado.clase_clasificador()(encoder, len(CLASES), h["dropout"])
    idx_clase = {c: i for i, c in enumerate(CLASES)}
    y_num = np.array([idx_clase[c] for c in y])

    conteo = np.bincount(y_num[tr], minlength=len(CLASES)).astype(float)
    pesos = torch.tensor(conteo.sum() / (len(CLASES) * np.maximum(conteo, 1)),
                         dtype=torch.float32)
    perdida = torch.nn.CrossEntropyLoss(weight=pesos)

    opt = torch.optim.AdamW(
        [{"params": modelo.encoder.parameters(), "lr": lr},
         {"params": modelo.cabeza.parameters(), "lr": h["lr_cabeza"]}],
        weight_decay=h["weight_decay"])
    pasos_epoca = int(np.ceil(len(tr) / h["lote"]))
    total = pasos_epoca * epocas_max
    sched = get_linear_schedule_with_warmup(opt, int(h["warmup"] * total), total)

    textos_np = np.asarray(textos, dtype=object)
    gen = np.random.default_rng(semilla)
    mejor, mejor_f1, mejor_epoca, sin_mejora = None, -1.0, 0, 0
    historia: list[float] = []
    for epoca in range(1, epocas_max + 1):
        modelo.train()
        orden = gen.permutation(tr)
        for i in range(0, len(orden), h["lote"]):
            b = orden[i : i + h["lote"]]
            enc = tok([cand["prefijo"] + t for t in textos_np[b]], padding=True,
                      truncation=True, max_length=h["max_len"], return_tensors="pt")
            opt.zero_grad()
            logits = modelo(enc["input_ids"], enc["attention_mask"])
            perdida(logits, torch.tensor(y_num[b])).backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), h["clip_gradiente"])
            opt.step()
            sched.step()

        pv = afinado.probas_de(modelo, tok, list(textos_np[va]), prefijo=cand["prefijo"],
                               max_len=h["max_len"])
        f1v = P.f1_macro(y[va], np.asarray(CLASES)[pv.argmax(1)], CLASES)
        historia.append(round(f1v, 4))
        if f1v > mejor_f1 + 1e-9:
            mejor_f1, mejor_epoca, sin_mejora = f1v, epoca, 0
            mejor = copy.deepcopy(modelo.state_dict())
        else:
            sin_mejora += 1
            if sin_mejora >= h["paciencia"]:
                break
    modelo.load_state_dict(mejor)
    return modelo, {"f1_val": round(mejor_f1, 4), "epoca_mejor": mejor_epoca,
                    "epocas_corridas": len(historia), "f1_val_por_epoca": historia}


def _registro_afinado(cand: dict[str, str], tok: Any, textos: list[str], y: np.ndarray,
                      grupos: np.ndarray, part: dict[str, Any],
                      lrs: list[float], epocas_max: int | None) -> dict[str, Any]:
    t0 = time.perf_counter()
    tr, te = np.asarray(part["entrenamiento"]), np.asarray(part["prueba"])
    tr2, va = P.dividir_validacion(
        tr, y, grupos, P.semilla_validacion(part["semilla"], part["particion"]))
    textos_np = np.asarray(textos, dtype=object)

    candidatos: list[dict[str, Any]] = []
    for lr in lrs:
        modelo, info = entrenar_una(cand, tok, textos, y, tr2, va, lr, part["semilla"],
                                    epocas_max)
        pv = afinado.probas_de(modelo, tok, list(textos_np[va]), prefijo=cand["prefijo"],
                               max_len=HIPER["max_len"])
        pt = afinado.probas_de(modelo, tok, list(textos_np[te]), prefijo=cand["prefijo"],
                               max_len=HIPER["max_len"])
        candidatos.append({"lr": lr, "info": info, "pv": pv, "pt": pt})
        del modelo

    # Eleccion solo con la validacion; ante empate, el lr menor (va primero).
    elegido = max(candidatos, key=lambda c: (c["info"]["f1_val"], -c["lr"]))
    ok_val = np.asarray(CLASES)[elegido["pv"].argmax(1)] == y[va]
    umbral = P.umbral_por_cobertura_exactitud(elegido["pv"].max(1), ok_val)
    pt = elegido["pt"]
    return {
        "id": part["id"], "repeticion": part["repeticion"], "semilla": part["semilla"],
        "particion": part["particion"], "prueba": te.tolist(),
        "pred": np.asarray(CLASES)[pt.argmax(1)].tolist(),
        "confianza": [round(float(v), 5) for v in pt.max(1)],
        "lr": elegido["lr"], "epoca_mejor": elegido["info"]["epoca_mejor"],
        "f1_val": elegido["info"]["f1_val"],
        "f1_val_por_lr": {str(c["lr"]): c["info"]["f1_val"] for c in candidatos},
        "n_entrenamiento": int(len(tr2)), "n_validacion": int(len(va)),
        "umbral": {k: v for k, v in umbral.items() if k != "curva"},
        "segundos": round(time.perf_counter() - t0, 1),
    }


def _guardar_oof(nombre: str, tipo: str, registros: list[dict[str, Any]],
                 destino: Path, extra: dict[str, Any] | None = None) -> None:
    destino.mkdir(parents=True, exist_ok=True)
    (destino / f"oof_{nombre}.json").write_text(json.dumps({
        "modelo": nombre, "tipo": tipo, "clases": CLASES, "commit": commit_actual(),
        "arbol_limpio": not _git("status", "--porcelain", "--", *ARCHIVOS_DE_CODIGO),
        **(extra or {}), "particiones": registros,
    }, ensure_ascii=False), encoding="utf-8")


def _cargar_oof(nombre: str, destino: Path) -> list[dict[str, Any]]:
    ruta = destino / f"oof_{nombre}.json"
    return json.loads(ruta.read_text("utf-8"))["particiones"] if ruta.exists() else []


def correr_cv(nombre: str, humo: bool = False) -> None:
    from transformers import AutoTokenizer

    if not humo:
        _exigir_reproducibilidad()
    cand = CANDIDATOS[nombre]
    textos, y, grupos = _cargar_dataset()
    partes = _particiones(y, grupos)
    destino = (RAIZ / "tmp" / "r2_humo") if humo else DIR_R2
    lrs = HIPER["lrs_encoder"][:1] if humo else HIPER["lrs_encoder"]
    epocas = 2 if humo else None
    if humo:
        partes = partes[:1]

    tok = AutoTokenizer.from_pretrained(cand["repo"], revision=cand["revision"])
    hechos = {} if humo else {r["id"]: r for r in _cargar_oof(nombre, destino)}
    for i, part in enumerate(partes, 1):
        if part["id"] in hechos:
            print(f"[{nombre}] {part['id']} ya calculada, se omite")
            continue
        reg = _registro_afinado(cand, tok, textos, y, grupos, part, lrs, epocas)
        hechos[part["id"]] = reg
        _guardar_oof(nombre, "afinado", list(hechos.values()), destino,
                     {"candidato": cand})
        print(f"[{nombre}] {i}/{len(partes)} {part['id']} lr={reg['lr']} "
              f"epoca={reg['epoca_mejor']} {reg['segundos']} s")


def correr_baselines() -> None:
    _exigir_reproducibilidad()
    textos, y, grupos = _cargar_dataset()
    partes = _particiones(y, grupos)
    destino = DIR_R2
    cand = CANDIDATOS["e5_small"]

    from core.llm import gemini
    if gemini.modo_degradado():
        sys.exit("B0 y B2 necesitan GOOGLE_API_KEY.")
    X_gemini = gemini.embed_clasificacion(textos)

    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cand["repo"], revision=cand["revision"])
    enc = AutoModel.from_pretrained(cand["repo"], revision=cand["revision"],
                                    use_safetensors=True, trust_remote_code=False)
    X_congelado = afinado.embeddings_de(
        afinado.clase_clasificador()(enc, len(CLASES)), tok, textos,
        prefijo=cand["prefijo"], max_len=HIPER["max_len"])

    fabricas = {
        "B0": lambda s: baselines.fabrica_b0(X_gemini, s),
        "B1": lambda s: baselines.fabrica_b1(textos, s),
        "B3": lambda s: baselines.fabrica_b3(X_congelado, s),
    }
    for nombre, f in fabricas.items():
        regs = [baselines.registro_particion(f(p["semilla"]), p, y, grupos, CLASES)
                for p in partes]
        _guardar_oof(nombre, "baseline", regs, destino)
        print(f"{nombre}: {len(regs)} particiones")

    b2 = baselines.predicciones_b2(textos)
    regs = []
    for p in partes:
        te = p["prueba"]
        regs.append({"id": p["id"], "repeticion": p["repeticion"], "semilla": p["semilla"],
                     "particion": p["particion"], "prueba": te,
                     "pred": [b2["pred"][i] for i in te],
                     "confianza": [b2["confianza"][i] for i in te], "umbral": None})
    _guardar_oof("B2", "baseline", regs, destino,
                 {"errores_llm": b2["errores"], "modelo_llm": b2["modelo"]})
    print(f"B2: {b2['errores']} respuestas invalidas de {len(textos)}")


# ---------------------------------------------------------------------------
# Latencia (condicion 2)
# ---------------------------------------------------------------------------
def medir_latencia(origen: str, n: int = 40) -> dict[str, Any]:
    """p50 y p95 por mensaje, un mensaje a la vez, sin cache."""
    import torch

    from core.llm import gemini

    textos, y, _ = _cargar_dataset()
    rng = np.random.default_rng(7)
    muestra = [textos[i] for i in rng.choice(len(textos), size=n, replace=False)]

    # B0: embedding de Gemini SIN cache (llamada directa) + producto punto.
    d = np.load(config.RUTA_MODELOS / "router_intencion.npz", allow_pickle=False)
    t_b0 = []
    for t in muestra:
        t0 = time.perf_counter()
        v = gemini._embed_gemini([t], gemini._TAREA_CLASIFICACION)[0]
        z = d["coef"] @ v + d["intercept"]
        z.argmax()
        t_b0.append((time.perf_counter() - t0) * 1000)

    m = afinado.cargar(origen)
    m.probas(["hola"])  # calentamiento
    t_af = []
    for t in muestra:
        t0 = time.perf_counter()
        m.probas([t])
        t_af.append((time.perf_counter() - t0) * 1000)

    def resumen(v: list[float]) -> dict[str, float]:
        return {"p50_ms": round(statistics.median(v), 1),
                "p95_ms": round(float(np.percentile(v, 95)), 1), "n": len(v)}

    res = {"B0": resumen(t_b0), "afinado": resumen(t_af), "modelo": origen,
           "maquina": platform.platform(), "hilos_torch": torch.get_num_threads(),
           "nota": "Una sola maquina, un mensaje a la vez. La latencia de B0 incluye "
                   "la red hacia Gemini; la del afinado es solo CPU local."}
    DIR_R2.mkdir(parents=True, exist_ok=True)
    (DIR_R2 / "latencia.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    return res


# ---------------------------------------------------------------------------
# Analisis y regla de adopcion
# ---------------------------------------------------------------------------
def _matriz(registros: list[dict[str, Any]], n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(pred[R,n], confianza[R,n], umbral[R,n]) para las particiones de CV."""
    reps = sorted({r["repeticion"] for r in registros if r["repeticion"] >= 0})
    pred = np.full((len(reps), n), "", dtype=object)
    conf = np.zeros((len(reps), n))
    umb = np.full((len(reps), n), np.nan)
    for r in registros:
        if r["repeticion"] < 0:
            continue
        for k, i in enumerate(r["prueba"]):
            pred[r["repeticion"], i] = r["pred"][k]
            conf[r["repeticion"], i] = r["confianza"][k]
            if r.get("umbral"):
                umb[r["repeticion"], i] = r["umbral"]["umbral"]
    assert (pred != "").all(), "alguna muestra no tiene prediccion fuera de muestra"
    return pred, conf, umb


def _cascada(pred_clf: np.ndarray, conf: np.ndarray, umbral: Any,
             pred_llm: np.ndarray) -> tuple[np.ndarray, float]:
    """Nivel 1 si conf >= umbral; si no, el LLM (o el nivel 1 si el LLM fallo)."""
    usa_clf = (conf >= umbral) | (pred_llm == "__error__")
    return np.where(usa_clf, pred_clf, pred_llm), float((conf >= umbral).mean())


def analizar() -> dict[str, Any]:
    textos, y, grupos = _cargar_dataset()
    n = len(y)
    protocolo = json.loads(RUTA_PROTOCOLO.read_text("utf-8"))
    nombres = ["B0", "B1", "B2", "B3"] + [c for c in CANDIDATOS if (DIR_R2 / f"oof_{c}.json").exists()]
    regs = {m: _cargar_oof(m, DIR_R2) for m in nombres}
    matrices = {m: _matriz(regs[m], n) for m in nombres}
    pred_b2 = matrices["B2"][0]

    resultados: dict[str, Any] = {}
    for m in nombres:
        pred, conf, umb = matrices[m]
        f1s = [P.f1_macro(y, pred[r], CLASES) for r in range(pred.shape[0])]
        accs = [P.exactitud(y, pred[r]) for r in range(pred.shape[0])]
        porc = {c: round(float(np.mean([
            f1_score(y, pred[r], labels=[c], average="macro", zero_division=0)
            for r in range(pred.shape[0])])), 4) for c in CLASES}
        fijo = next((r for r in regs[m] if r["id"] == "fijo"), None)
        fijo_m = None
        if fijo:
            yt = y[fijo["prueba"]]
            fijo_m = {"f1_macro": round(P.f1_macro(yt, fijo["pred"], CLASES), 4),
                      "exactitud": round(P.exactitud(yt, fijo["pred"]), 4),
                      "n_prueba": len(yt)}
        entrada: dict[str, Any] = {
            "f1_macro_media": round(float(np.mean(f1s)), 4),
            "f1_macro_por_repeticion": [round(v, 4) for v in f1s],
            "f1_macro_sd_entre_repeticiones": round(float(np.std(f1s, ddof=1)), 4),
            "exactitud_media": round(float(np.mean(accs)), 4),
            "f1_por_clase": porc,
            "split_fijo_semilla7": fijo_m,
        }
        propios = [r for r in regs[m] if r.get("umbral") and r["repeticion"] >= 0]
        if propios:
            us = [r["umbral"]["umbral"] for r in propios]
            entrada["umbral"] = {
                "mediana": round(float(np.median(us)), 4),
                "min": round(min(us), 4), "max": round(max(us), 4),
                "particiones_que_cumplen": sum(r["umbral"]["cumple"] for r in propios),
                "de": len(propios)}
            casc, cob = _cascada(pred, conf, umb, pred_b2)
            entrada["cascada_umbral_por_particion"] = {
                "f1_macro": round(float(np.mean([P.f1_macro(y, casc[r], CLASES)
                                                 for r in range(casc.shape[0])])), 4),
                "resuelto_en_nivel_1": round(cob, 4)}
        if m == "B0":
            casc, cob = _cascada(pred, conf, baselines.UMBRAL_PRODUCCION_B0, pred_b2)
            entrada["cascada_umbral_produccion_0.45"] = {
                "f1_macro": round(float(np.mean([P.f1_macro(y, casc[r], CLASES)
                                                 for r in range(casc.shape[0])])), 4),
                "resuelto_en_nivel_1": round(cob, 4)}
        if "lr" in (regs[m][0] if regs[m] else {}):
            lrs = [r["lr"] for r in regs[m] if r["repeticion"] >= 0]
            entrada["lr_elegidos"] = {str(v): lrs.count(v) for v in sorted(set(lrs))}
            entrada["epoca_mejor_mediana"] = float(np.median([r["epoca_mejor"] for r in regs[m]]))
        resultados[m] = entrada

    comparaciones = {}
    pred_b0 = matrices["B0"][0]
    for m in nombres:
        if m == "B0":
            continue
        comparaciones[f"{m}_menos_B0"] = P.bootstrap_diferencia_f1(
            y, matrices[m][0], pred_b0, grupos, CLASES,
            replicas=protocolo["bootstrap"]["replicas"], semilla=protocolo["bootstrap"]["semilla"])

    latencia = json.loads((DIR_R2 / "latencia.json").read_text("utf-8")) \
        if (DIR_R2 / "latencia.json").exists() else None

    decision = _aplicar_regla(resultados, comparaciones, latencia)
    metricas = {
        "protocolo": {"sha256_dataset": protocolo["dataset"]["sha256"],
                      "commit_padre_protocolo": protocolo["commit_padre"],
                      "sha256_particiones": protocolo["validacion_cruzada"]["sha256_particiones"]},
        "commits_de_resultados": {m: json.loads((DIR_R2 / f"oof_{m}.json").read_text("utf-8"))["commit"]
                                  for m in nombres},
        "versiones": _versiones(),
        "n_ejemplos": n, "n_repeticiones": len(P.SEMILLAS), "particiones": P.PARTICIONES,
        "por_modelo": resultados,
        "diferencias_frente_a_B0": comparaciones,
        "latencia": latencia,
        "decision": decision,
        "nota_metodologica": (
            "Fine-tuning de un encoder multilingue (todos los pesos) con cabeza lineal. "
            "No es fine-tuning de un LLM generativo. Datos en su mayoria sinteticos."),
    }
    RUTA_METRICAS.write_text(json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Metricas -> {RUTA_METRICAS.relative_to(RAIZ)}")
    print(json.dumps({"decision": decision,
                      "f1": {m: v["f1_macro_media"] for m, v in resultados.items()},
                      "diferencias": comparaciones}, ensure_ascii=False, indent=2))
    return metricas


def _aplicar_regla(resultados: dict, comparaciones: dict, latencia: dict | None) -> dict[str, Any]:
    p = REGLA_ADOPCION["candidato_principal"]
    if p not in resultados:
        return {"adoptar": None, "motivo": f"falta el candidato principal {p}"}
    c1 = comparaciones[f"{p}_menos_B0"]
    cond1 = c1["ic95_inferior"] >= -0.02
    cond2 = None if latencia is None else latencia["afinado"]["p50_ms"] < latencia["B0"]["p50_ms"]
    cond3 = resultados[p].get("umbral") is not None
    ok = [cond1, cond2, cond3]
    return {
        "candidato": p,
        "condicion_1_no_inferioridad": {"cumple": cond1, "ic95_inferior": c1["ic95_inferior"],
                                        "requerido": ">= -0.02"},
        "condicion_2_latencia": {"cumple": cond2,
                                 "afinado_p50_ms": latencia and latencia["afinado"]["p50_ms"],
                                 "B0_p50_ms": latencia and latencia["B0"]["p50_ms"]},
        "condicion_3_umbral_calibrado": {"cumple": cond3},
        "adoptar": None if None in ok else all(ok),
        "efecto": "VM_ROUTER=afinado como predeterminado" if all(x is True for x in ok)
                  else "queda disponible como alternativa; el valor predeterminado sigue en 'embeddings'",
    }


# ---------------------------------------------------------------------------
# Modelo final
# ---------------------------------------------------------------------------
def entrenar_final(nombre: str, destino: Path = RUTA_ARTEFACTO) -> Path:
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    _exigir_reproducibilidad()
    cand = CANDIDATOS[nombre]
    regs = [r for r in _cargar_oof(nombre, DIR_R2) if r["repeticion"] >= 0]
    if len(regs) != P.PARTICIONES * len(P.SEMILLAS):
        sys.exit("Faltan particiones de CV: el umbral y el lr salen de ellas.")
    lrs = [r["lr"] for r in regs]
    lr = sorted(set(lrs), key=lambda v: (-lrs.count(v), v))[0]
    umbral = float(np.median([r["umbral"]["umbral"] for r in regs]))

    textos, y, grupos = _cargar_dataset()
    todos = np.arange(len(y))
    semilla = P.SEMILLA_SPLIT_FIJO
    tr, va = P.dividir_validacion(todos, y, grupos, P.semilla_validacion(semilla, 99))
    tok = AutoTokenizer.from_pretrained(cand["repo"], revision=cand["revision"])
    modelo, info = entrenar_una(cand, tok, textos, y, tr, va, lr, semilla)

    destino.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in modelo.state_dict().items()},
              str(destino / "model.safetensors"))
    modelo.encoder.config.save_pretrained(destino)
    tok.save_pretrained(destino)
    meta = {
        "clases": CLASES, "modelo_base": cand["repo"], "revision_base": cand["revision"],
        "prefijo": cand["prefijo"], "max_len": HIPER["max_len"], "pooling": "promedio de tokens",
        "umbral_confianza": round(umbral, 4), "lr_encoder": lr,
        "semilla": semilla, "epoca_mejor": info["epoca_mejor"], "f1_val": info["f1_val"],
        "n_entrenamiento": int(len(tr)), "n_validacion": int(len(va)),
        "sha256_dataset": P.sha256_archivo(datos.RUTA_DATASET), "commit": commit_actual(),
        "versiones": _versiones(),
    }
    (destino / "etiquetas.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                            encoding="utf-8")

    # Prueba de ida y vuelta: el artefacto cargado debe dar lo mismo que el modelo en memoria.
    cargado = afinado.cargar(str(destino))
    muestra = [textos[i] for i in range(0, len(textos), 9)]
    a = afinado.probas_de(modelo, tok, muestra, prefijo=cand["prefijo"], max_len=HIPER["max_len"])
    b = cargado.probas(muestra)
    assert np.allclose(a, b, atol=1e-5), "el artefacto guardado no reproduce el modelo"
    print(f"Artefacto -> {destino} (lr={lr}, umbral={umbral:.3f}, ida y vuelta OK)")
    return destino


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="orden", required=True)
    sub.add_parser("congelar")
    sub.add_parser("baselines")
    c = sub.add_parser("cv")
    c.add_argument("candidato", choices=sorted(CANDIDATOS))
    c.add_argument("--humo", action="store_true", help="1 particion, 2 epocas, sin escribir en r2/")
    la = sub.add_parser("latencia")
    la.add_argument("--modelo", default=str(RUTA_ARTEFACTO))
    sub.add_parser("analizar")
    f = sub.add_parser("final")
    f.add_argument("candidato", choices=sorted(CANDIDATOS))
    a = ap.parse_args()

    if a.orden == "congelar":
        congelar()
    elif a.orden == "baselines":
        correr_baselines()
    elif a.orden == "cv":
        correr_cv(a.candidato, a.humo)
    elif a.orden == "latencia":
        medir_latencia(a.modelo)
    elif a.orden == "analizar":
        analizar()
    elif a.orden == "final":
        entrenar_final(a.candidato)


if __name__ == "__main__":
    main()
