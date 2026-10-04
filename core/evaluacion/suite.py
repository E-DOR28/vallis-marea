"""Suite de evaluacion del sistema.

Tres niveles, siguiendo la estrategia declarada en el plan de trabajo:

  Nivel 1 - Recuperacion:  recall@k, MRR sobre el golden set.
  Nivel 2 - Generacion:    tasa de citacion correcta y de abstencion correcta.
  Nivel 3 - Sistema:       monolito vs. ecosistema (latencia y numero de saltos).

El nivel 3 es el que sostiene la tesis del proyecto. Se ejecuta el MISMO
conjunto de preguntas por dos caminos:

  - "monolito": llamada directa a la funcion del agente, en proceso. Es como
    operaba el nodo unico de n8n.
  - "ecosistema": la misma capacidad, alcanzada por A2A sobre HTTP/JSON-RPC.

La diferencia de latencia entre ambos es, literalmente, el precio de la
interoperabilidad. Reportarlo con honestidad -- incluso cuando A2A pierde --
es lo que separa una evaluacion de un folleto.

Uso:  python -m core.evaluacion.suite
"""

from __future__ import annotations

import json
import statistics
import time
from typing import Any

from core import config
from core.a2a import cliente as a2a
from core.a2a import lanzador
from core.agentes.conocimiento import rag
from core.llm import gemini

RUTA_GOLDEN = config.RUTA_EVALUACION / "golden_set.json"
RUTA_RESULTADOS = config.RUTA_EVALUACION / "resultados.json"


def cargar_golden() -> list[dict[str, Any]]:
    return json.loads(RUTA_GOLDEN.read_text(encoding="utf-8"))["items"]


# ---------------------------------------------------------------------------
# Nivel 1: recuperacion
# ---------------------------------------------------------------------------
def evaluar_recuperacion(items: list[dict[str, Any]], k: int = 4) -> dict[str, Any]:
    """recall@k y MRR sobre los items que si tienen fuente esperada."""
    con_fuente = [i for i in items if i.get("fuente_esperada")]
    aciertos_fuente = 0
    aciertos_seccion = 0
    rangos_reciprocos: list[float] = []
    detalle: list[dict[str, Any]] = []

    for item in con_fuente:
        fragmentos, _ = rag.recuperar(item["pregunta"], k_final=k)
        fuentes = [f.fuente for f in fragmentos]
        secciones = [(f.fuente, f.seccion) for f in fragmentos]

        hit_fuente = item["fuente_esperada"] in fuentes
        hit_seccion = (item["fuente_esperada"], item["seccion_esperada"]) in secciones
        aciertos_fuente += hit_fuente
        aciertos_seccion += hit_seccion

        rr = 0.0
        for pos, (fu, se) in enumerate(secciones, start=1):
            if fu == item["fuente_esperada"] and se == item["seccion_esperada"]:
                rr = 1.0 / pos
                break
        rangos_reciprocos.append(rr)

        detalle.append({
            "id": item["id"],
            "pregunta": item["pregunta"],
            "acierto_fuente": hit_fuente,
            "acierto_seccion": hit_seccion,
            "rango_reciproco": round(rr, 3),
            "recuperado": [f"{f.fuente} > {f.seccion}" for f in fragmentos[:k]],
        })

    n = len(con_fuente) or 1
    return {
        "n_items": len(con_fuente),
        "k": k,
        f"recall_fuente@{k}": round(aciertos_fuente / n, 4),
        f"recall_seccion@{k}": round(aciertos_seccion / n, 4),
        "mrr": round(statistics.mean(rangos_reciprocos) if rangos_reciprocos else 0.0, 4),
        "detalle": detalle,
    }


# ---------------------------------------------------------------------------
# Nivel 2: generacion, citacion y abstencion
# ---------------------------------------------------------------------------
def evaluar_generacion(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Mide si cita la fuente correcta y si se abstiene cuando debe."""
    debe_responder = [i for i in items if i.get("fuente_esperada")]
    debe_abstenerse = [i for i in items if i.get("tipo") == "abstencion"]

    cito_bien = 0
    respondio = 0
    detalle: list[dict[str, Any]] = []

    for item in debe_responder:
        r = rag.responder(item["pregunta"])
        fuentes_citadas = [c["fuente"] for c in r.citas]
        ok_cita = item["fuente_esperada"] in fuentes_citadas
        cito_bien += ok_cita
        respondio += not r.abstencion
        detalle.append({
            "id": item["id"],
            "abstuvo": r.abstencion,
            "cito_fuente_correcta": ok_cita,
            "citas": fuentes_citadas,
            "confianza": round(r.confianza, 3),
            "respuesta": r.respuesta[:180],
        })

    abstuvo_bien = 0
    detalle_abstencion: list[dict[str, Any]] = []
    for item in debe_abstenerse:
        r = rag.responder(item["pregunta"])
        abstuvo_bien += r.abstencion
        detalle_abstencion.append({
            "id": item["id"],
            "pregunta": item["pregunta"],
            "abstuvo": r.abstencion,
            "motivo": r.motivo_abstencion,
            "respuesta_si_no_abstuvo": "" if r.abstencion else r.respuesta[:180],
        })

    n_resp = len(debe_responder) or 1
    n_abst = len(debe_abstenerse) or 1
    return {
        "n_debe_responder": len(debe_responder),
        "n_debe_abstenerse": len(debe_abstenerse),
        "tasa_respuesta": round(respondio / n_resp, 4),
        "precision_citacion": round(cito_bien / n_resp, 4),
        "tasa_abstencion_correcta": round(abstuvo_bien / n_abst, 4),
        "detalle": detalle,
        "detalle_abstencion": detalle_abstencion,
    }


# ---------------------------------------------------------------------------
# Nivel 3: monolito vs. ecosistema
# ---------------------------------------------------------------------------
def comparar_arquitecturas(
    items: list[dict[str, Any]], n: int = 10, repeticiones: int = 2
) -> dict[str, Any]:
    """Misma capacidad por llamada directa y por A2A. Mide el sobrecosto."""
    muestra = [i for i in items if i.get("fuente_esperada")][:n]

    # El descubrimiento se paga una vez al arrancar, no por mensaje. Se saca de
    # la medicion de latencia por turno y se reporta como costo de arranque.
    ms_descubrimiento = a2a.precalentar()

    ms_monolito: list[float] = []
    ms_ecosistema: list[float] = []

    for _ in range(repeticiones):
        for item in muestra:
            t0 = time.perf_counter()
            rag.recuperar(item["pregunta"])
            ms_monolito.append((time.perf_counter() - t0) * 1000)

            salto = a2a.invocar(
                "conocimiento", "obtener_fragmentos",
                {"consulta": item["pregunta"], "k": config.TOP_K_FINAL},
            )
            ms_ecosistema.append(salto.ms)

    def resumen(xs: list[float]) -> dict[str, float]:
        xs_ord = sorted(xs)
        return {
            "p50": round(statistics.median(xs_ord), 1),
            "p95": round(xs_ord[max(int(len(xs_ord) * 0.95) - 1, 0)], 1),
            "media": round(statistics.mean(xs_ord), 1),
            "min": round(xs_ord[0], 1),
            "max": round(xs_ord[-1], 1),
        }

    r_mono = resumen(ms_monolito)
    r_eco = resumen(ms_ecosistema)
    return {
        "n_consultas": len(ms_monolito),
        "monolito_llamada_directa_ms": r_mono,
        "ecosistema_a2a_ms": r_eco,
        "descubrimiento_agent_card_ms": ms_descubrimiento,
        "nota_descubrimiento": (
            "Costo de arranque: leer el Agent Card y abrir la conexion, una vez "
            "por agente y por sesion. No se paga en cada mensaje."
        ),
        "sobrecosto_p50_ms": round(r_eco["p50"] - r_mono["p50"], 1),
        "sobrecosto_p50_pct": (
            round((r_eco["p50"] / r_mono["p50"] - 1) * 100, 1) if r_mono["p50"] else None
        ),
        "lectura": (
            "El sobrecosto es el precio de serializar, cruzar HTTP/JSON-RPC y "
            "deserializar en cada delegacion. A cambio se obtiene desacoplamiento, "
            "reutilizacion por clientes externos y trazabilidad por salto."
        ),
    }


# ---------------------------------------------------------------------------
def ejecutar(verbose: bool = True) -> dict[str, Any]:
    log = print if verbose else (lambda *a, **k: None)
    items = cargar_golden()

    log("Levantando agentes A2A...")
    estado = lanzador.levantar_en_hilos()
    if not estado["ok"]:
        log(f"  AVISO: no responden {estado['no_responden']}; se omite el nivel 3.")

    log(f"\nNivel 1 - Recuperacion ({len(items)} items en el golden set)")
    n1 = evaluar_recuperacion(items)
    k = n1["k"]
    log(f"  recall_fuente@{k}:   {n1[f'recall_fuente@{k}']}")
    log(f"  recall_seccion@{k}:  {n1[f'recall_seccion@{k}']}")
    log(f"  MRR:                {n1['mrr']}")

    log("\nNivel 2 - Generacion, citacion y abstencion")
    n2 = evaluar_generacion(items)
    log(f"  precision de citacion:      {n2['precision_citacion']}")
    log(f"  tasa de abstencion correcta: {n2['tasa_abstencion_correcta']}")

    n3: dict[str, Any] = {}
    if estado["ok"]:
        log("\nNivel 3 - Monolito vs. ecosistema")
        n3 = comparar_arquitecturas(items)
        log(f"  monolito  p50: {n3['monolito_llamada_directa_ms']['p50']} ms")
        log(f"  ecosistema p50: {n3['ecosistema_a2a_ms']['p50']} ms")
        log(f"  sobrecosto:     +{n3['sobrecosto_p50_ms']} ms "
            f"({n3['sobrecosto_p50_pct']}%)")

    metricas_router = {}
    ruta_router = config.RUTA_EVALUACION / "metricas_router.json"
    if ruta_router.exists():
        metricas_router = json.loads(ruta_router.read_text(encoding="utf-8"))

    resultados = {
        "fecha": time.strftime("%Y-%m-%d %H:%M"),
        "modo_degradado": gemini.modo_degradado(),
        "advertencia": (
            "EJECUTADO EN MODO DEGRADADO (sin GOOGLE_API_KEY): los embeddings son "
            "locales por hashing y no hay generacion. Las metricas NO son "
            "representativas del sistema real."
            if gemini.modo_degradado() else
            f"Ejecutado con Gemini ({config.MODELO_GENERACION} / "
            f"{config.MODELO_EMBEDDING})."
        ),
        "configuracion": {
            "modelo_generacion": config.MODELO_GENERACION,
            "modelo_embedding": config.MODELO_EMBEDDING,
            "dim_embedding": config.DIM_EMBEDDING,
            "top_k_denso": config.TOP_K_DENSO,
            "top_k_lexico": config.TOP_K_LEXICO,
            "top_k_final": config.TOP_K_FINAL,
            "rrf_k": config.RRF_K,
        },
        "nivel_1_recuperacion": n1,
        "nivel_2_generacion": n2,
        "nivel_3_arquitecturas": n3,
        "router": {k: v for k, v in metricas_router.items()
                   if k in ("exactitud", "f1_macro", "nota_metodologica",
                            "embeddings_degradados", "dataset")},
    }
    RUTA_RESULTADOS.write_text(
        json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"\nResultados -> {RUTA_RESULTADOS}")
    return resultados


if __name__ == "__main__":
    ejecutar()
