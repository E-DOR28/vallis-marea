"""Pipeline de ingesta: corpus -> fragmentos -> embeddings -> almacenes.

Se ejecuta con:  python -m core.ingesta.indexar [--recrear]
"""

from __future__ import annotations

import argparse
import sys
import time

from core import config
from core.almacen import estructurado, lexico, vectorial
from core.ingesta import chunking
from core.llm import gemini


def indexar_todo(recrear: bool = True, verbose: bool = True) -> dict:
    """Reconstruye indices lexico y vectorial, y carga los datos estructurados."""
    t0 = time.time()
    log = print if verbose else (lambda *a, **k: None)

    log("1/4  Troceando el corpus...")
    fragmentos = chunking.trocear_corpus()
    if not fragmentos:
        return {"ok": False, "error": f"No hay .md en {config.RUTA_CORPUS}"}
    fuentes = sorted({f["fuente"] for f in fragmentos})
    log(f"     {len(fragmentos)} fragmentos de {len(fuentes)} documentos: {', '.join(fuentes)}")

    log("2/4  Calculando embeddings...")
    if gemini.modo_degradado():
        log("     AVISO: sin GOOGLE_API_KEY -> embeddings locales (modo degradado)")
    vectores = gemini.embed_documentos([f["texto"] for f in fragmentos])
    log(f"     matriz {vectores.shape} con modelo "
        f"{'local-degradado' if gemini.modo_degradado() else config.MODELO_EMBEDDING}")

    log("3/4  Indexando (vectorial + lexico)...")
    vectorial.inicializar(recrear=recrear)
    n_vec = vectorial.indexar(fragmentos, vectores)
    lexico.inicializar(recrear=recrear)
    n_lex = lexico.indexar(fragmentos)
    log(f"     vectorial: {n_vec} | lexico: {n_lex}")

    log("4/4  Cargando datos estructurados (flota, rutas, reservas)...")
    conteos = estructurado.inicializar(recrear=recrear)
    log(f"     {conteos}")

    return {
        "ok": True,
        "fragmentos": len(fragmentos),
        "documentos": len(fuentes),
        "dimension": int(vectores.shape[1]),
        "degradado": gemini.modo_degradado(),
        "estructurado": conteos,
        "segundos": round(time.time() - t0, 2),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Indexa el corpus de Vallis Marea.")
    p.add_argument("--recrear", action="store_true",
                   help="Borra y reconstruye los indices desde cero.")
    args = p.parse_args()

    r = indexar_todo(recrear=args.recrear or True)
    if not r.get("ok"):
        print(f"\nERROR: {r.get('error')}", file=sys.stderr)
        return 1
    print(f"\nListo en {r['segundos']}s. "
          f"{r['fragmentos']} fragmentos, dimension {r['dimension']}"
          f"{'  [MODO DEGRADADO]' if r['degradado'] else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
