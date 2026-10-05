"""Prueba de carga corta contra una URL (build-spec 7: T3 y T11).

    python scripts/carga.py --url http://127.0.0.1:8080 --usuarios 20

Lanza `--usuarios` turnos simultaneos, cada uno en su propia sesion, y mide lo
que importa para un servicio publico con un LLM detras: que lo que se admite
termine bien, que lo que sobra reciba un 503 limpio con Retry-After (y no un
error de servidor ni un cuelgue), y que el servicio siga sano despues.

Cada turno cuesta una llamada a Gemini: usar pocos usuarios contra el servicio
publicado. Solo usa la biblioteca estandar.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter

MENSAJES = [
    "Cuanto cuesta ir a Playa Blanca el 12 de octubre para 6 personas?",
    "Puedo llevar a mi perro?",
    "Cuanto me devuelven si cancelo con dos dias de anticipacion?",
    "Somos 8 amigos y queremos algo con musica",
    "Tienen lancha para 12 personas el sabado para Cholon?",
]


def llamar(base: str, ruta: str, cuerpo: dict | None = None, timeout: float = 150):
    datos = json.dumps(cuerpo).encode() if cuerpo is not None else (b"" if ruta == "/api/sesion" else None)
    req = urllib.request.Request(base + ruta, data=datos,
                                 method="POST" if datos is not None else "GET",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            crudo = r.read()
            return r.status, (json.loads(crudo) if crudo else None), dict(r.headers)
    except urllib.error.HTTPError as e:
        crudo = e.read()
        try:
            return e.code, json.loads(crudo), dict(e.headers)
        except json.JSONDecodeError:
            return e.code, {"crudo": crudo[:120].decode("utf-8", "replace")}, dict(e.headers)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--usuarios", type=int, default=20)
    ap.add_argument("--admitidos-max", type=int, default=0,
                    help="si se indica, se exige que no se admitan mas de estos turnos a la vez (concurrencia + cola)")
    a = ap.parse_args()
    base = a.url.rstrip("/")

    estado, salud, _ = llamar(base, "/api/salud")
    if estado != 200 or salud.get("estado") != "listo":
        print(f"El servicio no esta listo: {estado} {salud}")
        return 1

    sesiones = []
    for _ in range(a.usuarios):
        e, s, _ = llamar(base, "/api/sesion")
        if e != 200:
            print(f"No se pudo crear una sesion ({e}): {s}")
            return 1
        sesiones.append(s["session_id"])

    resultados: list[tuple[int, float, dict, dict]] = [None] * a.usuarios  # type: ignore[list-item]
    arranque = threading.Barrier(a.usuarios)

    def usuario(i: int) -> None:
        arranque.wait()
        t = time.time()
        e, cuerpo, cab = llamar(base, "/api/turno", {
            "session_id": sesiones[i], "texto": MENSAJES[i % len(MENSAJES)], "request_id": str(uuid.uuid4())})
        resultados[i] = (e, time.time() - t, cuerpo or {}, cab)

    hilos = [threading.Thread(target=usuario, args=(i,)) for i in range(a.usuarios)]
    t0 = time.time()
    for h in hilos:
        h.start()
    for h in hilos:
        h.join()
    total = time.time() - t0

    por_estado = Counter(r[0] for r in resultados)
    ok = [r for r in resultados if r[0] == 200]
    rechazados = [r for r in resultados if r[0] == 503]
    otros = [r for r in resultados if r[0] not in (200, 503)]
    print(f"{a.usuarios} turnos simultaneos en {total:.1f} s: {dict(sorted(por_estado.items()))}")
    if ok:
        lat = sorted(r[1] for r in ok)
        print(f"  200: p50 {statistics.median(lat):.1f} s, max {lat[-1]:.1f} s")
    fallos: list[str] = []
    if otros:
        fallos.append(f"respuestas distintas de 200 y 503: {[(r[0], r[2]) for r in otros][:3]}")
    if rechazados:
        sin_cabecera = [r for r in rechazados if "Retry-After" not in r[3] and "retry-after" not in r[3]]
        sin_codigo = [r for r in rechazados if (r[2].get("error") or {}).get("codigo") != "saturado"]
        print(f"  503: {len(rechazados)} (con Retry-After: {len(rechazados) - len(sin_cabecera)}, "
              f"codigo 'saturado': {len(rechazados) - len(sin_codigo)})")
        if sin_cabecera:
            fallos.append("hay 503 sin Retry-After")
        if sin_codigo:
            fallos.append(f"hay 503 sin el codigo 'saturado': {[r[2] for r in sin_codigo][:2]}")
    degradados = [r for r in ok if (r[2].get("contrato") or {}).get("metadatos", {}).get("degradado")]
    if degradados:
        fallos.append(f"{len(degradados)} respuestas en modo degradado")
    if a.admitidos_max and len(ok) > a.admitidos_max:
        fallos.append(f"se admitieron {len(ok)} turnos, mas que el maximo esperado ({a.admitidos_max})")
    if not ok:
        fallos.append("ningun turno termino bien")

    time.sleep(1)
    estado, salud, _ = llamar(base, "/api/salud")
    print(f"  salud despues: {estado} {salud.get('estado') if isinstance(salud, dict) else salud}")
    if estado != 200 or salud.get("estado") != "listo":
        fallos.append("el servicio no quedo sano")
    # Se reutiliza una sesion: crear una mas toparia con el limite de sesiones por IP y por hora.
    e2, r, _ = llamar(base, "/api/turno", {"session_id": sesiones[-1], "texto": "Hola", "request_id": str(uuid.uuid4())})
    print(f"  un turno nuevo despues de la rafaga: HTTP {e2}")
    if e2 != 200:
        fallos.append("tras la rafaga el servicio no atiende un turno nuevo")

    for f in fallos:
        print("  FALLO:", f)
    print("OK" if not fallos else "FALLO")
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
