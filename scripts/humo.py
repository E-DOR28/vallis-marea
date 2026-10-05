"""Prueba de humo del servicio web: los 6 ejemplos del chat contra una URL.

    python scripts/humo.py                          # http://127.0.0.1:8080
    python scripts/humo.py --url https://mi-servicio.run.app

Solo usa la biblioteca estandar y habla con la API por HTTP: sirve igual contra
un servidor local, un contenedor o el servicio publicado. No escribe en la
memoria episodica (el servicio web la tiene apagada) ni toca archivos locales.
Sale con codigo 1 si falla alguna comprobacion.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid

TIPOS = {"texto", "texto_con_opciones", "escalamiento"}

# (mensaje, habilidad que debe aparecer entre los saltos A2A, exige citas)
CASOS = [
    ("Hola, buenas tardes", None, False),
    ("Tienen lancha para 12 personas el sabado para Cholon?", "disponibilidad.consultar_disponibilidad", False),
    ("Cuanto me devuelven si cancelo con dos dias de anticipacion?", "conocimiento.buscar_conocimiento", True),
    ("Somos 8 amigos y queremos algo con musica para celebrar", "recomendador.recomendar", False),
    ("Puedo llevar a mi perro?", "conocimiento.buscar_conocimiento", None),
    ("Cuanto cuesta ir a Playa Blanca el 10 de octubre para 6 personas?",
     "disponibilidad.consultar_disponibilidad", False),
]


def llamar(base: str, metodo: str, ruta: str, cuerpo: dict | None = None, timeout: float = 120):
    datos = json.dumps(cuerpo).encode() if cuerpo is not None else (b"" if metodo == "POST" else None)
    req = urllib.request.Request(base + ruta, data=datos, method=metodo,
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
            return e.code, {"crudo": crudo[:200].decode("utf-8", "replace")}, dict(e.headers)


class Comprobaciones:
    def __init__(self) -> None:
        self.fallos: list[str] = []
        self.total = 0

    def que(self, condicion: bool, texto: str) -> None:
        self.total += 1
        print(f"  {'ok ' if condicion else 'FALLA'} {texto}")
        if not condicion:
            self.fallos.append(texto)


def esperar_listo(base: str, espera: float) -> dict:
    """Espera a que el servicio salga de 'iniciando' (arranque en frio)."""
    limite = time.time() + espera
    ultimo: dict = {}
    while time.time() < limite:
        try:
            estado, cuerpo, _ = llamar(base, "GET", "/api/salud", timeout=10)
            if estado == 200:
                ultimo = cuerpo
                if cuerpo.get("estado") != "iniciando":
                    return cuerpo
        except Exception:
            pass
        time.sleep(2)
    return ultimo


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--espera", type=float, default=90, help="segundos para esperar el arranque")
    a = ap.parse_args()
    base = a.url.rstrip("/")
    c = Comprobaciones()

    print(f"Servicio: {base}")
    t0 = time.time()
    salud = esperar_listo(base, a.espera)
    print(f"  salud tras {time.time() - t0:.1f} s: {json.dumps(salud, ensure_ascii=False)}")
    c.que(salud.get("estado") == "listo", "el servicio esta 'listo' (no iniciando ni degradado)")
    c.que(salud.get("gemini") is True, "hay clave de Gemini (sin ella las respuestas serian degradadas)")
    c.que(all(salud.get("agentes", {}).values()) and len(salud.get("agentes", {})) == 3,
          "los tres agentes estan en linea")
    if salud.get("estado") != "listo":
        print("\nNo tiene sentido seguir: el servicio no esta listo.")
        return 1

    estado, sesion, _ = llamar(base, "POST", "/api/sesion")
    c.que(estado == 200 and bool(sesion and sesion.get("session_id")), "se crea una sesion")
    sid = sesion["session_id"]

    for texto, habilidad, exige_citas in CASOS:
        print(f"\n> {texto}")
        rid = str(uuid.uuid4())
        t = time.time()
        estado, r, _ = llamar(base, "POST", "/api/turno", {
            "session_id": sid, "texto": texto, "request_id": rid, "consentimiento_registro": False})
        seg = time.time() - t
        c.que(estado == 200, f"HTTP 200 ({seg:.1f} s)")
        if estado != 200:
            print(f"      {r}")
            continue
        k = r["contrato"]
        c.que(r.get("request_id") == rid, "eco del request_id")
        c.que(k.get("tipo") in TIPOS, f"tipo valido ({k.get('tipo')})")
        c.que(bool(k.get("mensaje", "").strip()) or k.get("escalar_a_humano") is True, "hay mensaje o escalamiento")
        c.que(not k.get("escalar_a_humano") or bool(k.get("motivo_escalamiento")), "un escalamiento trae motivo")
        c.que("[MODO DEGRADADO" not in k.get("mensaje", ""), "la respuesta no es del modo degradado")
        c.que("_trazas" not in k and "_trazas" not in r, "no se filtran trazas internas")
        saltos = [f"{s['agente']}.{s['habilidad']}" for s in r["trazas"]["saltos_a2a"]]
        if habilidad is None:
            c.que(saltos == [], f"saludo sin delegar (saltos={saltos})")
        else:
            c.que(habilidad in saltos, f"salto esperado {habilidad} (saltos={saltos})")
        if exige_citas is True:
            c.que(len(k["citas"]) >= 1 and not k["escalar_a_humano"], f"{len(k['citas'])} cita(s) en una consulta de conocimiento")
        if exige_citas is None:
            c.que(len(k["citas"]) >= 1 or k["escalar_a_humano"], "o cita una fuente o se abstiene y escala")
        print(f"      intencion={r['trazas']['router'].get('intencion')} "
              f"metodo={r['trazas']['router'].get('metodo')} causa={k['metadatos'].get('causa')}")
        print(f"      {k['mensaje'][:140]!r}")

    print("\n> idempotencia: repetir el ultimo request_id")
    estado, otra, _ = llamar(base, "POST", "/api/turno", {
        "session_id": sid, "texto": CASOS[-1][0], "request_id": rid})
    c.que(estado == 200 and otra == r, "la misma respuesta, sin volver a ejecutar")

    print("\n> entrada invalida")
    estado, e, _ = llamar(base, "POST", "/api/turno", {
        "session_id": sid, "texto": "", "request_id": str(uuid.uuid4())})
    c.que(estado == 400 and e["error"]["codigo"] == "entrada_invalida", "texto vacio -> 400")

    print(f"\n{c.total - len(c.fallos)}/{c.total} comprobaciones")
    if c.fallos:
        print("Fallaron:")
        for f in c.fallos:
            print("  -", f)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
