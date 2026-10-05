"""Comprueba que los diagramas y el Anexo A del informe coincidan con el codigo.

    python documentacion/diagramas/verificar_trazas.py

Cuatro verificaciones:

  1. Las llamadas `a2a.invocar` de grafo.py son exactamente las habilidades que
     valida `core/orquestador/validacion.py`.
  2. Los Agent Cards publican las nueve herramientas MCP, y los parametros del
     Anexo A (nombre, tipo, obligatorio) coinciden con el esquema de cada una.
  3. La validacion de arranque detecta una habilidad ausente y un agente caido.
  4. La traza real de cada camino coincide con `secuencias_esperadas.json`.

La cuarta usa Gemini y arranca los agentes en hilos de este proceso. La reserva de
prueba se escribe en una COPIA temporal de la base de datos, nunca en la real.
Sale con codigo 1 si alguna verificacion falla.
"""

from __future__ import annotations

import ast
import asyncio
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))

from core import config  # noqa: E402

INFORME = RAIZ / "documentacion" / "informe_final.md"
ESPERADAS = Path(__file__).with_name("secuencias_esperadas.json")

_RESULTADOS: list[tuple[str, bool, str]] = []


def registrar(nombre: str, ok: bool, detalle: str = "") -> None:
    _RESULTADOS.append((nombre, ok, detalle))
    print(f"[{'ok' if ok else 'FALLA'}] {nombre}" + (f": {detalle}" if detalle else ""))


# ---------------------------------------------------------------------------
# 1. grafo.py contra validacion.py
# ---------------------------------------------------------------------------
def verificar_habilidades_requeridas() -> None:
    from core.orquestador.validacion import HABILIDADES_REQUERIDAS

    arbol = ast.parse((RAIZ / "core" / "orquestador" / "grafo.py").read_text(encoding="utf-8"))
    en_grafo: set[str] = set()
    for nodo in ast.walk(arbol):
        if (
            isinstance(nodo, ast.Call)
            and isinstance(nodo.func, ast.Attribute)
            and nodo.func.attr == "invocar"
            and len(nodo.args) >= 2
            and all(isinstance(a, ast.Constant) for a in nodo.args[:2])
        ):
            en_grafo.add(f"{nodo.args[0].value}.{nodo.args[1].value}")

    declaradas = {f"{a}.{h}" for a, hs in HABILIDADES_REQUERIDAS.items() for h in hs}
    registrar(
        "grafo.py invoca solo las habilidades que valida validacion.py",
        en_grafo == declaradas and len(en_grafo) == 4,
        f"grafo={sorted(en_grafo)} validacion={sorted(declaradas)}",
    )


# ---------------------------------------------------------------------------
# 2. Agent Cards y Anexo A contra el esquema MCP
# ---------------------------------------------------------------------------
def _esquemas_mcp() -> dict[str, dict]:
    from core.agentes import servidores_mcp

    async def leer() -> dict[str, dict]:
        salida: dict[str, dict] = {}
        for clave in servidores_mcp.CONSTRUCTORES:
            for t in await servidores_mcp.construir(clave).list_tools():
                esquema = getattr(t, "input_schema", None) or getattr(t, "inputSchema", {})
                salida[t.name] = {"agente": clave, "esquema": esquema}
        return salida

    return asyncio.run(leer())


def _anexo_a3(texto: str) -> dict[str, dict[str, tuple[str, bool]]]:
    """{herramienta: {parametro: (tipo, obligatorio)}} leido del Anexo A.3."""
    inicio = texto.index("### A.3 ")
    fin = texto.index("### A.4 ")
    bloques = re.split(r"^#### A\.3\.\d+ ", texto[inicio:fin], flags=re.M)[1:]
    herramientas: dict[str, dict[str, tuple[str, bool]]] = {}
    for bloque in bloques:
        nombre = re.match(r"`(\w+)`", bloque).group(1)
        filas = re.findall(r"^\| `(\w+)` \| (\w+) \| (sí|no) \|", bloque, flags=re.M)
        herramientas[nombre] = {p: (t, o == "sí") for p, t, o in filas}
    return herramientas


def verificar_anexo_y_tarjetas() -> None:
    from core.a2a import servidor

    esquemas = _esquemas_mcp()
    anexo = _anexo_a3(INFORME.read_text(encoding="utf-8"))

    registrar("el anexo describe las 9 herramientas publicadas",
              set(anexo) == set(esquemas) and len(esquemas) == 9,
              f"anexo={len(anexo)} mcp={len(esquemas)}")

    for nombre, datos in sorted(esquemas.items()):
        props = datos["esquema"].get("properties", {})
        requeridos = set(datos["esquema"].get("required", []))
        real = {p: (v.get("type"), p in requeridos) for p, v in props.items()}
        registrar(f"parametros de {nombre} coinciden con el anexo",
                  real == anexo.get(nombre), f"mcp={real} anexo={anexo.get(nombre)}")

    for clave in config.AGENTES:
        ids = {s.id for s in servidor.construir_agent_card(clave).skills}
        mcp = {n for n, d in esquemas.items() if d["agente"] == clave}
        registrar(f"Agent Card de {clave} publica sus herramientas MCP", ids == mcp,
                  f"tarjeta={sorted(ids)} mcp={sorted(mcp)}")


# ---------------------------------------------------------------------------
# 3 y 4. Con los agentes arriba
# ---------------------------------------------------------------------------
def verificar_validacion_de_arranque() -> None:
    from core.a2a import cliente as a2a
    from core.orquestador import validacion

    real = validacion.validar_agent_cards()
    registrar("validar_agent_cards: con los agentes arriba el estado es listo",
              real["estado"] == "listo", str(real["faltantes"] or real["sin_respuesta"]))

    original = a2a.descubrir

    def sin_bloquear(clave, timeout=5.0, usar_cache=True):
        d = original(clave, timeout, usar_cache)
        if clave == "disponibilidad" and d.get("ok"):
            tarjeta = dict(d["tarjeta"])
            tarjeta["skills"] = [s for s in tarjeta.get("skills", [])
                                 if s.get("id") != "bloquear_reserva"]
            return {**d, "tarjeta": tarjeta}
        return d

    def sin_recomendador(clave, timeout=5.0, usar_cache=True):
        if clave == "recomendador":
            return {"ok": False, "error": "ConnectError: simulado"}
        return original(clave, timeout, usar_cache)

    try:
        a2a.descubrir = sin_bloquear
        r1 = validacion.validar_agent_cards()
        a2a.descubrir = sin_recomendador
        r2 = validacion.validar_agent_cards()
    finally:
        a2a.descubrir = original

    registrar("validar_agent_cards detecta una habilidad ausente",
              r1["estado"] == "degradado" and r1["faltantes"] == ["disponibilidad.bloquear_reserva"],
              str(r1["faltantes"]))
    registrar("validar_agent_cards detecta un agente que no responde",
              r2["estado"] == "degradado" and r2["sin_respuesta"] == ["recomendador"],
              str(r2["sin_respuesta"]))


def verificar_trazas() -> None:
    from core.a2a import cliente as a2a
    from core.a2a import lanzador
    from core.aprendizaje import continuo
    from core.llm import gemini
    from core.orquestador import grafo

    casos = json.loads(ESPERADAS.read_text(encoding="utf-8"))["casos"]
    ya_arriba = [c for c in config.AGENTES if lanzador._responde(c)]

    temporal = Path(tempfile.mkdtemp(prefix="vm_verificar_"))
    original_db = config.RUTA_SQLITE
    original_episodios = continuo.RUTA_EPISODIOS
    try:
        copia = temporal / "vallis.db"
        shutil.copy2(original_db, copia)
        # Cada turno agrega un episodio a la memoria del proyecto; aqui no.
        continuo.RUTA_EPISODIOS = temporal / "memoria_episodica.jsonl"
        # Los agentes leen config.RUTA_SQLITE en cada operacion, asi que mientras
        # corran en hilos de este proceso escriben en la copia.
        if not ya_arriba:
            config.RUTA_SQLITE = copia
        arranque = lanzador.levantar_en_hilos()
        registrar("los tres agentes responden", arranque["ok"], str(arranque["en_linea"]))
        if not arranque["ok"]:
            return
        a2a.precalentar()

        verificar_validacion_de_arranque()

        modo = "DEGRADADO (sin clave de Gemini)" if gemini.modo_degradado() else "Gemini real"
        print(f"\nTrazas por camino ({modo}):")
        for caso in casos:
            if caso.get("escribe_datos") and ya_arriba:
                print(f"[omitido] {caso['id']}: los agentes ya corrian en otro proceso y "
                      "escribirian en la base real")
                continue
            r = grafo.responder(caso["texto"], historial=[], context_id=f"verificar-{caso['id']}")
            tz = r["_trazas"]
            real = [f"{s['agente']}.{s['habilidad']}" for s in tz["saltos_a2a"]]
            ok = real == caso["saltos"]
            detalle = f"figura {caso['figura']}; esperado={caso['saltos']} real={real}"

            if "escalar" in caso:
                ok = ok and r["escalar_a_humano"] is caso["escalar"]
                detalle += f" escalar={r['escalar_a_humano']}"
            if "motivo" in caso:
                motivo = config.MOTIVO_ESCALAMIENTO[caso["motivo"]]
                ok = ok and r["motivo_escalamiento"] == motivo
            if "acciones" in caso:
                tipos = [a.get("tipo") for a in r["acciones"]]
                ok = ok and tipos == caso["acciones"]
                detalle += f" acciones={tipos}"
            detalle += (f" | router={tz['router'].get('intencion')}"
                        f"/{tz['router'].get('metodo')}")
            registrar(f"traza real coincide con la figura: {caso['id']}", ok, detalle)
    finally:
        config.RUTA_SQLITE = original_db
        continuo.RUTA_EPISODIOS = original_episodios
        shutil.rmtree(temporal, ignore_errors=True)


# ---------------------------------------------------------------------------
def main() -> int:
    verificar_habilidades_requeridas()
    verificar_anexo_y_tarjetas()
    verificar_trazas()

    fallas = [n for n, ok, _ in _RESULTADOS if not ok]
    print(f"\n{len(_RESULTADOS) - len(fallas)} de {len(_RESULTADOS)} verificaciones pasan.")
    for n in fallas:
        print(f"  FALLA: {n}")
    return 1 if fallas else 0


if __name__ == "__main__":
    raise SystemExit(main())
