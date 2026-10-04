"""Vallis Marea - consola de demostracion del ecosistema de agentes.

Unico .py fuera de core/: es el punto de entrada.

    streamlit run app.py

La interfaz tiene un proposito doble. Es el entregable de "interfaz de usuario
funcional" del enunciado, y es el instrumento de demostracion: junto a cada
respuesta muestra que agente se invoco, por que protocolo, con que herramienta
MCP y con cuanta latencia. Sin ese panel, un ecosistema A2A y un monolito se
ven exactamente igual desde afuera.
"""

from __future__ import annotations

import logging
import uuid

import streamlit as st

# Las librerias de red son ruidosas en la consola de Streamlit.
for _n in ("httpx", "httpcore", "a2a", "uvicorn", "mcp", "chromadb"):
    logging.getLogger(_n).setLevel(logging.ERROR)

from core import config
from core.a2a import cliente as a2a
from core.a2a import lanzador
from core.almacen import lexico, vectorial
from core.llm import gemini
from core.orquestador import grafo

st.set_page_config(page_title="Vallis Marea - Agentes MCP/A2A", page_icon="⛵",
                   layout="wide")

EJEMPLOS = [
    "Hola, buenas tardes",
    "Tienen lancha para 12 personas el sabado para Cholon?",
    "Cuanto me devuelven si cancelo con dos dias de anticipacion?",
    "Somos 8 amigos y queremos algo con musica para celebrar",
    "Puedo llevar a mi perro?",
    "Cuanto cuesta ir a Playa Blanca el 10 de octubre para 6 personas?",
]


# ---------------------------------------------------------------------------
def inicializar_estado() -> None:
    if "mensajes" not in st.session_state:
        st.session_state.mensajes = []
    if "context_id" not in st.session_state:
        st.session_state.context_id = str(uuid.uuid4())
    if "agentes_arrancados" not in st.session_state:
        st.session_state.agentes_arrancados = False


def arrancar_agentes() -> None:
    with st.spinner("Levantando los agentes A2A..."):
        resultado = lanzador.levantar_en_hilos()
    st.session_state.agentes_arrancados = resultado["ok"]
    if resultado["ok"]:
        # Descubrimiento una sola vez: asi la latencia que muestra el panel es
        # la de la delegacion, no la del Agent Card.
        a2a.precalentar()
        st.success(f"Agentes en linea: {', '.join(resultado['en_linea'])}")
    else:
        st.error(f"No responden: {', '.join(resultado['no_responden'])}")


# ---------------------------------------------------------------------------
def barra_lateral() -> None:
    with st.sidebar:
        st.title("⛵ Vallis Marea")
        st.caption("Ecosistema de agentes interoperables (MCP + A2A)")

        if gemini.modo_degradado():
            st.error(
                "**MODO DEGRADADO**\n\n"
                "No hay `GOOGLE_API_KEY`. Los embeddings son locales y no hay "
                "generacion con LLM. Crea un archivo `.env` en la raiz con:\n\n"
                "`GOOGLE_API_KEY=tu_clave`"
            )
        else:
            st.success(f"Gemini activo · `{config.MODELO_GENERACION}`")

        st.divider()
        st.subheader("Agentes pares")
        estado = a2a.estado_agentes()
        for clave, info in estado.items():
            icono = "🟢" if info["en_linea"] else "🔴"
            with st.expander(f"{icono} {info['nombre']}", expanded=False):
                st.caption(f"`{info['url']}`")
                if info["en_linea"]:
                    st.caption("Habilidades publicadas en su Agent Card:")
                    for h in info["habilidades"]:
                        st.code(h, language=None)
                else:
                    st.caption(info.get("error") or "Sin respuesta")

        if not all(i["en_linea"] for i in estado.values()):
            if st.button("Levantar agentes", type="primary", use_container_width=True):
                arrancar_agentes()
                st.rerun()

        st.divider()
        st.subheader("Indices")
        col1, col2 = st.columns(2)
        col1.metric("Vectorial", vectorial.contar())
        col2.metric("Lexico", lexico.contar())

        with st.expander("Mantenimiento"):
            if st.button("Reindexar corpus", use_container_width=True):
                from core.ingesta import indexar

                with st.spinner("Indexando..."):
                    r = indexar.indexar_todo(recrear=True, verbose=False)
                st.success(f"{r['fragmentos']} fragmentos") if r["ok"] else st.error(r)
                st.rerun()

            if st.button("Reentrenar router", use_container_width=True):
                from core.router import entrenar

                with st.spinner("Entrenando..."):
                    m = entrenar.entrenar(verbose=False)
                st.success(f"F1 macro: {m['f1_macro']}")

        st.divider()
        st.subheader("Aprendizaje continuo")
        from core.aprendizaje import continuo as _apr

        _r = _apr.resumen()
        cA, cB = st.columns(2)
        cA.metric("Episodios", _r["memoria_episodica"]["episodios_indexados"])
        cB.metric("Correcciones", _r["correcciones"]["total"])
        _tasa = _r["memoria_episodica"]["tasa_resolucion_autonoma"]
        if _tasa is not None:
            st.caption(f"Resolucion sin humano: {_tasa:.0%}")

        st.divider()
        if st.button("Nueva conversacion", use_container_width=True):
            st.session_state.mensajes = []
            st.session_state.context_id = str(uuid.uuid4())
            st.rerun()


# ---------------------------------------------------------------------------
def panel_trazas(contrato: dict) -> None:
    """El panel que hace visible la interoperabilidad."""
    meta = contrato.get("metadatos", {})
    trazas = contrato.get("_trazas", {})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Intencion", meta.get("intencion", "?"))
    c2.metric("Saltos A2A", meta.get("numero_saltos_a2a", 0))
    c3.metric("Latencia A2A", f"{meta.get('ms_saltos_a2a', 0):.0f} ms")
    c4.metric("Total", f"{meta.get('ms_total', 0):.0f} ms")

    router = trazas.get("router", {})
    st.markdown(
        f"**Router** · metodo `{router.get('metodo','?')}` · "
        f"confianza `{router.get('confianza','?')}` · "
        f"destino `{router.get('agente_destino') or 'ninguno (resuelto por el orquestador)'}`"
    )
    if router.get("alternativas"):
        st.caption("Alternativas: " + " · ".join(
            f"{c} {p:.2f}" for c, p in router["alternativas"]))

    slots = trazas.get("slots") or {}
    if slots:
        st.markdown(
            f"**Datos extraidos** · fecha `{slots.get('fecha') or '—'}` · "
            f"pasajeros `{slots.get('pasajeros') or '—'}` · "
            f"ruta `{slots.get('ruta') or '—'}`"
        )

    saltos = trazas.get("saltos_a2a", [])
    if saltos:
        st.markdown("**Delegaciones A2A → herramientas MCP**")
        for s in saltos:
            icono = "✅" if s.get("ok") else "❌"
            st.markdown(
                f"{icono} `{s['agente']}` → herramienta MCP `{s['habilidad']}` "
                f"· **{s.get('ms', 0):.0f} ms**"
            )
            with st.expander("parametros y resultado", expanded=False):
                st.json({"parametros": s.get("parametros"),
                         "resultado": s.get("resultado")}, expanded=False)
    else:
        st.caption("Sin delegacion: el orquestador resolvio el turno por si mismo.")

    if meta.get("modelo"):
        st.caption(
            f"Modelo `{meta['modelo']}` · tokens {meta.get('tokens_entrada', 0)} entrada "
            f"/ {meta.get('tokens_salida', 0)} salida"
        )

    with st.expander("Contrato JSON publicado (el mismo de produccion)"):
        st.json({k: v for k, v in contrato.items() if k != "_trazas"})


def mostrar_mensaje(m: dict) -> None:
    with st.chat_message(m["rol"]):
        st.markdown(m["texto"])

        contrato = m.get("contrato")
        if not contrato:
            return

        if contrato.get("citas"):
            st.markdown("**Fuentes citadas**")
            for c in contrato["citas"]:
                st.caption(
                    f"📄 {c.get('titulo','')} — seccion «{c.get('seccion','')}» "
                    f"· `{c.get('fuente','')}` v{c.get('version','')}"
                )

        if contrato.get("escalar_a_humano"):
            st.warning(f"Escalado a un humano · {contrato.get('motivo_escalamiento','')}")

        for a in contrato.get("acciones", []):
            if a.get("tipo") == "reserva_bloqueada":
                st.success(
                    f"Reserva bloqueada · codigo **{a['codigo']}** · "
                    f"anticipo {a.get('anticipo', 0):,} COP".replace(",", ".")
                )

        with st.expander("🔍 Trazas del ecosistema", expanded=False):
            panel_trazas(contrato)


# ---------------------------------------------------------------------------
def main() -> None:
    inicializar_estado()
    barra_lateral()

    st.title("Consola de demostracion")
    st.caption(
        "De agente monolitico a ecosistema de agentes interoperables · "
        "SI7016 Procesamiento del Lenguaje Natural Aplicado · Universidad EAFIT"
    )

    if vectorial.contar() == 0:
        st.warning(
            "El indice esta vacio. Ejecuta `python -m core.ingesta.indexar --recrear` "
            "o usa **Reindexar corpus** en la barra lateral."
        )

    if not st.session_state.mensajes:
        st.markdown("**Prueba con:**")
        cols = st.columns(3)
        for i, ejemplo in enumerate(EJEMPLOS):
            if cols[i % 3].button(ejemplo, key=f"ej{i}", use_container_width=True):
                st.session_state.pendiente = ejemplo
                st.rerun()

    for m in st.session_state.mensajes:
        mostrar_mensaje(m)

    entrada = st.chat_input("Escribe como si fueras un cliente por WhatsApp...")
    if "pendiente" in st.session_state:
        entrada = st.session_state.pop("pendiente")

    if not entrada:
        return

    st.session_state.mensajes.append({"rol": "usuario", "texto": entrada})
    with st.chat_message("usuario"):
        st.markdown(entrada)

    estado = a2a.estado_agentes()
    if not all(i["en_linea"] for i in estado.values()):
        arrancar_agentes()

    historial = [
        {"rol": m["rol"], "texto": m["texto"]} for m in st.session_state.mensajes[:-1]
    ]

    with st.spinner("El orquestador esta delegando..."):
        contrato = grafo.responder(
            entrada, historial=historial, context_id=st.session_state.context_id
        )

    mensaje = {
        "rol": "assistant",
        "texto": contrato.get("mensaje", ""),
        "contrato": contrato,
    }
    st.session_state.mensajes.append(mensaje)
    mostrar_mensaje(mensaje)


if __name__ == "__main__":
    main()
