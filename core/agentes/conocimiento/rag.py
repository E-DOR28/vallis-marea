"""Agente de Conocimiento: RAG agentic con busqueda hibrida y citacion.

Tres decisiones de diseno que sostienen la calidad de las respuestas:

1. Busqueda hibrida. La densa entiende parafrasis ("puedo echar para atras el
   paseo" -> politica de cancelacion); la lexica no pierde literales ("72
   horas", "VM-04"). Se fusionan con Reciprocal Rank Fusion, que combina
   *rangos* y no *puntajes*, evitando tener que normalizar escalas
   incomparables (coseno contra BM25).

2. Citacion obligatoria. El LLM recibe fragmentos etiquetados [F1]..[Fn] y
   debe declarar en su salida estructurada cuales uso. Esas referencias se
   resuelven contra los metadatos reales del fragmento, asi que la cita no la
   inventa el modelo: la construye el codigo.

3. Abstencion en dos guardas. Una numerica antes de llamar al LLM (si no hay
   evidencia, no se gasta la llamada) y otra del propio LLM, que debe declarar
   `abstencion: true` cuando los fragmentos no contienen la respuesta. Para un
   negocio real, decir "no se" es mucho mas barato que inventar una politica de
   reembolso.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from core import config
from core.almacen import lexico, vectorial
from core.llm import gemini


# ---------------------------------------------------------------------------
# Recuperacion
# ---------------------------------------------------------------------------
@dataclass
class Fragmento:
    chunk_id: str
    texto: str
    fuente: str
    titulo: str
    seccion: str
    categoria: str = ""
    version: str = ""
    fecha: str = ""
    puntaje_denso: float = 0.0
    puntaje_lexico: float = 0.0
    puntaje_fusion: float = 0.0
    rango_denso: int | None = None
    rango_lexico: int | None = None

    def cita(self) -> dict[str, str]:
        return {
            "fuente": self.fuente,
            "titulo": self.titulo,
            "seccion": self.seccion,
            "version": self.version,
            "fecha": self.fecha,
        }


def recuperar(
    consulta: str,
    k_final: int = config.TOP_K_FINAL,
) -> tuple[list[Fragmento], dict[str, Any]]:
    """Busqueda hibrida densa + lexica fusionada con RRF.

    Devuelve (fragmentos ordenados, diagnostico) donde el diagnostico alimenta
    el visor de trazas de la interfaz.
    """
    t0 = time.time()

    vector = gemini.embed_consulta(consulta)
    densos = vectorial.buscar(vector, k=config.TOP_K_DENSO)
    t_denso = time.time()

    lexicos = lexico.buscar(consulta, k=config.TOP_K_LEXICO)
    t_lexico = time.time()

    # Reciprocal Rank Fusion: score(d) = sum_l 1 / (K + rank_l(d))
    acumulado: dict[str, Fragmento] = {}

    for rango, d in enumerate(densos, start=1):
        f = Fragmento(
            chunk_id=d["chunk_id"],
            texto=d["texto"],
            fuente=d["fuente"],
            titulo=d["titulo"],
            seccion=d["seccion"],
            categoria=d.get("categoria", ""),
            version=d.get("version", ""),
            fecha=d.get("fecha", ""),
            puntaje_denso=d["puntaje_denso"],
            rango_denso=rango,
        )
        f.puntaje_fusion = 1.0 / (config.RRF_K + rango)
        acumulado[f.chunk_id] = f

    for rango, l in enumerate(lexicos, start=1):
        cid = l["chunk_id"]
        aporte = 1.0 / (config.RRF_K + rango)
        if cid in acumulado:
            acumulado[cid].puntaje_lexico = l["puntaje_lexico"]
            acumulado[cid].rango_lexico = rango
            acumulado[cid].puntaje_fusion += aporte
        else:
            f = Fragmento(
                chunk_id=cid,
                texto=l["texto"],
                fuente=l["fuente"],
                titulo=l["titulo"],
                seccion=l["seccion"],
                puntaje_lexico=l["puntaje_lexico"],
                rango_lexico=rango,
                puntaje_fusion=aporte,
            )
            acumulado[cid] = f

    ordenados = sorted(acumulado.values(), key=lambda f: f.puntaje_fusion, reverse=True)
    seleccion = ordenados[:k_final]

    diagnostico = {
        "consulta": consulta,
        "candidatos_densos": len(densos),
        "candidatos_lexicos": len(lexicos),
        "candidatos_unicos": len(acumulado),
        "seleccionados": len(seleccion),
        "mejor_denso": max((d["puntaje_denso"] for d in densos), default=0.0),
        "mejor_lexico": max((l["puntaje_lexico"] for l in lexicos), default=0.0),
        "ms_denso": round((t_denso - t0) * 1000, 1),
        "ms_lexico": round((t_lexico - t_denso) * 1000, 1),
        "en_ambas_listas": sum(
            1 for f in seleccion if f.rango_denso and f.rango_lexico
        ),
    }
    return seleccion, diagnostico


# ---------------------------------------------------------------------------
# Generacion con citacion
# ---------------------------------------------------------------------------
SISTEMA = f"""Eres el agente de conocimiento de {config.NOMBRE_NEGOCIO}, un negocio de \
alquiler de lanchas en {config.CIUDAD}.

Reglas que no puedes romper:
1. Responde UNICAMENTE con informacion contenida en los fragmentos entregados.
2. Si los fragmentos no contienen la respuesta, declara abstencion. No completes \
con conocimiento general ni con suposiciones razonables. Inventar una politica de \
cancelacion o un precio le cuesta dinero real al negocio.
3. Declara en `fragmentos_usados` los numeros de los fragmentos en los que \
efectivamente te apoyaste. No listes fragmentos que no usaste.
4. Escribe en espanol neutro, claro y breve, como un asesor por WhatsApp: maximo \
tres parrafos cortos. No uses markdown ni vinetas.
5. No confirmes reservas ni disponibilidad: eso lo maneja otro agente. Si te \
preguntan por disponibilidad de una fecha concreta, dilo y declara abstencion."""

ESQUEMA_RESPUESTA = {
    "type": "object",
    "properties": {
        "respuesta": {
            "type": "string",
            "description": "Respuesta al cliente, o cadena vacia si hay abstencion.",
        },
        "abstencion": {
            "type": "boolean",
            "description": "true si los fragmentos no contienen la respuesta.",
        },
        "motivo_abstencion": {
            "type": "string",
            "description": "Por que no se pudo responder. Vacio si abstencion es false.",
        },
        "fragmentos_usados": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Numeros de fragmento efectivamente usados, ej. [1,3].",
        },
        "confianza": {
            "type": "number",
            "description": "Confianza de 0 a 1 en que la respuesta esta respaldada.",
        },
    },
    "required": ["respuesta", "abstencion", "fragmentos_usados", "confianza"],
}


def _construir_prompt(consulta: str, fragmentos: list[Fragmento], historial: str = "") -> str:
    bloques = []
    for i, f in enumerate(fragmentos, start=1):
        bloques.append(
            f"[F{i}] (fuente: {f.fuente} | documento: {f.titulo} | seccion: {f.seccion} "
            f"| version: {f.version})\n{f.texto}"
        )
    contexto = "\n\n".join(bloques)
    previo = f"\nContexto de la conversacion previa:\n{historial}\n" if historial else ""
    return (
        f"FRAGMENTOS RECUPERADOS:\n\n{contexto}\n"
        f"{previo}\n"
        f"PREGUNTA DEL CLIENTE:\n{consulta}\n\n"
        f"Responde siguiendo estrictamente las reglas."
    )


@dataclass
class RespuestaConocimiento:
    respuesta: str
    abstencion: bool
    citas: list[dict[str, str]] = field(default_factory=list)
    confianza: float = 0.0
    motivo_abstencion: str = ""
    fragmentos: list[Fragmento] = field(default_factory=list)
    diagnostico: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def a_dict(self) -> dict[str, Any]:
        return {
            "respuesta": self.respuesta,
            "abstencion": self.abstencion,
            "citas": self.citas,
            "confianza": round(self.confianza, 3),
            "motivo_abstencion": self.motivo_abstencion,
            "diagnostico": self.diagnostico,
            "fragmentos_recuperados": [
                {
                    "chunk_id": f.chunk_id,
                    "fuente": f.fuente,
                    "seccion": f.seccion,
                    "puntaje_denso": round(f.puntaje_denso, 4),
                    "puntaje_lexico": round(f.puntaje_lexico, 4),
                    "puntaje_fusion": round(f.puntaje_fusion, 5),
                    "rango_denso": f.rango_denso,
                    "rango_lexico": f.rango_lexico,
                }
                for f in self.fragmentos
            ],
            "error": self.error,
        }


def _sin_evidencia(diagnostico: dict[str, Any]) -> bool:
    """Guarda numerica previa al LLM.

    En modo degradado los puntajes no son comparables con los de Gemini, asi
    que la guarda se desactiva y se confia solo en la guarda del LLM.
    """
    if gemini.modo_degradado():
        return False
    return (
        diagnostico["mejor_denso"] < config.UMBRAL_DENSO_ABSTENCION
        and diagnostico["mejor_lexico"] < config.UMBRAL_LEXICO_ABSTENCION
    )


def responder(consulta: str, historial: str = "") -> RespuestaConocimiento:
    """Punto de entrada del agente: recupera, decide y genera con citas."""
    t0 = time.time()
    fragmentos, diagnostico = recuperar(consulta)

    if not fragmentos:
        return RespuestaConocimiento(
            respuesta="",
            abstencion=True,
            motivo_abstencion="El indice de conocimiento esta vacio. Ejecuta la ingesta.",
            diagnostico=diagnostico,
        )

    if _sin_evidencia(diagnostico):
        diagnostico["ms_total"] = round((time.time() - t0) * 1000, 1)
        diagnostico["abstencion_por"] = "guarda_numerica"
        return RespuestaConocimiento(
            respuesta="",
            abstencion=True,
            motivo_abstencion=(
                "No hay documentos en el corpus que respalden esta pregunta."
            ),
            fragmentos=fragmentos,
            diagnostico=diagnostico,
        )

    prompt = _construir_prompt(consulta, fragmentos, historial)
    datos, resp_llm = gemini.generar_json(
        prompt,
        ESQUEMA_RESPUESTA,
        sistema=SISTEMA,
        temperatura=config.TEMPERATURA_RESPUESTA,
    )

    diagnostico["ms_total"] = round((time.time() - t0) * 1000, 1)
    diagnostico["modelo"] = resp_llm.modelo
    diagnostico["tokens_entrada"] = resp_llm.tokens_entrada
    diagnostico["tokens_salida"] = resp_llm.tokens_salida
    diagnostico["degradado"] = resp_llm.degradado

    if resp_llm.error:
        return RespuestaConocimiento(
            respuesta="",
            abstencion=True,
            motivo_abstencion="Error al consultar el modelo de lenguaje.",
            fragmentos=fragmentos,
            diagnostico=diagnostico,
            error=resp_llm.error,
        )

    if datos is None:
        # Modo degradado o JSON irrecuperable: se devuelve el texto tal cual,
        # citando el fragmento mejor rankeado, y con confianza baja explicita.
        return RespuestaConocimiento(
            respuesta=resp_llm.texto,
            abstencion=False,
            citas=[fragmentos[0].cita()],
            confianza=0.2,
            fragmentos=fragmentos,
            diagnostico=diagnostico,
        )

    # Las citas las construye el codigo a partir de los indices declarados por
    # el modelo: el LLM elige CUALES, nunca inventa los metadatos.
    citas: list[dict[str, str]] = []
    for n in datos.get("fragmentos_usados", []):
        if isinstance(n, int) and 1 <= n <= len(fragmentos):
            cita = fragmentos[n - 1].cita()
            if cita not in citas:
                citas.append(cita)

    abstencion = bool(datos.get("abstencion", False))

    # Guardrail final: una respuesta afirmativa sin ninguna cita no es
    # verificable, asi que se trata como abstencion.
    if not abstencion and not citas:
        abstencion = True
        datos["motivo_abstencion"] = (
            "El modelo respondio sin declarar en que fragmento se apoyo."
        )
        diagnostico["abstencion_por"] = "sin_citas"

    return RespuestaConocimiento(
        respuesta="" if abstencion else datos.get("respuesta", ""),
        abstencion=abstencion,
        citas=citas,
        confianza=float(datos.get("confianza", 0.0)),
        motivo_abstencion=datos.get("motivo_abstencion", ""),
        fragmentos=fragmentos,
        diagnostico=diagnostico,
    )
