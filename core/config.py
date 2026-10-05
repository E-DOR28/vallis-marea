"""Configuracion centralizada del proyecto Vallis Marea.

Toda ruta, URL, puerto, nombre de modelo y catalogo fijo vive aqui.
Ningun otro modulo debe tener valores de este tipo escritos a mano.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------------
# RUTA_CORE  -> lo que vive junto a este modulo (core/)
# RAIZ_PROYECTO -> la raiz del proyecto (donde estan app.py, requirements.txt)
RUTA_CORE = Path(__file__).resolve().parent
RAIZ_PROYECTO = RUTA_CORE.parent

RUTA_DATA = RAIZ_PROYECTO / "data"
RUTA_CORPUS = RUTA_DATA / "corpus"
# VM_RUTA_ALMACEN permite construir un almacen limpio fuera de data/ (imagen del contenedor).
RUTA_ALMACEN = Path(os.getenv("VM_RUTA_ALMACEN") or (RUTA_DATA / "almacen")).resolve()
RUTA_MODELOS = RUTA_DATA / "modelos"
RUTA_EVALUACION = RUTA_DATA / "evaluacion"
RUTA_TRAZAS = RUTA_DATA / "trazas"
RUTA_APRENDIZAJE = RUTA_DATA / "aprendizaje"
RUTA_DOCUMENTACION = RAIZ_PROYECTO / "documentacion"

# Almacenes concretos.
# En produccion (Vallis Marea en AWS) estos dos son un unico Postgres con
# pgvector + tsvector. En el demo reproducible se separan en Chroma + SQLite.
RUTA_CHROMA = RUTA_ALMACEN / "chroma"
RUTA_SQLITE = RUTA_ALMACEN / "vallis.db"

for _ruta in (
    RUTA_DATA,
    RUTA_CORPUS,
    RUTA_ALMACEN,
    RUTA_MODELOS,
    RUTA_EVALUACION,
    RUTA_TRAZAS,
    RUTA_APRENDIZAJE,
):
    _ruta.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Credenciales
# --------------------------------------------------------------------------
load_dotenv(RAIZ_PROYECTO / ".env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()


def hay_api_key() -> bool:
    """True si hay una API key de Google AI Studio configurada."""
    return bool(GOOGLE_API_KEY)


# --------------------------------------------------------------------------
# Modelos
# --------------------------------------------------------------------------
# Se dejan como variables de entorno para poder cambiarlos sin tocar codigo:
# la disponibilidad de modelos de Gemini cambia con frecuencia.
MODELO_GENERACION = os.getenv("VM_MODELO_GENERACION", "gemini-2.5-flash")
MODELO_GENERACION_RAPIDO = os.getenv("VM_MODELO_RAPIDO", "gemini-2.5-flash-lite")
MODELO_EMBEDDING = os.getenv("VM_MODELO_EMBEDDING", "gemini-embedding-001")

# Dimension de los embeddings. gemini-embedding-001 admite truncamiento
# (Matryoshka); 768 es suficiente para un corpus de este tamano y es mas rapido.
DIM_EMBEDDING = int(os.getenv("VM_DIM_EMBEDDING", "768"))

TEMPERATURA_RESPUESTA = 0.2
TEMPERATURA_ROUTER = 0.0

# --------------------------------------------------------------------------
# Recuperacion (RAG)
# --------------------------------------------------------------------------
CHUNK_OBJETIVO_CARACTERES = 900
CHUNK_SOLAPE_CARACTERES = 150

TOP_K_DENSO = 8          # candidatos de la busqueda vectorial
TOP_K_LEXICO = 8         # candidatos de la busqueda BM25 / FTS5
TOP_K_FINAL = 4          # fragmentos que llegan al LLM
RRF_K = 60               # constante de Reciprocal Rank Fusion

# Abstencion. Se combinan dos guardas independientes:
#  1. numerica: si la mejor similitud densa y el mejor puntaje lexico son ambos
#     debiles, no hay evidencia y no se llama siquiera al LLM.
#  2. del propio LLM: se le exige declarar abstencion en su salida estructurada
#     cuando los fragmentos no contienen la respuesta.
# La segunda es la que de verdad evita alucinaciones; la primera ahorra la llamada.
UMBRAL_DENSO_ABSTENCION = 0.45
UMBRAL_LEXICO_ABSTENCION = 1.0

COLECCION_CONOCIMIENTO = "vallis_conocimiento"
TABLA_FTS = "conocimiento_fts"

# --------------------------------------------------------------------------
# Agentes y protocolo A2A
# --------------------------------------------------------------------------
HOST_AGENTES = os.getenv("VM_HOST_AGENTES", "127.0.0.1")

# Cada agente par es un servicio A2A independiente con su propio Agent Card.
AGENTES: dict[str, dict] = {
    "conocimiento": {
        "nombre": "Agente de Conocimiento",
        "puerto": int(os.getenv("VM_PUERTO_CONOCIMIENTO", "8101")),
        "descripcion": (
            "Responde preguntas sobre politicas, FAQs y rutas turisticas de "
            "Vallis Marea usando RAG agentic sobre el corpus indexado, citando "
            "siempre la fuente recuperada."
        ),
        "habilidades": ["consultar_politicas", "consultar_faq", "describir_ruta"],
    },
    "disponibilidad": {
        "nombre": "Agente de Disponibilidad y Reservas",
        "puerto": int(os.getenv("VM_PUERTO_DISPONIBILIDAD", "8102")),
        "descripcion": (
            "Consulta disponibilidad de la flota en tiempo real, cotiza paseos "
            "y bloquea reservas de forma idempotente."
        ),
        "habilidades": ["consultar_disponibilidad", "cotizar", "bloquear_reserva"],
    },
    "recomendador": {
        "nombre": "Agente Recomendador",
        "puerto": int(os.getenv("VM_PUERTO_RECOMENDADOR", "8103")),
        "descripcion": (
            "Sugiere embarcaciones y paseos segun las preferencias expresadas "
            "por el cliente durante la conversacion."
        ),
        "habilidades": ["recomendar_embarcacion", "recomendar_paseo"],
    },
}

VERSION_AGENTES = "1.0.0"

# Ruta del Agent Card. El spec de A2A la publica bajo /.well-known/.
# Se exponen las dos rutas conocidas por compatibilidad entre versiones.
RUTA_AGENT_CARD = "/.well-known/agent-card.json"
RUTA_AGENT_CARD_LEGACY = "/.well-known/agent.json"


def url_agente(clave: str) -> str:
    """URL base del servicio A2A de un agente par."""
    return f"http://{HOST_AGENTES}:{AGENTES[clave]['puerto']}"


TIMEOUT_A2A_SEGUNDOS = 60.0

# --------------------------------------------------------------------------
# Catalogo de intenciones (router del orquestador)
# --------------------------------------------------------------------------
# Cada intencion se mapea al agente par que la atiende. "otro" y "saludo" los
# resuelve el orquestador directamente, sin delegar.
INTENCIONES: dict[str, str | None] = {
    "saludo": None,
    "disponibilidad": "disponibilidad",
    "precio": "disponibilidad",
    "reserva": "disponibilidad",
    "politica": "conocimiento",
    "faq": "conocimiento",
    "ruta_turistica": "conocimiento",
    "recomendacion": "recomendador",
    "otro": None,
}

UMBRAL_CONFIANZA_ROUTER = 0.45  # por debajo, se consulta al LLM como desempate

# --------------------------------------------------------------------------
# Guardrails y escalamiento
# --------------------------------------------------------------------------
# El agente nunca confirma un precio o una reserva sin haber llamado a la
# herramienta correspondiente: se escala a un humano en Chatwoot.
MAX_TURNOS_MEMORIA_CORTA = 12
MOTIVO_ESCALAMIENTO = {
    "baja_confianza": "El agente no tuvo evidencia suficiente para responder.",
    "sin_fuente": "Se pidio informacion de politicas sin fuente recuperable.",
    "error_herramienta": "Una herramienta MCP fallo de forma irrecuperable.",
}

# --------------------------------------------------------------------------
# Despliegue web (R3)
# --------------------------------------------------------------------------
def _entero(nombre: str, defecto: int) -> int:
    try:
        return int(os.getenv(nombre, str(defecto)))
    except ValueError:
        return defecto


# Fecha de referencia del negocio. Vacio = fecha fija del demo (reproducible,
# la que usan las evaluaciones); "hoy" = reloj real en hora de Colombia;
# AAAA-MM-DD = esa fecha.
FECHA_HOY = os.getenv("VM_FECHA_HOY", "").strip()

# Una reserva "bloqueada" que no recibe anticipo se libera sola. 0 = no vence.
# Las reservas "confirmadas" de la semilla nunca vencen.
TTL_RESERVA_MINUTOS = _entero("VM_TTL_RESERVA_DEMO_MIN", 0)

# Topes que protegen el inventario de un demo publico. 0 = sin tope.
MAX_RESERVAS_POR_SESION = _entero("VM_MAX_RESERVAS_SESION", 0)
MAX_RESERVAS_BLOQUEADAS = _entero("VM_MAX_RESERVAS_BLOQUEADAS", 0)

# Reintentos ante fallos transitorios de Gemini (429, 5xx, red).
LLM_REINTENTOS = _entero("VM_LLM_REINTENTOS", 2)
LLM_ESPERA_BASE_SEGUNDOS = float(os.getenv("VM_LLM_ESPERA_BASE", "0.8"))
LLM_TIMEOUT_SEGUNDOS = float(os.getenv("VM_LLM_TIMEOUT", "45"))

# Aprendizaje continuo: indexar cada turno como memoria episodica. Se apaga en
# el despliegue publico (un usuario no debe poder sembrar la memoria de otro).
APRENDIZAJE_ACTIVO = os.getenv("VM_APRENDIZAJE", "1").strip() not in ("0", "false", "no")

# --------------------------------------------------------------------------
# Negocio (datos semilla del demo)
# --------------------------------------------------------------------------
NOMBRE_NEGOCIO = "Vallis Marea"
CIUDAD = "Cartagena de Indias, Colombia"
MONEDA = "COP"
HORARIO_OPERACION = "07:00 a 17:00, todos los dias"
PUNTO_EMBARQUE = "Muelle de La Bodeguita"

# --------------------------------------------------------------------------
# Router afinado (R2)
# --------------------------------------------------------------------------
# `embeddings` (predeterminado): regresion logistica sobre embeddings de Gemini.
# `afinado`: encoder multilingue afinado, local. Solo reemplaza el nivel 1 de la
# cascada; el LLM de desempate y las reglas siguen igual.
ROUTER_NIVEL1 = os.getenv("VM_ROUTER", "embeddings")
# Ruta local o `usuario/repo@<hash de 40 caracteres>` del Hub (revision obligatoria).
ROUTER_MODELO = os.getenv("VM_ROUTER_MODELO", "")
# Solo para descargar un modelo privado del Hub. Lectura es suficiente.
HF_TOKEN = os.getenv("HF_TOKEN", "")
