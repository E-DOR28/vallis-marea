"""Capa de acceso a Gemini: generacion de texto y embeddings.

Toda llamada al LLM del proyecto pasa por aqui. Dos razones:

1. Cache en disco. Los embeddings del corpus no cambian entre corridas, y
   reindexar pagando de nuevo cada vector es absurdo durante el desarrollo.
2. Modo degradado. Si no hay GOOGLE_API_KEY, el sistema no explota: usa un
   embedding local determinista y respuestas extractivas. Sirve para probar la
   tuberia de recuperacion sin credenciales, y esta marcado como degradado en
   todas las trazas para que nadie confunda un demo sin key con uno real.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from core import config

_LOCK = threading.Lock()
_CLIENTE = None
_RUTA_CACHE = config.RUTA_ALMACEN / "cache_llm.db"


# ---------------------------------------------------------------------------
# Cache en disco
# ---------------------------------------------------------------------------
def _conexion_cache() -> sqlite3.Connection:
    con = sqlite3.connect(_RUTA_CACHE, check_same_thread=False)
    con.execute(
        "CREATE TABLE IF NOT EXISTS cache ("
        " clave TEXT PRIMARY KEY, valor BLOB NOT NULL, tipo TEXT NOT NULL)"
    )
    return con


def _clave(*partes: str) -> str:
    return hashlib.sha256("\x1f".join(partes).encode("utf-8")).hexdigest()


def _cache_leer(clave: str) -> bytes | None:
    with _LOCK:
        con = _conexion_cache()
        try:
            fila = con.execute("SELECT valor FROM cache WHERE clave = ?", (clave,)).fetchone()
            return fila[0] if fila else None
        finally:
            con.close()


def _cache_escribir(clave: str, valor: bytes, tipo: str) -> None:
    with _LOCK:
        con = _conexion_cache()
        try:
            con.execute(
                "INSERT OR REPLACE INTO cache (clave, valor, tipo) VALUES (?, ?, ?)",
                (clave, valor, tipo),
            )
            con.commit()
        finally:
            con.close()


# ---------------------------------------------------------------------------
# Cliente
# ---------------------------------------------------------------------------
def cliente():
    """Cliente de google-genai, creado una sola vez. None si no hay API key."""
    global _CLIENTE
    if not config.hay_api_key():
        return None
    if _CLIENTE is None:
        from google import genai

        _CLIENTE = genai.Client(api_key=config.GOOGLE_API_KEY)
    return _CLIENTE


def modo_degradado() -> bool:
    """True cuando no hay API key y el sistema opera sin Gemini."""
    return not config.hay_api_key()


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
_TAREA_DOCUMENTO = "RETRIEVAL_DOCUMENT"
_TAREA_CONSULTA = "RETRIEVAL_QUERY"
_TAREA_CLASIFICACION = "CLASSIFICATION"


def _embedding_local(texto: str, dim: int) -> np.ndarray:
    """Embedding determinista sin red, para el modo degradado.

    Proyecta n-gramas de caracteres y palabras sobre un espacio de `dim`
    dimensiones mediante hashing (hashing trick). No tiene semantica real, pero
    es estable y da similitud lexica util para validar la tuberia completa.
    """
    vec = np.zeros(dim, dtype=np.float32)
    t = re.sub(r"\s+", " ", texto.lower().strip())
    unidades: list[str] = re.findall(r"\w+", t)
    unidades += [t[i : i + 4] for i in range(0, max(len(t) - 3, 0))]
    for u in unidades:
        h = int(hashlib.md5(u.encode("utf-8")).hexdigest()[:8], 16)
        vec[h % dim] += 1.0
    norma = np.linalg.norm(vec)
    return vec / norma if norma > 0 else vec


def _embed_gemini(textos: list[str], tarea: str) -> list[np.ndarray]:
    from google.genai import types

    cli = cliente()
    salida: list[np.ndarray] = []
    # La API acepta lotes; se usan lotes chicos para no pasarse de limites.
    TAM_LOTE = 32
    for i in range(0, len(textos), TAM_LOTE):
        lote = textos[i : i + TAM_LOTE]
        resp = cli.models.embed_content(
            model=config.MODELO_EMBEDDING,
            contents=lote,
            config=types.EmbedContentConfig(
                task_type=tarea,
                output_dimensionality=config.DIM_EMBEDDING,
            ),
        )
        for emb in resp.embeddings:
            v = np.asarray(emb.values, dtype=np.float32)
            # gemini-embedding-001 requiere renormalizar si se trunca la dimension.
            norma = np.linalg.norm(v)
            salida.append(v / norma if norma > 0 else v)
    return salida


def embed(textos: Iterable[str], tarea: str = _TAREA_DOCUMENTO) -> np.ndarray:
    """Devuelve una matriz (n, DIM_EMBEDDING) de vectores normalizados."""
    textos = [t if t.strip() else " " for t in textos]
    if not textos:
        return np.zeros((0, config.DIM_EMBEDDING), dtype=np.float32)

    resultado: list[np.ndarray | None] = [None] * len(textos)
    faltantes: list[int] = []

    modelo = "local" if modo_degradado() else config.MODELO_EMBEDDING
    for i, texto in enumerate(textos):
        c = _clave("emb", modelo, tarea, str(config.DIM_EMBEDDING), texto)
        crudo = _cache_leer(c)
        if crudo is not None:
            resultado[i] = np.frombuffer(crudo, dtype=np.float32)
        else:
            faltantes.append(i)

    if faltantes:
        pendientes = [textos[i] for i in faltantes]
        if modo_degradado():
            vectores = [_embedding_local(t, config.DIM_EMBEDDING) for t in pendientes]
        else:
            vectores = _embed_gemini(pendientes, tarea)
        for i, v in zip(faltantes, vectores):
            resultado[i] = v
            _cache_escribir(
                _clave("emb", modelo, tarea, str(config.DIM_EMBEDDING), textos[i]),
                v.astype(np.float32).tobytes(),
                "embedding",
            )

    return np.vstack([r for r in resultado])  # type: ignore[misc]


def embed_documentos(textos: Iterable[str]) -> np.ndarray:
    return embed(textos, _TAREA_DOCUMENTO)


def embed_consulta(texto: str) -> np.ndarray:
    return embed([texto], _TAREA_CONSULTA)[0]


def embed_clasificacion(textos: Iterable[str]) -> np.ndarray:
    return embed(textos, _TAREA_CLASIFICACION)


# ---------------------------------------------------------------------------
# Generacion
# ---------------------------------------------------------------------------
@dataclass
class RespuestaLLM:
    texto: str
    modelo: str
    degradado: bool = False
    tokens_entrada: int = 0
    tokens_salida: int = 0
    error: str | None = None
    datos: dict[str, Any] | None = field(default=None)


def generar(
    prompt: str,
    *,
    sistema: str | None = None,
    temperatura: float = config.TEMPERATURA_RESPUESTA,
    modelo: str | None = None,
    esquema_json: dict | None = None,
    max_tokens: int = 2048,
    usar_cache: bool = False,
) -> RespuestaLLM:
    """Genera texto con Gemini.

    `esquema_json` activa salida estructurada obligatoria (response_schema).
    `usar_cache` sirve para evaluaciones reproducibles; en conversacion se deja
    en False porque repetir la misma respuesta ante el mismo prompt empobrece
    la interaccion.
    """
    modelo = modelo or config.MODELO_GENERACION

    if modo_degradado():
        return RespuestaLLM(
            texto=_respuesta_degradada(prompt, esquema_json),
            modelo="degradado-sin-api-key",
            degradado=True,
        )

    clave_cache = _clave(
        "gen", modelo, sistema or "", prompt, str(temperatura), json.dumps(esquema_json or {})
    )
    if usar_cache:
        crudo = _cache_leer(clave_cache)
        if crudo is not None:
            guardado = json.loads(crudo.decode("utf-8"))
            return RespuestaLLM(**guardado)

    from google.genai import types

    cfg: dict[str, Any] = {
        "temperature": temperatura,
        "max_output_tokens": max_tokens,
    }
    if sistema:
        cfg["system_instruction"] = sistema
    if esquema_json:
        cfg["response_mime_type"] = "application/json"
        cfg["response_json_schema"] = esquema_json

    try:
        resp = cliente().models.generate_content(
            model=modelo,
            contents=prompt,
            config=types.GenerateContentConfig(**cfg),
        )
        texto = (resp.text or "").strip()
        uso = getattr(resp, "usage_metadata", None)
        salida = RespuestaLLM(
            texto=texto,
            modelo=modelo,
            tokens_entrada=getattr(uso, "prompt_token_count", 0) or 0,
            tokens_salida=getattr(uso, "candidates_token_count", 0) or 0,
        )
    except Exception as exc:  # la UI no puede caerse por un error de red
        return RespuestaLLM(
            texto="",
            modelo=modelo,
            error=f"{exc.__class__.__name__}: {exc}",
        )

    if usar_cache and not salida.error:
        _cache_escribir(
            clave_cache,
            json.dumps(salida.__dict__, ensure_ascii=False).encode("utf-8"),
            "generacion",
        )
    return salida


def generar_json(
    prompt: str,
    esquema_json: dict,
    *,
    sistema: str | None = None,
    temperatura: float = 0.0,
    modelo: str | None = None,
) -> tuple[dict | None, RespuestaLLM]:
    """Genera y parsea una respuesta JSON validada contra un esquema."""
    resp = generar(
        prompt,
        sistema=sistema,
        temperatura=temperatura,
        modelo=modelo,
        esquema_json=esquema_json,
    )
    if resp.error or not resp.texto:
        return None, resp
    try:
        return json.loads(resp.texto), resp
    except json.JSONDecodeError:
        # Rescate: extraer el primer objeto JSON del texto.
        m = re.search(r"\{.*\}", resp.texto, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0)), resp
            except json.JSONDecodeError:
                pass
        resp.error = "respuesta_no_es_json"
        return None, resp


def _respuesta_degradada(prompt: str, esquema_json: dict | None) -> str:
    """Salida de emergencia sin API key: extractiva y honesta.

    Si el prompt trae fragmentos recuperados, devuelve el primero. Nunca
    inventa: el objetivo es poder ejercitar la tuberia, no simular calidad.
    """
    if esquema_json:
        return json.dumps({"_modo": "degradado", "_nota": "sin GOOGLE_API_KEY"})
    m = re.search(r"\[F1\](.{0,700})", prompt, re.DOTALL)
    if m:
        extracto = m.group(1).strip().split("\n[F2]")[0]
        return (
            "[MODO DEGRADADO - sin GOOGLE_API_KEY] Segun la fuente recuperada:\n\n"
            f"{extracto.strip()}"
        )
    return (
        "[MODO DEGRADADO] No hay GOOGLE_API_KEY configurada, asi que no puedo "
        "generar una respuesta. Configura la clave en el archivo .env."
    )
