"""Servicio web de Vallis Marea: API HTTP y pagina de chat estatica.

    python -m core.web            # lee PORT (8080 por defecto)

Un solo proceso y un solo worker. Los tres agentes A2A corren como servidores
en hilos de este mismo proceso (`lanzador.levantar_en_hilos`): la comunicacion
sigue siendo HTTP/JSON-RPC real, pero el contenedor ocupa la memoria de uno.

El contrato HTTP esta en documentacion/build_spec.md, seccion 6.1. Ninguna
respuesta de error incluye trazas de pila, rutas del sistema ni claves.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as TiempoAgotado
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from core import config
from core.web import seguridad
from core.web.registro import Registro
from core.web.sesiones import ID_VALIDO, REQUEST_ID_VALIDO, Almacen, Sesion, SinCupo

log = logging.getLogger("vallis.web")

VERSION = os.getenv("VM_VERSION", "dev")
RUTA_ESTATICOS = Path(__file__).resolve().parent / "static"
_MAX_CUERPO = 16 * 1024
_MAX_COMENTARIO = 500

# Campos de `metadatos` que se pueden mostrar. Lista permitida y no prohibida:
# un campo nuevo no sale al navegador hasta que alguien lo decida.
_METADATOS_PUBLICOS = (
    "intencion", "confianza_router", "metodo_router", "agentes_consultados",
    "habilidades_invocadas", "ms_saltos_a2a", "numero_saltos_a2a", "degradado",
    "ms_total", "contrato_valido", "causa", "saltos_fallidos", "motivo_agente",
    "tokens_entrada", "tokens_salida", "modelo",
)


# ---------------------------------------------------------------------------
# Entradas
# ---------------------------------------------------------------------------
class TurnoIn(BaseModel):
    session_id: str
    texto: str
    request_id: str
    consentimiento_registro: bool = False

    @field_validator("session_id")
    @classmethod
    def _sesion(cls, v: str) -> str:
        if not ID_VALIDO.match(v):
            raise ValueError("session_id invalido")
        return v

    @field_validator("request_id")
    @classmethod
    def _solicitud(cls, v: str) -> str:
        if not REQUEST_ID_VALIDO.match(v):
            raise ValueError("request_id invalido")
        return v


class FeedbackIn(BaseModel):
    session_id: str
    request_id: str
    valor: Literal["up", "down"]
    comentario: str | None = Field(default=None, max_length=_MAX_COMENTARIO)

    @field_validator("session_id")
    @classmethod
    def _sesion(cls, v: str) -> str:
        if not ID_VALIDO.match(v):
            raise ValueError("session_id invalido")
        return v

    @field_validator("request_id")
    @classmethod
    def _solicitud(cls, v: str) -> str:
        if not REQUEST_ID_VALIDO.match(v):
            raise ValueError("request_id invalido")
        return v


# ---------------------------------------------------------------------------
# Respuestas
# ---------------------------------------------------------------------------
@dataclass
class Resultado:
    status: int
    cuerpo: Any
    cabeceras: dict[str, str] = field(default_factory=dict)

    def a_respuesta(self) -> Response:
        if self.cuerpo is None:
            return Response(status_code=self.status, headers=self.cabeceras)
        return JSONResponse(self.cuerpo, status_code=self.status, headers=self.cabeceras)


_MENSAJES = {
    "sesion_desconocida": "La sesión no existe o venció. Recarga la página para empezar una nueva.",
    "solicitud_desconocida": "No encuentro ese mensaje en tu sesión.",
    "entrada_invalida": "El mensaje no es válido. Revisa que no esté vacío ni sea demasiado largo.",
    "limite_sesion": "Llegaste al máximo de mensajes de esta sesión. Recarga la página para empezar otra.",
    "limite_ritmo": "Estás escribiendo muy rápido. Espera unos segundos.",
    "tope_diario": "El demo alcanzó su tope de mensajes de hoy. Vuelve mañana.",
    "saturado": "El servicio está atendiendo a muchas personas. Intenta de nuevo en unos segundos.",
    "iniciando": "El servicio está arrancando. Intenta de nuevo en unos segundos.",
    "sin_cupo": "Hay demasiadas sesiones abiertas. Intenta de nuevo en unos minutos.",
    "tiempo_agotado": "La respuesta tardó demasiado. Intenta de nuevo.",
    "no_autorizado": "Falta el token de administración.",
    "prohibido": "Token inválido.",
    "error_interno": "Ocurrió un error inesperado. Intenta de nuevo.",
}


def error(status: int, codigo: str, retry_after: int | None = None) -> Resultado:
    cab = {"Retry-After": str(retry_after)} if retry_after else {}
    return Resultado(status, {"error": {"codigo": codigo, "mensaje": _MENSAJES[codigo]}}, cab)


def _plano(x: Any) -> Any:
    """Convierte tipos de numpy y tuplas a JSON puro."""
    if isinstance(x, dict):
        return {str(k): _plano(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_plano(v) for v in x]
    if hasattr(x, "item") and not isinstance(x, (str, bytes)):
        try:
            return x.item()
        except Exception:
            return str(x)
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    return str(x)


def _texto(valor: Any, limite: int = 600) -> str:
    return str(valor if valor is not None else "")[:limite]


# ---------------------------------------------------------------------------
# Valores del demo publico
# ---------------------------------------------------------------------------
def aplicar_valores_del_demo() -> None:
    """Valores de un demo publico, salvo que el entorno diga otra cosa.

    Las evaluaciones y la consola de desarrollo siguen con la fecha fija y sin
    topes (config.py); este servicio es el unico que usa reloj real.
    """
    if "VM_FECHA_HOY" not in os.environ:
        config.FECHA_HOY = "hoy"
    if "VM_TTL_RESERVA_DEMO_MIN" not in os.environ:
        config.TTL_RESERVA_MINUTOS = 30
    if "VM_MAX_RESERVAS_SESION" not in os.environ:
        config.MAX_RESERVAS_POR_SESION = 2
    if "VM_MAX_RESERVAS_BLOQUEADAS" not in os.environ:
        config.MAX_RESERVAS_BLOQUEADAS = 40
    if "VM_APRENDIZAJE" not in os.environ:
        config.APRENDIZAJE_ACTIVO = False


def _responder_con_el_grafo(texto: str, historial: list[dict[str, str]], context_id: str) -> dict[str, Any]:
    from core.orquestador import grafo

    return grafo.responder(texto, historial=historial, context_id=context_id)


# ---------------------------------------------------------------------------
# Servicio
# ---------------------------------------------------------------------------
class Servicio:
    def __init__(
        self,
        limites: seguridad.Limites | None = None,
        registro: Registro | None = None,
        responder: Callable[[str, list[dict[str, str]], str], dict[str, Any]] | None = None,
    ) -> None:
        self.limites = limites or seguridad.leer_limites()
        self.registro = registro or Registro()
        self.responder = responder or _responder_con_el_grafo
        self.sesiones = Almacen(self.limites.ttl_sesion_min, self.limites.max_sesiones)
        self.ritmo = seguridad.VentanaDeslizante()
        self.tope = seguridad.TopeDiario(self.limites.tope_diario)
        self.ejecutor = ThreadPoolExecutor(
            max_workers=self.limites.concurrencia, thread_name_prefix="turno"
        )
        # Admite los que ejecutan mas una cola corta; el resto recibe 503.
        self.admision = threading.BoundedSemaphore(self.limites.concurrencia + self.limites.cola)
        self.estado = "iniciando"
        self.avisos: list[str] = []
        self._detener = threading.Event()
        self._cache_agentes: tuple[float, dict[str, bool]] = (0.0, {})

    # -- arranque ----------------------------------------------------------
    def arrancar(self) -> None:
        """Prepara datos, indice y agentes (en un hilo, o antes de abrir el puerto)."""
        t0 = time.perf_counter()
        try:
            aplicar_valores_del_demo()
            from core.a2a import cliente as a2a
            from core.a2a import lanzador
            from core.almacen import estructurado, lexico, vectorial
            from core.llm import gemini
            from core.orquestador.validacion import validar_agent_cards

            estructurado.inicializar()
            self._asegurar_indice(vectorial, lexico)
            res = lanzador.levantar_en_hilos()
            a2a.precalentar()
            val = validar_agent_cards(usar_cache=False)
            if not res["ok"]:
                self.avisos.append("agentes_sin_respuesta:" + ",".join(res["no_responden"]))
            if val["faltantes"]:
                self.avisos.append("habilidades_faltantes:" + ",".join(val["faltantes"]))
            if gemini.modo_degradado():
                self.avisos.append("sin_api_key")
            self.estado = "degradado" if self.avisos else "listo"
        except Exception:
            log.exception("Fallo el arranque")
            self.avisos.append("fallo_de_arranque")
            self.estado = "degradado"
        log.info("Arranque terminado en %.1f s: %s %s", time.perf_counter() - t0,
                 self.estado, self.avisos)

    @staticmethod
    def _asegurar_indice(vectorial, lexico) -> None:
        try:
            faltan = vectorial.contar() == 0 or lexico.contar() == 0
        except Exception:
            faltan = True
        if faltan:
            from core.ingesta import indexar

            log.warning("Indice ausente: se reconstruye desde data/corpus")
            r = indexar.indexar_todo(recrear=True, verbose=False)
            if not r.get("ok", True):
                raise RuntimeError("no se pudo construir el indice")

    def barrer(self) -> None:
        """Quita sesiones inactivas y libera sus reservas de demo."""
        from core.almacen import estructurado

        for contexto in self.sesiones.expirar():
            estructurado.liberar_por_sesion(contexto)
        estructurado.liberar_vencidas()

    def _bucle_de_barrido(self) -> None:
        while not self._detener.wait(60):
            try:
                self.barrer()
            except Exception:
                log.exception("Fallo el barrido de sesiones")

    # -- salud -------------------------------------------------------------
    def _agentes_en_linea(self) -> dict[str, bool]:
        ts, valor = self._cache_agentes
        if valor and time.monotonic() - ts < 5:
            return valor
        from core.a2a import lanzador

        valor = {c: lanzador._responde(c, timeout=0.5) for c in config.AGENTES}
        self._cache_agentes = (time.monotonic(), valor)
        return valor

    @staticmethod
    def _router_activo() -> str:
        from core.llm import gemini
        from core.router import predecir

        if gemini.modo_degradado():
            return "reglas"
        modo = os.getenv("VM_ROUTER", "embeddings")
        if modo == "afinado" and hasattr(predecir, "_por_afinado"):
            return "afinado"
        return "embeddings" if predecir.RUTA_MODELO.exists() else "reglas"

    def salud(self) -> dict[str, Any]:
        from core.agentes.disponibilidad import logica
        from core.llm import gemini

        if self.estado == "iniciando":
            return {"estado": "iniciando", "agentes": {}, "gemini": not gemini.modo_degradado(),
                    "router": "reglas", "version": VERSION}
        agentes = self._agentes_en_linea()
        hay_gemini = not gemini.modo_degradado()
        estado = "listo" if (all(agentes.values()) and hay_gemini and self.estado == "listo") else "degradado"
        return {
            "estado": estado,
            "agentes": agentes,
            "gemini": hay_gemini,
            "router": self._router_activo(),
            "version": VERSION,
            "fecha_referencia": logica.fecha_hoy().isoformat(),
            "ttl_reserva_demo_min": config.TTL_RESERVA_MINUTOS,
        }

    def agentes(self) -> dict[str, Any]:
        from core.a2a import cliente as a2a

        salida = []
        for clave, datos in config.AGENTES.items():
            d = a2a.descubrir(clave, timeout=2.0, usar_cache=True)
            habilidades = []
            if d.get("ok"):
                for s in d["tarjeta"].get("skills", []):
                    habilidades.append({"id": _texto(s.get("id"), 80), "nombre": _texto(s.get("name"), 120)})
            salida.append({"clave": clave, "nombre": datos["nombre"],
                           "en_linea": bool(d.get("ok")), "habilidades": habilidades})
        return {"agentes": salida}

    # -- sesion ------------------------------------------------------------
    def nueva_sesion(self, ip: str) -> Resultado:
        ok, espera = self.ritmo.intentar("nueva:" + ip, self.limites.sesiones_por_ip_hora, 3600)
        if not ok:
            return error(429, "limite_ritmo", espera)
        try:
            s = self.sesiones.crear(seguridad.hash_corto(ip))
        except SinCupo:
            return error(503, "sin_cupo", 30)
        return Resultado(200, {
            "session_id": s.id,
            "turnos_restantes": self.limites.max_turnos_sesion,
            "max_caracteres": self.limites.max_caracteres,
            "ttl_reserva_demo_min": config.TTL_RESERVA_MINUTOS,
        })

    # -- turno -------------------------------------------------------------
    def turno(self, entrada: TurnoIn, ip: str) -> Resultado:
        if self.estado == "iniciando":
            return error(503, "iniciando", 5)
        sesion = self.sesiones.obtener(entrada.session_id)
        if sesion is None:
            return error(404, "sesion_desconocida")
        texto = seguridad.limpiar_texto(entrada.texto)
        if not texto or len(texto) > self.limites.max_caracteres:
            return error(400, "entrada_invalida")
        rid = entrada.request_id

        with sesion.lock:
            if rid in sesion.respuestas:
                return Resultado(200, sesion.respuestas[rid])
            futuro = sesion.pendientes.get(rid)
            if futuro is None:
                rechazo = self._admitir(sesion, ip)
                if rechazo is not None:
                    return rechazo
                futuro = self.ejecutor.submit(
                    self._ejecutar, sesion, rid, texto, entrada.consentimiento_registro
                )
                sesion.pendientes[rid] = futuro
                futuro.add_done_callback(lambda _f: self.admision.release())
        try:
            return futuro.result(timeout=self.limites.timeout_turno_s)
        except TiempoAgotado:
            return error(504, "tiempo_agotado")

    def _admitir(self, sesion: Sesion, ip: str) -> Resultado | None:
        """Decide si un turno nuevo entra. Devuelve el rechazo o None. Con `sesion.lock`."""
        lim = self.limites
        if sesion.pendientes:
            return error(429, "limite_ritmo", 2)
        if sesion.turnos >= lim.max_turnos_sesion:
            return error(429, "limite_sesion")
        ok, espera = self.ritmo.intentar("s:" + sesion.id, lim.turnos_por_minuto_sesion, 60)
        if not ok:
            return error(429, "limite_ritmo", espera)
        ok, espera = self.ritmo.intentar("i:" + ip, lim.turnos_por_minuto_ip, 60)
        if not ok:
            return error(429, "limite_ritmo", espera)
        if not self.admision.acquire(blocking=False):
            return error(503, "saturado", 3)
        if not self.tope.consumir():
            self.admision.release()
            return error(429, "tope_diario")
        return None

    def _ejecutar(self, sesion: Sesion, rid: str, texto: str, consentimiento: bool) -> Resultado:
        shash = seguridad.hash_corto(sesion.id)
        try:
            contrato = self.responder(texto, list(sesion.historial), sesion.context_id)
            cuerpo, trazas, privado = self._armar(contrato, rid, texto, consentimiento)
        except Exception:
            log.exception("Fallo un turno")
            self.registro.error(sesion_hash=shash, request_id=rid, codigo="error_interno",
                                longitud_texto=len(texto))
            with sesion.lock:
                sesion.pendientes.pop(rid, None)
            return error(500, "error_interno")

        with sesion.lock:
            sesion.turnos += 1
            sesion.recordar(texto, str(cuerpo["contrato"].get("mensaje", "")))
            cuerpo["sesion"] = {"turnos_restantes": max(0, self.limites.max_turnos_sesion - sesion.turnos)}
            sesion.guardar_respuesta(rid, cuerpo, privado)
            sesion.pendientes.pop(rid, None)
        try:
            self.registro.turno(sesion_hash=shash, request_id=rid, contrato=cuerpo["contrato"],
                                trazas=trazas, texto=texto, consentimiento=consentimiento)
        except Exception:
            log.exception("No se pudo registrar el turno")
        return Resultado(200, cuerpo)

    @staticmethod
    def _armar(contrato: dict[str, Any], rid: str, texto: str, consentimiento: bool):
        contrato = dict(contrato)
        crudas = contrato.pop("_trazas", {}) or {}
        meta = contrato.get("metadatos", {}) or {}

        publico = {
            "tipo": _texto(contrato.get("tipo"), 40),
            "mensaje": _texto(contrato.get("mensaje"), 4000),
            "citas": [
                {k: _texto(v, 300) for k, v in (c or {}).items()}
                for c in contrato.get("citas", [])[:8]
            ],
            "acciones": [
                {"tipo": _texto(a.get("tipo"), 40), "codigo": _texto(a.get("codigo"), 40),
                 "anticipo": a.get("anticipo") if isinstance(a.get("anticipo"), (int, float)) else None}
                for a in contrato.get("acciones", [])[:4]
            ],
            "escalar_a_humano": bool(contrato.get("escalar_a_humano")),
            "motivo_escalamiento": _texto(contrato.get("motivo_escalamiento"), 300),
            "metadatos": _plano({k: meta[k] for k in _METADATOS_PUBLICOS if k in meta}),
        }

        saltos = []
        for s in crudas.get("saltos_a2a", []):
            # Un rechazo de negocio trae mensaje propio del agente; una falla de
            # transporte trae texto de excepcion, que no debe salir.
            propio = bool(s.get("resultado"))
            saltos.append({
                "agente": _texto(s.get("agente"), 40),
                "habilidad": _texto(s.get("habilidad"), 60),
                "ms": s.get("ms"),
                "ok": bool(s.get("ok")),
                **({} if s.get("ok") else {
                    "error": _texto(s.get("error"), 200) if propio else "fallo de comunicacion"
                }),
            })
        router = {k: v for k, v in (crudas.get("router") or {}).items()}
        slots = {k: (_texto(v, 200) if isinstance(v, str) else v)
                 for k, v in (crudas.get("slots") or {}).items()}
        trazas = _plano({"router": router, "slots": slots, "saltos_a2a": saltos})

        cuerpo = {
            "request_id": rid,
            "contrato": publico,
            "trazas": trazas,
            "sesion": {"turnos_restantes": 0},
        }
        privado = {
            "consentimiento": consentimiento,
            "texto": texto if consentimiento else "",
            "intencion_predicha": router.get("intencion"),
        }
        return cuerpo, trazas, privado

    # -- feedback ----------------------------------------------------------
    def feedback(self, entrada: FeedbackIn, ip: str) -> Resultado:
        ok, espera = self.ritmo.intentar("f:" + ip, self.limites.turnos_por_minuto_ip, 60)
        if not ok:
            return error(429, "limite_ritmo", espera)
        sesion = self.sesiones.obtener(entrada.session_id)
        if sesion is None:
            return error(404, "sesion_desconocida")
        with sesion.lock:
            privado = sesion.privado.get(entrada.request_id)
        if privado is None:
            return error(404, "solicitud_desconocida")
        candidato = None
        if privado.get("consentimiento") and privado.get("texto"):
            candidato = {"texto": privado["texto"], "intencion_predicha": privado.get("intencion_predicha")}
        self.registro.feedback(
            sesion_hash=seguridad.hash_corto(sesion.id),
            request_id=entrada.request_id,
            valor=entrada.valor,
            comentario=seguridad.limpiar_texto(entrada.comentario or ""),
            candidato=candidato,
        )
        return Resultado(204, None)

    def detener(self) -> None:
        self._detener.set()
        self.ejecutor.shutdown(wait=False, cancel_futures=True)


# ---------------------------------------------------------------------------
# Aplicacion
# ---------------------------------------------------------------------------
def _ip(request: Request, limites: seguridad.Limites) -> str:
    peer = request.client.host if request.client else None
    return seguridad.ip_cliente(peer, request.headers.get("x-forwarded-for"), limites.saltos_proxy)


def crear_app(servicio: Servicio | None = None, *, arrancar: bool = True) -> FastAPI:
    srv = servicio or Servicio()
    frame_ancestors = os.getenv("VM_FRAME_ANCESTORS", "'none'")
    token_admin = os.getenv("ADMIN_TOKEN", "")

    @asynccontextmanager
    async def ciclo_de_vida(_app: FastAPI):
        if not arrancar:
            srv.estado = "listo"
        elif os.getenv("VM_ARRANQUE_SINCRONO", "") == "1":
            # Uvicorn abre el puerto despues del arranque: en Cloud Run la CPU se
            # limita fuera de las solicitudes, asi que el trabajo pesado no puede
            # quedar en un hilo que corre cuando ya nadie lo esta mirando.
            await asyncio.to_thread(srv.arrancar)
        else:
            threading.Thread(target=srv.arrancar, daemon=True, name="arranque").start()
        threading.Thread(target=srv._bucle_de_barrido, daemon=True, name="barrido").start()
        yield
        srv.detener()
        if arrancar:
            from core.a2a import lanzador

            lanzador.detener()

    # Sin /docs, /redoc ni /openapi.json: el contrato esta en el build-spec.
    app = FastAPI(title="Vallis Marea", version=VERSION, docs_url=None, redoc_url=None,
                  openapi_url=None, lifespan=ciclo_de_vida)
    app.state.servicio = srv

    @app.middleware("http")
    async def encabezados(request: Request, call_next):
        largo = request.headers.get("content-length")
        if request.method == "POST" and largo and largo.isdigit() and int(largo) > _MAX_CUERPO:
            resp: Response = JSONResponse(
                {"error": {"codigo": "entrada_invalida", "mensaje": _MENSAJES["entrada_invalida"]}},
                status_code=413)
        else:
            try:
                resp = await call_next(request)
            except Exception:
                log.exception("Error no controlado")
                resp = JSONResponse(
                    {"error": {"codigo": "error_interno", "mensaje": _MENSAJES["error_interno"]}},
                    status_code=500)
        for k, v in seguridad.encabezados_seguros(
                request.url.path.startswith("/api/"), frame_ancestors).items():
            resp.headers[k] = v
        return resp

    @app.exception_handler(RequestValidationError)
    async def _entrada_invalida(_request: Request, _exc: RequestValidationError):
        return error(400, "entrada_invalida").a_respuesta()

    @app.exception_handler(404)
    async def _no_encontrado(_request: Request, _exc):
        return JSONResponse({"error": {"codigo": "no_encontrado", "mensaje": "No encontrado."}},
                            status_code=404)

    @app.exception_handler(405)
    async def _metodo(_request: Request, _exc):
        return JSONResponse({"error": {"codigo": "metodo_no_permitido", "mensaje": "Método no permitido."}},
                            status_code=405)

    @app.get("/", include_in_schema=False)
    def inicio() -> Response:
        return FileResponse(RUTA_ESTATICOS / "index.html", media_type="text/html; charset=utf-8")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/api/salud")
    def salud() -> Response:
        return JSONResponse(srv.salud())

    @app.get("/api/agentes")
    def agentes() -> Response:
        return JSONResponse(srv.agentes())

    @app.post("/api/sesion")
    def sesion(request: Request) -> Response:
        return srv.nueva_sesion(_ip(request, srv.limites)).a_respuesta()

    @app.post("/api/turno")
    def turno(entrada: TurnoIn, request: Request) -> Response:
        return srv.turno(entrada, _ip(request, srv.limites)).a_respuesta()

    @app.post("/api/feedback")
    def feedback(entrada: FeedbackIn, request: Request) -> Response:
        return srv.feedback(entrada, _ip(request, srv.limites)).a_respuesta()

    @app.get("/api/admin/logs")
    def logs(request: Request) -> Response:
        if not token_admin:
            return JSONResponse({"error": {"codigo": "no_encontrado", "mensaje": "No encontrado."}},
                                status_code=404)
        codigo = seguridad.token_admin_valido(request.headers.get("authorization"), token_admin)
        if codigo == 401:
            return error(401, "no_autorizado").a_respuesta()
        if codigo == 403:
            return error(403, "prohibido").a_respuesta()
        return StreamingResponse(srv.registro.exportar(), media_type="application/x-ndjson")

    app.mount("/static", StaticFiles(directory=RUTA_ESTATICOS), name="static")
    return app


def app_por_defecto() -> FastAPI:
    logging.basicConfig(level=os.getenv("VM_LOG", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    for ruidoso in ("httpx", "httpcore", "a2a", "mcp", "chromadb"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)
    return crear_app()
