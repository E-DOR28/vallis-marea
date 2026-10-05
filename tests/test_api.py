"""Pruebas de la API HTTP (build-spec 6.1 y 8.3). No usan red ni clave de API.

El grafo se sustituye por un responder simulado, salvo en la prueba de
concurrencia de reservas, que usa la logica real del agente de disponibilidad.

    python -m unittest tests.test_api -v
"""

from __future__ import annotations

import dataclasses
import io
import json
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from core import config
from core.agentes.disponibilidad import logica
from core.almacen import estructurado
from core.web import app as web
from core.web import seguridad
from core.web.registro import Registro
from tests.base_db import ConBaseTemporal


def contrato_simulado(mensaje: str = "Hola, soy el asistente.", **extra) -> dict:
    c = {
        "tipo": "texto", "mensaje": mensaje, "citas": [], "acciones": [],
        "escalar_a_humano": False, "motivo_escalamiento": "",
        "metadatos": {"intencion": "faq", "confianza_router": 0.9, "metodo_router": "clasificador",
                      "ms_total": 12.0, "degradado": False, "modelo": "m",
                      "error_llm": "Traceback SECRETO /Users/kin/.env KEY=abc"},
        "_trazas": {
            "router": {"intencion": "faq", "confianza": 0.9, "metodo": "clasificador", "ms": 1.0,
                       "alternativas": [("faq", 0.9)], "agente_destino": "conocimiento"},
            "slots": {"fecha": None, "pasajeros": 0},
            "saltos_a2a": [{"agente": "conocimiento", "habilidad": "buscar_conocimiento", "ms": 5.0,
                            "ok": True, "parametros": {"clave_idempotencia": "SECRETA|VM-01|x"},
                            "resultado": {"ok": True}}],
        },
    }
    c.update(extra)
    return c


class Base(unittest.TestCase):
    ajustes: dict = {}

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.salida = io.StringIO()
        self.registro = Registro(Path(self._tmp.name), salida=self.salida)
        self.llamadas: list[tuple] = []
        self.comportamiento = lambda texto, historial, ctx: contrato_simulado()
        valores = dict(concurrencia=4, cola=4, timeout_turno_s=10, turnos_por_minuto_sesion=1000,
                       turnos_por_minuto_ip=1000, sesiones_por_ip_hora=1000)
        valores.update(self.ajustes)
        base = dataclasses.replace(seguridad.leer_limites(), **valores)
        self.srv = web.Servicio(base, self.registro, self._responder)
        self.cliente = TestClient(web.crear_app(self.srv, arrancar=False))
        self.cliente.__enter__()
        self.addCleanup(self.cliente.__exit__, None, None, None)

    def _responder(self, texto, historial, ctx):
        self.llamadas.append((texto, list(historial), ctx))
        return self.comportamiento(texto, historial, ctx)

    def nueva_sesion(self, **kw) -> str:
        r = self.cliente.post("/api/sesion", **kw)
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()["session_id"]

    def turno(self, sid: str, texto: str = "hola", rid: str | None = None, **kw):
        cuerpo = {"session_id": sid, "texto": texto, "request_id": rid or str(uuid.uuid4()),
                  "consentimiento_registro": False}
        cuerpo.update(kw.pop("cuerpo", {}))
        return self.cliente.post("/api/turno", json=cuerpo, **kw)


class ContratoTest(Base):
    def test_una_sesion_y_un_turno(self) -> None:
        r = self.cliente.post("/api/sesion")
        d = r.json()
        self.assertEqual(d["turnos_restantes"], self.srv.limites.max_turnos_sesion)
        self.assertEqual(d["max_caracteres"], self.srv.limites.max_caracteres)
        self.assertGreaterEqual(len(d["session_id"]), 20)

        rid = str(uuid.uuid4())
        r = self.turno(d["session_id"], "Hola", rid)
        self.assertEqual(r.status_code, 200)
        j = r.json()
        self.assertEqual(j["request_id"], rid)
        self.assertEqual(j["contrato"]["mensaje"], "Hola, soy el asistente.")
        self.assertEqual(j["sesion"]["turnos_restantes"], d["turnos_restantes"] - 1)
        self.assertEqual(set(j["trazas"]), {"router", "slots", "saltos_a2a"})
        self.assertNotIn("_trazas", j["contrato"])

    def test_el_navegador_no_recibe_parametros_ni_errores_internos(self) -> None:
        j = self.turno(self.nueva_sesion()).json()
        crudo = json.dumps(j)
        for prohibido in ("SECRETA", "SECRETO", "/Users/kin", "clave_idempotencia", "error_llm", "KEY="):
            self.assertNotIn(prohibido, crudo)
        self.assertNotIn("parametros", j["trazas"]["saltos_a2a"][0])

    def test_salud_tiene_la_forma_del_contrato(self) -> None:
        with mock.patch.object(web.Servicio, "_agentes_en_linea",
                               return_value={"conocimiento": True, "disponibilidad": True,
                                             "recomendador": True}):
            j = self.cliente.get("/api/salud").json()
        self.assertIn(j["estado"], {"listo", "degradado", "iniciando"})
        self.assertEqual(set(j["agentes"]), {"conocimiento", "disponibilidad", "recomendador"})
        self.assertIsInstance(j["gemini"], bool)
        self.assertIn(j["router"], {"embeddings", "afinado", "reglas"})
        self.assertTrue(j["version"])
        self.assertNotIn("GOOGLE_API_KEY", json.dumps(j))

    def test_mientras_inicia_la_salud_responde_y_los_turnos_esperan(self) -> None:
        self.srv.estado = "iniciando"
        self.assertEqual(self.cliente.get("/api/salud").json()["estado"], "iniciando")
        r = self.turno(self.nueva_sesion())
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.json()["error"]["codigo"], "iniciando")
        self.assertIn("Retry-After", r.headers)

    def test_sin_clave_de_gemini_la_salud_es_degradada(self) -> None:
        with mock.patch.object(web.Servicio, "_agentes_en_linea", return_value={"conocimiento": True}), \
                mock.patch.object(config, "GOOGLE_API_KEY", ""):
            j = self.cliente.get("/api/salud").json()
        self.assertEqual(j["estado"], "degradado")
        self.assertFalse(j["gemini"])


class IdempotenciaTest(Base):
    def test_un_request_id_repetido_devuelve_lo_mismo_sin_volver_a_ejecutar(self) -> None:
        sid, rid = self.nueva_sesion(), str(uuid.uuid4())
        a = self.turno(sid, "hola", rid).json()
        b = self.turno(sid, "hola", rid).json()
        self.assertEqual(a, b)
        self.assertEqual(len(self.llamadas), 1)
        self.assertEqual(a["sesion"]["turnos_restantes"], self.srv.limites.max_turnos_sesion - 1)

    def test_el_doble_clic_simultaneo_ejecuta_una_sola_vez(self) -> None:
        liberar = threading.Event()

        def lento(texto, historial, ctx):
            liberar.wait(5)
            return contrato_simulado()

        self.comportamiento = lento
        sid, rid = self.nueva_sesion(), str(uuid.uuid4())
        resultados: list = []
        hilos = [threading.Thread(target=lambda: resultados.append(self.turno(sid, "hola", rid).json()))
                 for _ in range(2)]
        for h in hilos:
            h.start()
        time.sleep(0.3)
        liberar.set()
        for h in hilos:
            h.join(10)
        self.assertEqual(len(self.llamadas), 1)
        self.assertEqual(len(resultados), 2)
        self.assertEqual(resultados[0], resultados[1])


class SesionTest(Base):
    def test_el_context_id_lo_pone_el_servidor_y_no_es_el_session_id(self) -> None:
        a, b = self.nueva_sesion(), self.nueva_sesion()
        self.turno(a), self.turno(a), self.turno(b)
        ctx_a1, ctx_a2, ctx_b = (c[2] for c in self.llamadas)
        self.assertEqual(ctx_a1, ctx_a2)
        self.assertNotEqual(ctx_a1, ctx_b)
        self.assertNotIn(a, ctx_a1)
        self.assertNotIn("|", ctx_a1)

    def test_el_cliente_no_puede_inyectar_un_context_id(self) -> None:
        sid = self.nueva_sesion()
        self.turno(sid, cuerpo={"context_id": "ajeno|VM-01|2026-10-20"})
        self.assertNotIn("ajeno", self.llamadas[0][2])

    def test_el_historial_guarda_los_ultimos_doce_mensajes(self) -> None:
        sid = self.nueva_sesion()
        for i in range(10):
            self.turno(sid, f"mensaje {i}")
        historial = self.llamadas[-1][1]
        self.assertEqual(len(historial), config.MAX_TURNOS_MEMORIA_CORTA)
        self.assertEqual(historial[-2], {"rol": "usuario", "texto": "mensaje 8"})
        self.assertEqual(historial[-1]["rol"], "agente")

    def test_una_sesion_ajena_o_inventada_es_404(self) -> None:
        for sid in ("x" * 30, "A" * 32, "no-valida"):
            r = self.turno(sid)
            self.assertIn(r.status_code, (400, 404), sid)
        self.assertEqual(self.turno("A" * 32).json()["error"]["codigo"], "sesion_desconocida")

    def test_una_sesion_inactiva_vence_y_libera_sus_reservas(self) -> None:
        reloj = [0.0]
        self.srv.sesiones = web.Almacen(1, 10, reloj=lambda: reloj[0])
        s = self.srv.sesiones.crear("h")
        reloj[0] = 61.0
        self.assertIsNone(self.srv.sesiones.obtener(s.id))
        self.assertEqual(self.srv.sesiones.expirar(), [s.context_id])

    def test_el_maximo_de_sesiones(self) -> None:
        self.srv.sesiones = web.Almacen(60, 2)
        self.nueva_sesion(), self.nueva_sesion()
        r = self.cliente.post("/api/sesion")
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.json()["error"]["codigo"], "sin_cupo")


class EntradaInvalidaTest(Base):
    def setUp(self) -> None:
        super().setUp()
        self.sid = self.nueva_sesion()

    def assert400(self, r, secreto: str = "") -> None:
        self.assertEqual(r.status_code, 400, r.text)
        self.assertEqual(r.json()["error"]["codigo"], "entrada_invalida")
        if secreto:
            self.assertNotIn(secreto, r.text)
        self.assertEqual(len(self.llamadas), 0)

    def test_texto_vacio_o_solo_espacios(self) -> None:
        self.assert400(self.turno(self.sid, ""))
        self.assert400(self.turno(self.sid, "   \n\t "))

    def test_texto_de_solo_caracteres_de_control(self) -> None:
        self.assert400(self.turno(self.sid, "\x00\x01\u202e\u200b"))

    def test_texto_demasiado_largo(self) -> None:
        self.assert400(self.turno(self.sid, "a" * (self.srv.limites.max_caracteres + 1)))

    def test_el_limite_exacto_es_valido(self) -> None:
        r = self.turno(self.sid, "a" * self.srv.limites.max_caracteres)
        self.assertEqual(r.status_code, 200)

    def test_tipos_incorrectos(self) -> None:
        for cuerpo in ({"texto": 123}, {"texto": ["hola"]}, {"texto": None},
                       {"request_id": 5}, {"consentimiento_registro": "tal vez"}):
            self.assert400(self.turno(self.sid, cuerpo=cuerpo))

    def test_request_id_con_caracteres_raros(self) -> None:
        self.assert400(self.turno(self.sid, rid="<script>alert(1)</script>"), "<script>")
        self.assert400(self.turno(self.sid, rid="corto"))

    def test_faltan_campos_y_json_roto(self) -> None:
        self.assert400(self.cliente.post("/api/turno", json={"texto": "hola"}))
        r = self.cliente.post("/api/turno", content=b"{no es json",
                              headers={"content-type": "application/json"})
        self.assert400(r)

    def test_un_cuerpo_enorme_se_rechaza_antes_de_leerlo(self) -> None:
        r = self.cliente.post("/api/turno", content=b"x" * 40_000,
                              headers={"content-type": "application/json"})
        self.assertEqual(r.status_code, 413)

    def test_los_caracteres_de_control_se_quitan_pero_el_texto_pasa(self) -> None:
        self.assertEqual(self.turno(self.sid, "ho\x00la\u202e").status_code, 200)
        self.assertEqual(self.llamadas[0][0], "hola")

    def test_texto_con_html_pasa_como_texto_sin_alterarse(self) -> None:
        self.turno(self.sid, "<img src=x onerror=alert(1)>")
        self.assertEqual(self.llamadas[0][0], "<img src=x onerror=alert(1)>")


class LimitesTest(Base):
    ajustes = {"max_turnos_sesion": 3}

    def test_el_limite_de_turnos_por_sesion(self) -> None:
        sid = self.nueva_sesion()
        for _ in range(3):
            self.assertEqual(self.turno(sid).status_code, 200)
        r = self.turno(sid)
        self.assertEqual(r.status_code, 429)
        self.assertEqual(r.json()["error"]["codigo"], "limite_sesion")
        self.assertEqual(len(self.llamadas), 3)

    def test_un_turno_fallido_no_gasta_cupo(self) -> None:
        self.comportamiento = lambda t, h, c: (_ for _ in ()).throw(RuntimeError("x"))
        sid = self.nueva_sesion()
        with self.assertLogs("vallis.web", level="ERROR"):
            for _ in range(5):
                self.assertEqual(self.turno(sid).status_code, 500)
        self.assertEqual(self.srv.sesiones.obtener(sid).turnos, 0)


class TopeDiarioTest(Base):
    ajustes = {"tope_diario": 2}

    def test_el_tope_diario_corta_para_todos(self) -> None:
        a, b = self.nueva_sesion(), self.nueva_sesion()
        self.assertEqual(self.turno(a).status_code, 200)
        self.assertEqual(self.turno(b).status_code, 200)
        r = self.turno(a)
        self.assertEqual(r.status_code, 429)
        self.assertEqual(r.json()["error"]["codigo"], "tope_diario")
        self.assertEqual(self.turno(b).json()["error"]["codigo"], "tope_diario")

    def test_el_tope_se_reinicia_al_cambiar_el_dia(self) -> None:
        from datetime import date

        dia = [date(2026, 10, 4)]
        tope = seguridad.TopeDiario(1, hoy=lambda: dia[0])
        self.assertTrue(tope.consumir())
        self.assertFalse(tope.consumir())
        dia[0] = date(2026, 10, 5)
        self.assertTrue(tope.consumir())


class RitmoTest(Base):
    ajustes = {"turnos_por_minuto_sesion": 2}

    def test_demasiado_rapido_devuelve_429_con_retry_after(self) -> None:
        # `base` fija 1000 en el setUp; aqui se aprieta despues de crear el servicio.
        self.srv.limites = dataclasses.replace(self.srv.limites, turnos_por_minuto_sesion=2)
        sid = self.nueva_sesion()
        self.assertEqual(self.turno(sid).status_code, 200)
        self.assertEqual(self.turno(sid).status_code, 200)
        r = self.turno(sid)
        self.assertEqual(r.status_code, 429)
        self.assertEqual(r.json()["error"]["codigo"], "limite_ritmo")
        self.assertGreaterEqual(int(r.headers["Retry-After"]), 1)

    def test_el_limite_de_sesiones_por_ip(self) -> None:
        self.srv.limites = dataclasses.replace(self.srv.limites, sesiones_por_ip_hora=2)
        self.nueva_sesion(), self.nueva_sesion()
        self.assertEqual(self.cliente.post("/api/sesion").status_code, 429)

    def test_x_forwarded_for_falsificado_no_evade_el_limite_detras_de_un_proxy(self) -> None:
        self.srv.limites = dataclasses.replace(self.srv.limites, sesiones_por_ip_hora=2, saltos_proxy=1)
        for i in range(2):
            self.assertEqual(
                self.cliente.post("/api/sesion", headers={"X-Forwarded-For": f"9.9.9.{i}, 203.0.113.7"}
                                  ).status_code, 200)
        r = self.cliente.post("/api/sesion", headers={"X-Forwarded-For": "1.2.3.4, 203.0.113.7"})
        self.assertEqual(r.status_code, 429)

    def test_sin_proxy_se_ignora_x_forwarded_for(self) -> None:
        self.srv.limites = dataclasses.replace(self.srv.limites, sesiones_por_ip_hora=2, saltos_proxy=0)
        for i in range(2):
            self.cliente.post("/api/sesion", headers={"X-Forwarded-For": f"9.9.9.{i}"})
        r = self.cliente.post("/api/sesion", headers={"X-Forwarded-For": "5.5.5.5"})
        self.assertEqual(r.status_code, 429)

    def test_un_segundo_mensaje_con_el_primero_en_curso_espera(self) -> None:
        liberar = threading.Event()
        self.comportamiento = lambda t, h, c: (liberar.wait(5), contrato_simulado())[1]
        sid = self.nueva_sesion()
        hilo = threading.Thread(target=lambda: self.turno(sid, "uno"))
        hilo.start()
        time.sleep(0.3)
        r = self.turno(sid, "dos")
        liberar.set()
        hilo.join(10)
        self.assertEqual(r.status_code, 429)
        self.assertEqual(r.json()["error"]["codigo"], "limite_ritmo")


class SaturacionTest(Base):
    ajustes = {"concurrencia": 1, "cola": 0}

    def test_sobre_el_limite_de_concurrencia_responde_503_con_retry_after(self) -> None:
        liberar = threading.Event()
        self.comportamiento = lambda t, h, c: (liberar.wait(5), contrato_simulado())[1]
        a, b = self.nueva_sesion(), self.nueva_sesion()
        hilo = threading.Thread(target=lambda: self.turno(a))
        hilo.start()
        time.sleep(0.3)
        r = self.turno(b)
        liberar.set()
        hilo.join(10)
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.json()["error"]["codigo"], "saturado")
        self.assertIn("Retry-After", r.headers)
        # Tras liberar, el servicio se recupera.
        self.assertEqual(self.turno(b).status_code, 200)


class ErroresTest(Base):
    def test_un_fallo_interno_no_filtra_nada(self) -> None:
        def explota(t, h, c):
            raise RuntimeError("clave sk-SECRETA en /Users/kin/Desktop/.env")

        self.comportamiento = explota
        with self.assertLogs("vallis.web", level="ERROR"):
            r = self.turno(self.nueva_sesion())
        self.assertEqual(r.status_code, 500)
        self.assertEqual(r.json()["error"]["codigo"], "error_interno")
        for prohibido in ("SECRETA", "/Users", "Traceback", "RuntimeError"):
            self.assertNotIn(prohibido, r.text)

    def test_un_turno_que_tarda_demasiado_responde_504(self) -> None:
        self.srv.limites = dataclasses.replace(self.srv.limites, timeout_turno_s=1)
        liberar = threading.Event()
        self.comportamiento = lambda t, h, c: (liberar.wait(5), contrato_simulado())[1]
        r = self.turno(self.nueva_sesion())
        liberar.set()
        self.assertEqual(r.status_code, 504)
        self.assertEqual(r.json()["error"]["codigo"], "tiempo_agotado")

    def test_rutas_inexistentes_y_metodos_no_permitidos(self) -> None:
        self.assertEqual(self.cliente.get("/api/nada").status_code, 404)
        self.assertEqual(self.cliente.get("/api/turno").status_code, 405)
        for ruta in ("/docs", "/openapi.json", "/redoc"):
            self.assertEqual(self.cliente.get(ruta).status_code, 404, ruta)

    def test_no_se_puede_salir_de_la_carpeta_estatica(self) -> None:
        for ruta in ("/static/../app.py", "/static/%2e%2e/app.py", "/static/..%2fseguridad.py"):
            self.assertNotEqual(self.cliente.get(ruta).status_code, 200, ruta)


class EncabezadosTest(Base):
    def test_todas_las_respuestas_llevan_los_encabezados_de_seguridad(self) -> None:
        sid = self.nueva_sesion()
        respuestas = [
            self.cliente.get("/"), self.cliente.get("/api/salud"), self.cliente.get("/static/app.js"),
            self.cliente.get("/api/nada"), self.turno(sid, ""), self.turno("A" * 32),
        ]
        for r in respuestas:
            csp = r.headers.get("content-security-policy", "")
            self.assertIn("default-src 'none'", csp, r.request.url)
            self.assertIn("script-src 'self'", csp)
            self.assertNotIn("unsafe-inline", csp)
            self.assertNotIn("unsafe-eval", csp)
            self.assertIn("frame-ancestors", csp)
            self.assertEqual(r.headers.get("x-content-type-options"), "nosniff")
            self.assertEqual(r.headers.get("referrer-policy"), "no-referrer")

    def test_la_api_no_se_guarda_en_cache(self) -> None:
        self.assertEqual(self.cliente.get("/api/salud").headers["cache-control"], "no-store")

    def test_la_pagina_se_sirve_como_html_utf8(self) -> None:
        r = self.cliente.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("text/html", r.headers["content-type"])


class AdminTest(Base):
    def _app_con_token(self, token: str):
        with mock.patch.dict("os.environ", {"ADMIN_TOKEN": token}):
            return TestClient(web.crear_app(self.srv, arrancar=False))

    def test_sin_token_configurado_el_endpoint_no_existe(self) -> None:
        self.assertEqual(self.cliente.get("/api/admin/logs").status_code, 404)
        self.assertEqual(self.cliente.get("/api/admin/logs",
                                          headers={"Authorization": "Bearer algo"}).status_code, 404)

    def test_401_sin_cabecera_o_mal_formada_y_403_con_token_equivocado(self) -> None:
        with self._app_con_token("secreto-largo") as c:
            self.assertEqual(c.get("/api/admin/logs").status_code, 401)
            self.assertEqual(c.get("/api/admin/logs", headers={"Authorization": "secreto-largo"}).status_code, 401)
            self.assertEqual(c.get("/api/admin/logs", headers={"Authorization": "Bearer "}).status_code, 401)
            self.assertEqual(c.get("/api/admin/logs", headers={"Authorization": "Bearer otro"}).status_code, 403)
            self.assertEqual(c.get("/api/admin/logs", headers={"Authorization": "Bearer secreto-larg"}).status_code, 403)

    def test_con_token_correcto_entrega_ndjson(self) -> None:
        sid = self.nueva_sesion()
        self.turno(sid, "mensaje privado", cuerpo={"consentimiento_registro": False})
        with self._app_con_token("secreto-largo") as c:
            r = c.get("/api/admin/logs", headers={"Authorization": "Bearer secreto-largo"})
        self.assertEqual(r.status_code, 200)
        self.assertIn("ndjson", r.headers["content-type"])
        lineas = [json.loads(x) for x in r.text.splitlines()]
        self.assertTrue(lineas)
        self.assertEqual(lineas[0]["tipo"], "turno")
        self.assertNotIn("mensaje privado", r.text)


class RegistroTest(Base):
    def _turnos(self) -> list[dict]:
        return [json.loads(x) for x in self.registro.ruta_turnos.read_text().splitlines()]

    def test_sin_consentimiento_no_se_guarda_el_texto_ni_la_respuesta(self) -> None:
        self.turno(self.nueva_sesion(), "dato muy personal 3001234567")
        r = self._turnos()[0]
        self.assertFalse(r["consentimiento"])
        self.assertNotIn("texto", r)
        self.assertNotIn("respuesta", r)
        self.assertEqual(r["longitud_texto"], len("dato muy personal 3001234567"))
        self.assertNotIn("dato muy personal", self.registro.ruta_turnos.read_text())
        self.assertNotIn("dato muy personal", self.salida.getvalue())

    def test_con_consentimiento_se_guarda_texto_y_respuesta(self) -> None:
        self.turno(self.nueva_sesion(), "quiero ir a Baru", cuerpo={"consentimiento_registro": True})
        r = self._turnos()[0]
        self.assertEqual(r["texto"], "quiero ir a Baru")
        self.assertEqual(r["respuesta"], "Hola, soy el asistente.")

    def test_el_registro_tiene_los_campos_de_la_seccion_6_3_y_ningun_secreto(self) -> None:
        sid = self.nueva_sesion()
        self.turno(sid)
        r = self._turnos()[0]
        for campo in ("ts", "sesion_hash", "request_id", "intencion", "metodo_router", "confianza",
                      "saltos", "ms_total", "escalado", "abstencion", "tokens_entrada",
                      "tokens_salida", "modelo", "longitud_texto"):
            self.assertIn(campo, r)
        self.assertNotIn(sid, json.dumps(r))
        self.assertNotIn("SECRETA", json.dumps(r))
        self.assertEqual(r["saltos"][0].keys(), {"agente", "habilidad", "ms", "ok"})

    def test_cada_linea_tambien_sale_por_la_salida_estandar(self) -> None:
        self.turno(self.nueva_sesion())
        self.assertEqual(json.loads(self.salida.getvalue().splitlines()[0])["tipo"], "turno")


class FeedbackTest(Base):
    def _feedback(self, sid, rid, valor="up", **extra):
        return self.cliente.post("/api/feedback", json={
            "session_id": sid, "request_id": rid, "valor": valor, **extra})

    def test_feedback_valido_devuelve_204(self) -> None:
        sid, rid = self.nueva_sesion(), str(uuid.uuid4())
        self.turno(sid, rid=rid)
        r = self._feedback(sid, rid, "up", comentario="muy bien")
        self.assertEqual(r.status_code, 204)
        self.assertEqual(r.content, b"")
        fila = json.loads(self.registro.ruta_feedback.read_text().splitlines()[0])
        self.assertEqual((fila["valor"], fila["comentario"]), ("up", "muy bien"))

    def test_un_turno_ajeno_o_inexistente_es_404(self) -> None:
        sid = self.nueva_sesion()
        self.assertEqual(self._feedback(sid, str(uuid.uuid4())).status_code, 404)

    def test_valores_y_comentarios_invalidos(self) -> None:
        sid, rid = self.nueva_sesion(), str(uuid.uuid4())
        self.turno(sid, rid=rid)
        self.assertEqual(self._feedback(sid, rid, "meh").status_code, 400)
        self.assertEqual(self._feedback(sid, rid, "up", comentario="a" * 501).status_code, 400)
        self.assertEqual(self._feedback(sid, rid, "up", comentario="a" * 500).status_code, 204)

    def test_un_down_con_consentimiento_deja_un_candidato_para_revision(self) -> None:
        sid, rid = self.nueva_sesion(), str(uuid.uuid4())
        self.turno(sid, "texto que clasifico mal", rid=rid, cuerpo={"consentimiento_registro": True})
        self._feedback(sid, rid, "down")
        c = json.loads(self.registro.ruta_candidatos.read_text().splitlines()[0])
        self.assertEqual(c["texto"], "texto que clasifico mal")
        self.assertEqual(c["intencion_predicha"], "faq")

    def test_un_down_sin_consentimiento_no_guarda_el_texto(self) -> None:
        sid, rid = self.nueva_sesion(), str(uuid.uuid4())
        self.turno(sid, "texto privado", rid=rid)
        self._feedback(sid, rid, "down")
        self.assertFalse(self.registro.ruta_candidatos.exists())

    def test_el_comentario_con_html_se_guarda_tal_cual_como_dato(self) -> None:
        sid, rid = self.nueva_sesion(), str(uuid.uuid4())
        self.turno(sid, rid=rid)
        self._feedback(sid, rid, "down", comentario="<script>alert(1)</script>")
        fila = json.loads(self.registro.ruta_feedback.read_text().splitlines()[0])
        self.assertEqual(fila["comentario"], "<script>alert(1)</script>")


class MetadatosTest(Base):
    def test_la_causa_de_un_escalamiento_llega_al_navegador(self) -> None:
        escala = contrato_simulado(escalar_a_humano=True, motivo_escalamiento="m", tipo="escalamiento")
        escala["metadatos"]["causa"] = "error_llm"
        self.comportamiento = lambda t, h, c: escala
        j = self.turno(self.nueva_sesion()).json()
        self.assertEqual(j["contrato"]["metadatos"]["causa"], "error_llm")
        self.assertTrue(j["contrato"]["escalar_a_humano"])

    def test_los_numeros_de_numpy_se_serializan(self) -> None:
        import numpy as np

        c = contrato_simulado()
        c["_trazas"]["router"]["confianza"] = np.float32(0.87)
        self.comportamiento = lambda t, h, ctx: c
        j = self.turno(self.nueva_sesion()).json()
        self.assertAlmostEqual(j["trazas"]["router"]["confianza"], 0.87, places=2)


class ReservasConcurrentesPorApiTest(ConBaseTemporal):
    """20 solicitudes simultaneas contra la logica real de reservas (R3, T6)."""

    def test_veinte_solicitudes_mezcladas_no_dejan_dobles_reservas_ni_codigos_repetidos(self) -> None:
        fecha = "2026-10-20"
        flota = [e["id"] for e in estructurado.listar_embarcaciones()]

        def responder(texto: str, historial, context_id: str) -> dict:
            if texto.startswith("consulta"):
                r = logica.consultar_disponibilidad(fecha, 4)
                msg = f"libres={r['total_disponibles']}"
            else:
                emb = flota[int(texto.split()[1]) % len(flota)]
                r = logica.bloquear_reserva(emb, fecha, "rosario", 4,
                                            clave_idempotencia=f"{context_id}|{emb}|{fecha}")
                msg = r.get("codigo_reserva") or r.get("error", "")
            acciones = ([{"tipo": "reserva_bloqueada", "codigo": r["codigo_reserva"],
                          "anticipo": r["anticipo_requerido"]}] if r.get("codigo_reserva") else [])
            c = contrato_simulado(msg, acciones=acciones)
            c["_trazas"]["saltos_a2a"] = []
            return c

        with tempfile.TemporaryDirectory() as tmp:
            limites = dataclasses.replace(
                seguridad.leer_limites(), concurrencia=8, cola=20, timeout_turno_s=30,
                turnos_por_minuto_sesion=1000, turnos_por_minuto_ip=1000, sesiones_por_ip_hora=1000)
            srv = web.Servicio(limites, Registro(Path(tmp), salida=io.StringIO()), responder)
            with TestClient(web.crear_app(srv, arrancar=False)) as cliente:
                sesiones = [cliente.post("/api/sesion").json()["session_id"] for _ in range(20)]
                textos = [f"reservar {i % 12}" if i % 3 else f"consulta {i}" for i in range(20)]
                respuestas: list = [None] * 20
                arranque = threading.Barrier(20)

                def uno(i: int) -> None:
                    arranque.wait()
                    respuestas[i] = cliente.post("/api/turno", json={
                        "session_id": sesiones[i], "texto": textos[i],
                        "request_id": str(uuid.uuid4())})

                hilos = [threading.Thread(target=uno, args=(i,)) for i in range(20)]
                for h in hilos:
                    h.start()
                for h in hilos:
                    h.join(60)

        self.assertTrue(all(r is not None and r.status_code == 200 for r in respuestas),
                        [getattr(r, "text", r) for r in respuestas if r is None or r.status_code != 200])
        codigos = [a["codigo"] for r in respuestas for a in r.json()["contrato"]["acciones"]]
        self.assertEqual(len(codigos), len(set(codigos)), "codigos repetidos")

        con = estructurado.conexion()
        try:
            vivas = con.execute(
                "SELECT embarcacion_id, fecha, COUNT(*) FROM reservas "
                "WHERE estado IN ('confirmada','bloqueada') GROUP BY embarcacion_id, fecha "
                "HAVING COUNT(*) > 1").fetchall()
        finally:
            con.close()
        self.assertEqual(vivas, [], "doble reserva")
        reservas_pedidas = sum(1 for t in textos if t.startswith("reservar"))
        distintas = len({int(t.split()[1]) % len(flota) for t in textos if t.startswith("reservar")})
        self.assertEqual(len(codigos), distintas)
        self.assertLessEqual(len(codigos), reservas_pedidas)


class ArranqueTest(unittest.TestCase):
    def test_los_valores_del_demo_solo_se_aplican_si_el_entorno_no_los_fija(self) -> None:
        viejos = (config.FECHA_HOY, config.TTL_RESERVA_MINUTOS, config.MAX_RESERVAS_POR_SESION,
                  config.MAX_RESERVAS_BLOQUEADAS, config.APRENDIZAJE_ACTIVO)
        self.addCleanup(lambda: (setattr(config, "FECHA_HOY", viejos[0]),
                                 setattr(config, "TTL_RESERVA_MINUTOS", viejos[1]),
                                 setattr(config, "MAX_RESERVAS_POR_SESION", viejos[2]),
                                 setattr(config, "MAX_RESERVAS_BLOQUEADAS", viejos[3]),
                                 setattr(config, "APRENDIZAJE_ACTIVO", viejos[4])))
        with mock.patch.dict("os.environ", {}, clear=True):
            web.aplicar_valores_del_demo()
            self.assertEqual(config.FECHA_HOY, "hoy")
            self.assertEqual(config.TTL_RESERVA_MINUTOS, 30)
            self.assertEqual(config.MAX_RESERVAS_POR_SESION, 2)
            self.assertFalse(config.APRENDIZAJE_ACTIVO)
        config.MAX_RESERVAS_POR_SESION = 7
        with mock.patch.dict("os.environ", {"VM_MAX_RESERVAS_SESION": "7"}, clear=True):
            web.aplicar_valores_del_demo()
            self.assertEqual(config.MAX_RESERVAS_POR_SESION, 7)

    def _estado_al_primer_pedido(self, entorno: dict[str, str]) -> tuple[str, bool]:
        """Devuelve (estado visto al abrir el servicio, si arrancar ya habia terminado)."""
        terminado = threading.Event()

        class Falso(web.Servicio):
            def arrancar(self) -> None:
                time.sleep(0.3)
                self.estado = "listo"
                terminado.set()

        with tempfile.TemporaryDirectory() as tmp:
            srv = Falso(seguridad.leer_limites(), Registro(Path(tmp), salida=io.StringIO()))
            # `/api/salud` solo dice "listo" si los agentes responden y hay clave de Gemini.
            # Se fijan ambos para que la prueba no dependa de un servicio local encendido.
            with mock.patch.dict("os.environ", entorno), \
                    mock.patch("core.a2a.lanzador.detener"), \
                    mock.patch("core.a2a.lanzador._responde", return_value=True), \
                    mock.patch("core.llm.gemini.modo_degradado", return_value=False):
                with TestClient(web.crear_app(srv, arrancar=True)) as cliente:
                    visto = cliente.get("/api/salud").json()["estado"]
                    visto_terminado = terminado.is_set()
                    terminado.wait(2)
        return visto, visto_terminado

    def test_arranque_sincrono_termina_antes_de_atender(self) -> None:
        estado, terminado = self._estado_al_primer_pedido({"VM_ARRANQUE_SINCRONO": "1"})
        self.assertTrue(terminado)
        self.assertEqual(estado, "listo")

    def test_arranque_en_hilo_deja_atender_mientras_se_prepara(self) -> None:
        estado, terminado = self._estado_al_primer_pedido({"VM_ARRANQUE_SINCRONO": ""})
        self.assertFalse(terminado)
        self.assertEqual(estado, "iniciando")


if __name__ == "__main__":
    unittest.main()
