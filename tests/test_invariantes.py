"""Pruebas de las invariantes del build-spec (seccion 4.5).

Se escribieron ANTES de tocar los modulos que las contienen y deben seguir
pasando despues. No usan red ni clave de API.

    python -m unittest tests.test_invariantes -v
"""

from __future__ import annotations

import json
import time
import unittest
from unittest import mock

from core import config
from core.agentes.conocimiento import rag
from core.agentes.disponibilidad import logica
from core.almacen import estructurado
from core.llm import gemini
from core.orquestador import contrato, grafo
from tests.base_db import ConBaseTemporal

FECHA_BAJA = "2026-10-20"
FECHA_ALTA = "2026-12-20"


def _tarifas() -> tuple[list[dict], list[dict], float]:
    flota = json.loads((config.RUTA_DATA / "flota.json").read_text(encoding="utf-8"))
    rutas = json.loads((config.RUTA_DATA / "rutas.json").read_text(encoding="utf-8"))
    return flota["embarcaciones"], rutas["rutas"], float(rutas["descuento_temporada_baja"])


class CapacidadTest(ConBaseTemporal):
    def test_cotizar_rechaza_mas_pasajeros_que_la_capacidad(self) -> None:
        for emb in estructurado.listar_embarcaciones():
            r = logica.cotizar(emb["id"], FECHA_BAJA, "rosario", emb["capacidad"] + 1)
            self.assertFalse(r["ok"], emb["id"])

    def test_cotizar_acepta_exactamente_la_capacidad(self) -> None:
        emb = estructurado.listar_embarcaciones()[0]
        r = logica.cotizar(emb["id"], FECHA_BAJA, "rosario", emb["capacidad"])
        self.assertTrue(r["ok"], r)

    def test_la_disponibilidad_nunca_ofrece_una_embarcacion_pequena(self) -> None:
        for pax in range(1, 17):
            r = logica.consultar_disponibilidad(FECHA_BAJA, pax)
            self.assertTrue(r["ok"])
            for e in r["disponibles"]:
                self.assertGreaterEqual(e["capacidad"], pax, (pax, e["id"]))

    def test_la_pesca_admite_como_maximo_seis(self) -> None:
        self.assertFalse(logica.consultar_disponibilidad(FECHA_BAJA, 7, "pesca")["ok"])
        self.assertTrue(logica.consultar_disponibilidad(FECHA_BAJA, 6, "pesca")["ok"])
        self.assertFalse(logica.cotizar("VM-06", FECHA_BAJA, "pesca", 7)["ok"])


class PreciosTest(ConBaseTemporal):
    def test_todo_precio_sale_de_la_tabla_de_tarifas(self) -> None:
        flota, rutas, descuento = _tarifas()
        for emb in flota:
            for ruta in rutas:
                for fecha, alta in ((FECHA_BAJA, False), (FECHA_ALTA, True)):
                    base = (emb["tarifa_dia_completo"] if ruta["duracion"] == "dia_completo"
                            else emb["tarifa_medio_dia"])
                    esperado = int(round(base * (1 if alta else 1 - descuento)))
                    r = logica.cotizar(emb["id"], fecha, ruta["codigo"], 1)
                    self.assertTrue(r["ok"], r)
                    self.assertEqual(r["valor_alquiler"], esperado,
                                     (emb["id"], ruta["codigo"], fecha))
                    self.assertEqual(r["anticipo_50"], int(round(esperado * 0.5)))


class ReservaTest(ConBaseTemporal):
    def _bloquear(self, clave: str, emb: str = "VM-01", fecha: str = FECHA_BAJA) -> dict:
        return logica.bloquear_reserva(emb, fecha, "rosario", 4, clave_idempotencia=clave)

    def test_reservar_dos_veces_con_la_misma_clave_no_duplica(self) -> None:
        antes = self.contar_reservas()
        a = self._bloquear("ctx-1|VM-01|" + FECHA_BAJA)
        b = self._bloquear("ctx-1|VM-01|" + FECHA_BAJA)
        self.assertTrue(a["ok"] and b["ok"])
        self.assertEqual(a["codigo_reserva"], b["codigo_reserva"])
        self.assertTrue(b["reutilizada_por_idempotencia"])
        self.assertEqual(self.contar_reservas(), antes + 1)

    def test_no_hay_dos_reservas_vivas_de_la_misma_embarcacion_y_fecha(self) -> None:
        self.assertTrue(self._bloquear("ctx-1|a")["ok"])
        segunda = self._bloquear("ctx-2|b")
        self.assertFalse(segunda["ok"])
        self.assertEqual(self.contar_reservas(("bloqueada",)), 1)

    def test_el_indice_unico_existe_en_la_base(self) -> None:
        con = estructurado.conexion()
        try:
            nombres = {f["name"] for f in con.execute("PRAGMA index_list('reservas')")}
        finally:
            con.close()
        self.assertIn("ux_reserva_embarcacion_fecha", nombres)
        self.assertIn("ux_reserva_idempotencia", nombres)

    def test_distintas_embarcaciones_el_mismo_dia_si_se_pueden_reservar(self) -> None:
        self.assertTrue(self._bloquear("k1", "VM-01")["ok"])
        self.assertTrue(self._bloquear("k2", "VM-02")["ok"])


def _fragmento(n: int = 1) -> rag.Fragmento:
    return rag.Fragmento(chunk_id=f"c{n}", texto="Texto de la politica.", fuente="politicas.md",
                         titulo="Politica", seccion="Reembolsos", version="v2", fecha="2026-01-01")


class ConocimientoTest(unittest.TestCase):
    def _responder(self, datos_llm: dict) -> rag.RespuestaConocimiento:
        diag = {"mejor_denso": 0.9, "mejor_lexico": 9.0}
        llm = gemini.RespuestaLLM(texto="{}", modelo="falso")
        with mock.patch.object(rag, "recuperar", return_value=([_fragmento()], diag)), \
                mock.patch.object(gemini, "generar_json", return_value=(datos_llm, llm)):
            return rag.responder("cuanto devuelven")

    def test_una_respuesta_afirmativa_sin_cita_se_convierte_en_abstencion(self) -> None:
        r = self._responder({"respuesta": "Devolvemos todo.", "abstencion": False,
                             "fragmentos_usados": [], "confianza": 0.9})
        self.assertTrue(r.abstencion)
        self.assertEqual(r.respuesta, "")
        self.assertEqual(r.citas, [])

    def test_las_citas_las_construye_el_codigo_no_el_modelo(self) -> None:
        r = self._responder({
            "respuesta": "Se devuelve el 50 %.", "abstencion": False,
            "fragmentos_usados": [1], "confianza": 0.9,
            "citas": [{"fuente": "inventada.md", "titulo": "Falsa"}],
        })
        self.assertFalse(r.abstencion)
        self.assertEqual(r.citas, [_fragmento().cita()])

    def test_un_indice_de_fragmento_inexistente_no_crea_citas(self) -> None:
        r = self._responder({"respuesta": "Algo", "abstencion": False,
                             "fragmentos_usados": [7, 0, -1, "1"], "confianza": 0.9})
        self.assertTrue(r.abstencion)


class ContratoTest(unittest.TestCase):
    CAMPOS = {"tipo", "mensaje", "citas", "acciones", "escalar_a_humano",
              "motivo_escalamiento", "metadatos"}

    def test_la_forma_del_contrato_no_cambia(self) -> None:
        self.assertEqual(set(contrato.construir(mensaje="hola")), self.CAMPOS)
        self.assertEqual(set(contrato.escalamiento("x")), self.CAMPOS)

    def test_validar_detecta_un_escalamiento_sin_motivo(self) -> None:
        c = contrato.construir(mensaje="", escalar_a_humano=True)
        ok, problemas = contrato.validar(c)
        self.assertFalse(ok)
        self.assertTrue(problemas)

    def test_un_contrato_normal_es_valido(self) -> None:
        self.assertTrue(contrato.validar(contrato.construir(mensaje="hola"))[0])


class EscalamientoTest(unittest.TestCase):
    def _estado(self, saltos: list[dict], datos: dict | None = None) -> dict:
        return {"texto": "hola", "saltos": saltos, "datos_agente": datos or {"ok": True},
                "prediccion": {"intencion": "precio", "confianza": 0.9, "metodo": "clasificador"},
                "t_inicio": time.perf_counter(), "historial": []}

    def test_un_salto_a2a_fallido_escala_a_un_humano_sin_llamar_al_llm(self) -> None:
        salto = {"agente": "disponibilidad", "habilidad": "cotizar", "ok": False, "ms": 3.0}
        with mock.patch.object(gemini, "generar", side_effect=AssertionError("no debe llamarse")):
            c = grafo.nodo_componer(self._estado([salto]))["contrato"]
        self.assertTrue(c["escalar_a_humano"])
        self.assertEqual(c["motivo_escalamiento"], config.MOTIVO_ESCALAMIENTO["error_herramienta"])
        self.assertEqual(c["tipo"], "escalamiento")

    def test_un_error_del_llm_al_redactar_escala(self) -> None:
        resp = gemini.RespuestaLLM(texto="", modelo="falso", error="Timeout")
        with mock.patch.object(gemini, "generar", return_value=resp):
            c = grafo.nodo_componer(self._estado([]))["contrato"]
        self.assertTrue(c["escalar_a_humano"])

    def test_una_abstencion_del_agente_escala_por_baja_confianza(self) -> None:
        estado = self._estado([], {"abstencion": True, "motivo_abstencion": "sin datos"})
        c = grafo.nodo_componer(estado)["contrato"]
        self.assertTrue(c["escalar_a_humano"])
        self.assertEqual(c["motivo_escalamiento"], config.MOTIVO_ESCALAMIENTO["baja_confianza"])

    def test_el_contrato_de_salida_siempre_es_valido(self) -> None:
        resp = gemini.RespuestaLLM(texto="Hola, claro que si.", modelo="falso")
        with mock.patch.object(gemini, "generar", return_value=resp):
            c = grafo.nodo_componer(self._estado([]))["contrato"]
        self.assertTrue(c["metadatos"]["contrato_valido"])
        self.assertFalse(c["escalar_a_humano"])


if __name__ == "__main__":
    unittest.main()
