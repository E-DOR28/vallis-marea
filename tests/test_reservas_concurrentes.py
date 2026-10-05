"""Pruebas de reservas bajo concurrencia, topes, vencimiento y fecha de referencia.

Cubren los cambios de R3 sobre el almacen estructurado. No usan red.

    python -m unittest tests.test_reservas_concurrentes -v
"""

from __future__ import annotations

import re
import threading
import unittest
from datetime import date
from unittest import mock

from core import config
from core.agentes.disponibilidad import logica
from core.almacen import estructurado
from tests.base_db import ConBaseTemporal

FECHA = "2026-10-20"


def _en_paralelo(n: int, trabajo) -> list:
    """Lanza `n` hilos que arrancan a la vez y devuelve sus resultados en orden."""
    salida: list = [None] * n
    arranque = threading.Barrier(n)

    def correr(i: int) -> None:
        arranque.wait()
        salida[i] = trabajo(i)

    hilos = [threading.Thread(target=correr, args=(i,)) for i in range(n)]
    for h in hilos:
        h.start()
    for h in hilos:
        h.join(timeout=60)
    return salida


def _reservar(emb: str, fecha: str, clave: str, **extra):
    return estructurado.crear_reserva(
        embarcacion_id=emb, fecha=fecha, ruta="rosario", pasajeros=4,
        cliente=None, valor_total=1_000_000, clave_idempotencia=clave, **extra,
    )


class ConcurrenciaTest(ConBaseTemporal):
    def test_veinte_solicitudes_por_la_misma_embarcacion_y_fecha_dejan_una_sola_reserva(self) -> None:
        r = _en_paralelo(20, lambda i: _reservar("VM-01", FECHA, f"s{i}|VM-01|{FECHA}"))
        self.assertEqual(sum(1 for ok, _, _ in r if ok), 1, r)
        self.assertEqual(self.contar_reservas(("bloqueada",)), 1)

    def test_los_codigos_nunca_se_repiten_bajo_concurrencia(self) -> None:
        # 8 embarcaciones x fechas distintas: todas las reservas son validas.
        flota = [e["id"] for e in estructurado.listar_embarcaciones()]
        pares = [(emb, f"2026-11-{dia:02d}") for dia in range(1, 6) for emb in flota]
        r = _en_paralelo(len(pares), lambda i: _reservar(pares[i][0], pares[i][1], f"c{i}"))
        self.assertTrue(all(ok for ok, _, _ in r), [x for x in r if not x[0]])
        codigos = [c for _, c, _ in r]
        self.assertEqual(len(set(codigos)), len(codigos))
        for c in codigos:
            self.assertRegex(c, r"^VM-\d{4}-\d{4,}$")

    def test_la_misma_clave_simultanea_crea_una_reserva(self) -> None:
        r = _en_paralelo(20, lambda i: _reservar("VM-02", FECHA, "misma|VM-02|x"))
        self.assertTrue(all(ok for ok, _, _ in r), r)
        self.assertEqual(len({c for _, c, _ in r}), 1)
        self.assertEqual(sum(1 for _, _, reutilizada in r if not reutilizada), 1)
        self.assertEqual(self.contar_reservas(("bloqueada",)), 1)

    def test_bloquear_reserva_completo_en_paralelo_no_duplica(self) -> None:
        r = _en_paralelo(
            12, lambda i: logica.bloquear_reserva("VM-04", FECHA, "rosario", 6,
                                                   clave_idempotencia=f"u{i}|VM-04|{FECHA}")
        )
        self.assertEqual(sum(1 for x in r if x["ok"]), 1, r)
        self.assertEqual(self.contar_reservas(("bloqueada",)), 1)

    def test_el_codigo_no_se_reutiliza_tras_liberar(self) -> None:
        _, c1, _ = _reservar("VM-01", FECHA, "a|VM-01|x")
        estructurado.liberar_por_sesion("a")
        _, c2, _ = _reservar("VM-01", FECHA, "b|VM-01|x")
        self.assertNotEqual(c1, c2)


class TopesTest(ConBaseTemporal):
    def test_el_tope_por_sesion_se_respeta_aun_en_paralelo(self) -> None:
        flota = [e["id"] for e in estructurado.listar_embarcaciones()]
        r = _en_paralelo(
            len(flota),
            lambda i: _reservar(flota[i], FECHA, f"unica|{flota[i]}|{FECHA}", limite_sesion=2),
        )
        self.assertEqual(sum(1 for ok, _, _ in r if ok), 2, r)
        self.assertEqual(sum(1 for ok, m, _ in r if not ok and m == "limite_sesion"), len(flota) - 2)

    def test_el_tope_por_sesion_no_afecta_a_otras_sesiones(self) -> None:
        for emb in ("VM-01", "VM-02"):
            self.assertTrue(_reservar(emb, FECHA, f"A|{emb}|x", limite_sesion=2)[0])
        self.assertFalse(_reservar("VM-03", FECHA, "A|VM-03|x", limite_sesion=2)[0])
        self.assertTrue(_reservar("VM-03", FECHA, "B|VM-03|x", limite_sesion=2)[0])

    def test_el_tope_total(self) -> None:
        self.assertTrue(_reservar("VM-01", FECHA, "a|1|x", limite_total=2)[0])
        self.assertTrue(_reservar("VM-02", FECHA, "b|2|x", limite_total=2)[0])
        ok, motivo, _ = _reservar("VM-03", FECHA, "c|3|x", limite_total=2)
        self.assertFalse(ok)
        self.assertEqual(motivo, "limite_total")

    def test_una_reserva_liberada_no_cuenta_para_el_tope(self) -> None:
        _reservar("VM-01", FECHA, "A|VM-01|x", limite_sesion=1)
        self.assertFalse(_reservar("VM-02", FECHA, "A|VM-02|x", limite_sesion=1)[0])
        estructurado.liberar_por_sesion("A")
        self.assertTrue(_reservar("VM-02", FECHA, "A|VM-02|x", limite_sesion=1)[0])

    def test_las_reservas_confirmadas_de_la_semilla_no_cuentan(self) -> None:
        with mock.patch.object(config, "MAX_RESERVAS_BLOQUEADAS", 1):
            r = logica.bloquear_reserva("VM-01", FECHA, "rosario", 4, clave_idempotencia="s|1|x")
        self.assertTrue(r["ok"], r)

    def test_el_mensaje_de_tope_llega_al_agente_como_rechazo_de_negocio(self) -> None:
        with mock.patch.object(config, "MAX_RESERVAS_POR_SESION", 1):
            logica.bloquear_reserva("VM-01", FECHA, "rosario", 4, clave_idempotencia="s|VM-01|x")
            r = logica.bloquear_reserva("VM-02", FECHA, "rosario", 4, clave_idempotencia="s|VM-02|x")
        self.assertFalse(r["ok"])
        self.assertEqual(r["motivo"], "limite_sesion")


class VencimientoTest(ConBaseTemporal):
    def _envejecer(self, minutos: int) -> None:
        con = estructurado.conexion()
        try:
            con.execute("UPDATE reservas SET creada_en = datetime('now', ?) WHERE estado = 'bloqueada'",
                        (f"-{minutos} minutes",))
            con.commit()
        finally:
            con.close()

    def test_sin_ttl_nada_vence(self) -> None:
        _reservar("VM-01", FECHA, "a|VM-01|x")
        self._envejecer(10_000)
        self.assertEqual(estructurado.liberar_vencidas(0), 0)
        self.assertIn("VM-01", estructurado.embarcaciones_ocupadas(FECHA))

    def test_una_reserva_vencida_libera_la_embarcacion_y_su_clave(self) -> None:
        _, codigo, _ = _reservar("VM-01", FECHA, "a|VM-01|x")
        self._envejecer(45)
        with mock.patch.object(config, "TTL_RESERVA_MINUTOS", 30):
            self.assertNotIn("VM-01", estructurado.embarcaciones_ocupadas(FECHA))
        fila = estructurado.obtener_reserva(codigo)
        self.assertEqual(fila["estado"], "liberada")
        self.assertIsNone(fila["clave_idempotencia"])
        # Quien reintente la misma solicitud crea una reserva nueva, no recibe la muerta.
        ok, nuevo, reutilizada = _reservar("VM-01", FECHA, "a|VM-01|x")
        self.assertTrue(ok)
        self.assertFalse(reutilizada)
        self.assertNotEqual(nuevo, codigo)

    def test_una_reserva_reciente_no_vence(self) -> None:
        _reservar("VM-01", FECHA, "a|VM-01|x")
        self._envejecer(5)
        with mock.patch.object(config, "TTL_RESERVA_MINUTOS", 30):
            self.assertIn("VM-01", estructurado.embarcaciones_ocupadas(FECHA))

    def test_las_reservas_confirmadas_nunca_vencen(self) -> None:
        con = estructurado.conexion()
        try:
            con.execute("UPDATE reservas SET creada_en = datetime('now', '-9999 minutes')")
            con.commit()
        finally:
            con.close()
        antes = self.contar_reservas(("confirmada",))
        self.assertGreater(antes, 0)
        self.assertEqual(estructurado.liberar_vencidas(1), 0)
        self.assertEqual(self.contar_reservas(("confirmada",)), antes)


class FechaDeReferenciaTest(unittest.TestCase):
    def test_por_defecto_es_la_fecha_fija_del_demo(self) -> None:
        with mock.patch.object(config, "FECHA_HOY", ""):
            self.assertEqual(logica.fecha_hoy(), date(2026, 9, 13))
            self.assertEqual(logica.interpretar_fecha("manana"), "2026-09-14")

    def test_una_fecha_iso_configurada(self) -> None:
        with mock.patch.object(config, "FECHA_HOY", "2026-11-02"):
            self.assertEqual(logica.fecha_hoy(), date(2026, 11, 2))
            self.assertEqual(logica.interpretar_fecha("manana"), "2026-11-03")

    def test_hoy_usa_el_reloj_real_de_colombia(self) -> None:
        with mock.patch.object(config, "FECHA_HOY", "HOY"):
            self.assertRegex(logica.fecha_hoy().isoformat(), r"^\d{4}-\d{2}-\d{2}$")

    def test_un_valor_mal_escrito_falla_en_vez_de_cambiar_la_fecha_en_silencio(self) -> None:
        with mock.patch.object(config, "FECHA_HOY", "pasado-manana"):
            with self.assertRaises(ValueError):
                logica.fecha_hoy()


class FechaPasadaTest(ConBaseTemporal):
    def test_no_se_consulta_ni_se_cotiza_ni_se_reserva_una_fecha_pasada(self) -> None:
        with mock.patch.object(config, "FECHA_HOY", "2026-10-04"):
            self.assertFalse(logica.consultar_disponibilidad("2026-10-03", 4)["ok"])
            self.assertFalse(logica.cotizar("VM-01", "2026-10-03", "rosario", 4)["ok"])
            r = logica.bloquear_reserva("VM-01", "2026-10-03", "rosario", 4)
            self.assertFalse(r["ok"])
            self.assertEqual(self.contar_reservas(("bloqueada",)), 0)

    def test_hoy_mismo_si_se_puede(self) -> None:
        with mock.patch.object(config, "FECHA_HOY", "2026-10-04"):
            self.assertTrue(logica.consultar_disponibilidad("2026-10-04", 4)["ok"])

    def test_cotizar_con_formato_invalido_no_lanza_excepcion(self) -> None:
        r = logica.cotizar("VM-01", "manana", "rosario", 4)
        self.assertFalse(r["ok"])


class FormatoDeCodigoTest(ConBaseTemporal):
    def test_el_primer_codigo_nuevo_sigue_a_la_semilla(self) -> None:
        total = self.contar_reservas()
        _, codigo, _ = _reservar("VM-01", FECHA, "a|1|x")
        self.assertRegex(codigo, rf"^VM-\d{{4}}-{1000 + total + 1}$")


if __name__ == "__main__":
    unittest.main()
