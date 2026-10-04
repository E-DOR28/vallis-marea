"""Dataset de entrenamiento del router de intencion.

Se combinan dos fuentes, y la distincion importa para el informe:

1. **En dominio (sintetico).** Frases de clientes de Vallis Marea escritas por
   el equipo a partir de como se pregunta por WhatsApp en Cartagena. Son
   sinteticas y hay que declararlo: no son conversaciones reales de produccion.

2. **Fuera de dominio (publico y real).** MASSIVE en espanol, via el dataset
   `mteb/amazon_massive_intent`, config `es`. Son enunciados humanos reales de
   un asistente de proposito general (alarmas, musica, clima...). Alimentan la
   clase `otro`.

Ese segundo bloque no es decorativo. Un router entrenado unicamente con frases
del dominio clasifica con alta confianza cualquier cosa que le llegue, incluido
"pon musica", porque nunca vio un ejemplo de algo que no le corresponde. Las
negativas reales son lo que le ensena a decir "esto no es mio".
"""

from __future__ import annotations

import json
import random
import time
from typing import Any

import httpx

from core import config

RUTA_DATASET = config.RUTA_APRENDIZAJE / "dataset_router.json"

# ---------------------------------------------------------------------------
# En dominio
# ---------------------------------------------------------------------------
FRASES_DOMINIO: dict[str, list[str]] = {
    "saludo": [
        "hola", "buenas", "buenos dias", "buenas tardes", "hola buenas noches",
        "hola, buen dia", "que tal", "hola equipo", "hey", "saludos",
        "hola, me pueden ayudar?", "buenas, una pregunta", "hola! como estan",
        "buen dia, tienen atencion ahora?", "hola, quien me atiende",
        "gracias, muy amables", "listo, muchas gracias", "ok gracias",
        "perfecto, gracias por la info", "hasta luego", "chao, gracias",
    ],
    "disponibilidad": [
        "tienen lancha disponible el sabado",
        "hay disponibilidad para el 15 de octubre",
        "queda algo libre para manana",
        "tienen cupo para 12 personas el domingo",
        "esta libre alguna lancha el 3 de octubre",
        "hay disponibilidad en semana santa",
        "que lanchas tienen libres el proximo viernes",
        "me sirve el sabado, hay algo",
        "tienen algo para 8 personas ese dia",
        "quedan lanchas para el puente",
        "hay disponibilidad para islas del rosario el jueves",
        "tienen lancha libre pasado manana",
        "para el 24 de diciembre tienen algo",
        "hay cupo el fin de semana",
        "que disponibilidad tienen para 16 personas",
        "esta ocupada la lancha grande el sabado",
        "tienen disponible algo para maniana en la tarde",
        "hay lancha para el 10 de octubre a baru",
    ],
    "precio": [
        "cuanto cuesta el paseo a islas del rosario",
        "cual es el precio para 10 personas",
        "cuanto vale alquilar una lancha por el dia",
        "me pasas la tarifa de cholon",
        "cuanto me sale medio dia",
        "que precio tiene la lancha de 12 personas",
        "cuanto cobran por el atardecer",
        "me cotizas un dia completo a baru",
        "cuanto seria el total por 8 personas",
        "cual es el valor del paseo de pesca",
        "que vale la mas economica",
        "cuanto cuesta con todo incluido",
        "me das un estimado de precio",
        "cuanto es el anticipo",
        "el precio incluye almuerzo",
        "cuanto vale el yate grande",
        "tienen descuento en temporada baja",
        "cuanto me cobran por ir a playa blanca",
    ],
    "reserva": [
        "quiero reservar para el sabado",
        "confirmo la reserva",
        "listo, separame esa lancha",
        "quiero apartar el paseo a baru",
        "hagamosle, reservo el domingo",
        "si, confirmo la manglar para el 10",
        "quiero hacer la reserva ya",
        "como hago para reservar",
        "separame el cupo por favor",
        "ya hice la transferencia, confirmo",
        "quiero bloquear esa fecha",
        "dale, reservemos",
        "necesito reservar para 14 personas",
        "puedo dejar el anticipo hoy",
        "quiero cancelar mi reserva",
        "necesito cambiar la fecha de mi reserva",
        "quiero reprogramar el paseo",
    ],
    "politica": [
        "cuanto me devuelven si cancelo",
        "cual es la politica de cancelacion",
        "si llueve me devuelven la plata",
        "que pasa si no puedo ir",
        "puedo cambiar la fecha sin costo",
        "hay reembolso si cancelo un dia antes",
        "que documentos necesito para abordar",
        "los ninos necesitan documento",
        "puedo ir embarazada",
        "que pasa si llego tarde al muelle",
        "es obligatorio el chaleco salvavidas",
        "puedo llevar mascotas",
        "se puede fumar en la lancha",
        "que pasa si cierran el puerto",
        "hay seguro para los pasajeros",
        "a que hora debo llegar al muelle",
        "puedo cancelar por enfermedad",
        "cuantas veces puedo reprogramar",
    ],
    "faq": [
        "que incluye el alquiler",
        "el precio es por persona o por lancha",
        "puedo llevar mi propia comida",
        "hay bano en la lancha",
        "como se paga",
        "aceptan tarjeta",
        "que debo llevar al paseo",
        "hay que pagar entrada al parque",
        "cuanto es el impuesto de muelle",
        "con cuanta anticipacion debo reservar",
        "de donde sale la lancha",
        "cuantas horas dura el paseo",
        "incluye almuerzo",
        "puedo llevar cerveza",
        "hay nevera a bordo",
        "el paseo incluye snorkel",
        "donde queda el punto de encuentro",
        "aceptan nequi o daviplata",
    ],
    "ruta_turistica": [
        "que hay para hacer en islas del rosario",
        "como es el paseo a playa blanca",
        "cuentame del plan a cholon",
        "que se ve en el recorrido por la bahia",
        "que incluye la ruta a tierra bomba",
        "a que hora sale el paseo a baru",
        "que paseos ofrecen",
        "cuales son los destinos disponibles",
        "como es el atardecer en la bahia",
        "que tal el plan de pesca deportiva",
        "cuanto se demora llegar a las islas",
        "el fuerte de san fernando se puede visitar",
        "que rutas hacen",
        "que playa recomiendan",
        "el paseo a cholon tiene playa",
        "cuanto dura la travesia a rosario",
    ],
    "recomendacion": [
        "que me recomiendan para ir con ninos",
        "somos 8 amigos y queremos rumba, que nos sugieren",
        "cual lancha me sirve para un aniversario",
        "que plan me recomiendan para una despedida de soltero",
        "vamos en familia, que sugieren",
        "quiero algo economico y tranquilo, que me ofrecen",
        "cual es la mejor opcion para 14 personas",
        "que me conviene para un dia con mi pareja",
        "necesito algo con bano para adultos mayores",
        "que lancha es la mejor para fotos",
        "para un grupo corporativo que recomiendan",
        "cual plan es el mas popular",
        "que me sugieren para primera vez en cartagena",
        "quiero sorprender a mi esposa, ideas",
        "que opcion tienen para un cumpleanos",
    ],
}


# ---------------------------------------------------------------------------
# Fuera de dominio: MASSIVE (es) via datasets-server de Hugging Face
# ---------------------------------------------------------------------------
_URL_HF = "https://datasets-server.huggingface.co/rows"
DATASET_PUBLICO = {
    "dataset": "mteb/amazon_massive_intent",
    "config": "es",
    "split": "train",
    "cita": (
        "FitzGerald et al. (2022). MASSIVE: A 1M-Example Multilingual Natural "
        "Language Understanding Dataset. Accedido via mteb/amazon_massive_intent (es)."
    ),
}

# Intenciones de MASSIVE que SI se parecen a algo del dominio y por tanto no
# sirven como negativas limpias (transporte, reservas, recomendaciones).
_INTENCIONES_AMBIGUAS = {
    "transport_query", "transport_ticket", "transport_taxi", "transport_traffic",
    "recommendation_locations", "recommendation_events", "recommendation_movies",
    "qa_currency", "datetime_query", "general_quirky",
}


def _get_con_reintento(params: dict[str, Any], timeout: float, intentos: int = 3):
    """GET con reintento: la resolucion DNS de este endpoint falla a ratos."""
    ultimo: Exception | None = None
    for i in range(intentos):
        try:
            r = httpx.get(_URL_HF, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as exc:
            ultimo = exc
            time.sleep(1.5 * (i + 1))
    raise ultimo  # type: ignore[misc]


def descargar_fuera_de_dominio(n: int = 400, timeout: float = 30.0) -> dict[str, Any]:
    """Baja enunciados reales de MASSIVE (es) para la clase `otro`."""
    filas: list[dict[str, Any]] = []
    intentos: list[str] = []
    try:
        # El endpoint pagina de a 100 como maximo.
        for offset in range(0, n, 100):
            r = _get_con_reintento(
                params={**{k: v for k, v in DATASET_PUBLICO.items() if k != "cita"},
                        "offset": offset, "length": 100},
                timeout=timeout,
            )
            for fila in r.json().get("rows", []):
                d = fila.get("row", {})
                texto = (d.get("text") or "").strip()
                etiqueta = d.get("label_text") or d.get("label") or ""
                if not texto:
                    continue
                intentos.append(etiqueta)
                if etiqueta in _INTENCIONES_AMBIGUAS:
                    continue
                filas.append({"texto": texto, "intencion_original": etiqueta})
    except Exception as exc:
        return {"ok": False, "error": f"{exc.__class__.__name__}: {exc}", "filas": []}

    return {
        "ok": True,
        "filas": filas,
        "descartadas_por_ambiguedad": len(intentos) - len(filas),
        "intenciones_vistas": len(set(intentos)),
    }


# ---------------------------------------------------------------------------
def construir_dataset(
    n_fuera_dominio: int = 400, semilla: int = 7, forzar_descarga: bool = False
) -> dict[str, Any]:
    """Arma el dataset completo y lo guarda en data/aprendizaje/."""
    if RUTA_DATASET.exists() and not forzar_descarga:
        return json.loads(RUTA_DATASET.read_text(encoding="utf-8"))

    rng = random.Random(semilla)
    ejemplos: list[dict[str, Any]] = []

    for intencion, frases in FRASES_DOMINIO.items():
        for f in frases:
            ejemplos.append({"texto": f, "intencion": intencion, "origen": "dominio_sintetico"})

    externo = descargar_fuera_de_dominio(n_fuera_dominio)
    if externo["ok"]:
        muestras = externo["filas"]
        rng.shuffle(muestras)
        # Se limita el tamano de `otro` para no desbalancear brutalmente: como
        # maximo el doble de la clase de dominio mas grande.
        tope = max(len(v) for v in FRASES_DOMINIO.values()) * 2
        for fila in muestras[:tope]:
            ejemplos.append({
                "texto": fila["texto"],
                "intencion": "otro",
                "origen": "massive_es",
                "intencion_original": fila["intencion_original"],
            })

    rng.shuffle(ejemplos)
    datos = {
        "ejemplos": ejemplos,
        "clases": sorted({e["intencion"] for e in ejemplos}),
        "conteo_por_clase": {
            c: sum(1 for e in ejemplos if e["intencion"] == c)
            for c in sorted({e["intencion"] for e in ejemplos})
        },
        "fuente_publica": DATASET_PUBLICO,
        "fuera_de_dominio_ok": externo["ok"],
        "fuera_de_dominio_error": externo.get("error"),
        "descartadas_por_ambiguedad": externo.get("descartadas_por_ambiguedad", 0),
    }
    RUTA_DATASET.write_text(
        json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return datos


if __name__ == "__main__":
    d = construir_dataset(forzar_descarga=True)
    print(f"Ejemplos: {len(d['ejemplos'])}")
    print(f"Dataset publico OK: {d['fuera_de_dominio_ok']} "
          f"(descartadas por ambiguedad: {d['descartadas_por_ambiguedad']})")
    for c, n in d["conteo_por_clase"].items():
        print(f"   {c:16} {n}")
