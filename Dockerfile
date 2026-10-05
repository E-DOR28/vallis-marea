# Imagen del servicio web de Vallis Marea.
#   python scripts/preparar_almacen.py     (una vez, genera build/almacen)
#   gcloud run deploy --source .           (o: docker build -t vallis-marea .)
FROM python:3.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ANONYMIZED_TELEMETRY=False \
    VM_ARRANQUE_SINCRONO=1 \
    PORT=8080

WORKDIR /app

COPY requirements-web.txt requirements.lock ./
RUN pip install -r requirements-web.txt -c requirements.lock

RUN useradd --system --uid 10001 --home-dir /app vm \
    && mkdir -p data/trazas data/aprendizaje data/evaluacion data/modelos \
    && chown -R vm:vm /app

COPY --chown=vm:vm core ./core
COPY --chown=vm:vm data/corpus ./data/corpus
COPY --chown=vm:vm data/flota.json data/rutas.json ./data/
COPY --chown=vm:vm data/modelos/router_intencion.npz ./data/modelos/
COPY --chown=vm:vm build/almacen ./data/almacen

USER vm

# Falla aqui, al construir, si algun archivo quedo fuera de la imagen (por ejemplo por un
# patron de .gcloudignore), en vez de fallar al arrancar la revision.
RUN python -c "import core.web.app, core.orquestador.grafo, core.router.predecir"

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os,urllib.request as u; u.urlopen('http://127.0.0.1:%s/api/salud' % os.environ.get('PORT','8080'), timeout=4)"

# Un solo proceso, sin recarga automatica: los agentes A2A corren en hilos y
# las sesiones viven en memoria.
CMD ["python", "-m", "core.web"]
