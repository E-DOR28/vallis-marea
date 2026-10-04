# Vallis Marea — Ecosistema de agentes interoperables (MCP + A2A)

Proyecto final de **SI7016 Procesamiento del Lenguaje Natural Aplicado**
Maestría en Ciencias de Datos y Analítica · Universidad EAFIT · 2026-2

**Equipo:** Elkin David Ortiz Rodriguez · Dilan Monsalve Monsalve · Juan David Trujillo Velez

---

## Qué es

Vallis Marea es un chatbot multimodal **en producción** para alquiler de lanchas en
Cartagena (AWS EC2, Chatwoot, WhatsApp vía Meta Cloud API, n8n, Postgres+pgvector). Hoy su
inteligencia vive en **un solo nodo de IA** dentro de n8n, que concentra clasificación,
conocimiento, disponibilidad y recomendación.

Este proyecto descompone ese monolito en un **ecosistema de agentes especializados** que
colaboran mediante protocolos abiertos:

| Agente | Rol | Expone |
|---|---|---|
| **Orquestador** | Enruta, mantiene contexto y delega. Sin lógica de negocio propia | cliente A2A |
| **Conocimiento** | RAG agentic híbrido con citación obligatoria y abstención | MCP + A2A |
| **Disponibilidad y Reservas** | Inventario, cotización y bloqueo idempotente | MCP + A2A |
| **Recomendador** | Afinidad semántica entre preferencias y catálogo | MCP + A2A |

Cada agente par publica **las mismas capacidades por dos vías**: MCP (para cualquier cliente
compatible) y A2A (para el orquestador). Una sola implementación, dos protocolos.

---

## Puesta en marcha

### 1. Entorno

```bash
python -m venv C:\envs\vallis_marea
C:\envs\vallis_marea\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Credenciales

Crear un archivo `.env` en la raíz del proyecto:

```
GOOGLE_API_KEY=tu_clave_de_google_ai_studio
```

Se obtiene gratis en <https://aistudio.google.com/apikey>.

> **Sin la clave el sistema arranca igual, en modo degradado**: embeddings locales por
> hashing y sin generación. Sirve para verificar la tubería completa, pero la calidad de las
> respuestas no es representativa y la interfaz lo advierte en rojo.

### 3. Indexar el corpus

```bash
C:\envs\vallis_marea\Scripts\python.exe -m core.ingesta.indexar --recrear
```

### 4. Entrenar el router de intención

```bash
C:\envs\vallis_marea\Scripts\python.exe -m core.router.entrenar
```

### 5. Levantar la aplicación

```bash
C:\envs\vallis_marea\Scripts\streamlit.exe run app.py
```

La interfaz levanta los tres agentes A2A sola. Para correrlos como procesos separados (como
en producción):

```bash
C:\envs\vallis_marea\Scripts\python.exe -m core.a2a.lanzador
```

### 6. Evaluación

```bash
C:\envs\vallis_marea\Scripts\python.exe -m core.evaluacion.suite
```

Resultados en `data/evaluacion/resultados.json`.

---

## Usar los agentes desde otro cliente MCP

Esto es lo que demuestra que la interoperabilidad es real y no decorativa: cualquier cliente
MCP puede usar estas capacidades sin saber nada de Vallis Marea.

```bash
C:\envs\vallis_marea\Scripts\python.exe -m core.agentes.servidores_mcp disponibilidad
```

Para registrarlo en un cliente MCP (por ejemplo Claude Desktop):

```json
{
  "mcpServers": {
    "vallis-disponibilidad": {
      "command": "C:\\envs\\vallis_marea\\Scripts\\python.exe",
      "args": ["-m", "core.agentes.servidores_mcp", "disponibilidad"],
      "cwd": "<ruta del proyecto>"
    }
  }
}
```

## Inspeccionar los Agent Cards de A2A

Con los agentes arriba:

```bash
curl http://127.0.0.1:8101/.well-known/agent-card.json
```

(8101 conocimiento · 8102 disponibilidad · 8103 recomendador)

---

## Estructura

```
app.py                      # entrypoint: consola Streamlit con visor de trazas
requirements.txt
core/
  config.py                 # TODA la configuración: rutas, puertos, modelos, umbrales
  llm/gemini.py             # generación + embeddings, con caché y modo degradado
  almacen/
    vectorial.py            # Chroma  (producción: pgvector)
    lexico.py               # SQLite FTS5 / BM25  (producción: tsvector)
    estructurado.py         # SQLite flota, rutas, reservas  (producción: Postgres)
  ingesta/
    chunking.py             # troceado consciente de la estructura, con metadatos
    indexar.py              # pipeline completo de ingesta
  agentes/
    conocimiento/rag.py     # híbrido + RRF + citación + doble guarda de abstención
    disponibilidad/logica.py# inventario, precios, reserva idempotente
    recomendador/logica.py  # afinidad semántica sobre el catálogo
    servidores_mcp.py       # los 3 servidores MCP (9 herramientas)
  a2a/
    servidor.py             # Agent Cards + ejecutor que despacha sobre MCP
    cliente.py              # cliente con conexiones persistentes y trazas por salto
    lanzador.py             # arranque en hilos o en procesos
  orquestador/
    grafo.py                # LangGraph: enrutar → delegar → componer
    contrato.py             # contrato JSON de salida (el mismo de producción)
  router/
    datos.py                # dataset: dominio sintético + MASSIVE (es) público
    entrenar.py             # clasificador sobre embeddings congelados
    predecir.py             # cascada clasificador → LLM → reglas
  aprendizaje/continuo.py   # correcciones humanas + memoria episódica
  evaluacion/suite.py       # 3 niveles de evaluación
data/
  corpus/                   # 5 documentos del negocio (sintéticos, realistas)
  flota.json, rutas.json    # catálogo estructurado
  evaluacion/golden_set.json# 35 preguntas con fuente esperada
documentacion/              # memoria, plan, informe final
```

---

## Advertencias de honestidad

Están en el informe con más detalle, pero conviene no descubrirlas leyendo el código:

1. **Los datos son sintéticos.** El corpus, la flota y las reservas fueron construidos por el
   equipo. Son realistas para Cartagena, pero no son los datos reales de Vallis Marea.
2. **El router no es fine-tuning de un LLM.** Es una regresión logística sobre embeddings
   **congelados** de Gemini. Los pesos del modelo de embeddings no se tocan.
3. **El almacenamiento del demo no es el de producción.** Chroma + SQLite replican el rol de
   pgvector + `tsvector`, detrás de la misma interfaz, porque la máquina de desarrollo no
   tiene Docker.
4. **No hubo prueba con usuarios reales.**
