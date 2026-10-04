# Memoria del Proyecto — Vallis Marea (SI7016 PLN Aplicado, EAFIT 2026-2)

> Archivo de contexto persistente. **Se actualiza cada vez que se toma una decisión, se
> descubre un dato relevante o se cierra un entregable.** Si retomas el proyecto después
> de un tiempo, lee este archivo primero.
>
> Última actualización: 2026-09-13 (desarrollo completo ejecutado)

---

## 1. Identidad del proyecto

| Campo | Valor |
|---|---|
| Curso | SI7016 Procesamiento del Lenguaje Natural Aplicado, 2026-2 |
| Programa | Maestría en Ciencias de Datos y Analítica — Universidad EAFIT |
| Proyecto | **Vallis Marea: De Agente Monolítico a Ecosistema de Agentes Interoperables (MCP/A2A)** |
| Contexto de aplicación (según enunciado) | Opción 5 — libre / atención al cliente (Contexto 4) en dominio turístico |
| Equipo | Elkin David Ortiz Rodriguez · Dilan Monsalve Monsalve · Juan David Trujillo Velez |
| **Fecha máxima de entrega** | **30 de septiembre de 2026** |
| Días hábiles restantes al 2026-09-13 | 17 días calendario |
| Documentos fuente | `../SI7016-ProyectoFinal-2026-2.pdf` (enunciado), `../SI7016_propuesta_vallis_marea (1).pdf` (propuesta aprobada) |

### Resuelto
- ✅ Equipo de tres integrantes (la propuesta listaba solo a Elkin; agregar los tres al informe).
- ✅ Alcance: **demo reproducible**. Desarrollo ejecutado y cerrado el 2026-09-13.
- ✅ **`GOOGLE_API_KEY` configurada** (2026-09-13, misma tarde). Reindexado + reentrenado +
      reevaluado con Gemini real. Métricas del informe ya son representativas (ver §7 Bitácora).

### Pendiente crítico (checklist de entrega, ver `plan_trabajo.md` para el detalle)
- [ ] **Presentación final (10% de la rúbrica): no existe nada todavía.** Falta diagrama visual
      de arquitectura, guion de demo y slides.
- [ ] Empaquetar el proyecto para entrega (no hay repositorio git todavía).
- [ ] Revisar el informe final una vez más antes de entregar (ya actualizado con métricas reales).

---

## 2. Qué es Vallis Marea (estado actual, YA en producción)

Chatbot conversacional **multimodal** para un negocio de **alquiler de lanchas en Cartagena**.

### Infraestructura existente
- **AWS EC2** (Ubuntu), todo dockerizado.
- **Caddy** como reverse proxy con TLS.
- **Chatwoot** como bandeja unificada / consola de supervisión humana.
- **Meta Cloud API (WhatsApp)** → Chatwoot → **webhooks a n8n**.
- **Postgres con pgvector** + **Redis** (colas de Chatwoot).

### Capacidades ya resueltas
- Entrada: **texto, voz e imagen**. Salida: **texto, imágenes o PDFs**.
- n8n clasifica el mensaje por modalidad; **Gemini** transcribe audio / describe imágenes.
- Respuesta con **contrato JSON estructurado** validado en producción, publicada de vuelta
  en Chatwoot.

### El problema que ataca el proyecto
El sistema es un **agente monolítico**: un único nodo de IA en n8n concentra clasificación,
conocimiento, disponibilidad y recomendación. Sin separación de responsabilidades ni
comunicación formal entre capacidades → limita escalabilidad y trazabilidad.

### Desafíos de ingeniería ya superados (activo del equipo, mencionar en el informe)
timeouts en webhooks · bucles de retroalimentación · duplicación de eventos ·
autenticación segura de credenciales.

---

## 3. Qué se va a construir (alcance del proyecto final)

Descomponer el nodo único de IA en un **ecosistema de agentes especializados** que
colaboran vía **protocolos abiertos**:

| Agente | Rol | Protocolo |
|---|---|---|
| **Conversador / Orquestador** | Recibe el mensaje clasificado, mantiene memoria, delega | Cliente A2A |
| **Disponibilidad y Reservas** | Consulta y bloquea inventario en tiempo real (Sheets/Postgres) | Servidor MCP + A2A |
| **Conocimiento (RAG agentic)** | Indexa políticas, FAQs y rutas turísticas en pgvector; **cita la fuente** | Servidor MCP + A2A |
| **Recomendador** | Sugiere paseos/embarcaciones según preferencias del cliente | Servidor MCP + A2A |

Cada agente conserva el contrato de salida JSON ya validado.

### Componente de innovación declarado en la propuesta
**Interoperabilidad de agentes bajo protocolos abiertos (MCP + A2A)**: migrar un sistema
*real*, con *datos reales*, hacia un ecosistema desacoplado, y **evaluar honestamente qué
se gana y qué fricciones introduce**. Ese "qué fricciones introduce" es el diferenciador
frente a proyectos de juguete — hay que medirlo, no solo afirmarlo.

---

## 4. Fuentes de datos

- **Estructurados**: inventario, precios, disponibilidad (Sheets → Postgres); historial de
  conversaciones multimodal indexado.
- **No estructurados**: políticas de cancelación, FAQs, descripciones de rutas turísticas
  de Cartagena → a indexar en pgvector (**pipeline nuevo, no existe hoy**).
- **Interacción real**: conversaciones históricas de producción → línea base para evaluar
  el rediseño.
- **Datasets públicos** (requisito del enunciado, **no estaba en la propuesta**): ver
  `plan_trabajo.md` §Brecha G3.

---

## 5. Rúbrica de evaluación (pesos)

| Criterio | Peso |
|---|---|
| Calidad técnica (NLP, LLMs, RAG, embeddings) | **25 %** |
| Funcionalidad de la aplicación (incluye interfaz) | **20 %** |
| Originalidad e innovación (componente de futuro del NLP) | **15 %** |
| Uso adecuado de tecnologías (frameworks) | **15 %** |
| Evaluación y validación | **10 %** |
| Presentación final / demo | **10 %** |
| Informe escrito | **5 %** |

Escala 1–5 por criterio.

---

## 6. Decisiones técnicas tomadas

> Formato: `[YYYY-MM-DD] Decisión — Justificación — Alternativa descartada`

- [2026-09-13] **Gemini** para generación y embeddings — es lo que ya usa producción — desc.: OpenAI, Anthropic.
- [2026-09-13] **Datos sintéticos del dominio** — no había acceso a datos reales el día del desarrollo — desc.: export de producción.
- [2026-09-13] **Chroma + SQLite/FTS5 en vez de Docker+pgvector** — la máquina NO tiene Docker, ni WSL, ni Postgres, y habilitarlos exige admin y reinicio — el acceso queda abstraído en `core/almacen/` para que migrar sea reescribir un módulo.
- [2026-09-13] **n8n como transporte, no como cerebro** — un ecosistema A2A necesita servicios con endpoint y Agent Card propios — desc.: orquestar con los nodos MCP nativos de n8n (dejaría A2A como adorno e impediría medir su costo).
- [2026-09-13] **El ejecutor A2A despacha sobre el servidor MCP del mismo agente** — una implementación, dos protocolos; es lo que hace verificable la promesa de la propuesta.
- [2026-09-13] **LangGraph con grafo explícito** en vez de un agente ReAct — hacen falta fronteras nítidas para medir latencia por salto.
- [2026-09-13] **Router = regresión logística sobre embeddings congelados**, NO fine-tuning — declarado así en el informe; el fine-tune real queda como trabajo siguiente.
- [2026-09-13] **MASSIVE (es) como fuera de dominio** para la clase `otro` — sin negativas reales el router clasifica con alta confianza cualquier cosa.

### APIs verificadas en vivo (no asumidas)
- `mcp` **2.x renombró `FastMCP` → `MCPServer`** (`mcp.server.mcpserver`). `Tool.input_schema`, no `inputSchema`.
- `a2a-sdk` 1.1.2 usa **protobuf**, no Pydantic. Agent Card en `/.well-known/agent-card.json`.
  Montaje: `add_a2a_routes_to_fastapi` + `create_agent_card_routes` + `create_jsonrpc_routes(request_handler=..., rpc_url=...)`.

---

## 7. Bitácora de avance

> Formato: `- [YYYY-MM-DD] hito / hallazgo`

- [2026-09-13] Leídos enunciado y propuesta. Plan de trabajo y memoria. Identificadas 6
  brechas entre la propuesta aprobada y el enunciado (G1–G6).
- [2026-09-13] Confirmado equipo de 3, alcance de demo reproducible y ejecución en **un día**.
  Plan recomprimido.
- [2026-09-13] **Docker/WSL/Postgres ausentes en la máquina** → pivot de almacenamiento.
- [2026-09-13] Sistema completo construido y verificado: 5 documentos de corpus (34
  fragmentos), 8 embarcaciones, 6 rutas, **9 herramientas MCP** en 3 servidores, **3 agentes
  A2A** con Agent Cards descubribles, orquestador LangGraph, UI Streamlit con visor de trazas,
  router entrenado, ciclo de aprendizaje y suite de evaluación de 3 niveles.
- [2026-09-13] **Bug corregido (idempotencia)**: se evaluaba después de cotizar, y el
  reintento fallaba porque la lancha figuraba ocupada por la reserva que el propio reintento
  creó. Se resuelve antes de cotizar.
- [2026-09-13] **Error de medición corregido**: el sobrecosto A2A medía 1.098 ms (+8.379%)
  porque el cliente redescubría el Agent Card y reabría conexión en cada llamada. Con event
  loop persistente y tarjeta cacheada: primera corrida real +21,4 ms (+72,8%). Dos órdenes de
  magnitud de diferencia frente a la medición ingenua original.
- [2026-09-13] Primera corrida completa (modo degradado, sin API key): recall_fuente@4 = 0,967
  · recall_sección@4 = 0,867 · MRR = 0,664 · abstención correcta = 1,00 · precisión de citación
  = 0,00 (no evaluable sin generación) · router F1 macro = 0,499 (no representativo, embeddings
  de hashing).
- [2026-09-13, misma tarde] **`GOOGLE_API_KEY` configurada por el equipo.** Reindexado el
  corpus, reconstruido el dataset del router y reentrenado, y vuelta a correr la suite de
  evaluación completa, todo con Gemini real (`gemini-2.5-flash` / `gemini-embedding-001`).
  **Resultados finales (los que quedan en el informe):**
  - Recuperación: recall_fuente@4 = **1,000** · recall_sección@4 = **0,933** · MRR = **0,842**
    (30/30 preguntas respondibles recuperan la fuente correcta).
  - Generación: precisión de citación = **1,000** · abstención correcta = **1,000** · tasa de
    respuesta = **0,967** (30/30 citas correctas, 5/5 abstenciones correctas).
  - Monolito vs. ecosistema: monolito p50 = 6,6 ms · ecosistema (A2A) p50 = 14,3 ms ·
    sobrecosto = **+7,7 ms (+116,7%)**. Descubrimiento del Agent Card: 365–434 ms por agente,
    una vez por sesión.
  - Router: exactitud = **0,870** · F1 macro = **0,847** (subió desde 0,499 en degradado —
    confirma que el clasificador dependía de semántica real, no de coincidencias léxicas).
  - `documentacion/informe_final.md` actualizado con estas cifras (§1, §4, §5, §6, §7).
- [2026-09-13] Generado checklist de cierre alineado a los 7 criterios de la rúbrica
  (Calidad técnica 25% · Funcionalidad 20% · Originalidad 15% · Tecnologías 15% · Evaluación
  10% · Presentación 10% · Informe 5%). Único vacío real: **Presentación final** — no hay
  diagrama visual, guion de demo ni slides. R6 cerrado; ver R7 en §8.

---

## 8. Riesgos vivos

| # | Riesgo | Mitigación |
|---|---|---|
| R1 | **Solo 17 días** para un cronograma diseñado a 5 semanas | Plan comprimido con entregables mínimos viables; ver priorización por rúbrica |
| R2 | Romper el bot **en producción** (clientes reales por WhatsApp) | Inbox/número de pruebas separado; feature flag para enrutar al ecosistema nuevo; no tocar el flujo productivo hasta validar |
| R3 | Latencia: WhatsApp + webhooks ya dieron timeouts; A2A añade saltos de red | Presupuesto de latencia explícito; respuesta rápida + "typing"; medir p50/p95 desde el día 1 |
| R4 | Madurez del spec A2A y de sus SDKs | Verificar versión del SDK y ruta del Agent Card antes de codificar; tener fallback HTTP/JSON-RPC propio documentado |
| R5 | El enunciado pide fine-tuning; lo implementado es un clasificador sobre embeddings congelados | **Declarado explícitamente** en informe y código. No llamarlo fine-tuning ante el jurado |
| ~~R6~~ | ~~Métricas de generación y de router corridas en modo degradado~~ | **Cerrado 2026-09-13**: reindexado, reentrenado y reevaluado con la API key. Ver bitácora |
| R7 | **Presentación final (10% de la rúbrica) sin nada construido**: sin diagrama visual, sin guion de demo, sin slides | Priorizar antes de la sustentación; ver `plan_trabajo.md` para el checklist de cierre |
| R8 | El proyecto no es un repositorio git; no hay forma clara de empaquetar la entrega | Definir con el equipo si se entrega como zip o se sube a GitHub/GitLab antes del 30-sep |

---

## 9. Glosario de siglas del proyecto

- **MCP** (Model Context Protocol): estandariza cómo un agente se conecta a herramientas y
  datos externos. La capacidad se expone con un esquema común descubrible.
- **A2A** (Agent-to-Agent): estandariza la comunicación *entre* agentes autónomos. Cada
  agente publica qué sabe hacer; otros lo descubren y le delegan tareas, aun si fueron
  construidos con frameworks distintos.
- **RAG agentic**: el agente *decide* cuándo y qué recuperar, en vez de recuperar siempre.
