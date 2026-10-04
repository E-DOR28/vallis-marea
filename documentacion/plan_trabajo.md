# Plan de Trabajo — Vallis Marea (SI7016) · **Ejecución en 1 día**

Contexto completo en [memoria_proyecto.md](memoria_proyecto.md).

**Equipo:** Elkin David Ortiz Rodriguez · Dilan Monsalve Monsalve · Juan David Trujillo Velez
**Fecha de ejecución:** 2026-09-13 · **Entrega:** 30-sep-2026 (el desarrollo se cierra hoy)

---

## 0. Restricciones reales (verificadas en máquina, no supuestas)

| Decisión del equipo | Estado | Resolución |
|---|---|---|
| LLM: **Gemini (Google AI Studio)** | ✅ Viable | `google-genai` para generación **y** embeddings. Coherente con producción |
| Datos: **sintéticos del dominio** | ✅ Viable | Corpus y flota los construimos nosotros, realistas para Cartagena |
| Infra: **Docker + Postgres/pgvector** | ❌ **Inviable hoy** | **No hay Docker, ni WSL, ni Postgres en la máquina.** Instalar Docker Desktop exige WSL2, permisos de administrador y reinicio |

### Pivot de almacenamiento (decisión forzada, documentada en el informe)

El demo **replica 1:1 la arquitectura de producción** con equivalentes locales sin infraestructura:

| Producción (Vallis Marea) | Demo reproducible | Rol |
|---|---|---|
| Postgres + **pgvector** | **Chroma** (persistente en disco) | Búsqueda densa |
| Postgres + **`tsvector`** | **SQLite + FTS5** | Búsqueda léxica (BM25) |
| Postgres (inventario, reservas) | **SQLite** | Datos estructurados |
| Redis (memoria corto plazo) | SQLite / memoria de proceso | Estado de conversación |

Todo el acceso pasa por `core/almacen/`, con una interfaz única. **Migrar a pgvector es
reimplementar un módulo, no reescribir los agentes** — y eso es exactamente la tesis del
proyecto sobre desacoplamiento. Chroma, además, está en la lista de tecnologías del enunciado.

> Esto no es una degradación del alcance: los cuatro agentes, MCP, A2A, el RAG con citación
> y la evaluación se construyen completos. Cambia el motor de persistencia, no la arquitectura.

---

## 1. Alcance de hoy (y qué se sacrifica si el tiempo aprieta)

### Núcleo intocable — sostiene el 75 % de la rúbrica
1. **RAG agentic con citación** (híbrido denso+léxico, fusión RRF, abstención)
2. **3 servidores MCP** (Conocimiento, Disponibilidad, Recomendador)
3. **A2A real** entre orquestador y agentes pares, con Agent Cards descubribles
4. **Orquestador LangGraph** con router, memoria y contrato JSON
5. **UI Streamlit** con visor de trazas (qué agente, qué herramienta, qué fuente)
6. **Evaluación** con golden set y métricas
7. **Informe + README reproducible + guion de demo**

### Orden de sacrificio si falta tiempo
1. Fine-tuning con transformer → se queda el router por embeddings + clasificador sklearn
2. Aprendizaje continuo automatizado → queda el mecanismo, ejecutado a mano una vez
3. Recomendador con perfil elaborado → versión por similitud simple
4. Comparación monolito vs. ecosistema → se reduce a latencia y nº de saltos

---

## 2. Brechas del enunciado y cómo las cierra el código de hoy

| # | Exigencia | Dónde se resuelve |
|---|---|---|
| **G1** | RAG agentic **con citación** | `core/agentes/conocimiento/` — híbrido + RRF + citas obligatorias + abstención |
| **G2** | Interfaz web funcional | `app.py` (Streamlit) con panel de trazas A2A/MCP |
| **G3** | **Datasets públicos** | Dataset público de intención (dominio viajes/atención al cliente) mezclado con frases sintéticas del dominio, en `core/router/` |
| **G4** | **Fine-tuning** | Clasificador de intención **entrenado sobre embeddings de Gemini** (sklearn). ⚠️ Se reporta en el informe con ese nombre exacto — *entrenamiento de un clasificador*, no *fine-tuning de un LLM*. El fine-tune de un transformer pequeño queda como notebook opcional |
| **G5** | **Aprendizaje continuo** | `core/aprendizaje/` — cada corrección humana se vuelve ejemplo etiquetado + memoria episódica; reentrenamiento del router bajo demanda |
| **G6** | **Evaluación** | `core/evaluacion/` — golden set, recall@k, groundedness, precisión de citación, latencia por salto A2A |

**Nota de honestidad académica (G4):** entrenar un clasificador sobre embeddings congelados
**no es** fine-tuning. Se declara así en el informe y se justifica por la restricción de
tiempo, dejando el fine-tune real como trabajo inmediato siguiente. Vender lo uno como lo
otro es el tipo de cosa que un jurado detecta.

---

## 3. Arquitectura que se implementa hoy

```
                  ┌──────────────────────────────┐
   Streamlit  ──► │  Orquestador (LangGraph)     │
   (app.py)       │   · router de intención      │
   + visor de     │   · memoria de conversación  │
     trazas       │   · cliente A2A              │
                  └──────┬───────────────────────┘
                         │  A2A (JSON-RPC / HTTP)
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
   ┌────────────┐ ┌────────────┐ ┌────────────┐
   │Conocimiento│ │Disponib. y │ │Recomendador│   cada uno:
   │  (RAG)     │ │ Reservas   │ │            │   · servidor A2A + Agent Card
   └─────┬──────┘ └─────┬──────┘ └─────┬──────┘   · servidor MCP con sus tools
      MCP│            MCP│           MCP│
         ▼               ▼              ▼
   Chroma + FTS5     SQLite        embeddings
   (políticas,      (flota,        del catálogo
    FAQs, rutas)     reservas)
```

Los tres agentes pares exponen **las mismas capacidades por dos vías**: MCP (para cualquier
cliente compatible, incl. Claude Desktop) y A2A (para el orquestador). Esa dualidad es
literalmente lo que la propuesta prometió demostrar.

---

## 4. Cronograma de hoy por bloques

| Bloque | Qué se produce | Estado |
|---|---|---|
| **B1** | Entorno (`C:\envs\vallis_marea`), `requirements.txt`, estructura, `core/config.py` | ⏳ |
| **B2** | Datos semilla: corpus (políticas, FAQs, rutas de Cartagena) + flota e inventario | ⏳ |
| **B3** | Ingesta: chunking con metadatos → embeddings Gemini → Chroma + FTS5 | ⏳ |
| **B4** | Búsqueda híbrida + fusión RRF + generación con citas y abstención | ⏳ |
| **B5** | 3 servidores MCP con sus herramientas | ⏳ |
| **B6** | Capa A2A: Agent Cards + endpoints + cliente en el orquestador | ⏳ |
| **B7** | Orquestador LangGraph: router, delegación, memoria, contrato JSON, guardrails | ⏳ |
| **B8** | Router de intención entrenado (G3+G4) + bucle de aprendizaje (G5) | ⏳ |
| **B9** | UI Streamlit con visor de trazas | ⏳ |
| **B10** | Golden set + suite de evaluación + resultados | ⏳ |
| **B11** | README reproducible, informe final, guion de demo | ⏳ |

El avance real se registra en la bitácora de [memoria_proyecto.md](memoria_proyecto.md).

---

## 5. Evaluación (versión de un día)

**Nivel componentes**
- Recuperación: recall@k y MRR sobre el golden set
- Generación: precisión de la citación y tasa de abstención correcta
- Router: accuracy y F1 macro en held-out, contra baseline LLM-como-router

**Nivel sistema — el aporte del proyecto**
- Latencia **por salto A2A** y total extremo a extremo
- Nº de llamadas MCP por consulta
- Costo/tokens por conversación
- **Fricciones cualitativas de MCP/A2A**: qué costó más, qué se ganó en reutilización

No hay prueba con usuarios reales hoy; se declara como limitación explícita del estudio.

---

## 6. Requisito pendiente del equipo

~~**API key de Google AI Studio** en un archivo `.env` en la raíz.~~ **Resuelto 2026-09-13**:
configurada, verificada y usada para reindexar el corpus, reentrenar el router y recorrer
toda la suite de evaluación. Los números reales quedaron en `informe_final.md` §4.

---

## 7. Checklist de cierre — alineado a la rúbrica del taller

*(Generado 2026-09-13, misma tarde, tras confirmar la API key funcionando.)*

| Criterio (peso) | Estado | Qué falta, si algo |
|---|---|---|
| Calidad técnica (25%) | ✅ Completo, con métricas reales | Nada bloqueante |
| Funcionalidad de la app (20%) | ✅ Completo y verificado extremo a extremo | Nada bloqueante |
| Originalidad e innovación (15%) | ✅ Completo — es el punto más fuerte | Nada bloqueante |
| Uso de tecnologías (15%) | ✅ Completo y justificado en el informe | Nada bloqueante |
| Evaluación y validación (10%) | ✅ Completo, con números reales (Gemini) | Nada bloqueante |
| **Presentación final (10%)** | ❌ **No existe nada** | Diagrama visual de arquitectura, guion de demo (8–10 min), slides |
| Informe escrito (5%) | ✅ Completo y actualizado con métricas reales | Revisión final de redacción antes de entregar |

### Pendientes logísticos (no puntúan directo, pero bloquean la entrega)
- [ ] **Empaquetar para entrega**: el proyecto no es un repositorio git. Decidir con el equipo
      si se entrega como `.zip` o se sube a un repo (GitHub/GitLab) antes del 30-sep.
- [ ] **Grabación de respaldo de la demo**, por si algo falla en vivo el día de la sustentación.

### Orden recomendado de lo que queda
1. **Diagrama de arquitectura como imagen** (no solo el ASCII del informe) — es lo que más
   rápido se nota en una presentación y hoy no existe en ningún formato visual.
2. **Guion de demo de 8–10 minutos** sobre la consola Streamlit, apoyado en el panel de trazas
   que ya muestra qué agente se invocó, con qué herramienta MCP y en cuántos ms.
3. **Slides** (o un documento de apoyo) con el resumen ejecutivo: problema, arquitectura,
   hallazgo del sobrecosto A2A medido (+7,7 ms / +116,7%), y el veredicto del §6 del informe.
4. Definir formato de entrega (zip vs. repo) y empaquetar.
