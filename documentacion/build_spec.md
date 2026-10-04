# Build-spec: cierre de Vallis Marea (SI7016)

| Campo | Valor |
|---|---|
| Versión | 0.4, borrador para validación (historial en la §11) |
| Fecha | 2026-10-04 |
| Estado | Sin implementar. Este documento es lo único que se ha creado; ningún otro archivo del proyecto fue modificado. |
| Proyecto | `/Users/kin/Desktop/Proyecto NLP` (no es un repositorio git) |
| Equipo | Elkin David Ortiz Rodriguez, Dilan Monsalve Monsalve, Juan David Trujillo Velez |
| Contexto | Proyecto final de SI7016 Procesamiento del Lenguaje Natural Aplicado, EAFIT 2026-2 |

---

## 0. Solicitud de validación (para el revisor)

Eres un revisor independiente. Este documento especifica cómo ejecutar, de forma segura, un plan de cierre sobre un proyecto que ya funciona. **Tu trabajo es validar la especificación, no ejecutarla.**

### Reglas para el revisor

- Solo lectura. No modifiques, crees ni borres archivos del proyecto. No instales dependencias, no entrenes modelos y no llames a APIs externas con credenciales.
- No leas ni imprimas el contenido de `.env`. Contiene una clave de API.
- Contrasta cada afirmación de la §2 con el código o los datos, usando las referencias `archivo:línea`. Si una referencia no coincide, repórtalo como hallazgo.
- Distingue lo que verificaste de lo que no pudiste verificar.

### Qué validar

1. **Exactitud de la §2.** ¿Los hechos citados coinciden con el código y los datos?
2. **Consistencia.** ¿Algún requerimiento contradice el código existente? ¿Faltan dependencias entre fases?
3. **Seguridad.** ¿El modelo de amenazas (§7) omite algo relevante para una página pública con una clave de LLM y una herramienta que escribe en base de datos?
4. **Criterios de aceptación (§5 y §8).** ¿Son medibles, suficientes y no triviales? ¿Las tolerancias de la §8.2 son razonables dado que los LLM no son deterministas?
5. **Rigor del fine-tuning (R2).** Fuga de datos, tamaño de muestra, regla de adopción.
6. **Riesgos no listados** en la §10.
7. **Ambigüedades** que obligarían a un implementador a adivinar.

### Formato de respuesta

1. **Veredicto:** Aprobado, Aprobado con cambios o Rechazado, con una frase de justificación.
2. **Tabla de hallazgos** con columnas `ID | Sección | Severidad | Evidencia | Recomendación`. Severidades: Bloqueante, Importante, Menor, Sugerencia.
3. **Preguntas abiertas** que debe responder el equipo.
4. **No verificado:** lo que no pudiste comprobar y por qué.

---

## 1. Contexto y objetivo

Vallis Marea es un ecosistema de agentes para un negocio de alquiler de lanchas en Cartagena. Un orquestador (LangGraph) delega por A2A a tres agentes pares (Conocimiento, Disponibilidad y Reservas, Recomendador), cada uno publicado también como servidor MCP. El sistema está implementado y evaluado (ver §2.3). Todos los datos son sintéticos salvo 42 ejemplos públicos del router.

El cierre del proyecto tiene cuatro brechas frente al enunciado y al cronograma:

| Brecha | Requerimiento |
|---|---|
| Falta un diagrama visual y un flujo detallado de qué pide y entrega cada agente | R1 |
| El router es una regresión logística sobre embeddings congelados, no fine-tuning | R2 |
| No hay interfaz desplegada ni entrega empaquetada | R3 |
| El informe debe salir también en PDF | R4 |
| No hubo prueba con usuarios (semana 5 del cronograma) | R5 |

R0 es el prerrequisito común (entorno, control de versiones, línea base).

---

## 2. Estado de partida verificado

Todo lo de esta sección se leyó directamente en el proyecto. Las referencias son `archivo:línea`.

### 2.1 Código

| # | Hecho | Evidencia |
|---|---|---|
| H1 | El orquestador invoca exactamente 4 herramientas, con el nombre cableado en el código | `core/orquestador/grafo.py:162` (`buscar_conocimiento`), `:173` (`recomendar`), `:203` (`consultar_disponibilidad`), `:225` (`bloquear_reserva`) |
| H2 | Se publican 9 herramientas MCP. 5 no las invoca el orquestador: `obtener_fragmentos`, `cotizar`, `interpretar_fecha`, `listar_rutas`, `extraer_perfil` | `core/agentes/servidores_mcp.py` y `core/a2a/servidor.py` (`_HABILIDADES`) |
| H3 | El informe afirma que el orquestador descubre las capacidades por Agent Card en lugar de tenerlas cableadas. Contradice H1. | `documentacion/informe_final.md:74` y `:304` |
| H4 | El bloqueo de una reserva exige: intención `reserva`, `confirma_reserva`, `embarcacion_id`, `ruta` y un salto previo exitoso | `core/orquestador/grafo.py:218-224` (la línea `:220` evalúa `slots["confirma_reserva"]`) |
| H5 | `confirma_reserva` lo extrae un LLM con salida estructurada | `core/orquestador/grafo.py:78` (`ESQUEMA_SLOTS`) |
| H6 | El router carga su modelo con `np.load(..., allow_pickle=True)` | `core/router/predecir.py:53` |
| H7 | La cascada del router usa el clasificador si su confianza es al menos 0,45 (config); si no, pregunta a un LLM y por último a reglas | `core/config.py:169`, `core/router/predecir.py:165` |
| H8 | Los agentes se atan a `HOST_AGENTES` (por defecto `127.0.0.1`) y el cliente construye la URL con ese mismo host | `core/config.py:103` y `:145-147`; `core/a2a/lanzador.py:34` y `:112` |
| H9 | `levantar_en_hilos` es idempotente y arranca los servidores uvicorn en hilos del proceso actual | `core/a2a/lanzador.py:52` |
| H10 | `RUTA_TRAZAS` se define y se crea, pero ningún módulo `.py` la usa para escribir | `core/config.py:27` y `:43`; búsqueda en `core/` y `app.py` |
| H11 | `MAX_TURNOS_MEMORIA_CORTA = 12` está definida. **No verifiqué si algún módulo la usa.** | `core/config.py:176` |
| H12 | El historial de conversación lo aporta quien llama a `grafo.responder`. Hoy es la sesión de Streamlit. | `core/orquestador/grafo.py:394`, `app.py` |
| H13 | Una embarcación no puede tener dos reservas vivas el mismo día (índice único parcial, solo estados `confirmada` y `bloqueada`) | `core/almacen/estructurado.py:62-64` |
| H14 | El índice de idempotencia es único sobre `clave_idempotencia` **sin condición de estado** | `core/almacen/estructurado.py:66-68` |
| H15 | `siguiente_codigo_reserva` calcula `COUNT(*) + 1001` sobre la tabla. Dos reservas concurrentes pueden obtener el mismo código (la clave primaria lo rechaza), y borrar filas haría reutilizar códigos. | `core/almacen/estructurado.py:273-277`, `:47-48` |
| H16 | `crear_reserva` captura `IntegrityError` y lo devuelve como `(False, "conflicto: ...")` | `core/almacen/estructurado.py:245-247` |
| H17 | Cada operación abre una conexión SQLite nueva (`check_same_thread=False`) | `core/almacen/estructurado.py:72-76` |
| H18 | Chroma usa `PersistentClient` sobre disco local | `core/almacen/vectorial.py:30` |
| H19 | La salida de `grafo.responder` es el contrato JSON más una clave `_trazas` (router, slots, saltos A2A) | `core/orquestador/contrato.py`, `core/orquestador/grafo.py:394-431` |
| H20 | No hay carpeta de pruebas automáticas ni repositorio git. La única verificación de regresión es la suite de evaluación. | listado de `/Users/kin/Desktop/Proyecto NLP` |
| H21 | `.gitignore` excluye `.env`, `__pycache__/`, `*.pyc`, `data/almacen/`, `data/trazas/`, `data/modelos/`, `.req_installed` | `.gitignore` |
| H22 | `requirements.txt` usa solo cotas inferiores (`>=`) | `requirements.txt` |

### 2.2 Entorno de desarrollo (esta máquina)

- macOS arm64, Apple M5 Pro, 24 GB de RAM, unos 746 GB libres.
- Python del sistema: 3.9.6, sin torch, transformers ni streamlit. Homebrew tiene `python3.13` y `python3.14`; los `.pyc` del proyecto son `cpython-313`. No hay entorno virtual del proyecto.
- Docker CLI 29.3.1 con contexto Colima; el daemon está apagado.
- Node v25.8.2 y Google Chrome instalados. No hay pandoc, LaTeX, mermaid-cli ni graphviz.

### 2.3 Datos y línea base

Las métricas provienen de `data/evaluacion/resultados.json` y `data/evaluacion/metricas_router.json`, ejecutadas el 2026-09-13 con Gemini. **No han sido reproducidas en esta máquina.**

| Elemento | Valor |
|---|---|
| Corpus | 5 documentos Markdown, 34 fragmentos indexados |
| Estructurados | 8 embarcaciones, 6 rutas, 12 reservas preexistentes |
| Dataset del router | 183 ejemplos (141 sintéticos del dominio, 42 de MASSIVE en español), 9 clases. Se descartaron 40 ejemplos de MASSIVE por ambigüedad. Split con semilla 7: 137 de entrenamiento y 46 de prueba. |
| Golden set | 35 preguntas (30 respondibles, 5 de abstención) |
| Recuperación | recall de fuente@4 = 1,000; recall de sección@4 = 0,9333; MRR = 0,8417 |
| Generación | precisión de citación = 1,000; abstención correcta = 1,000 (5 de 5); tasa de respuesta = 0,967 (29 de 30) |
| Monolito contra A2A | p50 6,6 ms contra 14,3 ms (sobrecosto de 7,7 ms) |
| Router | exactitud 0,8696; F1 macro 0,8471. Clases débiles: `faq` (F1 0,50) y `politica` (F1 0,667). |
| Aprendizaje continuo | `correcciones.jsonl` con 1 registro; `memoria_episodica.jsonl` con 5 episodios, 2 de ellos del modo degradado |

### 2.4 Supuestos NO verificados

El revisor debe tratarlos como riesgo, no como hecho.

- Lo verificado el 2026-10-04 en documentación oficial (ver §2.5) puede cambiar. Se reconfirma el día del despliegue.
- Que la RAM de la capa gratuita elegida alcance para el encoder afinado más los cuatro procesos (orquestador y tres agentes). Se mide en R2 y R3.
- Que el hosting elegido admita cuatro procesos en un solo contenedor y el puerto que se le asigne.
- Los identificadores exactos de los modelos candidatos del R2 (se confirman al empezar).
- Que `indexar.indexar_todo` inicialice también las tablas estructuradas en un arranque en limpio.
- La seguridad de hilos del grafo compilado de LangGraph, de la caché SQLite de `gemini.py` y del cliente A2A ante muchas sesiones simultáneas.
- Todas las estimaciones de esfuerzo, memoria e imagen.

### 2.5 Hechos de hosting verificados el 2026-10-04

Fuentes: documentación oficial de cada proveedor.

| Opción | Hecho verificado | Consecuencia |
|---|---|---|
| Hugging Face, Spaces | Los Spaces estáticos son gratis. Los Docker y Gradio "requieren un plan de pago para crearse" en cuentas personales (PRO, 9 USD al mes). La cuenta gratuita solo aloja hasta 2 Spaces Gradio con ZeroGPU. CPU Basic (2 vCPU, 16 GB, 50 GB de disco no persistente) cuesta 0 USD por hora, pero crear el Space exige plan de pago. En hardware gratuito el Space se duerme por inactividad. | Un Space Docker no es gratis. |
| Hugging Face, visibilidad | Un Space privado devuelve 404 a terceros. Un Space "protegido" (código privado, app pública) exige PRO. | Para usuarios externos el Space debe ser público (o protegido con PRO). |
| Hugging Face, modelos | La cuenta gratuita incluye 100 GB de almacenamiento privado. | Un modelo privado es gratis. |
| Azure Container Apps con Azure for Students | Azure for Students: sin tarjeta de crédito, 100 USD de crédito válidos 12 meses, solo para estudiantes universitarios de tiempo completo. Container Apps tiene una cuota gratuita permanente de 180 000 vCPU-s, 360 000 GiB-s y 2 millones de solicitudes al mes. | Misma cuota que Cloud Run y sin tarjeta. No se verificó que el correo institucional de EAFIT sea aceptado ni las restricciones de región de la suscripción. Si se agota el crédito, los servicios se deshabilitan. |
| Google Cloud Run (facturación por solicitud) | Misma cuota mensual en us-central1: 180 000 vCPU-s, 360 000 GiB-s, 2 millones de solicitudes y 1 GB de salida desde Norteamérica. Con 1 vCPU y 2 GiB equivale a unas 50 h de uso activo al mes. Escala a cero. La cuota se calcula por cuenta de facturación, no por proyecto. El Free Tier exige una cuenta de facturación, y el alta pide tarjeta u otro medio de pago (retención de 0 a 1 USD). | Alcanza para 5 usuarios, pero necesita una cuenta de facturación. Hay arranque en frío. **Elegido por el equipo (v0.4).** |
| Google Cloud, servicios de apoyo | Free Trial: 300 USD de crédito por 90 días para clientes nuevos, con tarjeta. Al terminar, la cuenta se cierra y los recursos se detienen (30 días de gracia) salvo que se pase a cuenta de pago; el Free Tier sigue activo. Cloud Build: 2 500 min al mes (`e2-standard-2`). Artifact Registry: 0,5 GB. Secret Manager: 6 versiones activas y 10 000 accesos al mes. Los presupuestos y alertas de facturación avisan, **no cortan el gasto**. | El crédito de 300 USD **no paga la API de Gemini de AI Studio**. Una imagen con `torch` puede superar los 0,5 GB gratuitos de Artifact Registry; se mide el tamaño real en R3 y, si pasa, se evalúa ONNX o se borran las imágenes viejas. |
| Render, plan gratuito | Servicio web gratis con 512 MB de RAM, 750 h al mes. Se apaga tras 15 min sin tráfico y tarda cerca de 1 min en volver. Disco efímero. No declara tarjeta para el plan gratuito. | 512 MB no alcanza para `torch`. Solo es viable con el router actual (embeddings de Gemini) o con un encoder ONNX cuantizado, y hay que medirlo. |
| Modal, plan Starter | 30 USD de crédito gratis al mes, sin cuota mensual, escala a cero, CPU a 0,0000131 USD por núcleo-segundo y memoria a 0,00000222 USD por GiB-segundo. Con 1 núcleo y 2 GiB serían unas 470 h activas al mes. Retención de logs de 1 día. | Holgado para 5 usuarios. Exige adaptar el arranque al SDK de Modal. No se verificó si el alta pide tarjeta. |
| Oracle Cloud, Always Free | VM Ampere A1 con 2 OCPU y 12 GB en total, 200 GB de disco, 10 TB de salida al mes. Oracle puede reclamar instancias inactivas y a veces falta capacidad. | Es la opción más holgada, pero exige administrar una VM. No se verificó si el alta pide tarjeta. |
| Koyeb | La página de precios no muestra capa gratuita de cómputo. | Descartada. |

---

## 3. Alcance y decisiones

### 3.1 Dentro del alcance

R0 a R5, definidos en la §5.

### 3.2 Fuera del alcance

- WhatsApp, Chatwoot, n8n y EC2 (descartados por el equipo).
- Slides, guion de la demo y grabación de respaldo (siguen pendientes, pesan 10 % de la rúbrica).
- Fine-tuning de un LLM generativo.
- Datos reales de producción y migración a pgvector.

### 3.3 Decisiones tomadas por el equipo

- La interfaz es una página web propia, no WhatsApp.
- Se conservan los datos sintéticos actuales.
- El informe sale en `.md` y en PDF.
- El modelo afinado se publica en un repositorio **privado** de Hugging Face (la cuenta existe y es gratuita).
- La prueba con usuarios es de **5 participantes**. Con n = 5 el análisis es cualitativo y no se reportan porcentajes como evidencia estadística.
- La fecha de entrega no condiciona el alcance: se ejecuta el plan completo.
- Quien ejecuta todas las ramas es el agente de implementación. No hay fusiones concurrentes entre personas.
- Coste objetivo: 0 USD. El equipo pidió alternativas gratuitas al Space Docker.
- El servicio corre en un hosting **remoto**. No se usa ninguna máquina personal del equipo, ni siquiera con túnel.
- El hosting es **Google Cloud** (Cloud Run). El equipo dice contar con "una key"; su tipo y el estado de facturación del proyecto siguen sin confirmar (Q2, Q6).

### 3.4 Decisiones asumidas por este documento, pendientes de confirmación

| Decisión | Valor asumido | Alternativa |
|---|---|---|
| Nivel de la interfaz | **Nivel 2**: API FastAPI y página HTML/JS a medida, empaquetada en un solo contenedor Docker | Nivel 1: la app Streamlit actual (menos esfuerzo). Nivel 3: nivel 2 más `docker-compose` con agentes en contenedores separados. |
| Hosting remoto para la prueba | **Google Cloud Run** (decisión del equipo, v0.4), con la imagen construida en Cloud Build y los secretos en Secret Manager. Pendiente: tipo de clave, proyecto y cuenta de facturación (Q2, Q6). | Si el proyecto no tiene facturación: Azure Container Apps con Azure for Students, o Render gratuito (512 MB). El contenedor es el mismo. Ver §2.5. |
| Llamadas a Gemini en el servicio desplegado | Se conserva la clave de AI Studio (`genai.Client(api_key=...)`, un único punto en `core/llm/gemini.py:82`) guardada en Secret Manager. | Modo Vertex AI (`vertexai=True`, credenciales de la cuenta de servicio de Cloud Run), que se factura contra el proyecto de Google Cloud. Cambia el origen de la cuota y exige verificar que los embeddings coincidan con los del índice. Solo si la clave resulta ser de AI Studio sin cuota suficiente. |
| Aumento de datos con paráfrasis | Solo si el primer resultado no mejora en `faq` y `politica` | No usarlo |

---

## 4. Reglas de desarrollo seguro

### 4.1 Línea base y control de versiones

1. Antes de modificar nada: copia de seguridad del proyecto completo **fuera** del repositorio, con permisos restringidos (incluye `.env`).
2. `git init`, commit base y etiqueta `base-preintervencion`.
3. Antes de volver a ejecutar la suite o el entrenamiento, copiar `data/evaluacion/resultados.json`, `data/evaluacion/metricas_router.json` y `data/modelos/router_intencion.npz` a `data/evaluacion/historico/`. Ambos comandos sobrescriben esos archivos.
4. No ejecutar `core.ingesta.indexar --recrear` sobre `data/almacen/` sin copia previa.

### 4.2 Ramas y propiedad de archivos

| Rama | Responsable de | Archivos que puede modificar |
|---|---|---|
| `req0-entorno` | R0 | `.gitignore`, `.env.example`, `requirements*.txt`, lockfile |
| `req1-docs` | R1 y R4 | `documentacion/**`, `README.md`; en `core/` solo la validación de arranque de Agent Cards |
| `req2-finetune` | R2 | `core/router/**` (archivos nuevos `afinar.py` y `afinado.py`; cambios mínimos en `predecir.py`), `data/modelos/**`, `data/evaluacion/**` |
| `req3-web` | R3 y R5 | `core/web/**` (nuevo), `Dockerfile`, `scripts/**`, `tests/**`; cambios acotados en `estructurado.py` y `lanzador.py` |

`core/config.py` lo tocan R2 y R3. Regla: cada rama agrega sus variables en un bloque propio al final, con commits pequeños y rebase antes de fusionar.

Orden de fusión: R0, luego R1, R2 y R3 en cualquier orden, luego R5 y R4 al final.

### 4.3 Secretos

- `.env` nunca se versiona. Se entrega `.env.example` sin valores.
- Escaneo de secretos (por ejemplo gitleaks) antes del primer push y sobre la imagen Docker.
- La clave de Gemini se carga solo por variable de entorno o secreto del hosting. Nunca en logs, trazas, respuestas de error ni en la imagen.
- Si la clave actual salió alguna vez de la máquina (por ejemplo en un zip), se rota antes de publicar.

### 4.4 Dependencias y cadena de suministro

- Generar un lockfile con versiones exactas en R0 (`mcp` 2.x y `a2a-sdk` 1.x ya cambiaron de API una vez).
- `requirements-train.txt` separado: torch, transformers y herramientas de entrenamiento no van en la imagen más de lo necesario para inferencia.
- Modelo descargado del Hub: revisión fijada por hash de commit, solo `safetensors`, `trust_remote_code=False`.
- `np.load(..., allow_pickle=False)` en `predecir.py:53`. Los arreglos guardados no requieren pickle (a validar en la implementación).
- Imagen base fijada por versión.

### 4.5 Invariantes que no pueden regresar

| Invariante | Dónde vive |
|---|---|
| Nunca se excede la capacidad autorizada | `core/agentes/disponibilidad/logica.py` (`cotizar`, `consultar_disponibilidad`) |
| Ningún precio fuera de la tabla de tarifas | `core/agentes/disponibilidad/logica.py` (`calcular_precio`) |
| Reservar es idempotente | `logica.py` (`bloquear_reserva`, idempotencia antes de cotizar) y `estructurado.py:66-68` |
| Nunca dos reservas vivas de la misma embarcación y fecha | `estructurado.py:62-64` |
| Respuesta afirmativa sin cita se convierte en abstención | `core/agentes/conocimiento/rag.py` |
| Las citas las construye el código, no el modelo | `core/agentes/conocimiento/rag.py` |
| El contrato JSON de salida no cambia de forma | `core/orquestador/contrato.py` |
| Un salto A2A fallido escala a un humano | `core/orquestador/grafo.py` (`nodo_componer`) |

Cada invariante debe tener al menos una prueba automática antes de tocar el módulo que lo contiene (§8.3).

### 4.6 Cambios aditivos

Todo flag nuevo arranca con el comportamiento actual. `VM_ROUTER` vale `embeddings` por defecto y solo cambia a `afinado` cuando se cumple la regla de adopción del R2.

### 4.7 Datos y privacidad

- Sin teléfonos ni nombres. Identificador de sesión hasheado con sal en los logs.
- El texto de los mensajes se guarda solo si el usuario dio consentimiento explícito.
- Los logs de usuarios no entran al repositorio (`data/trazas/` ya está ignorado; se agregan los demás).
- Antes de empaquetar, `memoria_episodica.jsonl` y `correcciones.jsonl` se vacían (contienen pruebas del modo degradado). `dataset_router.json` se conserva: es insumo del R2.
- Retención: los logs de la prueba con usuarios se borran al cerrar el informe. Cualquier cita textual en el informe se anonimiza.

---

## 5. Requerimientos

Las estimaciones son del autor del documento, no medidas.

### R0. Entorno, base segura y línea base (1 a 2 h)

**Entregables**
- Copia de seguridad, repositorio git, commit base y etiqueta (§4.1).
- Entorno virtual con Python 3.13, `requirements.lock` y `requirements-train.txt`.
- `.gitignore` auditado (agregar `.DS_Store`, `.venv/`, `tmp/`, logs de usuarios) y `.env.example`.
- La app actual (`streamlit run app.py`) levantada y la suite de evaluación re-ejecutada.

**Criterios de aceptación**
- Las métricas re-ejecutadas están dentro de las tolerancias de la §8.2, o cada discrepancia queda documentada con su causa probable.
- El hallazgo de monolito contra A2A se re-mide y se compara; no se exige igualdad (depende de la máquina).
- Existen copias de los resultados previos en `data/evaluacion/historico/`.

### R1. Diagrama y flujo detallado de los agentes (3 a 4 h)

**Entregables**
- Fuentes Mermaid en `documentacion/diagramas/*.mmd` y exportación a SVG o PNG:
  - Diagrama de componentes con las fronteras A2A, MCP y llamada local.
  - Cuatro diagramas de secuencia (Conocimiento; Disponibilidad con sus variantes sin fecha, consulta y consulta más reserva; Recomendador; Directo) y el camino de escalamiento.
  - Diagrama de la página web y la API.
- Anexo "Contratos entre agentes" en el informe: las 9 herramientas con parámetros, campos de respuesta, quién las invoca y qué hace el orquestador con el resultado. Se marcan las 5 no invocadas como capacidad pública para clientes MCP.
- Sección "Lo que no ocurre": no hay comunicación entre agentes pares.
- Corrección de `informe_final.md` §2.2 y §6 (H3), más una **validación de arranque**: al iniciar, verificar que las 4 habilidades que el orquestador invoca estén publicadas en los Agent Cards; si falta alguna, registrar el error y reportar estado `degradado` en `/api/salud`.

**Criterios de aceptación**
- Para 4 consultas de ejemplo (una por camino), la traza real (`_trazas.saltos_a2a`) coincide con el diagrama correspondiente en agente, herramienta y orden.
- El anexo cubre las 9 herramientas y sus parámetros coinciden con las firmas de `servidores_mcp.py`.
- Ninguna afirmación del informe contradice H1.

### R2. Fine-tuning real del router (~1 día)

**Qué se afina:** un encoder multilingüe para clasificación de intención, que reemplaza el nivel 1 de la cascada de `predecir.py`. No es un LLM generativo.

**Protocolo (se congela y se registra en un commit antes de entrenar)**
1. Datos: `data/aprendizaje/dataset_router.json` sin modificar. Se registra su SHA-256.
2. Casi-duplicados: se agrupan (texto normalizado y similitud de caracteres) antes de dividir. La validación cruzada es estratificada por grupo.
3. Evaluación principal: validación cruzada estratificada por grupo, 5 particiones repetidas 3 veces. Métrica primaria: F1 macro. Intervalo de confianza del 95 % por bootstrap sobre las diferencias pareadas.
4. Comparabilidad: se reporta también el split fijo con semilla 7 (137 y 46), idéntico al de la línea base.
5. El conjunto de prueba nunca se usa para elegir hiperparámetros, umbral ni modelo.

**Baselines**
- B0: regresión logística sobre embeddings de Gemini (actual).
- B1: TF-IDF con regresión logística.
- B2: Gemini `flash-lite` sin entrenamiento, con el prompt de `predecir.py` (`_SISTEMA_ROUTER`).
- B3: embeddings multilingües locales congelados con regresión logística (aísla el efecto de afinar frente al de cambiar de embedding).

**Candidatos (2 o 3; identificadores a confirmar):** BETO (`dccuchile/bert-base-spanish-wwm-cased`), `multilingual-e5-small` o un MiniLM multilingüe, y XLM-R base como contraste. Fine-tuning completo, con LoRA como experimento opcional. Tasa de aprendizaje elegida con validación interna usando solo datos de entrenamiento. Parada temprana sobre una partición de validación tomada del entrenamiento. Pesos de clase. Semillas fijas (7, 11, 23).

**Aumento de datos (opcional):** paráfrasis con Gemini aplicadas solo al entrenamiento de cada partición, nunca a validación ni prueba. Se declara el porcentaje sintético.

**Regla de adopción (pre-registrada, a validar)**: el modelo afinado pasa a ser el predeterminado (`VM_ROUTER=afinado`) solo si cumple las tres condiciones:
1. El límite inferior del IC del 95 % de la diferencia de F1 macro frente a B0 es al menos -0,02 (no inferioridad).
2. La latencia p50 de inferencia local es menor que la de B0 (embedding de Gemini más regresión logística), medida en la misma máquina.
3. El umbral de confianza se recalibra sobre validación (curva cobertura-exactitud) y se fija antes de mirar la prueba.

Si no se cumple, el modelo queda disponible como alternativa y el informe reporta el resultado igual.

**Entregables**
- `core/router/afinar.py` (entrenamiento, reproducible con una orden y una semilla) y `core/router/afinado.py` (inferencia).
- `data/modelos/router_afinado/` con `model.safetensors`, archivos del tokenizador y `etiquetas.json`, más una model card.
- `data/evaluacion/metricas_router_afinado.json` con resultados por partición, intervalos, hash del dataset, semillas, versiones de librerías y commit.
- Actualización de los docstrings de `entrenar.py`, `nota_metodologica` y los §4.4, §5.2 y §7 del informe. El texto dice "fine-tuning de un encoder, no de un LLM generativo".

**Criterios de aceptación**
- Un tercero reproduce las métricas con la orden documentada y la misma semilla (tolerancia numérica documentada).
- No hay filtración demostrable entre entrenamiento y prueba (agrupación aplicada y verificada con una prueba).
- La decisión de adopción se toma aplicando la regla anterior, sin ajustes posteriores.

### R3. API, página web y despliegue (2 a 2,5 días)

**Estructura**
- `core/web/app.py` (FastAPI con un único worker; el arranque lanza los agentes con `lanzador.levantar_en_hilos()` y precalienta), `seguridad.py`, `sesiones.py`, `registro.py` y `static/` (HTML, JS y CSS, sin paso de compilación).
- `Dockerfile` (usuario no root, sin recarga automática, comprobación de salud sobre `/api/salud`, puerto leído de la variable `PORT`), `scripts/humo.py` y `tests/`.
- `app.py` (Streamlit) se conserva como consola de desarrollo y respaldo.

**Comportamiento**
- El `context_id` que usa el grafo (y por tanto la clave de idempotencia de reservas) lo emite o valida el servidor, nunca texto libre del cliente.
- Memoria por sesión en el servidor: últimos 12 turnos, vencimiento por inactividad.
- Cada turno se registra (§6.3).
- Arranque en limpio: si no existen `vallis.db` o el índice, se reconstruyen desde `data/corpus` y los JSON de flota y rutas. Sin clave de Gemini el servicio arranca en modo degradado y lo indica en `/api/salud` y en la página.
- Reservas de demo: se liberan automáticamente tras `VM_TTL_RESERVA_DEMO_MIN` cambiando su estado a `liberada` (**no se borran filas**, por H15). Al liberar, se anula `clave_idempotencia` (por H14). Máximo 2 reservas por sesión.
- `siguiente_codigo_reserva` se hace seguro ante concurrencia (reintento ante colisión de clave primaria o secuencia monótona). Es un cambio en un módulo con invariantes (§4.5) y requiere sus pruebas.
- Límite de concurrencia global (semáforo y cola corta); sobre ese límite, 503 con `Retry-After`.

**Frontend**
- Chat adaptado a celular, con los 6 ejemplos de `app.py` como atajos.
- Citas como chips de fuente (título, sección, versión), tarjeta de reserva bloqueada, aviso de escalamiento, indicador de agentes en línea y panel desplegable de trazas.
- Aviso visible de que es un demo académico con datos sintéticos, y casilla de consentimiento para registrar el texto.
- **Todo texto proveniente del LLM, de las citas o del comentario de feedback se inserta como texto plano (`textContent`), nunca como HTML.**

**Despliegue**
- Un solo contenedor Docker, que se prueba primero en local. El modelo afinado se descarga del repositorio privado del Hub al arrancar (con respaldo a la regresión logística si no está disponible).
- Variables de entorno o secretos del hosting: `GOOGLE_API_KEY`, `HF_TOKEN` (solo lectura, porque el modelo es privado), `ADMIN_TOKEN`.
- Prueba con usuarios: el contenedor corre en Google Cloud Run (ver §3.4). Ninguna parte depende de una máquina personal del equipo.
- Google Cloud: la imagen se construye con Cloud Build (la máquina del equipo es `arm64` y Cloud Run exige `linux/amd64`; además el daemon de Docker local no está activo). Una región de Tier 1 con cuota gratuita (por ejemplo `us-central1`). Los tres secretos van en Secret Manager y se montan como variables de entorno del servicio, con una cuenta de servicio propia que solo puede leerlos. `--max-instances=1`, `--concurrency` acotada, `--min-instances=0`, `--no-cpu-throttling` solo si la medición lo exige (cambia a facturación por instancia, con otra cuota gratuita).
- Control de gasto: alerta de presupuesto en la cuenta de facturación (avisa, no corta) más el tope diario de turnos de la aplicación, que es el límite efectivo. El servicio es público, así que cada turno cuesta una llamada a Gemini.
- Parámetros comunes del servicio: 2 GiB de memoria, 1 vCPU, una réplica como máximo, concurrencia limitada y mínimo de réplicas en 0. Se mide el arranque en frío (descarga del modelo y carga del índice) y se documenta.
- Si se elige Render gratuito: 512 MB es el techo. El router pasa a ONNX cuantizado, o se conserva el actual basado en embeddings de Gemini. No se instala `torch` en la imagen.
- El encoder afinado agrega `torch` y `transformers` al contenedor. R2 mide la RAM en reposo y evalúa exportarlo a ONNX con cuantización si la capa gratuita elegida no alcanza.
- Persistencia de logs: exportación a un dataset privado del Hub o descarga con `ADMIN_TOKEN`. El disco del contenedor se asume efímero.

**Criterios de aceptación**
- Arranque en limpio sin archivos previos: `/api/salud` pasa a `listo`.
- `scripts/humo.py` pasa con los 6 ejemplos: contrato válido, citas presentes en las consultas de conocimiento, saltos esperados.
- 20 solicitudes concurrentes (mezcla de consultas y reservas de distintas embarcaciones): 0 dobles reservas, 0 códigos duplicados, 0 errores espurios no reintentados.
- La batería adversarial de la §7 (T2) pasa.
- Una respuesta simulada que contenga `<img src=x onerror=...>` no ejecuta código en la página (prueba con Chrome).
- El límite de turnos por sesión y el tope diario cortan correctamente.
- La imagen construye en Docker local, y el servicio responde desde la URL pública del hosting remoto elegido, con el modelo privado descargado por token.

### R4. Informe también en PDF (2 a 3 h)

**Entregables**
- `documentacion/informe_final.pdf`, generado desde `informe_final.md` (fuente única). Se regenera cuando cambie el `.md`.
- Portada (título, materia, universidad, integrantes, fecha), tabla de contenido, numeración de páginas, tablas con encabezado y diagramas a ancho de página.
- El diagrama ASCII de la §2 y los símbolos fuera del repertorio latino básico (cajas `─ │ ┌ ┐ └ ┘ ┬ ┼`, `▼`, `►`, `⇒`) se reemplazan por imágenes o texto. Hay 158 caracteres `─` y otros símbolos en el `.md` actual.

**Criterios de aceptación**
- Cada página se convierte a imagen y se revisa: sin tablas cortadas ni diagramas ilegibles.
- Cero caracteres perdidos: el texto extraído del PDF contiene todas las palabras del `.md` (comparación automatizada).
- Toda cifra del `.md` aparece idéntica en el PDF (comparación automatizada de números).

### R5. Prueba con usuarios por enlace (0,5 día más coordinación)

**Entregables**
- Protocolo escrito **antes** de la prueba: 5 a 8 participantes, 3 a 5 tareas guionadas, métricas (éxito de tarea, escalamientos, abstenciones, `ms_total`, feedback) y encuesta de 5 preguntas en escala 1 a 5.
- Consentimiento visible antes del primer mensaje.
- Resultados con sus límites declarados (muestra pequeña, no representativa, datos sintéticos).

**Criterios de aceptación**
- Al menos 5 sesiones completas.
- El informe incluye una sección de la prueba y las limitaciones.
- Se aplica la política de datos de la §4.7.

---

## 6. Interfaces y contratos

### 6.1 API HTTP

| Endpoint | Entrada | Salida |
|---|---|---|
| `POST /api/turno` | `{"session_id": str, "texto": str (1 a 1000 caracteres), "request_id": str (uuid), "consentimiento_registro": bool}` | `200 {"contrato": {...sin _trazas}, "trazas": {"router": {...}, "slots": {...}, "saltos_a2a": [...]}, "sesion": {"turnos_restantes": int}, "request_id": str}` |
| `GET /api/salud` | n/a | `{"estado": "listo" \| "iniciando" \| "degradado", "agentes": {clave: bool}, "gemini": bool, "router": "embeddings" \| "afinado" \| "reglas", "version": str}`. Sin secretos. |
| `GET /api/agentes` | n/a | Estado de cada agente y habilidades de su Agent Card |
| `POST /api/feedback` | `{"session_id": str, "request_id": str, "valor": "up" \| "down", "comentario": str (hasta 500 caracteres, opcional)}` | `204` |
| `GET /api/admin/logs` | Cabecera `Authorization: Bearer <ADMIN_TOKEN>` | Registros en NDJSON. Deshabilitado si `ADMIN_TOKEN` no está definido. |

Errores: `400` entrada inválida (vacía, demasiado larga, tipos incorrectos), `429` límite de uso (con `Retry-After`), `503` servicio iniciando o saturado, `500` genérico. **Ninguna respuesta de error incluye trazas de pila, rutas del sistema ni claves.** `request_id` repetido devuelve la respuesta ya calculada (evita el doble clic).

### 6.2 Variables de entorno nuevas

| Variable | Valor por defecto | Uso |
|---|---|---|
| `VM_ROUTER` | `embeddings` | `embeddings` o `afinado` |
| `VM_ROUTER_MODELO` | vacío | Ruta local o id del repositorio del Hub, con revisión fijada |
| `HF_TOKEN` | vacío | Lectura del modelo si es privado |
| `ADMIN_TOKEN` | vacío | Habilita la exportación de logs |
| `VM_MAX_TURNOS_SESION` | 30 | Turnos por sesión |
| `VM_TOPE_DIARIO` | 500 | Turnos globales por día |
| `VM_MAX_CARACTERES` | 1000 | Longitud máxima del mensaje |
| `VM_TTL_SESION_MIN` | 120 | Vencimiento por inactividad |
| `VM_TTL_RESERVA_DEMO_MIN` | 30 | Liberación automática de reservas de demo |
| `VM_URL_<AGENTE>` | vacío | URL completa del agente (para contenedores separados) |

Los valores por defecto son propuestas a validar.

### 6.3 Formato de registros (JSONL)

Trazas, una línea por turno: `ts`, `sesion_hash`, `request_id`, `intencion`, `metodo_router`, `confianza`, `saltos` (lista de `{agente, habilidad, ms, ok}`), `ms_total`, `escalado`, `abstencion`, `tokens_entrada`, `tokens_salida`, `modelo`, `longitud_texto`. Los campos `texto` y `respuesta` solo se incluyen si `consentimiento_registro` es verdadero.

Feedback: `ts`, `sesion_hash`, `request_id`, `valor`, `comentario`. Un `down` genera además un candidato en un archivo aparte; **nunca se integra solo al dataset del router**.

### 6.4 Artefacto del modelo

`router_afinado/` con `model.safetensors`, archivos del tokenizador, `etiquetas.json` (las mismas 9 clases que las claves de `config.INTENCIONES`; el orden de los logits se guarda explícitamente en el archivo, porque el modelo actual ordena las clases alfabéticamente y no en el orden de `config.INTENCIONES`) y `README.md` (model card con métricas, datos, limitaciones y la declaración de que los datos son en su mayoría sintéticos).

---

## 7. Modelo de amenazas de la página pública

| ID | Amenaza | Mitigación requerida | Verificación |
|---|---|---|---|
| T1 | XSS por salida del LLM, citas o comentario de feedback | `textContent`, cabecera CSP restrictiva, escape del comentario | Prueba con Chrome y una respuesta simulada maliciosa |
| T2 | Inyección de instrucciones en el mensaje del usuario (reservas no deseadas, revelar prompts, saltarse la abstención) | Slots validados por tipo, enum y expresión regular; el bloqueo exige las 4 condiciones de H4; máximo 2 reservas por sesión; sin secretos en los prompts; salida del LLM nunca ejecutada | Batería de 10 a 15 prompts adversariales con resultado esperado: no reserva sin confirmación, no revela el prompt, no altera precios |
| T3 | Abuso de cuota y costo de Gemini | Límite por sesión e IP, tope diario, longitud máxima, 503 limpio al agotarse | Prueba de carga corta |
| T4 | Filtración de secretos | §4.3; no registrar cabeceras ni variables | Escaneo del historial git y de la imagen |
| T5 | Endpoints administrativos expuestos | Token, comparación en tiempo constante, deshabilitado sin token | Pruebas 401 y 403 |
| T6 | Concurrencia: doble reserva o códigos duplicados | Índice único (H13), reintento ante colisión, liberación por estado | 20 solicitudes concurrentes (R3) |
| T7 | Divulgación en trazas y errores (rutas, claves, pilas) | Sanear errores; la traza solo expone lo de §6.1 | Revisión de respuestas de error provocadas |
| T8 | Cadena de suministro (dependencias, modelo del Hub, pickle) | §4.4 | Revisión del lockfile y de la carga del modelo |
| T9 | Privacidad de los participantes | §4.7 | Revisión de logs y del repositorio |
| T10 | Clickjacking o incrustación indebida | `frame-ancestors` configurable. **Debe permitir el marco de huggingface.co**, a verificar. | Prueba de incrustación |
| T11 | Saturación por turnos lentos (cada turno ocupa un hilo varios segundos) | Semáforo global y cola corta | Prueba de carga |
| T12 | Falsificación de IP vía `X-Forwarded-For` | Confiar en ese encabezado solo detrás del proxy del hosting remoto; el límite por sesión es el principal | Revisión de configuración |
| T13 | Modo degradado silencioso (respuestas "[MODO DEGRADADO]" al usuario) | `/api/salud` y la página lo indican; el despliegue falla visiblemente sin clave | Arranque sin clave |

---

## 8. Plan de pruebas y puertas

### 8.1 Puertas por fase

| Fase | Puerta para cerrar |
|---|---|
| R0 | Línea base reproducida o discrepancias documentadas; copia de seguridad y etiqueta creadas |
| R1 | Diagramas coinciden con 4 trazas reales; anexo completo; §2.2 corregido |
| R2 | Protocolo congelado antes de entrenar; resultados reproducibles; regla de adopción aplicada |
| R3 | Todos los criterios del R3; suite de regresión sin degradación (§8.2) |
| R4 | Comparaciones automáticas de texto y cifras en verde; revisión visual de todas las páginas |
| R5 | Protocolo previo, 5 sesiones o más, política de datos aplicada |

### 8.2 Tolerancias de regresión (propuesta a validar)

Los LLM no son deterministas. Una pregunta de las 30 respondibles equivale a 0,033.

| Métrica | Línea base | Mínimo aceptable |
|---|---|---|
| recall de fuente@4 | 1,000 | 0,967 |
| recall de sección@4 | 0,933 | 0,900 |
| MRR | 0,842 | 0,800 |
| precisión de citación | 1,000 | 0,967 |
| abstención correcta (5 de 5) | 1,000 | 1,000 |
| tasa de respuesta | 0,967 | 0,933 |
| F1 macro del router actual | 0,847 | 0,800 |

### 8.3 Pruebas nuevas

- Pruebas unitarias de cada invariante de la §4.5 antes de tocar su módulo.
- Pruebas de la API con cliente de prueba de FastAPI: contrato, errores, límites, idempotencia por `request_id`.
- Prueba de concurrencia de reservas (R3).
- Recorrido de la página con Chrome automatizado: los 6 ejemplos, comprobación de XSS y capturas para el informe.
- Prueba de arranque en limpio y de arranque sin clave.

---

## 9. Despliegue y reversión

1. Cada despliegue sale de una etiqueta. Se revierte reconstruyendo la imagen desde la etiqueta anterior (en Azure Container Apps y Cloud Run, redirigiendo el tráfico a la revisión previa).
2. Lista de verificación antes de publicar: escaneo de secretos limpio, `.gitignore` auditado, datasets sin datos personales, aviso de demo académico visible, límites activos, `scripts/humo.py` y recorrido con Chrome en verde, model card completa, `memoria_episodica.jsonl` y `correcciones.jsonl` vaciados.
3. Si se filtra la clave: rotarla en Google AI Studio, actualizar el Secret, revisar el uso y registrar el incidente en `documentacion/memoria_proyecto.md`.
4. Entrega empaquetada: repositorio y zip **sin** `.env`, `cache_llm.db`, `data/trazas/` ni logs; con `.env.example`, `requirements.lock`, README actualizado (comandos independientes del sistema operativo; hoy usa rutas de Windows) y los diagramas.

---

## 10. Riesgos y preguntas abiertas

| ID | Riesgo o pregunta | Impacto |
|---|---|---|
| Q1 | ~~Fecha de entrega~~ | Resuelta: no condiciona el alcance (§3.3) |
| Q2 | Hosting decidido: Google Cloud Run. Falta saber si el proyecto ya tiene cuenta de facturación activa (Free Trial o de pago) | Sin facturación, Cloud Run no se puede usar ni dentro de la cuota gratuita |
| Q6 | ¿Qué es "la key"? Una clave de Gemini (AI Studio), una clave de API de Google Cloud, una cuenta de servicio o un acceso al proyecto | Define cómo se despliega y de dónde sale la cuota de Gemini (§3.4) |
| Q3 | ~~Quién ejecuta~~ | Resuelta: el agente de implementación |
| Q4 | La cuenta de Hugging Face existe. ¿El token de escritura está disponible para publicar el modelo privado? | Bloquea la publicación del modelo |
| Q5 | ~~Participantes~~ | Resuelta: 5. El análisis es cualitativo |
| R1 | El fine-tuning puede quedar empatado con la línea base (46 ejemplos de prueba) | Se reporta igual; no se fuerza la adopción |
| R2 | La cuota gratuita de Gemini limita las pruebas de carga y la prueba con usuarios | Reducir concurrencia en pruebas; tope diario |
| R3 | Los límites de hosting de la §2.5 cambian, o la RAM gratuita no alcanza para el encoder | Reconfirmar el día del despliegue; usar ONNX cuantizado o cambiar de proveedor (el contenedor es el mismo) |
| R4 | Los cambios de R3 en `estructurado.py` introducen regresiones | Obligan a pruebas previas (§4.5) |
| R5 | La línea base no se reproduce en esta máquina | Se documenta y se replantean las tolerancias |
| R6 | Seguridad de hilos no verificada en el grafo, la caché de `gemini.py` y el cliente A2A | La prueba de 20 solicitudes concurrentes puede revelar fallos |
| R7 | Ya hay documentación (README, plan, memoria) con rutas de Windows y estados desactualizados | Actualizar en R1 y R4 |

---

## 11. Registro de cambios del documento

| Versión | Fecha | Cambio |
|---|---|---|
| 0.1 | 2026-10-04 | Primer borrador para validación |
| 0.2 | 2026-10-04 | Se verificó el hosting (§2.5): el Space Docker de Hugging Face exige PRO. Se fijan modelo privado en el Hub, 5 participantes, ejecución por el agente y coste objetivo 0 USD. El hosting queda como propuesta a confirmar. |
| 0.3 | 2026-10-04 | El equipo descarta la máquina personal y el túnel: el hosting debe ser remoto. Se agregan Azure Container Apps (Azure for Students), Modal y la RAM de Render, y se aclara que Cloud Run pide tarjeta. |
| 0.4 | 2026-10-04 | R0 cerrado (etiqueta `r0-linea-base`). El equipo elige Google Cloud. Se verifican Free Trial, Free Tier, Secret Manager y la exclusión de la API de AI Studio del crédito. Se agrega Q6 y la opción de modo Vertex AI. |
