# Vallis Marea: de agente monolítico a ecosistema de agentes interoperables (MCP/A2A)

**Informe técnico final**
SI7016 Procesamiento del Lenguaje Natural Aplicado · Maestría en Ciencias de Datos y Analítica
Universidad EAFIT · 2026-2

**Equipo:** Elkin David Ortiz Rodriguez · Dilan Monsalve Monsalve · Juan David Trujillo Velez

---

## 1. Resumen

Vallis Marea es un chatbot multimodal **en producción** para un negocio de alquiler de lanchas
en Cartagena: AWS EC2 dockerizado, Caddy como proxy TLS, Chatwoot como bandeja unificada,
WhatsApp vía Meta Cloud API, webhooks a n8n, Postgres con pgvector y Redis. Recibe texto, voz
e imágenes y responde con texto, imágenes o PDFs.

Su limitación no es de capacidad sino de **arquitectura**: un único nodo de IA en n8n
concentra clasificación, conocimiento, disponibilidad y recomendación, sin separación de
responsabilidades ni comunicación formal entre capacidades.

Este trabajo descompone ese monolito en cuatro agentes que colaboran mediante **MCP** (Model
Context Protocol, para exponer herramientas y datos) y **A2A** (Agent-to-Agent, para que los
agentes se descubran y se deleguen tareas), y **mide qué se gana y qué se paga** con esa
migración.

**Hallazgo principal:** la interoperabilidad cuesta **+7,7 ms por delegación** (p50: 14,3 ms
por A2A contra 6,6 ms por llamada directa, +116,7 %), más un costo de arranque de
~0,37–0,43 s por agente para descubrir su Agent Card, que se paga una vez por sesión y no
por mensaje. A cambio se obtiene desacoplamiento real, reutilización de las capacidades por
clientes
externos y trazabilidad por salto.

---

## 2. Arquitectura

```
                  ┌──────────────────────────────┐
   Streamlit  ──► │  Orquestador (LangGraph)     │
   (app.py)       │   · router de intención      │
   + visor de     │   · memoria de conversación  │
     trazas       │   · cliente A2A              │
                  └──────┬───────────────────────┘
                         │  A2A (JSON-RPC sobre HTTP)
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

### 2.1 La decisión de diseño central

El ejecutor A2A de cada agente **no reimplementa nada**: despacha sobre el mismo servidor MCP
de ese agente (`core/a2a/servidor.py`, clase `EjecutorMCP`). Cada capacidad queda por tanto
publicada dos veces sin duplicar lógica:

- **MCP** → cualquier cliente compatible (Claude Desktop, un IDE, otro agente).
- **A2A** → el orquestador de Vallis Marea, o un agente de otro proveedor.

Esa dualidad es lo que hace verificable la promesa de la propuesta, y lo que permite medir el
desacoplamiento: la capacidad es una, los protocolos de acceso son dos.

### 2.2 El orquestador no sabe nada del negocio

No conoce precios, ni políticas, ni la flota. Solo sabe **a quién preguntarle**, y lo averigua
leyendo los Agent Cards, no por configuración cableada. Es la diferencia concreta contra el
nodo monolítico que reemplaza.

### 2.3 El contrato de salida no cambió

El orquestador publica el **mismo contrato JSON** que n8n ya entrega a Chatwoot
(`core/orquestador/contrato.py`). Esa frontera estable es lo que permitiría sustituir el
cerebro sin tocar la capa de transporte ni el frontend de WhatsApp.

### 2.4 Equivalencias de almacenamiento

La máquina de desarrollo no tiene Docker, WSL ni Postgres, y habilitarlos exigía permisos de
administrador y reinicio. El demo replica la arquitectura de producción con equivalentes
locales, detrás de una interfaz única (`core/almacen/`):

| Producción | Demo | Rol |
|---|---|---|
| Postgres + **pgvector** | **Chroma** persistente | búsqueda densa |
| Postgres + **`tsvector`** | **SQLite + FTS5** | búsqueda léxica (BM25) |
| Postgres | **SQLite** | inventario y reservas |

Migrar a pgvector es reescribir `core/almacen/vectorial.py`; los agentes no se enteran. Eso
mismo es una demostración del desacoplamiento que el proyecto defiende.

---

## 3. Capacidades de NLP implementadas

### 3.1 RAG agentic con citación obligatoria

**Troceado consciente de la estructura.** Se trocea por sección (`##`) y solo se subdivide
cuando excede el tamaño objetivo, respetando límites de párrafo y de frase. Cada fragmento
arrastra `fuente`, `titulo`, `seccion`, `version` y `fecha`: **esos metadatos son lo que
permite citar**. Sin ellos el agente puede acertar, pero no puede probarlo.

**Búsqueda híbrida con RRF.** La densa entiende paráfrasis ("puedo echar para atrás el paseo"
→ política de cancelación); la léxica no pierde literales ("72 horas", "VM-04"). Se fusionan
con *Reciprocal Rank Fusion*, que combina **rangos** en vez de **puntajes**, evitando
normalizar escalas incomparables (coseno contra BM25).

**Citación construida por código, no por el modelo.** El LLM recibe fragmentos etiquetados
`[F1]..[Fn]` y declara en su salida estructurada cuáles usó. Esos índices se resuelven contra
los metadatos reales. El modelo elige *cuáles*; nunca redacta los metadatos.

**Abstención con dos guardas.**
1. *Numérica*, antes de llamar al LLM: si no hay evidencia, no se gasta la llamada.
2. *Del propio LLM*, que debe declarar `abstencion: true` cuando los fragmentos no contienen
   la respuesta.

Y un tercer guardrail: **una respuesta afirmativa sin ninguna cita se convierte en
abstención**, porque no es verificable. Para un negocio real, decir "no sé" es mucho más
barato que inventar una política de reembolso.

### 3.2 Router de intención (embeddings + clasificador)

Cascada de tres niveles: clasificador entrenado (~ms) → LLM como desempate (~cientos de ms)
→ reglas por palabras clave (sin red). El punto no es reemplazar al LLM, sino **reservarlo
para los casos dudosos**.

**Datos:** frases del dominio escritas por el equipo + **MASSIVE en español** (dataset público
real, vía `mteb/amazon_massive_intent`, config `es`) como distribución **fuera de dominio**
para la clase `otro`. Ese segundo bloque no es decorativo: un router entrenado solo con frases
del dominio clasifica con alta confianza cualquier cosa que le llegue, incluido "pon música",
porque nunca vio un ejemplo de algo que no le corresponde.

### 3.3 Embeddings más allá del RAG

El Recomendador compara la descripción en lenguaje natural de lo que el cliente quiere contra
una representación textual de cada ruta y cada embarcación. Es recuperación semántica sobre
datos **estructurados**. La ventaja sobre filtrar por etiquetas: el cliente no dice "rumba",
dice "algo para celebrar con mis amigos y que suene música".

### 3.4 Guardrails y escritura segura

El agente de Disponibilidad es el único con efectos sobre el mundo. Tres invariantes:

- **Nunca se excede la capacidad autorizada** (falta ante la Capitanía y anula el seguro).
- **Ningún precio que no salga de la tabla de tarifas.**
- **Reservar es idempotente**: un mensaje de WhatsApp reenviado no reserva dos veces.

La idempotencia tiene además un índice único parcial en la base como última línea de defensa
contra la doble reserva.

> **Nota de implementación con valor didáctico.** La primera versión evaluaba la idempotencia
> *después* de cotizar, y el reintento fallaba: la embarcación figuraba ocupada por la reserva
> que el propio reintento había creado. Se corrigió resolviendo la idempotencia antes que
> nada. Es exactamente la clase de error que la propuesta menciona haber enfrentado en
> producción con la duplicación de eventos.

### 3.5 Aprendizaje continuo

Vallis Marea ya produce la señal de aprendizaje más valiosa y no la usa: **cada intervención
de un operador en Chatwoot es un ejemplo etiquetado por un experto del negocio, gratis**. El
módulo `core/aprendizaje/continuo.py` la convierte en dos mecanismos:

1. **Correcciones de intención** acumuladas e integradas al dataset del router; el
   reentrenamiento es incremental, no desde cero. Solo se incorporan las correcciones donde
   el humano discrepó: los aciertos ya están representados y agregarlos refuerza el sesgo.
2. **Memoria episódica**: cada turno resuelto se indexa con su embedding, y ante un mensaje
   nuevo se pueden recuperar casos parecidos. Es aprendizaje sin gradientes: el sistema
   mejora porque acumula experiencia recuperable.

---

## 4. Evaluación

### 4.1 Nivel 1 — Recuperación (35 preguntas del golden set)

Medido con embeddings reales de `gemini-embedding-001` (768 dimensiones) sobre el corpus
reindexado.

| Métrica | Valor |
|---|---|
| recall de fuente @4 | **1,000** |
| recall de sección @4 | **0,933** |
| MRR | **0,842** |

La recuperación acierta la fuente correcta en las 30 de 30 preguntas respondibles del golden
set, y la sección exacta en 28 de 30. Es el resultado más sólido del trabajo: la búsqueda
híbrida (densa con RRF + léxica) recupera con precisión casi perfecta sobre un corpus de este
tamaño. Estas cifras reemplazan una primera corrida en modo degradado (con embeddings locales
por hashing) que ya daba 0,967 / 0,867 / 0,664 gracias a la mitad léxica; con semántica real
la recuperación llega al techo.

### 4.2 Nivel 2 — Generación, citación y abstención

Medido con `gemini-2.5-flash` generando la respuesta y decidiendo qué fragmentos citar.

| Métrica | Valor | Lectura |
|---|---|---|
| tasa de abstención correcta | **1,000** | 5 de 5 preguntas fuera del corpus |
| precisión de citación | **1,000** | 30 de 30 preguntas respondibles citan la fuente correcta |
| tasa de respuesta | **0,967** | 29 de 30 preguntas respondibles obtuvieron respuesta (1 se abstuvo por precaución) |

Con la clave configurada, el sistema nunca cita una fuente incorrecta ni se abstiene quedando
la pregunta sin resolver injustificadamente: las 5 preguntas fuera del corpus (clima, hotel,
helicóptero, criptomonedas, sede en otra ciudad) se reconocen como fuera de alcance, y las 30
dentro del corpus se responden citando exactamente el documento esperado. Es el resultado que
en la corrida anterior (modo degradado) aparecía en 0,000 porque, sin generación, el guardrail
"sin citas ⇒ abstención" se activaba siempre — no era un fallo del sistema sino la ausencia de
la clave.

### 4.3 Nivel 3 — Monolito contra ecosistema *(el hallazgo del proyecto)*

Misma capacidad, alcanzada por dos caminos: llamada directa en proceso (como el nodo único de
n8n) y delegación A2A sobre HTTP/JSON-RPC. 20 consultas.

| | p50 | p95 | media |
|---|---|---|---|
| Monolito (llamada directa) | 6,6 ms | 11,0 ms | 7,6 ms |
| Ecosistema (A2A) | 14,3 ms | 21,9 ms | 15,2 ms |
| **Sobrecosto** | **+7,7 ms (+116,7 %)** | +10,9 ms | +7,6 ms |

Descubrimiento del Agent Card: 434 ms (conocimiento), 365 ms (disponibilidad), 366 ms
(recomendador) — **una vez por agente y por sesión**, no por mensaje.

> **Cómo casi publicamos una cifra falsa.** La primera medición arrojó **+1.098 ms (+8.379 %)**
> de sobrecosto. La causa no era el protocolo: el cliente abría una conexión HTTP nueva y
> volvía a descargar el Agent Card **en cada invocación**. Se estaba midiendo el costo del
> *descubrimiento repetido*, no el de la delegación. Con un event loop persistente,
> conexiones reutilizadas y la tarjeta cacheada, el sobrecosto real resultó ser **dos órdenes
> de magnitud menor**.
>
> La lección vale más que el número: al evaluar arquitecturas es fácil medir el costo de una
> implementación ingenua y atribuírselo al paradigma. Un sobrecosto de 8.379 % habría
> "demostrado" que A2A es inviable en WhatsApp. El dato correcto, unos 8 ms, es despreciable
> frente al segundo largo que tarda una llamada al LLM — y sigue siéndolo aun expresado como
> porcentaje (+116,7 %): en terminos absolutos, ambos caminos resuelven la consulta en menos
> de 25 ms, muy por debajo de cualquier umbral perceptible en una conversación de WhatsApp.

### 4.4 Router

Reentrenado sobre el dataset combinado (frases sintéticas del dominio + MASSIVE en español
como fuera de dominio) usando embeddings reales de Gemini como features.

| Métrica | Valor |
|---|---|
| exactitud (held-out) | **0,870** |
| F1 macro | **0,847** |
| dataset | 183 ejemplos, 9 clases (42 de MASSIVE es) |

El salto frente a la corrida en modo degradado (exactitud 0,565, F1 macro 0,499) confirma la
hipótesis del diseño: un clasificador lineal sobre embeddings **sin semántica** apenas supera
el azar ponderado; el mismo clasificador sobre embeddings **con semántica real** separa las 9
clases con un F1 macro de 0,847. Las clases con menos ejemplos (`faq`, `ruta_turistica`, con
4 casos de prueba cada una) son las que más error concentran — es la señal más clara de que
el siguiente dataset debe crecer ahí, no de que el enfoque falle.

---

## 5. Limitaciones y honestidad metodológica

Se declaran explícitamente porque afectan cómo deben leerse los resultados.

1. **Los datos son sintéticos.** Corpus, flota y reservas fueron construidos por el equipo:
   realistas para Cartagena, pero no son los datos reales de Vallis Marea. En consecuencia,
   **no hay comparación contra la línea base de conversaciones históricas de producción**, que
   era lo previsto en la propuesta.

2. **El router NO es fine-tuning de un LLM.** Es una regresión logística sobre embeddings
   **congelados**. Los pesos del modelo de embeddings no se modifican. Llamarlo fine-tuning
   sería incorrecto. Afinar un transformer pequeño (XLM-R o DistilBERT multilingüe) sobre
   este mismo dataset es el paso siguiente inmediato.

3. **No hubo prueba con usuarios reales.** El plan contemplaba una sesión con el equipo
   operativo; el desarrollo se comprimió a un día y se descartó.

4. **El aprendizaje continuo está implementado y es ejecutable, pero no alimentado con volumen
   real.** Se demuestra el mecanismo, no una curva de mejora.

5. **El golden set (35 preguntas) y el dataset del router (183 ejemplos) son de tamaño
   modesto.** Las métricas del §4 son representativas del mecanismo, pero un conjunto más
   grande —sobre todo para las clases `faq` y `ruta_turistica` del router, las que más error
   concentran— daría intervalos de confianza más ajustados.

6. **Los tres agentes corren como hilos de un mismo proceso** en el modo por defecto de la
   interfaz. La comunicación sigue siendo HTTP/JSON-RPC real —el protocolo se ejercita de
   verdad—, pero no se midió el efecto de la latencia de red entre máquinas distintas, que en
   producción sería mayor.

---

## 6. Qué se gana y qué se paga con MCP/A2A

Esta es la pregunta que la propuesta se comprometió a responder con honestidad.

### Se gana

- **Reutilización comprobable.** Los tres agentes son servidores MCP ejecutables de forma
  independiente. Cualquier cliente compatible puede usarlos sin conocer Vallis Marea.
- **Descubrimiento en vez de cableado.** El orquestador lee las capacidades del Agent Card.
  Agregar una habilidad no obliga a tocar el orquestador.
- **Trazabilidad por salto.** Se sabe qué agente se invocó, con qué herramienta y en cuántos
  ms. En el monolito esa información simplemente no existe.
- **Fronteras de falla.** Un salto A2A fallido se detecta y escala a un humano; en el nodo
  único, un error de una capacidad contamina toda la respuesta.

### Se paga

- **+7,7 ms por delegación (+116,7 %)** y ~0,4 s de descubrimiento por agente al arrancar la sesión.
- **Código de protocolo**: `core/a2a/` es infraestructura que en el monolito no existía.
- **Más superficie operativa**: tres servicios que pueden caerse por separado.
- **Trampas de medición** como la de §4.3, que en un monolito no pueden ocurrir.

### Veredicto

Para Vallis Marea la migración **se justifica**, y no por la latencia: menos de 8 ms son
despreciables frente al segundo largo de una llamada al LLM y frente a los timeouts de
webhook que el equipo ya enfrentó. Se justifica por la **trazabilidad y el desacoplamiento**,
que es precisamente lo que hoy le falta al nodo único cuando el negocio crece.

La reserva honesta: con tres agentes en una sola máquina, buena parte del beneficio es
potencial. El valor de MCP/A2A escala con el número de agentes, de equipos y de proveedores
distintos. En un sistema de tres agentes mantenidos por una persona, la misma separación de
responsabilidades se podría lograr con módulos bien definidos y sin protocolo. **Lo que los
protocolos abiertos compran no es la separación: es que la frontera sea pública y
consumible desde fuera.**

---

## 7. Trabajo siguiente

1. Fine-tuning real de un transformer pequeño para el router (cierra la brecha de §5.2).
2. Reemplazar `core/almacen/vectorial.py` por pgvector y medir de nuevo contra producción.
3. Replay de conversaciones históricas reales contra ambas arquitecturas — la comparación que
   la propuesta prometía y que los datos sintéticos no permiten.
4. Observabilidad con Langfuse: las trazas ya se generan, falta exportarlas.
5. Prueba con el equipo operativo del negocio.
6. Desplegar los agentes en procesos o máquinas separadas y medir la latencia real de red.
7. Ampliar el golden set y el dataset del router, sobre todo en `faq` y `ruta_turistica`.

---

## 8. Reproducibilidad

Instrucciones completas en [`README.md`](../README.md). Resumen:

```bash
python -m venv C:\envs\vallis_marea
C:\envs\vallis_marea\Scripts\python.exe -m pip install -r requirements.txt
echo GOOGLE_API_KEY=tu_clave > .env
C:\envs\vallis_marea\Scripts\python.exe -m core.ingesta.indexar --recrear
C:\envs\vallis_marea\Scripts\python.exe -m core.router.entrenar
C:\envs\vallis_marea\Scripts\streamlit.exe run app.py
```
