---
language:
  - es
license: mit
base_model: intfloat/multilingual-e5-small
library_name: transformers
pipeline_tag: text-classification
tags:
  - intent-classification
  - router
  - spanish
  - negative-result
---

# Router de intención afinado (resultado negativo): Vallis Marea

**Este modelo no se usa en producción.** Se publica como evidencia reproducible de un experimento cuyo resultado fue negativo. El servicio sigue usando un clasificador sobre embeddings de Gemini (`VM_ROUTER=embeddings`).

## Qué es

Es `intfloat/multilingual-e5-small` (revisión `614241f622f53c4eeff9890bdc4f31cfecc418b3`) afinado con **todos sus pesos** y una cabeza lineal de 9 clases. Clasifica el mensaje de un usuario en una intención:

`disponibilidad`, `faq`, `otro`, `politica`, `precio`, `recomendacion`, `reserva`, `ruta_turistica`, `saludo`.

Es parte de Vallis Marea, un asistente de alquiler de embarcaciones con varios agentes, construido para el curso SI7016 Procesamiento del Lenguaje Natural Aplicado (EAFIT, 2026-2). Es un encoder, no un LLM generativo.

## Resultado

El protocolo se registró antes de ver los resultados. La condición principal exigía que el modelo afinado no fuera peor que la línea base B0 por más de 0,02 de F1 macro (límite inferior del IC95 de la diferencia).

| Modelo | F1 macro |
|---|---|
| B0: regresión logística sobre embeddings congelados de Gemini (el router actual) | 0,863 |
| B2: Gemini `flash-lite` sin entrenar | 0,923 |
| Este modelo (e5-small afinado, protocolo v2) | 0,719 |

- Diferencia frente a B0: **−0,144** (IC95 −0,201 a −0,091). **La condición no se cumple.**
- Latencia p50: 4,8 ms (CPU local) contra 344,6 ms de B0. La de B0 incluye la red hacia Gemini, así que la comparación no es estricta.
- Decisión: **no se adopta**. La segunda ronda (v2) cambió solo la regla de parada y se declaró exploratoria.

## Datos y límites

- 183 ejemplos, en su mayoría sintéticos, con validación cruzada por grupos de 5 particiones y 3 repeticiones. SHA-256 del dataset: `4860684639f4d452cf371591fd921ea9bf0ce231ea572fbd87572d32207eb2b4`.
- Con tan pocos ejemplos y datos sintéticos, no hay base para concluir que más datos o más épocas cambiarían el resultado.
- No se evaluó fuera del dominio de alquiler de embarcaciones en Colombia.
- La latencia se midió en un solo Mac con chip arm64, un mensaje a la vez.

## Uso

El archivo `model.safetensors` contiene el encoder (prefijo `encoder.`) y la cabeza (prefijo `cabeza.`). `etiquetas.json` fija el orden de los logits, el prefijo `query: `, la longitud máxima (64), el promedio de tokens como pooling y el umbral de confianza (0,8406).

Se carga con `core/router/afinado.py` del repositorio del proyecto, que exige una revisión fijada:

```
VM_ROUTER=afinado
VM_ROUTER_MODELO=<usuario>/<repositorio>@<hash de 40 caracteres>
HF_TOKEN=<token de solo lectura>
```

## Licencia y atribución

MIT, igual que el modelo base. Modelo base: Wang et al., *Multilingual E5 Text Embeddings: A Technical Report*, arXiv:2402.05672.
