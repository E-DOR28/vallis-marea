# Línea base R0 (2026-10-04)

Re-ejecución de la suite y del router antes de cualquier cambio de código. Máquina: Apple M5 Pro, 24 GB, macOS. Python 3.13.13. Versiones exactas en `requirements.lock`. Modelos: `gemini-2.5-flash` y `gemini-embedding-001`.

## Insumos congelados

| Archivo | SHA-256 |
|---|---|
| `data/aprendizaje/dataset_router.json` | `4860684639f4d452cf371591fd921ea9bf0ce231ea572fbd87572d32207eb2b4` |
| `data/evaluacion/golden_set.json` | `3f95ce10e166975640df8ce7efc4c1d9ee62d38faa1d8e04014655708b7c3d11` |

Copia previa a las ejecuciones (corrida del 2026-09-13): `data/evaluacion/historico/*_20260913.*`. Corridas de hoy: `resultados_20261004_corrida{1,2,3}.json`.

## Resultados frente a la corrida anterior y a las tolerancias del build-spec (§8.2)

| Métrica | 2026-09-13 | Corrida 1 | Corrida 2 | Corrida 3 | Tolerancia mínima | Estado |
|---|---|---|---|---|---|---|
| recall de fuente@4 | 1,000 | 1,000 | 1,000 | 1,000 | 0,967 | Cumple |
| recall de sección@4 | 0,933 | 0,933 | 0,933 | 0,933 | 0,900 | Cumple |
| MRR | 0,842 | 0,842 | 0,842 | 0,842 | 0,800 | Cumple |
| precisión de citación | 1,000 | 1,000 | 0,967 | 0,967 | 0,967 | Cumple en el límite |
| abstención correcta | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | Cumple |
| tasa de respuesta | 0,967 | 0,933 | 0,933 | 0,933 | 0,933 | Cumple en el límite |
| exactitud del router | 0,870 | 0,870 (reentrenado) | | | | Idéntica |
| F1 macro del router | 0,847 | 0,847 (reentrenado) | | | 0,800 | Idéntica |

## Discrepancias y causa probable

1. **Recuperación y router: reproducen exactamente.** El nivel 1 no cambia en 4 corridas. El router reentrenado da las mismas métricas. Los coeficientes del `.npz` difieren en un máximo de 1,7e-5 (ruido numérico de float32), por eso el hash del archivo cambia.
2. **`tasa_respuesta` baja de 0,967 a 0,933 (g02), de forma persistente.** Una reproducción aislada de g02 en tres intentos dio tres abstenciones, con dos causas distintas:
   - Un intento devolvió "Error al consultar el modelo de lenguaje" (confianza 0,0): un error del LLM que el código convierte en abstención sin dejar rastro del error.
   - Dos intentos devolvieron una abstención legítima del modelo ("los fragmentos no especifican..."), con confianza 0,9 y 1,0. El 13 de septiembre el mismo modelo respondió.
   Causa probable: generación no determinista de `gemini-2.5-flash` más errores transitorios sin reintento. No se descarta un cambio en el comportamiento del modelo desde septiembre. No se investigó más porque R0 no modifica código.
3. **`precisión de citación` baja a 0,967 en las corridas 2 y 3** (un ítem sin cita correcta). Mismo origen: variación de la generación. El límite de tolerancia (0,967) deja cero margen.
4. **Latencia A2A.** Monolito p50 1,1 ms contra A2A p50 3,0 ms (sobrecosto de +1,9 ms, 172,7 %). El 13 de septiembre fue 6,6 ms contra 14,3 ms (+7,7 ms, 116,7 %). Cambió la máquina (de otra a Apple M5 Pro). La conclusión cualitativa se mantiene: A2A cuesta unos milisegundos frente a la llamada directa. El porcentaje sube porque el denominador es más pequeño.
5. **El arranque de la app Streamlit responde** (`/_stcore/health` devuelve `ok`).

## Implicaciones para las fases siguientes

- Con n = 30 preguntas que deben responderse, un ítem pesa 0,033. Las tolerancias de la §8.2 para `tasa_respuesta` y `precisión de citación` quedan en el límite del ruido. Para las comparaciones de R2 y R3 se promedian al menos 3 corridas y no se concluye por una diferencia de un ítem.
- R3 debe reintentar con espera creciente los errores transitorios del LLM, distinguir "error del modelo" de "abstención" en la respuesta y en los logs, y registrar el tipo de error sin datos sensibles.
