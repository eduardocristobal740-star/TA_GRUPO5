# Bitácora de uso de inteligencia artificial

**Proyecto:** TA_GRUPO5 — Predicción de coeficientes de presión de viento (datos TPU)
**Curso:** IA en Estructuras, Programa de Ingeniería de Diseño Estructural, UPC 2026

> Este registro documenta qué se pidió a cada herramienta de IA, qué entregó y qué verificó el equipo.
> Los campos marcados **[COMPLETAR]** no pueden ser conocidos por la herramienta y deben ser llenados por el grupo antes de entregar.

## 1. Resumen de herramientas

| Herramienta | Fase | Uso documentado | Prompts textuales | Refinamientos posteriores |
|---|---|---|---:|---:|
| Gemini | Entregable 1 | Orientación y generación del informe inicial (problema, dataset, marco PCS) | 3 | 2 |
| Copilot | Entregable 1 | Revisión del informe y redacción del README.md | 2 | 1 |
| Claude (modelo configurado: `claude-sonnet-5-5`) | Entregable 2 | Procesamiento de los .mat, modelado, código del repositorio, artículo IEEE y esta bitácora | 2 | 0 |
| **[COMPLETAR]** otras herramientas usadas en el notebook original `Prediccion_coeficientes.ipynb` | Entregable 2 | **[COMPLETAR]** | **[COMPLETAR]** | **[COMPLETAR]** |

Las filas de Gemini y Copilot se trasladan del registro ya existente en el README del Entregable 1. El conteo de Claude corresponde a los dos mensajes de la sesión del 10–11 de noviembre de 2026; la lectura de archivos adjuntos no se cuenta como prompt textual.

## 2. Entregable 1 — Gemini y Copilot (resumen del registro existente)

**Gemini.** Tres prompts: (1) solicitud del informe inicial según las consignas del curso (problema estructural, dataset, marco PCS de Yu y Barter, 2024); (2) incorporación de la descripción de la base TPU; (3) indicación de basarse en los PDF del enlace. Como la herramienta no pudo consultar los PDF desde el enlace, se le adjuntaron los documentos.

**Copilot.** Dos prompts: (1) mejorar el informe de Gemini con base en el PDF cargado; (2) generar un README.md listo para copiar.

El texto textual de estos prompts está en el README del Entregable 1.

## 3. Entregable 2 — Claude

### 3.1 Prompt 1 (10 de noviembre de 2026)

**Archivos entregados a la herramienta:** `Prediccion_coeficientes.ipynb`, `README` del Lab 2, `README` del Lab 1 y los siete archivos `time_series_of_point_wind_pressure_{0,15,30,45,60,75,90}.mat`.

**Solicitud (resumen fiel):** actuar como científico de datos senior e ingeniero civil; analizar, limpiar y unificar los datos de los Laboratorios 1 y 2; construir un modelo de Machine Learning para predecir coeficientes de presión replicando la metodología del repositorio de referencia y mejorándola; comparar resultados con RMSE, R² y MAE; entregar código estructurado y comentado; y redactar un artículo en formato IEEE con Resumen, Introducción, Metodología, Resultados y Discusión, Conclusiones y Referencias.

**Qué hizo la herramienta:**

- Leyó los .mat, verificó su estructura (9 000 × 128 muestras por ángulo, una sola geometría, sin valores NaN ni infinitos) y construyó la tabla de 896 registros sensor-ángulo; las estadísticas coincidieron con las del README del Lab 2.
- Entrenó y comparó, con validación por ángulo reservado (LeaveOneGroupOut), seis algoritmos, dos conjuntos de variables, dos referencias de interpolación angular por sensor y un modelo híbrido; ajustó hiperpárametros de Extra Trees e HistGradientBoosting; midió importancia por permutación y estabilidad (semillas, bootstrap de sensores, ruido).
- Escribió el repositorio modular (`src/tpu_cp/`, scripts numerados, pruebas, README) y generó las figuras.
- Redactó el artículo en formato IEEE (documento editable) con las cifras obtenidas de las ejecuciones.
- Todo el código se ejecutó en un entorno aislado de la sesión y las cifras del artículo provienen de esas ejecuciones, no de memoria de la herramienta.

**Limitaciones declaradas por la herramienta:**

- Solo se entrenó con los datos del Laboratorio 2: el material del Laboratorio 1 es un planteamiento del problema sin registros.
- No pudo abrir el artefacto de referencia de Claude; del repositorio del curso y del repositorio TA_GRUPO5 solo se vio la estructura, no los resultados.
- La convención de orientación del viento (muro 1 a barlovento a 0° y muro 2 a 90°) se infirió del signo del Cp medio y no de la documentación de TPU.
- No se incluyó XGBoost (no estaba disponible en el entorno).
- El mejor modelo por objetivo se eligió con los mismos pliegues con los que se reporta, por lo que las cifras son algo optimistas.
- La asistencia de IA no sustituye la NTE E.020 ni la revisión de un ingeniero estructural.

### 3.2 Prompt 2 (11 de noviembre de 2026)

**Solicitud:** elaborar una bitácora de uso de IA para el repositorio `TA_GRUPO5`.

**Resultado:** este documento. Se consultó la página pública del repositorio; no se pudo confirmar si ya existe un archivo `BITACORA_IA.md`, por lo que conviene revisarlo antes de reemplazarlo.

## 4. Verificación humana del grupo

La IA no puede certificar esta sección. Marquen lo que efectivamente hicieron:

- [ ] Se ejecutó el código desde cero (`scripts/01` a `05`) en un equipo del grupo y se obtuvieron las mismas métricas.
- [ ] Se contrastó la convención de orientación y las coordenadas con el plano de tomas de la base TPU.
- [ ] Se revisó que las cifras del artículo coincidan con `reports/tables/`.
- [ ] Se verificaron las referencias bibliográficas con sus fuentes originales.
- [ ] Se revisó el texto del artículo y se corrigió o amplió donde fue necesario.
- **Responsable(s) de la revisión:** **[COMPLETAR]**
- **Fecha:** **[COMPLETAR]**

## 5. Declaración de uso

Las herramientas de IA se emplearon como apoyo para análisis, programación y redacción. El grupo es responsable del contenido final entregado. Los resultados no deben emplearse para dimensionar elementos estructurales reales sin la verificación de un profesional competente y de la norma vigente.
