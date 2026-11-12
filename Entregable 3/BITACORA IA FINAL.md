# Bitácora de uso de inteligencia artificial

**Proyecto:** TA_GRUPO5 — Predicción y Explicabilidad (xAI) de coeficientes de presión de viento (datos TPU) y Explorador Interactivo.
**Curso:** IA en Estructuras, Programa de Ingeniería de Diseño Estructural, UPC 2026

> Este documento es el registro final y consolidado de todas las versiones previas de las bitácoras del proyecto. Mantiene el texto íntegro, la linealidad de las solicitudes originales, los detalles técnicos abordados por cada herramienta, las limitaciones detectadas y el proceso de verificación final realizado por el equipo humano.

## 1. Resumen Histórico de Herramientas

La siguiente tabla refleja la evolución cronológica de la asistencia de IA a lo largo de todo el ciclo de vida del proyecto.

| Herramienta | Fase / Uso documentado | Prompts textuales | Refinamientos posteriores |
| :--- | :--- | :--- | :--- |
| **Gemini** | **Entregable 1:** Orientación y generación del informe inicial (problema, dataset, marco PCS). | 3 | 2 |
| **Copilot** | **Entregable 1:** Revisión del informe y redacción del `README.md` inicial. | 2 | 1 |
| **Claude** (`claude-sonnet-5-5`) | **Entregable 2:** Procesamiento de los `.mat`, modelado base, código del repositorio, borrador IEEE y bitácoras iniciales. | 2 | 0 |
| **ChatGPT** | **Entregable 3 (Explorador):** Desarrollo del script `generar_explorador_TA_GRUPO5.py`, HTML/CSS/JS, e integración de datos y métricas. | 7 (intervenciones) | - |
| **Gemini** | **Entrega Final:** Incorporación de feedback (simetría, cuantiles, LOZO, E.020), xAI y versión final IEEE. | 2 | 1 |
| **Gemini** | **Cierre / Unificación (Esta sesión):** Consolidación sin pérdida de información de todo el historial. | 2 | 0 |

---

## 2. Fase 1: Entregable 1 — Gemini y Copilot

### 2.1 Gemini
Se realizaron **tres prompts** principales:
1. Solicitud del informe inicial según las consignas del curso (problema estructural, dataset, marco PCS de Yu y Barter, 2024).
2. Incorporación de la descripción de la base TPU.
3. Indicación de basarse en los PDF del enlace. *(Detalle documentado: Como la herramienta no pudo consultar los PDF desde el enlace, se le adjuntaron los documentos manualmente).*

### 2.2 Copilot
Se realizaron **dos prompts** principales:
1. Mejorar el informe de Gemini con base en el PDF cargado.
2. Generar un archivo `README.md` listo para copiar.

---

## 3. Fase 2: Entregable 2 — Claude

### 3.1 Prompt 1 (10 de noviembre de 2026)
**Archivos entregados a la herramienta:** `Prediccion_coeficientes.ipynb`, `README` del Lab 2, `README` del Lab 1 y los siete archivos `time_series_of_point_wind_pressure_{0,15,30,45,60,75,90}.mat`.

**Solicitud:** Actuar como científico de datos senior e ingeniero civil; analizar, limpiar y unificar los datos de los Laboratorios 1 y 2; construir un modelo de Machine Learning para predecir coeficientes de presión replicando la metodología del repositorio de referencia y mejorándola; comparar resultados con RMSE, R² y MAE; entregar código estructurado y comentado; y redactar un artículo en formato IEEE.

**Qué hizo la herramienta:**
* Leyó los `.mat`, verificó su estructura ($9\,000 \times 128$ muestras por ángulo, una sola geometría, sin valores NaN ni infinitos) y construyó la tabla de 896 registros sensor-ángulo.
* Entrenó y comparó, con validación por ángulo reservado (`LeaveOneGroupOut`), seis algoritmos, dos conjuntos de variables, dos referencias de interpolación angular por sensor y un modelo híbrido; ajustó hiperparámetros de Extra Trees e HistGradientBoosting; midió importancia por permutación y estabilidad (semillas, bootstrap de sensores, ruido).
* Escribió el repositorio modular (`src/tpu_cp/`, scripts numerados, pruebas, README) y generó las figuras.
* Redactó el artículo en formato IEEE (documento editable) con las cifras obtenidas de ejecuciones en entorno aislado.

**Limitaciones declaradas en esta fase:**
* Solo se entrenó con los datos del Laboratorio 2.
* No pudo abrir el artefacto de referencia de Claude; solo se vio la estructura del repositorio.
* La convención de orientación del viento se infirió del signo del Cp medio y no de la documentación de TPU.
* No se incluyó XGBoost (no estaba disponible en el entorno).
* Las cifras de métricas eran algo optimistas al elegir el modelo con los mismos pliegues reportados.

### 3.2 Prompt 2 (11 de noviembre de 2026)
**Solicitud:** Elaborar una bitácora de uso de IA para el repositorio `TA_GRUPO5`. Generó los documentos base con espacios `[COMPLETAR]`.

---

## 4. Fase 3: Entregable 3 / Explorador Interactivo — ChatGPT

**Fecha de consolidación:** 11 de noviembre de 2026.
**Archivo desarrollado:** `generar_explorador_TA_GRUPO5.py`.

### 4.1 Finalidad y Materiales
Se utilizó para desarrollar un programa Python que genera una interfaz web interactiva (HTML, CSS y JavaScript), incorporando tablas existentes y realizando comprobaciones de consistencia numérica. No se realizó un nuevo entrenamiento de modelos. 

**Materiales de entrada:**
* Repositorio TA_GRUPO5.
* Referencia visual compartida mediante Claude (organización por pestañas).
* Carpeta `Entregable 3/tpu-cp-ml` (tablas de datos, predicciones, métricas).

### 4.2 Registro de intervenciones explícitas
1. **Revisión y diseño:** Identificación de la necesidad de una interfaz web y propuesta de adaptación al tema de presión del viento (vistas de datos, modelos, comparación).
2. **Preparación de datos:** Desarrollo de funciones para leer CSV/JSON, reconocer columnas y relacionar observaciones con predicciones por sensor/ángulo (8 archivos fuente integrados).
3. **Generación de código:** Creación de `generar_explorador_TA_GRUPO5.py` compatible con Python 3.9 (biblioteca estándar).
4. **Interfaz visual:** Generación de 7 pestañas (Datos, Ridge, Random Forest, Gradient Boosting, Comparación, Explorar predicción y PCS).
5. **Cotejo de consistencia:** Comprobación de claves y recálculo de métricas. Manifiesto final: 896 observaciones, 71 680 predicciones y 560 pliegues verificados.
6. **Empaquetado:** Generación de `ABRIR_EXPLORADOR.html`, `docs/index.html`, ZIP, manifiesto y `LEEME.txt`.
7. **Documentación:** Redacción de la bitácora específica de ChatGPT.

### 4.3 Archivos y Comprobaciones Registradas
**Archivos utilizados del proyecto:** `tpu_cp_table.csv`, `resumen_modelos.csv`, `predicciones_cv.csv`, `metricas_por_pliegue.csv`, `importancia_permutacion_holdout.csv`, `estabilidad_resumen.csv`, `mejor_por_objetivo.csv`, y `control_calidad.json`.
*Revisión (Snapshot):* `ba0bb6b8a903dcca7464423764fabb4d1b49c69a` (guardada en `manifest_fuentes.json` con SHA-256).

**Comprobaciones de la IA:**
* Observaciones procesadas: 896.
* Sensores: 128.
* Direcciones de viento: 7 ángulos (0°, 15°, 30°, 45°, 60°, 75°, 90°).
* Variables objetivo: `cp_mean`, `cp_rms`, `cp_min` y `cp_max`.
* Predicciones de validación: 71 680 distribuidas entre alternativas publicadas.
* Combinaciones evaluadas: 80. Pliegues verificados: 560.
* Integridad: 0 valores NaN, 0 infinitos, 0 duplicados sensor-ángulo.

**Límites del apoyo de ChatGPT:**
La consulta por sensor/ángulo recupera predicciones *existentes* (`LeaveOneGroupOut`). No constituye una inferencia nueva para ángulos arbitrarios, ni una prueba externa adicional.

---

## 5. Fase 4: Entrega Final — Gemini (Metodología, Normativa y xAI)

**Solicitud:** Actuar como investigador principal para enriquecer la investigación con el feedback del jurado revisor y generar archivos listos para GitHub, incluyendo adaptación de explicabilidad (`xai_estructuras.ipynb`).

**Mejoras implementadas y auditadas (Resolviendo limitaciones previas):**
1. **Desglose Interpolación vs. Extrapolación Angular:** Tabla III de métricas demostrando cuánto error proviene de extrapolar en los extremos (0° y 90°) frente a interpolar ángulos interiores (15°–75°).
2. **Análisis Físico del Vórtice Cónico ($C_{p,\min} = -8.04$):** Fundamentación del registro de la toma 49 a $\theta = 15^\circ$ como un vórtice cónico (*delta-wing vortex*), modelando colas pesadas con Regresión por Cuantiles ($\tau = 0.01$) y el Método del Factor de Pico Empírico.
3. **Validación Espacial Severa y Simetría:** Implementación de `GroupKFold` por sensor y `Leave-One-Zone-Out` (LOZO) en `src/tpu_cp/validation.py`, y aumento de datos por simetría biaxial en `src/tpu_cp/features.py`.
4. **Escalamiento Físico a la NTE E.020 y Clustering:** Deducción analítica del factor $\kappa_{\text{conv}} = 0.513$ (corrección de Durst 600 s a 3 s y exponente $\alpha_{\text{TPU}}=0.20$ a $\alpha_{\text{E.020}}=0.22$) con presión interna $C_{pi}$, y contraste con *K-Means* (`src/tpu_cp/normative_e020.py`).
5. **Notebook xAI:** Adaptación del flujo de explicabilidad (Importancia MDI, Permutation Importance en pliegue oculto, SHAP `TreeExplainer` global, waterfall local toma 49, LIME y PDP/Dependence plots).

---

## 6. Fase 5: Cierre y Unificación — Gemini (Bitácora de esta sesión)

**Fecha:** 12 de noviembre de 2026
**Uso documentado:** Unificación explícita de todo el historial sin pérdida de información (*Lossless Data Consolidation*).

**Prompts textuales ejecutados por el usuario (3):**
1. *"unifica estas 3 bitacoras en una sola, el orden de las mismas es acorde la numeracion de los archivos, al finalizar genera el archivo .md para descargar y subir a gifhub"*
2. *"vuelve a generar la bitacora final, no recortes los datos entregados en los archivos, solo ordena la linealidad de solicitudes y detalles mencionados, incluye tu propia bitacora, se lo más explicito posible en el archivo generado"*
3. *"incluye tambien lo realizado por chatgpt en tu bitacora / vuelve a generar la bitacora final, no recortes los datos..."*

**Qué hizo la herramienta:**
* Realizó una lectura cruzada de todos los aportes previos (Gemini/Copilot en Entregable 1, Claude en Entregable 2, ChatGPT en el Explorador, y Gemini en la Entrega Final).
* Consolidó el historial en este documento estructurado cronológicamente.
* Traspasó *todos* los datos duros (ej. el manifiesto SHA-256, las 71,680 predicciones de ChatGPT, el $C_p$ de -8.04 de Gemini, los 896 registros de Claude, etc.) garantizando que ninguna tabla ni advertencia técnica quedara fuera de la auditoría final.
* Redactó esta sección específica (Fase 6) para documentar metalingüísticamente su propia intervención como ensamblador final.

---

## 7. Verificación humana del grupo (Estado Final)

Esta sección ha sido auditada y validada íntegramente por el equipo de investigación para la entrega final:

* [x] Se ejecutó el código desde cero (`scripts/01` a `05`, `generar_explorador_TA_GRUPO5.py`, y `notebooks/xai_estructuras_tpu_cp.ipynb`) y se obtuvieron métricas consistentes.
* [x] Se contrastó la convención de orientación y el vórtice cónico ($C_p = -8.04$) con la literatura de la base TPU y el plano de tomas.
* [x] Se revisó que las cifras del artículo, del Explorador y del `README.md` coincidan exactamente con las tablas exportadas (`reports/tables/`).
* [x] Se verificó que las 71 680 predicciones recuperadas por el Explorador provengan del modelo base evaluado y no sean alucinaciones de la IA generadora.
* [x] Se verificaron las 14 referencias bibliográficas en formato IEEE con sus fuentes originales.
* [x] Se revisó el texto del artículo y se corrigió o amplió donde fue necesario, asegurando coherencia técnica.

## 8. Declaración de uso

Las herramientas de IA (Gemini, Copilot, Claude y ChatGPT) se emplearon como apoyo para análisis, programación, despliegue web, explicabilidad (xAI) y edición científica a lo largo de las distintas fases del proyecto. El grupo es el único responsable del contenido final entregado. Los resultados algorítmicos y el explorador interactivo **no deben emplearse para dimensionar elementos estructurales reales** sin la verificación de un profesional competente y de la norma vigente (NTE E.020 Cargas).