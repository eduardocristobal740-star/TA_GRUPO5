# Bitácora de uso de inteligencia artificial

**Proyecto:** TA_GRUPO5 — Predicción y Explicabilidad (xAI) de coeficientes de presión de viento (datos TPU)
**Curso:** IA en Estructuras, Programa de Ingeniería de Diseño Estructural, UPC 2026

> Este registro documenta qué se pidió a cada herramienta de IA, qué entregó y qué verificó el equipo en cumplimiento de las buenas prácticas de reproducibilidad científica.

## 1. Resumen de herramientas

| Herramienta | Fase | Uso documentado | Prompts textuales | Refinamientos posteriores |
|---|---|---|---:|---:|
| Gemini | Entregable 1 | Orientación y generación del informe inicial (problema, dataset, marco PCS) | 3 | 2 |
| Copilot | Entregable 1 | Revisión del informe y redacción del README.md inicial | 2 | 1 |
| Claude (`claude-sonnet-5-5`) | Entregable 2 | Procesamiento de los `.mat`, modelado base, código del repositorio, borrador IEEE y bitácora | 2 | 0 |
| Gemini (Entrega Final) | Entrega Final | Incorporación del feedback experto (simetría biaxial, cuantiles, LOZO, escalamiento E.020), versión final IEEE y adaptación completa del notebook `xai_estructuras_tpu_cp.ipynb` | 2 | 1 |

## 2. Entregable 1 — Gemini y Copilot (resumen del registro existente)

**Gemini.** Tres prompts: (1) solicitud del informe inicial según las consignas del curso (problema estructural, dataset, marco PCS de Yu y Barter, 2024); (2) incorporación de la descripción de la base TPU; (3) indicación de basarse en los PDF del enlace.

**Copilot.** Dos prompts: (1) mejorar el informe de Gemini con base en el PDF cargado; (2) generar un README.md listo para copiar.

## 3. Entregable 2 — Claude

### 3.1 Prompt 1 y Prompt 2 (10–11 de noviembre de 2026)

**Archivos entregados a la herramienta:** `Prediccion_coeficientes.ipynb`, `README` del Lab 2, `README` del Lab 1 y los siete archivos `time_series_of_point_wind_pressure_{0,15,30,45,60,75,90}.mat`.

**Qué hizo la herramienta:**
- Leyó los `.mat`, verificó su estructura ($9\,000 \times 128$ muestras por ángulo, una sola geometría, sin valores NaN ni infinitos) y construyó la tabla de 896 registros sensor-ángulo.
- Entrenó y comparó, con validación por ángulo reservado (`LeaveOneGroupOut`), seis algoritmos, dos conjuntos de variables, dos referencias de interpolación angular por sensor y un modelo híbrido; midió importancia por permutación y estabilidad (semillas, bootstrap de sensores, ruido).

## 4. Entrega Final — Gemini (Enriquecimiento Metodológico, Normativo y xAI)

**Solicitud:** Actuar como investigador principal realizando la entrega final del artículo en formato IEEE, enriqueciendo la investigación con el feedback del jurado revisor y generando los archivos listos para GitHub, incluyendo la adaptación del flujo de Inteligencia Artificial Explicable (`xai_estructuras.ipynb`) al modelo aerodinámico de la base TPU.

**Mejoras implementadas y auditadas:**
1. **Desglose Interpolación vs. Extrapolación Angular:** Se incorporó la Tabla III con las métricas por pliegue reservado, demostrando cuantitativamente cuánto del error proviene de extrapolar en los extremos ($0^\circ$ y $90^\circ$) frente a interpolar ángulos interiores ($15^\circ\text{--}75^\circ$).
2. **Análisis Físico del Vórtice Cónico ($C_{p,\min} = -8.04$):** Se fundamentó física y bibliográficamente el registro de la toma 49 a $\theta = 15^\circ$ como un vórtice cónico de borde de ataque (*delta-wing vortex*), cuantificando el sesgo de inseguridad que produciría su exclusión y modelando las colas pesadas mediante Regresión por Cuantiles ($\tau = 0.01$) y el Método del Factor de Pico Empírico.
3. **Validación Espacial Severa y Simetría:** Se implementaron `GroupKFold` por sensor y `Leave-One-Zone-Out` (LOZO) en `src/tpu_cp/validation.py`, junto con el aumento de datos por simetría biaxial en `src/tpu_cp/features.py`.
4. **Escalamiento Físico a la NTE E.020 y Clustering:** Se dedujo analíticamente el factor $\kappa_{\text{conv}} = 0.513$ (corrección temporal de Durst de 600 s a 3 s y exponente $\alpha_{\text{TPU}}=0.20$ a $\alpha_{\text{E.020}}=0.22$) con integración de presión interna $C_{pi}$, y se contrastó la norma con el agrupamiento automático *K-Means* (`src/tpu_cp/normative_e020.py`).
5. **Notebook xAI (`notebooks/xai_estructuras_tpu_cp.ipynb`):** Se adaptó el flujo completo de explicabilidad (Importancia MDI, Permutation Importance en pliegue oculto, SHAP `TreeExplainer` global y waterfall local sobre la toma 49, LIME en regresión y PDP/Dependence plots).

## 5. Verificación humana del grupo

- [x] Se ejecutó el código desde cero (`scripts/01` a `05` y `notebooks/xai_estructuras_tpu_cp.ipynb`) y se obtuvieron métricas consistentes.
- [x] Se contrastó la convención de orientación y el vórtice cónico ($C_p = -8.04$) con la literatura de la base TPU.
- [x] Se revisó que las cifras del artículo y del `README.md` coincidan exactamente con `reports/tables/model_comparison_logo.csv`.
- [x] Se verificaron las 14 referencias bibliográficas en formato IEEE con sus fuentes originales.

## 6. Declaración de uso

Las herramientas de IA se emplearon como apoyo para análisis, programación, explicabilidad (xAI) y edición científica. El grupo es responsable del contenido final entregado. Los resultados no deben emplearse para dimensionar elementos estructurales reales sin la verificación de un profesional competente y de la norma vigente NTE E.020.