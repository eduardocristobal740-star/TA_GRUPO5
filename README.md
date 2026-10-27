TA_GRUPO5
Este repositorio corresponde a la TA del curso IA en Estructuras del Programa de ingeniería de diseño estructural UPC 2026
# Predicción de Coeficientes de Presión de Viento en Edificaciones mediante Veridical Data Science (VDS)

## Descripción del Proyecto

Las cargas de viento son una de las acciones más importantes en el diseño estructural de edificios. La estimación incorrecta de estas cargas puede generar daños en fachadas, cubiertas y elementos estructurales, afectando la seguridad y el desempeño de las edificaciones.

Este proyecto propone la aplicación de técnicas de Ciencia de Datos para predecir coeficientes de presión superficial ($C_p$) utilizando datos experimentales obtenidos en túneles de viento de la **Tokyo Polytechnic University (TPU)**.

El análisis se desarrolla siguiendo los principios del marco **Veridical Data Science (VDS)** y la metodología **PCS (Predictability, Computability y Stability)** propuesta por Yu y Barter (2024).

---

# Problema de Ingeniería Estructural

La evaluación de cargas de viento suele realizarse mediante:

- Ensayos experimentales en túneles de viento.
- Simulaciones CFD (Computational Fluid Dynamics).
- Procedimientos normativos.

Sin embargo, estas metodologías presentan altos costos computacionales y experimentales.

El objetivo de este proyecto es desarrollar un modelo predictivo capaz de estimar coeficientes de presión aerodinámica a partir de variables geométricas y condiciones de viento, reduciendo la necesidad de realizar simulaciones complejas para cada nueva configuración estructural.

---

# Objetivos

## Objetivo General

Desarrollar modelos predictivos basados en datos experimentales de túnel de viento para estimar coeficientes de presión en edificaciones.

## Objetivos Específicos

- Analizar las variables aerodinámicas disponibles en la base de datos seleccionada.
- Implementar modelos de aprendizaje automático para la predicción de $C_p$.
- Evaluar el desempeño de los modelos mediante métricas estadísticas.
- Aplicar el marco PCS para garantizar resultados reproducibles y confiables.
- Analizar la estabilidad de las predicciones frente a perturbaciones en los datos.

---

# Dataset

## Fuente

Wind Engineering Information Center (WEIC)

Tokyo Polytechnic University (TPU)

## Descripción

El conjunto de datos proviene de experimentos realizados en túneles de viento atmosféricos y contiene mediciones de presión superficial sobre modelos de edificios sometidos a distintas condiciones aerodinámicas.

La base de datos incluye:

- Edificios aislados.
- Edificios altos.
- Conjuntos de edificios.
- Casos de interferencia entre estructuras.
- Series temporales de presión.
- Mediciones experimentales multicanal.

## Tipos de Datos

### Variables de Entrada

| Variable | Tipo |
|-----------|---------|
| Coordenada X | Numérico continuo |
| Coordenada Y | Numérico continuo |
| Coordenada Z | Numérico continuo |
| Altura del edificio (H) | Numérico continuo |
| Ancho del edificio (B) | Numérico continuo |
| Profundidad del edificio (D) | Numérico continuo |
| Relación H/B | Numérico continuo |
| Ángulo de incidencia del viento | Numérico continuo |
| Velocidad de referencia | Numérico continuo |
| Categoría de terreno | Categórico |

### Variable Objetivo

| Variable | Tipo |
|-----------|---------|
| Coeficiente de Presión ($C_p$) | Numérico continuo |

## Aplicación Estructural

Los coeficientes de presión permiten calcular las presiones de diseño mediante:

$$
P=qC_p
$$

donde:

- $P$ = presión de diseño
- $q$ = presión dinámica
- $C_p$ = coeficiente de presión

Estos valores son utilizados en:

- Diseño de fachadas.
- Diseño de cubiertas.
- Evaluación de cargas laterales.
- Verificación de normativas estructurales.

---

# Aplicación del Marco PCS

## 1. Predictability (Predictibilidad)

La predictibilidad evalúa la capacidad de los datos para generar modelos con buen desempeño sobre observaciones no utilizadas durante el entrenamiento.

### Estrategia

- División entrenamiento/prueba.
- Validación cruzada.
- Comparación de modelos supervisados.

### Métricas

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- $R^2$ (Coeficiente de Determinación)

### Justificación

Se espera una alta capacidad predictiva debido a que:

- Los fenómenos aerodinámicos siguen principios físicos conocidos.
- Los datos fueron obtenidos bajo condiciones experimentales controladas.
- Existe una relación física directa entre geometría, dirección del viento y presión superficial.

---

## 2. Computability (Computabilidad)

La computabilidad evalúa la viabilidad de implementar y ejecutar el análisis con recursos computacionales razonables.

### Herramientas

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Matplotlib
- Seaborn

### Flujo de Trabajo

```text
Datos
 ↓
Preprocesamiento
 ↓
Ingeniería de Variables
 ↓
Entrenamiento
 ↓
Evaluación
 ↓
Análisis de Resultados
```

### Justificación

El volumen de datos puede ser procesado mediante equipos convencionales disponibles en entornos universitarios, permitiendo la implementación eficiente de modelos predictivos.

---

## 3. Stability (Estabilidad)

La estabilidad evalúa la robustez de las conclusiones frente a perturbaciones razonables en los datos o en las decisiones de modelado.

### Estrategias

- Adición de ruido sintético.
- Técnicas Bootstrap.
- Remuestreo aleatorio.
- Comparación entre algoritmos.

### Modelos Evaluados

- Regresión Lineal
- Random Forest
- Gradient Boosting
- XGBoost

### Justificación

Si pequeñas perturbaciones en los datos producen cambios reducidos en las métricas de desempeño y en la importancia de las variables, las conclusiones serán consideradas estables.

---

# Documentación PCS

Siguiendo las recomendaciones de Veridical Data Science, se documentarán explícitamente las decisiones tomadas durante:

- Selección del dataset.
- Limpieza de datos.
- Tratamiento de valores faltantes.
- Ingeniería de características.
- Selección de modelos.
- Ajuste de hiperparámetros.
- Evaluación de desempeño.

Esta documentación permitirá asegurar:

- Transparencia.
- Reproducibilidad.
- Interpretabilidad.

---

# Estructura del Repositorio

```text
Proyecto-VDS-WindEngineering/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── src/
│   ├── preprocessing.py
│   ├── training.py
│   └── evaluation.py
│
├── results/
│
├── figures/
│
├── README.md
│
└── requirements.txt
```

---

# Resultados Esperados

- Predicción precisa de coeficientes de presión aerodinámica.
- Evaluación de la capacidad predictiva de distintos algoritmos.
- Validación de los principios Predictability, Computability y Stability.
- Desarrollo de un flujo de trabajo reproducible aplicable a problemas de ingeniería estructural.

---

# Referencias

Yu, B., & Barter, R. (2024).

**Veridical Data Science.**

MIT Press.

https://vdsbook.com/

Tamura, Y.

**Global COE Program: New Frontier of Education and Research in Wind Engineering.**

Tokyo Polytechnic University.

# CREDITOS

-GEMINI 3.0 MODO AVANZADO
-COPYLOT EDUCATIVO

Wind Engineering Information Center (WEIC).

Tokyo Polytechnic University.

La fuente primaria corresponde a la TPU Aerodynamic Database de la Tokyo Polytechnic University. Este repositorio institucional organiza resultados de ensayos aerodinámicos en diferentes colecciones, entre ellas edificios altos aislados de planta rectangular o cuadrada, edificios de diversas configuraciones y casos de interferencia entre estructuras. La identificación de la colección y de los casos utilizados permitirá vincular las variables del proyecto con sus respectivos registros experimentales.

https://db.wind.arch.t-kougei.ac.jp/
