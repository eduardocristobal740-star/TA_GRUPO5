# TA_GRUPO5
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

- $P$ = presión de
