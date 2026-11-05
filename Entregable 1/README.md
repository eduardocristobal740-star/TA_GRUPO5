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

El aprovechamiento de bases aerodinámicas experimentales cuenta con aplicaciones documentadas en ingeniería estructural. Park et al. (2019) utilizaron registros de presión de la Tokyo Polytechnic University mediante el enfoque de diseño asistido por bases de datos para analizar la respuesta de un edificio alto ante diferentes direcciones del viento. Evaluaron momentos de vuelco e índices de demanda-capacidad de elementos estructurales. Este antecedente respalda la pertinencia de utilizar los registros de TPU para estudiar acciones del viento y destaca la importancia de conservar la relación entre las presiones, la geometría y la dirección de incidencia. Su aplicación al aprendizaje automático requiere una evaluación predictiva específica.

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

## Organización e interpretación de los datos experimentales

La base aerodinámica de TPU proporciona series temporales de coeficientes de presión en puntos de medición, disponibles en archivos de formato MATLAB (.mat). También presenta resultados estadísticos de los coeficientes locales, entre ellos valores medios, RMS, máximos y mínimos. Estas representaciones contienen información diferente sobre las presiones registradas, por lo que su identificación permite precisar qué magnitud se utilizará como respuesta del modelo predictivo (Tokyo Polytechnic University, s. f.).

La documentación de los datos distinguirá el modelo geométrico, la dirección del viento, los puntos de medición y las muestras temporales de cada registro. Esta distinción resulta necesaria porque múltiples mediciones procedentes de un mismo ensayo no equivalen a experimentos independientes. Asimismo, permitirá relacionar cada observación con sus condiciones experimentales y establecer posteriormente una evaluación predictiva que considere la dependencia entre los registros.

## Tipos de Datos

### Variables de entrada

| Variable | Tipo de dato | Definición | Unidad o referencia |
|---|---|---|---|
| Coordenada X | Numérico continuo | Posición del punto de medición respecto al eje X del modelo. | Unidad de longitud y origen definidos en los metadatos del ensayo. |
| Coordenada Y | Numérico continuo | Posición del punto de medición respecto al eje Y del modelo. | Unidad de longitud y origen definidos en los metadatos del ensayo. |
| Coordenada Z | Numérico continuo | Posición vertical del punto de medición. | Unidad de longitud y nivel de referencia definidos en el ensayo. |
| Altura del edificio (H) | Numérico continuo | Dimensión vertical del edificio representado en el ensayo. | Unidad de longitud del modelo, identificando su escala. |
| Ancho del edificio (B) | Numérico continuo | Dimensión horizontal identificada como ancho en la documentación experimental. | Misma unidad de longitud y escala que H y D. |
| Profundidad del edificio (D) | Numérico continuo | Dimensión horizontal identificada como profundidad en la documentación experimental. | Misma unidad de longitud y escala que H y B. |
| Relación H/B | Numérico continuo derivado | Cociente entre la altura y el ancho del edificio. | Adimensional; calculado con dimensiones expresadas en unidades y escalas compatibles. |
| Ángulo de incidencia del viento | Numérico angular | Orientación del viento respecto a la referencia establecida para el modelo. | Grados, según la convención angular del ensayo. |
| Velocidad de referencia | Numérico continuo | Velocidad del flujo utilizada como referencia para caracterizar el ensayo y normalizar las presiones. | Unidad de velocidad y altura de referencia indicadas en los metadatos. |
| Categoría de terreno | Categórico | Clasificación de la exposición o rugosidad del entorno representado experimentalmente. | Categorías establecidas por la fuente, sin unidad física. |

### Variable objetivo

| Variable | Tipo de dato | Definición | Unidad o referencia |
|---|---|---|---|
| Coeficiente de presión (Cp) | Numérico continuo | Presión relativa normalizada mediante una presión dinámica de referencia. | Adimensional; se identificará si corresponde al valor medio, RMS, máximo, mínimo o instantáneo. |

### Consideraciones sobre las variables

El esquema propuesto considera diez variables de entrada y una variable respuesta. Su disponibilidad y variación se comprobarán en los archivos seleccionados. Las coordenadas, dimensiones y velocidades conservarán las unidades y referencias documentadas por la fuente.

La relación H/B es una variable calculada a partir de la geometría. Las características que permanezcan constantes dentro del conjunto seleccionado se identificarán como condiciones del ensayo y se distinguirán de las variables que aporten información para la predicción.Tipos de Datos

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

Los coeficientes de presión caracterizan la acción aerodinámica sobre las superficies del edificio. Su utilización para evaluar la respuesta estructural requiere integrar esta información con las propiedades mecánicas y dinámicas de la edificación. Park et al. (2019) combinaron registros experimentales de presión con información del sistema estructural para determinar momentos de vuelco y demandas en elementos, considerando efectos dinámicos y de segundo orden.

En el presente proyecto, las predicciones constituirán información de apoyo para estudiar la distribución de los coeficientes de presión bajo las condiciones representadas en los datos. Su interpretación considerará las geometrías, direcciones del viento y condiciones experimentales utilizadas. La extensión de los resultados a configuraciones distintas requerirá una validación adicional, mientras que su incorporación al diseño estructural deberá considerar los demás parámetros y verificaciones correspondientes.

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

La procedencia experimental de los datos permite plantear una relación predictiva entre las características del edificio, las condiciones del viento y los coeficientes de presión. Sin embargo, las condiciones controladas de los ensayos no garantizan por sí solas un desempeño adecuado en observaciones nuevas. Según Yu y Barter (2024), la evaluación debe utilizar datos reservados que representen el escenario en el que se pretende aplicar el modelo.

En este proyecto, la separación de los datos considerará su procedencia experimental. Si se evalúa la predicción para geometrías nuevas, se reservarán modelos geométricos completos; si se evalúan direcciones del viento no utilizadas durante el entrenamiento, se reservarán direcciones completas dentro de las geometrías estudiadas. Esta planificación permitirá reducir el riesgo de obtener resultados excesivamente optimistas por distribuir mediciones relacionadas entre los conjuntos de entrenamiento y evaluación.

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

La viabilidad computacional se comprobará considerando el tamaño de los archivos seleccionados, las dimensiones de sus matrices y los recursos disponibles para el procesamiento. En los registros de presión se distinguirán los puntos de medición y las muestras temporales, debido a que ambos determinan el volumen de información que deberá manejarse. Si el conjunto supera la memoria disponible, el procesamiento se organizará por archivos o lotes.

Para facilitar la reproducción del análisis, se documentarán la versión de Python, las bibliotecas utilizadas, las características del equipo y las operaciones aplicadas a los datos. Los archivos originales se conservarán separados de los datos transformados, manteniendo la correspondencia entre cada resultado y su ensayo de procedencia. Estas decisiones permitirán evaluar la factibilidad del procedimiento y repetirlo bajo condiciones computacionales identificables, en concordancia con el marco PCS de Yu y Barter (2024).

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

La estabilidad se examinará mediante cambios razonables en la selección de ensayos, las particiones de evaluación y las decisiones de preprocesamiento. El remuestreo considerará la dependencia entre las mediciones: cuando se utilicen series temporales, se emplearán bloques que conserven su estructura temporal; cuando se comparen ensayos, se mantendrá identificada su procedencia experimental. La incorporación de ruido requerirá justificar las variables afectadas y la magnitud de la perturbación.

Se comparará la variación de las predicciones y de las métricas de desempeño entre los escenarios considerados. Además de los indicadores globales, se revisará si las diferencias se concentran en determinadas geometrías, direcciones del viento o puntos de medición. Esto permitirá identificar las condiciones en las que las conclusiones sean sensibles a las decisiones del análisis y delimitar el alcance de aplicación del modelo, siguiendo el principio de estabilidad de Yu y Barter (2024).

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

Tokyo Polytechnic University. (s. f.). TPU aerodynamic database [Base de datos]. Recuperado el 27 de octubre de 2026, de https://db.wind.arch.t-kougei.ac.jp/

La fuente primaria corresponde a la TPU Aerodynamic Database de la Tokyo Polytechnic University (s. f.).

Park, S., Duthinh, D., Simiu, E., & Yeo, D. (2019). Wind effects on a tall building with square cross-section and mid-side base columns: Database-assisted design approach. Journal of Structural Engineering, 145(5), Article 06019001. https://doi.org/10.1061/(ASCE)ST.1943-541X.0002328

# CREDITOS

-GEMINI 3.0 MODO AVANZADO
-COPYLOT EDUCATIVO

Wind Engineering Information Center (WEIC).

Tokyo Polytechnic University.

La fuente primaria corresponde a la TPU Aerodynamic Database de la Tokyo Polytechnic University. Este repositorio institucional organiza resultados de ensayos aerodinámicos en diferentes colecciones, entre ellas edificios altos aislados de planta rectangular o cuadrada, edificios de diversas configuraciones y casos de interferencia entre estructuras. La identificación de la colección y de los casos utilizados permitirá vincular las variables del proyecto con sus respectivos registros experimentales.

https://db.wind.arch.t-kougei.ac.jp/

## Bitácora de uso de inteligencia artificial

### Finalidad y herramientas utilizadas

Se utilizaron Gemini y posteriormente Copilot como herramientas de apoyo para la redacción, organización y programación del trabajo. El registro documenta las solicitudes empleadas para preparar el informe inicial y organizar su presentación en el repositorio.

| Herramienta | Uso documentado | Prompts textuales registrados | Refinamientos posteriores al primer prompt |
|---|---|---:|---:|
| Gemini | Orientación para elaborar el informe inicial y generación de contenido a partir de información y documentos de la fuente seleccionada. | 3 | 2 |
| Copilot | Revisión y mejora del informe generado con Gemini y preparación del contenido del README.md. | 2 | 1 |
| **Total** | **Preparación y organización del contenido inicial.** | **5** | **3** |

### Solicitudes realizadas a Gemini

#### Prompt 1

> Documento (Word o LaTeX): Informe inicial que describa el problema de ingenier´ıa estructural a resolver. Datos: Descripci´on detallada del dataset seleccionado (fuente, cantidad de variables, tipo de datos). Si tienen dificultades, pueden elegir uno de los datasets sugeridos en el Anexo. Marco VDS: Aplicaci´on de los principios de Veridical Data Science (PCS framework) basados en el trabajo de Bin Yu. Deben justificar la predictibilidad, computabilidad y estabilidad esperada de sus datos. Referencia obligatoria: Yu, B., & Barter, R. (2024). Veridical Data Science. MIT Press. https://vdsbook.com/
>
> se me pidio realizar eso en gifthub, ya cree el archivo readme y tengo los datos de entrada para buscar el problema de analisis, como debo crear el informe inicial?

#### Prompt 2

> toma en cuenta los siguientes datos para realizar el informe:
>
> Descripcion: Base de datos de la Universidad Politecnica de Tokio que contiene series temporales de presiones de viento medidas en tuneles de viento para modelos de edificios de diversas formas y alturas. Aplicacion: Modelos de regresion y analisis de series de tiempo para predecir coeficientes de presion y cargas de viento, evitando simulaciones costosas de dinamica de fluidos computacional (CFD). Enlace: http://wind.arch.t-kougei.ac.jp

#### Prompt 3

> los que tomes hazlo a partir de los pdf contenidos en el enlace que te envie

#### Documentación complementaria

Ante la imposibilidad de consultar directamente los PDF desde el enlace, se adjuntaron los documentos a Gemini. Con esta documentación, la herramienta generó un informe inicial.

### Solicitudes realizadas a Copilot

#### Prompt 1

> El siguiente informe fue creado por gemini, mejoralo en base al pdf cargado, el objetivo es conseguir que cumpla con lo siguientes puntos: PUNTOS SOLICITADOS: -Documento (Word o LaTeX): Informe inicial que describa el problema de ingenierıa estructural a resolver.

Se adjuntaron los PDF de la fuente para apoyar la revisión y mejora del contenido.

#### Prompt 2

> Crea un README.md listo para copiar.

### Alcance del registro

El conteo corresponde a los cinco prompts textuales documentados. La incorporación de archivos PDF se registra como suministro de documentación complementaria y no se contabiliza como un prompt textual adicional.

La revisión realizada con Copilot corresponde a una revisión asistida por IA. Este registro no documenta pruebas de ejecución ni auditorías humanas de código.
