# Predicción de coeficientes de presión superficial con datos TPU

## 1. Descripción y alcance

Este entregable aborda la preparación y el análisis de datos experimentales
de presión superficial de la Tokyo Polytechnic University —TPU—, como base
para desarrollar modelos de regresión.

Las variables objetivo consideradas son el coeficiente de presión medio,
la fluctuación RMS y los coeficientes mínimo y máximo de cada serie temporal.

La selección publicada comprende siete direcciones del viento de una misma
geometría. Las tablas se calcularon a partir de los MAT de `data/raw/`.

## 2. Archivos principales

| Ruta | Contenido |
|---|---|
| `data/raw/` | Siete archivos MAT con series temporales de Cp |
| `scripts/01_limpieza_eda_tpu.py` | Limpieza y análisis inicial |
| `Prediccion_coeficientes.ipynb` | Estructura del laboratorio y fragmentos del flujo predictivo |
| `src/tpu_cp/` | Módulos del flujo general propuesto |
| `configs/schema.yml` | Configuración del procesamiento general |
| `requirements.txt` | Dependencias |
| `BITACORA_IA.md` | Registro del uso de inteligencia artificial |
| `REFERENCIAS.md` | Referencias del proyecto |

## 3. Características de los datos

Los metadatos y la documentación interna de los MAT indican:

| Característica | Valor |
|---|---|
| Tipo de cubierta | A dos aguas: gable roof |
| Altura al alero H | 4 m |
| Ancho B | 16 m |
| Profundidad D | 16 m |
| Relación H/B | 0.25 |
| Inclinación de cubierta | 5° |
| Direcciones del viento | 0°, 15°, 30°, 45°, 60°, 75° y 90° |
| Sensores por dirección | 128 |
| Superficies identificadas | 6 |
| Frecuencia de muestreo | 15 Hz |
| Duración por dirección | 600 s |
| Muestras por sensor y dirección | 9000 |

Las dimensiones corresponden a escala real, según la documentación interna.
Las series incorporan un promedio móvil de 0.2 s a escala real.

## 4. Inventario y control de calidad

| Archivo | Ángulo | Muestras por sensor | Sensores | Cp faltantes | Cp infinitos |
|---|---:|---:|---:|---:|---:|
| `time_series_of_point_wind_pressure_0.mat` | 0° | 9000 | 128 | 0 | 0 |
| `time_series_of_point_wind_pressure_15.mat` | 15° | 9000 | 128 | 0 | 0 |
| `time_series_of_point_wind_pressure_30.mat` | 30° | 9000 | 128 | 0 | 0 |
| `time_series_of_point_wind_pressure_45.mat` | 45° | 9000 | 128 | 0 | 0 |
| `time_series_of_point_wind_pressure_60.mat` | 60° | 9000 | 128 | 0 | 0 |
| `time_series_of_point_wind_pressure_75.mat` | 75° | 9000 | 128 | 0 | 0 |
| `time_series_of_point_wind_pressure_90.mat` | 90° | 9000 | 128 | 0 | 0 |

Las matrices contienen 8 064 000 valores de Cp. Al resumir cada serie
temporal se obtienen 896 registros: 128 sensores por siete direcciones.

| Comprobación adicional | Resultado |
|---|---|
| Coincidencia de ubicaciones entre archivos | Verificada |
| Identificadores de sensor duplicados dentro de cada archivo | 0 |
| Combinaciones sensor–ángulo duplicadas | 0 |
| Valores faltantes en la tabla consolidada | 0 |
| Geometrías diferentes | 1 |

Los registros corresponden a combinaciones sensor–ángulo. Su cantidad
no implica que sean observaciones estadísticamente independientes.

## 5. Definición de las estadísticas de presión

| Abreviatura | Columna generada por el script | Definición |
|---|---|---|
| `cp_mean` | `cp_mean_10min` | Media temporal de Cp |
| `cp_rms` | `cp_rms_fluctuation_10min` | Raíz de la media de las fluctuaciones cuadráticas respecto de la media |
| `cp_min` | `cp_min_0p2s_moving_average` | Menor Cp observado en la serie |
| `cp_max` | `cp_max_0p2s_moving_average` | Mayor Cp observado en la serie |

Las estadísticas se calculan mediante:

```python
cp_mean = np.mean(serie)
cp_rms = np.sqrt(np.mean((serie - cp_mean) ** 2))
cp_min = np.min(serie)
cp_max = np.max(serie)
```

Los cuatro coeficientes son adimensionales. Los extremos corresponden
a las series disponibles y a su procesamiento temporal.

## 6. Resumen estadístico

Estadísticas calculadas entre los 896 registros sensor–ángulo:

| Variable | Media | Desviación estándar | Mínimo | Mediana | Máximo |
|---|---:|---:|---:|---:|---:|
| `cp_mean` | -0.2786 | 0.4377 | -1.6956 | -0.2973 | 0.8319 |
| `cp_rms` | 0.2635 | 0.1241 | 0.1229 | 0.2193 | 0.9124 |
| `cp_min` | -1.6330 | 1.1167 | -8.0377 | -1.3147 | -0.1243 |
| `cp_max` | 0.6964 | 0.8324 | -0.2124 | 0.3644 | 3.5038 |

El promedio de `cp_mean` es negativo para el conjunto considerado.
El mínimo global registrado es -8.0377 y el máximo global es 3.5038.

Estos extremos requieren interpretación considerando el sensor,
la superficie, la dirección del viento y las características del ensayo.

## 7. Variación con la dirección del viento

Cada valor es el promedio de la estadística correspondiente entre
los 128 sensores de esa dirección.

| Ángulo | Sensores | Media de cp_mean | Media de cp_rms | Media de cp_min | Media de cp_max |
|---|---:|---:|---:|---:|---:|
| 0° | 128 | -0.3203 | 0.2782 | -1.7087 | 0.6684 |
| 15° | 128 | -0.2823 | 0.2686 | -1.6621 | 0.7233 |
| 30° | 128 | -0.2801 | 0.2658 | -1.6960 | 0.6852 |
| 45° | 128 | -0.2887 | 0.2794 | -1.7236 | 0.7317 |
| 60° | 128 | -0.2484 | 0.2524 | -1.5847 | 0.6598 |
| 75° | 128 | -0.2690 | 0.2573 | -1.5268 | 0.6895 |
| 90° | 128 | -0.2613 | 0.2426 | -1.5287 | 0.7170 |

Los promedios permiten comparar el comportamiento global por dirección.
El análisis por sensor y superficie es necesario para localizar las
presiones críticas.

## 8. Correlación de Spearman

Correlaciones calculadas entre las estadísticas de los 896 registros:

| Variable | cp_mean | cp_rms | cp_min | cp_max |
|---|---:|---:|---:|---:|
| `cp_mean` | 1.0000 | -0.2507 | 0.8124 | 0.6238 |
| `cp_rms` | -0.2507 | 1.0000 | -0.4555 | 0.3652 |
| `cp_min` | 0.8124 | -0.4555 | 1.0000 | 0.3157 |
| `cp_max` | 0.6238 | 0.3652 | 0.3157 | 1.0000 |

La mayor asociación positiva de esta matriz corresponde a `cp_mean`
y `cp_min`. Estas asociaciones describen los datos y no establecen
causalidad.

H, B, D y H/B son constantes en la selección actual; su correlación
no permite evaluar el efecto de cambios geométricos.

## 9. Distribución por superficie

Se conservan los identificadores de superficie de
`Location_of_measured_points`.

| Superficie | Sensores únicos | Registros sensor–ángulo | Promedio de cp_mean | Menor cp_min observado |
|---|---:|---:|---:|---:|
| 1 | 16 | 112 | 0.2480 | -4.2738 |
| 2 | 16 | 112 | 0.2406 | -4.5481 |
| 3 | 16 | 112 | -0.3934 | -3.8361 |
| 4 | 16 | 112 | -0.3942 | -3.5988 |
| 5 | 32 | 224 | -0.5436 | -8.0377 |
| 6 | 32 | 224 | -0.4213 | -6.2492 |
| **Total** | **128** | **896** | — | — |

La superficie 5 contiene la succión más negativa de la selección.

Los MAT almacenan dos coordenadas por punto, el identificador del sensor
y el identificador de superficie. La asignación a fachadas, faldones y
coordenadas tridimensionales debe verificarse con el plano de sensores TPU.

## 10. Estado del modelado predictivo

El repositorio contiene una propuesta para comparar DummyRegressor,
RandomForestRegressor y ExtraTreesRegressor.

| Análisis propuesto | Información que debe publicarse | Estado |
|---|---|---|
| Comparación de modelos | MAE, RMSE y R² por objetivo | Pendiente de ejecución y validación |
| Importancia por permutación | Cambio del MAE por variable y su variabilidad | Pendiente de ejecución |
| Observado frente a predicho | Valores observados, predicciones y residuos | Pendiente de ejecución |
| Estabilidad angular y espacial | Cambio absoluto medio y percentil 95 de las predicciones | Pendiente de ejecución |

El notebook publicado contiene 62 celdas Markdown y ninguna salida
ejecutada. Los fragmentos Python deben convertirse en celdas de código
y revisarse antes de ejecutar el laboratorio.

La selección actual representa una sola geometría. Para evaluar edificios
de geometrías no vistas se necesitan varias geometrías reales.

Los siete nombres de archivo identifican direcciones del viento; no
constituyen siete geometrías diferentes. Si se trabaja con esta selección,
debe definirse una validación angular o espacial acorde con ese alcance.

## 11. Reproducción del análisis inicial

Desde la raíz del repositorio:

```bash
cd "Entregable 2"
python -m pip install -r requirements.txt
python scripts/01_limpieza_eda_tpu.py
```

El script lee los siete MAT y genera:

| Archivo de salida | Contenido |
|---|---|
| `outputs/TPU_Gable_0-90_tabla_por_punto.csv` | Tabla consolidada de 896 registros |
| `outputs/cp_medio_por_direccion.png` | Promedio de Cp por dirección |
| `outputs/distribucion_cp_medio_por_direccion.png` | Distribución del Cp medio por dirección |
| `outputs/correlacion_variables.png` | Correlación de Pearson de las variables seleccionadas por el script |

La matriz de Spearman presentada en este README se calculó adicionalmente
a partir de la tabla consolidada. Puede reproducirse con:

```python
import pandas as pd

df = pd.read_csv("outputs/TPU_Gable_0-90_tabla_por_punto.csv")

columnas = {
    "cp_mean_10min": "cp_mean",
    "cp_rms_fluctuation_10min": "cp_rms",
    "cp_min_0p2s_moving_average": "cp_min",
    "cp_max_0p2s_moving_average": "cp_max",
}

correlacion = (
    df[list(columnas)]
    .rename(columns=columnas)
    .corr(method="spearman")
)

print(correlacion.round(4).to_string())
```

Los módulos generales de `src/tpu_cp/` requieren revisar la correspondencia
de variables y la representación espacial antes de utilizarlos con estos MAT.

## 12. Documentación complementaria

- [Bitácora de uso de IA](BITACORA_IA.md)
- [Referencias del proyecto](REFERENCIAS.md)
- [Notebook del laboratorio](Prediccion_coeficientes.ipynb)

Las tablas descriptivas se presentan directamente en Markdown.
Los resultados predictivos se incorporarán después de ejecutar y validar
el flujo correspondiente.
