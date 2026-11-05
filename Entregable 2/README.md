# Predicción de coeficientes de presión superficial con datos TPU

Este repositorio implementa un flujo reproducible para transformar ensayos de presión de viento de la **Tokyo Polytechnic University, TPU**, realizar análisis exploratorio de datos y entrenar modelos de aprendizaje automático para estimar:

- `cp_mean`: coeficiente de presión medio.
- `cp_rms`: variabilidad de la presión.
- `cp_min`: pico mínimo o succión crítica.
- `cp_max`: pico máximo o sobrepresión crítica.

El proyecto evita la fuga de datos mediante divisiones agrupadas por geometría o ensayo, incluye importancia por permutación y evalúa la estabilidad de las predicciones frente a perturbaciones del ángulo del viento y de la posición del sensor.

## 1. Estructura del repositorio

```text
tpu_cp_ml/
├── README.md
├── REFERENCIAS.md
├── requirements.txt
├── pyproject.toml
├── configs/
│   └── schema.yml
├── data/
├── models/
├── reports/
│   └── figures/
└── src/
    └── tpu_cp/
        ├── io.py
        ├── inspect_mat.py
        ├── build_dataset.py
        ├── eda.py
        ├── train.py
        └── visualize.py
```

## 2. Instalación

```bash
python -m venv .venv
```

En Linux o macOS:

```bash
source .venv/bin/activate
```

En Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Instalar el proyecto:

```bash
pip install -r requirements.txt
pip install -e .
```

## 3. Incorporación de los datos

Depositar los archivos `.mat`, `.csv` o `.parquet` dentro de:

```text
data/
```

Los datos originales no se versionan porque pueden ser pesados y estar sujetos a las condiciones de uso de la base TPU.

## 4. Inspección de archivos MAT

Antes de consolidar los datos, inspeccionar las variables internas:

```bash
python -m tpu_cp.inspect_mat \
  --data-dir data \
  --out reports/mat_inventory.json
```

Revisar `reports/mat_inventory.json` y adaptar los alias de `configs/schema.yml` si los archivos usan nombres diferentes para `Cp`, `H`, `B`, `D`, `theta`, `X`, `Y` o `Z`.

## 5. Construcción de la matriz consolidada

```bash
python -m tpu_cp.build_dataset \
  --data-dir data \
  --config configs/schema.yml \
  --out data/processed.parquet
```

La matriz consolidada contiene:

```text
H, B, D, H_B, theta, X, Y, Z,
sensor_id, geometry_id, experiment_id,
cp_mean, cp_rms, cp_min, cp_max
```

Si el archivo contiene series temporales, las estadísticas se calculan por sensor. Los extremos `cp_min` y `cp_max` representan los extremos observados durante la duración del ensayo y no deben interpretarse directamente como valores universales de diseño.

## 6. Análisis exploratorio de datos

```bash
python -m tpu_cp.eda \
  --data data/processed.parquet \
  --config configs/schema.yml \
  --out-dir reports/figures/eda
```

El módulo EDA produce:

- Porcentaje de datos faltantes.
- Histogramas de `cp_mean`, `cp_rms`, `cp_min` y `cp_max`.
- Matriz de correlación de Spearman.
- Variación de los objetivos con el ángulo del viento.
- Mapa espacial `X-Z` coloreado por `cp_min`.
- Cantidad de observaciones por geometría.
- Resumen estadístico con percentiles 1, 5, 50, 95 y 99.

### Figuras EDA

> Las siguientes imágenes aparecerán después de ejecutar el módulo EDA y subir la carpeta `reports` a GitHub.

#### Datos faltantes

![Datos faltantes](reports/figures/eda/01_datos_faltantes.png)

#### Distribuciones de las variables objetivo

![Distribuciones de Cp](reports/figures/eda/02_distribuciones_objetivos.png)

#### Correlación de Spearman

![Correlación de Spearman](reports/figures/eda/03_correlacion_spearman.png)

#### Coeficientes de presión según el ángulo del viento

![Cp frente al ángulo](reports/figures/eda/04_objetivos_vs_angulo.png)

#### Distribución espacial de Cp mínimo

![Mapa espacial de Cp mínimo](reports/figures/eda/05_mapa_espacial_cp_min.png)

#### Observaciones por geometría

![Observaciones por grupo](reports/figures/eda/06_observaciones_por_grupo.png)

## 7. Entrenamiento y validación

```bash
python -m tpu_cp.train \
  --data data/processed.parquet \
  --config configs/schema.yml \
  --out-dir models
```

Se comparan los siguientes modelos:

1. `DummyRegressor` con estrategia de mediana.
2. `RandomForestRegressor`.
3. `ExtraTreesRegressor`.

La selección se realiza con el menor MAE de validación cruzada mediante `GroupKFold`. La prueba externa se construye con `GroupShuffleSplit`, de modo que una geometría completa no aparezca simultáneamente en entrenamiento y prueba.

Las métricas reportadas son:

- MAE.
- RMSE.
- R².
- MAE de validación cruzada agrupada.

## 8. Importancia y visualización de resultados

```bash
python -m tpu_cp.visualize \
  --models-dir models \
  --out-dir reports/figures/modelos
```

Este comando genera:

- Comparación de MAE y RMSE.
- Importancia por permutación para cada objetivo.
- Gráficos observado frente a predicho.
- Sensibilidad de las predicciones frente a perturbaciones.

### Métricas en prueba

![Métricas de prueba](reports/figures/modelos/01_metricas_prueba.png)

### Importancia por permutación

![Importancia por permutación](reports/figures/modelos/02_importancia_permutacion.png)

La importancia se expresa como el aumento del MAE cuando una variable se permuta en el conjunto de prueba. Un valor alto indica que el modelo depende fuertemente de esa variable. Una importancia cercana a cero no demuestra irrelevancia física, porque puede existir colinealidad entre `H`, `B`, `D` y `H/B`.

### Valores observados frente a predichos

![Observado frente a predicho](reports/figures/modelos/03_observado_vs_predicho.png)

### Estabilidad frente a perturbaciones

![Estabilidad de las predicciones](reports/figures/modelos/04_estabilidad_perturbaciones.png)

## 9. Evaluación de estabilidad PCS/VDS

El modelo se evalúa con las siguientes perturbaciones:

| Variable | Perturbaciones |
|---|---:|
| Dirección del viento `theta` | ±1° y ±5° |
| Coordenada `X` | ±0.5 % y ±1 % del rango observado |
| Coordenada `Y` | ±0.5 % y ±1 % del rango observado |
| Coordenada `Z` | ±0.5 % y ±1 % del rango observado |

Se reportan el cambio absoluto medio y el percentil 95 del cambio absoluto de la predicción. Esto permite detectar regiones donde pequeñas variaciones de dirección o ubicación producen cambios importantes en Cp.

## 10. Evaluación de puntos sin sensores

Las coordenadas `X`, `Y` y `Z` se utilizan como variables continuas. Sin embargo, la capacidad de interpolar no debe evaluarse con una división aleatoria de filas. Para demostrar generalización espacial se recomienda:

1. Reservar una franja completa de sensores o una zona de fachada.
2. Entrenar con los sensores restantes.
3. Predecir la zona reservada.
4. Reportar MAE y RMSE según la distancia al sensor de entrenamiento más cercano.
5. Repetir el procedimiento para diferentes fachadas, alturas y ángulos.

## 11. Flujo completo

```bash
python -m tpu_cp.inspect_mat --data-dir data --out reports/mat_inventory.json
python -m tpu_cp.build_dataset --data-dir data --config configs/schema.yml --out data/processed.parquet
python -m tpu_cp.eda --data data/processed.parquet --config configs/schema.yml --out-dir reports/figures/eda
python -m tpu_cp.train --data data/processed.parquet --config configs/schema.yml --out-dir models
python -m tpu_cp.visualize --models-dir models --out-dir reports/figures/modelos
```

## 12. Reproducibilidad

- Semilla aleatoria: `42`.
- División externa: 20 % de grupos para prueba.
- Validación cruzada: hasta 5 pliegues agrupados.
- Importancia: 15 repeticiones de permutación.
- Figuras: PNG a 220 dpi.
- Datos procesados: formato Parquet.

## 13. Referencias principales

- Bre, F., Gimenez, J. M., y Fachinotti, V. D. (2018). *Prediction of wind pressure coefficients on building surfaces using artificial neural networks*. Energy and Buildings, 158, 1429-1441.
- Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., y Anandkumar, A. (2021). *Fourier Neural Operator for Parametric Partial Differential Equations*. ICLR 2021.
- Tokyo Polytechnic University. *TPU Aerodynamic Database*. Citar el año de consulta y la publicación específica del subconjunto empleado.

## 14. Nota sobre modelos avanzados

Las GNN y los Fourier Neural Operators se consideran extensiones posteriores. Primero debe establecerse una línea base tabular sin fuga de información. Una GNN requiere construir explícitamente las aristas entre sensores y un FNO requiere representar los campos de presión en mallas espaciales consistentes.
