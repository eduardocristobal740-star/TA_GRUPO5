<div align="center">

# TA Grupo 5 — IA en Estructuras

**Predicción y análisis de coeficientes de presión de viento en edificaciones**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Datos](https://img.shields.io/badge/Datos-TPU%20Wind%20Tunnel-2ea44f)](https://db.wind.arch.t-kougei.ac.jp/)
[![Marco](https://img.shields.io/badge/Marco-Veridical%20Data%20Science-6f42c1)](https://vdsbook.com/)
[![Curso](https://img.shields.io/badge/UPC-IA%20en%20Estructuras%202026-0055a4)](https://www.upc.edu.pe/)

</div>

## Sobre el proyecto

Este repositorio reúne los entregables del **Grupo 5** para el curso de Inteligencia Artificial en Estructuras del Programa de Ingeniería de Diseño Estructural de la UPC (2026). El trabajo estudia cómo analizar y estimar coeficientes de presión superficial del viento (`Cp`) a partir de datos experimentales de túnel de viento de la **Tokyo Polytechnic University (TPU)**.

El proyecto combina la formulación del problema de ingeniería estructural, la exploración de mediciones experimentales y un flujo reproducible de preparación y modelado. El análisis se organiza con el marco **Veridical Data Science**, considerando predictibilidad, computabilidad y estabilidad (PCS).

> **Alcance:** el caso de análisis de la segunda entrega corresponde a una geometría de cubierta a dos aguas y siete direcciones de viento. Los resultados son académicos y no sustituyen ensayos, estudios específicos de viento, la normativa aplicable ni la revisión de un profesional estructural.

## Entregables

### [Entregable 1](<Entregable 1/README.md>) — Definición del problema y marco PCS

Presenta el problema de estimar las presiones aerodinámicas sobre edificios, el contexto de uso estructural de `Cp`, la fuente de datos TPU y el planteamiento de variables. También describe cómo se propone evaluar la predictibilidad, computabilidad y estabilidad, junto con los límites de interpretación y referencias del trabajo.

### [Entregable 2](<Entregable 2/README.md>) — Análisis de datos y modelado

Incluye los datos experimentales seleccionados, código Python, configuración y notebook para:

- Revisar y consolidar las series temporales de presión por punto de medición.
- Calcular estadísticas de `Cp`: media, RMS de fluctuación, mínimo y máximo.
- Explorar la variación por dirección y superficie y producir visualizaciones.
- Comparar modelos de regresión reservando direcciones completas del viento, para reducir la fuga de información entre mediciones relacionadas.
- Documentar la norma peruana E.020 como contexto educativo.

La validación angular evalúa la generalización entre direcciones **dentro de la geometría observada**; no demuestra que los modelos generalicen a otros edificios o geometrías.

## Datos

Los archivos `.mat` provienen de la base aerodinámica de TPU. El subconjunto incluido corresponde a una edificación baja con cubierta a dos aguas, con mediciones para **0°, 15°, 30°, 45°, 60°, 75° y 90°**.

| Variable | Descripción |
|---|---|
| `cp_mean` | Coeficiente de presión medio |
| `cp_rms` | RMS de la fluctuación respecto a la media |
| `cp_min` | Mínimo observado en el registro |
| `cp_max` | Máximo observado en el registro |

Las mediciones de varios sensores y direcciones pertenecen a los mismos ensayos; no deben interpretarse como observaciones experimentales independientes. Las definiciones y metadatos se detallan en el README del [Entregable 2](<Entregable 2/README.md>).

## Estructura

```text
TA_GRUPO5/
├── Entregable 1/
│   └── README.md                         # Problema, objetivos y marco PCS
└── Entregable 2/
    ├── data/raw/                         # Series temporales TPU en archivos .mat
    ├── configs/schema.yml                # Esquema y configuración de datos
    ├── scripts/01_limpieza_eda_tpu.py    # Limpieza y análisis exploratorio inicial
    ├── src/tpu_cp/                       # Módulos de datos, EDA y modelado
    ├── Prediccion_coeficientes.ipynb     # Análisis y comparación de modelos
    ├── figuras_viento/                   # Figuras descriptivas
    ├── requirements.txt                  # Dependencias de Python
    └── README.md                         # Detalles, metodología y resultados
```

## Reproducir el análisis

Se requiere **Python 3.10 o superior**. Desde la raíz del repositorio:

```bash
cd "Entregable 2"
python -m pip install -r requirements.txt
python scripts/01_limpieza_eda_tpu.py
```

El script revisa y resume los archivos de presión y genera la tabla consolidada y gráficos en `outputs/`. Luego se puede abrir y ejecutar `Prediccion_coeficientes.ipynb` desde esa carpeta; el notebook utiliza la tabla consolidada para el análisis exploratorio, la validación y la comparación de modelos.

## Tecnologías y referencias

- **Python** con NumPy, pandas, SciPy, scikit-learn, Matplotlib y Seaborn.
- **Datos experimentales:** [TPU Aerodynamic Database](https://db.wind.arch.t-kougei.ac.jp/).
- **Marco metodológico:** Yu, B. y Barter, R. (2024), *Veridical Data Science*, [MIT Press](https://vdsbook.com/).
- **Contexto normativo:** Norma Técnica E.020, Cargas, del Reglamento Nacional de Edificaciones del Perú. La comparación presentada es educativa y no constituye un diseño estructural.

Para más información sobre variables, metodología, resultados y límites de validez, consulta el [README del Entregable 2](<Entregable 2/README.md>) y su [bibliografía](<Entregable 2/REFERENCIAS.md>).
```
