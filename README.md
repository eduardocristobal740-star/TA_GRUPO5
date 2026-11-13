<div align="center">

# Predicción de coeficientes de presión de viento

**Grupo 5 · Inteligencia Artificial en Estructuras · UPC 2026**

[![Python](https://img.shields.io/badge/Python-%3E%3D3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![uv](https://img.shields.io/badge/gestor%20opcional-uv-261230?logo=uv&logoColor=white)](https://docs.astral.sh/uv/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%3E%3D1.4-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Datos TPU](https://img.shields.io/badge/datos-TPU%20Wind%20Tunnel-008080)](https://db.wind.arch.t-kougei.ac.jp/)
![Python CI](https://img.shields.io/badge/Python%20CI-no%20configurado-lightgrey)
[![Marco PCS](https://img.shields.io/badge/marco-PCS-6f42c1)](https://vdsbook.com/)

</div>

## Descripción

Este proyecto académico estudia la predicción de los coeficientes de presión superficial del viento (**Cp**) en una edificación baja con cubierta a dos aguas. Utiliza datos de túnel de viento de la Tokyo Polytechnic University (TPU) y compara modelos de aprendizaje automático con métodos de interpolación.

El trabajo sigue el marco **Veridical Data Science**, mediante los principios de predictibilidad, computabilidad y estabilidad (PCS). También examina la interpretación de resultados en el contexto de la norma peruana NTE E.020.

## Datos y objetivos

El caso estudiado comprende una sola geometría: altura de 4 m, planta de 16 × 16 m y pendiente de cubierta de 5°. Incluye 128 sensores y siete direcciones de viento, de 0° a 90° en intervalos de 15°, para un total de **896 registros sensor-dirección**. Estos registros no son ensayos independientes.

Se predicen cuatro estadísticos por sensor y dirección: `cp_mean` (media), `cp_rms` (RMS de la fluctuación), `cp_min` (mínimo) y `cp_max` (máximo). El control de calidad reporta cero valores NaN o infinitos y ninguna pareja sensor-dirección duplicada. Consulta los [datos y el control de calidad](<Entregable 3/datos_fuente/control_calidad.json>).

## Metodología

1. Revisar las series originales `.mat` y construir estadísticas por sensor y dirección.
2. Explorar la distribución de Cp según la ubicación y la dirección del viento.
3. Comparar modelos de regresión e interpolación mediante validación angular: se reserva una dirección completa en cada uno de los siete pliegues.
4. Analizar estabilidad, errores en ángulos interiores y extremos, y límites de aplicación. El informe final también describe pruebas de validación espacial y un análisis normativo.

La interpolación por sensor utiliza información de ese mismo sensor en otras direcciones; sus métricas de validación angular no miden el desempeño en sensores nuevos.

## Resultados

La siguiente tabla resume el mejor candidato por objetivo según el [CSV de resultados](<Entregable 3/datos_fuente/mejor_por_objetivo.csv>). Son promedios de los siete pliegues de validación angular; la mejora de MAE se calcula respecto a una línea base de mediana.

| Objetivo | Modelo | MAE | RMSE | R² | Mejora de MAE |
|---|---|---:|---:|---:|---:|
| `cp_mean` | Interpolación angular PCHIP por sensor | 0.0654 | 0.0843 | 0.953 | 78.2 % |
| `cp_rms` | Extra Trees ajustado, variables físicas | 0.0272 | 0.0404 | 0.883 | 71.4 % |
| `cp_min` | Extra Trees, variables base | 0.2504 | 0.3849 | 0.879 | 68.8 % |
| `cp_max` | HistGradientBoosting, variables físicas | 0.1349 | 0.1892 | 0.946 | 72.6 % |

El error varía con la dirección: para `cp_mean`, PCHIP presenta un MAE de 0.0457 en los ángulos interiores y de 0.1145 en los extremos de 0° y 90°. Los resultados detallados y sus condiciones se presentan en el [informe final](<Entregable 3/informe_final_ieee.md>) y en las [tablas de datos](<Entregable 3/datos_fuente/>).

## Conclusiones y alcance

- Los candidatos seleccionados reducen el MAE entre **68.8 % y 78.2 %** frente a la línea base en validación angular, para esta geometría y estos datos.
- La interpolación angular funciona bien para la presión media, pero la extrapolación en los límites puede producir errores mayores que la interpolación entre ángulos observados.
- El informe analiza la utilidad de variables físicas y la estabilidad bajo perturbaciones; la validación angular no demuestra por sí sola generalización a sensores o geometrías no observados.
- Los resultados son académicos: **no sustituyen ensayos, estudios específicos de viento, la normativa vigente ni el juicio de un ingeniero estructural**. No deben extrapolarse a otras geometrías sin validación adicional.

## Contenido del repositorio

| Ruta | Contenido |
|---|---|
| [Entregable 1](<Entregable 1/README.md>) | Planteamiento del problema y marco PCS. |
| [Entregable 2](<Entregable 2/>) | Datos TPU, limpieza, análisis exploratorio, código y notebook. |
| [Entregable 3](<Entregable 3/>) | Informe final, tablas de resultados y explorador interactivo. |
| [docs](docs/) | Versión del explorador para GitHub Pages. |

## Uso

### Explorador de resultados

Abre [ABRIR_EXPLORADOR.html](<Entregable 3/ABRIR_EXPLORADOR.html>) en un navegador. Según las instrucciones incluidas, funciona sin servidor ni conexión a Internet. El explorador muestra resultados existentes; no entrena modelos ni predice nuevos ángulos.

### Análisis exploratorio con Python

Se requiere Python 3.10 o superior. Desde la raíz del repositorio:

```bash
cd "Entregable 2"
python -m pip install -r requirements.txt
python scripts/01_limpieza_eda_tpu.py
```

Como alternativa, con [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
cd "Entregable 2"
uv venv
uv pip install -r requirements.txt
uv run --no-project python scripts/01_limpieza_eda_tpu.py
```

El script genera una tabla consolidada y gráficos en `Entregable 2/outputs/`. El [notebook](<Entregable 2/Prediccion_coeficientes.ipynb>) contiene análisis adicionales. Estas instrucciones corresponden al análisis exploratorio del segundo entregable; no reproducen automáticamente todas las tablas del informe final.

## Referencias

- [TPU Aerodynamic Database](https://db.wind.arch.t-kougei.ac.jp/), fuente de los datos experimentales.
- Yu y Barter, [*Veridical Data Science*](https://vdsbook.com/), marco PCS.
- [Bibliografía del proyecto](<Entregable 2/REFERENCIAS.md>) y referencias del [informe final](<Entregable 3/informe_final_ieee.md>).
