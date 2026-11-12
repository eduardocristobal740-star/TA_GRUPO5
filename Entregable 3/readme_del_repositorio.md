# tpu-cp-ml — Predicción y Explicabilidad (xAI) de Coeficientes de Presión de Viento en un Edificio Bajo (Base TPU + NTE E.020)

Pipeline reproducible bajo el marco **PCS (Predictability, Computability, Stability)** [Yu & Barter, 2024]: series crudas `.mat` de túnel de viento (TPU) $\rightarrow$ auditoría de datos y tabla sensor-ángulo ($896$ registros) $\rightarrow$ ingeniería de variables aerodinámicas y simetría biaxial $\rightarrow$ validación angular (`LeaveOneGroupOut`) y espacial severa (`GroupKFold` / `LeaveOneZoneOut`) $\rightarrow$ modelado de extremos (Cuantiles / Factor de Pico), explicabilidad **xAI** (SHAP, LIME, PDP) y escalamiento normativo **NTE E.020**.

## Estructura del Repositorio

```text
TA_GRUPO5/
├── README.md                                   # Resumen ejecutivo, resultados e instrucciones
├── BITACORA_IA (1).md                          # Registro transparente y auditable de uso de IA
├── Predicción de estadísticos del coeficiente de presión de viento en un edificio b.md
├── requirements.txt                            # Dependencias del proyecto
├── data/
│   ├── raw/                                    # 7 archivos time_series_of_point_wind_pressure_{0..90}.mat
│   └── processed/                              # Tabla auditada tpu_cp_features.csv (896 filas)
├── models/                                     # Modelos serializados (.joblib) y cuantílicos
├── notebooks/
│   └── xai_estructuras_tpu_cp.ipynb            # Notebook interactivo xAI (SHAP, LIME, PDP, Permutation)
├── src/
│   └── tpu_cp/
│       ├── features.py                         # Variables físicas (fetch, incidencia) y aumento por simetría
│       ├── validation.py                       # Esquemas LOGO angular, GroupKFold espacial y LOZO
│       └── normative_e020.py                   # Escalamiento Durst/E.020 (κ=0.513) y Clustering K-Means
├── scripts/
│   ├── 01_build_dataset.py                     # Procesamiento y auditoría de archivos .mat
│   ├── 02_train_evaluate.py                    # Entrenamiento LOGO, GroupKFold, LOZO y Regresión Cuantílica
│   ├── 03_eda_figures.py                       # Figuras exploratorias y desarrollo plano
│   ├── 04_analysis.py                          # Estabilidad PCS (semillas, bootstrap de sensores, ruido)
│   └── 05_results_figures.py                   # Gráficos finales de paridad y errores por pliegue
├── reports/
│   ├── tables/                                 # Tablas CSV exportadas (model_comparison_logo.csv, etc.)
│   └── figures/                                # Figuras PNG de alta resolución para el artículo IEEE
└── tests/                                      # Pruebas unitarias y de humo (pytest)
```

## Tabla Consolidada de Resultados (Validación Angular LOGO — 7 Pliegues)

| Modelo Evaluado | $C_{p,\text{mean}}$ MAE | $C_{p,\text{mean}}$ RMSE | $C_{p,\text{mean}}$ $R^2$ | $C_{p,\text{rms}}$ MAE | $C_{p,\text{rms}}$ RMSE | $C_{p,\text{rms}}$ $R^2$ | $C_{p,\min}$ MAE | $C_{p,\min}$ RMSE | $C_{p,\min}$ $R^2$ | $C_{p,\max}$ MAE | $C_{p,\max}$ RMSE | $C_{p,\max}$ $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mediana (Dummy Baseline)** | 0.2995 | 0.4374 | -0.004 | 0.0952 | 0.1317 | -0.143 | 0.8029 | 1.1614 | -0.091 | 0.4922 | 0.8952 | -0.163 |
| **Ridge (Variables Físicas)** | 0.2105 | 0.2712 | 0.616 | 0.0461 | 0.0642 | 0.727 | 0.4965 | 0.6510 | 0.655 | 0.2954 | 0.3980 | 0.770 |
| **Random Forest (Físicas)** | 0.0979 | 0.1380 | 0.899 | 0.0380 | 0.0565 | 0.772 | 0.3107 | 0.4720 | 0.817 | 0.1523 | 0.2190 | 0.927 |
| **Extra Trees (Base)** | 0.0799 | 0.1145 | 0.928 | 0.0289 | 0.0451 | 0.853 | **0.2504** | **0.3849** | **0.879** | 0.1538 | 0.2360 | 0.915 |
| **Extra Trees (Físicas)** | 0.0833 | 0.1150 | 0.928 | 0.0317 | 0.0472 | 0.839 | 0.2785 | 0.4120 | 0.861 | 0.1477 | 0.2130 | 0.931 |
| **HistGradientBoosting (Físicas)** | 0.0906 | 0.1260 | 0.914 | 0.0376 | 0.0546 | 0.787 | 0.3188 | 0.4640 | 0.823 | **0.1349** | **0.1892** | **0.946** |
| **Interp. Angular Lineal** | 0.0731 | 0.0965 | 0.948 | 0.0307 | 0.0484 | 0.831 | 0.3184 | 0.5190 | 0.779 | 0.1617 | 0.2520 | 0.903 |
| **Interp. Angular PCHIP** | **0.0654** | **0.0843** | 0.953 | 0.0336 | 0.0543 | 0.789 | 0.3957 | 0.7410 | 0.550 | 0.2231 | 0.3510 | 0.812 |
| **Híbrido Extra Trees (Físicas)** | 0.0671 | 0.0896 | **0.955** | 0.0291 | 0.0432 | 0.865 | 0.2545 | 0.3835 | **0.880** | 0.1458 | 0.2125 | 0.931 |
| **Extra Trees Ajustado (Físicas)** | 0.0747 | 0.1030 | 0.941 | **0.0272** | **0.0404** | **0.883** | 0.2594 | 0.3980 | 0.870 | 0.1432 | 0.2015 | 0.938 |
| **Factor de Pico + Simetría ($f$)** | — | — | — | — | — | — | **0.2285** | **0.3510** | **0.898** | — | — | — |

## Ejecución Rápida

```bash
pip install -r requirements.txt
python scripts/01_build_dataset.py
python scripts/02_train_evaluate.py --symmetry
jupyter notebook notebooks/xai_estructuras_tpu_cp.ipynb
python -m pytest -q tests
```

## Alcance y Disclaimers Normativos
* **Geometría única ($H=4\text{ m}, B=D=16\text{ m}, \beta=5^\circ$):** El modelo predice orientaciones no ensayadas e interpola/extrapola zonas sobre esta relación geométrica ($H/B=0.25$). No debe extrapolarse a edificios con distinta pendiente o esbeltez sin reentrenar con múltiples geometrías de la base TPU.
* **Escalamiento NTE E.020 ($\kappa_{\text{conv}} = 0.513$):** Los coeficientes TPU están referidos a velocidad media horaria/10-min a $H=4\text{ m}$ ($\alpha=0.20$). Para usarlos con la velocidad de ráfaga de 3 s a $10\text{ m}$ de la NTE E.020 ($\alpha=0.22$), debe aplicarse el módulo `src/tpu_cp/normative_e020.py` e integrar la presión interna $C_{pi}$.