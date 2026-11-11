# tpu-cp-ml — Predicción de estadísticos de Cp (datos TPU, edificio bajo, cubierta a dos aguas)

Pipeline reproducible: series MAT → tabla sensor-ángulo (896 filas) → comparación de modelos con
**validación por ángulo reservado (LeaveOneGroupOut)** → figuras, estabilidad (PCS) e informe.

```text
src/tpu_cp/   config, data_loading, features, models, validation, evaluation, plotting
scripts/      01_build_dataset · 02_train_evaluate [--tune] · 03_eda_figures · 04_analysis · 05_results_figures
tests/        pruebas de humo (pytest)
data/raw/     7 archivos time_series_of_point_wind_pressure_{0..90}.mat  (no incluidos en el zip)
reports/      tables/ (CSV) · figures/ (PNG)
```

## Uso
```bash
pip install -r requirements.txt
python scripts/01_build_dataset.py
python scripts/02_train_evaluate.py --tune     # ~10 min en 2 CPU
python scripts/03_eda_figures.py
python scripts/04_analysis.py                  # importancia hold-out + estabilidad (~6 min)
python scripts/05_results_figures.py
python -m pytest -q tests
```

## Alcance y límites
* Una sola geometría (H=4 m, B=D=16 m, pendiente 5°) y 7 direcciones: la validación mide
  **generalización angular**, no a otras geometrías. 896 filas = 128 sensores × 7 ángulos (no son independientes).
* La convención de orientación (muro 1 a barlovento a 0°, muro 2 a 90°) se **infiere** del signo de Cp medio; verificar con el plano de TPU.
* Los extremos provienen de series ya promediadas en 0.2 s (escala real); no son valores de diseño por periodo de retorno.
* No sustituye la NTE E.020 ni el criterio de un ingeniero estructural.
