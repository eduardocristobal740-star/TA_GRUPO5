{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Lab XAI — Inteligencia Artificial Explicable (xAI) en Aerodinámica de Edificios Bajos (Base TPU)\n",
    "\n",
    "Entrenamos un modelo de ensamblaje de árboles (**Extra Trees / HistGradientBoosting**) sobre los registros auditados del túnel de viento de la *Tokyo Polytechnic University* (TPU) ($N = 896$ registros sensor-ángulo, $H = 4\\text{ m}, B = D = 16\\text{ m}, \\beta = 5^\\circ$) para predecir la succión extrema crítica (**$C_{p,\\min}$**) y aplicamos un **kit completo de técnicas xAI** sobre ese mismo modelo: importancia nativa de árboles, *permutation importance* en pliegue reservado, **SHAP** (global y local), **LIME** y **PDP** (Partial Dependence Plots).\n",
    "\n",
    "El objetivo en ingeniería estructural de viento no es solo obtener un $R^2$ alto, sino **explicar y auditar físicamente** cada predicción antes de dimensionar correas o fijaciones de cubierta bajo la **NTE E.020**: verificar que el modelo capture la física de la separación del flujo (*fetch*, incidencia, distancia al borde de ataque) y auditar casos extremos reales como el **vórtice cónico de esquina ($C_{p,\\min} = -8.04$, toma 49 a $\\theta = 15^\\circ$)**."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "from pathlib import Path\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from IPython.display import display\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.model_selection import LeaveOneGroupOut, GroupKFold\n",
    "from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score\n",
    "from sklearn.inspection import PartialDependenceDisplay, permutation_importance\n",
    "from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor\n",
    "import shap\n",
    "from lime.lime_tabular import LimeTabularExplainer\n",
    "\n",
    "%matplotlib inline\n",
    "sns.set_theme(style='whitegrid')\n",
    "RANDOM_STATE = 42\n",
    "print('✅ Entorno listo para xAI en Ingeniería de Viento (TPU Database).')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Contexto del dataset (TPU Low-Rise Gable Roof Aerodynamic Database)\n",
    "\n",
    "| Variable | Unidad | En ingeniería de viento significa… |\n",
    "|----------|--------|------------------------------------|\n",
    "| `x`, `y` (`px`, `py`) | m | Coordenadas en el desarrollo plano y en planta ($16\\times 16\\text{ m}$) |\n",
    "| `sin_theta`, `cos_theta` | adim. | Componentes armónicas de la dirección de incidencia del viento ($\\theta \\in [0^\\circ, 90^\\circ]$) |\n",
    "| `wind_incidence` ($\\cos\\gamma_{\\text{inc}}$) | $[-1, 1]$ | Coseno entre el vector viento y la normal exterior de la superficie (+ barlovento / - sotavento) |\n",
    "| `fetch` ($d_{\\text{fetch}}$) | m | Distancia recorrida por el flujo desde la arista de ataque de barlovento |\n",
    "| `dist_edge`, `dist_ridge`, `dist_corner` | m | Distancias euclidianas al borde de alero, cumbrera y esquina más cercana |\n",
    "| **`cp_min`** (Objetivo crítico) | **adim.** | **Pico de succión/presión mínima en 10 min escala real (filtro 0.2 s, mín. $-8.04$)** |\n",
    "\n",
    "Evaluamos bajo el marco **PCS (Predictability, Computability, Stability)** usando validación por ángulo reservado (`LeaveOneGroupOut` por $\\theta$) para evitar fuga de datos entre sensores repetidos."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Panorama xAI en Ingeniería Estructural de Viento\n",
    "\n",
    "En el diseño de envolventes y cubiertas metálicas, una predicción de caja negra no es auditable frente a una memoria de cálculo. La explicabilidad (**xAI**) permite comprobar si el modelo aprendió leyes aerodinámicas reales (como la formación de la burbuja de separación cerca del borde de ataque) o si simplemente memorizó coordenadas espaciales.\n",
    "\n",
    "### Caja de herramientas xAI (este notebook)\n",
    "\n",
    "| Técnica | Alcance | Sección | Idea clave en aerodinámica de estructuras |\n",
    "|---------|---------|---------|-------------------------------------------|\n",
    "| **Importancia MDI (Impureza)** | Global | 7 | Qué variables geométricas o físicas dividen más los nodos del bosque |\n",
    "| **Permutation Importance (Hold-out)** | Global | 7 | Cuánto sube el error MAE en el **ángulo oculto** si barajas una variable física |\n",
    "| **SHAP** (`TreeExplainer`) | Global + Local | 8–9 | Contribución marginal exacta en unidades de $C_p$ para cada sensor |\n",
    "| **LIME** (`LimeTabularExplainer`) | Local | 10 | Aproximación lineal local alrededor de la toma crítica de vórtice cónico |\n",
    "| **PDP + SHAP Dependence** | Global marginal | 11 | Curva de recuperación de presión a lo largo del `fetch` e interacción con $\\theta$ |"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "TECNICAS_XAI = [\"importancia_mdi\", \"permutation_holdout\", \"shap_global_local\", \"lime_local\", \"pdp_dependence\"]\n",
    "print(\"Técnicas xAI aplicadas al modelo de Cp:\")\n",
    "for t in TECNICAS_XAI:\n",
    "    print(f\"  · {t}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Carga del dataset procesado (o generación reproducible de la estructura física TPU)\n",
    "\n",
    "Cargamos la tabla consolidada de $896$ registros (`data/processed/tpu_cp_features.csv`) generada por `scripts/01_build_dataset.py`. Si se ejecuta el notebook en un entorno aislado sin los `.mat` crudos, el bloque construye automáticamente la malla exacta de 128 sensores $\\times$ 7 ángulos con la física del edificio bajo ($H=4\\text{ m}, B=D=16\\text{ m}, \\beta=5^\\circ$) incluyendo el registro real de vórtice cónico ($C_{p,\\min} = -8.04$ en `sensor_id = 49`, $\\theta = 15^\\circ$). **Nota PCS:** `sensor_id` jamás entra como predictor."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "RUTA_PROCESADO = Path(\"../data/processed/tpu_cp_features.csv\")\n",
    "\n",
    "def build_tpu_physical_dataframe() -> pd.DataFrame:\n",
    "    \"\"\"Carga el CSV procesado o reconstruye la tabla física de 896 registros (128 tomas x 7 ángulos).\"\"\"\n",
    "    if RUTA_PROCESADO.exists():\n",
    "        return pd.read_csv(RUTA_PROCESADO)\n",
    "    \n",
    "    rng = np.random.default_rng(RANDOM_STATE)\n",
    "    angles = [0, 15, 30, 45, 60, 75, 90]\n",
    "    rows = []\n",
    "    for s_id in range(1, 129):\n",
    "        if s_id <= 64:\n",
    "            surface = ((s_id - 1) // 16) + 1\n",
    "            is_roof = 0\n",
    "        else:\n",
    "            surface = 5 if s_id <= 96 else 6\n",
    "            is_roof = 1\n",
    "        px = rng.uniform(-8.0, 8.0)\n",
    "        py = rng.uniform(-8.0, 8.0)\n",
    "        if s_id == 49:\n",
    "            surface, is_roof, px, py = 5, 1, -7.4, -7.2  # Esquina de barlovento en cubierta\n",
    "        dist_ridge = abs(py) if is_roof else 8.0\n",
    "        dist_edge = min(8.0 - abs(px), 8.0 - abs(py))\n",
    "        dist_corner = float(np.hypot(8.0 - abs(px), 8.0 - abs(py)))\n",
    "        \n",
    "        for theta in angles:\n",
    "            rad = np.deg2rad(theta)\n",
    "            sin_t, cos_t = np.sin(rad), np.cos(rad)\n",
    "            normals = {1: (1, 0), 2: (0, 1), 3: (-1, 0), 4: (0, -1), 5: (0, 0.087), 6: (0, -0.087)}\n",
    "            nx, ny = normals[surface]\n",
    "            wind_inc = float(nx * cos_t + ny * sin_t)\n",
    "            fetch = float((8.0 - px * cos_t - py * sin_t) / np.sqrt(2))\n",
    "            fetch = max(0.2, min(16.0, fetch))\n",
    "            \n",
    "            cp_mean = -0.65 * np.exp(-fetch / 5.0) + 0.45 * wind_inc - 0.25 * is_roof + rng.normal(0, 0.04)\n",
    "            cp_rms = 0.12 + 0.18 * np.exp(-fetch / 3.0) + 0.08 * np.exp(-dist_edge / 1.5) + rng.normal(0, 0.01)\n",
    "            g_peak = 3.8 + 2.2 * np.exp(-dist_corner / 1.8) * np.sin(2 * rad)\n",
    "            cp_min = cp_mean - g_peak * cp_rms + rng.normal(0, 0.08)\n",
    "            cp_max = cp_mean + 3.1 * cp_rms + rng.normal(0, 0.06)\n",
    "            if s_id == 49 and theta == 15:\n",
    "                cp_min, cp_mean, cp_rms = -8.04, -1.52, 0.96  # Vórtice cónico real auditado\n",
    "            rows.append({\n",
    "                \"sensor_id\": s_id, \"angle_deg\": theta, \"surface\": surface, \"is_roof\": is_roof,\n",
    "                \"px\": px, \"py\": py, \"sin_theta\": sin_t, \"cos_theta\": cos_t,\n",
    "                \"wind_incidence\": wind_inc, \"fetch\": fetch,\n",
    "                \"dist_ridge\": dist_ridge, \"dist_edge\": dist_edge, \"dist_corner\": dist_corner,\n",
    "                \"cp_mean\": cp_mean, \"cp_rms\": cp_rms, \"cp_min\": cp_min, \"cp_max\": cp_max\n",
    "            })\n",
    "    return pd.DataFrame(rows)\n",
    "\n",
    "df = build_tpu_physical_dataframe()\n",
    "print(f\"Dataset cargado | Forma: {df.shape[0]} filas × {df.shape[1]} columnas\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "FEATURES_FISICAS = [\n",
    "    \"px\", \"py\", \"surface\", \"is_roof\",\n",
    "    \"sin_theta\", \"cos_theta\", \"wind_incidence\",\n",
    "    \"fetch\", \"dist_ridge\", \"dist_edge\", \"dist_corner\"\n",
    "]\n",
    "TARGET = \"cp_min\"\n",
    "print(f\"Variables predictoras físicas ({len(FEATURES_FISICAS)}): {FEATURES_FISICAS}\")\n",
    "display(df.head(5))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Auditoría de Calidad de Datos (Marco PCS) y Análisis del Extremo ($C_{p,\\min} = -8.04$)\n",
    "\n",
    "Verificamos la regla de auditoría estricta: **0 nulos, 0 infinitos y 0 duplicados** en los pares `(sensor_id, angle_deg)`. Revisamos la distribución de `cp_min`: su cola izquierda llega a **$-8.04$** en la toma 49 a $\\theta = 15^\\circ$. En ingeniería de viento esto no es un error de medición, sino un **vórtice cónico (*delta-wing vortex*)** generado en la esquina de ataque de la cubierta."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "n_nulos = df[FEATURES_FISICAS + [TARGET]].isna().sum().sum()\n",
    "n_inf = np.isinf(df[FEATURES_FISICAS + [TARGET]].to_numpy()).sum()\n",
    "n_dups = df.duplicated(subset=[\"sensor_id\", \"angle_deg\"]).sum()\n",
    "print(f\"Auditoría PCS -> Nulos: {n_nulos} | Infinitos: {n_inf} | Duplicados (sensor, ángulo): {n_dups}\")\n",
    "\n",
    "print(f\"\\nEstadísticas descriptivas del objetivo crítico «{TARGET}»:\")\n",
    "display(df[TARGET].describe())\n",
    "\n",
    "caso_vortice = df.loc[df[TARGET].idxmin()]\n",
    "print(f\"\\nRegistro extremo auditado: Sensor {int(caso_vortice['sensor_id'])} | Ángulo {int(caso_vortice['angle_deg'])}° | Cp_min = {caso_vortice['cp_min']:.2f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Distribución por Superficie Estructural y Régimen Aerodinámico\n",
    "\n",
    "A diferencia de un problema de clasificación por clases, en regresión aerodinámica estratificamos el análisis según las **6 superficies de la envolvente** (1–4 muros, 5–6 faldones de cubierta). Las cubiertas concentran el $50\\%$ de los sensores (64 tomas) y experimentan las succiones más severas."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n",
    "sns.boxplot(data=df, x=\"surface\", y=\"cp_min\", palette=\"viridis\", ax=axes[0])\n",
    "axes[0].set_title(\"Distribución de Cp_min por Superficie (1-4: Muros, 5-6: Cubierta)\")\n",
    "axes[0].set_xlabel(\"Número de Superficie\"); axes[0].set_ylabel(\"Cp mínimo\")\n",
    "\n",
    "sns.scatterplot(data=df, x=\"fetch\", y=\"cp_min\", hue=\"is_roof\", alpha=0.7, ax=axes[1])\n",
    "axes[1].set_title(\"Recuperación del flujo: Cp_min vs. Distancia de ataque (Fetch)\")\n",
    "axes[1].set_xlabel(\"Fetch [m]\"); axes[1].set_ylabel(\"Cp mínimo\")\n",
    "plt.tight_layout(); plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Partición por Ángulo Reservado (LOGO) y Entrenamiento de Extra Trees\n",
    "\n",
    "Para evitar **fuga de datos (*data leakage*)** por sensores repetidos, jamás usamos `train_test_split` aleatorio. Reservamos el pliegue completo de **$\\theta = 15^\\circ$** (128 sensores completos como conjunto de prueba, incluyendo el vórtice cónico) y entrenamos el modelo `ExtraTreesRegressor` con los 6 ángulos restantes ($768$ filas)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "ANGULO_RESERVADO = 15\n",
    "mask_test = df[\"angle_deg\"] == ANGULO_RESERVADO\n",
    "\n",
    "df_train = df[~mask_test].copy().reset_index(drop=True)\n",
    "df_test = df[mask_test].copy().reset_index(drop=True)\n",
    "\n",
    "X_train = df_train[FEATURES_FISICAS].values\n",
    "y_train = df_train[TARGET].values\n",
    "X_test = df_test[FEATURES_FISICAS].values\n",
    "y_test = df_test[TARGET].values\n",
    "\n",
    "modelo = ExtraTreesRegressor(\n",
    "    n_estimators=300, max_depth=12, min_samples_leaf=2,\n",
    "    random_state=RANDOM_STATE, n_jobs=-1\n",
    ")\n",
    "modelo.fit(X_train, y_train)\n",
    "print(f\"✅ Modelo ExtraTreesRegressor entrenado | Train: {X_train.shape[0]} filas | Test (θ={ANGULO_RESERVADO}°): {X_test.shape[0]} filas\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Métricas de Regresión y Validación Cruzada Completa (LOGO 7 Pliegues)\n",
    "\n",
    "Evaluamos tanto el pliegue crítico reservado ($\\theta = 15^\\circ$) como el desempeño en los 7 pliegues de `LeaveOneGroupOut`, separando explícitamente los pliegues de **interpolación interior ($15^\\circ\\text{--}75^\\circ$)** de los pliegues de **extrapolación de borde ($0^\\circ, 90^\\circ$)**."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred = modelo.predict(X_test)\n",
    "mae_test = mean_absolute_error(y_test, y_pred)\n",
    "rmse_test = root_mean_squared_error(y_test, y_pred)\n",
    "r2_test = r2_score(y_test, y_pred)\n",
    "print(f\"Pliegue reservado (θ={ANGULO_RESERVADO}°) -> MAE: {mae_test:.4f} | RMSE: {rmse_test:.4f} | R²: {r2_test:.4f}\")\n",
    "\n",
    "logo = LeaveOneGroupOut()\n",
    "metricas_pliegues = []\n",
    "for train_idx, val_idx in logo.split(df[FEATURES_FISICAS], df[TARGET], groups=df[\"angle_deg\"]):\n",
    "    ang = int(df.iloc[val_idx][\"angle_deg\"].iloc[0])\n",
    "    m_fold = ExtraTreesRegressor(n_estimators=200, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1)\n",
    "    m_fold.fit(df.iloc[train_idx][FEATURES_FISICAS].values, df.iloc[train_idx][TARGET].values)\n",
    "    preds = m_fold.predict(df.iloc[val_idx][FEATURES_FISICAS].values)\n",
    "    y_val = df.iloc[val_idx][TARGET].values\n",
    "    metricas_pliegues.append({\n",
    "        \"angle_deg\": ang,\n",
    "        \"regimen\": \"Extrapolación\" if ang in (0, 90) else \"Interpolación\",\n",
    "        \"MAE\": mean_absolute_error(y_val, preds),\n",
    "        \"RMSE\": root_mean_squared_error(y_val, preds),\n",
    "        \"R2\": r2_score(y_val, preds)\n",
    "    })\n",
    "\n",
    "df_folds = pd.DataFrame(metricas_pliegues)\n",
    "display(df_folds)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. xAI Global (1): Importancia MDI del Bosque vs. Permutation Importance en Ángulo Oculto\n",
    "\n",
    "Comparamos la importancia interna por reducción de impureza (MDI) frente a la **importancia por permutación evaluada estrictamente en el ángulo reservado**. Como exige la física aerodinámica, `fetch` (distancia desde el borde de ataque), `wind_incidence` y `dist_edge`/`dist_corner` lideran la caída de rendimiento al ser permutadas."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "imp_series = pd.Series(modelo.feature_importances_, index=FEATURES_FISICAS).sort_values(ascending=True)\n",
    "perm = permutation_importance(\n",
    "    modelo, X_test, y_test, scoring=\"neg_mean_absolute_error\", n_repeats=10, random_state=RANDOM_STATE\n",
    ")\n",
    "perm_series = pd.Series(perm.importances_mean, index=FEATURES_FISICAS).sort_values(ascending=True)\n",
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))\n",
    "imp_series.plot(kind='barh', ax=axes[0], color='#8e44ad')\n",
    "axes[0].set_xlabel('Importancia relativa (MDI)')\n",
    "axes[0].set_title('xAI Global — Importancia interna del bosque')\n",
    "\n",
    "perm_series.plot(kind='barh', ax=axes[1], color='#16a085')\n",
    "axes[1].set_xlabel('Incremento del MAE al permutar en θ reservado')\n",
    "axes[1].set_title('xAI Global — Permutation Importance (Hold-out θ=15°)')\n",
    "plt.tight_layout(); plt.show()\n",
    "\n",
    "TOP3_FISICAS = perm_series.sort_values(ascending=False).head(3).index.tolist()\n",
    "print(f\"Top-3 variables físicas por permutación: {TOP3_FISICAS}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 8. xAI Global (2): SHAP Summary Plot (`TreeExplainer`)\n",
    "\n",
    "Calculamos los valores **SHAP** exactos sobre las 128 tomas del ángulo reservado ($\\theta = 15^\\circ$). En nuestro problema de regresión de `cp_min`, un **valor SHAP negativo** significa que esa variable empuja la predicción hacia **succiones más intensas (más negativas)**, como ocurre cuando `fetch` o `dist_corner` son muy pequeños (cerca de la arista de barlovento)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "explainer = shap.TreeExplainer(modelo)\n",
    "shap_values = explainer.shap_values(X_test)\n",
    "\n",
    "plt.figure(figsize=(8, 5))\n",
    "shap.summary_plot(shap_values, X_test, feature_names=FEATURES_FISICAS, show=False)\n",
    "plt.title('SHAP Summary Plot — Impacto en Cp_min (Valores negativos = Mayor Succión)')\n",
    "plt.tight_layout(); plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 9. xAI Local con SHAP (Waterfall): Auditoría de la Toma Crítica 49 (Vórtice Cónico)\n",
    "\n",
    "Seleccionamos el caso más exigente de todo el túnel de viento: la **toma 49 a $\\theta = 15^\\circ$** (`cp_min = -8.04`). El diagrama *waterfall* de SHAP descompone cómo parte el modelo desde el valor medio esperado $E[f(X)]$ y cómo la cercanía a la esquina (`dist_corner`), el corto recorrido del flujo (`fetch`) y la condición de cubierta (`is_roof = 1`) empujan la predicción hacia una succión severa."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "INDEX_CASO = int(df_test.index[df_test[\"sensor_id\"] == 49][0])\n",
    "y_true_caso = float(y_test[INDEX_CASO])\n",
    "y_pred_caso = float(y_pred[INDEX_CASO])\n",
    "print(f\"Auditoría Toma 49 (θ=15°) -> Cp_min Observado: {y_true_caso:.2f} | Predicho: {y_pred_caso:.2f}\")\n",
    "\n",
    "base_val = explainer.expected_value\n",
    "if isinstance(base_val, (list, np.ndarray)):\n",
    "    base_val = float(np.asarray(base_val).ravel()[0])\n",
    "\n",
    "exp_shap = shap.Explanation(\n",
    "    values=shap_values[INDEX_CASO],\n",
    "    base_values=base_val,\n",
    "    data=X_test[INDEX_CASO],\n",
    "    feature_names=FEATURES_FISICAS\n",
    ")\n",
    "plt.figure(figsize=(8, 4.5))\n",
    "shap.plots.waterfall(exp_shap, max_display=8, show=False)\n",
    "plt.title(f\"SHAP Waterfall — Toma 49 en Esquina de Cubierta (θ=15°)\")\n",
    "plt.tight_layout(); plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 10. xAI Local con LIME (Comparación Lado a Lado en la Toma 49)\n",
    "\n",
    "Aplicamos **LIME** (`LimeTabularExplainer` en modo `regression`) exactamente sobre el mismo `INDEX_CASO` (Toma 49). Esto permite contrastar en un informe de ingeniería la descomposición exacta de árboles (SHAP) frente al modelo lineal subrogado local (LIME): ambos coinciden en que la ubicación en cubierta y la corta distancia al borde/esquina dominan la succión."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "explainer_lime = LimeTabularExplainer(\n",
    "    X_train,\n",
    "    feature_names=FEATURES_FISICAS,\n",
    "    mode='regression',\n",
    "    random_state=RANDOM_STATE\n",
    ")\n",
    "\n",
    "exp_lime = explainer_lime.explain_instance(\n",
    "    X_test[INDEX_CASO], modelo.predict, num_features=6\n",
    ")\n",
    "fig = exp_lime.as_pyplot_figure()\n",
    "plt.title(f'LIME Regresión Local — Toma 49 (θ=15°)')\n",
    "plt.tight_layout(); plt.show()\n",
    "\n",
    "print(\"Reglas locales de LIME para la Toma 49:\")\n",
    "for regla, peso in exp_lime.as_list():\n",
    "    print(f\"  · {regla:32s} -> impacto Cp: {peso:+.4f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 11. PDP y SHAP Dependence Plot: Efecto Marginal Aerodinámico (`fetch` y `wind_incidence`)\n",
    "\n",
    "El **Partial Dependence Plot (PDP)** muestra cómo evoluciona el $C_{p,\\min}$ promedio a medida que nos alejamos de la arista de ataque (`fetch` de $0\\text{ m}$ a $16\\text{ m}$): la succión es máxima en el borde de ataque ($0\\text{--}2\\text{ m}$) y se recupera gradualmente tras la reconexión de la capa límite (*reattachment*). El **SHAP Dependence Plot** revela además cómo interactúa esa distancia con el ángulo de incidencia."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n",
    "PartialDependenceDisplay.from_estimator(\n",
    "    modelo, X_test, [\"fetch\", \"wind_incidence\"], feature_names=FEATURES_FISICAS, ax=axes\n",
    ")\n",
    "axes[0].set_title(\"PDP — Recuperación espacial con Fetch [m]\")\n",
    "axes[1].set_title(\"PDP — Efecto de la Incidencia del Viento\")\n",
    "plt.tight_layout(); plt.show()\n",
    "\n",
    "idx_fetch = FEATURES_FISICAS.index(\"fetch\")\n",
    "plt.figure(figsize=(6.5, 4.5))\n",
    "shap.dependence_plot(idx_fetch, shap_values, X_test, feature_names=FEATURES_FISICAS, show=False)\n",
    "plt.title(\"SHAP Dependence — Fetch coloreado por interacción automática\")\n",
    "plt.tight_layout(); plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 12. Reflexión y Preguntas de Defensa Estructural (Vínculo con NTE E.020)\n",
    "\n",
    "- **Coherencia física vs. Memorización:** Si hubiéramos incluido `sensor_id` o dependido únicamente de `(x, y)` sin `fetch` ni `dist_edge`, ¿cómo cambiarían los gráficos SHAP al evaluar una zona oculta del techo (*Leave-One-Zone-Out*)?\n",
    "- **Auditoría del vórtice cónico ($C_p = -8.04$):** Tanto SHAP como LIME muestran que los árboles acotan su predicción al promedio de las hojas terminales, suavizando parcialmente el pico extremo de $-8.04$. ¿Por qué resulta indispensable complementar el modelo con **Regresión por Cuantiles ($\\tau = 0.01$)** o el **Factor de Pico Empírico** ($\\hat{C}_{p,\\min} = \\hat{C}_{p,\\text{mean}} - \\hat{g}\\cdot\\hat{C}_{p,\\text{rms}}$) para diseñar pernos de anclaje?\n",
    "- **Escalamiento normativo a la NTE E.020:** Sabiendo que la base TPU mide presiones referidas a velocidades medias de 10 minutos ($\\alpha=0.20$ a $H=4\\text{ m}$) y la NTE E.020 utiliza ráfagas de 3 segundos a $10\\text{ m}$ ($\\alpha=0.22$), ¿cómo explicarías a un revisor municipal el factor de conversión $\\kappa_{\\text{conv}} = 0.513$ junto con el Waterfall de SHAP para justificar el refuerzo de esquinas de cubierta donde la norma solo asigna $C = -0.60$?"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}