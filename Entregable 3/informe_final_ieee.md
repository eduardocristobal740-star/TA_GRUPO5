# Predicción de Estadísticos del Coeficiente de Presión de Viento en un Edificio Bajo con Cubierta a Dos Aguas Mediante Aprendizaje Automático, Validación Espacial-Angular y Escalamiento Normativo E.020

*Grupo 5 — Curso de Inteligencia Artificial en Estructuras, Programa de Ingeniería de Diseño Estructural, Universidad Peruana de Ciencias Aplicadas (UPC), Lima, Perú (2026)*

---

## Resumen (Abstract)

**Resumen—** Se presenta un marco metodológico reproducible para la predicción y explicabilidad (xAI) de cuatro estadísticos del coeficiente de presión aerodinámica ($C_p$: media, desviación estándar o RMS, mínimo y máximo) en un edificio bajo de cubierta a dos aguas ($H = 4\text{ m}$, $B = D = 16\text{ m}$, pendiente $\beta = 5^\circ$) ensayado en el túnel de viento de la *Tokyo Polytechnic University* (TPU) bajo siete direcciones de incidencia ($\theta \in [0^\circ, 90^\circ]$, paso de $15^\circ$) y 128 tomas de presión ($N = 896$ registros sensor-ángulo). Siguiendo el paradigma de Ciencia de Datos Veraz (*Predictability, Computability, Stability* — PCS), se evita la fuga de información espacial y serial contrastando esquemas de validación por ángulo reservado (*Leave-One-Group-Out* por $\theta$) frente a esquemas de generalización espacial pura (*GroupKFold* por sensor y *Leave-One-Zone-Out* ocultando bordes y faldones completos). Se evalúan seis algoritmos de aprendizaje supervisado (Ridge, Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting y modelos híbridos con interpolación PCHIP libre de fuga), incorporando ingeniería de variables aerodinámicas (distancia de recorrido o *fetch*, coseno de incidencia y distancia a aristas de separación), aumento de datos por simetría biaxial y modelado de valores extremos mediante regresión cuantílica ($\tau_{0.01}, \tau_{0.99}$) y factores de pico empíricos. Los modelos basados en árboles reducen el error absoluto medio (MAE) entre $68.8\%$ y $78.2\%$ respecto a la línea base ($R^2 \in [0.879, 0.955]$ en validación angular), demostrando que el error en los pliegues de extrapolación ($0^\circ$ y $90^\circ$) supera hasta en un $188.6\%$ al de interpolación interior ($15^\circ\text{--}75^\circ$) cuando no se emplean predictores físicos en zonas de desprendimiento vorticial ($C_{p,\min} = -8.04$). Finalmente, se formula el protocolo analítico de conversión de escala de velocidad y perfil de capa límite ($\alpha_{\text{TPU}} = 0.20$ vs. $\alpha_{\text{E.020}} = 0.22$, factor $\kappa_{\text{conv}} = 0.513$) junto con una zonificación aerodinámica no supervisada (*K-Means*) contrastada con la Norma Técnica Peruana NTE E.020.

**Palabras clave—** aprendizaje automático, base de datos aerodinámica TPU, coeficiente de presión de viento, explicabilidad xAI, ingeniería de viento, marco PCS, NTE E.020, regresión por cuantiles, vórtices cónicos.

---

## I. Introducción

Las acciones aerodinámicas del viento gobiernan el dimensionamiento estructural de las envolventes, correas y sistemas de fijación en edificaciones bajas (*low-rise buildings*), donde la separación del flujo en aristas vivas genera gradientes severos de succión local. Dado el elevado costo computacional de la dinámica de fluidos computacional (CFD) con simulación de grandes remolinos (LES) y de los ensayos en túnel de viento de capa límite, el Diseño Asistido por Bases de Datos (*Database-Assisted Design* — DAD) introducido por Main y Fritz [3] y expandido por Simiu y Yeo [14] se ha consolidado como el estándar moderno de la ingeniería estructural de viento. En este contexto, el repositorio experimental de la *Tokyo Polytechnic University* (TPU) [1], [2] provee registros de alta resolución temporal que permiten entrenar modelos subrogados de aprendizaje automático (*Machine Learning* — ML) capaces de estimar campos de presión ante configuraciones o ángulos no instrumentados.

Trabajos recientes han demostrado la viabilidad de redes neuronales y ensamblajes de árboles sobre la base TPU. Chen *et al.* [12] desarrollaron modelos subrogados para predecir coeficientes de presión en edificios bajos logrando reducciones sustanciales del error cuadrático medio frente a interpolaciones polinómicas tradicionales, mientras que Dongmei *et al.* [13] sistematizaron el uso de algoritmos de aprendizaje profundo y ensamblajes para la predicción espacial de presiones en envolventes edilicias. Asimismo, Park *et al.* [10] integraron registros de la TPU en flujos DAD para evaluar respuestas estructurales direccionales. Sin embargo, la literatura advierte dos deficiencias recurrentes al aplicar ML a series de túnel de viento:
1. **Fuga de datos (*data leakage*) por partición aleatoria:** Al existir múltiples orientaciones para un mismo arreglo fijo de sensores, dividir aleatoriamente por filas permite que el modelo memorice la coordenada espacial $(x, y)$ como un identificador encubierto, produciendo métricas artificialmente optimistas que colapsan ante direcciones o ubicaciones no vistas.
2. **Dilución de extremos aerodinámicos y desconexión normativa:** Los algoritmos entrenados bajo funciones de pérdida cuadráticas o absolutas sobre la media tienden a suavizar las colas pesadas de succión ($C_{p,\min}$) asociadas a vórtices cónicos de esquina, y suelen vincularse directamente con códigos de diseño como la NTE E.020 [5] sin corregir las diferencias de tiempo de promediación (3 s vs. 10 min), rugosidad de terreno ($\alpha$) ni presión interna ($C_{pi}$).

Adoptando el marco de Ciencia de Datos Veraz (*Predictability, Computability, Stability* — PCS) de Yu y Barter [4], este trabajo estructura una investigación auditada que responde simultáneamente al rigor estadístico y a la física de fluidos en estructuras. Los objetivos específicos del estudio son:
* Consolidar y auditar los $8\,064\,000$ registros temporales del túnel de viento TPU en un conjunto tabular libre de anomalías numéricas, verificando la naturaleza física de los valores extremos severos.
* Evaluar el desempeño predictivo de seis familias algorítmicas y modelos híbridos contrastando la interpolación/extrapolación angular (*Leave-One-Group-Out* por dirección $\theta$) frente a pruebas de generalización espacial pura (*GroupKFold* por sensor y *Leave-One-Zone-Out*).
* Cuantificar el impacto de variables aerodinámicas explícitas (distancia al borde de ataque, *fetch*, coseno de incidencia), el aumento de datos por simetría geométrica, la formulación de extremos vía regresión cuantílica y factor de pico, y la auditoría mediante técnicas de Inteligencia Artificial Explicable (xAI: SHAP, LIME y PDP).
* Establecer un protocolo matemático riguroso de escalamiento hacia la NTE E.020 y comparar la zonificación normativa estática frente a agrupamientos aerodinámicos no supervisados (*clustering*).

---

## II. Metodología

### A. Auditoría del Conjunto de Datos Experimental y Naturaleza del Extremo ($C_p = -8.04$)

El estudio emplea siete archivos matriciales (`.mat`) de la base aerodinámica de edificios bajos con cubierta a dos aguas de la TPU [1], [2], correspondientes a los ángulos de incidencia $\theta \in \{0^\circ, 15^\circ, 30^\circ, 45^\circ, 60^\circ, 75^\circ, 90^\circ\}$. El modelo a escala geométrica $1/100$ representa un prototipo de altura media de cubierta $H = 4\text{ m}$, planta cuadrada $B = D = 16\text{ m}$ (relaciones $H/B = 0.25$, $D/B = 1.0$) y pendiente de techo $\beta = 5^\circ$. Cada archivo registra series temporales de longitud $T = 9\,000$ pasos por cada una de las $128$ tomas de presión (*pressure taps*), muestreadas de forma equivalente a $15\text{ Hz}$ durante $600\text{ s}$ ($10\text{ min}$) a escala real, con filtro de media móvil de $0.2\text{ s}$ y referenciadas a la presión dinámica media a la altura de alero/cubierta $q_H$ bajo un perfil de ley potencial con exponente $\alpha_{\text{TPU}} = 0.20$ (terreno abierto/suburbano).

La auditoría computacional sobre los $8\,064\,000$ puntos de datos ($9\,000 \times 128 \times 7$) confirmó exactitud estructural: $0$ valores nulos (`NaN`), $0$ valores infinitos, invarianza geométrica de coordenadas entre archivos y $0$ duplicados en la llave primaria compuesta `(sensor_id, angle_deg)`. Para cada par sensor-ángulo $(i, \theta)$ se extrajeron cuatro estadísticos objetivo, conformando una matriz de diseño de $N = 896$ filas (que representan $128$ historias angulares correlacionadas y no $896$ ensayos independientes):

$$C_{p,\text{mean}} = \frac{1}{T}\sum_{t=1}^{T} C_p(t), \quad C_{p,\text{rms}} = \sqrt{\frac{1}{T}\sum_{t=1}^{T} \left(C_p(t) - C_{p,\text{mean}}\right)^2}$$

$$C_{p,\min} = \min_{1 \le t \le T} C_p(t), \quad C_{p,\max} = \max_{1 \le t \le T} C_p(t)$$

Las 128 tomas se distribuyen sobre el desarrollo plano de la envolvente en seis superficies codificadas sin coordenada $z$ explícita: superficies 1 y 3 (muros hastiales, 16 tomas cada uno), superficies 2 y 4 (muros longitudinales, 16 tomas cada uno) y superficies 5 y 6 (faldones de cubierta, 32 tomas cada uno).

**Análisis físico del valor extremo severo ($C_{p,\min} = -8.04$):** Durante el análisis exploratorio se detectó un valor mínimo absoluto de $C_{p,\min} = -8.04$ registrado en la toma 49 bajo una incidencia oblicua de $\theta = 15^\circ$. Desde la mecánica de fluidos computacional y experimental (Simiu y Yeo [14]; Quan *et al.* [2]), este registro **no constituye ruido instrumental ni un dato anómalo (*outlier*)**, sino la captura física directa de un **vórtice cónico de borde de ataque (*conical o delta-wing vortex*)**. Cuando el viento incide en ángulos oblicuos cercanos a $15^\circ\text{--}45^\circ$ sobre cubiertas de baja pendiente ($\beta = 5^\circ$), la capa límite se desprende en las aristas del alero de barlovento enrollándose en tubos de vorticidad altamente concentrada cuyos núcleos generan succiones instantáneas de entre $-6.0$ y $-9.0$. Si este registro se excluyera arbitrariamente del conjunto de evaluación en el pliegue de $15^\circ$, el RMSE del pliegue se reduciría artificialmente en un $14.2\%$ y el error absoluto máximo caería de $2.70$ a $1.85$, induciendo un sesgo no conservador (subestimación del riesgo de arrancamiento de fijaciones de esquina). Por consiguiente, bajo el principio de veracidad PCS, el registro se conserva íntegramente.

---

### B. Ingeniería de Características Físicas y Aumento de Datos por Simetría

Para impedir que las coordenadas planas $(x, y)$ operen únicamente como etiquetas de memorización espacial y dotar a los modelos de partición recursiva de invariantes aerodinámicas, se definieron dos espacios de predictores (excluyendo estrictamente `sensor_id`):

1. **Conjunto Base ($b$):** Coordenadas del desarrollo plano $(x, y)$, codificación *one-hot* de la superficie ($S_1 \dots S_6$) y componentes armónicas $\sin\theta, \cos\theta$ para preservar la continuidad angular.
2. **Conjunto Físico-Aerodinámico ($f$):** Transforma la geometría en distancias relativas al flujo de separación:
   * **Proyección en planta $(p_x, p_y)$ e indicador de cubierta ($I_{\text{roof}}$):** Mapeo tridimensional implícito del desarrollo plano sobre la huella de $16\text{ m} \times 16\text{ m}$.
   * **Coseno de incidencia ($\cos\gamma_{\text{inc}}$) e indicador de barlovento:** Producto escalar entre el vector unitario del viento $\mathbf{u}(\theta) = (\cos\theta, \sin\theta, 0)$ y la normal exterior de cada superficie $\mathbf{n}_S$, identificando zonas de estancamiento ($\cos\gamma_{\text{inc}} > 0$) frente a zonas de estela o sotavento.
   * **Distancia de recorrido aerodinámico (*fetch*, $d_{\text{fetch}}$) y coordenada transversal ($d_{\text{cross}}$):** Proyección de la posición de la toma medida desde la arista o esquina de ataque de barlovento en la dirección del flujo $\mathbf{u}(\theta)$. Gobierna físicamente el espesor de la burbuja de separación y la reconexión (*reattachment*) del flujo sobre el techo.
   * **Distancias a singularidades geométricas ($d_{\text{ridge}}, d_{\text{edge}}, d_{\text{corner}}$):** Distancias euclidianas mínimas desde cada sensor hacia la cumbrera central, el borde perimetral de alero/hastial y la esquina más cercana, capturando la intensidad de los gradientes de borde.

**Aumento de Datos por Simetría Biaxial (*Symmetry Data Augmentation*):** Dado que la planta es cuadrada ($B = D = 16\text{ m}$) y la cubierta a dos aguas posee dos ejes ortogonales de simetría (el eje paralelo a la cumbrera y el eje transversal a los hastiales), un flujo incidente a un ángulo $\theta$ produce un campo de presiones exactamente especular sobre las tomas simétricas para los ángulos suplementarios y reflejados ($\theta' \in \{180^\circ - \theta, 180^\circ + \theta, 360^\circ - \theta\}$). Al aplicar la transformación de reflexión $(p_x, p_y) \mapsto (p_x, -p_y)$ y el mapeo de superficies simétricas ($S_2 \leftrightarrow S_4$, $S_5 \leftrightarrow S_6$) dentro del conjunto de entrenamiento de cada pliegue, se duplica el tamaño muestral efectivo sin introducir datos sintéticos artificiales, regularizando las particiones de los árboles en los bordes del dominio.

---

### C. Algoritmos Evaluados, Estrategia Híbrida, Modelado de Extremos y xAI

Todos los estimadores se encapsularon en objetos `Pipeline` de *scikit-learn* [9] con imputación preventiva por mediana y escalado estándar (`StandardScaler`) ajustados exclusivamente sobre el subconjunto de entrenamiento de cada pliegue:
* **Líneas base:** Predictor constante por mediana (`DummyRegressor`) y dos referencias físicas univariadas que interpolan la historia angular propia de cada toma sin emplear covariables espaciales: interpolación lineal por tramos e interpolación cúbica monótona a trozos (**PCHIP**, *Piecewise Cubic Hermite Interpolating Polynomial*) de $C_p(\theta)$.
* **Modelos paramétricos y de ensamblaje:** Regresión lineal penalizada ($\ell_2$, `Ridge`), `RandomForestRegressor` [6], `ExtraTreesRegressor` (*Extremely Randomized Trees*) [7], `GradientBoostingRegressor` [8] e `HistGradientBoostingRegressor`.
* **Arquitectura Híbrida Libre de Fuga (*Leak-Free Hybrid Model*):** Combina la capacidad de interpolación suave de PCHIP en ángulos interiores con la robustez espacial de `ExtraTrees` en los bordes. Para construir la característica híbrida $\hat{C}_{p,\text{interp}}(i, \theta)$ durante el entrenamiento sin incurrir en fuga de datos, el valor interpolado de la fila $(i, \theta_k)$ se calcula ocultando temporalmente el ángulo $\theta_k$ dentro del propio conjunto de entrenamiento (*nested leave-one-angle-out*).
* **Modelado Robusto de Extremos ($C_{p,\min}, C_{p,\max}$):** Dado que el entrenamiento mediante pérdida cuadrática (MSE) estima la esperanza condicional $\mathbb{E}[C_{p,\min}|\mathbf{x}]$ y subestima los picos de succión de cola pesada, se implementan dos alternativas complementarias para la ingeniería de viento (Simiu y Yeo [14]):
  1. *Regresión por Cuantiles (`HistGradientBoostingRegressor` con pérdida `quantile`):* Optimización directa de la pérdida asimétrica *pinball* $L_\tau(y, \hat{y}) = \max(\tau(y-\hat{y}), (\tau-1)(y-\hat{y}))$ para los cuantiles extremos $\tau = 0.01$ (succión crítica) y $\tau = 0.99$ (sobrepresión crítica).
  2. *Método del Factor de Pico Empírico ($g_{\text{peak}}$):* Aprovechando la alta precisión alcanzada en los dos primeros momentos estadísticos ($\hat{C}_{p,\text{mean}}$ y $\hat{C}_{p,\text{rms}}$), el extremo se reconstruye indirectamente mediante:
     $$\hat{C}_{p,\min} = \hat{C}_{p,\text{mean}} - \hat{g}_{\min}(\mathbf{x}) \cdot \hat{C}_{p,\text{rms}}$$
     donde el factor de pico local $g_{\min} = |C_{p,\min} - C_{p,\text{mean}}| / C_{p,\text{rms}}$ (que oscila entre $3.5$ en zonas gaussianas y $6.8$ en vórtices de esquina) se estima con menor varianza relativa que el pico crudo.
* **Auditoría mediante Inteligencia Artificial Explicable (xAI):** Se incorpora un flujo completo de explicabilidad global y local (SHAP `TreeExplainer`, LIME tabular en modo regresión y gráficos de dependencia parcial PDP) para auditar físicamente el comportamiento del modelo en general y sobre la toma crítica 49 a $\theta = 15^\circ$.

---

### D. Esquemas de Validación: Generalización Angular vs. Generalización Espacial Severa

Para responder a la crítica metodológica sobre interpolación frente a capacidad predictiva espacial, el módulo `src/tpu_cp/validation.py` reemplaza cualquier partición aleatoria (`GroupShuffleSplit`) por tres esquemas deterministas de dificultad creciente:

1. **Validación por Ángulo Reservado (`LeaveOneGroupOut` — LOGO por $\theta$):** 7 pliegues exactos donde cada pliegue retiene las 128 tomas de una dirección completa $\theta_k$ ($128$ validación / $768$ entrenamiento). Evalúa la capacidad de predecir el campo de presiones ante un ángulo de viento no ensayado en el túnel. Los pliegues interiores ($\theta \in \{15^\circ, 30^\circ, 45^\circ, 60^\circ, 75^\circ\}$) operan en régimen de **interpolación angular** (flanqueados por vecinos a $\pm 15^\circ$), mientras que los pliegues de frontera ($\theta \in \{0^\circ, 90^\circ\}$) exigen **extrapolación angular pura**.
2. **Validación Espacial por Toma (`GroupKFold` por `sensor_id`, $K=8$):** Retiene $16$ sensores completos (sus 7 ángulos simultáneamente) en cada pliegue. Impide que el modelo disponga de la historia angular previa del sensor evaluado, midiendo interpolación espacial pura entre tomas vecinas.
3. **Validación Espacial Severa (*Leave-One-Zone-Out* — LOZO):** Oculta simultáneamente bloques geométricos continuos de alta complejidad aerodinámica, como todas las tomas del borde de ataque del techo o un faldón completo de cubierta ($S_5$, 32 sensores = 224 filas). Demuestra si las variables físicas ($d_{\text{fetch}}, \cos\gamma_{\text{inc}}, d_{\text{edge}}$) permiten transferir el aprendizaje desde una vertiente hacia una zona físicamente oculta.

---

### E. Protocolo Analítico de Escalamiento a la NTE E.020 y Zonificación por Clustering

Los coeficientes $C_p^{\text{TPU}}$ publicados por la TPU **no pueden multiplicarse directamente** por la presión dinámica de la Norma Técnica Peruana E.020 [5] debido a tres discrepancias físicas fundamentales: (i) la altura y el perfil de capa límite difieren ($\alpha_{\text{TPU}} = 0.20$ vs. $\alpha_{\text{E.020}} = 0.22$ en Zona Urbana/Terreno Rugoso), (ii) el tiempo de promediación de la velocidad de referencia en TPU es $T_{\text{TPU}} = 600\text{ s}$ ($10\text{ min}$) mientras que la velocidad de diseño $V_h$ de la NTE E.020 corresponde a ráfagas de $T_{\text{E.020}} = 3\text{ s}$ a $10\text{ m}$ de altura, y (iii) la norma E.020 emplea factores de forma globales envolventes ($C$) e integra presiones internas ($C_{pi}$).

Siguiendo a Simiu y Yeo [14], el factor de conversión de escala $\kappa_{\text{conv}}$ para transformar un coeficiente estadístico de la TPU ($C_p^{\text{TPU}}$, referido a $\bar{V}_H(600\text{ s})$) en un coeficiente equivalente compatible con la ecuación normativa peruana $P_h = 0.005 \cdot C_{\text{eq}} \cdot V_h^2$ (con $V_h$ en $\text{km/h}$ y $P_h$ en $\text{kgf/m}^2$) viene dado por:

$$C_{\text{eq}}^{\text{E.020}} = \frac{C_p^{\text{TPU}}}{G_q} \cdot \left( \frac{\bar{V}_{\text{TPU}}(H)}{\bar{V}_{\text{E.020}}(H)} \right)^2 = \frac{C_p^{\text{TPU}}}{\left( \frac{V_{3\text{s}}(z_{\text{ref}})}{\bar{V}_{600\text{s}}(z_{\text{ref}})} \right)^2} \cdot \left(\frac{H}{10}\right)^{2(\alpha_{\text{TPU}} - \alpha_{\text{E.020}})}$$

Adoptando la curva de Durst para ráfaga de $3\text{ s}$ frente a media horaria/10-min ($V_{3\text{s}} / \bar{V}_{600\text{s}} \approx 1.42 \Rightarrow G_q = 1.42^2 \approx 2.02$) y corrigiendo la relación de ley potencial a $H = 4\text{ m}$ respecto a $z_{10} = 10\text{ m}$:

$$\left(\frac{4}{10}\right)^{2(0.20 - 0.22)} = (0.4)^{-0.04} \approx 1.037 \implies \kappa_{\text{conv}} = \frac{1.037}{2.02} \approx \mathbf{0.513}$$

Así, la presión de diseño neta sobre un elemento de cierre en consultoría estructural real utilizando registros climáticos de estaciones SENAMHI (velocidad básica $V_{10}$ para periodo de retorno $T_R = 50\text{ años}$) se expresa como:

$$V_h = V_{10} \cdot \left(\frac{H}{10}\right)^{0.22}, \quad P_{\text{neta}} = 0.005 \cdot V_h^2 \cdot \left( \kappa_{\text{conv}} \cdot \hat{C}_{p,\text{extremo}}^{\text{TPU}} - C_{pi} \right)$$

donde $C_{pi} \in \{+0.7, -0.7\}$ o $\{+0.3, -0.3\}$ según el porcentaje de aberturas de la edificación estipulado en la NTE E.020 [5]. Adicionalmente, para superar la comparación punto a punto, se aplica agrupamiento no supervisado (*K-Means* con $K=4$ clústeres seleccionados por coeficiente de silueta) sobre el vector multivariado predicho $[\hat{C}_{p,\text{mean}}, \hat{C}_{p,\text{rms}}, \hat{C}_{p,\min}]$ de las 64 tomas de cubierta, contrastando las zonas emergentes con la zonificación normativa clásica.

---

## III. Resultados y Discusión

### A. Desempeño Global Frente a Líneas Base

La Tabla I sintetiza el rendimiento del mejor modelo por estadístico bajo el esquema de validación LOGO de 7 pliegues angulares. Los algoritmos no lineales reducen el MAE de la línea base constante (`DummyRegressor`) entre un $68.8\%$ y un $78.2\%$, y el RMSE entre un $66.9\%$ y un $80.7\%$, alcanzando coeficientes de determinación $R^2$ medios entre $0.879$ ($C_{p,\min}$) y $0.955$ ($C_{p,\text{mean}}$ con modelo híbrido). Estos niveles de precisión son plenamente consistentes con los reportados por Chen *et al.* [12] y Dongmei *et al.* [13] sobre geometrías de baja altura de la base TPU.

**Tabla I. Resumen del mejor candidato por objetivo en validación angular LOGO (7 pliegues).**

| Objetivo | Mejor Candidato | MAE | RMSE | $R^2$ | MAE Base | RMSE Base | Mejora MAE | Mejora RMSE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $C_{p,\text{mean}}$ | Híbrido Extra Trees ($f$) / PCHIP | 0.0654 | 0.0843 | 0.955 | 0.2995 | 0.4374 | 78.2 % | 80.7 % |
| $C_{p,\text{rms}}$ | Extra Trees ajustado ($f$) | 0.0272 | 0.0404 | 0.883 | 0.0952 | 0.1317 | 71.4 % | 69.3 % |
| $C_{p,\min}$ | Extra Trees ($b$) / Híbrido ($f$) | 0.2504 | 0.3849 | 0.880 | 0.8029 | 1.1614 | 68.8 % | 66.9 % |
| $C_{p,\max}$ | HistGradientBoosting ($f$) | 0.1349 | 0.1892 | 0.946 | 0.4922 | 0.8952 | 72.6 % | 78.9 % |

En la Tabla II se desglosa la comparativa completa entre familias algorítmicas y conjuntos de predictores ($b$: base, $f$: físicas). Se observa que la interpolación angular univariada por sensor (PCHIP) constituye una referencia sumamente exigente para el flujo estacionario medio ($C_{p,\text{mean}}$ MAE $= 0.0654$, $R^2 = 0.953$), igualando en la práctica al modelo `Híbrido Extra Trees (f)` (MAE $= 0.0671$, $R^2 = 0.955$). Sin embargo, en los estadísticos gobernados por turbulencia de pequeña escala y desprendimiento de vórtices ($C_{p,\min}$ y $C_{p,\max}$), la interpolación PCHIP se degrada drásticamente ($R^2 = 0.550$ en $C_{p,\min}$), siendo ampliamente superada por `Extra Trees` (MAE $= 0.2504$, $R^2 = 0.879$) e `HistGradientBoosting` (MAE $= 0.1349$, $R^2 = 0.946$).

**Tabla II. Comparativa integral de modelos bajo validación LOGO por ángulo (media de 7 pliegues).**

| Modelo Evaluado | $C_{p,\text{mean}}$ MAE | $C_{p,\text{mean}}$ $R^2$ | $C_{p,\text{rms}}$ MAE | $C_{p,\text{rms}}$ $R^2$ | $C_{p,\min}$ MAE | $C_{p,\min}$ $R^2$ | $C_{p,\max}$ MAE | $C_{p,\max}$ $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Mediana (Dummy) | 0.2995 | -0.004 | 0.0952 | -0.143 | 0.8029 | -0.091 | 0.4922 | -0.163 |
| Ridge ($f$) | 0.2105 | 0.616 | 0.0461 | 0.727 | 0.4965 | 0.655 | 0.2954 | 0.770 |
| Random Forest ($f$) | 0.0979 | 0.899 | 0.0380 | 0.772 | 0.3107 | 0.817 | 0.1523 | 0.927 |
| Extra Trees ($b$) | 0.0799 | 0.928 | 0.0289 | 0.853 | 0.2504 | 0.879 | 0.1538 | 0.915 |
| Extra Trees ($f$) | 0.0833 | 0.928 | 0.0317 | 0.839 | 0.2785 | 0.861 | 0.1477 | 0.931 |
| HistGradientBoosting ($f$) | 0.0906 | 0.914 | 0.0376 | 0.787 | 0.3188 | 0.823 | 0.1349 | 0.946 |
| Interp. Angular Lineal | 0.0731 | 0.948 | 0.0307 | 0.831 | 0.3184 | 0.779 | 0.1617 | 0.903 |
| Interp. Angular PCHIP | 0.0654 | 0.953 | 0.0336 | 0.789 | 0.3957 | 0.550 | 0.2231 | 0.812 |
| Híbrido Extra Trees ($f$) | 0.0671 | 0.955 | 0.0291 | 0.865 | 0.2545 | 0.880 | 0.1458 | 0.931 |
| Extra Trees Ajustado ($f$) | 0.0747 | 0.941 | 0.0272 | 0.883 | 0.2594 | 0.870 | 0.1432 | 0.938 |

---

### B. Análisis de Pliegues: Extrapolación ($0^\circ, 90^\circ$) vs. Interpolación Interior ($15^\circ\text{--}75^\circ$)

Para responder con precisión cuánto del error global proviene de extrapolar en las fronteras del dominio angular frente a interpolar entre direcciones vecinas, la Tabla III desglosa las métricas pliegue por pliegue para $C_{p,\text{mean}}$ y $C_{p,\min}$.

**Tabla III. Desglose de error MAE por pliegue angular reservado: Extrapolación de Borde vs. Interpolación Interior.**

| Régimen Angular | Ángulo ($\theta$) | $C_{p,\text{mean}}$ PCHIP | $C_{p,\text{mean}}$ Híbrido ET | $C_{p,\min}$ Lineal | $C_{p,\min}$ PCHIP | $C_{p,\min}$ Extra Trees ($b$) | $C_{p,\min}$ Híbrido ET ($f$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Extrapolación** | $0^\circ$ | 0.1182 | 0.0941 | 0.5120 | 0.7685 | 0.2914 | 0.2862 |
| **Interpolación** | $15^\circ$ | 0.0512 | 0.0618 | 0.2890 | 0.3110 | 0.2645 | 0.2680 |
| **Interpolación** | $30^\circ$ | 0.0428 | 0.0545 | 0.2415 | 0.2540 | 0.2280 | 0.2315 |
| **Interpolación** | $45^\circ$ | 0.0441 | 0.0562 | 0.2380 | 0.2495 | 0.2210 | 0.2260 |
| **Interpolación** | $60^\circ$ | 0.0435 | 0.0559 | 0.2460 | 0.2610 | 0.2315 | 0.2390 |
| **Interpolación** | $75^\circ$ | 0.0469 | 0.0601 | 0.2189 | 0.2102 | 0.2376 | 0.2578 |
| **Extrapolación** | $90^\circ$ | 0.1108 | 0.0869 | 0.4834 | 0.7155 | 0.2788 | 0.2730 |
| **Media Interior** | $15^\circ\text{--}75^\circ$ | **0.0457** | **0.0577** | **0.2467** | **0.2571** | **0.2365** | **0.2445** |
| **Media Borde** | $0^\circ, 90^\circ$ | **0.1145** | **0.0905** | **0.4977** | **0.7420** | **0.2851** | **0.2796** |

Los resultados demuestran dos fenómenos clave:
1. **Colapso de la interpolación polinómica en los bordes:** Al reservar $0^\circ$ o $90^\circ$, PCHIP debe proyectar derivadas cúbicas hacia afuera del dominio, incrementando su MAE en $C_{p,\text{mean}}$ un $+150.5\%$ ($0.1145$ vs. $0.0457$ interior) y casi triplicando su error en $C_{p,\min}$ ($0.7420$ vs. $0.2571$ interior, un aumento del $+188.6\%$).
2. **Amortiguamiento mediante aprendizaje estructural:** Los modelos de árboles (`Extra Trees` e `Híbrido`) contienen la degradación en los bordes gracias a que las variables espaciales y de incidencia anclan la predicción física: en $C_{p,\min}$, el MAE de borde del modelo híbrido ($0.2796$) es apenas un $+14.4\%$ superior al promedio interior ($0.2445$), reduciendo el error de extrapolación de PCHIP en un $62.3\%$. Asimismo, en el pliegue de $15^\circ$ (donde ocurre la succión extrema de $-8.04$ por vórtice cónico en la toma 49), el MAE sube moderadamente a $0.2645$ debido a que el cambio de $0^\circ$ (flujo ortogonal sin vórtice cónico) a $15^\circ$ (aparición abrupta del vórtice delta) es altamente no lineal.

---

### C. Impacto de Variables Físicas, Validación Espacial Severa, Extremos y Explicabilidad (xAI)

El análisis de importancia por permutación evaluado estrictamente sobre el ángulo reservado y la descomposición **SHAP** (`TreeExplainer`) confirman que los árboles priorizan las variables aerodinámicas frente a las coordenadas crudas: para $C_{p,\max}$ y $C_{p,\text{mean}}$ domina el coseno de incidencia $\cos\gamma_{\text{inc}}$ ($\Delta\text{MAE} = +0.316$ y $+0.152$), mientras que para $C_{p,\min}$ y $C_{p,\text{rms}}$ gobierna la distancia de recorrido desde la arista de ataque $d_{\text{fetch}}$ ($\Delta\text{MAE} = +0.365$ y $+0.065$) seguida del tipo de superficie ($\Delta\text{MAE} = +0.251$). La auditoría local mediante gráficos *Waterfall* de SHAP y LIME sobre la toma 49 a $\theta = 15^\circ$ demuestra que la cercanía simultánea a la esquina (`dist_corner`), el corto recorrido desde el borde de ataque (`fetch`) y la pertenencia a la cubierta (`is_roof = 1`) empujan aditivamente la predicción hacia el régimen de succión severa.

La razón por la cual las variables físicas ($f$) resultan indispensables en la generalización espacial queda demostrada al ejecutar los esquemas de **Validación Espacial Severa** (Tabla IV):

**Tabla IV. Comparación de esquemas de validación y técnicas avanzadas sobre $C_{p,\min}$.**

| Esquema de Validación / Metodología | Conjunto Base ($b$) MAE | Conjunto Físico ($f$) MAE | Conjunto Físico + Simetría MAE | Interpretación Física y Estadística |
| :--- | :---: | :---: | :---: | :--- |
| **LOGO por Ángulo ($\theta$)** (7 pliegues) | 0.2504 | 0.2785 | 0.2390 | Sensores repetidos en entrenamiento; $b$ memoriza $(x,y)$. |
| **GroupKFold por Sensor** ($K=8$, 16 tomas ocultas) | 0.3412 | 0.2840 | 0.2615 | Sin historial del sensor evaluado, las variables físicas reducen el MAE un $16.8\%$. |
| **Leave-One-Zone-Out (LOZO: Borde Alero)** | 0.6850 | 0.4120 | 0.3780 | $(x,y)$ cae fuera del soporte de los árboles; $d_{\text{fetch}}$ y $d_{\text{edge}}$ permiten extrapolar la zona ($-44.8\%$ MAE). |
| **Enfoque Factor de Pico** ($\hat{C}_{p,\text{mean}} - \hat{g}\cdot\hat{C}_{p,\text{rms}}$) | 0.2480 | 0.2410 | **0.2285** | Reduce la subestimación en colas pesadas ($C_p < -4.0$) un $21.5\%$. |
| **Regresión Cuantílica ($\tau = 0.01$, HGB)** | — | — | — | Cobertura empírica del $98.4\%$ de los picos reales (Pinball Loss $= 0.041$). |

Cuando se ocultan sensores completos (`GroupKFold`) o toda una franja de borde de cubierta (`LOZO`), el conjunto base ($b$) sufre una caída drástica de rendimiento (MAE $= 0.6850$ en LOZO) porque las particiones ortogonales sobre $(x, y)$ no pueden extrapolar coordenadas inéditas. Por el contrario, al emplear las distancias adimensionales al borde de ataque ($d_{\text{fetch}}, d_{\text{edge}}$) enriquecidas con el **aumento de datos por simetría biaxial**, el error en la zona oculta cae a $0.3780$ (mejora del $44.8\%$), demostrando verdadera capacidad predictiva física más allá de la mera interpolación tabular. Asimismo, la reconstrucción mediante el **Factor de Pico Empírico** ($\hat{C}_{p,\min} = \hat{C}_{p,\text{mean}} - \hat{g}_{\min}\hat{C}_{p,\text{rms}}$) combinada con la regresión por cuantiles ($\tau = 0.01$) corrige el sesgo de suavizado sobre los vórtices de esquina, logrando cubrir el $98.4\%$ de las succiones severas sin subestimar la demanda estructural.

---

### D. Estabilidad bajo Perturbaciones (Marco PCS)

En cumplimiento del pilar de Estabilidad (*Stability*) de Yu y Barter [4], la Tabla V reporta la sensibilidad de `Extra Trees (f)` frente a tres perturbaciones controladas: cambio de semilla pseudoaleatoria (8 réplicas), remuestreo *bootstrap* por bloques de sensores completos (12 réplicas, preservando la estructura de correlación intra-sensor) y adición de ruido gaussiano $\mathcal{N}(0, (0.05\sigma_y)^2)$ al objetivo de entrenamiento (8 réplicas).

**Tabla V. Estabilidad del MAE de Extra Trees (variables físicas) bajo perturbaciones de datos y modelo.**

| Objetivo | Semillas Aleatorias (8 réplicas) | Bootstrap de Sensores (12 réplicas) | Ruido Gaussiano 5 % (8 réplicas) | Variación Relativa Máx. |
| :--- | :---: | :---: | :---: | :---: |
| $C_{p,\text{mean}}$ | $0.0833 \pm 0.0004$ | $0.0920 \pm 0.0022$ | $0.0835 \pm 0.0010$ | $+10.4\%$ (Bootstrap) |
| $C_{p,\text{rms}}$ | $0.0317 \pm 0.0003$ | $0.0343 \pm 0.0012$ | $0.0316 \pm 0.0002$ | $+8.2\%$ (Bootstrap) |
| $C_{p,\min}$ | $0.2810 \pm 0.0031$ | $0.2918 \pm 0.0066$ | $0.2818 \pm 0.0025$ | $+3.8\%$ (Bootstrap) |
| $C_{p,\max}$ | $0.1466 \pm 0.0015$ | $0.1520 \pm 0.0034$ | $0.1477 \pm 0.0020$ | $+3.7\%$ (Bootstrap) |

La variación por semilla y ruido instrumental del $5\%$ es inferior al $1.0\%$, confirmando alta estabilidad numérica. El remuestreo *bootstrap* de sensores incrementa el MAE entre $3.7\%$ y $10.4\%$ debido a la reducción de densidad espacial al omitir un $\approx 36.8\%$ de las tomas únicas en cada bolsa, lo cual ratifica que la fuente dominante de incertidumbre es la cobertura geométrica y direccional del ensayo.

---

### E. Zonificación Aerodinámica Automatizada (Clustering) y Contraste con la NTE E.020

Al aplicar el algoritmo *K-Means* ($K=4$) sobre la envolvente de coeficientes predichos $[\hat{C}_{p,\text{mean}}, \hat{C}_{p,\text{rms}}, \hat{C}_{p,\min}]$ en las 64 tomas de cubierta (superficies 5 y 6) y escalar los valores al marco de ráfaga de 3 s de la NTE E.020 ($\kappa_{\text{conv}} = 0.513$), el aprendizaje no supervisado descubre de manera autónoma cuatro regiones aerodinámicas claramente diferenciadas (Tabla VI):

**Tabla VI. Comparación entre Zonificación Automatizada (*K-Means* sobre predicciones TPU escaladas) y Norma Peruana NTE E.020 [5].**

| Zona Descubierta por Clustering (IA) | Proporción de Cubierta | $C_{p,\text{mean}}^{\text{TPU}}$ Medio | $C_{p,\min}^{\text{TPU}}$ Crítico (10 min) | Coeficiente Equivalente Escalado $C_{\text{eq}}^{\text{E.020}}$ ($\kappa=0.513$) | Factor de Forma NTE E.020 ($C$ Barlovento / Sotavento) | Evaluación Estructural (Con $C_{pi} = \pm 0.3$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Clúster 1: Interior Central** | 46.9 % (30 tomas) | $-0.42$ | $-1.35$ | $-0.69$ | $-0.60$ | Concordancia alta ($13\%$ de diferencia). |
| **Clúster 2: Cumbrera y Alero Medio** | 31.2 % (20 tomas) | $-0.68$ | $-2.45$ | $-1.26$ | $-0.60$ | La NTE E.020 subestima la succión local en un $52\%$. |
| **Clúster 3: Arista de Ataque (Borde)** | 15.6 % (10 tomas) | $-1.05$ | $-4.10$ | $-2.10$ | $-0.60$ | Desprendimiento severo; requiere anclajes reforzados. |
| **Clúster 4: Esquina de Vórtice Cónico** | 6.3 % (4 tomas) | $-1.48$ | $-8.04$ | **$-4.12$** | $-0.60$ | Crítico para fijaciones: succión $6.8\times$ mayor al promedio E.020. |

Esta contrastación evidencia la distinción fundamental entre cargas para el **sistema principal resistente a fuerzas de viento (SPRFV)** y cargas para **componentes y revestimientos (C&R)**:
* En el **Clúster 1 (Campo Central, $46.9\%$ del área)**, el coeficiente extremo escalado ($C_{\text{eq}}^{\text{E.020}} = -0.69$) coincide estrechamente con el factor global de succión $C = -0.60$ de la NTE E.020 [5], validando que el código peruano calibra adecuadamente la resultante global para el diseño de pórticos principales.
* Sin embargo, en los **Clústeres 3 y 4 (Bordes de ataque y Esquinas vorticiales, $21.9\%$ del área)**, la succión local escalada alcanza $C_{\text{eq}}^{\text{E.020}} = -2.10$ y $-4.12$ (que al sumarse con una presión interna positiva $C_{pi} = +0.30$ arroja un coeficiente neto de diseño $C_{\text{neto}} = -4.42$). Esto demuestra que la zonificación uniforme de la NTE E.020 debe complementarse con microzonificación de bordes y esquinas (similar a ASCE 7) o con modelos subrogados auditados con xAI como el aquí desarrollado al diseñar pernos autorroscantes, clips de calamina y correas de borde.

---

## IV. Conclusiones

1. **Capacidad predictiva y auditoría PCS:** Los modelos de ensamblaje de árboles (`Extra Trees`, `HistGradientBoosting`) y el esquema híbrido libre de fuga predicen los cuatro estadísticos de $C_p$ en direcciones no ensayadas con reducciones del MAE de entre $68.8\%$ y $78.2\%$ frente a la línea base ($R^2 \in [0.880, 0.955]$) sobre los $896$ registros auditados de la base TPU.
2. **Descomposición entre interpolación y extrapolación angular:** El análisis por pliegue demuestra que la interpolación polinómica por sensor (PCHIP) es óptima en ángulos interiores ($15^\circ\text{--}75^\circ$, MAE $= 0.0457$ en $C_{p,\text{mean}}$), pero sufre una degradación severa al extrapolar en las fronteras de $0^\circ$ y $90^\circ$ (MAE $= 0.7420$ en $C_{p,\min}$), donde el modelo híbrido basado en variables físicas reduce el error de borde en un $62.3\%$ (MAE $= 0.2796$).
3. **Generalización espacial, ingeniería física y xAI:** Bajo validación espacial severa (*GroupKFold* por sensor y *Leave-One-Zone-Out* ocultando bordes completos), las coordenadas cartesianas crudas $(x, y)$ pierden validez predictiva, mientras que las variables aerodinámicas ($d_{\text{fetch}}, \cos\gamma_{\text{inc}}, d_{\text{edge}}$) combinadas con el aumento de datos por simetría biaxial reducen el error de extrapolación espacial en un $44.8\%$. Las pruebas xAI (SHAP, LIME y PDP) ratifican que el modelo captura la ley física de recuperación de presión tras la arista de ataque.
4. **Física de extremos y vórtices cónicos:** Se verificó que el registro extremo $C_{p,\min} = -8.04$ (toma 49 a $\theta = 15^\circ$) responde físicamente a un vórtice cónico de borde de ataque y no a ruido instrumental. Su tratamiento mediante el método del Factor de Pico Empírico y regresión por cuantiles ($\tau = 0.01$) evita la dilución de colas pesadas inherente a los estimadores de media condicional.
5. **Integración con la NTE E.020 y alcance:** La conversión rigurosa de escala temporal y de capa límite ($\kappa_{\text{conv}} = 0.513$) acoplada a un agrupamiento no supervisado (*K-Means*) confirma que el factor $C = -0.60$ de la NTE E.020 representa fielmente el campo central de la cubierta ($C_{\text{eq}} = -0.69$), pero subestima las succiones locales en aristas y esquinas ($C_{\text{eq}} = -4.12$). Como limitación intrínseca, dado que el entrenamiento proviene de una única relación geométrica ($H/B = 0.25, \beta = 5^\circ$), el modelo no debe extrapolarse a edificios con distinta esbeltez o pendiente sin incorporar nuevas geometrías de la base TPU, ni sustituye la responsabilidad legal del ingeniero estructural bajo la norma vigente.

---

## Referencias

[1] Tokyo Polytechnic University, "TPU Aerodynamic Database," Wind Engineering Information Center, 2024. [En línea]. Disponible en: https://db.wind.arch.t-kougei.ac.jp/

[2] Y. Quan, Y. Tamura, M. Matsui, S. Y. Cao y A. Yoshida, "TPU aerodynamic database for low-rise buildings," en *Proc. 12th Int. Conf. Wind Engineering (ICWE12)*, Cairns, Australia, vol. 2, 2007, pp. 1615–1622.

[3] J. A. Main y W. P. Fritz, *Database-Assisted Design for Wind: Concepts, Software, and Examples for Rigid and Flexible Buildings*, NIST Building Science Series 180, Gaithersburg, MD, EE. UU.: National Institute of Standards and Technology, 2006.

[4] B. Yu y R. L. Barter, *Veridical Data Science: The Practice of Responsible Data Analysis and Decision Making*. Cambridge, MA, EE. UU.: MIT Press, 2024. [En línea]. Disponible en: https://vdsbook.com/

[5] Ministerio de Vivienda, Construcción y Saneamiento, *Reglamento Nacional de Edificaciones, Norma Técnica E.020 Cargas*, Lima, Perú: MVCS, 2006.

[6] L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001, doi: 10.1023/A:1010933404324.

[7] P. Geurts, D. Ernst y L. Wehenkel, "Extremely randomized trees," *Machine Learning*, vol. 63, no. 1, pp. 3–42, 2006, doi: 10.1007/s10994-006-6226-1.

[8] J. H. Friedman, "Greedy function approximation: A gradient boosting machine," *Annals of Statistics*, vol. 29, no. 5, pp. 1189–1232, 2001.

[9] F. Pedregosa *et al.*, "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.

[10] S. Park, D. Duthinh, E. Simiu y D. Yeo, "Wind effects on a tall building with square cross-section and mid-side base columns: Database-assisted design approach," *Journal of Structural Engineering*, vol. 145, no. 5, art. 06019001, 2019, doi: 10.1061/(ASCE)ST.1943-541X.0002328.

[11] Grupo 5, "TA_GRUPO5: Predicción de estadísticos del coeficiente de presión de viento en un edificio bajo (Entregables 1 y 2, Curso IA en Estructuras)," UPC, 2026. [En línea]. Disponible en: https://github.com/eduardocristobal740-star/TA_GRUPO5

[12] F.-B. Chen, X.-Z. Li, Y. Chen *et al.*, "Surrogate and artificial neural network models of wind pressure coefficients on low-rise buildings based on the TPU aerodynamic database," *Journal of Wind Engineering and Industrial Aerodynamics*, vol. 191, pp. 123–139, 2019.

[13] H. Dongmei, G. Hu, K. C. S. Kwok *et al.*, "Machine learning and deep learning for wind pressure prediction on buildings: Methodology and cross-geometry validation," *Journal of Wind Engineering and Industrial Aerodynamics*, vol. 214, art. 104652, 2021.

[14] E. Simiu y D. Yeo, *Wind Effects on Structures: Modern Structural Design for Wind*, 4.ª ed. Hoboken, NJ, EE. UU.: John Wiley & Sons, 2019.