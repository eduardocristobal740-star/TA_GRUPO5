# Predicción de estadísticos del coeficiente de presión de viento en un edificio bajo con cubierta a dos aguas mediante aprendizaje automático y validación angular

*Grupo 5 — Curso IA en Estructuras, Programa de Ingeniería de Diseño Estructural, UPC (2026)*

## Resumen (Abstract)

**Resumen—** Se comparan seis algoritmos de regresión, dos conjuntos de variables y dos líneas base sobre cuatro estadísticos del coeficiente de presión (Cp) —media, RMS, mínimo y máximo— medidos en 128 tomas de un edificio bajo (H = 4 m, B = D = 16 m, cubierta a dos aguas de 5°) de la base aerodinámica de la Tokyo Polytechnic University (TPU) para siete direcciones de viento (0°–90°). La validación reserva un ángulo completo por pliegue (LeaveOneGroupOut) para evitar la fuga entre sensores repetidos. Los mejores modelos reducen el MAE de la línea base de mediana entre 68.8 % y 78.2 %, con R² entre 0.88 y 0.95 (media de los pliegues). La interpolación angular del propio sensor resulta una referencia muy competitiva para el Cp medio (MAE 0.065), y las variables físicas añadidas (incidencia del viento, distancia de recorrido, distancia a bordes) mejoran unos objetivos y empeoran otros. Los resultados son válidos únicamente para la geometría ensayada y no sustituyen la NTE E.020 ni un estudio de viento.

**Palabras clave—** coeficiente de presión, túnel de viento, base de datos TPU, aprendizaje automático, validación por grupos, ingeniería de viento.

## I. Introducción

Las acciones de viento gobiernan el diseño de cubiertas y cerramientos de edificios bajos, donde las succiones locales son críticas. Los ensayos en túnel de viento y la simulación CFD son costosos, por lo que las bases de datos aerodinámicas abiertas, como la de la TPU \[1\], \[2\], se emplean en el diseño asistido por bases de datos \[3\] y como insumo para modelos de aprendizaje automático.

Como antecedente, Park et al. \[10\] emplearon registros de presión de la TPU en diseño asistido por bases de datos para evaluar la respuesta de un edificio alto ante distintas direcciones de viento; su uso en aprendizaje automático requiere, sin embargo, una evaluación predictiva propia, que es lo que aborda este trabajo.

Un riesgo metodológico frecuente es la **fuga de información**: en estos ensayos el mismo sensor aparece en varias direcciones y los puntos vecinos están correlacionados, de modo que una partición aleatoria por filas produce métricas optimistas. Siguiendo el marco de ciencia de datos veraz (PCS: predecibilidad, computabilidad y estabilidad) \[4\], la evaluación debe reproducir el escenario de uso. Aquí el escenario es *predecir un ángulo de viento no ensayado en la misma geometría*.

**Objetivos.** (i) consolidar y auditar los registros de presión de los laboratorios; (ii) comparar algoritmos y conjuntos de variables con validación por ángulo reservado; (iii) evaluar la estabilidad de las conclusiones frente a perturbaciones; (iv) delimitar el alcance de uso.

**Alcance de los datos.** Los archivos suministrados corresponden a un único conjunto experimental (siete archivos MAT, una geometría). El documento del Laboratorio 1 aporta el planteamiento del problema y el marco PCS, pero no contiene registros adicionales; por ello el entrenamiento utiliza exclusivamente los datos del Laboratorio 2.

## II. Metodología

### A. Conjunto de datos

Se usaron siete archivos MAT de la base TPU para edificios bajos con cubierta a dos aguas \[1\], \[2\], uno por dirección de viento (0°, 15°, 30°, 45°, 60°, 75° y 90°). Cada archivo contiene una matriz de 9 000 × 128 de Cp (15 Hz, 600 s a escala real; presión de referencia a la altura media de cubierta), las coordenadas de las 128 tomas y su número de superficie. Según los metadatos, las series ya están promediadas en ventanas móviles de 0.2 s. La geometría es H = 4 m, B = D = 16 m y pendiente de 5°. En total hay 8 064 000 valores de Cp.

El control de calidad no encontró valores NaN ni infinitos, la geometría es idéntica en los siete archivos y no hay combinaciones sensor-ángulo duplicadas. Cada serie se resumió en cuatro estadísticos por sensor (media, desviación estándar poblacional, mínimo y máximo), lo que produce **896 registros** (128 sensores × 7 ángulos). Estos registros no son 896 ensayos independientes. Las coordenadas (x, y) están en un desarrollo plano de la envolvente (superficies 1 y 3: hastiales; 2 y 4: muros largos; 5 y 6: faldones de cubierta) y no existe coordenada z.

| Superficie | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- |
| Tomas | 16 | 16 | 16 | 16 | 32 | 32 |
| Tipo | muro | muro | muro | muro | cubierta | cubierta |

### B. Variables predictoras

Se evaluaron dos conjuntos. El conjunto **base** contiene x, y, la superficie (one-hot) y sen θ, cos θ. El conjunto **físico** añade: indicador de cubierta; posición en planta (px, py); distancia a la cumbrera, al borde y a la esquina más cercana; coseno de incidencia entre la normal exterior de la superficie y el viento (`wind_incidence`) y su indicador de barlovento; distancia recorrida por el flujo desde la esquina de barlovento (`fetch`) y coordenada transversal; y θ. No se usa el identificador del sensor como predictor. La convención de orientación (muro 1 a barlovento a 0° y muro 2 a 90°) es un **supuesto inferido** del signo del Cp medio por superficie (Fig. 1) y debe verificarse con el plano de tomas de la TPU.

&#91;image: Fig. 1. Cp medio y Cp mínimo medio por superficie y ángulo.\]

### C. Algoritmos y líneas base

Se compararon: mediana constante (línea base), Ridge, Random Forest, Extra Trees, Gradient Boosting e HistGradientBoosting, todos con escalado y codificación dentro de un `Pipeline` para que cada pliegue ajuste el preprocesamiento solo con su entrenamiento. Además se incluyeron dos referencias que **no usan las variables**, sino la historia angular de cada sensor: interpolación lineal e interpolación PCHIP de Cp frente a θ, con extrapolación en los ángulos extremos. Un modelo **híbrido** (Extra Trees o HistGradientBoosting) recibe, como variable adicional, la interpolación del sensor calculada sin usar el ángulo de la fila (leave-one-angle-out dentro del entrenamiento), lo que evita fuga. Para Extra Trees e HistGradientBoosting con variables físicas se realizó un ajuste de hiperparámetros con una malla pequeña (4 combinaciones) y validación interna por ángulo, anidada dentro de cada pliegue externo. No se incluyó XGBoost; el boosting se cubre con los dos modelos de scikit-learn.

### D. Validación y métricas

Se usó LeaveOneGroupOut con θ como grupo: siete pliegues, cada uno con un ángulo completo reservado. Los pliegues de 0° y 90° exigen extrapolar; los cinco restantes interpolan. Se reportan MAE, RMSE, R², error máximo y sesgo, como media (y desviación estándar) entre pliegues. Esta validación mide generalización angular dentro de una geometría y **no** generalización a otros edificios.

### E. Estabilidad

Sobre Extra Trees con variables físicas y el mismo esquema por ángulo se aplicaron tres perturbaciones: 8 semillas aleatorias, 12 remuestreos bootstrap de sensores (se remuestrean sensores completos, no filas, para respetar la dependencia) y 8 réplicas con ruido gaussiano de 5 % de la desviación del objetivo añadido al entrenamiento. La importancia de variables se calculó por permutación **en el ángulo reservado**.

## III. Resultados y Discusión

### A. Desempeño frente a la línea base

Todos los modelos no lineales superan con amplitud a la mediana constante. La Tabla I resume el mejor candidato por objetivo; el MAE baja entre 68.8 % y 78.2 % y el RMSE entre 66.9 % y 80.7 %. Los R² medios por pliegue van de 0.879 (Cp mínimo) a 0.953 (Cp medio). Como el mejor modelo se elige sobre los mismos pliegues con los que se reporta, estas cifras son algo optimistas.

**Tabla I. Mejor candidato por objetivo (media de 7 pliegues angulares).**

| Objetivo | Mejor candidato | MAE | RMSE | R² | MAE mediana | RMSE mediana | Mejora MAE | Mejora RMSE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cp medio | Interp. angular PCHIP | 0.0654 | 0.0843 | 0.953 | 0.2995 | 0.4374 | 78.2 % | 80.7 % |
| Cp RMS | Extra Trees ajustado (físicas) | 0.0272 | 0.0404 | 0.883 | 0.0952 | 0.1317 | 71.4 % | 69.3 % |
| Cp mínimo | Extra Trees (base) | 0.2504 | 0.3849 | 0.879 | 0.8029 | 1.1614 | 68.8 % | 66.9 % |
| Cp máximo | HistGradientBoosting (físicas) | 0.1349 | 0.1892 | 0.946 | 0.4922 | 0.8952 | 72.6 % | 78.9 % |

Para el Cp medio el mejor candidato es una referencia sin aprendizaje automático; el mejor modelo de aprendizaje es el híbrido Extra Trees con variables físicas (MAE 0.0671, RMSE 0.0896, R² 0.955), prácticamente igual dentro de la dispersión entre pliegues.

**Tabla II. Comparativa de modelos (MAE y R², media de 7 pliegues; f = variables físicas, b = base).**

| Modelo | Cp medio MAE | R² | Cp RMS MAE | R² | Cp mín. MAE | R² | Cp máx. MAE | R² |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mediana (línea base) | 0.2995 | −0.004 | 0.0952 | −0.143 | 0.8029 | −0.091 | 0.4922 | −0.163 |
| Ridge (f) | 0.2105 | 0.616 | 0.0461 | 0.727 | 0.4965 | 0.655 | 0.2954 | 0.770 |
| Random Forest (f) | 0.0979 | 0.899 | 0.0380 | 0.772 | 0.3107 | 0.817 | 0.1523 | 0.927 |
| Extra Trees (b) | 0.0799 | 0.928 | 0.0289 | 0.853 | 0.2504 | 0.879 | 0.1538 | 0.915 |
| Extra Trees (f) | 0.0833 | 0.928 | 0.0317 | 0.839 | 0.2785 | 0.861 | 0.1477 | 0.931 |
| HistGradientBoosting (f) | 0.0906 | 0.914 | 0.0376 | 0.787 | 0.3188 | 0.823 | 0.1349 | 0.946 |
| Interp. angular lineal | 0.0731 | 0.948 | 0.0307 | 0.831 | 0.3184 | 0.779 | 0.1617 | 0.903 |
| Interp. angular PCHIP | 0.0654 | 0.953 | 0.0336 | 0.789 | 0.3957 | 0.550 | 0.2231 | 0.812 |
| Híbrido Extra Trees (f) | 0.0671 | 0.955 | 0.0291 | 0.865 | 0.2545 | 0.880 | 0.1458 | 0.931 |
| Extra Trees ajustado (f) | 0.0747 | 0.941 | 0.0272 | 0.883 | 0.2594 | 0.870 | 0.1432 | 0.938 |
| HistGradientBoosting ajustado (f) | 0.0841 | 0.922 | 0.0383 | 0.781 | 0.3147 | 0.825 | 0.1371 | 0.946 |

&#91;image: Fig. 2. MAE de los siete mejores candidatos por objetivo (barras de error: desviación entre pliegues). Naranja: interpolación del sensor; verde: híbrido; azul: modelos con variables.\]

La dispersión entre pliegues (Fig. 2) es del mismo orden que las diferencias entre los mejores candidatos, por lo que **no se debe afirmar un ranking estricto** entre ellos; sí es clara la separación respecto de Ridge y de la mediana. Con solo siete pliegues no se aplicó una prueba de significancia.

### B. Efecto de las variables físicas y del ajuste

Las variables físicas ayudan de forma no uniforme. Mejoran a Ridge en los cuatro objetivos (p. ej. Cp medio 0.2521 → 0.2105) y a HistGradientBoosting en Cp medio (0.1097 → 0.0906) y Cp máximo (0.1768 → 0.1349). En cambio, Extra Trees y Random Forest empeoran en Cp RMS y Cp mínimo (Extra Trees: 0.0289 → 0.0317 y 0.2504 → 0.2785), probablemente porque añadir variables correlacionadas diluye las divisiones útiles con solo 6 ángulos de entrenamiento. El ajuste de hiperparámetros tampoco fue uniformemente útil: mejoró a Extra Trees en los cuatro objetivos (p. ej. Cp RMS 0.0317 → 0.0272) pero no a HistGradientBoosting en Cp máximo (0.1349 → 0.1371).

La importancia por permutación en el ángulo reservado (aumento del MAE al permutar) destaca a la incidencia del viento para Cp máximo (0.316) y Cp medio (0.152), y a la distancia de recorrido (*fetch*) para Cp mínimo (0.365), Cp RMS (0.065) y Cp medio (0.120), junto con la superficie (0.106 para Cp máximo y 0.251 para Cp mínimo). Las demás variables aportan poco. Esto es coherente con la física del problema: las presiones dependen de qué cara está a barlovento y de la distancia desde la arista de ataque.

### C. Interpolación frente a extrapolación angular

&#91;image: Fig. 3. MAE por ángulo reservado: mejor modelo, interpolación lineal del sensor y mediana.\]

La interpolación por sensor es muy buena en los ángulos interiores (Cp medio: MAE 0.0457 con PCHIP frente a 0.0577 del híbrido), pero se degrada al extrapolar en 0° y 90°: en Cp mínimo su MAE de borde es 0.4977 (lineal) y 0.7420 (PCHIP), frente a 0.2796 del híbrido y 0.2851 de Extra Trees. El híbrido combina ambas ventajas en Cp medio (MAE de borde 0.0905 frente a 0.1145 de PCHIP). Con solo siete direcciones separadas 15°, extrapolar fuera del rango \[0°, 90°\] no está respaldado por estos datos.

&#91;image: Fig. 4. Valores observados frente a predichos (predicciones fuera de muestra, mejor candidato por objetivo).\]

&#91;image: Fig. 5. Cp mínimo a 45° (pliegue reservado, Extra Trees base): observado, predicho y error sobre el desarrollo plano.\]

Los errores mayores se concentran en pocas tomas del hastial y en puntos de borde de la cubierta con gradientes fuertes (Fig. 5). Para Cp mínimo el error máximo de un pliegue llega a 2.70 con el mejor modelo (la succión más intensa del conjunto es −8.04, toma 49 a 15°), de modo que el promedio de error oculta fallos locales relevantes para el diseño.

### D. Estabilidad (PCS)

**Tabla III. MAE de Extra Trees (variables físicas) bajo perturbaciones, media ± desv. entre réplicas.**

| Objetivo | Semillas (8) | Bootstrap de sensores (12) | Ruido 5 % (8) |
| --- | --- | --- | --- |
| Cp medio | 0.0833 ± 0.0004 | 0.0920 ± 0.0022 | 0.0835 ± 0.0010 |
| Cp RMS | 0.0317 ± 0.0003 | 0.0343 ± 0.0012 | 0.0316 ± 0.0002 |
| Cp mínimo | 0.2810 ± 0.0031 | 0.2918 ± 0.0066 | 0.2818 ± 0.0025 |
| Cp máximo | 0.1466 ± 0.0015 | 0.1520 ± 0.0034 | 0.1477 ± 0.0020 |

La semilla y el ruido de 5 % apenas alteran el MAE. El bootstrap de sensores lo aumenta entre 3.7 % y 10.4 % (al perder y repetir sensores el modelo ve menos posiciones distintas), pero la variación entre réplicas sigue siendo pequeña. La mayor fuente de variabilidad es el ángulo reservado (desviaciones entre pliegues de Tabla II y Fig. 3), no el azar del algoritmo.

### E. Limitaciones

(1) Una sola geometría: no se puede inferir efecto de H, B, D, H/B ni pendiente. (2) Siete direcciones espaciadas 15° y sensores correlacionados. (3) Los extremos son del registro de 10 min a escala real con filtro de 0.2 s, no valores de diseño por periodo de retorno. (4) La convención de orientación y la geometría del desarrollo plano se infirieron; deben contrastarse con la documentación TPU. (5) La selección del mejor modelo sobre los mismos pliegues es optimista. (6) Los resultados no sustituyen la NTE E.020 \[5\], un estudio de viento específico ni la revisión de un ingeniero estructural.

## IV. Conclusiones

1. Con modelos de conjuntos de árboles y boosting \[6\]–\[9\] y validación por ángulo reservado, los cuatro estadísticos de Cp se predicen con R² medio entre 0.88 y 0.95 y una reducción del MAE de 68.8 % a 78.2 % frente a la mediana, para direcciones no ensayadas de esta geometría.
2. La interpolación angular de cada sensor es una línea base exigente: iguala a los modelos en Cp medio, pero falla al extrapolar (0° y 90°) en los extremos de Cp. Cualquier modelo propuesto debe compararse con ella.
3. Las variables físicas (incidencia del viento, distancia de recorrido) dominan la importancia, pero su efecto en el error depende del algoritmo y del objetivo; no mejoran de forma general. El ajuste de hiperparámetros aportó ganancias pocas y desiguales.
4. Las conclusiones son estables frente a semillas y ruido (variación del MAE inferior a 2 %) y moderadamente sensibles a la composición de sensores (aumento de 3.7 % a 10.4 % bajo bootstrap de sensores).
5. Los resultados no se extienden a otras geometrías ni a valores de diseño. Como trabajo futuro: validación con geometrías completas reservadas (otras relaciones H/B), más direcciones, y comparación cuantitativa con la NTE E.020 con un protocolo de escalamiento explícito. El código y las tablas de resultados reproducibles se entregan en el repositorio del proyecto \[11\].

## Referencias

&#91;1\] Tokyo Polytechnic University, "TPU Aerodynamic Database," Wind Engineering Information Center. \[En línea\]. Disponible: https://db.wind.arch.t-kougei.ac.jp/

&#91;2\] Y. Quan, Y. Tamura, M. Matsui, S. Y. Cao y A. Yoshida, "TPU aerodynamic database for low-rise buildings," en *Proc. 12th Int. Conf. Wind Engineering*, vol. 2, 2007, pp. 1615–1622.

&#91;3\] J. A. Main y W. P. Fritz, *Database-Assisted Design for Wind: Concepts, Software, and Examples for Rigid and Flexible Buildings*, NIST Building Science Series 180, 2006.

&#91;4\] B. Yu y R. L. Barter, *Veridical Data Science*. Cambridge, MA, EE. UU.: MIT Press, 2024. \[En línea\]. Disponible: https://vdsbook.com/

&#91;5\] Ministerio de Vivienda, Construcción y Saneamiento, *Reglamento Nacional de Edificaciones, Norma Técnica E.020 Cargas*, Perú.

&#91;6\] L. Breiman, "Random forests," *Machine Learning*, vol. 45, pp. 5–32, 2001.

&#91;7\] P. Geurts, D. Ernst y L. Wehenkel, "Extremely randomized trees," *Machine Learning*, vol. 63, pp. 3–42, 2006.

&#91;8\] J. H. Friedman, "Greedy function approximation: A gradient boosting machine," *Annals of Statistics*, vol. 29, no. 5, pp. 1189–1232, 2001.

&#91;9\] F. Pedregosa *et al.*, "Scikit-learn: Machine learning in Python," *J. Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.

&#91;10\] S. Park, D. Duthinh, E. Simiu y D. Yeo, "Wind effects on a tall building with square cross-section and mid-side base columns: Database-assisted design approach," *J. Structural Engineering*, vol. 145, no. 5, art. 06019001, 2019, doi: 10.1061/(ASCE)ST.1943-541X.0002328.

&#91;11\] Grupo 5, "TA\_GRUPO5: Entregables 1 y 2, curso IA en Estructuras," UPC, 2026. \[En línea\]. Disponible: https://github.com/eduardocristobal740-star/TA\_GRUPO5
