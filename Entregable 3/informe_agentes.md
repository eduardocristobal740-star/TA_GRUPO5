# Informe técnico de resultados

Modo de elaboración: reglas locales deterministas, sin LLM.

Las afirmaciones se reproducen desde evidencias verificadas por código. La aprobación del flujo corresponde a trazabilidad editorial, no a una certificación técnica independiente.

## Datos y procedencia

- Los archivos utilizados proceden de https://github.com/eduardocristobal740-star/TA\_GRUPO5; revisión o referencia: ba0bb6b8a903dcca7464423764fabb4d1b49c69a. [F_SOURCE]

- La base contiene 896 observaciones, 128 sensores y 7 direcciones del viento. [F_DATA]

- Se cotejaron 71680 predicciones publicadas y se recalcularon las métricas de 560 pliegues; este proceso no repite el entrenamiento. [F_VALIDATION]

- LeaveOneGroupOut por ángulo: cada dirección se predice sin entrenar con sus registros. [F_METHOD]

## Comparación de modelos

- Para cp\_mean, el menor MAE publicado corresponde a Interp. angular PCHIP con predictores sensor: MAE=0.06535046467, RMSE=0.08432198547 y R²=0.9533776995. La clasificación utiliza los mismos pliegues publicados y no constituye una prueba externa. [C_CP_MEAN]

- Para cp\_rms, el menor MAE publicado corresponde a ExtraTrees (ajustado) con predictores fisica: MAE=0.02723792415, RMSE=0.04036340846 y R²=0.8831349931. La clasificación utiliza los mismos pliegues publicados y no constituye una prueba externa. [C_CP_RMS]

- Para cp\_min, el menor MAE publicado corresponde a ExtraTrees con predictores base: MAE=0.2504460493, RMSE=0.3848800163 y R²=0.8786246074. La clasificación utiliza los mismos pliegues publicados y no constituye una prueba externa. [C_CP_MIN]

- Para cp\_max, el menor MAE publicado corresponde a HistGradientBoosting con predictores fisica: MAE=0.1348507678, RMSE=0.1892332708 y R²=0.946108828. La clasificación utiliza los mismos pliegues publicados y no constituye una prueba externa. [C_CP_MAX]

## Alcance y limitaciones

- La evaluación corresponde a una sola geometría y no acredita generalización a otros edificios. [L_01]

- La selección del mejor candidato sobre los mismos pliegues puede introducir optimismo de selección. [L_02]

- El explorador consulta predicciones existentes; no entrena modelos ni infiere resultados para direcciones arbitrarias. [L_03]

- La revisión de agentes verifica cobertura y trazabilidad del informe; no certifica la validez física del modelo ni sustituye la revisión del equipo responsable. [L_04]

- Se evalúa una sola geometría. La validación mide generalización angular, no a otros edificios. [L_05]

- Seleccionar el mejor candidato sobre los mismos pliegues puede producir optimismo de selección. [L_06]

- Los extremos proceden de un registro filtrado de diez minutos, no de un periodo de retorno de diseño. [L_07]

- La orientación y el desarrollo plano deben contrastarse con la documentación de TPU. [L_08]

## Revisión del flujo

Estado: aprobado por el control determinista de cobertura y por el agente revisor.

El detalle de roles, prompts del sistema, entradas, salidas y marcas de tiempo se conserva en el registro JSON de agentes.
