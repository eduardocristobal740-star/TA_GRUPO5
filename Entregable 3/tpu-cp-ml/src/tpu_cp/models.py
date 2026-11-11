"""Catálogo de modelos, preprocesamiento y mallas de hiperparámetros."""
from __future__ import annotations

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import RANDOM_STATE
from .features import CATEGORICAL


def make_preprocessor(features: list[str]) -> ColumnTransformer:
    """Escala numéricas y codifica en one-hot las categóricas (todo dentro del Pipeline)."""
    cat = [f for f in features if f in CATEGORICAL]
    num = [f for f in features if f not in CATEGORICAL]
    return ColumnTransformer(
        [("num", StandardScaler(), num),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)],
        remainder="drop")


def base_estimators() -> dict:
    """Estimadores base (sin ajuste de hiperparámetros)."""
    rs = RANDOM_STATE
    return {
        "Dummy (mediana)": DummyRegressor(strategy="median"),
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=rs, n_jobs=-1),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, random_state=rs, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=250, learning_rate=0.05, max_depth=3,
                                                      subsample=0.8, random_state=rs),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05,
                                                              l2_regularization=1.0, random_state=rs),
    }


# Mallas pequeñas: con 6 ángulos de entrenamiento no se justifica una búsqueda amplia.
PARAM_GRIDS = {
    "ExtraTrees": {"modelo__min_samples_leaf": [1, 3], "modelo__max_features": [0.5, 1.0]},
    "HistGradientBoosting": {"modelo__learning_rate": [0.05, 0.1], "modelo__min_samples_leaf": [10, 30]},
}


def make_pipeline(estimator, features: list[str]) -> Pipeline:
    return Pipeline([("pre", make_preprocessor(features)), ("modelo", clone(estimator))])


def make_tuned(estimator, grid: dict, features: list[str]) -> GridSearchCV:
    """Pipeline + búsqueda interna con LeaveOneGroup (los grupos se pasan en fit)."""
    return GridSearchCV(make_pipeline(estimator, features), grid, cv=LeaveOneGroupOut(),
                        scoring="neg_mean_absolute_error", n_jobs=-1, refit=True)
