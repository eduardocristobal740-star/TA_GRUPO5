"""Pruebas de humo: forma de la tabla, ausencia de fuga y coherencia de variables."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tpu_cp.data_loading import build_dataset, summarize_series
from tpu_cp.features import add_features
from tpu_cp.validation import sensor_interpolation_cv


def test_summarize_series():
    cp = np.array([[1.0, -1.0], [3.0, -3.0]])
    s = summarize_series(cp)
    assert np.allclose(s["cp_mean"], [2, -2]) and np.allclose(s["cp_rms"], [1, 1])
    assert np.allclose(s["cp_min"], [1, -3]) and np.allclose(s["cp_max"], [3, -1])


def test_dataset_shape_and_features():
    df, qc = build_dataset()
    assert df.shape[0] == 896 and qc["geometrias_distintas"] == 1 and qc["valores_nan"] == 0
    f = add_features(df)
    assert f[["fetch", "wind_incidence", "edge_dist"]].notna().all().all()
    assert (f.loc[(f.surface == 1) & (f.theta == 0), "wind_incidence"] == 1).all()


def test_interpolation_never_uses_heldout_angle():
    df, _ = build_dataset()
    p = sensor_interpolation_cv(df, "cp_mean", "linear")
    assert len(p) == 896 and set(p.theta) == set(df.theta)
