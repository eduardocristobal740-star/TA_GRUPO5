"""Paso 1: MAT -> tabla consolidada (data/processed/tpu_cp_table.csv) + QC."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tpu_cp.config import PROCESSED_DIR, TAB_DIR
from tpu_cp.data_loading import build_dataset

if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True); TAB_DIR.mkdir(parents=True, exist_ok=True)
    df, qc = build_dataset()
    df.to_csv(PROCESSED_DIR / "tpu_cp_table.csv", index=False)
    (TAB_DIR / "control_calidad.json").write_text(json.dumps(qc, indent=2, ensure_ascii=False))
    print(json.dumps(qc, indent=2, ensure_ascii=False)); print(df.describe().T.round(4))
