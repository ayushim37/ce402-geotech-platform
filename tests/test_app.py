"""Smoke test: the app must run without exceptions on (a) the generic starter data and (b) a bundled example."""
import sys, os, json
import pandas as pd
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from streamlit.testing.v1 import AppTest

COLS = ["Name", "Type", "Top (m)", "Bottom (m)", "γ moist (kN/m³)", "γ sat (kN/m³)", "c′ (kPa)", "φ′ (°)",
        "cu (kPa)", "SPT N (blows/0.3 m)", "qc/N (kPa per blow)", "E user (kPa)", "Cc (–)", "Cr (–)",
        "e0 top (–)", "e0 bottom (–)", "cv (m²/yr)", "OCR (–)", "σ′p add (kPa)"]
APP = os.path.join(ROOT, "app.py")


def run(state=None):
    at = AppTest.from_file(APP, default_timeout=90)
    for k, v in (state or {}).items():
        at.session_state[k] = v
    return at.run()


at = run()
assert not at.exception, [e.value for e in at.exception]
print("generic starter data: OK, tabs =", len(at.tabs), ", tables =", len(at.dataframe))

for bh in ("BH-1", "BH-2", "BH-3"):
    d = json.load(open(os.path.join(ROOT, "examples", f"mandi_hostel_{bh}.json")))
    st_ = {"ver": 1, "layers": pd.DataFrame(d["layers"]).reindex(columns=COLS), "wt_meas": d["wt_measured"],
           "wt_des": d["wt_design"], "spt": pd.DataFrame(d["spt"]), "ds": pd.DataFrame(d["direct_shear"])}
    at = run(st_)
    assert not at.exception, (bh, [e.value for e in at.exception])
    assert not at.error, (bh, [e.value for e in at.error])
    print(bh, "example: OK, metrics:", [(m.label, m.value) for m in at.metric][:8])
print("APP SMOKE TESTS PASSED")
