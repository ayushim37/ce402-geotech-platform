"""
CE 402 - Shallow foundation design platform (Streamlit).
Modules: P1 soil profile | P2 parameter tools | P3 bearing capacity | P4 stress increase | P5 settlement.
NO site values are hard-coded: all data are typed in, uploaded as JSON, or loaded from examples/*.json.
All calculations live in geotech.py, where every equation carries a source comment.
SI units throughout: kN, kPa, m, mm.
"""
import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import geotech as gt

st.set_page_config(page_title="Foundation design platform", layout="wide")

KINDS = ["fill", "sand", "gravel", "clay"]
COLS = ["Name", "Type", "Top (m)", "Bottom (m)", "γ moist (kN/m³)", "γ sat (kN/m³)", "c′ (kPa)", "φ′ (°)",
        "cu (kPa)", "SPT N (blows/0.3 m)", "qc/N (kPa per blow)", "E user (kPa)", "Cc (–)", "Cr (–)",
        "e0 top (–)", "e0 bottom (–)", "cv (m²/yr)", "OCR (–)", "σ′p add (kPa)"]
GENERIC = [  # neutral placeholder so the page opens; replace with your own data
    ["Sand", "sand", 0.0, 3.0, 17.0, 19.0, 0.0, 32.0, 0.0, 12.0, 400.0, None, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    ["Clay", "clay", 3.0, 8.0, 18.0, 18.0, 0.0, 0.0, 40.0, None, 400.0, None, 0.25, 0.04, 0.90, 0.85, 3.0, 1.0, 0.0],
    ["Gravel", "gravel", 8.0, 20.0, 20.0, 21.0, 0.0, 38.0, 0.0, 50.0, 800.0, None, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
]
SPT_COLS = ["Depth (m)", "N (blows/0.3 m)", "Saturated fine/silty sand? (dilatancy)"]
DS_COLS = ["σn (kPa)", "τf (kPa)"]


# ------------------------------------------------------------------ state helpers
def init_state():
    ss = st.session_state
    ss.setdefault("ver", 0)
    ss.setdefault("layers", pd.DataFrame(GENERIC, columns=COLS))
    ss.setdefault("wt_meas", 2.0)
    ss.setdefault("wt_des", 1.0)
    ss.setdefault("spt", pd.DataFrame([[1.5, 10, False], [3.0, 14, False]], columns=SPT_COLS))
    ss.setdefault("ds", pd.DataFrame([[50.0, 40.0], [100.0, 72.0], [150.0, 105.0]], columns=DS_COLS))


def load_json(d):
    ss = st.session_state
    ss["layers"] = pd.DataFrame(d["layers"]).reindex(columns=COLS)
    ss["wt_meas"] = float(d["wt_measured"]); ss["wt_des"] = float(d["wt_design"])
    if "spt" in d:
        ss["spt"] = pd.DataFrame(d["spt"]).reindex(columns=SPT_COLS)
    if "direct_shear" in d:
        ss["ds"] = pd.DataFrame(d["direct_shear"]).reindex(columns=DS_COLS)
    ss["ver"] += 1


def nz(v, default=0.0):
    return default if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def validate(df, wm, wd):
    errs = []
    if df.empty:
        return ["Enter at least one layer."]
    d = df.reset_index(drop=True)
    for i, r in d.iterrows():
        if r["Type"] not in KINDS: errs.append(f"Layer {i+1}: type must be one of {KINDS}.")
        if nz(r["Bottom (m)"]) <= nz(r["Top (m)"]): errs.append(f"Layer {i+1}: bottom must be deeper than top.")
        if nz(r["γ sat (kN/m³)"]) < nz(r["γ moist (kN/m³)"]) - 1e-9:
            errs.append(f"Layer {i+1}: saturated unit weight is below moist unit weight (check units).")
        if i > 0 and abs(nz(r["Top (m)"]) - nz(d.loc[i-1, "Bottom (m)"])) > 1e-6:
            errs.append(f"Layer {i+1}: top does not equal the bottom of layer {i} (gap or overlap).")
    if abs(nz(d.loc[0, "Top (m)"])) > 1e-6: errs.append("First layer must start at 0 m.")
    if wm < 0 or wd < 0: errs.append("Water-table depths must be ≥ 0 m.")
    return errs


def to_profile(df, wm, wd):
    L = []
    for _, r in df.reset_index(drop=True).iterrows():
        N = r["SPT N (blows/0.3 m)"]; E = r["E user (kPa)"]
        L.append(gt.Layer(top=nz(r["Top (m)"]), bot=nz(r["Bottom (m)"]), name=str(r["Name"]), kind=r["Type"],
                          gamma=nz(r["γ moist (kN/m³)"]), gamma_sat=nz(r["γ sat (kN/m³)"]),
                          c=nz(r["c′ (kPa)"]), phi=nz(r["φ′ (°)"]), cu=nz(r["cu (kPa)"]),
                          N=None if pd.isna(N) else float(N), Cc=nz(r["Cc (–)"]), Cr=nz(r["Cr (–)"]),
                          e0_top=nz(r["e0 top (–)"]), e0_bot=nz(r["e0 bottom (–)"]), cv=nz(r["cv (m²/yr)"]),
                          ocr=nz(r["OCR (–)"], 1.0), sp_add=nz(r["σ′p add (kPa)"]),
                          E=None if pd.isna(E) else float(E), qc_factor=nz(r["qc/N (kPa per blow)"], 400.0)))
    return gt.Profile(layers=L, wt_measured=wm, wt_design=wd)


def fig_profile(p):
    fig, ax = plt.subplots(1, 2, figsize=(9, 5), gridspec_kw={"width_ratios": [1, 1.2]}, sharey=True)
    colors = {"fill": "#c9b79c", "sand": "#f0d98c", "gravel": "#b0b0b0", "clay": "#9db7a8"}
    for L in p.layers:
        ax[0].add_patch(plt.Rectangle((0, L.top), 1, L.bot - L.top, fc=colors.get(L.kind, "#ddd"), ec="k"))
        ax[0].text(0.5, (L.top + L.bot) / 2, f"{L.name}\n{L.top:g}–{L.bot:g} m", ha="center", va="center", fontsize=8)
    zmax = p.layers[-1].bot
    ax[0].axhline(p.wt_measured, color="tab:blue", ls="--", label=f"Measured WT {p.wt_measured:g} m")
    ax[0].axhline(p.wt_design, color="tab:red", label=f"Design WT {p.wt_design:g} m")
    ax[0].set_xlim(0, 1); ax[0].set_ylim(zmax, 0); ax[0].set_xticks([]); ax[0].set_ylabel("Depth below ground (m)")
    ax[0].legend(fontsize=7, loc="lower left"); ax[0].set_title("Soil profile")
    zs = np.linspace(0, zmax, 200)
    st_, se_ = zip(*[gt.sigma_v_total_eff(p, z) for z in zs])
    ax[1].plot(st_, zs, label="σv total"); ax[1].plot([s - t for s, t in zip(st_, se_)], zs, label="u (design WT)")
    ax[1].plot(se_, zs, label="σ′v effective", lw=2); ax[1].set_xlabel("Stress (kPa)"); ax[1].legend(fontsize=8)
    ax[1].set_title("Vertical stress at design WT"); ax[1].grid(alpha=.3)
    fig.tight_layout()
    return fig


def layer_options(p):
    return {f"{L.name} ({L.top:g}–{L.bot:g} m)": L for L in p.layers}


# ------------------------------------------------------------------ app
init_state()
ss = st.session_state
st.title("Shallow foundation design platform")
st.caption("IS 6403:1981 · IS 2131:1981 · Boussinesq · Schmertmann et al. (1978) · Terzaghi 1-D consolidation. "
           "SI units: kN, kPa, m, mm. Every equation is sourced in geotech.py and listed in the *Sources* tab.")

t1, t2, t3, t4, t5, t6 = st.tabs(["P1 Soil profile", "P2 Parameter tools", "P3 Bearing capacity",
                                  "P4 Stress increase", "P5 Settlement", "Sources"])

# ------------------------------------------------------------------ P1
with t1:
    st.subheader("P1 · Layered soil profile")
    c1, c2, c3 = st.columns(3)
    ex = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples", "*.json")))
    with c1:
        pick = st.selectbox("Bundled example (optional)", ["—"] + [os.path.basename(e) for e in ex])
        if st.button("Load example") and pick != "—":
            load_json(json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples", pick))))
            st.rerun()
    with c2:
        up = st.file_uploader("…or upload a profile (.json)", type="json")
        if up is not None and st.button("Load uploaded file"):
            load_json(json.load(up)); st.rerun()
    with c3:
        ss["wt_meas"] = st.number_input("Measured water table (m below ground)", 0.0, 100.0, float(ss["wt_meas"]), 0.1,
                                        help="Level observed at the time of the investigation. Used for SPT corrections.")
        ss["wt_des"] = st.number_input("Design water table (m below ground)", 0.0, 100.0, float(ss["wt_des"]), 0.1,
                                       help="Conservative seasonal level. Used for bearing capacity and settlement.")
    st.markdown("**Layers** (edit, add or delete rows; depths in m, unit weights kN/m³, strengths kPa). "
                "Leave cells blank when not applicable. **OCR** = overconsolidation ratio (1 = normally consolidated); "
                "σ′p = OCR·σ′v0 + σ′p add.")
    edited = st.data_editor(
        ss["layers"], num_rows="dynamic", width='stretch', key=f"layers_{ss['ver']}",
        column_config={"Type": st.column_config.SelectboxColumn("Type", options=KINDS, required=True)})
    ss["layers"] = edited
    errs = validate(edited, ss["wt_meas"], ss["wt_des"])
    if errs:
        for e in errs: st.error(e)
        st.stop()
    prof = to_profile(edited, ss["wt_meas"], ss["wt_des"])
    st.pyplot(fig_profile(prof))
    pts = sorted({0.0, *[L.bot for L in prof.layers], prof.wt_design})
    rows = []
    for z in pts:
        s, se = gt.sigma_v_total_eff(prof, z)
        rows.append({"Depth (m)": z, "σv total (kPa)": round(s, 1), "u (kPa)": round(s - se, 1), "σ′v (kPa)": round(se, 1)})
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
    st.download_button("Download this profile as JSON", json.dumps({
        "wt_measured": ss["wt_meas"], "wt_design": ss["wt_des"], "layers": json.loads(edited.to_json(orient="records")),
        "spt": json.loads(ss["spt"].to_json(orient="records")),
        "direct_shear": json.loads(ss["ds"].to_json(orient="records"))}, indent=1), "profile.json")
    st.caption("Capillary rise above the water table is ignored: moist unit weight is used above it, saturated below.")

prof = to_profile(ss["layers"], ss["wt_meas"], ss["wt_des"])

# ------------------------------------------------------------------ P2
with t2:
    st.subheader("P2 · Parameter tools")
    st.markdown("### IS 2131 corrections to SPT N")
    wtc = st.radio("Water table used for the correction", ["Measured (stress state during the test)", "Design"], horizontal=True,
                   help="The correction describes the stress when the test was done, so the measured level is the default.")
    wt_used = prof.wt_measured if wtc.startswith("Measured") else prof.wt_design
    cap = st.number_input("Cap on CN (–)", 1.0, 3.0, 2.0, 0.1, help="Limits the overburden factor at very low stress.")
    if st.button("Set 'saturated fine sand' flag from the profile (below chosen water table, sand layers)"):
        d = ss["spt"].copy()
        d[SPT_COLS[2]] = [(z >= wt_used) and prof.layer_at(z).kind == "sand" for z in d[SPT_COLS[0]]]
        ss["spt"] = d; ss["ver"] += 1; st.rerun()
    ss["spt"] = st.data_editor(ss["spt"], num_rows="dynamic", width='stretch', key=f"spt_{ss['ver']}")
    out = []
    for _, r in ss["spt"].dropna(subset=[SPT_COLS[0], SPT_COLS[1]]).iterrows():
        _, sv = gt.sigma_v_total_eff(prof, float(r[SPT_COLS[0]]), wt_used)
        c = gt.spt_corrections(float(r[SPT_COLS[1]]), sv, bool(r[SPT_COLS[2]]), cn_cap=cap)
        out.append({"Depth (m)": r[SPT_COLS[0]], "Layer": prof.layer_at(float(r[SPT_COLS[0]])).name, "N (field)": c["N"],
                    "σ′v (kPa)": round(c["sigma_v_eff"], 1), "CN = 0.77 log10(1920/σ′v) (–)": round(c["CN"], 3),
                    "N′ = CN·N": round(c["N_overburden"], 1), "N′ > 15 and saturated fine sand?": c["dilatancy_applied"],
                    "N″ = 15 + 0.5(N′−15)": round(c["N_design"], 1)})
    if out:
        st.dataframe(pd.DataFrame(out), width='stretch', hide_index=True)
    st.caption("Step 1: σ′v at the test depth. Step 2: overburden factor CN and N′ = CN·N (closed form of the "
               "IS 2131 chart: Peck, Hanson & Thornburn, quoted in Das; check against the chart in your copy). "
               "Step 3: dilatancy N″ = 15 + 0.5(N′ − 15) only if N′ > 15 in saturated fine or silty sand "
               "(Terzaghi & Peck, IS 2131). Gravelly or coarse sand: untick the flag.")
    with st.expander("Cross-check of φ′ from N′ (Wolff 1989; use only as a sanity check)"):
        for o in out:
            st.write(f"z = {o['Depth (m)']} m: φ′ ≈ {gt.phi_from_N(o['N′ = CN·N']):.1f}°")

    st.markdown("### Direct shear: c′ and φ′ by least squares")
    ss["ds"] = st.data_editor(ss["ds"], num_rows="dynamic", width='stretch', key=f"ds_{ss['ver']}")
    dsd = ss["ds"].dropna()
    if len(dsd) >= 2 and dsd[DS_COLS[0]].nunique() >= 2:
        sn, tau = list(dsd[DS_COLS[0]]), list(dsd[DS_COLS[1]])
        c, phi, r2 = gt.fit_cphi(sn, tau)
        sxx = sum(x * x for x in sn); phi0 = math.degrees(math.atan(sum(x * y for x, y in zip(sn, tau)) / sxx))
        m1, m2, m3 = st.columns(3)
        m1.metric("c′ (kPa)", f"{c:.2f}"); m2.metric("φ′ (°)", f"{phi:.2f}"); m3.metric("R² (–)", f"{r2:.4f}")
        st.write(f"Fit forced through the origin (c′ = 0): φ′ = {phi0:.2f}°")
        f, a = plt.subplots(figsize=(5, 3.5)); xs = np.linspace(0, max(sn) * 1.1, 10)
        a.plot(sn, tau, "o"); a.plot(xs, c + xs * math.tan(math.radians(phi)), label="least squares")
        a.set_xlabel("σn (kPa)"); a.set_ylabel("τf (kPa)"); a.grid(alpha=.3); a.legend(); st.pyplot(f)
        st.caption("τf = c′ + σn tanφ′, ordinary least squares. A small c′ from a soaked or remoulded test is often "
                   "set to zero in design; judge whether suction (unsoaked) or sampling effects are present.")
    else:
        st.info("Enter at least two different normal stresses.")

# ------------------------------------------------------------------ P3
with t3:
    st.subheader("P3 · Bearing capacity (IS 6403:1981)")
    a1, a2, a3 = st.columns(3)
    with a1:
        shape_ui = st.selectbox("Footing shape", ["Square", "Rectangular", "Strip", "Circular"])
        shape = {"Square": "square", "Rectangular": "rect", "Strip": "strip", "Circular": "circle"}[shape_ui]
        B = st.number_input("Width B, or diameter for circular (m)", 0.1, 100.0, 2.0, 0.05)
        Lf = st.number_input("Length L (m)", 0.1, 500.0, 3.0, 0.05, disabled=shape != "rect") if shape == "rect" else B
        Df = st.number_input("Founding depth Df (m)", 0.0, 50.0, 1.5, 0.1)
    with a2:
        src = st.radio("Strength parameters", ["Drained from profile layer at founding depth (c′, φ′)",
                                               "Undrained from profile layer (c = cu, φ = 0)", "Manual"])
        lay = prof.layer_at(Df)
        if src.startswith("Drained"):
            c_in, phi_in = lay.c, lay.phi
        elif src.startswith("Undrained"):
            c_in, phi_in = lay.cu, 0.0
        else:
            c_in = st.number_input("c (kPa)", 0.0, 1000.0, 0.0, 1.0); phi_in = st.number_input("φ (°)", 0.0, 50.0, 30.0, 0.5)
        st.write(f"Founding layer: **{lay.name}** · c = {c_in:g} kPa · φ = {phi_in:g}°")
        FoS = st.number_input("Factor of safety (–)", 1.0, 10.0, 3.0, 0.1)
    with a3:
        alpha = st.number_input("Load inclination α from vertical (°)", 0.0, 60.0, 0.0, 1.0)
        ex_ = st.number_input("Eccentricity e_B along B (m)", 0.0, 10.0, 0.0, 0.01)
        ey_ = st.number_input("Eccentricity e_L along L (m)", 0.0, 10.0, 0.0, 0.01, disabled=shape != "rect") if shape == "rect" else 0.0
        mode = st.radio("Shear failure mode", ["General", "Local", "Blend by SPT N″"], horizontal=True)
        if mode.startswith("Blend"):
            Nd = st.number_input("Design N″ at founding level (blows/0.3 m)", 0.0, 100.0, 15.0, 1.0)
            lo = st.number_input("N″ at or below which failure is fully local", 0.0, 50.0, 10.0, 1.0)
            hi = st.number_input("N″ at or above which failure is fully general", 1.0, 100.0, 30.0, 1.0)
            w = gt.local_general_weight(Nd, lo, hi); st.write(f"Weight on local shear = {w:.2f}")
        else:
            w = 1.0 if mode == "Local" else 0.0
    if shape == "strip":
        Lf = 1e6
    kw = dict(profile=prof, B=B, L=Lf, shape=shape, Df=Df, c=c_in, phi=phi_in, alpha=alpha, FoS=FoS, ex=ex_, ey=ey_)
    if shape == "circle" and ex_ > 0:
        st.warning("Eccentric circular footing: B′ = D − 2e is used as an approximation; IS 6403 gives no circular rule.")
    G = gt.bearing_capacity(**kw, local_shear=False)
    Lc = gt.bearing_capacity(**kw, local_shear=True)
    R = Lc if w == 1.0 else (G if w == 0.0 else None)
    show = {"General shear": G, "Local shear": Lc}
    cols = st.columns(2)
    for col, (nm, r) in zip(cols, show.items()):
        with col:
            st.markdown(f"**{nm}**")
            st.dataframe(pd.DataFrame([
                ["Effective B′ (m)", r["B_eff"]], ["Effective L′ (m)", r["L_eff"] if shape != "strip" else float("inf")],
                ["c used (kPa)", r["c_used"]], ["φ used (°)", r["phi_used"]],
                ["Nc (–)", r["Nc"]], ["Nq (–)", r["Nq"]], ["Nγ (–)", r["Ng"]],
                ["sc (–)", r["sc"]], ["sq (–)", r["sq"]], ["sγ (–)", r["sg"]],
                ["dc (–)", r["dc"]], ["dq (–)", r["dq"]], ["dγ (–)", r["dg"]],
                ["ic (–)", r["ic"]], ["iq (–)", r["iq"]], ["iγ (–)", r["ig"]],
                ["Water-table factor W′ (–)", r["W"]], ["γ in third term (kN/m³)", r["gamma_used"]],
                ["Effective overburden q at base (kPa)", r["q_overburden"]],
                ["Term 1: c Nc sc dc ic (kPa)", r["term_c"]], ["Term 2: q(Nq−1) sq dq iq (kPa)", r["term_q"]],
                ["Term 3: 0.5 B′ γ Nγ sγ dγ iγ W′ (kPa)", r["term_g"]],
                ["Net ultimate q_nu (kPa)", r["qnu"]], [f"Net safe q_ns = q_nu/{FoS:g} (kPa)", r["qns"]],
                ["Gross safe q_s = q_ns + q (kPa)", r["qs_gross"]]], columns=["Quantity (unit)", "Value"]).round(3)
                .astype({"Value": float}), width='stretch', hide_index=True, height=620)
    qns = (1 - w) * G["qns"] + w * Lc["qns"]
    st.success(f"**Design net safe bearing capacity ({mode.lower()}): {qns:.1f} kPa**")
    st.caption("q_nu is the NET ultimate capacity: q(Nq−1) already removes the overburden q. With W′ the third term uses the "
               "bulk/saturated unit weight, not the buoyant one, otherwise the water table is counted twice.")
    P = st.number_input("Column load P for a check (kN; kN/m for strip)", 0.0, 1e7, 0.0, 50.0)
    if P > 0:
        area = (G["B_eff"] * (G["L_eff"] if shape != "strip" else 1.0)) if shape != "circle" else math.pi / 4 * G["B_eff"] ** 2
        qa = P / area
        st.write(f"Applied pressure on effective area = {qa:.1f} kPa → "
                 + ("**OK**" if qa <= qns else "**EXCEEDS q_ns**") + f" (utilisation {qa/qns:.2f})")
        if shape != "strip" and st.button("Find the smallest B that carries P"):
            ratio = Lf / B
            for Bt in np.arange(0.5, 30, 0.05):
                kk = dict(kw, B=Bt, L=Bt * ratio if shape == "rect" else Bt)
                qq = (1 - w) * gt.bearing_capacity(**kk)["qns"] + w * gt.bearing_capacity(**kk, local_shear=True)["qns"]
                Gx = gt.bearing_capacity(**kk)
                ar = Gx["B_eff"] * Gx["L_eff"] if shape != "circle" else math.pi / 4 * Gx["B_eff"] ** 2
                if P / ar <= qq:
                    st.info(f"B = {Bt:.2f} m (L = {kk['L']:.2f} m), q_ns = {qq:.1f} kPa, applied {P/ar:.1f} kPa"); break
            else:
                st.error("No size up to 30 m is adequate.")

# ------------------------------------------------------------------ P4
with t4:
    st.subheader("P4 · Vertical stress increase")
    b1, b2 = st.columns(2)
    with b1:
        q4 = st.number_input("Uniform net pressure q (kPa)", 0.1, 5000.0, 150.0, 5.0)
        B4 = st.number_input("Width B (m)", 0.1, 200.0, 2.0, 0.1, key="B4")
        L4 = st.number_input("Length L (m)", 0.1, 500.0, 3.0, 0.1, key="L4")
    with b2:
        zmax = st.number_input("Maximum depth below the footing base (m)", 0.5, 200.0, 10.0, 0.5)
        dz = st.number_input("Depth step (m)", 0.05, 5.0, 0.5, 0.05)
        px = st.number_input("Extra point: x from footing centre along B (m)", -100.0, 100.0, 0.0, 0.5)
        py = st.number_input("Extra point: y from footing centre along L (m)", -100.0, 100.0, 0.0, 0.5)
    zs = np.arange(dz, zmax + 1e-9, dz)
    tab = pd.DataFrame({
        "z (m)": zs,
        "Δσz centre (kPa)": [gt.dsig_center(q4, B4, L4, z) for z in zs],
        "Δσz corner (kPa)": [gt.dsig_corner(q4, B4, L4, z) for z in zs],
        "Δσz 2:1 (kPa)": [gt.dsig_2to1(q4, B4, L4, z) for z in zs],
        f"Δσz at ({px:g},{py:g}) m (kPa)": [gt.dsig_rect_point(q4, -B4 / 2, B4 / 2, -L4 / 2, L4 / 2, px, py, z) for z in zs],
        "I corner (–)": [gt.I_corner(B4 / z, L4 / z) for z in zs]})
    f, a = plt.subplots(figsize=(6, 5))
    for k in tab.columns[1:5]:
        a.plot(tab[k], tab["z (m)"], label=k)
    a.invert_yaxis(); a.set_xlabel("Δσz (kPa)"); a.set_ylabel("Depth below base (m)"); a.grid(alpha=.3); a.legend(fontsize=8)
    st.pyplot(f)
    st.dataframe(tab.round(3), width='stretch', hide_index=True)
    st.caption("Centre = 4 × corner of a B/2 × L/2 rectangle; any other point by signed corner superposition; "
               "2:1 method Δσz = qBL/((B+z)(L+z)).")

# ------------------------------------------------------------------ P5
with t5:
    st.subheader("P5 · Settlement")
    d1, d2, d3 = st.columns(3)
    with d1:
        B5 = st.number_input("Footing width B (m)", 0.1, 200.0, 2.0, 0.1, key="B5")
        L5 = st.number_input("Footing length L (m)", 0.1, 500.0, 2.0, 0.1, key="L5")
        Df5 = st.number_input("Founding depth Df (m)", 0.0, 50.0, 1.5, 0.1, key="Df5")
    with d2:
        q5 = st.number_input("Net pressure Δp at founding level (kPa)", 0.1, 5000.0, 100.0, 5.0,
                             help="Gross pressure minus the overburden removed (net pressure).")
        yrs = st.number_input("Time for creep factor C2 (years)", 0.1, 100.0, 5.0, 0.5)
        sub = st.number_input("Clay sublayer thickness (m, ≤ 0.5)", 0.05, 0.5, 0.5, 0.05)
    with d3:
        where = st.selectbox("Stress under the footing for clay layers", ["Centre (Boussinesq)", "Corner (Boussinesq)", "2:1 method"])
        nc = st.checkbox("Treat all clay as normally consolidated (ignore OCR)", value=False)
    Bs, Ls = min(B5, L5), max(B5, L5)
    ss_ = gt.schmertmann(prof, Bs, Ls, Df5, q5, years=yrs)
    fn = {"Centre (Boussinesq)": lambda z: gt.dsig_center(q5, B5, L5, z - Df5),
          "Corner (Boussinesq)": lambda z: gt.dsig_corner(q5, B5, L5, z - Df5),
          "2:1 method": lambda z: gt.dsig_2to1(q5, B5, L5, z - Df5)}[where]
    cs = gt.consolidation(prof, fn, sub=sub, force_nc=nc)
    st.markdown("### Granular layers · strain influence factor method (Schmertmann et al. 1978)")
    m = st.columns(5)
    m[0].metric("Settlement (mm)", f"{ss_['S_mm']:.1f}"); m[1].metric("C1 (–)", f"{ss_['C1']:.3f}")
    m[2].metric("C2 (–)", f"{ss_['C2']:.3f}"); m[3].metric("Iz peak (–)", f"{ss_['Izp']:.3f}")
    m[4].metric("Depth of peak / of zero (m)", f"{ss_['z_peak']:.2f} / {ss_['z_max']:.2f}")
    rs = pd.DataFrame(ss_["rows"])
    if not rs.empty:
        f, a = plt.subplots(figsize=(4.5, 4)); a.plot(rs["Iz"], rs["z"]); a.invert_yaxis(); a.set_xlabel("Iz (–)")
        a.set_ylabel("Depth below base (m)"); a.grid(alpha=.3); st.pyplot(f)
        with st.expander("Slice table (Δz = 0.1 m)"):
            st.dataframe(rs.rename(columns={"z": "z (m)", "E": "Es (kPa)", "strain_term": "Iz Δz/Es (1/kPa)"}), width='stretch', hide_index=True)
    st.caption("S = C1 C2 Δp Σ(Iz/Es)Δz; C1 = 1 − 0.5σ′v0/Δp ≥ 0.5; C2 = 1 + 0.2 log10(t/0.1); Es = 2.5qc (square) to 3.5qc (L/B ≥ 10); "
               "qc = (qc/N)·N from the profile unless an Es is entered. Clay slices are skipped here and handled below.")

    st.markdown("### Clay layers · one-dimensional consolidation")
    st.metric("Consolidation settlement (mm)", f"{cs['S_mm']:.1f}")
    if cs["rows"]:
        cr = pd.DataFrame(cs["rows"]).rename(columns={"layer": "Layer", "z": "z mid (m)", "h": "h (m)", "e0": "e0 (–)", "s0": "σ′v0 (kPa)",
                                                      "sp": "σ′p (kPa)", "ds": "Δσ (kPa)", "state": "State", "S_mm": "S (mm)"})
        st.dataframe(cr.round(3), width='stretch', hide_index=True)
        st.caption("NC: S = Cc h/(1+e0) log10((σ′v0+Δσ)/σ′v0). OC with σ′v0+Δσ ≤ σ′p: Cr in place of Cc. "
                   "OC with σ′v0+Δσ > σ′p: Cr h/(1+e0) log10(σ′p/σ′v0) + Cc h/(1+e0) log10((σ′v0+Δσ)/σ′p). Water table: design.")
    else:
        st.info("No clay layer in the profile.")
    tot = ss_["S_mm"] + cs["S_mm"]
    st.success(f"**Total settlement at the end of consolidation = {ss_['S_mm']:.1f} + {cs['S_mm']:.1f} = {tot:.1f} mm**")

    st.markdown("### Time–settlement")
    clays = [L for L in prof.layers if L.kind == "clay"]
    if clays:
        lo = layer_options(prof)
        names = [k for k, v in lo.items() if v.kind == "clay"]
        sel = st.selectbox("Clay layer", names)
        Ls_ = lo[sel]
        e1, e2, e3 = st.columns(3)
        with e1:
            drain = st.selectbox("Drainage", ["Double (top and bottom)", "Single (top only)", "Single (bottom only)"])
        with e2:
            cv = st.number_input("cv (m²/year)", 0.001, 1000.0, float(Ls_.cv) if Ls_.cv > 0 else 1.0, 0.1)
        with e3:
            H = st.number_input("Layer thickness H (m)", 0.1, 100.0, float(Ls_.bot - Ls_.top), 0.1)
        Hdr = H / 2 if drain.startswith("Double") else H
        Sc = sum(r["S_mm"] for r in cs["rows"] if r["layer"] == Ls_.name)
        ts = np.logspace(-2, 2, 80)
        curve = gt.time_settlement(Sc, cv, H, drain.startswith("Double"), list(ts))
        f, a = plt.subplots(figsize=(7, 4))
        a.semilogx(ts, [ss_["S_mm"] + c[2] for c in curve], label="granular + consolidation")
        a.semilogx(ts, [c[2] for c in curve], "--", label="consolidation of this layer")
        a.invert_yaxis(); a.set_xlabel("Time (years)"); a.set_ylabel("Settlement (mm)"); a.grid(alpha=.3, which="both"); a.legend()
        st.pyplot(f)
        t50 = 0.197 * Hdr ** 2 / cv; t90 = 0.848 * Hdr ** 2 / cv
        st.write(f"H_dr = {Hdr:g} m · consolidation of this layer = {Sc:.1f} mm · t50 = {t50:.2f} years · t90 = {t90:.2f} years (Tv = 0.197, 0.848)")
        st.caption("Tv = cv t / H_dr²; Tv = (π/4)U² for U < 0.6, else Tv = −0.933 log10(1−U) − 0.085. The granular settlement is taken as immediate.")
    else:
        st.info("Add a clay layer in P1 to get a time–settlement curve.")

# ------------------------------------------------------------------ Sources
with t6:
    st.subheader("Equation sources")
    st.table(pd.DataFrame([
        ["Vertical stress, pore pressure", "σv = Σγh, u = γw(z − zw), σ′ = σ − u", "Terzaghi, Peck & Mesri (1996), Ch. 1"],
        ["SPT overburden correction", "CN = 0.77 log10(1920/σ′v), N′ = CN·N", "IS 2131:1981 cl. 5.2.1 / Fig. 2 (chart); closed form Peck et al. (1974) in Das"],
        ["SPT dilatancy correction", "N″ = 15 + 0.5(N′ − 15) for N′ > 15, saturated fine sand", "IS 2131:1981; Terzaghi & Peck (1948)"],
        ["c′, φ′ from direct shear", "τ = c′ + σn tanφ′, least squares", "Mohr–Coulomb; Das, Principles of Foundation Engineering"],
        ["Bearing capacity factors", "Nq = e^(π tanφ) tan²(45+φ/2); Nc = (Nq−1)cotφ; Nγ = 2(Nq+1)tanφ", "IS 6403:1981 cl. 5.1.2 and Table 1"],
        ["Net ultimate capacity", "q_nu = c Nc sc dc ic + q(Nq−1) sq dq iq + 0.5Bγ Nγ sγ dγ iγ W′", "IS 6403:1981 cl. 5.1.2"],
        ["Shape factors", "Square 1.3 / 1.2 / 0.8; rectangular 1+0.2B/L, 1+0.2B/L, 1−0.4B/L; circle 1.3 / 1.2 / 0.6", "IS 6403:1981 Table 2"],
        ["Depth factors", "dc = 1 + 0.2(Df/B)√Nφ; dq = dγ = 1 + 0.1(Df/B)√Nφ for φ > 10°", "IS 6403:1981 cl. 5.1.2.2"],
        ["Inclination factors", "ic = iq = (1 − α/90)²; iγ = (1 − α/φ)²", "IS 6403:1981 cl. 5.1.2.3"],
        ["Water-table factor", "W′ = 0.5 (zw ≤ Df) … 1.0 (zw ≥ Df + B), linear between", "IS 6403:1981 cl. 5.1.2.2"],
        ["Local shear", "c* = (2/3)c, tanφ* = (2/3)tanφ", "Terzaghi (1943); IS 6403:1981 cl. 5.1.2.1"],
        ["Eccentric load", "B′ = B − 2e_B, L′ = L − 2e_L", "IS 6403:1981; Meyerhof (1953)"],
        ["Stress under a corner", "Δσz = q I(m,n); I from Newmark (1935) closed form", "Das, Principles of Foundation Engineering, Boussinesq/Newmark"],
        ["Stress at any point", "Signed sum of corner contributions (superposition)", "Newmark (1935)"],
        ["2:1 method", "Δσz = qBL/((B+z)(L+z))", "Das, approximate stress distribution"],
        ["Granular settlement", "S = C1 C2 Δp Σ(Iz/Es)Δz; C1, C2, Izp, Es", "Schmertmann, Hartman & Brown (1978), J. Geotech. Eng. Div. ASCE 104(GT8)"],
        ["Clay consolidation", "NC / OC settlement with Cc, Cr, σ′p", "Terzaghi (1925); Das; Terzaghi, Peck & Mesri (1996)"],
        ["Time–settlement", "Tv = cv t/Hdr², U–Tv relations", "Terzaghi 1-D consolidation; Das"]],
        columns=["Item", "Equation", "Source"]))
    st.caption("Some code-table values were checked against secondary sources only; verify them against your copy of the standard before submission.")
