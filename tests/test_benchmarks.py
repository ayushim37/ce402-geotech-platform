import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
"""Benchmarks against published values / hand calculations. Run: python test_benchmarks.py"""
import math
from geotech import *

def close(a, b, tol, label):
    ok = abs(a - b) <= tol
    print(("PASS " if ok else "FAIL ") + f"{label}: got {a:.4f}, expected {b:.4f} (tol {tol})")
    assert ok, label

# IS 6403 Table 1 bearing capacity factors
for phi, nc, nq, ng in [(0, 5.14, 1.0, 0.0), (20, 14.83, 6.40, 5.39), (30, 30.14, 18.40, 22.40),
                        (35, 46.12, 33.30, 48.03), (40, 75.31, 64.20, 109.41)]:
    a, b, c = bc_factors(phi)
    close(a, nc, 0.05, f"Nc phi={phi}"); close(b, nq, 0.05, f"Nq phi={phi}"); close(c, ng, 0.05, f"Ng phi={phi}")

# Boussinesq: square corner m=n=1 -> 0.1752 ; centre of square at z=B -> 0.336 (Das table)
close(I_corner(1, 1), 0.1752, 0.0002, "I_corner(1,1)")
close(dsig_center(100, 2, 2, 2) / 100, 0.336, 0.001, "centre square z=B")
# deep-branch check: m=n=5 (m2n2 > m2+n2+1) must stay positive and ~0.2486
close(I_corner(5, 5), 0.2486, 0.002, "I_corner(5,5) branch")
# point outside rectangle by superposition = corner formula
close(dsig_rect_point(100, 0, 2, 0, 3, 0, 0, 2.0), dsig_corner(100, 2, 3, 2.0), 1e-9, "superposition corner")
# point at centre by general routine equals 4-corner centre formula
close(dsig_rect_point(100, -1, 1, -1.5, 1.5, 0, 0, 2.0), dsig_center(100, 2, 3, 2.0), 1e-9, "superposition centre")
# 2:1 at z=0 returns q
close(dsig_2to1(100, 2, 2, 1e-9), 100, 1e-3, "2:1 at z=0")

# Simple hand calculation, dry c=0 sand, phi=30, gamma=18, B=2 square, Df=1, WT deep.
p = Profile([Layer(0, 20, kind="sand", gamma=18, gamma_sat=20, phi=30)], wt_measured=50, wt_design=50)
r = bearing_capacity(p, 2, 2, "square", 1.0, 0, 30)
# hand: q=18 ; Nq=18.4 Ng=22.4 ; sq=1.2 sg=0.8 ; dq=dg=1+0.1*(0.5)*sqrt(3)=1.0866 ; W'=1
hand = 18 * 17.4 * 1.2 * 1.0866 + 0.5 * 2 * 18 * 22.4 * 0.8 * 1.0866
close(r["qnu"], hand, 0.5, "qnu hand check (dry sand)")
# water table at base: W'=0.5 ; WT at ground: W'=0.5 ; far below: 1
close(water_table_factor(1.0, 1.0, 2.0), 0.5, 1e-9, "W' WT at base")
close(water_table_factor(2.0, 1.0, 2.0), 0.75, 1e-9, "W' WT B/2 below base")
close(water_table_factor(9.0, 1.0, 2.0), 1.0, 1e-9, "W' deep WT")

# consolidation hand check: single clay 2 m thick (4 sublayers), NC, Cc=0.3 e0=1.0, s0 from profile, uniform ds
cl = Profile([Layer(0, 5, kind="clay", gamma=18, gamma_sat=18, Cc=0.3, Cr=0.05, e0_top=1.0, e0_bot=1.0)], 50, 50)
res = consolidation(cl, lambda z: 50.0)
hand = sum(0.3 / 2.0 * 0.5 * math.log10((18 * (0.25 + 0.5 * i) + 50) / (18 * (0.25 + 0.5 * i))) for i in range(10))
close(res["S_mm"] / 1000, hand, 1e-6, "consolidation NC hand check")

# Terzaghi time factor: U=50% -> Tv=0.196 ; U=90% -> 0.848
close(Tv_from_U(0.5), 0.196, 0.001, "Tv(50%)"); close(Tv_from_U(0.9), 0.848, 0.002, "Tv(90%)")
# least squares exact line tau = 10 + sigma tan(30)
c, ph, r2 = fit_cphi([50, 100, 150], [10 + 50 * math.tan(math.radians(30)), 10 + 100 * math.tan(math.radians(30)),
                                       10 + 150 * math.tan(math.radians(30))])
close(c, 10, 1e-6, "fit c"); close(ph, 30, 1e-6, "fit phi")
print("ALL BENCHMARKS PASSED")

# ---- INDEPENDENT HAND CALCULATION (done separately from the engine, see report) ----
# BH-1 interior footing: B = 3.10 m square, Df = 1.5 m, design WT 1.0 m. Fill 0-0.6 (16.5), SM 0.6-3.4 (18/20), SP 3.4-5.8 (19/20.4)
# q = 0.6*16.5 + 0.4*18 + 0.5*20 - 0.5*9.81 = 22.195 kPa ; W' = 0.5 ; gamma(3rd term) = (1.9*20 + 1.2*20.4)/3.1 = 20.155
# general (phi=30): q_nu = 805.8 -> q_ns = 268.6 ; local (phi*=21.05): q_nu = 257.3 -> q_ns = 85.8
p1 = Profile([Layer(0, 0.6, kind="fill", gamma=16.5, gamma_sat=18, phi=25),
              Layer(0.6, 3.4, kind="sand", gamma=18, gamma_sat=20, phi=30),
              Layer(3.4, 5.8, kind="sand", gamma=19, gamma_sat=20.4, phi=34)], wt_measured=2.6, wt_design=1.0)
g = bearing_capacity(p1, 3.1, 3.1, "square", 1.5, 0, 30)
l = bearing_capacity(p1, 3.1, 3.1, "square", 1.5, 0, 30, local_shear=True)
close(g["q_overburden"], 22.195, 0.01, "hand: effective overburden q")
close(g["gamma_used"], 20.155, 0.02, "hand: gamma in third term")
close(g["qns"], 268.6, 0.6, "hand: general-shear q_ns")
close(l["qns"], 85.8, 0.4, "hand: local-shear q_ns")
print("HAND-CALC BENCHMARKS PASSED")
