# usb_s1223_camber_based_mm_fixed.py
"""
USB solver using CAMBER LINE for influence matrices and full airfoil only for visualization.
All geometry internal units: mm (per your Option 2). Physics (lift, m_dot) use SI where needed.
Panel counts: Na=100 (camber panels), Njt=100, Njb=100.

CHANGES made in this patched version:
1) Parameterised air densities (rho_out, rho_jet) as user inputs in the GUI and passed through solver.
2) Fixed a bug that used V_jet instead of V_out when forming the free-stream velocity vector for camber BCs
   (this caused increasing alpha to sometimes *decrease* computed circulation). Now V_inf_vec_mm uses V_out.
3) compute_CLs_from_Gamma_mm now accepts rho_out rather than relying on a hardcoded value.
4) All places that previously hard-coded 1.225 now read the rho variables from GUI inputs.
5) Minor cleanups to avoid repeated recomputation of camber when finding hinge_y.
"""

import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# ---------------- user geometry constants (mm)
CHORD_MM = 175.0
HINGE_X_MM = 122.5
NOZZLE_X_FRAC = 0.18764
JET_TOP_Z_MM = 109.6705
JET_BOT_Z_MM = 25.6705
JET_THICKNESS_MM = 86.0
JET_END_X_FACTOR = 2.5
N_A = 100
N_JT = 100
N_JB = 100

# ---------------- load normalized S1223 dataset (the surface coordinates you provided)
def load_s1223_norm():
    data = """1.00000 0.00000
0.99838 0.00126
0.99417 0.00494
0.98825 0.01037
0.98075 0.01646
0.97111 0.02250
0.95884 0.02853
0.94389 0.03476
0.92639 0.04116
0.90641 0.04768
0.88406 0.05427
0.85947 0.06089
0.83277 0.06749
0.80412 0.07402
0.77369 0.08044
0.74166 0.08671
0.70823 0.09277
0.67360 0.09859
0.63798 0.10412
0.60158 0.10935
0.56465 0.11425
0.52744 0.11881
0.49025 0.12303
0.45340 0.12683
0.41721 0.13011
0.38193 0.13271
0.34777 0.13447
0.31488 0.13526
0.28347 0.13505
0.25370 0.13346
0.22541 0.13037
0.19846 0.12594
0.17286 0.12026
0.14863 0.11355
0.12591 0.10598
0.10482 0.09770
0.08545 0.08879
0.06789 0.07940
0.05223 0.06965
0.03855 0.05968
0.02694 0.04966
0.01755 0.03961
0.01028 0.02954
0.00495 0.01969
0.00155 0.01033
0.00005 0.00178
0.00044 -0.00561
0.00264 -0.01120
0.00789 -0.01427
0.01718 -0.01550
0.03006 -0.01584
0.04627 -0.01532
0.06561 -0.01404
0.08787 -0.01202
0.11282 -0.00925
0.14020 -0.00563
0.17006 -0.00075
0.20278 0.00535
0.23840 0.01213
0.27673 0.01928
0.31750 0.02652
0.36044 0.03358
0.40519 0.04021
0.45139 0.04618
0.49860 0.05129
0.54639 0.05534
0.59428 0.05820
0.64176 0.05976
0.68832 0.05994
0.73344 0.05872
0.77660 0.05612
0.81729 0.05219
0.85500 0.04706
0.88928 0.04088
0.91966 0.03387
0.94573 0.02624
0.96693 0.01822
0.98255 0.01060
0.99268 0.00468
0.99825 0.00115
1.00000 0.00000"""
    pts=[]
    for ln in data.splitlines():
        s=ln.strip()
        if not s: continue
        a,b = s.split()
        pts.append([float(a), float(b)])
    return np.array(pts)

# ---------------- basic transforms in mm
def rotate_points_mm(points, angle_deg, center=(0.0,0.0)):
    rad = np.radians(angle_deg)
    c,s = np.cos(rad), np.sin(rad)
    pts = np.array(points, dtype=float)
    ox,oy = center
    px = pts[:,0] - ox
    py = pts[:,1] - oy
    qx = ox + c*px - s*py
    qy = oy + s*px + c*py
    return np.column_stack((qx,qy))

def discretize_mm(points):
    panels=[]
    for i in range(len(points)-1):
        p1 = points[i]; p2 = points[i+1]
        vec = p2 - p1
        L = np.linalg.norm(vec)
        if L < 1e-12: continue
        t = vec / L
        n = np.array([-t[1], t[0]])
        mid = p1 + 0.5*vec
        panels.append({'p1':p1, 'p2':p2, 'center':mid, 'tangent':t, 'normal':n, 'length':L})
    return panels

def resample_by_arclength(pts, Npan):
    # pts: Nx2 array ordered along surface; returns Npan+1 points
    d = np.sqrt(np.sum(np.diff(pts,axis=0)**2,axis=1))
    s = np.concatenate(([0.0], np.cumsum(d)))
    s_norm = s / s[-1]
    xi = np.linspace(0.0,1.0,Npan+1)
    fx = interp1d(s_norm, pts[:,0], kind='linear')
    fz = interp1d(s_norm, pts[:,1], kind='linear')
    return np.column_stack((fx(xi), fz(xi)))

# ---------------- compute camber line from surface coordinates (normalized -> mm)
def compute_camber_mm(raw_norm, chord_mm, Ncamber_pts):
    pts_mm = np.zeros_like(raw_norm)
    pts_mm[:,0] = raw_norm[:,0] * chord_mm
    pts_mm[:,1] = raw_norm[:,1] * chord_mm
    idx_le = np.argmin(pts_mm[:,0])
    upper = pts_mm[:idx_le+1]
    lower = pts_mm[idx_le:]
    if upper[0,0] < upper[-1,0]:
        upper = upper[::-1]
    if lower[0,0] > lower[-1,0]:
        lower = lower[::-1]
    x_up = upper[:,0]; y_up = upper[:,1]
    x_lo = lower[:,0]; y_lo = lower[:,1]
    x_common = np.linspace(0.0, chord_mm, Ncamber_pts+1)
    f_up = interp1d(x_up, y_up, bounds_error=False, fill_value="extrapolate")
    f_lo = interp1d(x_lo, y_lo, bounds_error=False, fill_value="extrapolate")
    y_up_s = f_up(x_common)
    y_lo_s = f_lo(x_common)
    y_camber = 0.5*(y_up_s + y_lo_s)
    camber_pts = np.column_stack((x_common, y_camber))
    return camber_pts  # (Ncamber_pts+1 x 2)

# ---------------- panel induction formula (2D straight panel with vortex+source) in mm units
def induced_panel_velocity_mm(target, p1, p2, gamma, sigma):
    target = np.array(target); p1 = np.array(p1); p2 = np.array(p2)
    vec = p2 - p1
    L = np.linalg.norm(vec)
    if L < 1e-12:
        return np.array([0.0,0.0])
    t = vec / L
    n = np.array([-t[1], t[0]])
    d = target - p1
    xloc = np.dot(d, t); yloc = np.dot(d, n)
    r1sq = xloc**2 + yloc**2
    r2sq = (xloc - L)**2 + yloc**2
    if r1sq < 1e-12 or r2sq < 1e-12:
        # regularized local approx
        u_loc = 0.0
        v_loc = 0.5 * sigma
    else:
        theta1 = np.arctan2(yloc, xloc)
        theta2 = np.arctan2(yloc, xloc - L)
        val_log = 0.5 * np.log(r2sq / r1sq)
        val_atan = theta2 - theta1
        u_src = (sigma / (2.0*np.pi)) * val_log
        v_src = (sigma / (2.0*np.pi)) * val_atan
        u_vtx = -(gamma / (2.0*np.pi)) * val_atan
        v_vtx = (gamma / (2.0*np.pi)) * val_log
        u_loc = u_src + u_vtx
        v_loc = v_src + v_vtx
    # rotate back
    u = u_loc * t[0] - v_loc * t[1]
    w = u_loc * t[1] + v_loc * t[0]
    return np.array([u,w])

# ---------------- build full geometry (camber panels for matrices + airfoil surface for plotting)
def build_geometry_cam_airfoil_mm(chord_mm, alpha_deg, delta_j_deg, res_factor=1.0):
    raw_norm = load_s1223_norm()
    camber_pts = compute_camber_mm(raw_norm, chord_mm, N_A)  # returns N_A+1 points
    # compute hinge y once
    # find index closest to hinge_x
    hinge_idx = np.argmin(np.abs(camber_pts[:,0] - HINGE_X_MM))
    hinge_y = camber_pts[hinge_idx,1]
    camber_pts_rot = []
    for p in camber_pts:
        if p[0] > HINGE_X_MM:
            pr = rotate_points_mm(np.array([p]), delta_j_deg, center=(HINGE_X_MM, hinge_y))[0]
            camber_pts_rot.append(pr)
        else:
            camber_pts_rot.append(p)
    camber_pts_rot = np.array(camber_pts_rot)
    camber_pts_final = rotate_points_mm(camber_pts_rot, -alpha_deg)
    camber_panels = discretize_mm(camber_pts_final)

    # full airfoil surface for plotting
    raw_mm = np.zeros_like(raw_norm)
    raw_mm[:,0] = raw_norm[:,0] * chord_mm
    raw_mm[:,1] = raw_norm[:,1] * chord_mm
    af_pts = []
    for row in raw_mm:
        if row[0] > HINGE_X_MM:
            af_pts.append(rotate_points_mm(np.array([row]), delta_j_deg, center=(HINGE_X_MM, hinge_y))[0])
        else:
            af_pts.append(row)
    af_pts = np.array(af_pts)
    af_pts_rot = rotate_points_mm(af_pts, -alpha_deg)
    af_plot = resample_by_arclength(af_pts_rot, 200)

    # jet sheets
    nozzle_x = NOZZLE_X_FRAC * chord_mm
    n_up = max(10, int(40*res_factor))
    n_down = max(10, int(50*res_factor))
    x_up = np.linspace(nozzle_x, HINGE_X_MM, n_up)
    top_up = np.column_stack((x_up, np.full_like(x_up, JET_TOP_Z_MM)))
    jet_end_x_mm = JET_END_X_FACTOR * chord_mm
    x_down = np.linspace(HINGE_X_MM, jet_end_x_mm, n_down)
    top_down = np.column_stack((x_down, np.full_like(x_down, JET_TOP_Z_MM)))
    pivot_top = (HINGE_X_MM, JET_TOP_Z_MM)
    top_down_rot = rotate_points_mm(top_down, delta_j_deg, center=pivot_top)
    jet_top_pts = np.vstack((top_up, top_down_rot))
    bottom_up = np.column_stack((x_up, np.full_like(x_up, JET_TOP_Z_MM - JET_THICKNESS_MM)))
    bottom_down = np.column_stack((x_down, np.full_like(x_down, JET_TOP_Z_MM - JET_THICKNESS_MM)))
    pivot_bot = (HINGE_X_MM, JET_TOP_Z_MM - JET_THICKNESS_MM)
    bottom_down_rot = rotate_points_mm(bottom_down, delta_j_deg, center=pivot_bot)
    jet_bot_pts = np.vstack((bottom_up, bottom_down_rot))
    jet_top_pts = rotate_points_mm(jet_top_pts, -alpha_deg)
    jet_bot_pts = rotate_points_mm(jet_bot_pts, -alpha_deg)
    jt_res = resample_by_arclength(jet_top_pts, N_JT)
    jb_res = resample_by_arclength(jet_bot_pts, N_JB)
    jt_panels = discretize_mm(jt_res)
    jb_panels = discretize_mm(jb_res)

    return camber_panels, af_plot, jt_panels, jb_panels

# ---------------- assemble full matrix using camber_panels for airfoil unknowns
def assemble_and_solve_camber(af_camber_panels, jt_panels, jb_panels, V_out, V_jet, rho_out, rho_jet, t_j_mm, reg=1e-8):
    Na = len(af_camber_panels); Nj = len(jt_panels); Njb = len(jb_panels)
    N_unknowns = Na + Nj + Nj + Njb + Njb
    N_eq = Na + 2*Nj + 2*Njb + 1
    A = np.zeros((N_eq, N_unknowns))
    b = np.zeros(N_eq)
    idx_ga = 0
    idx_gjt = idx_ga + Na
    idx_sjt = idx_gjt + Nj
    idx_gjb = idx_sjt + Nj
    idx_sjb = idx_gjb + Njb

    if abs(V_jet) < 1e-8: V_jet = 1e-8
    mu = V_out / V_jet
    K_p = (rho_out * V_out) / (rho_jet * V_jet)
    # BUG FIX: use V_out for the free-stream velocity vector (previous code used V_jet here)
    V_inf_vec_mm = np.array([V_out * 1000.0, 0.0])  # mm/s (V in m/s -> mm/s)

    # Airfoil (camber) normal BCs
    for i in range(Na):
        Pi = af_camber_panels[i]
        b[i] = -np.dot(V_inf_vec_mm, Pi['normal'])
        # influences from camber (ga)
        for j in range(Na):
            Pj = af_camber_panels[j]
            u = induced_panel_velocity_mm(Pi['center'], Pj['p1'], Pj['p2'], gamma=1.0, sigma=0.0)
            A[i, idx_ga + j] = np.dot(u, Pi['normal'])
        # influences from jets
        for j in range(Nj):
            Pj = jt_panels[j]
            u_g = induced_panel_velocity_mm(Pi['center'], Pj['p1'], Pj['p2'], gamma=1.0, sigma=0.0)
            u_s = induced_panel_velocity_mm(Pi['center'], Pj['p1'], Pj['p2'], gamma=0.0, sigma=1.0)
            A[i, idx_gjt + j] = np.dot(u_g, Pi['normal'])
            A[i, idx_sjt + j] = np.dot(u_s, Pi['normal'])
        for j in range(Njb):
            Pj = jb_panels[j]
            u_g = induced_panel_velocity_mm(Pi['center'], Pj['p1'], Pj['p2'], gamma=1.0, sigma=0.0)
            u_s = induced_panel_velocity_mm(Pi['center'], Pj['p1'], Pj['p2'], gamma=0.0, sigma=1.0)
            A[i, idx_gjb + j] = np.dot(u_g, Pi['normal'])
            A[i, idx_sjb + j] = np.dot(u_s, Pi['normal'])

    # Jet BCs (top & bottom)
    row = Na
    for panels, gj_idx, sj_idx, Nj_curr in [(jt_panels, idx_gjt, idx_sjt, Nj), (jb_panels, idx_gjb, idx_sjb, Njb)]:
        for i in range(Nj_curr):
            Pi = panels[i]
            A[row, sj_idx + i] += 1.0
            factor = (1.0 - mu)
            b[row] = - factor * np.dot(V_inf_vec_mm, Pi['normal'])
            # influences
            for j in range(Na):
                Pj = af_camber_panels[j]
                u = induced_panel_velocity_mm(Pi['center'], Pj['p1'], Pj['p2'], gamma=1.0, sigma=0.0)
                A[row, idx_ga + j] += factor * np.dot(u, Pi['normal'])
            for j in range(Nj):
                Pj = jt_panels[j]
                u_g = induced_panel_velocity_mm(Pi['center'], Pj['p1'], Pj['p2'], gamma=1.0, sigma=0.0)
                u_s = induced_panel_velocity_mm(Pi['center'], Pj['p1'], Pj['p2'], gamma=0.0, sigma=1.0)
                A[row, idx_gjt + j] += factor * np.dot(u_g, Pi['normal'])
                A[row, idx_sjt + j] += factor * np.dot(u_s, Pi['normal'])
            for j in range(Njb):
                Pj = jb_panels[j]
                u_g = induced_panel_velocity_mm(Pi['center'], Pj['p1'], Pj['p2'], gamma=1.0, sigma=0.0)
                u_s = induced_panel_velocity_mm(Pi['center'], Pj['p1'], Pj['p2'], gamma=0.0, sigma=1.0)
                A[row, idx_gjb + j] += factor * np.dot(u_g, Pi['normal'])
                A[row, idx_sjb + j] += factor * np.dot(u_s, Pi['normal'])
            row += 1

            # tangential BC
            A[row, gj_idx + i] += 1.0
            factor_tan = (1.0 - K_p)
            b[row] = - factor_tan * np.dot(V_inf_vec_mm, Pi['tangent'])
            for j in range(Na):
                Pj = af_camber_panels[j]
                u = induced_panel_velocity_mm(Pi['center'], Pj['p1'], Pj['p2'], gamma=1.0, sigma=0.0)
                A[row, idx_ga + j] += factor_tan * np.dot(u, Pi['tangent'])
            for j in range(Nj):
                Pj = jt_panels[j]
                u_g = induced_panel_velocity_mm(Pi['center'], Pj['p1'], Pj['p2'], gamma=1.0, sigma=0.0)
                u_s = induced_panel_velocity_mm(Pi['center'], Pj['p1'], Pj['p2'], gamma=0.0, sigma=1.0)
                A[row, idx_gjt + j] += factor_tan * np.dot(u_g, Pi['tangent'])
                A[row, idx_sjt + j] += factor_tan * np.dot(u_s, Pi['tangent'])
            for j in range(Njb):
                Pj = jb_panels[j]
                u_g = induced_panel_velocity_mm(Pi['center'], Pj['p1'], Pj['p2'], gamma=1.0, sigma=0.0)
                u_s = induced_panel_velocity_mm(Pi['center'], Pj['p1'], Pj['p2'], gamma=0.0, sigma=1.0)
                A[row, idx_gjb + j] += factor_tan * np.dot(u_g, Pi['tangent'])
                A[row, idx_sjb + j] += factor_tan * np.dot(u_s, Pi['tangent'])
            row += 1

    # Kutta at TE (camber endpoints)
    A[row, idx_ga + 0] = 1.0
    A[row, idx_ga + (Na - 1)] = 1.0
    b[row] = 0.0
    row += 1
    assert row == N_eq

    # regularize and solve via augmented least squares
    A_aug = np.vstack((A, np.sqrt(reg) * np.eye(N_unknowns)))
    b_aug = np.concatenate((b, np.zeros(N_unknowns)))
    sol, *_ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
    gamma_air = sol[0:Na]
    # compute Gamma integral (note: gamma units are mm/s per induction formulation; integrate multiply by panel length mm -> mm^2/s)
    Gamma_mm2s = 0.0
    for i in range(Na):
        Gamma_mm2s +=( gamma_air[i] * af_camber_panels[i]['length'])
    return Gamma_mm2s, sol

# ---------------- convert gamma->Cl and jet reaction
def compute_CLs_from_Gamma_mm(Gamma_mm2s, V_out, V_jet, t_j_mm, chord_mm, rho_out=1.225):
    Gamma_m2s = Gamma_mm2s / 1e6
    chord_m = chord_mm / 1000.0
    Lp = rho_out * V_out * Gamma_m2s
    qinf = 0.5 * rho_out * V_out**2
    Cl_circ = Lp / (qinf * chord_m)
    return Cl_circ

# ---------------- GUI driver
class USBCamberApp:
    def __init__(self, root):
        self.root = root
        root.title("USB S1223 (Camber-based matrices) - mm units (patched)")
        root.geometry("1250x760")
        self.var_alpha = tk.DoubleVar(value=0.0)
        self.var_delta = tk.DoubleVar(value=0.0)
        self.var_Vj = tk.DoubleVar(value=40.0)
        self.var_V0 = tk.DoubleVar(value=40.0)
        self.var_res = tk.DoubleVar(value=1.0)
        self.var_rho_out = tk.DoubleVar(value=1.225)
        self.var_rho_jet = tk.DoubleVar(value=1.225)

        left = ttk.Frame(root, padding=8); left.pack(side=tk.LEFT, fill=tk.Y)
        right = ttk.Frame(root, padding=8); right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        ttk.Label(left, text="Inputs", font=("Arial",14,"bold")).pack(pady=6)
        self._add_input(left, "alpha (deg):", self.var_alpha)
        self._add_input(left, "delta_j (deg):", self.var_delta)
        self._add_input(left, "V_jet (m/s):", self.var_Vj)
        self._add_input(left, "V_out (m/s):", self.var_V0)
        self._add_input(left, "rho_out (kg/m^3):", self.var_rho_out)
        self._add_input(left, "rho_jet (kg/m^3):", self.var_rho_jet)
        ttk.Label(left, text="res factor").pack()
        tk.Scale(left, from_=0.5, to=2.0, resolution=0.1, orient=tk.HORIZONTAL, variable=self.var_res).pack(fill=tk.X)
        ttk.Button(left, text="Run", command=self.run).pack(fill=tk.X,pady=6)
        ttk.Button(left, text="Sweep CL vs alpha", command=self.sweep).pack(fill=tk.X,pady=6)
        self.txt = tk.StringVar(value="Run simulation")
        ttk.Label(left, textvariable=self.txt, font=("Consolas",10)).pack(pady=8)

        self.fig = Figure(figsize=(8,6), dpi=110); self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right); self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.run()

    def _add_input(self, parent, label, var):
        f = ttk.Frame(parent); f.pack(fill=tk.X,pady=4)
        ttk.Label(f, text=label).pack(anchor=tk.W); ttk.Entry(f, textvariable=var).pack(fill=tk.X)

    def run(self):
        try:
            alpha = float(self.var_alpha.get()); delta = float(self.var_delta.get())
            Vj = float(self.var_Vj.get()); V0 = float(self.var_V0.get()); rf = float(self.var_res.get())
            rho_out = float(self.var_rho_out.get()); rho_jet = float(self.var_rho_jet.get())
        except Exception as e:
            self.txt.set("Invalid input: "+str(e)); return
        chord_mm = CHORD_MM
        camber_panels, af_plot_pts, jt_panels, jb_panels = build_geometry_cam_airfoil_mm(chord_mm, alpha, delta, res_factor=rf)
        Gamma_mm2s, sol = assemble_and_solve_camber(camber_panels, jt_panels, jb_panels, V0, Vj, rho_out, rho_jet, JET_THICKNESS_MM, reg=1e-8)
        Cl_circ = compute_CLs_from_Gamma_mm(Gamma_mm2s, V0, Vj, JET_THICKNESS_MM, chord_mm, rho_out=rho_out)
        # jet reaction lift:
        t_j_m = JET_THICKNESS_MM / 1000.0
        m_dot = rho_jet * t_j_m * Vj * 1.0
        theta = np.radians(alpha + abs(delta))
        L_jet = m_dot * Vj * np.sin(theta)
        qinf = 0.5 * rho_out * V0**2
        chord_m = chord_mm / 1000.0
        Cl_jet = L_jet / (qinf * chord_m)
        Cl_total = Cl_circ + Cl_jet
        self.txt.set(f"Gamma (mm^2/s): {Gamma_mm2s:.6e}\nCl_circ: {Cl_circ:.4f}\nCl_jet: {Cl_jet:.4f}\nCl_total: {Cl_total:.4f}")
        # plot full airfoil + jets for viz
        self.ax.clear()
        af = af_plot_pts
        jt = np.array([p['p1'] for p in jt_panels] + [jt_panels[-1]['p2']])
        jb = np.array([p['p1'] for p in jb_panels] + [jb_panels[-1]['p2']])
        self.ax.plot(af[:,0], af[:,1], 'k-', lw=1.6, label='Airfoil (surface, viz only)')
        self.ax.plot(jt[:,0], jt[:,1], 'r--', label='Jet top')
        self.ax.plot(jb[:,0], jb[:,1], 'b--', label='Jet bottom')
        self.ax.axvline(HINGE_X_MM, color='g', linestyle=':', label='Hinge x=122.5 mm')
        self.ax.set_xlabel('x (mm)'); self.ax.set_ylabel('z (mm)')
        self.ax.set_title(f"S1223 USB geometry (Cl_total={Cl_total:.3f})")
        self.ax.set_aspect('equal','box'); self.ax.grid(True); self.ax.legend()
        self.canvas.draw()

    def sweep(self):
        try:
            delta = float(self.var_delta.get()); Vj = float(self.var_Vj.get()); V0 = float(self.var_V0.get()); rf = float(self.var_res.get())
            rho_out = float(self.var_rho_out.get()); rho_jet = float(self.var_rho_jet.get())
        except Exception as e:
            self.txt.set("Invalid input: "+str(e)); return
        alphas = np.linspace(-6.0,16.0,23)
        Cl_tot = []; Cl_c = []; Cl_j = []
        for a in alphas:
            camber_panels, af_plot_pts, jt_panels, jb_panels = build_geometry_cam_airfoil_mm(CHORD_MM, a, delta, res_factor=rf)
            Gamma_mm2s, sol = assemble_and_solve_camber(camber_panels, jt_panels, jb_panels, V0, Vj, rho_out, rho_jet, JET_THICKNESS_MM, reg=1e-8)
            Clc = compute_CLs_from_Gamma_mm(Gamma_mm2s, V0, Vj, JET_THICKNESS_MM, CHORD_MM, rho_out=rho_out)
            t_j_m = JET_THICKNESS_MM / 1000.0
            m_dot = rho_jet * t_j_m * Vj * 1.0
            theta = np.radians(a + abs(delta))
            L_jet = m_dot * Vj * np.sin(theta)
            qinf = 0.5 * rho_out * V0**2
            Clj = L_jet / (qinf * (CHORD_MM/1000.0))
            Cl_tot.append(Clc+Clj); Cl_c.append(Clc); Cl_j.append(Clj)
        fig, ax = plt.subplots(figsize=(7,5))
        ax.plot(alphas, Cl_tot, '-o', label='CL_total')
        ax.plot(alphas, Cl_c, '-s', label='CL_circ')
        ax.plot(alphas, Cl_j, '-^', label='CL_jet')
        ax.set_xlabel('alpha (deg)'); ax.set_ylabel('CL'); ax.grid(True); ax.legend(); ax.set_title('S1223: CL vs alpha (camber-based matrices)')
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = USBCamberApp(root)
    root.mainloop()
