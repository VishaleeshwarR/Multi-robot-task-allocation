"""
Multi-Robot Task Allocation Experiments
Based on Park, Zhong, Leonard (ICRA 2021)

Implements and runs all 4 experiments from the paper using the same
mathematical model as the Webots simulation:
  Eq. (2): Resource dynamics  dq_i/dt = -F_i(q_i, x_i) + w_i
  Eq. (3): Consumption rate   F_i = R_i * tanh(alpha_i*q_i/2) * x_i^beta_i
  Eq. (6): Payoff mechanism   p_i = q_i + nu*(-F_i(gamma*, x_i) + w_i)
  Eq. (7): Revision protocol  rho * [p_j - p_i]_+
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys

# ─── Model parameters (match Webots controllers exactly) ─────────────────────
NUM_PATCHES  = 4
NUM_ROBOTS   = 40
R_BASE       = np.array([3.44, 3.44, 3.44, 3.44])
ALPHA        = np.array([0.036, 0.036, 0.036, 0.036])
BETA         = np.array([0.91,  0.91,  0.91,  0.91])
W_BASE       = np.array([0.5,   0.5,   0.5,   0.5])
RHO          = 1.0 / 600.0
NU_DEFAULT   = 0

OUTPUT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PATCH_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
PATCH_LABELS = ["Area 1", "Area 2", "Area 3", "Area 4"]

# ─── Core equations ──────────────────────────────────────────────────────────

def F(q, x, R=R_BASE):
    q = np.maximum(q, 0.0)
    x = np.maximum(x, 0.0)
    sat = np.tanh(ALPHA * q / 2.0)
    return R * sat * np.power(x, BETA)


def payoff(q, x, nu, w):
    if nu == 0:
        return q.copy()
    gamma_star = np.max(q)
    F_star = F(np.full(NUM_PATCHES, gamma_star), x)
    return q + nu * (-F_star + w)


def dx_dt(x, p, rho=RHO):
    n = len(x)
    out = np.zeros(n)
    for i in range(n):
        inflow  = sum(x[j] * rho * max(0.0, p[i] - p[j]) for j in range(n) if j != i)
        outflow = x[i] * sum(rho * max(0.0, p[j] - p[i]) for j in range(n) if j != i)
        out[i]  = inflow - outflow
    return out


def make_ode(w_fn, n_fn=None, nu=NU_DEFAULT):
    def ode(t, y):
        q    = np.maximum(y[:NUM_PATCHES], 0.0)
        xraw = y[NUM_PATCHES:]
        s    = np.sum(xraw)
        x    = xraw / s if s > 1e-10 else np.full(NUM_PATCHES, 1.0 / NUM_PATCHES)
        n_eff = n_fn(t) if n_fn else NUM_ROBOTS
        scale = n_eff / NUM_ROBOTS
        w     = w_fn(t)
        Fi    = F(q, x) * scale
        dqdt  = -Fi + w
        p     = payoff(q, x, nu, w)
        dxdt_ = dx_dt(x, p) * n_eff
        return np.concatenate([dqdt, dxdt_])
    return ode


def run(duration, w_fn, n_fn=None, nu=0, q0=None, x0=None):
    if q0 is None:
        q0 = np.zeros(NUM_PATCHES)
    if x0 is None:
        x0 = np.full(NUM_PATCHES, 1.0 / NUM_PATCHES)
    y0  = np.concatenate([q0, x0])
    sol = solve_ivp(make_ode(w_fn, n_fn, nu), [0, duration], y0,
                    method="RK45", max_step=1.0, rtol=1e-5, atol=1e-7)
    t   = sol.t
    q   = np.maximum(sol.y[:NUM_PATCHES], 0.0)
    xr  = sol.y[NUM_PATCHES:]
    s   = np.sum(xr, axis=0, keepdims=True)
    x   = np.where(s > 1e-10, xr / s, np.full_like(xr, 0.25))
    return t, q, x


# ─── Utilities ───────────────────────────────────────────────────────────────

def snap(t, q, x, t_target, label=""):
    idx = np.argmin(np.abs(t - t_target))
    alloc = x[:, idx] * 100
    print(f"\n  {label}  (actual t={t[idx]:.1f}s):")
    print(f"  {'Area':<10} {'Resource q_i':>14} {'Allocation':>12}")
    print(f"  {'-'*40}")
    for i in range(NUM_PATCHES):
        print(f"  {PATCH_LABELS[i]:<10} {q[i, idx]:>14.3f} {alloc[i]:>11.1f}%")
    return alloc


def plot_qx(ax_q, ax_x, t, q, x, evts=None):
    for i in range(NUM_PATCHES):
        ax_q.plot(t, q[i], label=PATCH_LABELS[i], color=PATCH_COLORS[i], lw=2.2, alpha=0.9)
        ax_x.plot(t, x[i]*100, label=PATCH_LABELS[i], color=PATCH_COLORS[i], lw=2.2, alpha=0.9)
    for ax in (ax_q, ax_x):
        if evts:
            for et, el in evts:
                ax.axvline(et, color="grey", ls="--", lw=1.3, alpha=0.7)
                ax.text(et+5, ax.get_ylim()[1]*0.95 if ax.get_ylim()[1] > 0 else 5,
                        el, fontsize=8, color="gray")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    ax_q.set(xlabel="Time (s)", ylabel="Resource level q_i",
             title="Resource Levels per Area")
    ax_q.set_ylim(bottom=0)
    ax_x.set(xlabel="Time (s)", ylabel="Robot allocation (%)",
             title="Robot Allocation per Area")
    ax_x.set_ylim(0, 100)


def savefig(fig, name):
    p = os.path.join(OUTPUT_DIR, name)
    fig.tight_layout()
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {p}")
    return p


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1 — Temporary surge (t=500–600 s)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("EXPERIMENT 1: Temporary Surge in Area 1 (t=500-600 s, w1=5.0)")
print("="*60)

def w1(t):
    w = W_BASE.copy()
    if 500 <= t <= 600:
        w[0] = 5.0
    return w

t1, q1, x1 = run(1000, w1)
a1_pre  = snap(t1, q1, x1, 490,  "Pre-surge  (490 s)")
a1_peak = snap(t1, q1, x1, 600,  "Peak surge (600 s)")
a1_post = snap(t1, q1, x1, 800,  "Post-surge (800 s)")
a1_end  = snap(t1, q1, x1, 1000, "End        (1000 s)")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Experiment 1 — Temporary Surge in Area 1 (t=500–600 s)", fontweight="bold")
plot_qx(axes[0], axes[1], t1, q1, x1, evts=[(500, "Surge start"), (600, "Surge end")])
p1 = savefig(fig, "exp1_temporary_surge.png")


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — Permanent increase (t > 500 s)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("EXPERIMENT 2: Permanent Growth-Rate Increase in Area 1 (t>500 s, w1=5.0)")
print("="*60)

def w2(t):
    w = W_BASE.copy()
    if t >= 500:
        w[0] = 5.0
    return w

t2, q2, x2 = run(2000, w2)
snap(t2, q2, x2, 490,  "Before change (490 s)")
snap(t2, q2, x2, 800,  "Mid           (800 s)")
a2_end = snap(t2, q2, x2, 2000, "End          (2000 s)")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Experiment 2 — Permanent Growth-Rate Increase in Area 1", fontweight="bold")
plot_qx(axes[0], axes[1], t2, q2, x2, evts=[(500, "Rate increase")])
p2 = savefig(fig, "exp2_permanent_increase.png")


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3 — 15 robot failures (t=500–1500 s)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("EXPERIMENT 3: 15 Robot Failures (t=500-1500 s)")
print("="*60)

FAILED = 15

def n3(t):
    return NUM_ROBOTS - FAILED if 500 <= t <= 1500 else NUM_ROBOTS

t3, q3, x3 = run(2000, lambda t: W_BASE.copy(), n_fn=n3)
snap(t3, q3, x3,  490,  "Before failures  (490 s)")
snap(t3, q3, x3, 1000,  "Mid failure     (1000 s)")
snap(t3, q3, x3, 1500,  "Recovery start  (1500 s)")
snap(t3, q3, x3, 2000,  "End             (2000 s)")

fig = plt.figure(figsize=(16, 5))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)
ax_q, ax_x, ax_n = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
fig.suptitle("Experiment 3 — 15 Robot Failures (t=500–1500 s)", fontweight="bold")
plot_qx(ax_q, ax_x, t3, q3, x3, evts=[(500, "Fail"), (1500, "Recover")])
n_trace = np.array([n3(ti) for ti in t3])
ax_n.plot(t3, n_trace, color="#8e44ad", lw=2.5)
ax_n.axhline(NUM_ROBOTS, color="gray", ls=":", lw=1.2)
ax_n.axhline(NUM_ROBOTS-FAILED, color="#e74c3c", ls=":", lw=1.2)
ax_n.set(xlabel="Time (s)", ylabel="Active robots", title="Active Robot Count")
ax_n.set_ylim(0, NUM_ROBOTS+5)
ax_n.grid(True, alpha=0.3)
p3 = savefig(fig, "exp3_robot_failures.png")


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4 — nu comparison (0, 40, 800) with unequal initial w
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("EXPERIMENT 4: Effect of Payoff Parameter nu (0, 40, 800)")
print("="*60)

# Use unequal growth rates so nu effect is visible
W4 = np.array([1.5, 0.5, 0.5, 0.5])
NU_VALS  = [0, 40, 800]
NU_COLS  = {0: "#e74c3c", 40: "#27ae60", 800: "#2980b9"}

results4 = {}
spreads4 = []

for nu in NU_VALS:
    t4, q4, x4 = run(1000, lambda t, _w=W4: _w.copy(), nu=nu)
    results4[nu] = (t4, q4, x4)
    snap(t4, q4, x4, 1000, f"nu={nu} at t=1000 s")
    sp = float(np.max(q4[:, -1]) - np.min(q4[:, -1]))
    spreads4.append(sp)
    print(f"  nu={nu:<4}: max_q={np.max(q4[:,-1]):.3f}  spread={sp:.4f}")

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
axes = axes.flatten()
fig.suptitle("Experiment 4 — Resource Dynamics for Different ν Values", fontweight="bold")
for idx, nu in enumerate(NU_VALS):
    t4, q4, x4 = results4[nu]
    ax = axes[idx]
    for i in range(NUM_PATCHES):
        ax.plot(t4, q4[i], label=PATCH_LABELS[i], color=PATCH_COLORS[i], lw=2.0, alpha=0.9)
    ax.set_title(f"ν = {nu}", fontsize=12, fontweight="bold")
    ax.set(xlabel="Time (s)", ylabel="Resource level q_i")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

ax4 = axes[3]
bars = ax4.bar([str(nu) for nu in NU_VALS], spreads4,
               color=[NU_COLS[nu] for nu in NU_VALS], alpha=0.85, edgecolor="white", lw=1.5)
for bar, val in zip(bars, spreads4):
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
             f"{val:.3f}", ha="center", va="bottom", fontweight="bold")
ax4.set(xlabel="ν value", ylabel="Resource spread (max–min q_i)",
        title="Resource Spread at t=1000 s")
ax4.grid(True, alpha=0.3, axis="y")
p4 = savefig(fig, "exp4_nu_comparison.png")

# Allocation comparison
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
fig2.suptitle("Experiment 4 — Robot Allocation for Different ν Values", fontweight="bold")
for ax, nu in zip(axes2, NU_VALS):
    t4, q4, x4 = results4[nu]
    for i in range(NUM_PATCHES):
        ax.plot(t4, x4[i]*100, label=PATCH_LABELS[i], color=PATCH_COLORS[i], lw=2.0, alpha=0.9)
    ax.set_title(f"ν = {nu}")
    ax.set(xlabel="Time (s)", ylabel="Allocation (%)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
p4b = savefig(fig2, "exp4_nu_allocation.png")


# ─── Final summary ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("CONSOLIDATED RESULTS SUMMARY")
print("="*60)

print("\n[ Exp 1 ] Peak allocation at t=600 s (temporary surge):")
for i, lbl in enumerate(PATCH_LABELS):
    print(f"  {lbl}: {a1_peak[i]:.1f}%")

print("\n[ Exp 2 ] Final allocation at t=2000 s (permanent increase):")
for i, lbl in enumerate(PATCH_LABELS):
    print(f"  {lbl}: {a2_end[i]:.1f}%")

print("\n[ Exp 3 ] Resource levels during failure (15 bots, t=500-1500 s):")
idx = np.argmin(np.abs(t3 - 1000))
for i, lbl in enumerate(PATCH_LABELS):
    print(f"  {lbl}: q={q3[i, idx]:.3f}, alloc={x3[i, idx]*100:.1f}%")

print("\n[ Exp 4 ] Resource spread at t=1000 s for each nu:")
for nu, sp in zip(NU_VALS, spreads4):
    print(f"  nu={nu:<4}: spread={sp:.4f}")

print(f"\nAll plots saved to: {os.path.abspath(OUTPUT_DIR)}")
