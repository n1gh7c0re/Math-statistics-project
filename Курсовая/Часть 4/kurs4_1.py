import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.special import erf, ndtr

os.makedirs("figures", exist_ok=True)
rng = np.random.default_rng(42)

A = np.sqrt(3.0)
L = 2 * A
INV_L = 1.0 / L

# ============================================================
# 1. ТОЧНЫЕ АНАЛИТИЧЕСКИЕ ФОРМУЛЫ
# ============================================================
def delta_exact(xi, n):
    xi = np.maximum(np.asarray(xi, dtype=float), 1e-12)
    term1 = 1.0
    term2 = np.sqrt(3.0) / (np.sqrt(np.pi) * n * xi)
    term3 = (1.0 - 1.0 / n) * (np.sqrt(3.0) / np.pi) * (
        (np.pi / np.sqrt(3.0)) * erf(np.sqrt(3.0) / xi) +
        (xi * np.sqrt(np.pi) / 3.0) * (np.exp(-3.0 / xi**2) - 1.0)
    )
    term4 = -2.0 * (np.sqrt(3.0) / np.pi) * (
        (np.pi / np.sqrt(3.0)) * erf(np.sqrt(6.0) / xi) +
        (xi * np.sqrt(np.pi) / (3.0 * np.sqrt(2.0))) * (np.exp(-6.0 / xi**2) - 1.0)
    )
    return term1 + term2 + term3 + term4

def optimize_xi_exact(n):
    res = minimize_scalar(lambda x: float(delta_exact(x, n)), bounds=(0.01, 5.0), method='bounded')
    return res.x, res.fun

# ============================================================
# 2. ЧИСЛЕННОЕ ВЫЧИСЛЕНИЕ ISE 
# ============================================================
def relative_ise_sample(sample, h):
    """Точное вычисление ISE для одной выборки (закрытая формула)"""
    x = np.asarray(sample, dtype=float)
    n = x.size
    d2 = (x[:, None] - x[None, :]) ** 2
    
    # ∫ hat{f}^2 dx
    term1 = np.exp(-d2 / (4.0 * h * h)).sum() / (n * n * 2.0 * np.sqrt(np.pi) * h)
    
    # ∫ hat{f} * f0 dx
    term2 = np.mean(ndtr((A - x) / h) - ndtr((-A - x) / h)) * INV_L
    
    ise = term1 - 2.0 * term2 + INV_L
    return float(max(ise * L, 1e-12))

# ============================================================
# 3. ММК ОЦЕНКА КРИВОЙ delta_n(h)
# ============================================================
def mc_delta_curve(n, h_grid, n_rep, rng):
    h_grid = np.asarray(h_grid, dtype=float)
    curve = np.zeros_like(h_grid, dtype=float)
    for _ in range(n_rep):
        sample = rng.uniform(-A, A, size=n)
        for j, h in enumerate(h_grid):
            curve[j] += relative_ise_sample(sample, h)
    curve /= n_rep
    return curve

# ============================================================
# 4. ПАРАМЕТРЫ
# ============================================================
n_values = [10, 50, 250, 1000]
# h_grid = np.logspace(-2.0, 0.6, 35)          # для графика 1
h_grid = np.logspace(-2.0, 0.6, 60)
n_fine = np.unique(np.round(np.logspace(1, 4, 45)).astype(int))
# N_mc_base = lambda n: 200 if n <= 20 else 100 if n <= 50 else 50 if n <= 100 else 25 if n <= 250 else 10
N_mc_base = lambda n: max(12, int(2500 / np.sqrt(n)))

# ============================================================
# ГРАФИК 1: delta_n(ξ) — точное + ММК
# ============================================================
fig1, ax1 = plt.subplots(figsize=(11, 6))
colors = ['tab:red', 'tab:orange', 'tab:green', 'tab:blue']

for n, color in zip(n_values, colors):
    exact_curve = delta_exact(h_grid, n)
    mc_curve = mc_delta_curve(n, h_grid, N_mc_base(n), rng)
    
    ax1.plot(h_grid, exact_curve, color=color, lw=2.8, label=f'Точное, n={n}')
    ax1.plot(h_grid, mc_curve, color=color, lw=1.8, ls='--', label=f'ММК, n={n}')

ax1.set_title('1. ОИСКО $\\bar{\\delta}_n(\\xi)$ — точное решение и Монте-Карло', fontsize=15, fontweight='bold')
ax1.set_xlabel('$\\xi$')
ax1.set_ylabel('$\\bar{\\delta}_n$')
ax1.set_xlim(0.01, 1.5)
ax1.set_ylim(0, 0.6)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(fontsize=10)
plt.tight_layout()
plt.savefig("figures/f12.png", dpi=300, bbox_inches='tight')

# ============================================================
# ГРАФИК 2: ξ_opt(n) — точное + ММК
# ============================================================
xi_exact = []
xi_mc = []
for n in n_fine:
    xi_e, _ = optimize_xi_exact(n)
    xi_exact.append(xi_e)
    # ММК-оптимум
    curve = mc_delta_curve(n, h_grid, N_mc_base(n)//2, rng)  # чуть меньше реплик
    xi_mc.append(h_grid[np.argmin(curve)])

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(n_fine, xi_exact, 'b-', lw=2.8, label='Точное')
ax2.plot(n_fine, xi_mc, 'r--', lw=1.8, label='ММК')
ax2.scatter(n_fine, xi_mc, color='red', s=12, zorder=10)
ax2.set_title('2. Оптимальный параметр $\\xi_{opt}$ от $n$', fontsize=15, fontweight='bold')
ax2.set_xlabel('$n$')
ax2.set_ylabel('$\\xi_{opt}$')
ax2.set_xscale('log')
ax2.plot(n_fine, xi_mc, 'r--', lw=1.5)
ax2.scatter(n_fine, xi_mc, color='red', s=10)
ax2.grid(True, which='both', linestyle='--', alpha=0.6)
ax2.legend()
plt.tight_layout()
plt.savefig("figures/f13.png", dpi=300, bbox_inches='tight')

# ============================================================
# ГРАФИК 3: δ_min(n) — точное + ММК
# ============================================================
delta_exact_list = [optimize_xi_exact(n)[1] for n in n_fine]
delta_mc_list = []
for n in n_fine:
    curve = mc_delta_curve(n, h_grid, N_mc_base(n)//2, rng)
    delta_mc_list.append(np.min(curve))

fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.plot(n_fine, delta_exact_list, 'b-', lw=2.8, label='Точное')
ax3.plot(n_fine, delta_mc_list, 'r--', lw=2.0, label='ММК')
ax3.set_title('3. Нижняя граница $\\bar{\\delta}_{n,\\min}$ от $n$', fontsize=15, fontweight='bold')
ax3.set_xlabel('$n$')
ax3.set_ylabel('$\\bar{\\delta}_{n,\\min}$')
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.grid(True, which='both', linestyle='--', alpha=0.6)
ax3.legend()
plt.tight_layout()
plt.savefig("figures/f14.png", dpi=300, bbox_inches='tight')

# ============================================================
# ГРАФИК 4: Закон распределения ОИСКО (гистограммы)
# ============================================================
n_dist = 250
xi_opt = optimize_xi_exact(n_dist)[0]
N_values = [200, 1000, 5000]

fig4, axes = plt.subplots(1, 3, figsize=(15, 5))
fig4.suptitle(f'4. Закон распределения $\\bar{{\\delta}}_n$ (n = {n_dist}, $\\xi_{{opt}}$ = {xi_opt:.3f})', fontsize=15, fontweight='bold')

for ax, N in zip(axes, N_values):
    deltas = []
    for _ in range(N):
        sample = rng.uniform(-A, A, n_dist)
        deltas.append(relative_ise_sample(sample, xi_opt))
    ax.hist(deltas, bins=35, density=True, alpha=0.75, color='tab:purple', edgecolor='black')
    ax.set_title(f'N = {N}')
    ax.set_xlabel('$\\bar{\\delta}_n$')
    ax.set_ylabel('Плотность')
    ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("figures/f15.png", dpi=300, bbox_inches='tight')

plt.show()