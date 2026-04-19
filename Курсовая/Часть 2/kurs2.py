import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import scipy.special as sp
import os

os.makedirs("figures", exist_ok=True)

# ====================== 1. ЯДЕРНАЯ ОЦЕНКА (из части 1) ======================
def I1_const(): 
    return np.sqrt(np.pi)

def I4_const_kernel(): 
    return np.pi / np.sqrt(3)

def I2_func(xi):
    xi = np.maximum(xi, 1e-9)
    term1 = (np.pi / np.sqrt(3)) * sp.erf(np.sqrt(3) / xi)
    term2 = (xi * np.sqrt(np.pi) / 3) * (np.exp(-3 / xi**2) - 1)
    return term1 + term2

def I3_func(xi):
    xi = np.maximum(xi, 1e-9)
    term1 = (np.pi / np.sqrt(3)) * sp.erf(np.sqrt(6) / xi)
    term2 = (xi * np.sqrt(np.pi) / (3 * np.sqrt(2))) * (np.exp(-6 / xi**2) - 1)
    return term1 + term2

def calculate_mise_kernel(xi, n):
    term1 = 1.0
    term2 = I1_const() / (n * xi * I4_const_kernel())
    term3 = (1 - 1.0 / n) * I2_func(xi) / I4_const_kernel()
    term4 = -2.0 * I3_func(xi) / I4_const_kernel()
    return term1 + term2 + term3 + term4

def optimize_kernel(n):
    res = minimize_scalar(lambda x: calculate_mise_kernel(x, n), bounds=(0.01, 5.0), method='bounded')
    return res.x, res.fun

# ====================== 2. ГИСТОГРАММА (Векторизованная) ======================
A_VAL = np.sqrt(3)
L_VAL = 2 * A_VAL
INT_F2 = 1.0 / L_VAL

def sum_p2_vec(xi):
    """Векторизованный подсчет суммы p_i^2 для гладкой оптимизации"""
    xi = max(xi, 1e-9)
    i_min = np.floor(-A_VAL / xi) - 1
    i_max = np.ceil(A_VAL / xi) + 1
    i = np.arange(i_min, i_max + 1)
    
    left = i * xi
    right = (i + 1) * xi
    inter_l = np.maximum(left, -A_VAL)
    inter_r = np.minimum(right, A_VAL)
    
    lengths = np.maximum(0, inter_r - inter_l)
    p = lengths / L_VAL
    return np.sum(p**2)

def calculate_mise_hist(xi, n):
    xi = max(xi, 1e-9)
    s_p2 = sum_p2_vec(xi)
    term_var = 1.0 / (n * xi * INT_F2)
    term_other = -(1.0 + 1.0 / n) * s_p2 / (xi * INT_F2)
    return 1.0 + term_var + term_other

def optimize_hist(n):
    res = minimize_scalar(lambda x: calculate_mise_hist(x, n), bounds=(0.01, 10.0), method='bounded')
    return res.x, res.fun

# ====================== ПАРАМЕТРЫ ======================
n_values = [10, 50, 250, 1000]
colors = ['tab:red', 'tab:orange', 'tab:green', 'tab:blue']
xi_grid = np.linspace(0.01, 3.0, 800)
n_range = np.logspace(1, 4, 120).astype(int)
n_range = np.unique(n_range) # Защита от дублей при округлении logspace

# Предвычисление для гистограммы (График 1)
opt_hist = {}
min_hist = {}
for n in n_values:
    ox, mm = optimize_hist(n)
    opt_hist[n] = ox
    min_hist[n] = mm

# Предвычисление для диапазонов (Графики 2, 3, 4)
opt_xis_hist, min_mises_hist, min_mises_kernel = [], [], []
for n in n_range:
    ox_h, mm_h = optimize_hist(n)
    _, mm_k = optimize_kernel(n)
    opt_xis_hist.append(ox_h)
    min_mises_hist.append(mm_h)
    min_mises_kernel.append(mm_k)

# ====================== ГРАФИК 1 ======================
fig1 = plt.figure(figsize=(9, 6))
ax1 = fig1.add_subplot(111)
for n, color in zip(n_values, colors):
    mise_vals = [calculate_mise_hist(xi, n) for xi in xi_grid]
    ox = opt_hist[n]
    mm = min_hist[n]
    ax1.plot(xi_grid, mise_vals, color=color, lw=2.5, label=f'$n={n}$, $\\xi^*={ox:.3f}$, $\\bar{{\\delta}}_{{min}}={mm:.4f}$')
    ax1.plot(ox, mm, 'o', color=color, markeredgecolor='black', markersize=8, zorder=10)

ax1.set_title('1. ОИСКО гистограммы $\\bar{\\delta}_n(\\xi)$', fontsize=14, fontweight='bold')
ax1.set_xlabel('$\\xi$ (ширина разряда)')
ax1.set_ylabel('$\\bar{\\delta}_n$')
ax1.set_xlim(0, 2.0)
ax1.set_ylim(0, 0.6)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(title='Параметры', fontsize=9)
plt.tight_layout()
plt.savefig("figures/f5.png", dpi=300, bbox_inches='tight')

# ====================== ГРАФИК 2 ======================
fig2 = plt.figure(figsize=(9, 6))
ax2 = fig2.add_subplot(111)
ax2.plot(n_range, opt_xis_hist, color='tab:purple', lw=2.5)
ax2.set_title('2. Оптимальный параметр $\\xi_{opt}$ (гистограмма)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Объем выборки $n$')
ax2.set_ylabel('$\\xi_{opt}$')
# ax2.set_xscale('log')
ax2.grid(True, which="both", linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("figures/f6.png", dpi=300, bbox_inches='tight')

# ====================== ГРАФИК 3 ======================
fig3 = plt.figure(figsize=(9, 6))
ax3 = fig3.add_subplot(111)
ax3.plot(n_range, min_mises_hist, color='tab:brown', lw=2.5)
ax3.set_title('3. Нижняя граница $\\bar{\\delta}_{n,\\min}$ (гистограмма)', fontsize=14, fontweight='bold')
ax3.set_xlabel('Объем выборки $n$')
ax3.set_ylabel('$\\bar{\\delta}_{n,\\min}$')
# ax3.set_xscale('log')
ax3.grid(True, which="both", linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("figures/f7.png", dpi=300, bbox_inches='tight')

# ====================== ГРАФИК 4 (сравнение) ======================
fig4 = plt.figure(figsize=(9, 6))
ax4 = fig4.add_subplot(111)

ax4.plot(n_range, min_mises_kernel, 'b-', lw=2.5, label='Ядерная оценка')
ax4.plot(n_range, min_mises_hist, 'r-', lw=2.5, label='Гистограмма')

# Точный поиск пересечения через интерполяцию (лучшая практика)
diff = np.array(min_mises_kernel) - np.array(min_mises_hist)
cross_idx = np.where(np.diff(np.sign(diff)))[0]

n_cr = None
if len(cross_idx) > 0:
    idx = cross_idx[0]
    n1, n2 = n_range[idx], n_range[idx+1]
    d1, d2 = diff[idx], diff[idx+1]
    # Линейная интерполяция
    n_cr = n1 - d1 * (n2 - n1) / (d2 - d1)
    
    y1, y2 = min_mises_kernel[idx], min_mises_kernel[idx+1]
    y_cr = y1 + (n_cr - n1) * (y2 - y1) / (n2 - n1)
    
    ax4.axvline(x=n_cr, color='k', linestyle='--', alpha=0.7)
    ax4.plot(n_cr, y_cr, 'ko', markersize=7, zorder=10)
    ax4.annotate(f'$n_{{кр}} \\approx {int(n_cr)}$', xy=(n_cr, y_cr), xytext=(n_cr*30.5, y_cr + 0.005),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"), fontsize=12)

ax4.set_title('4. Сравнение ОИСКО: ядерная оценка vs гистограмма', fontsize=14, fontweight='bold')
ax4.set_xlabel('Объём выборки $n$')
ax4.set_ylabel('$\\bar{\\delta}_{n,\\min}$')
# ax4.set_xscale('log')
ax4.grid(True, which="both", linestyle='--', alpha=0.6)
ax4.legend(fontsize=11)

plt.tight_layout()
plt.savefig("figures/f8.png", dpi=300, bbox_inches='tight')

plt.show()

if n_cr:
    print(f"   n_cr ≈ {int(n_cr)}")