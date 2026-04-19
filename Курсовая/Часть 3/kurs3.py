import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import scipy.special as sp
import os

os.makedirs("figures", exist_ok=True)

# ====================== 1. ОМП (точная аналитическая формула) ======================
def calculate_mise_omp(n):
    return 2.0 / (n - 2) if n > 2 else 1.0

# ====================== 2. Ядерная оценка ======================
def I1_const(): return np.sqrt(np.pi)
def I4_const_kernel(): return np.pi / np.sqrt(3)

def I2_func(xi):
    xi = np.maximum(xi, 1e-9)
    return (np.pi / np.sqrt(3)) * sp.erf(np.sqrt(3) / xi) + (xi * np.sqrt(np.pi) / 3) * (np.exp(-3 / xi**2) - 1)

def I3_func(xi):
    xi = np.maximum(xi, 1e-9)
    return (np.pi / np.sqrt(3)) * sp.erf(np.sqrt(6) / xi) + (xi * np.sqrt(np.pi) / (3 * np.sqrt(2))) * (np.exp(-6 / xi**2) - 1)

def calculate_mise_kernel(xi, n):
    return (1.0 +
            I1_const() / (n * xi * I4_const_kernel()) +
            (1 - 1.0 / n) * I2_func(xi) / I4_const_kernel() -
            2.0 * I3_func(xi) / I4_const_kernel())

def optimize_kernel(n):
    res = minimize_scalar(lambda x: calculate_mise_kernel(x, n), bounds=(0.01, 5.0), method='bounded')
    return res.x, res.fun

# ====================== 3. Гистограмма ======================
A_VAL = np.sqrt(3)
L_VAL = 2 * A_VAL
INT_F2 = 1.0 / L_VAL

def sum_p2(xi):
    xi = max(xi, 1e-9)
    i_min = int(np.floor(-A_VAL / xi)) - 5
    i_max = int(np.ceil(A_VAL / xi)) + 5
    s = 0.0
    for i in range(i_min, i_max + 1):
        left = i * xi
        right = (i + 1) * xi
        inter_l = max(left, -A_VAL)
        inter_r = min(right, A_VAL)
        if inter_r > inter_l:
            p = (inter_r - inter_l) / L_VAL
            s += p * p
    return s

def calculate_mise_hist(xi, n):
    xi = max(xi, 1e-9)
    s_p2 = sum_p2(xi)
    denom = xi * INT_F2
    return 1.0 + 1.0 / (n * denom) - (1.0 + 1.0 / n) * s_p2 / denom

def optimize_hist(n):
    res = minimize_scalar(lambda x: calculate_mise_hist(x, n), bounds=(0.05, 10.0), method='bounded')
    return res.x, res.fun

# ====================== ПАРАМЕТРЫ ======================
n_values = [10, 50, 250, 1000]
colors = ['tab:red', 'tab:orange', 'tab:green', 'tab:blue']
xi_grid = np.linspace(0.01, 3.0, 800)
n_range = np.logspace(1, 5, 150).astype(int)

# Предвычисляем значения
omp_vals = [calculate_mise_omp(n) for n in n_range]
kernel_min = [optimize_kernel(n)[1] for n in n_range]
hist_min = [optimize_hist(n)[1] for n in n_range]

# # ====================== ГРАФИК 1: ОМП от ξ (горизонтальные линии) ======================
# fig1 = plt.figure(figsize=(9, 6))
# ax1 = fig1.add_subplot(111)

# for n, color in zip(n_values, colors):
#     delta = calculate_mise_omp(n)
#     ax1.axhline(y=delta, color=color, lw=2.5, label=f'$n={n}$, $\\bar{{\\delta}}_{{ОМП}}={delta:.4f}$')

# ax1.set_title('1. ОИСКО ОМП $\\bar{\\delta}_n(\\xi)$', fontsize=14, fontweight='bold')
# ax1.set_xlabel('$\\xi$ (параметр сглаживания — фиктивный)')
# ax1.set_ylabel('$\\bar{\\delta}_n$')
# ax1.set_xlim(0, 3.0)
# ax1.set_ylim(0, 0.4)
# ax1.grid(True, linestyle='--', alpha=0.7)
# ax1.legend(title='ОМП (не зависит от $\\xi$)', fontsize=10)
# plt.tight_layout()
# plt.savefig("figures/f1.png", dpi=300, bbox_inches='tight')

# ====================== ГРАФИК 2: Нижняя граница ОМП от n ======================
fig2 = plt.figure(figsize=(9, 6))
ax2 = fig2.add_subplot(111)
ax2.plot(n_range, omp_vals, color='tab:brown', lw=3)
ax2.set_title('1. Нижняя граница $\\bar{\\delta}_{n,\\min}$ (ОМП)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Объём выборки $n$')
ax2.set_ylabel('$\\bar{\\delta}_{n,\\min}$')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.grid(True, which="both", linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("figures/f9.png", dpi=300, bbox_inches='tight')

# ====================== ГРАФИК 3: Ядерная оценка vs ОМП ======================
fig3 = plt.figure(figsize=(9, 6))
ax3 = fig3.add_subplot(111)
ax3.plot(n_range, kernel_min, 'b-', lw=2.5, label='Ядерная оценка')
ax3.plot(n_range, omp_vals, 'r-', lw=3, label='ОМП')

# Точное нахождение n_cr
def find_n_cr(arr1, arr2, n_arr):
    diff = np.array(arr1) - np.array(arr2)
    idx = np.where(np.diff(np.sign(diff)))[0]
    if len(idx) > 0:
        i = idx[0]
        n1, n2 = n_arr[i], n_arr[i+1]
        n_cr = n1 - diff[i] * (n2 - n1) / (diff[i+1] - diff[i])
        return n_cr
    return None

n_cr1 = find_n_cr(kernel_min, omp_vals, n_range)
if n_cr1:
    ax3.axvline(x=n_cr1, color='k', linestyle='--', alpha=0.8)
    ax3.annotate(f'$n_{{кр}} \\approx {int(n_cr1)}$', xy=(n_cr1, 0.15),
                 xytext=(n_cr1 * 1.8, 0.22), arrowprops=dict(arrowstyle="->"), fontsize=11)

ax3.set_title('2. Сравнение: Ядерная оценка vs ОМП', fontsize=14, fontweight='bold')
ax3.set_xlabel('$n$')
ax3.set_ylabel('$\\bar{\\delta}_{n,\\min}$')
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.grid(True, which="both", linestyle='--', alpha=0.7)
ax3.legend()
plt.tight_layout()
plt.savefig("figures/f10.png", dpi=300, bbox_inches='tight')

# ====================== ГРАФИК 4: Гистограмма vs ОМП ======================
fig4 = plt.figure(figsize=(9, 6))
ax4 = fig4.add_subplot(111)
ax4.plot(n_range, hist_min, 'g-', lw=2.5, label='Гистограмма')
ax4.plot(n_range, omp_vals, 'r-', lw=3, label='ОМП')

n_cr2 = find_n_cr(hist_min, omp_vals, n_range)
if n_cr2:
    ax4.axvline(x=n_cr2, color='k', linestyle='--', alpha=0.8)
    ax4.annotate(f'$n_{{кр}} \\approx {int(n_cr2)}$', xy=(n_cr2, 0.15),
                 xytext=(n_cr2 * 1.8, 0.22), arrowprops=dict(arrowstyle="->"), fontsize=11)

ax4.set_title('3. Сравнение: Гистограмма vs ОМП', fontsize=14, fontweight='bold')
ax4.set_xlabel('$n$')
ax4.set_ylabel('$\\bar{\\delta}_{n,\\min}$')
ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.grid(True, which="both", linestyle='--', alpha=0.7)
ax4.legend()
plt.tight_layout()
plt.savefig("figures/f11.png", dpi=300, bbox_inches='tight')

plt.show()