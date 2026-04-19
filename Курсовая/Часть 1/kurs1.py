import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import scipy.special as sp
import os

# Создаём папку figures (если её ещё нет)
os.makedirs("figures", exist_ok=True)

# --- 1. АНАЛИТИЧЕСКИЕ ФУНКЦИИ ---
def I1_const(): 
    return np.sqrt(np.pi)

def I4_const(): 
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

def calculate_mise(xi, n):
    term1 = 1.0
    term2 = I1_const() / (n * xi * I4_const())
    term3 = (1 - 1/n) * I2_func(xi) / I4_const()
    term4 = -2 * I3_func(xi) / I4_const()
    return term1 + term2 + term3 + term4

def optimize_xi(n):
    res = minimize_scalar(lambda x: calculate_mise(x, n), bounds=(0.01, 5.0), method='bounded')
    return res.x, res.fun

# --- 2. НАСТРОЙКИ ПАРАМЕТРОВ ---
n_values = [10, 50, 250, 1000]
colors = ['tab:red', 'tab:orange', 'tab:green', 'tab:blue']
xi_grid = np.linspace(0.01, 2.0, 1000)

# Данные для графиков 2 и 3
n_range = np.logspace(1, 4, 100).astype(int)
opt_xis = []
min_mises = []
for n in n_range:
    ox, mm = optimize_xi(n)
    opt_xis.append(ox)
    min_mises.append(mm)

# Предвычисляем оптимумы для n_values (чтобы не считать дважды)
opt_dict = {}
min_dict = {}
for n in n_values:
    ox, mm = optimize_xi(n)
    opt_dict[n] = ox
    min_dict[n] = mm

# ====================== ГРАФИК 1 ======================
fig1 = plt.figure(figsize=(9, 6))
ax1 = fig1.add_subplot(111)

for n, color in zip(n_values, colors):
    mise_vals = calculate_mise(xi_grid, n)
    ox = opt_dict[n]
    mm = min_dict[n]
    label_txt = f'$n={n}$, $\\xi^*={ox:.3f}$, $\\bar{{\\delta}}_{{min}}={mm:.4f}$'
    ax1.plot(xi_grid, mise_vals, color=color, lw=2.5, label=label_txt)
    ax1.plot(ox, mm, 'o', color=color, markeredgecolor='black', markersize=8, zorder=10)

ax1.set_title('1. ОИСКО $\\bar{\\delta}_n(\\xi)$', fontsize=14, fontweight='bold')
ax1.set_xlabel('$\\xi$')
ax1.set_ylabel('$\\bar{\\delta}_n$')
ax1.set_xlim(0, 1.5)
ax1.set_ylim(0, 0.6)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(title='Параметры оптимума', fontsize=10)

plt.tight_layout()
plt.savefig("figures/f1.png", dpi=300, bbox_inches='tight')

# ====================== ГРАФИК 2 ======================
fig2 = plt.figure(figsize=(9, 6))
ax2 = fig2.add_subplot(111)
ax2.plot(n_range, opt_xis, color='tab:purple', lw=2.5)
ax2.set_title('2. Зависимость $\\xi_{opt}$ от $n$', fontsize=14, fontweight='bold')
ax2.set_xlabel('$n$')
ax2.set_ylabel('$\\xi_{opt}$')
ax2.grid(True, which="both", linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("figures/f2.png", dpi=300, bbox_inches='tight')

# ====================== ГРАФИК 3 ======================
fig3 = plt.figure(figsize=(9, 6))
ax3 = fig3.add_subplot(111)
ax3.plot(n_range, min_mises, color='tab:brown', lw=2.5)
ax3.set_title('3. Нижняя граница $\\bar{\\delta}_{n, min}$ от $n$', fontsize=14, fontweight='bold')
ax3.set_xlabel('$n$')
ax3.set_ylabel('$\\bar{\\delta}_{n, min}$')
ax3.grid(True, which="both", linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("figures/f3.png", dpi=300, bbox_inches='tight')

# ====================== ГРАФИК 4 ======================
fig4 = plt.figure(figsize=(9, 6))
ax4 = fig4.add_subplot(111)

for n, color in zip(n_values, colors):
    ox = opt_dict[n]
    mm = min_dict[n]
    mise_vals = calculate_mise(xi_grid, n)
    eff_vals = mm / mise_vals
    
    label_txt = f'$n={n}$, $\\xi^*={ox:.3f}$'
    ax4.plot(xi_grid, eff_vals, color=color, lw=2.5, label=label_txt)
    ax4.plot(ox, 1.0, 'o', color=color, markeredgecolor='black', markersize=8, zorder=10)

ax4.set_title('4. Эффективность ядерной оценки $e_n(\\xi)$', fontsize=14, fontweight='bold')
ax4.set_xlabel('$\\xi$')
ax4.set_ylabel('Эффективность')
ax4.set_xlim(0, 1.5)
ax4.set_ylim(0, 1.1)
ax4.grid(True, linestyle='--', alpha=0.6)
ax4.legend(title='Объем выборки', fontsize=10)

plt.tight_layout()
plt.savefig("figures/f4.png", dpi=300, bbox_inches='tight')

# Показываем все 4 графика на экране
plt.show()
