import numpy as np
from scipy.special import laguerre
import matplotlib.pyplot as plt
from scipy.integrate import quad

# ========================= НАСТРОЙКИ =========================
np.random.seed(42)
n = 200
Lambda = 2.0

# Генерация выборки
u1 = np.random.uniform(0, 1, n)
u2 = np.random.uniform(0, 1, n)
sample = - (1 / Lambda) * np.log(u1 * u2)

def f(x):
    return 4 * x * np.exp(-2 * x)

def phi(i, x):
    L = laguerre(i)
    return np.exp(-x / 2.0) * L(x)

# ====================== КОЭФФИЦИЕНТЫ ======================
MAX_N = 25
true_c = np.zeros(MAX_N + 1)
hat_c = np.zeros(MAX_N + 1)

print("Вычисляем коэффициенты...")
for i in range(MAX_N + 1):
    integrand = lambda x: f(x) * phi(i, x)
    true_c[i], _ = quad(integrand, 0, np.inf, epsabs=1e-12, limit=500)
    hat_c[i] = np.mean(phi(i, sample))

# ====================== КРАСИВАЯ ТАБЛИЦА КОЭФФИЦИЕНТОВ ======================
print("\n" + "="*80)
print(" i | true_c_i      | hat_c_i")
print("-" * 80)
for i in range(MAX_N + 1):
    print(f"{i:2d} | {true_c[i]:.10f} | {hat_c[i]:.10f}")
print("="*50)

# ====================== ОИСКО ДЛЯ N = 5..25 (шаг 1) ======================
N_oisco = list(range(5, 26))          
delta_n = []

print("Вычисляем ОИСКО для N = 5..25...")
for N in N_oisco:
    disp = np.sum((hat_c[:N+1] - true_c[:N+1]) ** 2)
    tail = np.sum(true_c[N+1:] ** 2)
    ise = disp + tail
    delta_n.append(ise / 0.5)

delta_n = np.array(delta_n)
best_N = N_oisco[np.argmin(delta_n)]

# ====================== КРАСИВАЯ ТАБЛИЦА ОИСКО ======================
print("\n" + "="*50)
print("N  |  ОИСКО δ_n")
print("-" * 50)
for N, d in zip(N_oisco, delta_n):
    print(f"{N:2d} |  {d:.6f}")
print("="*50)

# ====================== ПРОЕКЦИОННЫЕ ОЦЕНКИ ======================
N_plot = [5, 10, 15, 20, 25]
x_plot = np.linspace(0, 5, 1000)
f_true = f(x_plot)

f_hats = []
for N in N_plot:
    f_hat = np.zeros_like(x_plot)
    for i in range(N + 1):
        f_hat += hat_c[i] * phi(i, x_plot)
    f_hats.append(f_hat)

# ====================== ГРАФИКИ: 2 СТРОКИ ПО 3 ======================
fig, axs = plt.subplots(2, 3, figsize=(18, 11))

# Первая строка: N=5, 10, 15
for idx, N in enumerate(N_plot[:3]):
    ax = axs[0, idx]
    ax.plot(x_plot, f_true, 'r-', lw=2.5, label='Теоретическая f(x)')
    ax.plot(x_plot, f_hats[idx], 'b--', lw=2.2, label=f'$\\widehat{{f}}_{{{N}}}(x)$')
    ax.set_title(f'Проекционная оценка, N = {N}', fontsize=13)
    ax.set_xlabel('x')
    ax.set_ylabel('Плотность')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

# Вторая строка: N=20, 25 + График ОИСКО
for idx, N in enumerate(N_plot[3:]):
    ax = axs[1, idx]
    ax.plot(x_plot, f_true, 'r-', lw=2.5, label='Теоретическая f(x)')
    ax.plot(x_plot, f_hats[3 + idx], 'b--', lw=2.2, label=f'$\\widehat{{f}}_{{{N}}}(x)$')
    ax.set_title(f'Проекционная оценка, N = {N}', fontsize=13)
    ax.set_xlabel('x')
    ax.set_ylabel('Плотность')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

# График ОИСКО (правый нижний)
ax_delta = axs[1, 2]
ax_delta.plot(N_oisco, delta_n, 'o-', color='purple',
              markersize=5, linewidth=2.5, label=r'$\overline{\delta}_n(N)$')

min_idx = np.argmin(delta_n)
ax_delta.plot(N_oisco[min_idx], delta_n[min_idx], 'o', color='green',
              markersize=12, label=f'Минимум при N={N_oisco[min_idx]}')

ax_delta.set_title('Зависимость ОИСКО $\\overline{\\delta}_n$ от N', fontsize=13)
ax_delta.set_xlabel('Число членов разложения N')
ax_delta.set_ylabel(r'$\overline{\delta}_n(N)$')
ax_delta.grid(True, alpha=0.3)
ax_delta.set_xticks(range(5, 26, 2))
ax_delta.legend(fontsize=11)

plt.tight_layout()
plt.show()

# ====================== ИТОГ ======================
print(f"\n{'='*50}")
print(f"МИНИМУМ ОИСКО достигается при N = {best_N}")
print(f"Значение: δ_n = {delta_n[min_idx]:.6f}")