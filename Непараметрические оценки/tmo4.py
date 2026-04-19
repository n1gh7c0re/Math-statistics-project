import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# ========================= НАСТРОЙКИ =========================
n = 200
Lambda = 2.0
np.random.seed(42)

# Генерация выборки
u1 = np.random.uniform(0, 1, n)
u2 = np.random.uniform(0, 1, n)
sample = - (1 / Lambda) * np.log(u1 * u2)

def f(x):
    return 4 * x * np.exp(-2 * x)

# Треугольное ядро
def K(u):
    a = np.sqrt(6)
    return np.where(np.abs(u) <= a, (1/a)*(1 - np.abs(u)/a), 0.0)

# Векторизованная ядерная оценка
def f_hat(x, h, sample):
    return np.sum(K((x[:, None] - sample)/h), axis=1) / (n * h)

Norm_L2 = 0.5

# ====================== ОИСКО ======================
h_values = np.arange(0.05, 2.01, 0.01)
h_values_2 = np.arange(0.1, 100.1, 1)
delta_values = []
delta_values_2 = []

print("Вычисляем ОИСКО...")
for h in h_values:
    def integrand(x):
        return (f_hat(np.array([x]), h, sample)[0] - f(x))**2
    ise, _ = quad(integrand, 0, np.inf, limit=300)
    delta_values.append(ise / Norm_L2)

for h_2 in h_values_2:
    def integrand(x):
        return (f_hat(np.array([x]), h_2, sample)[0] - f(x))**2
    ise_2, _ = quad(integrand, 0, np.inf, limit=300)
    delta_values_2.append(ise_2 / Norm_L2)    

delta_values = np.array(delta_values)
best_idx = np.argmin(delta_values)
best_h = h_values[best_idx]

delta_values_2 = np.array(delta_values_2)
best_idx_2 = np.argmin(delta_values_2)
best_h_2 = h_values_2[best_idx_2]

# ====================== ТАБЛИЦА ======================
print("\n" + "="*55)
print("  h     |   ОИСКО δ_n")
print("-" * 55)
for i in range(0, len(h_values), 8):
    print(f"{h_values[i]:.3f}   |   {delta_values[i]:.6f}")
print("="*55)
print(f"МИНИМУМ: h = {best_h:.3f},  δ_n = {delta_values[best_idx]:.6f}\n")

# ====================== ГРАФИКИ ======================
fig, axs = plt.subplots(1, 3, figsize=(14, 6))

# 1. Зависимость ОИСКО от h
axs[0].plot(h_values, delta_values, 'o-', color='purple', markersize=3, lw=2.5)
axs[0].plot(best_h, delta_values[best_idx], 'o', color='green', markersize=12)
axs[0].set_title('Зависимость ОИСКО от ширины окна h')
axs[0].set_xlabel('h')
axs[0].set_ylabel(r'$\overline{\delta}_n(h)$')
axs[0].grid(True, alpha=0.3)
axs[0].legend(['ОИСКО', f'Минимум при h={best_h:.3f}'])

# 1. Зависимость ОИСКО от h
axs[1].plot(h_values_2, delta_values_2, 'o-', color='purple', markersize=3, lw=2.5)
axs[1].plot(best_h_2, delta_values_2[best_idx_2], 'o', color='green', markersize=12)
axs[1].set_title('Зависимость ОИСКО от ширины окна h')
axs[1].set_xlabel('h')
axs[1].set_ylabel(r'$\overline{\delta}_n(h)$')
axs[1].grid(True, alpha=0.3)
axs[1].legend(['ОИСКО', f'Минимум при h={best_h_2:.3f}'])

# 2. Сравнение лучшей оценки + СИНЯЯ ГИСТОГРАММА 
x_plot = np.linspace(0, 5, 1000)
y_hat = f_hat(x_plot, best_h, sample)

axs[2].plot(x_plot, f(x_plot), 'r-', lw=2.5, label='Теоретическая f(x)')
axs[2].plot(x_plot, y_hat, 'b--', lw=2.2, label=f'Ядерная оценка (h={best_h:.3f})')
axs[2].hist(sample, bins=20, density=True, alpha=0.7, color='skyblue', 
            edgecolor='black', label='Гистограмма')
axs[2].set_title('Сравнение лучшей ядерной оценки')
axs[2].set_xlabel('x')
axs[2].set_ylabel('Плотность')
axs[2].grid(True, alpha=0.3)
axs[2].legend()

plt.tight_layout()
plt.show()