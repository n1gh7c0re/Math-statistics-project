import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# ========================= НАСТРОЙКИ =========================
n = 200
Lambda = 2.0
np.random.seed(42)

def pdf(x):
    return 4 * x * np.exp(-2 * x)

Norm_L2_sq = 0.5

# ====================== 1. ГЕНЕРАЦИЯ ОДНОЙ ВЫБОРКИ (для гистограмм) ======================
u1 = np.random.uniform(0, 1, n)
u2 = np.random.uniform(0, 1, n)
sample_single = - (1 / Lambda) * np.log(u1 * u2)

x_min = np.min(sample_single)
x_max = np.max(sample_single)
Delta = x_max - x_min

print(f"Выборка сгенерирована (n={n}):")
print(f"   x_min = {x_min:.6f},  x_max = {x_max:.6f},  Δ = {Delta:.6f}\n")

# ====================== ФУНКЦИЯ ДЛЯ ТОЧНОГО ОИСКО ======================
def compute_delta_n(m, sample, x_min, x_max):
    if m < 1:
        return np.nan
    h = (x_max - x_min) / m
    bins = np.linspace(x_min, x_max, m + 1)
    counts, _ = np.histogram(sample, bins=bins)
    
    ise = 0.0
    for i in range(m):
        left, right = bins[i], bins[i + 1]
        f_hat = counts[i] / (n * h)
        integrand = lambda x: (f_hat - pdf(x)) ** 2
        integral, _ = quad(integrand, left, right, epsabs=1e-10)
        ise += integral
    
    if x_min > 0:
        left_tail, _ = quad(lambda x: pdf(x)**2, 0, x_min, epsabs=1e-10)
        ise += left_tail
    right_tail, _ = quad(lambda x: pdf(x)**2, x_max, np.inf, epsabs=1e-10)
    ise += right_tail
    
    return ise / Norm_L2_sq

# ====================== ПЕРВАЯ ФИГУРА: ГИСТОГРАММЫ m=10,20,30 ======================
fig1, axs1 = plt.subplots(1, 3, figsize=(18, 5))

m_hist = [10, 20, 30]
for idx, m in enumerate(m_hist):
    h = Delta / m
    bins = np.linspace(x_min, x_max, m + 1)
    counts, _ = np.histogram(sample_single, bins=bins)
    bin_centers = x_min + (np.arange(m) + 0.5) * h
    empirical_density = counts / (n * h)
    
    ax = axs1[idx]
    ax.bar(bin_centers, empirical_density, width=h, alpha=0.7, 
           color='skyblue', edgecolor='black', label='Гистограмма')
    
    x_plot = np.linspace(0, x_max + 0.5, 1000)
    ax.plot(x_plot, pdf(x_plot), 'r-', lw=2.5, label='Теоретическая f(x)')
    
    ax.set_title(f'Гистограмма, m = {m}\n(h = {h:.4f})', fontsize=12)
    ax.set_xlabel('x')
    ax.set_ylabel('Плотность')
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.show()

# Вывод в консоль (как в старой версии)
print(f"m = 10 | h = {Delta/10:.5f} | пустых разрядов = {np.sum(np.histogram(sample_single, bins=np.linspace(x_min,x_max,11))[0] == 0):2d} | "
      f"ОИСКО δ_n = {compute_delta_n(10, sample_single, x_min, x_max):.6f}")
print(f"m = 20 | h = {Delta/20:.5f} | пустых разрядов = {np.sum(np.histogram(sample_single, bins=np.linspace(x_min,x_max,21))[0] == 0):2d} | "
      f"ОИСКО δ_n = {compute_delta_n(20, sample_single, x_min, x_max):.6f}")
print(f"m = 30 | h = {Delta/30:.5f} | пустых разрядов = {np.sum(np.histogram(sample_single, bins=np.linspace(x_min,x_max,31))[0] == 0):2d} | "
      f"ОИСКО δ_n = {compute_delta_n(30, sample_single, x_min, x_max):.6f}")

# ====================== ВТОРАЯ ФИГУРА: ОИСКО ДЛЯ РАЗНЫХ N ======================
N_list = [1, 10, 100, 1000]
m_all = list(range(1, 31))
delta_curves = {}

print("\nВычисляем ОИСКО для разных N...")
for N_sim in N_list:
    print(f"  N = {N_sim}...")
    delta_avg = np.zeros(len(m_all))
    
    for _ in range(N_sim):
        # Новая независимая выборка
        u1 = np.random.uniform(0, 1, n)
        u2 = np.random.uniform(0, 1, n)
        sample = - (1 / Lambda) * np.log(u1 * u2)
        x_min_s = np.min(sample)
        x_max_s = np.max(sample)
        
        for i, m in enumerate(m_all):
            delta_avg[i] += compute_delta_n(m, sample, x_min_s, x_max_s)
    
    delta_avg /= N_sim
    delta_curves[N_sim] = delta_avg

# Построение второй фигуры
fig2 = plt.figure(figsize=(10, 6))
for N in N_list:
    plt.plot(m_all, delta_curves[N], 'o-', markersize=3, lw=2, label=f'N = {N}')

plt.title('Зависимость ОИСКО от числа разрядов m при разных N')
plt.xlabel('Число разрядов m')
plt.ylabel(r'$\overline{\delta}_n(m)$')
plt.grid(True, alpha=0.3)
plt.xticks(range(1, 31, 2))
plt.legend()
plt.tight_layout()
plt.show()

# ====================== ТАБЛИЦА ======================
print("\n" + "="*50)
print(" m | h       | пустых разрядов | ОИСКО δ_n")
print("-" * 50)
for m in m_all:
    h = Delta / m
    counts = np.histogram(sample_single, bins=np.linspace(x_min, x_max, m+1))[0]
    empty = np.sum(counts == 0)
    delta = compute_delta_n(m, sample_single, x_min, x_max)
    print(f"{m:2d} | {h:.5f} | {empty:2d}              | {delta:.6f}")
print("="*50)

# ====================== ИТОГ ======================
min_m = m_all[np.argmin(delta_curves[100])]
print(f"\nМИНИМУМ ОИСКО достигается при m = {min_m} "
      f"(δ_n ≈ {delta_curves[100][min_m-1]:.6f})")