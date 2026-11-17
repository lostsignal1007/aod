import numpy as np
import scipy.stats as stats
import statsmodels.stats.api as sms
import matplotlib.pyplot as plt

# 1. Исходные данные
data = np.array([434, 436, 443, 445, 445, 446, 447, 447, 448, 451, 
                 452, 453, 456, 458, 458, 462, 462, 468, 472, 477])
n = len(data)
alpha = 0.05

# 2. Первичный статистический анализ
mean = np.mean(data)
variance = np.var(data, ddof=1)
std_dev = np.std(data, ddof=1)
sem = stats.sem(data)

print(f"Выборочное среднее: {mean:.2f}")
print(f"Выборочная дисперсия: {variance:.2f}")
print(f"Стандартная ошибка среднего: {sem:.2f}\n")

# 3. Доверительный интервал для среднего (SciPy и statsmodels)
ci_mean_scipy = stats.t.interval(1 - alpha, df=n-1, loc=mean, scale=sem)
ci_mean_sm = sms.DescrStatsW(data).tconfint_mean(alpha=alpha)

print(f"95%-й ДИ для среднего (SciPy): ({ci_mean_scipy[0]:.2f}; {ci_mean_scipy[1]:.2f})")
print(f"95%-й ДИ для среднего (statsmodels): ({ci_mean_sm[0]:.2f}; {ci_mean_sm[1]:.2f})")

# 4. Доверительный интервал для дисперсии
df = n - 1
chi2_lower = stats.chi2.ppf(1 - alpha / 2, df)
chi2_upper = stats.chi2.ppf(alpha / 2, df)
ci_var = (df * variance / chi2_lower, df * variance / chi2_upper)

print(f"95%-й ДИ для генеральной дисперсии: ({ci_var[0]:.2f}; {ci_var[1]:.2f})\n")

# 5. Планирование объема выборки (погрешность <= 5)
z_crit = stats.norm.ppf(1 - alpha / 2)
n_required = int(np.ceil((z_crit * std_dev / 5) ** 2))
print(f"Минимальный необходимый размер выборки: {n_required}")

# 6. Визуализация результатов
plt.figure(figsize=(10, 5))

# Отрисовка исходных точек
plt.scatter(data, np.zeros_like(data) + 1, color='blue', alpha=0.6, label='Измерения прочности', zorder=3)

# Отображение среднего и доверительного интервала
plt.axvline(mean, color='red', linestyle='--', linewidth=1.5, label=f'Выборочное среднее ({mean:.2f})')
plt.axvspan(ci_mean_scipy[0], ci_mean_scipy[1], color='red', alpha=0.15, label='95%-й ДИ для среднего')

plt.title('Распределение прочности алюминия и интервальная оценка среднего')
plt.xlabel('Прочность (ед.)')
plt.gca().get_yaxis().set_visible(False) # убираем пустую ось Y
plt.legend()
plt.grid(axis='x', alpha=0.3)
plt.show()
