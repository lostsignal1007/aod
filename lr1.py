import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

#data = np.array([434,436,443,445,446,447,447,448,451,
#                 452,453,456, 458,458,462,462,468,472,477])
data = np.array([1,2,3,4,4,5])
np.random.seed(42)
#data = np.random.normal(loc=50, scale=10, size=1000)


mean_val = np.mean(data)                      # Среднее значение
variance_val = np.var(data, ddof=1)           # Дисперсия (несмещенная, ddof=1)
std_dev_val = np.std(data, ddof=1)            # Стандартное отклонение (несмещенное)

# Квантили (например: 25-й, 50-й/медиана, 75-й процентили)
quantiles = np.percentile(data, [25, 50, 75,90])


print("=== Основные статистические понятия ===")
print(f"Среднее значение (Mean):         {mean_val:.4f}")
print(f"Дисперсия (Variance):             {variance_val:.4f}")
print(f"Стандартное отклонение (Std Dev): {std_dev_val:.4f}")
print(f"25-й квантиль (1-й квартиль):     {quantiles[0]:.4f}")
print(f"50-й квантиль (Медиана):          {quantiles[1]:.4f}")
print(f"75-й квантиль (3-й квартиль):     {quantiles[2]:.4f}")
print(f"90-й квантиль:                    {quantiles[3]:.4f}")


plt.figure(figsize=(10, 6))


plt.hist(data, bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Данные')

# Добавляем линию теоретической плотности вероятности распределения из scipy.stats
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mean_val, std_dev_val)
plt.plot(x, p, 'r-', linewidth=2, label='Плотность вероятности (SciPy)')

# Отмечаем вертикальной линией среднее значение
plt.axvline(mean_val, color='darkred', linestyle='dashed', linewidth=2, label=f'Среднее: {mean_val:.2f}')

# Настройка внешнего вида графика
plt.title('Распределение данных и статистические метрики')
plt.xlabel('Значения')
plt.ylabel('Плотность')
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()