import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)

normal_data   = stats.norm.rvs(loc=0, scale=1, size=1000)
lognorm_data  = stats.lognorm.rvs(s=0.1, scale=0.5, size=1000)
weibull_data  = stats.weibull_min.rvs(c=1.5, size=1000)
#scale = сигма, loc - е^мю

def print_statistics(name, data):
    print(f"--- {name} ---")
    print(f"Минимум:             {np.min(data):.4f}")
    print(f"Максимум:            {np.max(data):.4f}")
    print(f"Среднее:             {np.mean(data):.4f}")
    print(f"Медиана:             {np.median(data):.4f}")
    print(f"Дисперсия:           {np.var(data, ddof=1):.4f}")
    print(f"Стандартное откл.:   {np.std(data, ddof=1):.4f}")
    print(f"Коэфф. асимметрии:   {stats.skew(data):.4f}")
    print()

print_statistics("Нормальное распределение",   normal_data)
print_statistics("Логнормальное распределение", lognorm_data)
print_statistics("Распределение Вейбулла",      weibull_data)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

distributions = [
    ("Нормальное",    normal_data,  stats.norm,        dict(loc=0, scale=1)),
    ("Логнормальное",  lognorm_data, stats.lognorm,     dict(s=1, scale=1)),
    ("Вейбулла",       weibull_data, stats.weibull_min, dict(c=1.5)),
]



for ax, (name, data, dist, params) in zip(axes, distributions):
    ax.hist(data, bins=30, density=True, alpha=0.6,
            color="steelblue", edgecolor="white")
    x_min, x_max = ax.get_xlim()
    x = np.linspace(x_min, x_max, 300)
    ax.plot(x, dist.pdf(x, **params), "r-", lw=2, label="Теорет. плотность")
    ax.set_title(name)
    ax.set_xlabel("Значение")
    ax.set_ylabel("Плотность")
    ax.legend()

plt.tight_layout()
plt.savefig("distributions.png", dpi=150)
plt.show()