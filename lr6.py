import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# 1. Фиксация состояния генератора и моделирование выборок
np.random.seed(42)

# Независимые группы
group1 = np.random.exponential(scale=100, size=30) + 350
group2 = np.random.lognormal(mean=6.1, sigma=0.3, size=35)
group3 = np.random.weibull(a=2.0, size=25) * 80 + 380

# Связанные выборки (на основе первых 25 элементов Группы 1)
paired_base = group1[:25]
paired_group = paired_base + np.random.normal(loc=0, scale=10, size=25)

# 2. Расчёт описательных статистик
stats_dict = {}
for name, data in [("Группа 1", group1), ("Группа 2", group2), ("Группа 3", group3)]:
    stats_dict[name] = {
        "N": len(data),
        "Mean": np.mean(data),
        "Median": np.median(data),
        "Std": np.std(data, ddof=1)
    }

print("--- ОПИСАТЕЛЬНЫЕ СТАТИСТИКИ ---")
for name, metrics in stats_dict.items():
    print(f"{name}: N={metrics['N']}, Среднее={metrics['Mean']:.4f}, "
          f"Медиана={metrics['Median']:.4f}, Станд. откл.={metrics['Std']:.4f}")

# 3. Проведение статистических тестов (без сдвига)
mw_stat, mw_p = stats.mannwhitneyu(group1, group2, alternative='two-sided')
ks_stat, ks_p = stats.ks_2samp(group1, group2)
kw_stat, kw_p = stats.kruskal(group1, group2, group3)
wilc_stat, wilc_p = stats.wilcoxon(paired_base, paired_group, alternative='two-sided')

# Реализация теста знаков через биномиальный критерий для ненулевых разностей
diffs = paired_base - paired_group
pos_diffs = np.sum(diffs > 0)
n_nonzero = np.sum(diffs != 0)
sign_p = stats.binomtest(pos_diffs, n=n_nonzero, p=0.5, alternative='two-sided').pvalue

# 4. Повторный парный эксперимент со сдвигом +15
paired_group_shift = paired_group + 15
wilc_shift_stat, wilc_shift_p = stats.wilcoxon(paired_base, paired_group_shift, alternative='two-sided')

diffs_shift = paired_base - paired_group_shift
pos_diffs_shift = np.sum(diffs_shift > 0)
n_nonzero_shift = np.sum(diffs_shift != 0)
sign_shift_p = stats.binomtest(pos_diffs_shift, n=n_nonzero_shift, p=0.5, alternative='two-sided').pvalue

# Вывод результатов в формате таблицы (для п. 13 текстового ответа)
print("\n--- РЕЗУЛЬТАТЫ СТАТИСТИЧЕСКИХ КРИТЕРИЕВ ---")
print(f"{'Критерий':<25} | {'Статистика':<12} | {'p-value':<10} | {'Решение при alpha=0.05':<20}")
print("-" * 78)
print(f"{'Манна — Уитни':<25} | {mw_stat:<12.6f} | {mw_p:<10.6f} | {'H0 не отвергается' if mw_p >= 0.05 else 'H0 отвергается'}")
print(f"{'Колмогорова — Смирнова':<25} | {ks_stat:<12.6f} | {ks_p:<10.6f} | {'H0 не отвергается' if ks_p >= 0.05 else 'H0 отвергается'}")
print(f"{'Крускаля — Уоллиса':<25} | {kw_stat:<12.6f} | {kw_p:<10.6f} | {'H0 не отвергается' if kw_p >= 0.05 else 'H0 отвергается'}")
print(f"{'Уилкоксона':<25} | {wilc_stat:<12.6f} | {wilc_p:<10.6f} | {'H0 не отвергается' if wilc_p >= 0.05 else 'H0 отвергается'}")
print(f"{'Тест знаков':<25} | {f'{pos_diffs} пол. разн.':<12} | {sign_p:<10.6f} | {'H0 не отвергается' if sign_p >= 0.05 else 'H0 отвергается'}")

print("\n--- РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА СО СДВИГОМ +15 ---")
print(f"{'Уилкоксона (со сдвигом)':<25} | {wilc_shift_stat:<12.6f} | {wilc_shift_p:<10.7f} | {'H0 не отвергается' if wilc_shift_p >= 0.05 else 'H0 отвергается'}")
print(f"{'Тест знаков (со сдвигом)':<25} | {f'{pos_diffs_shift} пол. разн.':<12} | {sign_shift_p:<10.7f} | {'H0 не отвергается' if sign_shift_p >= 0.05 else 'H0 отвергается'}")

# 5. Визуализация: построение гистограмм
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
titles = ["Группа 1 (Экспоненциальное)", "Группа 2 (Логнормальное)", "Группа 3 (Вейбулла)"]
colors = ['#4C72B0', '#55A868', '#C44E52']

for i, (data, title, color) in enumerate(zip([group1, group2, group3], titles, colors)):
    axes[i].hist(data, bins=8, edgecolor='black', alpha=0.7, color=color)
    axes[i].set_title(title, fontsize=11)
    axes[i].set_xlabel("Значение признака")
    axes[i].set_ylabel("Частота")
    axes[i].grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
