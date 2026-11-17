import numpy as np
import scipy.stats as stats
from statsmodels.stats.power import TTestIndPower

# 1. Исходные данные (выборки)
sample_1 = np.array([220, 223, 234, 245, 257])
sample_2 = np.array([234, 246, 259, 262, 278, 280, 285, 290])

# 2. Расчет описательных статистик
mean_1 = np.mean(sample_1)
mean_2 = np.mean(sample_2)
var_1 = np.var(sample_1, ddof=1)  # несмещенная дисперсия (объем-1)
var_2 = np.var(sample_2, ddof=1)
n1 = len(sample_1)
n2 = len(sample_2)

# 3. Двусторонний F-тест равенства дисперсий
f_stat = var_1 / var_2
df1 = n1 - 1
df2 = n2 - 1
# Корректный двусторонний p-value для F-распределения
p_f_one_sided = stats.f.sf(f_stat, df1, df2) if f_stat > 1 else stats.f.cdf(f_stat, df1, df2)
p_f_two_sided = min(1.0, 2.0 * p_f_one_sided)

# 4. t-тест Стьюдента (равные дисперсии) и t-тест Уэлча (разные дисперсии)
t_stat_student, p_student = stats.ttest_ind(sample_1, sample_2, equal_var=True)
t_stat_welch, p_welch = stats.ttest_ind(sample_1, sample_2, equal_var=False)

# 5. Тест Шапиро—Уилка на нормальность
w_stat_1, p_shapiro_1 = stats.shapiro(sample_1)
w_stat_2, p_shapiro_2 = stats.shapiro(sample_2)

# 6. Однофакторный дисперсионный анализ (ANOVA)
f_anova, p_anova = stats.f_oneway(sample_1, sample_2)

# 7. Размер эффекта Коэна d и статистическая мощность
# Объединенное стандартное отклонение (pooled standard deviation)
pooled_std = np.sqrt(((n1 - 1) * var_1 + (n2 - 1) * var_2) / (n1 + n2 - 2))
cohen_d = abs(mean_1 - mean_2) / pooled_std
effect_size_comment = "Большой эффект" if cohen_d > 0.8 else "Средний эффект"

# Расчет мощности для t-теста
analysis = TTestIndPower()
# ratio = n2 / n1 (отношение размеров выборок)
power = analysis.solve_power(effect_size=cohen_d, nobs1=n1, ratio=n2/n1, alpha=0.05, alternative='two-sided')

# 8. Вывод результатов в консоль в виде таблицы
print("-" * 90)
print(f"{'Показатель или тест':<30} | {'Статистика':<20} | {'p-значение / результат':<30}")
print("-" * 90)
print(f"{'Среднее: выборка 1':<30} | {mean_1:<20.2f} МПа | {'—':<30}")
print(f"{'Среднее: выборка 2':<30} | {mean_2:<20.2f} МПа | {'—':<30}")
print(f"{'Выборочная дисперсия 1':<30} | {var_1:<20.2f} МПа² | {'—':<30}")
print(f"{'Выборочная дисперсия 2':<30} | {var_2:<20.2f} МПа² | {'—':<30}")
print(f"{'Двусторонний F-тест':<30} | F = {f_stat:<16.4f} | p = {p_f_two_sided:.4f}")
print(f"{'t-тест Стьюдента':<30} | t = {t_stat_student:<16.4f} | p = {p_student:.4f}")
print(f"{'t-тест Уэлча':<30} | t = {t_stat_welch:<16.4f} | p = {p_welch:.4f}")
print(f"{'Шапиро-Уилк: выборка 1':<30} | W = {w_stat_1:<16.4f} | p = {p_shapiro_1:.4f}")
print(f"{'Шапиро-Уилк: выборка 2':<30} | W = {w_stat_2:<16.4f} | p = {p_shapiro_2:.4f}")
print(f"{'Однофакторная ANOVA':<30} | F = {f_anova:<16.4f} | p = {p_anova:.4f}")
print(f"{'Размер эффекта Коэна':<30} | d = {cohen_d:<16.4f} | {effect_size_comment:<30}")
print(f"{'Мощность t-теста':<30} | {power:<20.4f} | {power * 100:.2f} %")
print("-" * 90)
