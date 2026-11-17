import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm, weibull_min, linregress

np.random.seed(42)

# Истинные значения параметров из таблицы
TRUE_PARAMS = {
    'normal': {'mu': 5.0, 'sigma': 2.0},
    'linear': {'a': 3.0, 'b': 1.7},
    'weibull': {'k': 1.8, 'lambda': 4.0}
}

sample_sizes = [30, 100, 500, 2000]

for n in sample_sizes:
    print(f"\n{'='*20} РЕЗУЛЬТАТЫ ДЛЯ N = {n} {'='*20}")
    
    # ==========================================
    # ГЕНЕРАЦИЯ ДАННЫХ
    # ==========================================
    # Выборка из нормального распределения
    data_norm = np.random.normal(TRUE_PARAMS['normal']['mu'], TRUE_PARAMS['normal']['sigma'], n)
    
    # Данные линейной регрессии (y = a*x + b + eps)
    x_lin = np.random.uniform(0, 10, n)
    eps = np.random.normal(0, 1.0, n)  # нормально распределённая ошибка
    y_lin = TRUE_PARAMS['linear']['a'] * x_lin + TRUE_PARAMS['linear']['b'] + eps
    
    # Выборка из распределения Вейбулла
    # В scipy.stats.weibull_min: c = k, scale = lambda
    data_weib = weibull_min.rvs(c=TRUE_PARAMS['weibull']['k'], scale=TRUE_PARAMS['weibull']['lambda'], size=n)
    
    # ==========================================
    # 1. НОРМАЛЬНОЕ РАСПРЕДЕЛЕНИЕ
    # ==========================================
    # Аналитические оценки ММП
    mu_mmp_analyt = np.mean(data_norm)
    sigma_mmp_analyt = np.std(data_norm)  # Смещенная оценка (классический ММП)
    
    # Численные оценки ММП (минимизация минус логарифмической функции правдоподобия)
    def neg_log_likelihood_norm(params):
        mu, sigma = params
        if sigma <= 0: return np.inf
        return -np.sum(norm.logpdf(data_norm, mu, sigma))
    
    res_norm = minimize(neg_log_likelihood_norm, [1.0, 1.0], method='Nelder-Mead')
    mu_mmp_num, sigma_mmp_num = res_norm.x
    
    err_mu = abs(mu_mmp_analyt - TRUE_PARAMS['normal']['mu'])
    err_sigma = abs(sigma_mmp_analyt - TRUE_PARAMS['normal']['sigma'])
    
    print(f"--- Нормальное распределение ---")
    print(f"Параметр mu (Истинное: {TRUE_PARAMS['normal']['mu']}):")
    print(f"  Оценка ММП (Аналит.): {mu_mmp_analyt:.4f} | Числ.: {mu_mmp_num:.4f} | Абс. ошибка: {err_mu:.4f}")
    print(f"Параметр sigma (Истинное: {TRUE_PARAMS['normal']['sigma']}):")
    print(f"  Оценка ММП (Аналит.): {sigma_mmp_analyt:.4f} | Числ.: {sigma_mmp_num:.4f} | Абс. ошибка: {err_sigma:.4f}")

    # ==========================================
    # 2. ЛИНЕЙНАЯ РЕГРЕССИЯ
    # ==========================================
    slope, intercept, r_value, p_value, std_err = linregress(x_lin, y_lin)
    
    err_a = abs(slope - TRUE_PARAMS['linear']['a'])
    err_b = abs(intercept - TRUE_PARAMS['linear']['b'])
    
    print(f"\n--- Линейная регрессия ---")
    print(f"Параметр a (Истинное: {TRUE_PARAMS['linear']['a']}): Оценка МНК = {slope:.4f} | Абс. ошибка = {err_a:.4f}")
    print(f"Параметр b (Истинное: {TRUE_PARAMS['linear']['b']}): Оценка МНК = {intercept:.4f} | Абс. ошибка = {err_b:.4f}")

    # ==========================================
    # 3. РАСПРЕДЕЛЕНИЕ ВЕЙБУЛЛА
    # ==========================================
    # Оценка ММП через scipy
    k_mmp, _, lambda_mmp = weibull_min.fit(data_weib, floc=0)
    
    # Оценка МНК (Линеаризация функции распределения)
    sorted_weib = np.sort(data_weib)
    # Метод медианных рангов для оценки F(x)
    i = np.arange(1, n + 1)
    F_i = (i - 0.3) / (n + 0.4)
    
    # Y = ln(-ln(1 - F_i)), X = ln(sorted_x)
    # Так как Y = k*X - k*ln(lambda), сделаем линейную аппроксимацию Y = A*X + B
    X_mnk = np.log(sorted_weib)
    Y_mnk = np.log(-np.log(1 - F_i))
    
    A, B, _, _, _ = linregress(X_mnk, Y_mnk)
    k_mnk = A
    lambda_mnk = np.exp(-B / k_mnk)
    
    err_k_mmp = abs(k_mmp - TRUE_PARAMS['weibull']['k'])
    err_lambda_mmp = abs(lambda_mmp - TRUE_PARAMS['weibull']['lambda'])
    err_k_mnk = abs(k_mnk - TRUE_PARAMS['weibull']['k'])
    err_lambda_mnk = abs(lambda_mnk - TRUE_PARAMS['weibull']['lambda'])
    
    print(f"\n--- Распределение Вейбулла ---")
    print(f"Параметр k (Истинное: {TRUE_PARAMS['weibull']['k']}):")
    print(f"  Оценка ММП = {k_mmp:.4f} (Ошибка: {err_k_mmp:.4f})")
    print(f"  Оценка МНК = {k_mnk:.4f} (Ошибка: {err_k_mnk:.4f})")
    print(f"Параметр lambda (Истинное: {TRUE_PARAMS['weibull']['lambda']}):")
    print(f"  Оценка ММП = {lambda_mmp:.4f} (Ошибка: {err_lambda_mmp:.4f})")
    print(f"  Оценка МНК = {lambda_mnk:.4f} (Ошибка: {err_lambda_mnk:.4f})")
    
    # ==========================================
    # ВИЗУАЛИЗАЦИЯ (для n = 100 построим графики)
    # ==========================================
    if n == 100:
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        
        # График нормального распределения
        axs[0].hist(data_norm, bins=15, density=True, alpha=0.6, color='g', label='Выборка')
        x_axis = np.linspace(mu_mmp_analyt - 3*sigma_mmp_analyt, mu_mmp_analyt + 3*sigma_mmp_analyt, 100)
        axs[0].plot(x_axis, norm.pdf(x_axis, mu_mmp_analyt, sigma_mmp_analyt), 'r-', lw=2, label='ММП')
        axs[0].set_title('Нормальное распределение (n=100)')
        axs[0].legend()
        
        # График регрессии
        axs[1].scatter(x_lin, y_lin, alpha=0.7, label='Данные')
        x_vals = np.array([0, 10])
        axs[1].plot(x_vals, slope*x_vals + intercept, 'r-', lw=2, label='МНК линия')
        axs[1].set_title('Линейная регрессия (n=100)')
        axs[1].legend()
        
        # График распределения Вейбулла
        axs[2].hist(data_weib, bins=15, density=True, alpha=0.6, color='b', label='Выборка')
        x_axis_w = np.linspace(0.1, np.max(data_weib), 100)
        axs[2].plot(x_axis_w, weibull_min.pdf(x_axis_w, k_mmp, scale=lambda_mmp), 'r-', lw=2, label='ММП')
        axs[2].plot(x_axis_w, weibull_min.pdf(x_axis_w, k_mnk, scale=lambda_mnk), 'm--', lw=2, label='МНК')
        axs[2].set_title('Распределение Вейбулла (n=100)')
        axs[2].legend()
        
        plt.tight_layout()
        plt.show()