import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2

# Параметри згідно варіанту
student_number = 17
criterion = "χ² (хи-квадрат)"
quadrature_method = "трапеції"

# Генерація даних (нормальний розподіл)
data = np.random.normal(loc=student_number, scale=5, size=224)

# Побудова варіаційного ряду
def build_variation_series(data, k=17):
    x_min, x_max = min(data), max(data)
    R = x_max - x_min
    h = R / k
    intervals = [(x_min + i * h, x_min + (i + 1) * h) for i in range(k)]
    frequencies = [np.sum((data >= a) & (data < b)) for a, b in intervals]
    return intervals, frequencies, h

intervals, frequencies, h = build_variation_series(data)
n_emp = np.array(frequencies)

# Теоретичні частоти (з використанням формули трапецій)
def theoretical_frequencies_trapezoid(intervals, n, mean, std):
    m = []
    for a, b in intervals:
        p = (norm.pdf(a, mean, std) + norm.pdf(b, mean, std)) * h / 2
        m.append(p * n)
    return np.array(m)

mean = np.mean(data)
std = np.std(data, ddof=1)  # Вибіркове стандартне відхилення
n = len(data)

m_theor = theoretical_frequencies_trapezoid(intervals, n, mean, std)

# Критерій χ²
def chi_squared_criterion(n_emp, m_theor):
    # Щоб уникнути ділення на 0, ігноруємо інтервали з m=0
    valid = m_theor > 0
    chi_squared = np.sum((n_emp[valid] - m_theor[valid])**2 / m_theor[valid])
    df = np.count_nonzero(valid) - 3  # Кількість степенів вільності (k - 1 - 2 параметри)
    return chi_squared, df

chi2_stat, df = chi_squared_criterion(n_emp, m_theor)
critical_value = chi2.ppf(0.95, df)

# Висновок
print(f"Критерій χ²: χ² = {chi2_stat:.4f}")
print(f"Кількість степенів вільності: {df}")
print(f"Критичне значення (α = 0.05): {critical_value:.4f}")

if chi2_stat <= critical_value:
    print("Гіпотеза про нормальний розподіл ПРИЙМАЄТЬСЯ (χ² ≤ критичне значення)")
else:
    print("Гіпотеза про нормальний розподіл ВІДХИЛЯЄТЬСЯ (χ² > критичне значення)")

# Візуалізація
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(data, bins=[i[0] for i in intervals] + [intervals[-1][1]], alpha=0.7, label='Емпіричні')
x = np.linspace(min(data) - 5, max(data) + 5, 200)
plt.plot(x, norm.pdf(x, mean, std) * n * h, 'r-', label='Теоретичний')
plt.title('Емпіричний vs Теоретичний розподіл')
plt.legend()

plt.subplot(1, 2, 2)
bar_width = h * 0.4
positions = [i[0] + h / 2 for i in intervals]
plt.bar([p - bar_width / 2 for p in positions], n_emp, width=bar_width, label='Емпіричні')
plt.bar([p + bar_width / 2 for p in positions], m_theor, width=bar_width, label='Теоретичні')
plt.title('Порівняння частот')
plt.legend()

plt.tight_layout()
plt.show()

# Результати
print("\nРезультати:")
print(f"- Кількість інтервалів (k): {len(intervals)}")
print(f"- Середнє значення: {mean:.2f}")
print(f"- Стандартне відхилення: {std:.2f}")
print(f"- Критерій згоди: {criterion}")
print(f"- Квадратурна формула: {quadrature_method}")
