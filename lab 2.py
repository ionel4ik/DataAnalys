import numpy as np
import matplotlib.pyplot as plt

# Приклад даних (заміни на свої дані з лабораторної роботи №1)
data = np.random.normal(loc=17, scale=5, size=224)  # Нормально розподілені числа

# 1. Побудова інтервального варіаційного ряду
def build_variation_series(data, k):
    x_min = min(data)
    x_max = max(data)
    R = x_max - x_min
    h = R / k  # Ширина інтервалу
    intervals = [(x_min + i * h, x_min + (i + 1) * h) for i in range(k)]
    frequencies = [sum((data >= interval[0]) & (data < interval[1])) for interval in intervals]
    return intervals, frequencies

k = 14  # Кількість інтервалів (номер студента <10, тому 10+4)
intervals, frequencies = build_variation_series(data, k)

# Виведення варіаційного ряду
print("Інтервали та частоти:")
for interval, freq in zip(intervals, frequencies):
    print(f"[{interval[0]:.2f}, {interval[1]:.2f}): {freq}")

# 2. Обчислення числових характеристик
def calculate_characteristics(intervals, frequencies):
    n = sum(frequencies)
    midpoints = [(interval[0] + interval[1]) / 2 for interval in intervals]
    mean = sum(midpoints[i] * frequencies[i] for i in range(k)) / n
    variance = sum((midpoints[i] - mean) ** 2 * frequencies[i] for i in range(k)) / n
    std_dev = np.sqrt(variance)
    return mean, variance, std_dev

mean, variance, std_dev = calculate_characteristics(intervals, frequencies)
print(f"\nСереднє значення: {mean:.2f}")
print(f"Дисперсія: {variance:.2f}")
print(f"Середнє квадратичне відхилення: {std_dev:.2f}")

# Модальний інтервал
modal_interval_index = np.argmax(frequencies)
modal_interval = intervals[modal_interval_index]
print(f"\nМодальний інтервал: [{modal_interval[0]:.2f}, {modal_interval[1]:.2f})")

# 3. Виявлення аномальних значень за правилом 3-х сигм
def detect_outliers(data, mean, std_dev):
    lower_bound = mean - 3 * std_dev
    upper_bound = mean + 3 * std_dev
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return outliers

outliers = detect_outliers(data, mean, std_dev)
print(f"\nАномальні значення: {outliers}")

# 4. Побудова емпіричної функції розподілу
def empirical_distribution_function(data, intervals):
    sorted_data = np.sort(data)
    n = len(data)
    ecdf = np.arange(1, n + 1) / n
    return sorted_data, ecdf

sorted_data, ecdf = empirical_distribution_function(data, intervals)

# Графік емпіричної функції розподілу
plt.step(sorted_data, ecdf, where="post")
plt.title("Емпірична функція розподілу")
plt.xlabel("Значення")
plt.ylabel("F(x)")
plt.grid(True)
plt.show()