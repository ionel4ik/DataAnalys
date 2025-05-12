import math
import random

# Генератор рівномірно розподілених псевдовипадкових чисел (мультиплікативний метод Лемера)
def lehmer_generator(seed, a, M, n):
    numbers = []
    x = seed
    for _ in range(n):
        x = (a * x) % M
        numbers.append(x / M)  # Нормалізація до інтервалу (0, 1)
    return numbers

# Тест на моменти
def check_moments(random_numbers):
    n = len(random_numbers)
    m = sum(random_numbers) / n
    D = sum((x - m)**2 for x in random_numbers) / n
    sigma = math.sqrt(D)
    print(f"\nМатематичне сподівання: {m}")
    print(f"Дисперсія: {D}")
    print(f"Середньоквадратичне відхилення: {sigma}")

# Метод Бокса-Мюллера для нормального розподілу
def box_muller(u1, u2):
    z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    z1 = math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.pi * u2)
    return z0, z1

def generate_normal_distribution(random_numbers, a, sigma):
    normal_numbers = []
    for i in range(0, len(random_numbers), 2):
        u1 = random_numbers[i]
        u2 = random_numbers[i+1] if i+1 < len(random_numbers) else random_numbers[0]
        z0, z1 = box_muller(u1, u2)
        normal_numbers.append(a + sigma * z0)
        normal_numbers.append(a + sigma * z1)
    return normal_numbers

# Хі-квадрат тест
def chi_square_test(random_numbers, k=10):
    n = len(random_numbers)
    expected = n / k
    observed = [0] * k
    for number in random_numbers:
        interval = min(int(number * k), k - 1)
        observed[interval] += 1
    chi_square = sum((obs - expected)**2 / expected for obs in observed)
    return chi_square

# Основна частина
if __name__ == "__main__":
    # Параметри для генератора Лемера (варіант A)
    q = 31
    M = 2**q  # M = 2^31
    seed = 123456789  # Непарне (1 ≤ a₀ ≤ M−1)
    beta = 13  # β mod 8 = 5 (умова виконується)

    n = 148  # Кількість чисел (HP = 100 + 12 * 4 = 224)

    # Генерація рівномірно розподілених чисел
    random_numbers = lehmer_generator(seed, beta, M, n)
    print("Згенеровані рівномірно розподілені числа:")
    print(random_numbers)

    # Перевірка на рівномірність за хі-квадрат критерієм
    chi_square_value = chi_square_test(random_numbers)
    print(f"\nЗначення критерію хі-квадрат: {chi_square_value}")
    critical_value = 16.919  # для k=10-1 ступенів свободи
    if chi_square_value < critical_value:
        print("Генератор якісний")
    else:
        print("Генератор неякісний")

    # Перевірка моментів
    check_moments(random_numbers)

    # Генерація нормально розподілених значень
    a_normal = 4  # Номер за списком
    sigma_normal = random.uniform(0.1, 1.0)  # σ ∈ (0,1)
    normal_numbers = generate_normal_distribution(random_numbers, a_normal, sigma_normal)
    print("\nЗгенеровані нормально розподілені числа:")
    print(normal_numbers)
