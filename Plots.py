import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def xy_fitted_residuals(y_true, y_pred):
    """Coordinates (x, y) for fitted residuals against true values."""
    residuals = y_true - y_pred
    return y_pred, residuals

def xy_normal_qq(y_true, y_pred):
    """Coordinates (x, y) for normal Q-Q plot."""
    residuals = y_true - y_pred

    # Нормализуем остатки
    normalized_residuals = (residuals - np.mean(residuals)) / np.std(residuals)
    sorted_residuals = np.sort(normalized_residuals)
    n = len(sorted_residuals)

    # Находим диапазон для квантилей
    quantiles_range = np.linspace(0, 100, n, endpoint=False)

    # Находим квантили для нормального распределения
    theoretical_quantiles = stats.norm.ppf(quantiles_range / 100)

    return theoretical_quantiles, sorted_residuals

def xy_scale_location(y_true, y_pred):
    """Coordinates (x, y) for scale-location plot."""
    residuals = y_true - y_pred
    normalized_residuals = (residuals - np.mean(residuals)) / np.std(residuals)
    sqrt_abs_residuals = np.sqrt(np.abs(normalized_residuals))
    return y_pred, sqrt_abs_residuals


# # Создаем более многочисленные тестовые данные
# y_true = np.arange(0, 20, 1)
# y_pred = np.random.normal(loc=y_true, scale=10, size=len(y_true))

# # # Тест и график для xy_fitted_residuals
# # x, y = xy_fitted_residuals(y_true, y_pred)
# # plt.scatter(x, y)
# # plt.xlabel("Fitted Values")
# # plt.ylabel("Residuals")
# # plt.title("Fitted vs. Residuals")
# # plt.show()

# # Тест и график для xy_normal_qq
# # x, y = xy_normal_qq(y_true, y_pred)
# # print()

# # plt.scatter(x, y)
# # plt.xlabel("Theoretical Quantiles")
# # plt.ylabel("Sorted Normalized Residuals")
# # plt.title("Normal Q-Q Plot")
# # plt.show()

# # Тест и график для xy_scale_location
# x, y = xy_scale_location(y_true, y_pred)
# plt.scatter(x, y)
# plt.xlabel("Mean of True Values")
# plt.ylabel("Square Root of Absolute Residuals")
# plt.title("Scale-Location Plot")
# plt.show()
