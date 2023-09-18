import numpy as np
from typing import List
from sklearn.metrics import log_loss

def fairness(residuals: np.ndarray) -> float:
    """Compute Gini fairness of array of values"""
    n = len(residuals)
    if n == 0:
        raise ValueError("The input array 'residuals' is empty.")

    # 1. Считаем модули остатков
    abs_residuals = np.abs(residuals)

    # 2. Сортируем полученное
    sorted_residuals = np.sort(abs_residuals)

    # 3. Считаем сумму абсолютной разности каждого значения и того, что стоит после этого остатка
    sum_diff = 0
    for i in range(n):
        for j in range(i+1, n):
            sum_diff += np.abs(sorted_residuals[i] - sorted_residuals[j])

    # 4. Полученную сумму делим на число остатков в квадрате
    # и на среднее значение абсолютного значения остатков
    gini = sum_diff / (n**2 * np.mean(abs_residuals))

    return 1 - gini

def logloss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """LogLoss terms"""
    if (y_pred == 0).any() or (y_pred == 1).any():
        raise ValueError("y_pred cannot be equal to 0 or 1 in logloss function")
    return y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)

def best_prediction(
    y_true: np.ndarray, y_preds: List[np.ndarray], fairness_drop: float = 0.05
) -> int:
    """Find index of best model"""
    if len(y_preds) == 0:
        raise ValueError("The input list 'y_preds' is empty.")

    baseline_loss = log_loss(y_true, y_preds[0])  # LogLoss для базовой модели
    best_model_index = 0  # Индекс лучшей модели
    best_log_loss = baseline_loss  # Начальное значение для нахождения минимального LogLoss
    base_residuals = logloss(y_true, y_preds[0])
    base_fairness = fairness(base_residuals)

    for i, y_pred in enumerate(y_preds):
        y_logloss = log_loss(y_true, y_pred)  # Рассчитываем LogLoss для текущей модели
        # Рассчитываем остатки
        residuals = logloss(y_true, y_pred) # y_true - y_pred
        # residuals = log_loss(y_true, y_pred, normalize=False)


        # Рассчитываем справедливость для текущей модели
        model_fairness = fairness(residuals)

        # Проверяем, удовлетворяет ли модель условию fairness_drop
        if model_fairness > base_fairness * (1 - fairness_drop):
            # Если LogLoss лучше, обновляем индекс лучшей модели и значение LogLoss
            if y_logloss < best_log_loss:
                best_log_loss = y_logloss
                best_model_index = i

    return best_model_index


# # Пример использования
# y_true = np.array([1, 0, 1, 0, 1])
# y_preds = [np.array([0.9, 0.1, 0.8, 0.2, 0.7]), np.array([0.7, 0.2, 0.6, 0.3, 0.8]), np.array([0.8, 0.1, 0.9, 0.3, 0.7])]

# fairness_drop = 0.05
# best_model_index = best_prediction(y_true, y_preds, fairness_drop)
# print("Best model index:", best_model_index)