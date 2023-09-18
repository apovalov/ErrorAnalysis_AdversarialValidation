from typing import Tuple, Optional
import numpy as np
from scipy import stats

def test_normality(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float = 0.05
) -> Tuple[float, bool]:
    """Normality test

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    alpha : float, optional (default=0.05)
        Significance level for the test

    Returns
    -------
    p_value : float
        p-value of the normality test

    is_rejected : bool
        True if the normality hypothesis is rejected, False otherwise

    """
    res = y_true - y_pred
    pvalue = stats.shapiro(res).pvalue

    is_rejected = pvalue < alpha
    return pvalue, is_rejected


def test_unbiased(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefer: Optional[str] = None,
    alpha: float = 0.05,
) -> Tuple[float, bool]:
    """Unbiasedness test

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    prefer : str, optional (default=None)
        If None or "two-sided", test whether the residuals are unbiased.
        If "positive", test whether the residuals are unbiased or positive.
        If "negative", test whether the residuals are unbiased or negative.

    alpha : float, optional (default=0.05)
        Significance level for the test

    Returns
    -------
    p_value : float
        p-value of the test

    is_rejected : bool
        True if the unbiasedness hypothesis is rejected, False otherwise

    """

    residuals = y_true - y_pred
    # n = len(residuals)

    if prefer is None or prefer == "two-sided":
        t_statistics, p_value = stats.ttest_1samp(residuals, 0)
    elif prefer == "positive":
        t_statistics, p_value = stats.ttest_1samp(residuals, 0, alternative = "greater")
    elif prefer == "negative":
        t_statistic, p_value = stats.ttest_1samp(residuals, 0, alternative="less")
    else:
        raise ValueError("Invalid value for 'prefer' parameter")
    is_rejected = p_value < alpha

    return p_value, is_rejected


def test_homoscedasticity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bins: int = 10,
    test: Optional[str] = None,
    alpha: float = 0.05,
) -> Tuple[float, bool]:
    """Homoscedasticity test"""

    residuals = y_true - y_pred
    sorted_indices = np.argsort(y_true)
    sorted_residuals = residuals[sorted_indices]

    # Вычисляем количество элементов в каждом бине (кроме последнего)
    bin_size = len(sorted_residuals) // bins
    bin_sizes = [bin_size] * (bins - 1)

    # Определяем размер последнего бина (может быть короче)
    last_bin_size = len(sorted_residuals) - sum(bin_sizes)
    bin_sizes.append(last_bin_size)

    # Разделяем упорядоченные остатки на бины
    bin_indices = np.repeat(np.arange(bins), bin_sizes)
    bins_residuals = [sorted_residuals[bin_indices == bin_idx] for bin_idx in range(bins)]

    if test is None or test == "bartlett":
        _, p_value = stats.bartlett(*bins_residuals)
    elif test == "levene":
        _, p_value = stats.levene(*bins_residuals)
    elif test == "fligner":
        _, p_value = stats.fligner(*bins_residuals)
    else:
        raise ValueError("Invalid value for 'test' parameter")

    is_rejected = p_value < alpha

    return p_value, is_rejected

# # Тестовые данные
# np.random.seed(0)
# y_true = np.random.rand(10)
# y_pred = y_true + np.random.normal(0, 0.1, 10)  # Добавляем случайный шум

# # Вызов функции для тестирования
# p_value, is_rejected = test_homoscedasticity(y_true, y_pred, 3)

# print(f"P-значение: {p_value}")
# print(f"Гипотеза о гомоскедастичности {'отвергнута' if is_rejected else 'не отвергнута'}")
