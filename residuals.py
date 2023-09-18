import numpy as np

def residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Residuals"""
    return y_true - y_pred

def squared_errors(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Squared errors"""
    res = residuals(y_true, y_pred)
    return res ** 2

def quantile_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float = 0.01) -> np.ndarray:
    """Quantile loss terms"""
    res = residuals(y_true, y_pred)
    return np.maximum(q * res, (q - 1) * res)

def logloss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """LogLoss terms"""
    if (y_pred == 0).any() or (y_pred == 1).any():
        raise ValueError("y_pred cannot be equal to 0 or 1 in logloss function")
    return y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)

def ape(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """MAPE terms"""
    if (y_true < 0).any() or (y_pred < 0).any():
        raise ValueError("Negative dimensions are not allowed in y_true or y_pred")

    try:
        return 1 - y_pred / y_true
    except ZeroDivisionError:
        raise ValueError("Zero division error occurred in ape function")
    except Exception as e:
        raise ValueError(f"An error occurred in ape function: {str(e)}")
