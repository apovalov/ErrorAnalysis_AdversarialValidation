from typing import Optional
import numpy as np
import pandas as pd
import residuals

def best_cases(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: pd.Series,
    top_k: int = 10,
    mask: Optional[pd.Series] = None,
    func: Optional[str] = None,
) -> dict:
    """Return top-k best cases according to the given function"""
    # Вычисляем остатки в зависимости от выбранной функции
    if func == "squared_errors":
        resid = residuals.squared_errors(y_test, y_pred)
    elif func == "logloss":
        resid = residuals.logloss(y_test, y_pred)
    elif func == "ape":
        resid = residuals.ape(y_test, y_pred)
    else:
        resid = residuals.residuals(y_test, y_pred)

    # Применяем маску, если она предоставлена
    if mask is not None:
        resid = resid[mask]
        X_test = X_test[mask]
        y_test = y_test[mask]
        y_pred = y_pred[mask]


    # Отсортировываем остатки и выбираем топ-K лучших случаев
    top_indices = np.abs(resid).nsmallest(top_k).index

    # Получаем соответствующие записи из X_test, y_test и y_pred
    top_X_test = X_test.loc[top_indices]
    top_y_test = y_test.loc[top_indices]
    top_y_pred = y_pred.loc[top_indices]

    result = {
        "X_test": top_X_test,
        "y_test": top_y_test,
        "y_pred": top_y_pred,
        "resid": np.abs(resid).nsmallest(top_k),
    }
    return result

def worst_cases(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: pd.Series,
    top_k: int = 10,
    mask: Optional[pd.Series] = None,
    func: Optional[str] = None,
) -> dict:
    """Return top-k worst cases according to the given function"""
    # Вычисляем остатки в зависимости от выбранной функции
    if func == "squared_errors":
        resid = residuals.squared_errors(y_test, y_pred)
    elif func == "logloss":
        resid = residuals.logloss(y_test, y_pred)
    elif func == "ape":
        resid = residuals.ape(y_test, y_pred)
    else:
        resid = residuals.residuals(y_test, y_pred)

    # Применяем маску, если она предоставлена
    if mask is not None:
        resid = resid[mask]
        X_test = X_test[mask]
        y_test = y_test[mask]
        y_pred = y_pred[mask]


    # Отсортировываем остатки и выбираем топ-K худших случаев
    top_indices = np.abs(resid).nlargest(top_k).index

    # Получаем соответствующие записи из X_test, y_test и y_pred
    top_X_test = X_test.loc[top_indices]
    top_y_test = y_test.loc[top_indices]
    top_y_pred = y_pred.loc[top_indices]

    result = {
        "X_test": top_X_test,
        "y_test": top_y_test,
        "y_pred": top_y_pred,
        "resid": np.abs(resid).nlargest(top_k),
    }
    return result


# # Чтение данных из файлов Excel
# # Загрузка данных из файлов
# x_test = pd.read_excel('x_test.xlsx')
# y_pred = pd.read_excel('y_pred.xlsx')
# y_test = pd.read_excel('y_test.xlsx')
# mask = pd.read_excel('mask.xlsx')

# # Преобразование столбцов к нужному типу данных (если необходимо)
# x_test['age'] = x_test['age'].astype(float)
# x_test['sex'] = x_test['sex'].astype(float)
# # Продолжите преобразование для остальных столбцов

# y_pred = y_pred['target'].astype(float)
# y_test = y_test['target'].astype(float)

# # Преобразование столбца mask к булевому типу
# mask = mask['target'] == 'TRUE'

# # Вызываем функцию best_cases для демонстрации
# best_top_cases = best_cases(x_test, y_test, y_pred, top_k=5, func="residuals")
# print("Top 5 Best Cases (residuals):")
# print()
# print('X_test')
# print(best_top_cases['X_test'])
# print()
# print('y_test')
# print(best_top_cases['y_test'])
# print()
# print('y_pred')
# print(best_top_cases['y_pred'])
# print()
# print('resid')
# print(best_top_cases['resid'])

# # Вызываем функцию worst_cases для демонстрации
# worst_top_cases = worst_cases(x_test, y_test, y_pred, top_k=5, func="residuals")
# print("\nTop 5 Worst Cases (residuals):")
# print()
# print('X_test')
# print(worst_top_cases['X_test'])
# print()
# print('y_test')
# print(worst_top_cases['y_test'])
# print()
# print('y_pred')
# print(worst_top_cases['y_pred'])
# print()
# print('resid')
# print(worst_top_cases['resid'])



# #  Создаем фиктивные данные для демонстрации
# data = {
#     'Age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
#     'Income': [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000],
#     'Expenses': [20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000],
# }

# X_test = pd.DataFrame(data)
# y_test = pd.Series([75000, 85000, 95000, 105000, 115000, 125000, 135000, 145000, 155000, 165000], name='Target')
# y_pred = pd.Series([76000, 86000, 94000, 108000, 116000, 124000, 136000, 144000, 157000, 167000], name='Predicted')

# # Вызываем функцию best_cases для демонстрации
# best_top_cases = best_cases(X_test, y_test, y_pred, top_k=3)
# print("Top 5 Best Cases:")
# print(best_top_cases['X_test'])
# print(best_top_cases['y_test'])
# print(best_top_cases['y_pred'])
# print(best_top_cases['resid'])

# # Вызываем функцию worst_cases для демонстрации
# worst_top_cases = worst_cases(X_test, y_test, y_pred, top_k=5)
# print("\nTop 5 Worst Cases:")
# print(worst_top_cases['X_test'])
# print(worst_top_cases['y_test'])
# print(worst_top_cases['y_pred'])
# print(worst_top_cases['resid'])