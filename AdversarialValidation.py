from typing import Optional
import pandas as pd
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
import residuals

def adversarial_validation(
    classifier: ClassifierMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: pd.Series,
    quantile: float = 0.1,
    func: Optional[str] = None,
) -> dict:
    """Adversarial validation residual analysis"""

    # Вычисляем остатки в зависимости от выбранной функции
    if func == "squared_errors":
        resid = residuals.squared_errors(y_test, y_pred)
    elif func == "logloss":
        resid = residuals.logloss(y_test, y_pred)
    elif func == "ape":
        resid = residuals.ape(y_test, y_pred)
    elif func == "quantile_loss":
        resid = residuals.quantile_loss(y_test, y_pred)
    else:
        resid = residuals.residuals(y_test, y_pred)

    # # Вычисляем квантиль
    # threshold = np.quantile(np.abs(resid), quantile)

    # # Создаем маску для адверсарной валидации
    # adversarial_mask = np.abs(resid) >= threshold

    # # Обучаем классификатор на тестовых данных
    # classifier.fit(X_test, adversarial_mask)
    # y_pred_proba = classifier.predict_proba(X_test)[:, 1]

    top_k = np.floor(len(X_test) * quantile).astype(int)
    top_indices = np.abs(resid).nlargest(top_k).index
    zeros_series = pd.Series(np.zeros(len(X_test)), index=X_test.index)
    zeros_series[top_indices] = 1

    classifier.fit(X_test, zeros_series)
    y_pred_proba = classifier.predict_proba(X_test)[:, 1]

    # Оценка ROC-AUC на полной выборке
    roc_auc = roc_auc_score(zeros_series, y_pred_proba)


    # Вычисляем важность признаков
    feature_importances = None
    if hasattr(classifier, "feature_importances_"):
        feature_importances = pd.Series(np.abs(classifier.feature_importances_), index=X_test.columns)
    elif hasattr(classifier, "coef_"):
        feature_importances = pd.Series(np.abs(classifier.coef_[0]), index=X_test.columns)

    result = {
        "ROC-AUC": roc_auc,
        "feature_importances": feature_importances,
    }

    return result


# Чтение данных из файлов Excel
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

# # clf = RandomForestClassifier(n_estimators=100, random_state=42)

# # # Вызываем функцию best_cases для демонстрации
# # best_top_cases = adversarial_validation(clf, x_test, y_test, y_pred)

# # print('ROC-AUC', best_top_cases['ROC-AUC'])
# # print('feature_importances-AUC', best_top_cases['feature_importances'])

# # Test with RandomForestClassifier
# clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
# result_rf = adversarial_validation(clf_rf, x_test, y_test, y_pred)
# print("RandomForestClassifier:")
# print('ROC-AUC', result_rf['ROC-AUC'])
# print('Feature Importances:')
# print(result_rf['feature_importances'])

# # Test with Logistic Regression
# clf_lr = LogisticRegression()
# result_lr = adversarial_validation(clf_lr, x_test, y_test, y_pred)
# print("\nLogistic Regression:")
# print('ROC-AUC', result_lr['ROC-AUC'])
# print('Feature Importances:')
# print(result_lr['feature_importances'])

# # Test with Support Vector Classifier
# clf_svc = SVC(probability=True)
# result_svc = adversarial_validation(clf_svc, x_test, y_test, y_pred)
# print("\nSupport Vector Classifier:")
# print('ROC-AUC', result_svc['ROC-AUC'])
# print('Feature Importances:')
# print(result_svc['feature_importances'])

# # Test with GradientBoostingClassifier
# clf_gb = GradientBoostingClassifier()
# result_gb = adversarial_validation(clf_gb, x_test, y_test, y_pred)
# print("\nGradient Boosting Classifier:")
# print('ROC-AUC', result_gb['ROC-AUC'])
# print('Feature Importances:')
# print(result_gb['feature_importances'])