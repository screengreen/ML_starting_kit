from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt


def evaluate_classification_model(y_true, y_pred, experiment):
    """
    Оценивает модель классификации и выводит различные метрики.

    Parameters:
    y_true (array-like): Истинные значения целевой переменной.
    y_pred (array-like): Предсказанные значения целевой переменной (бинарные).
    y_pred_prob (array-like, optional): Предсказанные вероятности принадлежности к классу 1 (для ROC AUC).

    Returns:
    None
    """
    # Вычисляем метрики
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Выводим метрики
    # print(f'Accuracy: {accuracy:.4f}')
    # print(f'Precision: {precision:.4f}')
    # print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    if experiment != 0:
        metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
        experiment.log_metrics(metrics, step=1)

    # # Если предоставлены вероятности, вычисляем ROC AUC
    # if y_pred_prob is not None:
    #     roc_auc = roc_auc_score(y_true, y_pred_prob)
    #     print(f'ROC AUC: {roc_auc:.4f}')

    

    # Матрица ошибок
    # confusion = confusion_matrix(y_true, y_pred)
    # print('\nConfusion Matrix:')
    # print(confusion)



def perform_cross_validation(model, X, y, scoring='accuracy', n_splits=5, random_state=None):
    """
    Выполняет кросс-валидацию модели.

    Parameters:
    model: Модель машинного обучения.
    X (array-like): Матрица признаков.
    y (array-like): Вектор целевой переменной.
    scoring (str, optional): Метрика для оценки производительности модели (по умолчанию 'accuracy').
    n_splits (int, optional): Количество фолдов в кросс-валидации (по умолчанию 5).
    random_state (int, optional): Зерно генератора случайных чисел для воспроизводимости (по умолчанию None).

    Returns:
    list: Список значений метрик для каждого фолда.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    return scores



def plot_roc_curve(fpr, tpr):
    """
    Визуализирует ROC-кривую.

    Parameters:
    fpr (array-like): Список значений false positive rate.
    tpr (array-like): Список значений true positive rate.

    Returns:
    None
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

