import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from colorama import Back, Fore, Style

from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    f1_score
)



class GraphicReport:
    """
    Класс для визуализации графических отчетов.

    Методы класса:
    - drow_confusion_matrix: метод получения масштабированных выборок.
    - drow_pr_curve: метод для построения pr-кривой и нахождения лучшего трешхолда.
    - plot_importance: метод для построения графика важности признаков.
    - get_preds (статический метод): статический метод для получения предсказаний.
    - get_pred_pro (статический метод): статический метод для получения предсказанных вероятностей.
    """

    def __init__(self):
        """
        Конструктор класса DatasetPreprocessor.
        """
        pass

    @staticmethod
    def get_preds(model, features):
        """
        Статический метод get_preds класса GraphicReport
        Метод для получения предсказаний модели.

        Параметры:
        - model (class object): модель для предсказаний.
        - features (pd.DataFrame): признаки, на основе которых делаются предсказания.

        Возвращает:
        - preds (np.array): предсказания модели.
        """
        preds = model.predict(features)
        return preds

    @staticmethod
    def get_pred_proba(model, features):
        """
        Статический метод get_pred_proba класса GraphicReport
        Метод для получения вероятностей предсказаний модели.

        Параметры:
        - model (class object): модель для предсказаний.
        - features (pd.DataFrame): признаки, на основе которых делаются предсказания.

        Возвращает:
        - pred_proba (np.array): предсказания модели.
        """
        pred_proba = model.predict_proba(features)[:, 1]
        return pred_proba

    def drow_confusion_matrix(self, model=None, features=None, target=None):
        """
        Метод drow_confusion_matrix класса GraphicReport.
        Метод для построения матрицы ошибок.

        Параметры:
        - model (class object): модель для предсказаний (по умолчанию None).
        - features (pd.DataFrame): признаки, на основе которых делаются предсказания (по умолчанию None).
        - target (pd.Series): целевой признак (по умолчанию None).
        Возвращает:
        - Нет возвращаемого значения
        """
        print(f"{Fore.GREEN}{Style.BRIGHT}Матрица ошибок{Style.RESET_ALL}")
        print()
        # получение предсказаний модели и построение confusion matrix
        preds = GraphicReport.get_preds(model, features)
        conf_matrix = confusion_matrix(target, preds)
        # создание графика с матрицей ошибок (confusion matrix)
        plt.figure(figsize=(12, 8))
        # определение классов и их значения
        classes = ["True Negative", "False Positive", "False Negative", "True Positive"]
        # форматирование значений и вычисление процентов
        values = ["{0:0.0f}".format(x) for x in conf_matrix.flatten()]
        percentages = [
            "{0:.1%}".format(x) for x in conf_matrix.flatten() / np.sum(conf_matrix)
        ]
        # комбинация значений для подписей в графике
        combined = [f"{i}\n{j}\n{k}" for i, j, k in zip(classes, values, percentages)]
        combined = np.asarray(combined).reshape(2, 2)
        # построение Heatmap с подписями - confusion matrix
        ax = sns.heatmap(conf_matrix, annot=combined, fmt="", cmap="YlGnBu")
        ax.set(title="Confusion Matrix")
        ax.set(xlabel="Predicted", ylabel="Actual")
        plt.show()
        # вычисление основных метрик качества модели
        TN = conf_matrix[0][0]
        FP = conf_matrix[0][1]
        FN = conf_matrix[1][0]
        TP = conf_matrix[1][1]
        total = TN + TP + FP + FN
        # вычисление основных метрик качества модели
        acc = round((TP + TN) / total, 4)
        recall = round(TP / (TP + FN), 4)
        precision = round(TP / (TP + FP), 4)

        print(
            f"{Fore.BLACK}{Style.BRIGHT}Всего предсказаний: {total}\n{Style.RESET_ALL}"
        )
        print(
            f"{Fore.GREEN}{Style.BRIGHT}Правильно предсказанные ответы (Accuracy): {acc}{Style.RESET_ALL}"
        )
        print(
            f"{Fore.RED}{Style.BRIGHT}Ошибки в прогнозах: {round(1 - acc, 4)}\n{Style.RESET_ALL}"
        )
        print(f"{Fore.BLUE}{Style.BRIGHT}Recall: {recall}{Style.RESET_ALL}")
        print(f"{Fore.BLUE}{Style.BRIGHT}Precision: {precision}\n{Style.RESET_ALL}")
        print(
            f"{Fore.BLACK}{Style.BRIGHT}1. По главной диагонали (от верхнего левого угла) выстроены правильные прогнозы:\n{Style.RESET_ALL}"
        )
        print(
            f"{Fore.GREEN}{Style.BRIGHT}  - TN в левом верхнем углу. {TN} правильных ответов.{Style.RESET_ALL}"
        )
        print(
            f"{Fore.GREEN}{Style.BRIGHT}  - TP в правом нижнем углу. {TP} правильных ответов.\n{Style.RESET_ALL}"
        )
        print(
            f"{Fore.BLACK}{Style.BRIGHT}2. Вне главной диагонали — ошибочные варианты:\n{Style.RESET_ALL}"
        )
        print(
            f"{Fore.RED}{Style.BRIGHT}  - FP в правом верхнем углу. {FP} ошибок ошибочных предсказаний первого рода.{Style.RESET_ALL}"
        )
        print(
            f"{Fore.RED}{Style.BRIGHT}  - FN в левом нижнем углу. {FN} ошибочных предсказаний второго рода.{Style.RESET_ALL}"
        )

    def drow_pr_curve(self, model=None, features=None, target=None, treshold=False):
        """
        Метод drow_pr_curve класса GraphicReport.
        Метод для построения pr-кривой и нахождения лучшего трешхолда.
        Используется для тренировочной выборки.

        Параметры:
        - model (class object): модель для предсказаний (по умолчанию None).
        - features (pd.DataFrame): признаки, на основе которых делаются предсказания (по умолчанию None).
        - target (pd.Series): целевой признак (по умолчанию None).
        - treshold (Boolean): флаг позволяющий отображать treshold на графике (по умолчанию None).

        Возвращает:
        - Нет возвращаемого значения
        """

        print(f"{Fore.GREEN}{Style.BRIGHT}PR-кривая{Style.RESET_ALL}")
        print()
        # получение вероятностей предсказаний целевого класса
        target_score = GraphicReport.get_pred_proba(model, features)
        # вычисление precision, recall и пороговых значений для PR-кривой
        precision, recall, thresholds = precision_recall_curve(target, target_score)
        # если указан параметр threshold, находим лучший порог
        if treshold:
            fscore = (2 * precision * recall) / (precision + recall)
            ix = np.argmax(fscore)
            best_treshold = thresholds[ix]
            print("Лучший Threshold = %.2f" % best_treshold)
        # построение PR-кривой
        fig, ax = plt.subplots()
        ax.plot(recall, precision, color="purple", label="Модель")
        # отображение лучшего порога на графике, если указан параметр threshold
        if treshold:
            ax.scatter(
                recall[ix],
                precision[ix],
                marker="o",
                color="black",
                label="Лучший Threshold",
            )
        # настройка графика и добавление подписей
        ax.set_title("Precision-Recall Curve")
        ax.set_ylabel("Precision")
        ax.set_xlabel("Recall")
        ax.legend()

        plt.show()

    def plot_importance(self, model=None, features=None, fig_size=(40, 20)):
        """
        Метод plot_importance GraphicReport.
        Метод для построения графика важности признаков.
        Используется для тренировочной выборки.

        Параметры:
        - model (class object): обученная модель (по умолчанию None).
        - features (pd.DataFrame): признаки, на основе которых делаются предсказания (по умолчанию None).
        - fig_size (tuple(int)): размер графика (по умолчанию (40, 20)).

        Возвращает:
        - Нет возвращаемого значения
        """

        # создаем датафрейм из значений и признаков
        feature_imp = pd.DataFrame(
            {"Value": model.feature_importances_, "Feature": features.columns}
        )
        # создаем barplot
        fig, ax = plt.subplots(figsize=fig_size)
        sns.barplot(
            x="Value",
            y="Feature",
            data=feature_imp.sort_values(by="Value", ascending=False),
            ax=ax,
        )
        # подписи к барам
        ax.bar_label(
            ax.containers[0],
            labels=feature_imp["Value"]
            .sort_values(ascending=False)
            .apply("{:.0f}".format),
            fontsize=24,
        )
        # заголовки
        ax.set_title("Важность признаков", fontsize=36)
        ax.set_xlabel("Значение", fontsize=30)
        ax.set_ylabel("Признак", fontsize=30)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        # вывод графика
        plt.tight_layout()
        plt.show()