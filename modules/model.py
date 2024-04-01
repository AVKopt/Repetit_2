import sys
import numpy as np
import pandas as pd
from colorama import Back, Fore, Style

from sklearn.model_selection import StratifiedKFold, cross_val_score
# модели
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
# метрики
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    f1_score
)
# подбор гиперпараметров
import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
# отбор признаков
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
# балансировка
from crucio import SMOTENC, SMOTETOMEK

from time_decorator import TimeitDecorator

RANDOM = 12345

class ModelOptuna:
    """
    Класс для подбора гиперпараметров модели с Optuna, обучения моделей и отбора признаков.

    Методы класса:
    - get_model_with_best_params: метод получает модель на лучших гиперпараметрах.
    - models_report: метод выводит отчет с полученными метриками разных моделей.
    - get_test_results: метод обучает модель, делает предсказания с калибровкой и без неё.
    - sequential_feature_selection: метод отбирает лучшие признаки модели.
    - objective: метод поиска лучших гиперпараметров для модели.
    - convert_param (статический метод): статический метод конвертирует параметр в формат, подходящий для оптимизации с помощью Optuna.
    - get_model (статический метод): статический метод создает модель с параметрами, на основаниия полученного названия модели и словарей с ее гиперпараметрами.
    """

    def __init__(self, random_state=RANDOM):
        """
        Конструктор класса ModelOptuna.
        Параметры:
        - random_state (int): значение random_state.
        """
        self.RANDOM = random_state

    @staticmethod
    def convert_param(param_name, param_value, trial=None):
        """
        Статический метод convert_paramt класса ModelOptuna.
        Конвертирует параметр в формат, подходящий для оптимизации с помощью Optuna.

        Параметры:
        - param_name (str): имя параметра.
        - param_value (any): значение параметра
        - trial (optuna.Trial, optional): экземпляр класса optuna.Trial, используемый для оптимизации параметров.
                                          (по умолчанию None).

        Возвращает:
        - значение параметра, подходящее для оптимизации с помощью Optuna.

        """
        # проверка значения param_value на тип np.ndarray
        if isinstance(param_value, np.ndarray):
            # если тип np.ndarray, проверяем на тип np.integer
            if np.issubdtype(param_value.dtype, np.integer):
                # если тип np.integer возвращаем trial.suggest_int c именем гиперпараметра, стартовым и конечным значением поиска
                # при условии что значение param_value это np.ndarray
                return trial.suggest_int(param_name, param_value[0], param_value[-1])
            # если тип np.ndarray, проверяем на тип np.float
            elif np.issubdtype(param_value.dtype, np.float):
                # если значение параметра является массивом numpy и содержит числа с плавающей запятой, используем suggest_float
                # c именем гиперпараметра, стартовым и конечным значением поиска при условии что значение param_value это np.ndarray
                return trial.suggest_float(
                    param_name, param_value[0], param_value[-1], log=True
                )
        # проверка значения param_value на тип list
        elif isinstance(param_value, list):
            # если все в списке string
            if all(isinstance(item, str) for item in param_value):
                # используем suggest_categorical c именем гиперпараметра и списком строковых значений
                return trial.suggest_categorical(param_name, param_value)
            # если все в списке int
            elif all(isinstance(item, int) for item in param_value):
                # используем suggest_int
                # c именем гиперпараметра, стартовым и конечным значением поиска при условии что значение param_value это list
                return trial.suggest_int(param_name, param_value[0], param_value[-1])
            # если какое-нибудь значение в списке float
            elif any(isinstance(item, float) for item in param_value):
                # используем suggest_float
                # c именем гиперпараметра, стартовым и конечным значением поиска при условии что значение param_value это list
                return trial.suggest_float(
                    param_name, param_value[0], param_value[-1], log=True
                )
            # в других случаях возвращаем исходное значение
            else:
                return param_value

    @staticmethod
    def get_model(
        model_name=None,
        fixed_params=None,
        dynamic_or_best_params=None,
        trial=None,
        random=None,
    ):
        """
        Статический метод get_model класса ModelOptuna.
        Возвращает модель с параметрами, на основаниия полученного названия модели и словарей с ее гиперпараметрами.

        Параметры:
        - model_name (str): имя модели (по умолчанию None).
        - fixed_params (dict): словарь фиксированных параметров (по умолчанию None)
        - dynamic_or_best_params (dict): словарь динамических или лучших параметров (по умолчанию None).
        - trial (optuna.Trial): экземпляр класса optuna.Trial, используемый для оптимизации параметров (по умолчанию None).
        - random (int): значение random_state (по умолчанию None)

        Возвращает:
        - model: экземпляр модели с оптимизированными параметрами.

        """
        # имициализируем переменную со значением None, в которую будем записывать словарь с гиперпараметрами.
        final_params = None
        # проверка на наличие словаря с гиперпараметрами для подбора оптуной или словаря с лучшими гиперпараметрами
        if dynamic_or_best_params:
            # проверка нужен ли экземпляр класса optuna.Trial
            if trial:
                # создаем словарь через вызов статического метода convert_param для конвертации в формат, подходящий
                # для оптимизации с помощью Optuna.
                final_params = {
                    k: ModelOptuna.convert_param(k, v, trial=trial)
                    for k, v in dynamic_or_best_params.items()
                }
            # если optuna.Trial не нужен
            else:
                # в переменную сохраняем словарь с лучшими гиперпараметрами
                final_params = dynamic_or_best_params
        # проверка на наличие словаря с фиксированными гиперпараметрами
        if fixed_params:
            # если передан словарь только с фикированными гиперпараметрами и переменная final_params пустая
            if final_params is None:
                # присваиваем в переменную словарь с фикированными гиперпараметрами
                final_params = fixed_params
            # иначе
            else:
                # объединем словари final_params, полученный при dynamic_or_best_params с fixed_params
                final_params.update(fixed_params)
        # создаем модель согласно имени модели model_name. sys.modules - это словарь,
        # содержащий все загруженные модули в текущем исполняющемся скрипте. name здесь - это строка с именем модуля,
        # из которого мы хотим получить атрибут.  getattr - это встроенная функция Python, которая возвращает значение
        # атрибута объекта по его имени. obj - это модуль, из которого нужно получить атрибут, а attr - это строка с именем атрибута.
        model_class = getattr(sys.modules[__name__], model_name)
        # заполняем модель гиперпараметрами из словаря final_params если он не пустой
        model = (
            lambda p: model_class(**p, random_state=random)
            if p is not None
            else model_class(random_state=random)
        )(final_params)

        return model

    def objective(
        self,
        trial,
        model_name,
        features_train,
        target_train,
        fixed_params,
        dynamic_or_best_params,
        metric_for_cv,
    ):
        """
        Метод objective класса ModelOptuna.
        Метод поиска лучших гиперпараметров для модели.

        Параметры:
        - trial (optuna.Trial): экземпляр класса optuna.Trial, используемый для оптимизации параметров.
        - model_name (str): имя модели.
        - features_train (pd.DataFrame): признаки для облучения.
        - target_train (pd.Series, np.array): целевой признак для обучения.
        - fixed_params (dict): словарь фиксированных параметров.
        - dynamic_or_best_params (dict): словарь динамических или лучших параметров.
        - metric_for_cv (str): метрика для кросс-валидации.

        Возвращает:
        - metric (float)- значение метрики.
        """

        # создаем модель вызывая статический метод get_model, передавая в него имя модели, фиксированные гиперпараметры,
        # гиперпараметры для подбора, trial и random_state
        model = ModelOptuna.get_model(
            model_name=model_name,
            fixed_params=fixed_params,
            dynamic_or_best_params=dynamic_or_best_params,
            trial=trial,
            random=self.RANDOM,
        )

        # получение метрики на кроссвалидации
        skf = StratifiedKFold(n_splits=5, random_state=self.RANDOM, shuffle=True)

        metric = cross_val_score(
            model, features_train, target_train, cv=skf, scoring=metric_for_cv
        ).mean()
        # возврат метрики
        return metric

    @TimeitDecorator.timeit  # декоратор подсчета времени выполнения метода
    def get_model_with_best_params(
        self,
        n_trials=10,
        features_train=None,
        target_train=None,
        model_name=None,
        fixed_params=None,
        dynamic_or_best_params=None,
        metric_for_cv="roc_auc",
    ):
        """
        Метод get_model_with_best_params класса ModelOptuna.
        Метод получает модель на лучших гиперпараметрах.

        Параметры:

        - n_trials (int): количество тестов модели (по умолчанию 10).
        - features_train (pd.DataFrame): признаки для облучения (по умолчанию None).
        - target_train (pd.Series, np.array): целевой признак для обучения (по умолчанию None).
        - model_name (srt): название модели, например DecisionTreeClassifier (по умолчанию None).
        - fixed_params (dict): словарь с фиксироваными гиперпараметрами (по умолчанию None).
        - dynamic_or_best_params (dict): словарь динамических или лучших параметров (по умолчанию None).
        - metric_for_cv (str): метрика (по умолчанию "roc_auc").

        Возвращает:
        - model: модель на лучших гиперпараметрах.
        """

        # создание сэмплера
        pruner = HyperbandPruner()
        sampler = TPESampler(seed=self.RANDOM)
        # инициализация подбора гиперпараметров
        study = optuna.create_study(
            study_name=model_name, direction="maximize", pruner=pruner, sampler=sampler
        )
        # процесс подбора гиперпараметров
        study.optimize(
            lambda trial: self.objective(
                trial,
                model_name,
                features_train,
                target_train,
                fixed_params,
                dynamic_or_best_params,
                metric_for_cv,
            ),
            n_trials=n_trials,
        )

        # вывод на экран результатов
        print()
        print(
            f"{Fore.GREEN}{Style.BRIGHT}Лучшие гиперпараметры подобраные Optuna:{Style.RESET_ALL}",
            study.best_params,
        )
        print(
            f"{Fore.BLUE}{Style.BRIGHT}Усредненная метрика {metric_for_cv} модели на тренировочной выборке с кроссвалидацией: {study.best_value}{Style.RESET_ALL}"
        )
        # создаем модель вызывая статический метод get_model, передавая в него имя модели, фиксированные гиперпараметры,
        # лучшие подобранные гиперпараметры и random_state
        model = ModelOptuna.get_model(
            model_name=model_name,
            fixed_params=fixed_params,
            dynamic_or_best_params=study.best_params,
            random=self.RANDOM,
        )
        # возврат модели
        return model

    @TimeitDecorator.timeit  # декоратор подсчета времени выполнения метода
    def models_report(self, model_names=None, model_metrics=None):
        """
        Метод models_report класса ModelOptuna.
        Метод выводит отчет с полученными метриками разных моделей.

        Параметры:

        - model_names (list): список с моделями (по умолчанию None).
        - model_metrics (list): список с метриками (по умолчанию None).

        Возвращает:
        - Нет возвращаемого значения.
        """
        # создаем словарь из списков
        data_dict = dict(zip(model_names, model_metrics))
        # создание DataFrame из словаря
        result_df = (
            pd.DataFrame(data_dict, index=[0])
            .T.rename_axis("Models")
            .rename(columns={0: "Metrics"})
        )
        # вывод DataFrame
        display(result_df)

    @TimeitDecorator.timeit  # декоратор подсчета времени выполнения метода
    def get_test_results(
        self,
        model=None,
        features_train=None,
        target_train=None,
        features_test=None,
        target_test=None,
        metric="roc_auc",
    ):
        """
        Метод get_test_results класса ModelOptuna.
        Метод обучает модель, делает предсказания с калибровкой и без неё.

        Параметры:

        - model (object): модель для обучения (по умолчанию None).
        - features_train (pd.DataFrame): признаки для обучения (по умолчанию None).
        - target_train (pd.Series, np.array): целевой признак для обучения (по умолчанию None).
        - features_test (pd.DataFrame): признаки для теста (по умолчанию None).
        - target_test (pd.Series, np.array): целевой признак для теста (по умолчанию None).
        - metric (str): метрика (по умолчанию "roc_auc")

        Возвращает:
        - model: обученная модель.
        - preds (np.array): предсказания модели.
        - predicted_probalities_test (np.array): предсказанные вероятности моделью.
        - metric_test (float): метрика.
        """

        # обучение модели
        model.fit(features_train, target_train)
        # предсказанные вероятности
        predicted_probalities_test = model.predict_proba(features_test)[:, 1]
        # предсказания
        preds = model.predict(features_test)
        # присваиваем имени метода пустую строку
        metric_method = ""
        # захватываем метрику
        match metric:
            # если метрика связана с roc_auc, то это будет метод из sklearn.metrics  - roc_auc_score
            case "roc_auc" | "roc_auc_ovr" | "roc_auc_ovo" | "roc_auc_ovr_weighted" | "roc_auc_ovo_weighted" | "auc_roc":
                metric_method = "roc_auc_score"
            # если метрика связана с f1, то это будет метод из sklearn.metrics  - f1_score
            case "f1" | "f1_micro" | "f1_macro" | "f1_weighted" | "f1_samples":
                metric_method = "f1_score"
            # если метрика связана с accuracy, то это будет метод из sklearn.metrics  - accuracy_score
            case "accuracy":
                metric_method = "accuracy_score"
            # если метрика связана с balanced_accuracy, то это будет метод из sklearn.metrics  - balanced_accuracy_score
            case "balanced_accuracy":
                metric_method = "balanced_accuracy_score"
            # если метрика связана с top_k_accuracy, то это будет метод из sklearn.metrics  - top_k_accuracy_score
            case "top_k_accuracy":
                metric_method = "top_k_accuracy_score"
            # если метрика связана с average_precision, то это будет метод из sklearn.metrics  - average_precision_score
            case "average_precision":
                metric_method = "average_precision_score"
            # если метрика связана с precision, то это будет метод из sklearn.metrics  - precision_score
            case "precision":
                metric_method = "precision_score"
            # если метрика связана с recall, то это будет метод из sklearn.metrics  - recall_score
            case "recall":
                metric_method = "recall_score"
            # если метрика связана с matthews_corrcoe, то это будет метод из sklearn.metrics  - matthews_corrcoe
            case "matthews_corrcoef":
                metric_method = "matthews_corrcoef"
            # если метрика связана с kappa_score, то это будет метод из sklearn.metrics  - cohen_kappa_score
            case "cohen_kappa_score" | "cohen_kappa":
                metric_method = "cohen_kappa_score"
            # если нету метрики
            case _:
                raise ValueError(f"Нет метрики для подсчета.")
        # создаем объект metric_scorer согласно имени метода metric_method. sys.modules - это словарь,
        # содержащий все загруженные модули в текущем исполняющемся скрипте. name здесь - это строка с именем модуля,
        # из которого мы хотим получить атрибут.  getattr - это встроенная функция Python, которая возвращает значение
        # атрибута объекта по его имени. obj - это модуль, из которого нужно получить атрибут, а attr - это строка с именем атрибута.
        try:
            metric_scorer = getattr(sys.modules[__name__], metric_method)
        # если в импортах нету, то выводим ошибку.
        except:
            raise ValueError(
                f"Метод {metric_method} не определен. Нужно импортировать from sklearn.metrics import {metric_method}"
            )
        # захват метрики
        match metric:
            # если это roc_auc с "ovr" и "weighted", то вызываем соответсвующий скорер с нужными доп. параметрами
            case _ if all(item in metric for item in ["ovr", "weighted"]):
                metric_test = metric_scorer(
                    target_test,
                    predicted_probalities_test,
                    average="weighted",
                    multi_class="ovr",
                )
            # если это roc_auc с "ovo" и "weighted", то вызываем соответсвующий скорер с нужными доп. параметрами
            case _ if all(item in metric for item in ["ovo", "weighted"]):
                metric_test = metric_scorer(
                    target_test,
                    predicted_probalities_test,
                    average="weighted",
                    multi_class="ovo",
                )
            # если это roc_auc с "ovo", то вызываем соответсвующий скорер с нужными доп. параметрами
            case _ if "ovo" in metric:
                metric_test = metric_scorer(
                    target_test, predicted_probalities_test, multi_class="ovo"
                )
            # если это roc_auc с "ovr", то вызываем соответсвующий скорер с нужными доп. параметрами
            case _ if "ovr" in metric:
                metric_test = metric_scorer(
                    target_test, predicted_probalities_test, multi_class="ovr"
                )
            # если это просто roc_auc , то вызываем соответсвующий скорер без доп. параметров
            case _ if "roc" in metric:
                metric_test = metric_scorer(target_test, predicted_probalities_test)
            # аналогичто для f1
            case _ if all(item in metric for item in ["f1", "micro"]):
                metric_test = metric_scorer(target_test, preds, average="micro")
            case _ if all(item in metric for item in ["f1", "macro"]):
                metric_test = metric_scorer(target_test, preds, average="macro")
            case _ if all(item in metric for item in ["f1", "weighted"]):
                metric_test = metric_scorer(target_test, preds, average="weighted")
            case _ if all(item in metric for item in ["f1", "samples"]):
                metric_test = metric_scorer(target_test, preds, average="samples")
            # для остальных метрик
            case _:
                metric_test = metric_scorer(target_test, preds)

        print(
            f"{Fore.GREEN}{Style.BRIGHT}Метрика {metric.upper()} модели на тестовой выборке: {metric_test:.6f}{Style.RESET_ALL}"
        )
        print(classification_report(target_test, preds))

        return model, preds, predicted_probalities_test, metric_test

    @TimeitDecorator.timeit  # декоратор подсчета времени выполнения метода
    def sequential_feature_selection(
        self,
        model=None,
        features_train=None,
        target_train=None,
        k_features=None,
        forward=False,
        floating=False,
        verbose=2,
        scoring="roc_auc",
        cv=5,
        n_jobs=2,
        top_rows=3,
    ):
        """
        Метод sequential_feature_selection ModelOptuna.
        Метод отбирает лучшие признаки модели при помощи mlxtend SFS
        (https://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/).

        Sequential Forward Selection (SFS) - это метод пошагового выбора признаков, который начинает
        с пустого подмножества и поочередно добавляет признак с наилучшим критерием качества.
        Этот процесс продолжается до тех пор, пока не будет достигнуто определенное условие остановки,
        такое как количество выбранных признаков или улучшение качества модели (forward=True, floating=False).

        Sequential Backward Selection (SBS), напротив, начинает с полного набора признаков и постепенно
        удаляет признаки, которые вносят наименьший вклад в качество модели (forward=False, floating=False).

        Sequential Forward Floating Selection (SFFS) интегрирует идею SFS с дополнительным шагом, в
        котором некоторые признаки могут быть временно удалены после того, как они были добавлены на
        предыдущих шагах. Это помогает улучшить способность алгоритма к обнаружению наилучшего подмножества
        признаков (forward=True, floating=True).

        Sequential Backward Floating Selection (SBFS), с другой стороны, объединяет идеи SBS и SFFS,
        позволяя как удаление, так и добавление признаков на различных этапах алгоритма (forward=False, floating=True).

        Параметры:

        - model (object): модель для выбора признаков (по умолчанию None).
        - features_train (pd.DataFrame): признаки для облучения (по умолчанию None).
        - target_train (pd.Series, np.array): целевой признак для обучения (по умолчанию None).
        - k_features (int): количество признаков для отбора (по умолчанию None).
        - forward (Boolean): (по умолчанию False).
        - floating (Boolean): (по умолчанию False).
        - verbose (int): вывод промежуточного отчета (по умолчанию 2).
        - scoring (string): метрика, по которой будет отбор признаков (по умолчанию "roc_auc").
        - cv (int): параметр кроссвалидации (по умолчанию 5).
        - n_jobs (int): количество процессоров (по умолчанию 2).
        - top_rows (int): количество строк датасета для вывода отчета (по умолчанию 3).

        Возвращает:
        - final_features (list): список лучших признаков.
        """

        # инициализация SFS
        sfs = SFS(
            model,
            k_features=k_features,
            forward=forward,
            floating=floating,
            verbose=verbose,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
        )
        # запуск процессаотбора признаков
        sfs = sfs.fit(features_train, target_train)
        # создание датасета с отчетом из атрибута get_metric_dict()
        report = pd.DataFrame.from_dict(sfs.get_metric_dict()).T
        # сортировка по убыванию по наилучшей метрике
        report = report.sort_values(by="avg_score", ascending=False).reset_index(
            drop=True
        )
        report["num_features"] = report["feature_names"].apply(lambda x: len(x))
        # создание списка с лучшими признаками
        final_features = list(report["feature_names"].loc[0])
        # проверка параметра top_rows, если он больше либо равен длине датасета, то выводим весь отчет
        if top_rows >= len(report):
            display(report)
        # в противном случае только top_rows строк
        else:
            display(report.head(top_rows))

        return final_features

    @TimeitDecorator.timeit  # декоратор подсчета времени выполнения метода
    def oversampling(self, features=None, target=None, method=SMOTENC()):
        """
        Метод oversampling класса ModelOptuna.
        Метод для борьбы с дисбалансом классов.
        Используется Crucio — это пакет, для устранения дисбаланса классов.
        Он использует некоторые классические методы балансировки классов,
        принимая в качестве параметров фрейм данных и целевой столбец.
        - ADASYN.
        - ICOTE (Immune Centroids Oversampling).
        - MTDF (Mega-Trend Difussion Function).
        - MWMOTE (Majority Weighted Minority Oversampling Technique).
        - SMOTE (Synthetic Minority Oversampling Technique).
        - SMOTENC (Synthetic Minority Over-sampling Technique for Nominal and Continuous).
        - SMOTETOMEK (Synthetic Minority Oversampling Technique + Tomek links for undersampling).
        - SMOTEENN (Synthetic Minority Oversampling Technique + ENN for undersampling).
        - SCUT (SMOTE and Clustered Undersampling Technique).
        - SLS (Safe-Level-Synthetic Minority Over-Sampling TEchnique).
        - TKRKNN (Top-K ReverseKNN).
        https://github.com/SigmoidAI/crucio

        Параметры:

        - features (pd.DataFrame): признаки для балансировки (по умолчанию None).
        - target (pd.Series, np.array): целевой признак для балансировки (по умолчанию None).
        - method (object): метод для балансировки (по умолчанию SMOTENC())ю

        Возвращает:
        - features_train (pd.DataFrame): сбалансированные признаки.
        - target_train (pd.Series, np.array): сбалансированный таргет.
        """

        # создаем копию features
        features_to_oversampling = features.copy(deep=True)
        # присоединяем target
        features_to_oversampling["target"] = target
        # сброс индексов
        features_to_oversampling = features_to_oversampling.reset_index()
        # получаем сбалансированную выборку
        features_to_oversampling = method.balance(features_to_oversampling, "target")
        # удаляем "index", "target"
        features_train = features_to_oversampling.drop(["index", "target"], axis=1)
        # выделяем таргет
        target_train = features_to_oversampling["target"]

        return features_train, target_train