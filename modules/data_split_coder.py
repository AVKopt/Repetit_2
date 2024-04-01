import pandas as pd
import numpy as np
from colorama import Fore, Style
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, RobustScaler, StandardScaler, OneHotEncoder
from time_decorator import TimeitDecorator

RANDOM = 12345

class DataSplitCoder:
    """
    Класс для разбиения датасета на выборки и масштабирования.

    Методы класса:
    - get_coded_data: метод получения масштабированных выборок.
    - data_split (статический метод): статический метод разбивает датасет на выборки.
    """

    def __init__(self):
        """
        Конструктор класса DataSplitCoder.
        """
        pass

    @staticmethod
    def data_split(
        dataset=None,
        target_col=None,
        test_size=None,
        valid=False,
        test_valid_size=0.4,
        report=False,
    ):
        """
        Статический метод data_split класса DataSplitCoder.
        Метод разбивает датасет на выборки.

        Параметры:

        - dataset (pd.DataFrame): датасет с данными (по умолчанию None).
        - target_col (str): целевой признак (по умолчанию None).
        - test_size (float): размер тестовой выборки (по умолчанию None).
        - valid (Boolean): флаг для выбора возможности добавлять валидационную выборку (по умолчанию False).
        - test_valid_size (float): размер тестовой и валидационной выборки (по умолчанию None).
        - report (Boolean): флаг для выбора возможности вывода отчета по разбиению (по умолчанию False).

        Возвращает (результат разбиения на выборки):

        - Если valid=False, возвращаются только трейн и тест:

           features_train, target_train, features_test, target_test

        - Если valid=True, возвращаются трейн, валид и тест:

           features_train, target_train, features_valid, target_valid, features_test, target_test

        """

        # выделение признаков
        features = dataset.drop([target_col], axis=1)
        # выделение целевого признака
        target = dataset[target_col]
        # если нужна валидация - разбиваем на train, valid, test в соотношении 60-20-20
        if valid:
            features_train, features_test, target_train, target_test = train_test_split(
                features, target, test_size=test_valid_size, random_state=RANDOM
            )
            features_valid, features_test, target_valid, target_test = train_test_split(
                features_test, target_test, test_size=0.5, random_state=RANDOM
            )
            # если True
            if report:
                print(
                    f"{Fore.BLACK}{Style.BRIGHT}Тренировочная выборка: {Style.RESET_ALL}",
                    features_train.shape,
                    target_train.shape,
                )
                print(
                    f"{Fore.BLACK}{Style.BRIGHT}Валидационная выборка: {Style.RESET_ALL}",
                    features_valid.shape,
                    target_valid.shape,
                )
                print(
                    f"{Fore.BLACK}{Style.BRIGHT}Тестовая выборка: {Style.RESET_ALL}",
                    features_test.shape,
                    target_test.shape,
                )

            return (
                features_train,
                target_train,
                features_valid,
                target_valid,
                features_test,
                target_test,
            )

        # если не нужна валидация - разбиваем на train и test в соотношении 80-20
        else:
            features_train, features_test, target_train, target_test = train_test_split(
                features,
                target,
                test_size=test_size,
                random_state=RANDOM,
                stratify=target,
            )
            # если True
            if report:
                print(
                    f"{Fore.BLACK}{Style.BRIGHT}Тренировочная выборка: {Style.RESET_ALL}",
                    features_train.shape,
                    target_train.shape,
                )
                print(
                    f"{Fore.BLACK}{Style.BRIGHT}Тестовая выборка: {Style.RESET_ALL}",
                    features_test.shape,
                    target_test.shape,
                )

            return features_train, target_train, features_test, target_test

    @TimeitDecorator.timeit  # декоратор подсчета времени выполнения метода
    def get_coded_data(
        self,
        dataset=None,
        target_col=None,
        cols_to_drop=None,
        test_size=0.25,
        valid=False,
        test_valid_size=0.4,
        report=False,
        scaler=StandardScaler(),
        one_hot=False,
        cols_to_one_hot=None,
        categorical=False,
        encoder=OrdinalEncoder(),
    ):
        """
        Метод get_coded_data класса DataSplitCoder.
        Метод получения масштабированных выборок.

        Параметры:

        - dataset (pd.DataFrame): датасет с данными (по умолчанию None).
        - target_col (str): целевой признак (по умолчанию None).
        - cols_to_drop (list): список столбцов, которые можно дополнительно удалить (по умолчанию пустой список).
        - test_size (float): размер тестовой выборки (по умолчанию 0.25).
        - valid (Boolean): флаг для выбора возможности добавлять валидационную выборку (по умолчанию False).
        - test_valid_size (float): размер тестовой и валидационной выборки (по умолчанию 0.4).
        - report (Boolean): флаг для выбора возможности вывода отчета по разбиению (по умолчанию False).
        - scaler (class object): скалер для масштабирования числовых признаков (по умолчанию StandardScaler()).
        - one_hot (Boolean): флаг для выбора OHE (по умолчанию False).
        - cols_to_one_hot (list): столбцы для OHE (по умолчанию None).
        - categorical(Boolean): флаг для выбора возможность кодирования категориальных признаков (по умолчанию False).
        - encoder (class object): энкодер для кодирования категориальных признаков (по умолчанию OrdinalEncoder()).


        Возвращает (результат разбиения на выборки):

        - Если valid=False, возвращаются только трейн и тест:

           features_train, target_train, features_test, target_test

        - Если valid=True, возвращаются трейн, валид и тест:

           features_train, target_train, features_valid, target_valid, features_test, target_test

        """
        # вызов метода pd..get_dummies для OHE
        if one_hot:
            # кодируем данные
            one_hot_encoded = pd.get_dummies(
                dataset[cols_to_one_hot], drop_first=True
            ).astype("int")
            # объединение закодированных данных с исходным датафреймом
            dataset = pd.concat([dataset, one_hot_encoded], axis=1)
            # датасет с кодировкой OHE
            dataset = dataset.drop(cols_to_one_hot, axis=1)
        # вызов метода data_split для разбиения на выборки
        # если True, то разбиваем на train, valid и test
        if valid:
            (
                features_train,
                target_train,
                features_valid,
                target_valid,
                features_test,
                target_test,
            ) = DataSplitCoder.data_split(
                dataset,
                target_col=target_col,
                test_size=test_valid_size,
                report=report,
                valid=valid,
            )
        # если False, то разбиваем на train и test
        else:
            (
                features_train,
                target_train,
                features_test,
                target_test,
            ) = DataSplitCoder.data_split(
                dataset,
                target_col=target_col,
                test_size=test_size,
                report=report,
                valid=valid,
            )
        # проверка наличия списка удаляемых столбцов из выборок
        if cols_to_drop is not None:
            # удаление из трейна и теста нужных столбцов
            features_train = features_train.drop(cols_to_drop, axis=1)
            features_test = features_test.drop(cols_to_drop, axis=1)
            if valid:
                features_valid = features_valid.drop(cols_to_drop, axis=1)
        # список числовых признаков
        num_cols = features_train.select_dtypes(include=np.number).columns.tolist()
        # масштабирование числовых признаков
        if len(num_cols) > 0:
            # объявление шкалера и его обучение на трейне
            scaler.fit(features_train[num_cols])
            # масштабирование числовых признаков в трейне и тесте
            features_train[num_cols] = scaler.transform(features_train[num_cols])
            features_test[num_cols] = scaler.transform(features_test[num_cols])
            if valid:
                features_valid[num_cols] = scaler.transform(features_valid[num_cols])
        # кодирование категориальных признаков
        if categorical:
            # список категориальных признаков
            cat_cols = features_train.select_dtypes(exclude=np.number).columns.tolist()
            if len(cat_cols) > 0:
                if isinstance(encoder, OrdinalEncoder):
                    # объявление кодера и его обучение на трейне
                    encoder.fit(features_train[cat_cols])
                    # кодирование категориальных признаков в трейне и тесте
                    features_train[cat_cols] = encoder.transform(
                        features_train[cat_cols]
                    )
                    features_test[cat_cols] = encoder.transform(features_test[cat_cols])
                    if valid:
                        features_valid[cat_cols] = encoder.transform(
                            features_valid[cat_cols]
                        )
                elif isinstance(encoder, OneHotEncoder):
                    train_encoded = pd.DataFrame(
                        encoder.fit_transform(features_train[cat_cols])
                    )
                    test_encoded = pd.DataFrame(
                        encoder.transform(features_test[cat_cols])
                    )  # Используется transform, а не fit_transform

                    # Переименование столбцов One-Hot Encoding
                    encoded_columns = encoder.get_feature_names_out(cat_cols)
                    train_encoded.columns = encoded_columns
                    test_encoded.columns = encoded_columns

                    # Удаление столбцов из train и test наборов данных
                    features_train = features_train.drop(cat_cols, axis=1)
                    features_test = features_test.drop(cat_cols, axis=1)

                    # Объединение закодированных данных с исходными датафреймами
                    features_train = pd.concat([features_train, train_encoded], axis=1)
                    features_test = pd.concat([features_test, test_encoded], axis=1)

                    if valid:
                        valid_encoded = pd.DataFrame(
                            encoder.transform(features_valid[cat_cols])
                        )
                        valid_encoded.columns = encoded_columns
                        features_valid = features_valid.drop(cat_cols, axis=1)
                        features_valid = pd.concat(
                            [features_valid, valid_encoded], axis=1
                        )

        # возврат выборок и таргетов
        if valid:
            return (
                features_train,
                target_train,
                features_valid,
                target_valid,
                features_test,
                target_test,
            )
        return features_train, target_train, features_test, target_test