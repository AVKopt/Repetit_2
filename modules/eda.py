import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from colorama import Back, Fore, Style
# корреляция
import phik
from phik.report import plot_correlation_matrix
# статистики
from scipy.stats import kurtosis, skew
from time_decorator import TimeitDecorator

class CustomEDA:
    """
    Класс для кастомного Exploratory Data Analysis (EDA).

    Методы класса:
    - custom_describe: cоздает кастомную описательную статистику для числовых признаков в датасете.
    - hist_and_box: метод для отображения гистограмм и ящиков с усами для числовых признаков датасета.
    - target_countplot: метод для построения barplot'а для подсчета баланса целевого признака.
    - phik_correlation: метод для вычисления и визуализации корреляции Phik между признаками.
    - pairplot: метод для построения pairplot между числовыми признаками.
    - color_negative_red (staticmethod): статический метод  для определения цвета текста в зависимости от значения.
    """

    def __init__(self):
        """
        Конструктор класса CustomEDA.
        """
        pass

    @staticmethod
    def color_negative_red(value):
        """
        Статический метод color_negative_red класса CustomEDA.
        Метод для определения цвета текста в зависимости от значения. Отрицательные значения - красный, положительные - зеленый, ноль - белый.

        Параметры:
        - value (float): значение, для которого необходимо определить цвет.

        Возвращает:
            str: CSS стиль для цвета текста.
        """
        if float(value) < 0:
            color = "#ff0000"  # касный цвет
        elif float(value) > 0:
            color = "#00ff00"  # зеленый цвет
        else:
            color = "#FFFFFF"  # Белый цвет

        return "color: %s" % color

    @TimeitDecorator.timeit  # декоратор подсчета времени выполнения метода
    def custom_describe(self, dataset=None):
        """
        Метод custom_describe класса CustomEDA.
        Метод для создания кастомной описательной статистики для числовых признаков в датасете.

        Параметры:
        - dataset (pd. Dataframe): исходный датафрейм Pandas (по умолчанию None).

        Возвращает:
        - Нету возвращаемого значения.
        """

        # вычисление статистик для числовых признаков
        print(f"{Fore.BLUE}{Style.BRIGHT}Описательная статистика.{Style.RESET_ALL}")
        print()
        num_cols = dataset.select_dtypes(include=np.number).columns.tolist()
        summary_data = pd.DataFrame(
            {
                col: dataset[col].agg(
                    [
                        lambda x: x.count(),
                        lambda x: x.values.mean(),
                        lambda x: x.median(),
                        lambda x: x.values.std(),
                        lambda x: x.min(),
                        lambda x: x.quantile(0.25),
                        lambda x: x.median(),
                        lambda x: x.quantile(0.75),
                        lambda x: x.max(),
                    ]
                )
                for col in num_cols
            }
        )
        # транспонируем датафрейм
        summary_data = summary_data.T
        # рассчет асимметрии и эксцесса числовых признаков
        skew_ = (
            dataset._get_numeric_data()
            .dropna()
            .apply(lambda x: skew(x))
            .to_frame(name="skewness")
        )
        kurt_ = (
            dataset._get_numeric_data()
            .dropna()
            .apply(lambda x: kurtosis(x))
            .to_frame(name="kurtosis")
        )
        skew_kurt = pd.concat([skew_, kurt_], axis=1)
        # объединяем в единый датасет
        summary_data = pd.concat([summary_data, skew_kurt], ignore_index=True, axis=1)
        # названия столбцов
        summary_data.columns = [
            "count",
            "mean",
            "median",
            "std",
            "min",
            "25%",
            "50%",
            "75%",
            "max",
            "skewness",
            "kurtosis",
        ]
        # формат вывода значений столбцов
        summary_data = summary_data.applymap(lambda x: "{:.2f}".format(x))
        # асимметрия и эксцесс в отдельный список для выделения цветов при помощи статического метода color_negative_red
        info_cols = ["skewness", "kurtosis"]
        display(
            summary_data.style.background_gradient(
                cmap="Spectral", subset=summary_data.columns[:-2]
            )
            .applymap(CustomEDA.color_negative_red, subset=info_cols)
            .set_properties(
                **{"background-color": "#000000", "font-weight": "bold"},
                subset=info_cols,
            )
            .set_properties(**{"font-weight": "bold"}, subset=summary_data.columns[:-2])
        )

    @TimeitDecorator.timeit  # декоратор подсчета времени выполнения метода
    def hist_and_box(self, dataset=None, target=None, cols_to_drop=None):
        """
        Метод hist_and_box класса CustomEDA.
        Метод для отображения гистограмм и ящиков с усами для числовых признаков датасета.

        Параметры:
        - dataset (pd. Dataframe): исходный датафрейм Pandas (по умолчанию None).
        - target (str): название целевого признака (по умолчанию None).
        - cols_to_drop

        Возвращает:
        - Нету возвращаемого значения.
        """

        print(
            f"{Fore.YELLOW}{Style.BRIGHT}Гистограммы и ящики с усами.{Style.RESET_ALL}"
        )
        print()
        # создание копии исходного датасета
        data = dataset.copy(deep=True)
        if cols_to_drop is not None:
            data = data.drop(cols_to_drop, axis=1)
        # создание списка числовых признаков и удаление оттуда таргета
        num_cols = data.select_dtypes(include=np.number).columns.tolist()
        num_cols.remove(target)
        # преобразование float16 в float32
        float16_columns = data.columns[data.dtypes == np.float16]
        data[float16_columns] = data[float16_columns].astype(np.float32)
        # построение гистограмм и boxplot'ов для каждой числовой колонки
        for col in num_cols:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            # гистограмма
            ax1.hist(data[col], bins=50, color="skyblue", edgecolor="black")
            ax1.set_xlabel("Значения")
            ax1.set_ylabel("Частота")
            ax1.set_title("Гистограмма")

            box_dict = {
                "boxprops": dict(color="#000000", linewidth=2),
                "capprops": dict(color="#000000", linewidth=1.5),
                "medianprops": dict(color="#000000", linewidth=1.5),
                "whiskerprops": dict(color="#000000", linewidth=1.5),
                "flierprops": dict(markeredgecolor="#ff9900"),
                "meanprops": dict(markeredgecolor="#000000"),
            }
            # построение ящика с усами с использованием параметра hue
            data.boxplot(
                by=target,
                column=[col],
                widths=0.5,
                showmeans=True,
                patch_artist=True,
                vert=False,
                **box_dict,
                ax=ax2,
            )
            ax2.set_xlabel("Данные")
            ax2.set_ylabel("Значения")
            ax2.set_title("Boxplot")

            boxes = ax2.findobj(matplotlib.artist.Artist)
            # цветовая палитра для boxplot'ов в зависимости от типа данных в целевом признаке
            if data[target].dtypes == "O":
                for i, box in enumerate(boxes):
                    if isinstance(box, matplotlib.patches.PathPatch):
                        if i < 3:
                            box.set_facecolor("#ea4b33")
                        if i > 3:
                            box.set_facecolor("#3490d6")
            else:
                for i, box in enumerate(boxes):
                    if isinstance(box, matplotlib.patches.PathPatch):
                        if i < 3:
                            box.set_facecolor("#3490d6")
                        if i > 3:
                            box.set_facecolor("#ea4b33")

            fig.suptitle(f"Гистограмма и ящик с усами для {col}", size=20, y=1.02)

            # Отображение графика
            plt.show()

    @TimeitDecorator.timeit  # декоратор подсчета времени выполнения метода
    def target_countplot(self, dataset=None, target_name=None):
        """
        Метод target_countplot класса CustomEDA.
        Метод для построения barplot'а для подсчета баланса целевого признака.

        Параметры:
        - dataset (pd. Dataframe): исходный датафрейм Pandas (по умолчанию None).
        - target_name (str): имя целевого признака (по умолчанию None).
        
        Возвращает:
        - Нету возвращаемого значения.
        """

        print(f"{Fore.GREEN}{Style.BRIGHT}Баланс целевого признака{Style.RESET_ALL}")
        print()
        # создание графика barplot для целевого признака
        plt.figure(figsize=(9, 5))
        plt.title("Целевой признак", fontsize=18)
        # построение barplot'а
        ax = sns.barplot(
            x=dataset[target_name].unique(),
            y=dataset[target_name].value_counts(ascending=True, normalize=True).values
            * 100,
            palette=["green", "red"],
        )

        # добавление дополнительной информации к графику
        abs_values = dataset[target_name].value_counts(ascending=False)
        rel_values = (
            dataset[target_name].value_counts(ascending=False, normalize=True).values * 100
        )
        lbls = [f"{p[0]} ({p[1]:.2f}%)" for p in zip(abs_values, rel_values)]
        ax.bar_label(container=ax.containers[0], labels=lbls)
        plt.xlabel(f"Целевой признак {target_name}", fontsize=15)
        plt.ylabel("Процентное соотношение", fontsize=15)
        plt.xticks(ticks=[0, 1], labels=["Не обманывает", "Обманывает"])

        plt.show()

    @TimeitDecorator.timeit  # декоратор подсчета времени выполнения метода
    def phik_correlation(self, dataset=None, cols_to_drop=None, multi_collinear=False, target_name=None):
        """
        Метод phik_correlation класса CustomEDA.
        Метод для вычисления и визуализации корреляции Phik между признаками.

        Параметры:
        - dataset (pd. Dataframe): исходный датафрейм Pandas (по умолчанию None).
        - cols_to_drop (list): список столбцов для удаления (по умолчанию None).
        - multi_collinear (Boolean): флаг для выбора фильтрации признаков в мультиколлинеарностью
                                     (по умолчанию False).
        - target_name (str): имя целевого признака (по умолчанию None).

        Возвращает:
        - dataset (pd. Dataframe): датасет с отфильтрованными признаками.
        """
        # копия датасета
        dataset_copy = dataset.copy(deep=True)
        # если есть столбцы для удаления - удаляем
        if cols_to_drop is not None:
            dataset_copy = dataset_copy.drop(cols_to_drop, axis=1)
        # список столбцов
        columns = list(dataset_copy.columns)
        # матрица корреляции
        phik_overview = dataset_copy.phik_matrix(interval_cols=columns)

        # фильтрация мультиколлениарных признаков
        if multi_collinear:
            to_drop = (
                set()
            )  # cоздаем множество для хранения названий признаков, которые нужно удалить
            # в цикле определяем признаки у которых взаимная корреляция больше 0.8 и оставляем тот из них,
            # у которого корреляция с таргетом наибольшая
            for i in range(len(phik_overview.columns)):
                for j in range(i + 1, len(phik_overview.columns)):
                    if abs(phik_overview.iloc[i, j]) > 0.8:
                        feature_i = phik_overview.columns[i]
                        feature_j = phik_overview.columns[j]
                        if (
                            phik_overview.loc[feature_i, target_name]
                            < phik_overview.loc[feature_j, target_name]
                        ):
                            to_drop.add(feature_i)
                        else:
                            to_drop.add(feature_j)
            # удаляем признаки с высокой корреляцией с целевым признаком
            phik_overview = phik_overview.drop(list(to_drop))
            phik_overview = phik_overview.drop(list(to_drop), axis=1)

        # визуализация матрицы корреляции
        plot_correlation_matrix(
            phik_overview.values,
            x_labels=phik_overview.columns,
            y_labels=phik_overview.index,
            vmin=0,
            vmax=1,
            color_map="Greens",
            title=r"Корреляция $\phi_K$",
            fontsize_factor=1,
            figsize=(26, 16),
        )
        plt.tight_layout()
        # обновление датасета с отфильтрованными признаками
        final_cols = list(phik_overview.columns)
        if cols_to_drop is not None:
            final_cols.extend(cols_to_drop)
        dataset = dataset[final_cols]

        return dataset

    @TimeitDecorator.timeit  # декоратор подсчета времени выполнения метода
    def pairplot(self, dataset=None, cols_to_drop=None, target_col=None):
        """
        Метод pairplot класса CustomEDA.
        Метод для построения pairplot между числовыми признаками.

        Параметры:
        - dataset (pd. Dataframe): исходный датафрейм Pandas (по умолчанию None).
        - cols_to_drop (list): список для удаления категориальных столбцов (по умолчанию None).
        - target_col (str): целевая переменная (по умолчанию None).

        Возвращает:
        - Нету возвращаемого значения.
        """

        print(f"{Fore.GREEN}{Style.BRIGHT}График Pairplot{Style.RESET_ALL}")
        print()
        # выделяем датасет с числовыми признаками
        if cols_to_drop is not None:
            dataset_num = dataset.drop(cols_to_drop, axis=1)
        else: 
            dataset_num = dataset.copy(deep=True)
        # строим pairplot
        sns.pairplot(dataset_num, hue=target_col)
        plt.show()