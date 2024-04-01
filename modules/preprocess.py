import gc
import numpy as np
import pandas as pd
from colorama import Back, Fore, Style
from functools import reduce

from time_decorator import TimeitDecorator
RANDOM = 12345

class Preprocessor:
    """
    Класс для предобработки датасета.

    Методы класса:
    - preprocess_dataset : метод для обработки датасетов генерации признаков.
    - optim_mem_types (staticmethod): статический метод оптимизатор памяти.
    """

    def __init__(self):
        """
        Конструктор класса Preprocessor.
        """
        pass

    @staticmethod
    def optim_mem_types(dataset):
        """
        Статический метод optim_mem_types класса DatasetPreprocessor.
        Перебирает все столбцы датафрейма и изменяет тип данных, чтобы
        уменьшить использование памяти.
        Параметры:
        - dataset (pd.DataFrame): исходный датасет.

        Возвращает:
        - dataset (pd.DataFrame): обработанный датасет.
        """

        print(
            f"{Fore.YELLOW}{Style.BRIGHT}Оптимизация типов и памяти.{Style.RESET_ALL}"
        )
        # глубокая копия датасета
        dataset_opt = dataset.copy(deep=True)
        # подсчет начального объема датасета
        start_mem = round(dataset_opt.memory_usage().sum() / 1024**2, 4)
        print(
            f"{Fore.BLACK}{Style.BRIGHT}Память занимаемая датасетом в ОП до обработки: {start_mem} MB.{Style.RESET_ALL}"
        )
        # цикл по колонкам в датасете
        for col in dataset_opt.columns:
            # проверка на то, что столбец числовой
            if dataset_opt[col].dtype.name not in [
                    "object", "category", "datetime"
            ]:
                # проверка на то, что в типе float после запятой только нули
                if dataset_opt[col].dtype.name[:5].lower() == "float":
                    # создаем временный датасет без пропусков
                    tmp = dataset_opt.loc[dataset_opt[col].notna()]
                    # список с масками, что после запятой нули
                    decimal_check = (tmp[col] % 1 == 0).tolist()
                    # если все значения списка True
                    if all(decimal_check):
                        # то переводи в тип Int64, который не критичен к nan
                        dataset_opt[col] = dataset_opt[col].astype("Int64")
                    # удаляем промежуточный датасет
                    del tmp
                # минимум и максимум в столбце
                c_min = dataset_opt[col].min()
                c_max = dataset_opt[col].max()
                #  проверка на тип int
                if dataset_opt[col].dtype.name[:3].lower() == "int":
                    # если тип int, то присваиваются значения в зависимости от диапазона от min до max
                    if (c_min >= np.iinfo(np.int8).min and
                            c_max <= np.iinfo(np.int8).max):
                        dataset_opt[col] = dataset_opt[col].astype(np.int8)
                    elif (c_min >= np.iinfo(np.int16).min and
                          c_max <= np.iinfo(np.int16).max):
                        dataset_opt[col] = dataset_opt[col].astype(np.int16)
                    elif (c_min >= np.iinfo(np.int32).min and
                          c_max <= np.iinfo(np.int32).max):
                        dataset_opt[col] = dataset_opt[col].astype(np.int32)
                    elif (c_min >= np.iinfo(np.int64).min and
                          c_max <= np.iinfo(np.int64).max):
                        dataset_opt[col] = dataset_opt[col].astype(np.int64)
                # # если тип float, то присваиваются значения в зависимости от диапазона от min до max
                else:
                    if (c_min >= np.finfo(np.float16).min and
                            c_max <= np.finfo(np.float16).max):
                        dataset_opt[col] = dataset_opt[col].astype(np.float16)
                    elif (c_min >= np.finfo(np.float32).min and
                          c_max <= np.finfo(np.float32).max):
                        dataset_opt[col] = dataset_opt[col].astype(np.float32)
                    else:
                        dataset_opt[col] = dataset_opt[col].astype(np.float64)
            # если тип не дата, то оставшиеся столбцы в тип object
            elif "datetime" not in dataset[col].dtype.name:
                dataset_opt[col] = dataset_opt[col].astype("object")

        # новый объем датасета и возможная экономия в процентах
        end_mem = round(dataset_opt.memory_usage().sum() / 1024**2, 4)
        economy = round(100 * (start_mem - end_mem) / start_mem, 2)
        print(
            f"{Fore.BLACK}{Style.BRIGHT}Память занимаемая датасетом в ОП после обработки: {end_mem} MB.{Style.RESET_ALL}"
        )

        print(
            f"{Fore.GREEN}{Style.BRIGHT}Экономия {economy}%.{Style.RESET_ALL}")
        # если удалось оптимизировать
        if start_mem - end_mem >= 0:
            print(
                f"{Fore.GREEN}{Style.BRIGHT}Удалось оптимизировать типы и память или выйти по нулям.{Style.RESET_ALL}"
            )
            # сбока мусора
            gc.collect()
            return dataset_opt
        # в противном случае
        else:
            print(
                f"{Fore.RED}{Style.BRIGHT}Не удалось оптимизировать типы и память, оставляем исходный датасет.{Style.RESET_ALL}"
            )
            # сбока мусора
            del dataset_opt
            gc.collect()
            return dataset

    @TimeitDecorator.timeit  # декоратор подсчета времени выполнения метода
    def preprocess_dataset(self,
                           dataset_lst=None,
                           teachers_agg_dict=None,
                           data_lessons_agg_dict=None):
        """
        Метод preprocess_dataset класса Preprocessor.
        Метод обрабатывает датасет и генерирует новые признаки.

        Параметры:
        - dataset_lst (list): список с датасетами для обработки (по умолчанию None).
        - teachers_agg_dict (dict): словарь со столбцами для агрегации по учителям (по умолчанию None).
        - data_lessons_agg_dict (dict): словарь со столбцами для агрегации по занятиям (по умолчанию None).

        Возвращает:
        - dataset (pd.DataFrame): финальный датасет.
        """

        # проходим циклом по списку с датасетами
        for dataset in dataset_lst:
            # фиксируем имя датасета
            match dataset.name:
            # если "orders", то создаем датасет orders
                case "orders":
                    orders = dataset.copy(deep=True)
                # если "teachers_info", то создаем датасет teachers_info
                case "teachers_info":
                    teachers_info = dataset.copy(deep=True)
                # если "teachers", то создаем датасет teachers
                case "teachers":
                    teachers = dataset.copy(deep=True)
                # если "teacher_prices", то создаем датасет teacher_prices
                case "teacher_prices":
                    teacher_prices = dataset.copy(deep=True)
                # если "lessons", то создаем датасет lessons
                case "lessons":
                    lessons = dataset.copy(deep=True)
                # если "lesson_course", то создаем датасет lesson_course
                case "lesson_course":
                    lesson_course = dataset.copy(deep=True)
                # в противном случае
                case _:
                    return f"Нету датасета для обработки"
        # переименовываем столбец "id" в "teacher_id"
        teachers = teachers.rename(columns={"id": "teacher_id"})
        # в датасете teacher_prices оставляем только некоторые столбцы
        teacher_prices = teacher_prices[[
            "teacher_id", "subject_id", "price", "price_external",
            "price_remote"
        ]]
        # переименовываем столбец "id" в "teacher_id"
        teachers_info = teachers_info.rename(columns={"id": "teacher_id"})
        # преобразуем столбцы в формат datetime
        teachers_info["birth_date"] = pd.to_datetime(
            teachers_info["birth_date"])
        teachers_info["teaching_start_date"] = pd.to_datetime(
            teachers_info["teaching_start_date"])
        # вычисляем возраст и стаж учителя
        teachers_info["teacher_age"] = (pd.to_datetime("now").year -
                                        teachers_info["birth_date"].dt.year)
        teachers_info["teacher_experience"] = (
            pd.to_datetime("now").year -
            teachers_info["teaching_start_date"].dt.year)
        # в датасете teachers_info оставляем только некоторые столбцы
        teachers_info = teachers_info[[
            "teacher_id", "teacher_age", "teacher_experience",
            "is_email_confirmed", "lesson_duration", "lesson_cost",
            "is_display", "is_cell_phone_confirmed", "area_id", "review_num"
        ]]
        # создаем список с датасетами для объединения по учителям 
        frames = [teachers, teacher_prices, teachers_info]
        # объединияем датасеты по teacher_id 
        data_teachers = reduce(
            lambda left, right: pd.merge(
                left, right, on=['teacher_id'], how='left'), frames)
        # заменяем пропуски на -1
        data_teachers = data_teachers.fillna(-1)
        # делаем датасет с аггрегированными признаками из словаря teachers_agg_dict
        teachers_agg = data_teachers.groupby("teacher_id").agg(
            **teachers_agg_dict).reset_index()
        # создаем список с уникальными teacher_id, у которых есть целевая переменная
        teacher_id_list = list(teachers_agg["teacher_id"].unique())
        # для занятий оставляем только те записи, у которых есть целевой признак
        lesson_course = lesson_course.loc[lesson_course["teacher_id"].isin(
            teacher_id_list)]
        # переименовываем столбцы
        lessons = lessons.rename(columns={
            "lesson_course_id": "client_id",
            "id": "lesson_id"
        })
        # объединяем lesson_course и lessons  по "client_id"
        data_lessons = pd.merge(lesson_course,
                                lessons,
                                on="client_id",
                                how="left")
        # удаляем не нужные столбцы
        data_lessons = data_lessons.drop([
            "id", "lesson_id", "time_from", "time_to", "home_task",
            "lesson_place", "date_updated", "suspend_till_date"
        ],
                                         axis=1)
        # переименовываем столец "id" в "order_id" в orders
        orders = orders.rename(columns={"id": "order_id"})
        # оставляем только эти столбцы в orders
        orders = orders[["order_id", "status_id", "planned_lesson_number"]]
        # объединяем data_lessons и orders по "order_id",
        data_lessons = pd.merge(data_lessons, orders, on="order_id", how="left")
        # присваиваем "lesson_date" тип "object"
        data_lessons["lesson_date"] = data_lessons["lesson_date"].astype(
            "object")
        # заполняем пропуски нулями
        data_lessons[[
            "is_regular", "amount_to_pay", "amount_paid", "lesson_date"
            ]] = data_lessons[[
            "is_regular", "amount_to_pay", "amount_paid", "lesson_date"
            ]].fillna(0)
        # удаляем пропуски
        data_lessons = data_lessons.dropna()
        # создаем признак "diff_amounts" как сравнение "amount_paid" и "amount_to_pay"
        # 1 если "amount_paid" < "amount_to_pay" - иначе 0
        data_lessons["diff_amounts"] = data_lessons.apply(
            lambda x: 1 if x["amount_paid"] < x["amount_to_pay"] else 0, axis=1)
        # создаем признак "diff_amount_paid_status". 1 если amount_paid" == 0
        # и какой-то из статусов равен 6, 13 или 14, то есть оплачено -
        # иначе 0.
        data_lessons["diff_amount_paid_status"] = data_lessons.apply(
            lambda x: 1
            if (x["amount_paid"] == 0) and (x["status_id"] == 6 or x[
                "status_id"] == 13 or x["status_id"] == 14) else 0,
            axis=1)
        # создаем датасет с агрегированными признаками из словаря data_lessons_agg_dict
        data_lessons_agg = data_lessons.groupby("teacher_id").agg(
            **data_lessons_agg_dict).reset_index()
        # объединяем data_lessons_agg и teachers_agg по "teacher_id"
        dataset = pd.merge(data_lessons_agg,
                           teachers_agg,
                           on="teacher_id",
                           how="left")
        # очисктка мусора
        del orders, teachers_info, teachers, teacher_prices, lessons, lesson_course 
        del frames, data_teachers, data_lessons, teachers_agg, data_lessons_agg
        gc.collect()
        if "pupils_unique" in list(dataset.columns):
            dataset["avg_money_per_pupil"] = dataset["money_recieved"] / dataset["pupils_unique"]
        # # оптимизируем типы данных при помощи optim_mem_types класса Preprocessor
        dataset = Preprocessor.optim_mem_types(dataset)
        # присваиваем столбцам категориальный тип
        dataset[["diff_amounts", "diff_amount_paid_status",
                 "area_id"]] = dataset[[
                     "diff_amounts", "diff_amount_paid_status", "area_id"
                 ]].astype("category")

        return dataset