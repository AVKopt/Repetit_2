import os
import magic
import re
import pandas as pd
from colorama import Fore, Style

from time_decorator import TimeitDecorator


class GetDataframe:
    """
    Класс для работы с датасетами.

    Методы класса:
    - get_dataframe_from_feather : метод создает датасет из файла формата feather.
    - get_dataframe_info: метод выводит информацию о датасете и пропущенных значениях.
    - is_accessible (staticmethod): статический метод для проверки доступности файлов в режиме чтения.
    - check_type (staticmethod): статический метод для типа переменной на строковое значение и ненулевую длину.
    - get_file_type (staticmethod): статический метод определяет MIME тип файла.
    - load_dataset (staticmethod): статический метод создает датасет методо пандас на основе MIME типа файла.
    """

    def __init__(self, data_dir):
        """
        Конструктор класса GetDataframe.

        Параметры:
        - data_dir (str): директория с данными для датасета.
        """
        self.data_dir = data_dir

    @staticmethod
    def is_accessible(file, data_dir, mode="r"):
        """
        Статический метод is_accessible класса GetDataframe.
        Проверка, является ли файл в рабочей директории доступным для работы в предоставленном режиме.

        Параметры:
        - file (str): файл с данными для датасета,
        - data_dir (str): рабочая директория с файлами для датасета,
        - mode (str): режим доступа к файлу, по умолчанию 'r' - чтение.

        Возвращает:
        - Boolean: True или False.
        """
        try:
            f = open(os.path.join(data_dir, file), mode)
            f.close()
        except IOError:
            return False
        return True

    @staticmethod
    def check_type(file_to_check):
        """
        Статический метод check_type класса GetDataframe.
        Проверяет тип переменной на строковое значение и ненулевую длину.

        Параметры:
        - file_to_check: переменная для проверки.

        Возвращает:
        - Boolean: True если проверка пройдена.
        """
        if not isinstance(file_to_check, str):
            raise TypeError(f"Тип переменной должен быть str. Датасет не будет создан.")
        if len(file_to_check) == 0:
            raise ValueError(
                f"Имя файла не должно быть пустым. Датасет не будет создан."
            )
        else:
            return True

    @staticmethod
    def get_file_type(file):
        """
        Статический метод check_type класса GetDataframe.
        Определяет MIME тип файла.
        (требуется установка библиотеки python-magic)

        Параметры:
        - file (str): имя файла для определения типа файла.

        Возвращает:
        - file_mime_type (str): MIME тип файла
           (https://wp-kama.ru/id_8643/spisok-rasshirenij-fajlov-i-ih-mime-tipov.html).
        """
        # создаем объект Magic, который позволяет определять тип файла. Параметр mime=True
        # указывает, что необходимо получить не просто описание типа файла, а именно его MIME-тип.
        mime = magic.Magic(mime=True)
        # используем метод from_file для определения MIME-типа файла,
        # этот метод анализирует файл и возвращает его MIME-тип на основе содержимого.
        file_mime_type = mime.from_file(file)

        return file_mime_type

    @staticmethod
    def load_dataset(file_dir, file, **kwargs):
        """
        Статический load_dataset класса GetDataframe.
        Создает датасет методо пандас на основе MIME типа файла.

        Параметры
        - file_dir (str): пусть к файлу.
        - file (str): имя файла для датасета.
        - **kwargs: дополнительные именованные аргументы для передачи в pd.read_csv().

        Возвращает:
        - dataset (pd.Dataframe): созданный датасет или инфо о невозможности создания датасета.
        """

        # статическим методом get_file_type получам MIME тип файла
        file_mime_type = GetDataframe.get_file_type(os.path.join(file_dir, file))
        # оператором match принимаем тип файла file_mime_type для дальнейшего сравнения
        match file_mime_type:
            # проверяем условие на text или csv, если True, то используем read_csv
            case "text/csv":
                return pd.read_csv(os.path.join(file_dir, file), **kwargs)
            # проверяем условие на xls или xlsx, если True, то используем read_excel
            case "application/vnd.ms-excel" | "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                return pd.read_excel(os.path.join(file_dir, file), **kwargs)
            # проверяем условие на json, если True, то используем read_json
            case "application/json":
                return pd.read_json(os.path.join(file_dir, file), **kwargs)
            # проверяем условие на html, если True, то используем read_html
            case "text/html":
                return pd.read_html(os.path.join(file_dir, file), **kwargs)
            # проверяем условие для feather или parquet, у них тип octet-stream
            case "application/octet-stream":
                # далее принимаем само имя файла
                match file:
                    # если в расширении feather то используем read_feather
                    case _ if file.endswith(".feather"):
                        return pd.read_feather(os.path.join(file_dir, file), **kwargs)
                    # если в расширении parquet то используем read_parquet
                    case _ if file.endswith(".parquet"):
                        return pd.read_parquet(os.path.join(file_dir, file), **kwargs)
                    # если нету нужного расширения
                    case _:
                        return f"Неподдерживаемый формат файла"
            # если нету нужного типа файла
            case _:
                return f"Неподдерживаемый формат файла"

    @TimeitDecorator.timeit  # декоратор подсчета времени выполнения метода
    def get_dataframe_from_file(self, file=None, **kwargs):
        """
        Создает датасет из файла.
        Для вывода информации требуется пакет colorama,

        Параметры:
        - file (str): имя файла для создания датасета.
        - **kwargs: дополнительные именованные аргументы для передачи в pd.read_csv().
          (передаются в виде словаря, например: additional_args = {'sep': ';'}, как
           **additional_args)

        Возвращает:
        - dataset (pd.Dataframe): созданный датасет.
        """
        # вызов статического метода проверки типа переменной
        check_file = GetDataframe.check_type(file)
        # вызов статического метода проверки читаемости файла
        flag = GetDataframe.is_accessible(file, self.data_dir)
        # если оба True
        if all([check_file, flag]):
            # создаем датасет
            print(
                f"{Fore.RED}{Style.BRIGHT}Создаем датасет из файла {file}.{Style.RESET_ALL}"
            )
            # вызов статического метода load_dataset
            dataset = GetDataframe.load_dataset(self.data_dir, file, **kwargs)
            # присваиваем имя датасету в соответсвии с именем файла
            dataset.name = re.findall("(\w+)\.", file)[0]
            print(f"{Fore.GREEN}{Style.BRIGHT}Датасет создан!{Style.RESET_ALL}")
            return dataset
        # в противном случай вызываем ошибку
        else:
            raise ValueError(f"Файл {file} не найден в директории {self.data_dir}.")

    def get_dataframe_info(self, dataset=None, info=True, miss_values=True):
        """
        Выводит информацию о датасете и пропущенных значениях.

        Параметры:
        - dataset: датасет для анализа,
        - info (bool): выводить информацию о датасете (по умолчанию True),
        - miss_values (bool): выводить информацию о пропущенных значениях (по умолчанию True).
        """
        # если выбран флаг info, выводим информацию о датасете
        if info:
            print(
                f"{Fore.RED}{Style.BRIGHT}Общая информация по датасету{Style.RESET_ALL}"
            )
            dataset.info()
            print()
        # если выбран флаг miss_values, считаем пропуски в столбцах
        if miss_values:
            percentage_missing = (dataset.isna().sum() / len(dataset)) * 100
            dataset_miss = pd.DataFrame({"percent_missing": percentage_missing})
            dataset_miss = dataset_miss.loc[dataset_miss["percent_missing"] > 0]
            # если в датасете нету пропусков
            if len(dataset_miss) == 0:
                print(
                    f"{Fore.GREEN}{Style.BRIGHT}В датасете нет пропущенных значений.{Style.RESET_ALL}"
                )
            # если они есть
            else:
                print(
                    f"{Fore.RED}{Style.BRIGHT}Столбцы с пропущенными значениями:{Style.RESET_ALL}"
                )
                display(dataset_miss)