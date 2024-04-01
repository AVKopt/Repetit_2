import time
from functools import wraps
from colorama import Fore, Style

class TimeitDecorator:
    """
    Класс для декоратора, который измеряет время выполнения функции.
    """

    def timeit(func):
        """
        Декоратор, который измеряет время выполнения функции.

        Параметры:
        - func: функция, для которой измеряется время выполнения

        Возвращает:
        - wrapper: обертка, которая измеряет время выполнения функции
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            """
            Обертка, которая измеряет время выполнения функции.

            Параметры:
            - args: позиционные аргументы функции
            - kwargs: именованные аргументы функции

            Возвращает:
            - result: результат выполнения функции
            """
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(
                f"{Fore.BLUE}{Style.BRIGHT}Время выполнения: {round(end_time - start_time, 2)} секунд(ы){Style.RESET_ALL}"
            )

            return result

        return wrapper