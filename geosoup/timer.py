import time
from functools import wraps


__all__ = ['Timer']


class Timer:
    """
    Decorator class to compute time a function takes to execute

    example usage:

    @Timer.timing(True)
    def my_function(a, b, c):
        return a + b + c

    >Time it took to run my_function: 0.44 seconds

    """
    def __init__(self,
                 func):
        """
        Constructor
        :param func: Input function
        """
        self.func = func

    def __repr__(self):
        """
        Object representation
        :return: String
        """
        return '<Timer class for {}>'.format(self.func.__name__)

    @staticmethod
    def display_time(seconds,
                     precision=3):
        """
        method to display time in human readable format
        :param seconds: Number of seconds
        :param precision: Decimal precision
        :return: String
        """

        # define denominations
        intervals = [('weeks', 604800),
                     ('days', 86400),
                     ('hours', 3600),
                     ('minutes', 60),
                     ('seconds', 1)]

        # initialize list
        result = list()

        # coerce to float
        dtype = type(seconds).__name__
        if dtype != 'int' or dtype != 'long' or dtype != 'float':
            try:
                seconds = float(seconds)
            except (TypeError, ValueError, NameError):
                print("Type not coercible to Float")

        # break denominations
        for name, count in intervals:
            if name != 'seconds':
                value = seconds // count
                if value:
                    seconds -= value * count
                    if value == 1:
                        name = name.rstrip('s')
                    value = str(int(value))
                    result.append("{v} {n}".format(v=value,
                                                   n=name))
            else:
                value = "{:.{p}f}".format(seconds,
                                          p=precision)
                result.append("{v} {n}".format(v=value,
                                               n=name))

        # join output
        return ' '.join(result)

    @classmethod
    def timing(cls,
               doit=False):
        """
        Function to compute timing for input function
        :param doit: (abbr: Do it.) Keyword to determine if the wrapper returns
                     the function with or without timing it
        :return: Function and prints time taken
        """

        def time_it(func):

            """
            Executes wrapper
            """
            @wraps(func)
            def wrapper(*args, **kwargs):
                """
                Wrapper for func
                :param args: Arguments
                :param kwargs: Key word arguments
                :return: Function func return
                """
                if doit:
                    t1 = time.time()
                    val = func(*args, **kwargs)
                    t2 = time.time()

                    # time to run
                    t = Timer.display_time(t2 - t1)

                    print("Time it took to run {}: {}\n".format(func.__name__,
                                                                t))
                    return val
                else:
                    return func(*args, **kwargs)
            return wrapper
        return time_it
