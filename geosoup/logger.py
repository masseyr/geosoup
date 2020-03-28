import logging


class Logger(object):
    """
    Class for logging output
    """

    def __init__(self,
                 name='default',
                 filename=None,
                 level='info',
                 stream=False):
        """
        Instantialize logger class object
        """

        self.logger = logging.getLogger(name)

        if level.lower() == 'debug':
            self.logger.setLevel(logging.DEBUG)
        elif level.lower() == 'info':
            self.logger.setLevel(logging.INFO)
        elif level.lower() == 'warn' or level.lower() == 'warning':
            self.logger.setLevel(logging.WARNING)
        else:
            self.logger.setLevel(logging.ERROR)

        # create a logging format
        formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')

        # file handler
        if filename is not None:
            file_handler = logging.FileHandler(filename,
                                               mode='w')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        # stream handler
        if stream:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)

    def __repr__(self):
        return '<Logger class object>'

    def lprint(self,
               message_str=None,
               level='info'):
        """
        Method to print to logger
        """

        if level.lower() == 'info':
            self.logger.info(message_str)
        elif level.lower() == 'warn' or level.lower() == 'warning':
            self.logger.warning(message_str)
        elif level.lower() == 'error':
            self.logger.error(message_str)
        else:
            self.logger.debug(message_str)

    def close(self):
        """
        Close logger
        """
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)
