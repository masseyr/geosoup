
__all__ = ['UninitializedError',
           'ObjectNotFound',
           'ImageProcessingError',
           'TileNotFound',
           'FileNotFound',
           'FieldError']


class UninitializedError(Exception):
    """
    Error class for use in returning fields that may be in error.
    """
    status_code = 500

    def __init__(self, error=None,
                 description="Object not initialized",
                 status_code=status_code,
                 headers=None):
        """
        :param error: name of error
        :param description: readable description
        :param status_code: the http status code
        :param headers: any applicable headers
        :return:
        """
        self.description = description
        self.status_code = status_code
        self.headers = headers
        self.error = error


class ObjectNotFound(Exception):
    """
    Error class for use in returning fields that may be in error.
    """
    status_code = 404

    def __init__(self, error=None,
                 description="Object not found",
                 status_code=status_code,
                 headers=None):
        """
        :param error: name of error
        :param description: readable description
        :param status_code: the http status code
        :param headers: any applicable headers
        :return:
        """
        self.description = description
        self.status_code = status_code
        self.headers = headers
        self.error = error


class FieldError(Exception):
    """
    Error class for use in returning fields that may be in error.
    """
    status_code = 200

    def __init__(self,
                 error="Bad Request",
                 description="Missing or invalid information",
                 status_code=status_code,
                 headers=None):
        """
        :param error: name of error
        :param description: readable description
        :param status_code: the http status code
        :param headers: any applicable headers
        :return:
        """
        self.description = description
        self.status_code = status_code
        self.headers = headers
        self.error = error


class ImageProcessingError(Exception):
    status_code = 500

    def __init__(self, error="Image Processing Error",
                 description="Something went wrong processing your image",
                 status_code=status_code,
                 headers=None):
        """
        :param error: name of error
        :param description: readable description
        :param status_code: the http status code
        :param headers: any applicable headers
        :return:
        """
        self.description = description
        self.status_code = status_code
        self.headers = headers
        self.error = error


class TileNotFound(Exception):
    status_code = 601

    def __init__(self, error="Image Processing Error",
                 description="Image tile not found",
                 status_code=status_code,
                 headers=None):
        """
        :param error: name of error
        :param description: readable description
        :param status_code: the http status code
        :param headers: any applicable headers
        :return:
        """
        self.description = description
        self.status_code = status_code
        self.headers = headers
        self.error = error


class FileNotFound(Exception):
    status_code = 600

    def __init__(self, error=None,
                 description="No matching file found on disk",
                 status_code=status_code,
                 headers=None):
        """
        :param error: name of error
        :param description: readable description
        :param status_code: the http status code
        :param headers: any applicable headers
        :return:
        """
        self.description = description
        self.status_code = status_code
        self.headers = headers
        self.error = error
