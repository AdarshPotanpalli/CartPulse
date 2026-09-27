import sys

def error_message_detail(error: Exception, error_detail: sys) -> str:
    """Extracts detailed error information including file name, line number, and message."""
    _, _, exc_tb = error_detail.exc_info()

    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        return f"Error occurred in script: [{file_name}] at line number [{line_number}]: {str(error)}"
    
    return f"Error occurred: {str(error)}"


class CustomException(Exception):
    """Custom exception class for MLOps pipeline errors."""

    def __init__(self, error_message: Exception, error_detail: sys):
        """
        :param error_message: The original Exception or error message.
        :param error_detail: The sys module to access traceback details.
        """
        super().__init__(str(error_message))
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self) -> str:
        return self.error_message