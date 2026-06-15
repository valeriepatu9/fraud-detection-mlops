import sys


class FraudDetectionException(Exception):
    """Exception maison qui capture le fichier et la ligne d'origine de l'erreur."""

    def __init__(self, error_message, error_detail: sys = sys):
        super().__init__(error_message)
        _, _, exc_tb = error_detail.exc_info()
        self.lineno = exc_tb.tb_lineno if exc_tb is not None else "?"
        self.file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb is not None else "?"
        self.error_message = str(error_message)

    def __str__(self):
        return (
            f"Erreur dans le script [{self.file_name}] "
            f"ligne [{self.lineno}] : {self.error_message}"
        )
