import logging
from colorama import init, Fore, Style

# Initializing colorama (for Windows really important)
init(autoreset=True)


class ColorFormatter(logging.Formatter):
    """
    A custom logging formatter that adds color to log messages based on severity level.

    Colors are applied using the `colorama` library:
    - DEBUG and INFO: Green
    - WARNING: Yellow
    - ERROR and CRITICAL: Red (CRITICAL is also bold)

    This formatter enhances readability, especially in terminal output.
    """
    COLORS = {
        "DEBUG": Fore.GREEN,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED + Style.BRIGHT,
    }

    def format(self, record: logging.LogRecord) -> str:
        """
        Apply color formatting to a log record based on its severity level.

        :param record: A log record containing all relevant log message information.
        :return: A string with ANSI color codes applied, ready for terminal output.
        """
        color = self.COLORS.get(record.levelname, "")
        message = super().format(record)
        
        return f"{color}{message}{Style.RESET_ALL}"


def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure the global logging settings with colorized output.

    :param level: Logging level (e.g., logging.INFO, logging.DEBUG).
    :return: None
    """
    handler = logging.StreamHandler()
    formatter = ColorFormatter(
        fmt="[%(asctime)s.%(msecs)03d] %(module)-30s:%(lineno)3d %(levelname)-7s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
