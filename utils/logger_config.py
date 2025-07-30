import logging


def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure the global logging settings for the application.

    :param level: Logging level (e.g., logging.INFO, logging.DEBUG).
    :return: None
    """
    logging.basicConfig(
        level=level,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="[%(asctime)s.%(msecs)03d] %(module)-20s:%(lineno)3d %(levelname)-7s - %(message)s",
    )
