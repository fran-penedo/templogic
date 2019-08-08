import logging

logger = logging.getLogger("femformal")
logger.addHandler(logging.NullHandler())

import sys
import os

FOCUSED = os.environ.get("FOCUSED", False)

if "nose" in sys.modules.keys() and FOCUSED:
    import logging.config

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "debug_formatter": {
                    "format": "%(levelname).1s %(module)s:%(lineno)d:%(funcName)s: %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "level": "DEBUG",
                    "class": "logging.StreamHandler",
                    "formatter": "debug_formatter",
                }
            },
            "loggers": {
                "stlmilp": {
                    "handlers": ["console"],
                    "level": "DEBUG",
                    "propagate": True,
                }
            },
        }
    )
