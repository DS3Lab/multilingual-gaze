import logging.config

CONFIG = {
    "version": 1,
    "formatters": {
        "simple": {
            "format": "[%(asctime)s - %(name)s - %(levelname)s] %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
            "level": "DEBUG",
            "stream": "ext://sys.stdout"
        }
    },
    "loggers": {
        "ceiling-eye-nlp": {
            "handlers": ["console"],
            "level": "DEBUG"
        }
    }
}

logging.config.dictConfig(CONFIG)
LOGGER = logging.getLogger("processing")
