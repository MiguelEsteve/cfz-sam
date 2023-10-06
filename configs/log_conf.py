import logging
from logging.config import dictConfig


def getLogger(name):
    dictConfig(LOGGING)
    logger = logging.getLogger(name=name)
    return logger


class Formatter(logging.Formatter):
    cri_fmt = "\033[7;31mCRITICAL: %(msg)s\033[0m"
    war_fmt = "\033[7;33mWARNING: %(msg)s\033[0m"
    err_fmt = "\033[0;36mERROR: %(msg)s\033[0m"
    inf_fmt = "\033[0;32mINFO:  %(module)s: %(msg)s\033[0m"
    dbg_fmt = "\033[0;33mDEBUG: %(module)s: %(funcName)s: %(lineno)d: %(msg)s\033[0m"

    def __init__(self):
        super().__init__(fmt="%(levelno)d: %(msg)s", datefmt=None, style='%')

    def format(self, record):

        # Save the original format configured by the user
        # when the logger formatter was instantiated
        format_orig = self._style._fmt

        # Replace the original format with one customized by logging level
        if record.levelno == logging.CRITICAL:
            self._style._fmt = Formatter.cri_fmt

        elif record.levelno == logging.ERROR:
            self._style._fmt = Formatter.err_fmt

        elif record.levelno == logging.WARNING:
            self._style._fmt = Formatter.war_fmt

        elif record.levelno == logging.INFO:
            self._style._fmt = Formatter.inf_fmt

        elif record.levelno == logging.DEBUG:
            self._style._fmt = Formatter.dbg_fmt

        # Call the original formatter class to do the grunt work
        result = logging.Formatter.format(self, record)

        # Restore the original format configured by the user
        self._style._fmt = format_orig

        return result

LOGGING = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
        'verbose': {
            'format': '\033[1;31m%(levelname)s %(asctime)s %(module)s process: %(process)s thread: %(thread)d %('
                      'message)s\033[0m'
        },
        'simple': {
            'format': '%(levelname)s %(message)s'
        },
        'myFormatter': {
            '()': Formatter
        }
    },
    'handlers': {
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'myFormatter'
        }
    },
    'loggers': {
        'main': {
            'handlers': ['console'],
            'level': 'INFO'
        },
        'src.runs.example_resnet50': {
            'handlers': ['console'],
            'level': 'DEBUG'
        },
        'src.examples.example_FastSAM': {
            'handlers': ['console'],
            'level': 'DEBUG'
        }
    }
}