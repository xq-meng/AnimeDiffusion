import logging
from .path import *


class Logger(object):

    def __init__(
        self,
        name=None,
        level='debug',
        console=True,
        logfile='',
        format='[%(asctime)s][%(levelname)s] %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S'
    ):
        self.level_mapping = {
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR,
            'critical': logging.CRITICAL
        }

        # get basic logger instance
        self._logger = logging.getLogger(name=name)
        self._logger.setLevel(self.level_mapping.get(level))
        logfmt = logging.Formatter(fmt=format, datefmt=datefmt)

        # output to console
        if console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logfmt)
            self._logger.addHandler(console_handler)

        # output to logfile
        if len(logfile) > 0:
            create_prefix_dir(logfile)
            logfile_handler = logging.FileHandler(filename=logfile, encoding='utf-8')
            logfile_handler.setFormatter(logfmt)
            self._logger.addHandler(logfile_handler)

    def debug(self, msg, *args, **kwargs):
        return self._logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        return self._logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        return self._logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        return self._logger.error(msg, *args, **kwargs)