"""
Logging utilities, adapted from tf_logging
"""

# pylint: disable=unused-import
# pylint: disable=g-bad-import-order
# pylint: disable=invalid-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging as _logging
import os as _os
import re
import sys as _sys
import traceback as _traceback
from logging import DEBUG
from logging import ERROR
from logging import FATAL
from logging import INFO
from logging import WARN

_name2level = {
    'FATAL': FATAL,
    'F': FATAL,

    'ERROR': ERROR,
    'ERR': ERROR,
    'E': ERROR,

    'WARNING': WARN,
    'WARN': WARN,
    'W': WARN,

    'INFO': INFO,
    'I': INFO,

    'DEBUG': DEBUG,
    'D': DEBUG,
}


def _init_log_level():
    pkg = __name__.split('.')[0]
    env_key = pkg.upper() + '_LOG_LEVEL'
    for k, v in _os.environ.items():
        if k.upper() == env_key:
            if v.upper() in _name2level.keys():
                return _name2level[v.upper()]
            elif re.match(r'^\d$', v):
                return int(v)
            else:
                print(f'Unrecognized log level {v}.', file=_sys.stderr)

    _dev_mode = __file__.lower().find('site-packages') < 0
    return INFO if _dev_mode else WARN


# settings
_log_level = _init_log_level()

# _log_format = '%(name)s %(levelname).1s%(asctime)s.%(msecs)d %(filename)s %(lineno)d - %(message)s'
# _log_format = '%(module)s %(levelname).1s %(filename)s %(lineno)d - %(message)s'
# _log_format = '%(levelname).1s %(sname)s.%(filename)s %(lineno)d - %(message)s'
_log_format = '%(asctime)s %(levelname).1s %(sname)s.%(filename)s %(lineno)d - %(message)s'

# _DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_date_format = '%m-%d %H:%M:%S'


# # detect code run in interactive mode or not
# _interactive = False
# try:
#     # This is only defined in interactive shells.
#     if _sys.ps1: _interactive = True
# except AttributeError:
#     # Even now, we may be in an interactive shell with `python -i`.
#     _interactive = _sys.flags.interactive


class CustomizedLogFormatter(_logging.Formatter):

    def __init__(self, fmt=None, datefmt=None, style='%'):
        super(CustomizedLogFormatter, self).__init__(fmt, datefmt, style)

        self.with_simple_name = fmt.find(style + '(sname)') >= 0

    def formatMessage(self, record):
        if self.with_simple_name:
            record.sname = self.get_simple_name(record.name)

        if isinstance(record.msg, Exception):
            ex = record.msg
            lines = _traceback.format_exception(type(ex), ex, ex.__traceback__)
            record.message = lines[-1] + ''.join(lines[:-1])

        return super(CustomizedLogFormatter, self).formatMessage(record)

    @staticmethod
    def get_simple_name(name):
        if name.endswith('@'):
            return name

        sa = name.split('.')
        if len(sa) <= 1:
            return name

        names = [sa[0]] + \
                [sa[i][0] for i in range(1, len(sa) - 1)]
        return '.'.join(names)


class CustomizedLogger(_logging.Logger):
    FATAL = FATAL
    ERROR = ERROR
    INFO = INFO
    DEBUG = DEBUG
    WARN = WARN

    def __init__(self, name, level=_log_level) -> None:
        super(CustomizedLogger, self).__init__(name, level)

        self.findCaller = _logger_find_caller
        self.setLevel(_log_level)
        self.propagate = False  # disable propagate to parent

        # Don't further configure the logger if the root logger is
        # already configured. This prevents double logging in those cases.
        if not self.handlers:
            # # Add the output handler.
            # stream = _sys.stdout if _interactive else _sys.stderr
            # handler = _logging.StreamHandler(stream)
            # handler.setFormatter(CustomizedLogFormatter(_log_format, _date_format))
            # self.addHandler(handler)

            stdout_handler = _logging.StreamHandler(_sys.stdout)
            stdout_handler.setFormatter(CustomizedLogFormatter(_log_format, _date_format))
            stdout_handler.addFilter(lambda rec: rec.levelno < WARN)
            self.addHandler(stdout_handler)

            stderr_handler = _logging.StreamHandler(_sys.stderr)
            stderr_handler.setFormatter(CustomizedLogFormatter(_log_format, _date_format))
            stderr_handler.addFilter(lambda rec: rec.levelno >= WARN)
            self.addHandler(stderr_handler)

    def getEffectiveLevel(self):
        return _log_level

    def log(self, level, msg, *args, **kwargs):
        super(CustomizedLogger, self).log(level, msg, *args, **kwargs)

    def fatal(self, msg, *args, **kwargs):
        self.log(FATAL, msg, *args, **kwargs)
        self.exception()

    def error(self, msg, *args, **kwargs):
        self.log(ERROR, msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.log(WARN, msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.log(INFO, msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self.log(DEBUG, msg, *args, **kwargs)

    def log_if(self, level, msg, condition, *args):
        """Log 'msg % args' at level 'level' only if condition is fulfilled."""
        if callable(condition):
            if condition():
                self.log(level, msg, *args)
        elif condition:
            self.log(level, msg, *args)

    def log_every_n(self, level, msg, n, *args):
        """Log 'msg % args' at level 'level' once per 'n' times.

        Logs the 1st call, (N+1)st call, (2N+1)st call,  etc.
        Not threadsafe.

        Args:
          level: The level at which to log.
          msg: The message to be logged.
          n: The number of times this should be called before it is logged.
          *args: The args to be substituted into the msg.
        """
        count = _get_next_log_count_per_token(_get_file_and_line())
        self.log_if(level, msg, not (count % n), *args)

    def log_first_n(self, level, msg, n, *args):  # pylint: disable=g-bad-name
        """Log 'msg % args' at level 'level' only first 'n' times.

        Not threadsafe.

        Args:
          level: The level at which to log.
          msg: The message to be logged.
          n: The number of times this should be called before it is logged.
          *args: The args to be substituted into the msg.
        """
        count = _get_next_log_count_per_token(_get_file_and_line())
        self.log_if(level, msg, count < n, *args)

    def is_debug_enabled(self):
        return self.isEnabledFor(DEBUG)

    def is_info_enabled(self):
        return self.isEnabledFor(INFO)

    def is_warning_enabled(self):
        return self.isEnabledFor(WARN)


def get_logger(name):
    original_logger_class = _logging.getLoggerClass()
    _logging.setLoggerClass(CustomizedLogger)
    logger = _logging.getLogger(name)
    _logging.setLoggerClass(original_logger_class)

    return logger


# compatible with pylog
def getLogger(name):
    return get_logger(name)


def get_level():
    """Return how much logging output will be produced for newer logger."""
    return _log_level


def set_level(v):
    """Sets newer logger threshold for what messages will be logged."""
    assert isinstance(v, (str, int))

    global _log_level

    _log_level = to_level(v)


def to_level(v):
    assert isinstance(v, (int, str)) or v is None

    if v is None:
        v = _log_level
    elif isinstance(v, str):
        if v.upper() in _name2level.keys():
            v = _name2level[v.upper()]
        elif re.match(r'^\d$', v):
            v = int(v)
        else:
            raise ValueError(f'Unrecognized log level {v}.')

    return v


def _get_caller(offset=3):
    """Returns a code and frame object for the lowest non-logging stack frame."""
    # Use sys._getframe().  This avoids creating a traceback object.
    # pylint: disable=protected-access
    f = _sys._getframe(offset)
    # pylint: enable=protected-access
    our_file = f.f_code.co_filename
    f = f.f_back
    while f:
        code = f.f_code
        if code.co_filename != our_file:
            return code, f
        f = f.f_back
    return None, None


# The definition of `findCaller` changed in Python 3.2,
# and further changed in Python 3.8
if _sys.version_info.major >= 3 and _sys.version_info.minor >= 8:

    def _logger_find_caller(stack_info=False, stacklevel=1):  # pylint: disable=g-wrong-blank-lines
        code, frame = _get_caller(4)
        sinfo = None
        if stack_info:
            sinfo = '\n'.join(_traceback.format_stack())
        if code:
            return (code.co_filename, frame.f_lineno, code.co_name, sinfo)
        else:
            return '(unknown file)', 0, '(unknown function)', sinfo
elif _sys.version_info.major >= 3 and _sys.version_info.minor >= 2:

    def _logger_find_caller(stack_info=False):  # pylint: disable=g-wrong-blank-lines
        code, frame = _get_caller(4)
        sinfo = None
        if stack_info:
            sinfo = '\n'.join(_traceback.format_stack())
        if code:
            return (code.co_filename, frame.f_lineno, code.co_name, sinfo)
        else:
            return '(unknown file)', 0, '(unknown function)', sinfo
else:
    def _logger_find_caller():  # pylint: disable=g-wrong-blank-lines
        code, frame = _get_caller(4)
        if code:
            return (code.co_filename, frame.f_lineno, code.co_name)
        else:
            return '(unknown file)', 0, '(unknown function)'

# Counter to keep track of number of log entries per token.
_log_counter_per_token = {}


def _get_next_log_count_per_token(token):
    """Wrapper for _log_counter_per_token.

    Args:
      token: The token for which to look up the count.

    Returns:
      The number of times this function has been called with
      *token* as an argument (starting at 0)
    """
    global _log_counter_per_token  # pylint: disable=global-variable-not-assigned
    _log_counter_per_token[token] = 1 + _log_counter_per_token.get(token, -1)
    return _log_counter_per_token[token]


def _get_file_and_line():
    """Returns (filename, linenumber) for the stack frame."""
    code, f = _get_caller()
    if not code:
        return '<unknown>', 0
    return code.co_filename, f.f_lineno
