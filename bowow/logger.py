#!/bin/python3
import datetime
import logging
import sys


class ColoredDateLogger(logging.Logger):
    COLOR_CODES = {
        logging.DEBUG: "\033[90m",  # Muted gray
        logging.INFO: "\033[36m",  # Cyan
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",  # Red
        logging.CRITICAL: "\033[1;31m",  # Bold red
    }
    PREFIXES = {logging.DEBUG: "DEBUG", logging.INFO: " INFO", logging.WARNING: " WARN", logging.ERROR: "ERROR", logging.CRITICAL: "CRIT!"}
    RESET = "\033[0m"

    def __init__(self, name, level=logging.NOTSET, *, color=None, date=False):
        """
        color:
            True  = force ANSI
            False = no ANSI
            None  = ANSI only if stdout is a TTY
        date:
            True = add date/time
        """
        super().__init__(name, level)

        self._color_enabled = color if color is not None else sys.stdout.isatty()
        self._date_enabled = date

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(self._make_formatter())
        self.addHandler(handler)

    # ------------------------------------------------------

    def _colorize(self, level, message):
        if not self._color_enabled:
            return message
        return f"{ColoredDateLogger.COLOR_CODES.get(level, '')}{ColoredDateLogger.PREFIXES.get(level, '')}: {message}{self.RESET}"

    def _timestamp(self):
        if not self._date_enabled:
            return ""
        now = datetime.datetime.now().strftime("%Y%m%d %H:%M:%S")
        return f"[{now}] "

    def _make_formatter(self):
        logger = self

        class _Formatter(logging.Formatter):
            def format(self, record):
                base = record.getMessage()
                base = logger._colorize(record.levelno, base)
                prefix = logger._timestamp()
                if logger._color_enabled:
                    prefix = ColoredDateLogger.COLOR_CODES.get(record.levelno, "") + prefix + ColoredDateLogger.RESET
                return prefix + base

        return _Formatter()


tlog = ColoredDateLogger(None, level=logging.DEBUG, date=True)
ntlog = ColoredDateLogger(None, level=logging.DEBUG, date=False)

if __name__ == "__main__":
    tlog.debug("This is a DEBUG message")
    tlog.info("This is an INFO message")
    tlog.warning("This is a WARNING message")
    tlog.error("This is an ERROR message")
    tlog.critical("This is a CRITICAL message")

    ntlog.debug("This is a DEBUG message")
    ntlog.info("This is an INFO message")
    ntlog.warning("This is a WARNING message")
    ntlog.error("This is an ERROR message")
    ntlog.critical("This is a CRITICAL message")
