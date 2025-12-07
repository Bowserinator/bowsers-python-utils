#!/bin/python3
import contextlib
import datetime
import functools
import importlib
import io
import multiprocessing as mp
import os
import re
import shlex
import signal
import string
import sys
import threading
import time
import traceback
from collections.abc import Callable, Iterator
from typing import TextIO

import psutil


def ttl_cache(seconds: int, maxsize: int = -1):
    """
    Decorator:
    functools.lru_cache with a ttl to each entry. Note entries
    are not proactively evicted, and probably not thread safe.
    """

    def wrapper(func):
        time_cache = {}

        def time_cache_key(*args, **kwargs) -> int:
            hash_key = (args, frozenset(kwargs.items()))
            last_update_time = time_cache.get(hash_key, min(-99999999, -seconds * 2))
            if time.monotonic() - last_update_time > seconds:
                time_cache[hash_key] = time.monotonic()
            return time_cache[hash_key]

        @functools.lru_cache(maxsize=maxsize)
        def time_wrap(_cache_buster=0, *args, **kwargs):
            return func(*args, **kwargs)

        return lambda *args, **kwargs: time_wrap(time_cache_key(*args, **kwargs), *args, **kwargs)

    return wrapper


def callable_once(*, err_on_subsequent: bool = True):
    """
    Decorator ensuring a function is only called once.
    If error_on_second_call=True, raises RuntimeError on second call.
    Otherwise, silently ignores subsequent calls.
    """

    def decorator(func):
        called = False

        def wrapper(*args, **kwargs):
            nonlocal called
            if called:
                if err_on_subsequent:
                    raise RuntimeError(f"{func.__name__} may only be called once")
                return None
            called = True
            return func(*args, **kwargs)

        return wrapper

    return decorator


def repeat_every(seconds: float, *, daemon: bool = True):
    """Run a function every `seconds` in a daemon (default) thread."""

    def decorator(func):
        def start_thread(*args, **kwargs):
            def runner():
                while True:
                    func(*args, **kwargs)
                    time.sleep(seconds)

            t = threading.Thread(target=runner, daemon=daemon)
            t.start()
            return t

        return start_thread

    return decorator


class timeout:  # noqa: N801
    """
    Interrupt block with TimeoutError after seconds, ie

    .. code-block: python

        with timeout(seconds=3, msg="Code took too long"):
            do_long_operation()
    """

    def __init__(self, seconds: int = 1, msg: str = "Timed out"):
        self.seconds = seconds
        self.err_msg = msg

    def handler(self, _signum, _frame):
        raise TimeoutError(self.err_msg)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handler)
        signal.alarm(self.seconds)
        return self

    def __exit__(self, _, value, traceback):
        signal.alarm(0)


# ----------------- IO / commands -----------------------


def x(cmd: str | list[str], *, dryrun: bool = False, quit_on_err: bool = True, err_msg: str | None = None, silent: bool = False) -> int:
    """
    Call os.system(cmd), returning 8 bit retcode value. Also
    prints the command to call to stderr

    :param cmd: Command to run, list or string
    :param dryrun: If true only print and return 0
    :param quit_on_err: Terminate program if cmd fails
    :param err_msg: Custom err msg to print on failure
    :param silent: Don't print unless error
    """
    if isinstance(cmd, list):
        cmd = shlex.join(cmd)

    if not silent:
        print("---", cmd, file=sys.stderr)
    if dryrun:
        return 0
    ec = os.system(cmd)
    if ec != 0:
        if silent:
            print("--- Failed cmd:", cmd, file=sys.stderr)
        if err_msg:
            print(err_msg, file=sys.stderr)
        if quit_on_err:
            sys.exit(ec >> 8)
    return ec >> 8


def iter_lines(file: str | TextIO, *, skip: int = 0, ignore_comments: bool = False, remove_empty: bool = False, encoding: str = "utf-8", errors: str = "strict", buffering: int = -1) -> Iterator[str]:
    """
    Iterate over lines of file

    :param file: File obj or path
    :param skip: Skip first n lines
    :param ignore_comments: Remove lines starting with #
    :param remove_empty: Ignore empty lines
    """
    should_close = False

    if isinstance(file, (str, os.PathLike)):
        file = open(file, encoding=encoding, errors=errors, buffering=buffering)  # noqa: SIM115
        should_close = True

    try:
        for i, line in enumerate(file):
            if i < skip:
                continue
            line = line.rstrip("\n")
            if ignore_comments and line.startswith("#"):
                continue
            if remove_empty and not line.strip():
                continue
            yield line
    finally:
        if should_close:
            file.close()


def read_lines(*args, **kwargs) -> list[str]:
    """Same as iter_lines but returns list"""
    return list(iter_lines(*args, **kwargs))


def iter_paragraphs(
    file: str | TextIO, *, skip: int = 0, ignore_comments: bool = False, remove_empty: bool = False, encoding: str = "utf-8", errors: str = "strict", buffering: int = -1
) -> Iterator[str]:
    """Iterate over paragraphs separated by two or more blank lines"""
    paragraph = []

    for line in iter_lines(
        file,
        skip=skip,
        ignore_comments=ignore_comments,
        remove_empty=False,  # handle emptiness differently here
        encoding=encoding,
        errors=errors,
        buffering=buffering,
    ):
        if line.strip():  # line is non-empty
            paragraph.append(line)
        elif paragraph:
            if not remove_empty or any(str(x).strip() for x in paragraph):
                yield "\n".join(paragraph)
            paragraph = []

    if paragraph:
        yield "\n".join(paragraph)


def eprint(*args, **kwargs):
    """Print to stderr"""
    print(*args, **kwargs, file=sys.stderr)


def die(*args, **kwargs):
    """Print and exit with retcode = -1"""
    eprint(*args, **kwargs)
    sys.exit(-1)


def quit(*args, **kwargs):  # noqa: A001
    """Print and exit with retcode = 0"""
    eprint(*args, **kwargs)
    sys.exit(0)


def set_flush():
    """Remove buffering on stdout"""
    sys.stdout = io.TextIOWrapper(open(sys.stdout.fileno(), "wb", 0), write_through=True)  # noqa: SIM115


# --------- Python and str utils ---------------


def split_maybe_csv_args(args: list[str]) -> list[str]:
    """For when nargs='+' but user might provide a csv instead, ie -a 1,2,3 instead of -a 1 2 3, normalizes"""
    if len(args) == 1 and "," in args[0]:
        args = [x.strip() for x in args[0].split()]
        return [x for x in args if x]
    return args


def remove_empty_keys(d: dict[any, any], pred: Callable[[any], bool] | None = None) -> dict[any, any]:
    """
    Remove keys from dict based on predicate.
    Default: remove keys that are falsy.
    """
    if pred is None:
        pred = lambda k: not k  # noqa: E731
    return {k: v for k, v in d.items() if not pred(k)}


_ansi_re = re.compile(r"\x1b\[[0-9;]*m")


def remove_ansi(s: str) -> str:
    """Strip ANSI escape sequences."""
    return _ansi_re.sub("", s)


def remove_unprintable(s: str) -> str:
    """Remove unprintable characters from a string."""
    return "".join(ch for ch in s if ch in string.printable)


def is_int(s: any) -> bool:
    """Return True if s can be converted to an int."""
    try:
        int(s)
        return True
    except Exception:
        return False


def is_float(s: any) -> bool:
    """Return True if s can be converted to a float."""
    try:
        float(s)
        return True
    except Exception:
        return False


def is_date(s: any) -> bool:
    """Return True if s is YYYYMMDD (string or int)."""
    try:
        datetime.strptime(str(s), "%Y%m%d")
        return True
    except Exception:
        return False


def ensure_trailing_slash(path: str) -> str:
    """Ensure path ends with a slash."""
    return path if path.endswith("/") else path + "/"


def ensure_no_trailing_slash(path: str) -> str:
    """Ensure path does NOT end with a slash."""
    return path.removesuffix("/")


def importfile(file: str):
    """Import python file as module"""
    modulename = os.path.basename(file).removesuffix(".py")
    spec = importlib.util.spec_from_file_location(modulename, file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[modulename] = module
    spec.loader.exec_module(module)
    return module


def lazy_importfile(file: str):
    """Import python file as module but lazy"""
    modulename = os.path.basename(file).removesuffix(".py")
    spec = importlib.util.spec_from_file_location(modulename, file)
    loader = importlib.util.LazyLoader(spec.loader)
    spec.loader = loader

    module = importlib.util.module_from_spec(spec)
    sys.modules[modulename] = module
    spec.loader.exec_module(module)
    return module


def do_ranges_overlap(a, b, c, d) -> bool:
    """Does [a, b] overlap [c, d]"""
    return max(a, c) <= min(b, d)


def is_primitive_type(value: any, *, include_none: bool = True) -> bool:
    """
    Return True if the value is a primitive scalar type.
    Primitive types: int, float, bool, str, bytes, and optionally None.
    """
    primitives = (int, float, bool, str, bytes, complex)
    if include_none and value is None:
        return True
    return isinstance(value, primitives)


def mp_run(func: Callable[..., any], *args, timeout: float = -1.0, **kwargs) -> any:
    """
    Run `func(*args, **kwargs)` inside a separate process. Can be used to isolate
    functions that may terminate the program on error (ie call exit())

    - If the function raises an exception, re-raise it in the parent.
    - If the process crashes, exits with non-zero status, or is killed,
      raise RuntimeError indicating the return code.
    - Ensures subprocess is joined within a short timeout by default.
    """

    def worker(_entry_queue, result_queue):
        try:
            result = func(*args, **kwargs)
            result_queue.put(("ok", result))
        except Exception:
            exc_type, exc_value, exc_tb = sys.exc_info()
            formatted = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            result_queue.put(("exception", (exc_type, exc_value, formatted)))

    result_queue = mp.Queue()
    entry_queue = mp.Queue()

    process = mp.Process(target=worker, args=(entry_queue, result_queue))
    process.start()
    process.join(timeout if timeout > 0 else None)

    if process.is_alive():
        process.terminate()
        process.join()
        raise RuntimeError(f"quarantine: subprocess timed out after {timeout} seconds")

    if process.exitcode != 0:
        raise RuntimeError(f"quarantine: subprocess terminated abnormally (exit code {process.exitcode})")

    if not result_queue.empty():
        status, payload = result_queue.get()
        if status == "ok":
            return payload
        if status == "exception":
            exc_type, exc_value, formatted = payload
            raise exc_type(f"{exc_value}\n(Subprocess traceback):\n{formatted}")
    raise RuntimeError("quarantine: subprocess exited without returning a result")


def recursive_kill(pid: int, sig: int = signal.SIGTERM):
    """Recursively signal PID and its children"""
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    try:
        children = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        children = []

    for child in children:
        with contextlib.suppress((ProcessLookupError, PermissionError)):
            os.kill(child.pid, sig)
    with contextlib.suppress((ProcessLookupError, PermissionError)):
        os.kill(pid, sig)
