#!/bin/python3
import contextlib
import errno
import fcntl
import logging
import time


class FileMutex:
    """
    fcntl.lockf-backed file mutex.
    """

    def __init__(self, filename, *, read_only=False, dont_kill=False, start=0, length=0, timeout=-1, no_log=False):
        """
        Initialize a file-backed mutex using fcntl.

        Parameters
        ----------
        filename : str
            Path to the lock file. The file will be created if it does not exist
            and an exclusive lock is requested.
        read_only : bool, default=False
            If True, acquire a shared lock (LOCK_SH). If False, acquire an exclusive lock (LOCK_EX).
        dont_kill : bool, default=False
            If True, failures to acquire the lock do not raise exceptions; instead,
            `self.bad` is set to True. If False, lock acquisition errors are raised.
        start : int, default=0
            Byte offset within the file where the lock begins.
        length : int, default=0
            Number of bytes to lock. Zero means "lock to EOF" (lockf semantics).
        timeout : float, default=-1
            Lock acquisition timeout in seconds:
                -1 : block indefinitely until lock is acquired
                0 : try once non-blocking
                >0 : wait up to `timeout` seconds, using signal-based interruption
        no_log : bool, default=False
            If True, suppress error logging. If False, acquisition errors are logged.

        Attributes
        ----------
        bad : bool
            True if lock acquisition failed but `dont_kill=True`, otherwise False.
        file : file object or None
            The underlying opened file object used for locking.
        locked : bool
            True if the lock was successfully acquired, False otherwise.
        """
        self.filename = filename
        self.read_only = read_only
        self.dont_kill = dont_kill
        self.start = start
        self.length = length
        self.timeout = timeout
        self.no_log = no_log

        self.file = None
        self.bad = False
        self.locked = False

        self._open_file()
        if self.file:
            self._acquire_lock()

    def _open_file(self):
        try:
            mode = "r" if self.read_only else "a+"
            self.file = open(self.filename, mode)  # noqa: SIM115
        except Exception:
            self.file = None
            self.bad = self.dont_kill
            if not self.no_log and not self.dont_kill:
                logging.exception(f"Failed to open lock file {self.filename}")
            if not self.dont_kill:
                raise

    def _timeout_handler(self, _signum, _frame):
        raise TimeoutError("File lock acquisition timed out")

    def _acquire_lock(self):
        LOCK_TYPE = fcntl.LOCK_SH if self.read_only else fcntl.LOCK_EX
        LOCK_FLAG = 0 if self.timeout == -1 else fcntl.LOCK_NB

        attempts = 0
        MAX_ATTEMPTS = 5
        if self.timeout >= 0:
            start = time.monotonic()

        while attempts < MAX_ATTEMPTS or (self.timeout >= 0 and time.monotonic() - start < self.timeout):
            try:
                attempts += 1
                fcntl.lockf(self.file.fileno(), LOCK_TYPE | LOCK_FLAG, self.length, self.start)
                self.locked = True
                break
            except (OSError, BlockingIOError) as e:
                # ignore ENOLCK as NFS3 is garbage and can sometimes fail to lock
                # for no reason. This is also why we retry locking repeatedly
                if e.errno == errno.ENOLCK and attempts < MAX_ATTEMPTS:
                    time.sleep(0.1)
                    continue
                if self.timeout == 0 or not (attempts < MAX_ATTEMPTS or (self.timeout >= 0 and time.monotonic() - start < self.timeout)):
                    self.bad = True
                    if not self.no_log:
                        logging.exception(f"Failed to acquire lock on {self.filename}")
                    if not self.dont_kill:
                        raise
                else:
                    time.sleep(0.1)

        if not self.locked:
            self.bad = True
            if not self.no_log:
                logging.exception(f"Failed to acquire lock on {self.filename}")
            if not self.dont_kill:
                raise TimeoutError

    def release(self):
        if self.locked and self.file:
            try:
                fcntl.lockf(self.file.fileno(), fcntl.LOCK_UN, self.length, self.start)
                self.locked = False
            except Exception:
                if not self.no_log:
                    logging.exception(f"Failed to release lock on {self.filename}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        if self.file:
            with contextlib.suppress(Exception):
                self.file.close()
