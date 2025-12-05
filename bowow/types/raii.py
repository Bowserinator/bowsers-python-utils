#!/bin/python3
import weakref
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class RAIIWrapper(Generic[T]):
    """
    RAII-style wrapper that acquires a context-managed resource immediately
    and ensures it is released when this wrapper is garbage-collected.

    Construction
    ------------
    - Pass either:
      * a context-manager factory (callable) that returns an object with
        __enter__/__exit__ (e.g. `open`, or a `contextlib.contextmanager` factory),
      * or an already-instantiated context manager (an object with __enter__
        and __exit__).

    Behavior
    --------
    - On construction the wrapper calls the context manager's __enter__ and
      stores the returned resource as `.value`.
    - A `weakref.finalize` finalizer is registered to call the context
      manager's `__exit__(None, None, None)` when this RAII object is
      garbage-collected (or when the interpreter exits).
    - You can call `.close()` to run the cleanup immediately — this will
      also prevent the finalizer from running again.
    - Attribute access is proxied to the held `.value`

    - CAVEAT: multiple references to an RAIIWrapper will not behave properly:
      __exit__ will only be called when the last RAIIWrapper wrapper drops

    Example
    -------
    >>> from contextlib import contextmanager
    >>> @contextmanager
    ... def trace(name):
    ...     print(f"enter {name}")
    ...     try:
    ...         yield {"name": name}
    ...     finally:
    ...         print(f"exit {name}")
    >>> r = RAII(trace, "x")  # prints "enter x"
    >>> print(r.value)  # {'name': 'x'}
    >>> # When `r` is deleted or goes out of scope and is GC'd, the context
    >>> # manager's __exit__ is called -> prints "exit x"
    >>> r.close()  # explicitly close early -> prints "exit x"

    Notes
    -----
    - The finalizer calls the context manager's __exit__(None, None, None),
    - If the underlying resource's __exit__ needs exception info, call .close()
      inside exception-handling code yourself
    """

    def __init__(self, cm_or_factory: Any, *args, **kwargs):
        """
        Acquire the resource immediately.

        Parameters
        ----------
        cm_or_factory:
            either a context-manager factory (callable) or a context-manager
            instance (has __enter__ and __exit__).
        *args, **kwargs:
            forwarded to cm_or_factory if it is a callable factory.
        """
        if hasattr(cm_or_factory, "__enter__") and hasattr(cm_or_factory, "__exit__"):
            self._cm = cm_or_factory
        elif callable(cm_or_factory):
            self._cm = cm_or_factory(*args, **kwargs)
        else:
            raise TypeError("cm_or_factory must be a context-manager instance or a callable")

        self.value = self._cm.__enter__()
        self._finalizer = weakref.finalize(self, self._cm.__exit__, None, None, None)
        self._closed = False

    def close(self):
        """
        Explicitly release the resource now by calling the context manager's
        __exit__(None, None, None). If already closed, this is a no-op.
        """
        if self._closed:
            return
        try:
            self._finalizer()
        finally:
            self._closed = True

    def __repr__(self) -> str:
        return f"RAII(value={self.value!r}, closed={self._closed})"


if __name__ == "__main__":
    import gc
    import unittest

    exited = False

    class DummyContext:
        """Simple context manager for testing RAIIWrapper."""

        def __init__(self):
            self.entered = False

        def __enter__(self):
            self.entered = True
            return "resource_value"

        def __exit__(self, exc_type, exc, tb):
            global exited  # noqa: PLW0603
            exited = True

    class TestRAIIWrapper(unittest.TestCase):
        def test_enter_called_on_construction(self):
            ctx = DummyContext()
            wrapper = RAIIWrapper(ctx)
            self.assertTrue(ctx.entered)
            self.assertEqual(wrapper.value, "resource_value")

        def test_exit_called_on_gc(self):
            global exited  # noqa: PLW0603
            exited = False

            ctx = DummyContext()
            wrapper = RAIIWrapper(ctx)
            del wrapper
            gc.collect()
            self.assertTrue(exited)

        def test_exit_called_once(self):
            global exited  # noqa: PLW0603
            exited = False

            ctx = DummyContext()
            wrapper = RAIIWrapper(ctx)
            wrapper.close()
            self.assertTrue(exited)
            wrapper.close()

    unittest.main()
