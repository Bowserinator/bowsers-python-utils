#!/bin/python3
import asyncio
import functools
import io
import sys
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import suppress
from typing import Any, TypeVar

T = TypeVar("T")
U = TypeVar("U")


# -----------------------------------------------------
class StdoutRedirect:
    """Redirect stdout and/or stderr to internal StringIO buffer."""

    def __init__(self, *, stdout=True, stderr=False):
        self.stdout_flag = stdout
        self.stderr_flag = stderr
        self._buf = io.StringIO()
        self._old_stdout = None
        self._old_stderr = None

    def __enter__(self):
        if self.stdout_flag:
            self._old_stdout = sys.stdout
            sys.stdout = self._buf
        if self.stderr_flag:
            self._old_stderr = sys.stderr
            sys.stderr = self._buf
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.stdout_flag:
            sys.stdout = self._old_stdout
        if self.stderr_flag:
            sys.stderr = self._old_stderr

    def __str__(self):
        return self._buf.getvalue()


# -----------------------------------------------------
class TmpSysPath:
    """Temporarily prepend a path to sys.path."""

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        sys.path.insert(0, self.path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        with suppress(ValueError):
            sys.path.remove(self.path)


# -----------------------------------------------------
class ContextBlock:
    """Executes a callback on __exit__."""

    def __init__(self, callback):
        self.callback = callback

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.callback()


# -----------------------------------------------------
class Executor:
    """Run a coroutine in ThreadPoolExecutor or ProcessPoolExecutor via asyncio."""

    def __init__(self, *, loop=None, thread=True, workers=None):
        self.loop = loop or asyncio.get_event_loop()
        self.pool = ThreadPoolExecutor(max_workers=workers) if thread else ProcessPoolExecutor(max_workers=workers)

    async def __call__(self, func, *args, **kwargs):
        partial_func = functools.partial(func, *args, **kwargs)
        return await self.loop.run_in_executor(self.pool, partial_func)


# -----------------------------------------------------
class KeyWrapper:
    """Wraps an iterable and applies a transform function on item access."""

    def __init__(self, iterable: Iterable[T], transform: Callable[[T], U]):
        self._iterable = iterable
        self._transform = transform

    def __getitem__(self, index) -> U:
        return self._transform(self._iterable[index])

    def __len__(self) -> int:
        return len(self._iterable)

    def __iter__(self):
        return (self._transform(x) for x in self._iterable)


# -----------------------------------------------------
def NamedList(type_name: str, field_names: Iterable[str]):  # noqa: N802
    """
    Like collections.namedtuple but mutable. Supports construction with kwargs,
    _asdict(), _replace(), and access by named members
    """
    field_names = list(field_names)
    n_fields = len(field_names)

    class _NamedList(list):
        __slots__ = ()
        _fields = field_names

        def __init__(self, values: Iterable[Any] = (), **kwargs):
            if values and kwargs:
                raise ValueError("Cannot mix positional values and keyword arguments")
            if kwargs:
                vals = [kwargs.get(name) for name in self._fields]
            elif values:
                if len(values) != n_fields:
                    raise ValueError(f"Expected {n_fields} values, got {len(values)}")
                vals = list(values)
            else:
                vals = [None] * n_fields
            super().__init__(vals)

        def __repr__(self):
            pairs = ", ".join(f"{name}={self[i]!r}" for i, name in enumerate(self._fields))
            return f"{type_name}({pairs})"

        def _asdict(self) -> dict[str, Any]:
            return {name: self[i] for i, name in enumerate(self._fields)}

        def _replace(self, **kwargs) -> "_NamedList":
            new_vals = [kwargs.get(name, self[i]) for i, name in enumerate(self._fields)]
            return _NamedList(new_vals)

    for i, name in enumerate(field_names):
        setattr(_NamedList, name, property(fget=lambda self, i=i: self[i], fset=lambda self, val, i=i: self.__setitem__(i, val)))

    _NamedList.__name__ = type_name
    return _NamedList


if __name__ == "__main__":
    import unittest

    class TestNamedList(unittest.TestCase):
        def test_basic_access_and_mutation(self):
            P = NamedList("P", ["a", "b"])
            p = P([10, 20])
            self.assertEqual(p.a, 10)
            self.assertEqual(p.b, 20)
            p.a = 100
            self.assertEqual(p[0], 100)
            p[1] = 200
            self.assertEqual(p.b, 200)

        def test_repr(self):
            P = NamedList("Point", ["x", "y"])
            p = P([7, 8])
            self.assertEqual(repr(p), "Point(x=7, y=8)")

        def test_default_values(self):
            P = NamedList("Pair", ["first", "second"])
            p = P()
            self.assertIsNone(p.first)
            self.assertIsNone(p.second)
            p.first = 1
            p.second = 2
            self.assertEqual(list(p), [1, 2])

        def test_length_mismatch(self):
            P = NamedList("Triplet", ["x", "y", "z"])
            with self.assertRaises(ValueError):
                P([1, 2])

        def test_asdict_and_replace(self):
            Point = NamedList("Point", ["x", "y", "z"])
            p = Point([1, 2, 3])
            self.assertEqual(p._asdict(), {"x": 1, "y": 2, "z": 3})
            p2 = p._replace(y=42)
            self.assertEqual(p2._asdict(), {"x": 1, "y": 42, "z": 3})
            # original not mutated
            self.assertEqual(p._asdict(), {"x": 1, "y": 2, "z": 3})

        def test_construct_from_kwargs(self):
            P = NamedList("P", ["a", "b", "c"])
            p = P(a=1, c=3)
            self.assertEqual(p._asdict(), {"a": 1, "b": None, "c": 3})

    unittest.main()
