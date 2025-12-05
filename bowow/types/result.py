#!/bin/python3
from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")
E = TypeVar("E")
R = TypeVar("R")


class Result(Generic[T, E]):
    """Base Result class. Do not instantiate directly."""

    @abc.abstractmethod
    def is_ok(self) -> bool:
        raise NotImplementedError

    def is_err(self) -> bool:
        return not self.is_ok()

    def ok(self) -> T | None:
        return self.value if self.is_ok() else None

    def err(self) -> E | None:
        return self.value if self.is_err() else None

    def unwrap(self) -> T:
        if self.is_ok():
            return self.value  # type: ignore
        raise RuntimeError(f"Called unwrap on Err: {self.value!r}")

    def unwrap_err(self) -> E:
        if self.is_err():
            return self.value  # type: ignore
        raise RuntimeError(f"Called unwrap_err on Ok: {self.value!r}")

    def expect(self, msg: str) -> T:
        if self.is_ok():
            return self.value  # type: ignore
        raise RuntimeError(f"{msg}: {self.value!r}")

    def map(self, func: Callable[[T], R]) -> Result[R, E]:
        if self.is_ok():
            return Ok(func(self.value))  # type: ignore
        return self  # type: ignore

    def map_err(self, func: Callable[[E], R]) -> Result[T, R]:
        if self.is_err():
            return Err(func(self.value))  # type: ignore
        return self  # type: ignore

    def and_then(self, func: Callable[[T], Result[R, E]]) -> Result[R, E]:
        if self.is_ok():
            return func(self.value)  # type: ignore
        return self  # type: ignore

    def or_else(self, func: Callable[[E], Result[T, R]]) -> Result[T, R]:
        if self.is_err():
            return func(self.value)  # type: ignore
        return self  # type: ignore

    def unwrap_or(self, default: T) -> T:
        if self.is_ok():
            return self.value  # type: ignore
        return default

    def unwrap_or_else(self, func: Callable[[E], T]) -> T:
        if self.is_ok():
            return self.value  # type: ignore
        return func(self.value)  # type: ignore

    def __repr__(self) -> str:
        typename = "Ok" if self.is_ok() else "Err"
        return f"{typename}({self.value!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Result):
            return False
        return self.is_ok() == other.is_ok() and self.value == other.value

    def __hash__(self) -> int:
        return hash((self.is_ok(), self.value))

    def __bool__(self) -> bool:
        return self.is_ok()


class Ok(Result[T, E]):
    def __init__(self, value: T | None = None):
        self.value: T | None = value

    def is_ok(self) -> bool:
        return True


class Err(Result[T, E]):
    def __init__(self, value: E | None = None):
        self.value: E | None = value

    def is_ok(self) -> bool:
        return False


def resultify(func: Callable[..., T]) -> Callable[..., Result[T, Exception]]:
    """Decorator to convert exceptions to Result."""

    def wrapper(*args, **kwargs) -> Result[T, Exception]:
        try:
            return Ok(func(*args, **kwargs))
        except Exception as e:
            return Err(e)

    return wrapper


if __name__ == "__main__":
    import unittest

    class TestResult(unittest.TestCase):
        def test_ok_basic(self):
            r = Ok(10)
            self.assertTrue(r.is_ok())
            self.assertFalse(r.is_err())
            self.assertEqual(r.ok(), 10)
            self.assertIsNone(r.err())
            self.assertEqual(r.unwrap(), 10)
            self.assertEqual(r.unwrap_or(0), 10)
            self.assertTrue(bool(r))

        def test_err_basic(self):
            e = Err("error")
            self.assertFalse(e.is_ok())
            self.assertTrue(e.is_err())
            self.assertEqual(e.err(), "error")
            self.assertIsNone(e.ok())
            self.assertEqual(e.unwrap_or(0), 0)
            self.assertFalse(bool(e))

            with self.assertRaises(RuntimeError):
                e.unwrap()

        def test_map_and_then(self):
            r = Ok(2)
            r2 = r.map(lambda x: x * 3)
            self.assertEqual(r2.unwrap(), 6)

            def f(x):
                return Ok(x + 1)

            r3 = r.and_then(f)
            self.assertEqual(r3.unwrap(), 3)

            e = Err("fail")
            self.assertEqual(e.map(lambda x: x * 10), e)
            self.assertEqual(e.and_then(f), e)

        def test_map_err_or_else(self):
            e = Err(5)
            e2 = e.map_err(lambda x: x * 2)
            self.assertEqual(e2.err(), 10)

            r = Ok(1)
            r2 = r.map_err(lambda x: x * 2)
            self.assertEqual(r2.unwrap(), 1)

            # or_else
            r3 = e.or_else(lambda x: Ok(x + 1))
            self.assertTrue(r3.is_ok())
            self.assertEqual(r3.unwrap(), 6)

        def test_eq_and_hash(self):
            a = Ok(10)
            b = Ok(10)
            c = Err(10)
            self.assertEqual(a, b)
            self.assertNotEqual(a, c)
            self.assertEqual(hash(a), hash(b))
            self.assertNotEqual(hash(a), hash(c))

        def test_resultify_ok(self):
            @resultify
            def f():
                return 42

            r = f()
            self.assertTrue(r.is_ok())
            self.assertEqual(r.unwrap(), 42)

        def test_resultify_err(self):
            @resultify
            def g():
                raise ValueError("fail")

            r = g()
            self.assertTrue(r.is_err())
            self.assertIsInstance(r.err(), ValueError)
            self.assertEqual(str(r.err()), "fail")

    unittest.main()
