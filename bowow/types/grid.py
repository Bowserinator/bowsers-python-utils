#!/bin/python3
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterator
from typing import Generic, TypeVar

T = TypeVar("T")


class AbstractGrid(ABC, Generic[T]):
    """Abstract base class for a 2D grid."""

    def is_in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self._width and 0 <= y < self._height

    @abstractmethod
    def __getitem__(self, pos: tuple[int, int]) -> T:
        raise NotImplementedError

    @abstractmethod
    def __setitem__(self, pos: tuple[int, int], value: T):
        raise NotImplementedError

    @abstractmethod
    def width(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def height(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def transpose(self) -> "AbstractGrid":
        raise NotImplementedError

    @abstractmethod
    def count_distinct(self, type_: T) -> int:
        raise NotImplementedError

    @abstractmethod
    def fill_all(self, value: T):
        raise NotImplementedError

    @abstractmethod
    def floodfill(self, x: int, y: int, value: T):
        raise NotImplementedError

    def iter_neumann(self, x: int, y: int) -> Iterator[tuple[int, int]]:
        """Yield 4-adjacent positions (N,E,S,W) skipping out-of-bounds."""
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            nx, ny = x + dx, y + dy
            if self.is_in_bounds(nx, ny):
                yield nx, ny

    def iter_moore(self, x: int, y: int) -> Iterator[tuple[int, int]]:
        """Yield 8-adjacent positions skipping out-of-bounds."""
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if self.is_in_bounds(nx, ny):
                    yield nx, ny

    def iterate_slice(self, x1: int, y1: int, x2: int, y2: int) -> Iterator[tuple[int, int]]:
        """Iterate positions in rectangular slice (inclusive x1,y1; exclusive x2,y2)."""
        for y in range(max(0, y1), min(self._height, y2)):
            for x in range(max(0, x1), min(self._width, x2)):
                yield x, y

    def iter_all(self) -> Iterator[tuple[int, int]]:
        """Iterate over all positions in the grid in row-major order (y, x)."""
        for y in range(self._height):
            for x in range(self._width):
                yield x, y


class Grid(AbstractGrid[T]):
    """2D grid backed by a 2D list."""

    def __init__(self, width: int, height: int, fill: T = None):
        self._width = width
        self._height = height
        self._data = [[fill for _ in range(width)] for _ in range(height)]

    @classmethod
    def from_string(cls, s: str) -> "Grid[str]":
        """Construct a grid from a multiline string."""
        lines = [line.strip() for line in s.strip().splitlines() if line.strip()]
        height = len(lines)
        width = max(len(line) for line in lines)
        g = cls(width, height, fill=None)
        for y, line in enumerate(lines):
            for x, c in enumerate(line):
                g[x, y] = c
        return g

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def __getitem__(self, pos: tuple[int, int]) -> T:
        x, y = pos
        return self._data[y][x]

    def __setitem__(self, pos: tuple[int, int], value: T):
        x, y = pos
        self._data[y][x] = value

    def transpose(self) -> "Grid[T]":
        g = Grid(self._height, self._width)
        for y in range(self._height):
            for x in range(self._width):
                g[y, x] = self[x, y]
        return g

    def count_distinct(self, type_: T) -> int:
        return sum(1 for y in range(self._height) for x in range(self._width) if self[x, y] == type_)

    def fill_all(self, value: T):
        for y in range(self._height):
            for x in range(self._width):
                self[x, y] = value

    def floodfill(self, x: int, y: int, value: T):
        if not self.is_in_bounds(x, y):
            return
        target = self[x, y]
        if target == value:
            return
        q = deque([(x, y)])
        while q:
            cx, cy = q.popleft()
            if not self.is_in_bounds(cx, cy):
                continue
            if self[cx, cy] != target:
                continue
            self[cx, cy] = value
            for nx, ny in self.iter_neumann(cx, cy):
                q.append((nx, ny))

    def __str__(self) -> str:
        return "\n".join("".join(str(self[x, y]) if self[x, y] is not None else "." for x in range(self._width)) for y in range(self._height))


class SparseGrid(AbstractGrid[T]):
    """Sparse 2D grid using a dict for storage. Only non-default values are stored."""

    def __init__(self, width: int, height: int, fill: T = None):
        self._width = width
        self._height = height
        self._fill = fill
        self._data: dict[tuple[int, int], T] = {}

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def __getitem__(self, pos: tuple[int, int]) -> T:
        x, y = pos
        if not self.is_in_bounds(x, y):
            raise IndexError("Position out of bounds")
        return self._data.get((x, y), self._fill)

    def __setitem__(self, pos: tuple[int, int], value: T):
        x, y = pos
        if not self.is_in_bounds(x, y):
            raise IndexError("Position out of bounds")
        if value == self._fill:
            self._data.pop((x, y), None)
        else:
            self._data[(x, y)] = value

    def fill_all(self, value: T):
        """Change the default fill value and clear all stored cells."""
        self._fill = value
        self._data.clear()

    def iter_all(self) -> Iterator[tuple[int, int]]:
        """Iterate all positions in row-major order."""
        for y in range(self._height):
            for x in range(self._width):
                yield x, y

    def iter_nondefault(self) -> Iterator[tuple[int, int, T]]:
        """Iterate over positions that are not the default fill."""
        for (x, y), val in self._data.items():
            yield x, y, val

    def count_distinct(self, type_: T) -> int:
        """Count occurrences of a value, including default fill."""
        count = sum(1 for val in self._data.values() if val == type_)
        if self._fill == type_:
            count += self._width * self._height - len(self._data)
        return count

    def transpose(self) -> "SparseGrid[T]":
        g = SparseGrid(self._height, self._width, self._fill)
        for (x, y), val in self._data.items():
            g[y, x] = val
        return g

    def floodfill(self, x: int, y: int, value: T):
        if not self.is_in_bounds(x, y):
            return
        target = self[x, y]
        if target == value:
            return
        from collections import deque

        q = deque([(x, y)])
        while q:
            cx, cy = q.popleft()
            if not self.is_in_bounds(cx, cy):
                continue
            if self[cx, cy] != target:
                continue
            self[cx, cy] = value
            for nx, ny in self.iter_neumann(cx, cy):
                q.append((nx, ny))


if __name__ == "__main__":
    import unittest

    class TestGrids(unittest.TestCase):
        def test_grid_construction_and_indexing(self):
            g = Grid(3, 2, fill=0)
            self.assertEqual(g.width, 3)
            self.assertEqual(g.height, 2)
            g[1, 1] = 42
            self.assertEqual(g[1, 1], 42)
            self.assertEqual(g[0, 0], 0)
            with self.assertRaises(IndexError):
                _ = g[3, 0]
            with self.assertRaises(IndexError):
                g[0, 2] = 1

        def test_grid_from_string(self):
            s = """
            AAA
            BBA
            ABB
            """
            g = Grid.from_string(s)
            self.assertEqual(g[0, 0], "A")
            self.assertEqual(g[1, 1], "B")
            self.assertEqual(g[2, 2], "B")
            self.assertEqual(g.width, 3)
            self.assertEqual(g.height, 3)

        def test_fill_all_and_count(self):
            g = Grid(2, 2, fill=0)
            g.fill_all(7)
            for x, y in g.iter_all():
                self.assertEqual(g[x, y], 7)
            g[0, 0] = 3
            self.assertEqual(g.count_distinct(7), 3)
            self.assertEqual(g.count_distinct(3), 1)

        def test_transpose(self):
            g = Grid(2, 3, fill=0)
            g[0, 0] = 1
            g[1, 2] = 2
            t = g.transpose()
            self.assertEqual(t[0, 0], 1)
            self.assertEqual(t[2, 1], 2)
            self.assertEqual(t.width, 3)
            self.assertEqual(t.height, 2)

        def test_iter_neighbors(self):
            g = Grid(3, 3, fill=0)
            neumann = list(g.iter_neumann(1, 1))
            moore = list(g.iter_moore(1, 1))
            self.assertCountEqual(neumann, [(1, 0), (2, 1), (1, 2), (0, 1)])
            self.assertCountEqual(moore, [(0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (0, 2), (1, 2), (2, 2)])

        def test_iterate_slice(self):
            g = Grid(4, 4, fill=0)
            slice_coords = list(g.iterate_slice(1, 1, 3, 3))
            self.assertCountEqual(slice_coords, [(1, 1), (2, 1), (1, 2), (2, 2)])

        def test_floodfill(self):
            s = """
            AAA
            ABA
            AAA
            """
            g = Grid.from_string(s)
            g.floodfill(1, 1, "X")
            self.assertEqual(g[1, 1], "X")
            self.assertEqual(g[0, 0], "A")  # outside floodfill
            g.floodfill(0, 0, "B")
            self.assertEqual(g[0, 0], "B")
            self.assertEqual(g[0, 1], "B")

        # ---------- SparseGrid tests ----------
        def test_sparsegrid_basic(self):
            sg = SparseGrid(3, 3, fill=0)
            self.assertEqual(sg[0, 0], 0)
            sg[1, 1] = 42
            self.assertEqual(sg[1, 1], 42)
            self.assertEqual(sg[0, 0], 0)
            sg[1, 1] = 0  # back to fill, should remove
            self.assertEqual(sg[1, 1], 0)

        def test_sparsegrid_fill_and_count(self):
            sg = SparseGrid(2, 2, fill=0)
            sg.fill_all(7)
            self.assertEqual(sg[0, 0], 7)
            sg[0, 0] = 3
            self.assertEqual(sg.count_distinct(7), 3)
            self.assertEqual(sg.count_distinct(3), 1)

        def test_sparsegrid_transpose_and_neighbors(self):
            sg = SparseGrid(2, 3, fill=0)
            sg[0, 0] = 1
            sg[1, 2] = 2
            t = sg.transpose()
            self.assertEqual(t[0, 0], 1)
            self.assertEqual(t[2, 1], 2)
            self.assertEqual(list(t.iter_neumann(1, 1)), [(1, 0), (2, 1), (0, 1)])

        def test_sparsegrid_iter_all(self):
            sg = SparseGrid(2, 2, fill=0)
            positions = list(sg.iter_all())
            self.assertCountEqual(positions, [(0, 0), (1, 0), (0, 1), (1, 1)])

        def test_sparsegrid_floodfill(self):
            sg = SparseGrid(3, 3, fill=0)
            sg[1, 1] = 1
            sg.floodfill(0, 0, 2)
            self.assertEqual(sg[0, 0], 2)
            self.assertEqual(sg[1, 1], 1)  # not overwritten
            sg.floodfill(1, 1, 3)
            self.assertEqual(sg[1, 1], 3)

    unittest.main()
