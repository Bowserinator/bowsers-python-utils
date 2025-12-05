#!/bin/python3
from collections.abc import Iterator


class BitSet:
    """
    A simple unbounded bitset (internally stored as a python integer)

    Notes
    -----
    This class does not impose a fixed width: the bitset grows to accommodate
    the highest bit that is set.

    Example
    -------
    >>> b = BitSet(0)
    >>> b[0] = 1
    >>> b[3] = True
    >>> print(b)
    0b1001

    >>> b.toggle(1)
    >>> b.as_binstr()
    '1011'

    >>> (b & BitSet("1101")).as_binstr()
    '1001'

    >>> int(b)
    11

    >>> list(b.iter())
    [True, True, False, True]

    >>> b.count()
    3
    """

    def __init__(self, value: int | str = 0):
        if isinstance(value, str):
            self._bits = int(value.strip(), 2)
        else:
            self._bits = int(value)

    def _validate_index(self, index: int):
        """Internal index checking, throws on bad index"""
        if not isinstance(index, int):
            raise TypeError("index must be an int")
        if index < 0:
            raise IndexError("negative indexing not supported")

    def as_binstr(self, width: int | None = None) -> str:
        """
        Return the bits as a binary string without '0b' prefix.
        If width is given, left-pad with zeros to that width.
        """
        s = f"{self._bits:b}"
        if width is not None:
            if width <= 0:
                raise ValueError("width must be positive")
            s = s.rjust(width, "0")
        return s

    def all(self) -> bool:
        """Returns true if all bits are set"""
        return all(x for x in self.iter())

    def any(self) -> bool:
        """Returns true if any of the bits are set"""
        return self._bits != 0

    def none(self) -> bool:
        """Returns true if none of the bits are set"""
        return self._bits == 0

    def clear(self):
        """Clear all bits to 0"""
        self._bits = 0

    def iter(self) -> Iterator[bool]:
        """
        Yield all bits until the last 1 bit (or just 0 if
        a currently empty bitset)
        """
        if self._bits == 0:
            yield False
            return

        bits = self._bits
        while bits != 0:
            yield (bits & 1) == 1
            bits >>= 1

    def set(self, index: int):
        """Set bit at index to True"""
        self[index] = 1

    def unset(self, index: int):
        """Set bit at index to False"""
        self[index] = 0

    def toggle(self, index: int):
        """Toggle bit at index"""
        self._validate_index(index)
        self._bits ^= 1 << index

    def count(self) -> int:
        """Return number of set bits"""
        return self._bits.bit_count()

    # --- Comparison ---
    def __eq__(self, other) -> bool:
        if isinstance(other, BitSet):
            return self._bits == other._bits
        if isinstance(other, int):
            return self._bits == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._bits)

    # --- Bitwise operators ---
    def __and__(self, other) -> "BitSet":
        if isinstance(other, BitSet):
            return BitSet(self._bits & other._bits)
        return BitSet(self._bits & int(other))

    def __or__(self, other) -> "BitSet":
        if isinstance(other, BitSet):
            return BitSet(self._bits | other._bits)
        return BitSet(self._bits | int(other))

    def __xor__(self, other) -> "BitSet":
        if isinstance(other, BitSet):
            return BitSet(self._bits ^ other._bits)
        return BitSet(self._bits ^ int(other))

    def __invert__(self) -> "BitSet":
        # Infinite bit width is not well-defined; assume two's complement infinite
        # but users typically mask after. Provide exact Python int inversion.
        return BitSet(~self._bits)

    # --- Shifts ---
    def __lshift__(self, n: int) -> "BitSet":
        return BitSet(self._bits << n)

    def __rshift__(self, n: int) -> "BitSet":
        return BitSet(self._bits >> n)

    # --- Others ----
    def __int__(self) -> int:
        return self._bits

    def __bool__(self) -> bool:
        return self._bits != 0

    def __getitem__(self, index: int) -> bool:
        self._validate_index(index)
        return ((self._bits >> index) & 1) == 1

    def __setitem__(self, index: int, value: bool | int):
        self._validate_index(index)
        if value:
            self._bits |= 1 << index
        else:
            self._bits &= ~(1 << index)

    def __str__(self) -> str:
        return bin(self._bits)

    def __repr__(self) -> str:
        return f"BitSet({bin(self._bits)})"


if __name__ == "__main__":
    import unittest

    class TestBitSet(unittest.TestCase):
        def test_init_from_int(self):
            b = BitSet(13)
            self.assertEqual(int(b), 13)
            self.assertEqual(str(b), "0b1101")

        def test_init_from_binstr(self):
            b = BitSet("1101")
            self.assertEqual(int(b), 13)
            self.assertEqual(str(b), "0b1101")

            b2 = BitSet("00101")
            self.assertEqual(int(b2), 5)

        def test_get_set_item(self):
            b = BitSet(0)
            b[0] = 1
            b[3] = True
            self.assertTrue(b[0])
            self.assertFalse(b[1])
            self.assertTrue(b[3])
            self.assertEqual(int(b), 0b1001)

            b[3] = 0
            self.assertFalse(b[3])

        def test_set_unset_toggle(self):
            b = BitSet(0)
            b.set(2)
            self.assertEqual(int(b), 4)
            b.unset(2)
            self.assertEqual(int(b), 0)
            b.toggle(3)
            self.assertEqual(int(b), 8)
            b.toggle(3)
            self.assertEqual(int(b), 0)

        def test_iter(self):
            # zero case returns a single False
            b = BitSet(0)
            it = list(b.iter())
            self.assertEqual(it, [False])

            # normal case
            b2 = BitSet(0b10110)
            it2 = list(b2.iter())
            # Bits from LSB upward: 0,1,1,0,1
            self.assertEqual(it2, [False, True, True, False, True])

        def test_all_any_none(self):
            b = BitSet(0)
            self.assertTrue(b.none())
            self.assertFalse(b.any())
            self.assertFalse(b.all())

            b2 = BitSet(0b111)
            self.assertTrue(b2.any())
            self.assertTrue(b2.all())
            self.assertFalse(b2.none())

            b3 = BitSet(0b101)
            self.assertFalse(b3.all())

        def test_count(self):
            self.assertEqual(BitSet(0).count(), 0)
            self.assertEqual(BitSet(0b11101).count(), 4)

        def test_as_binstr(self):
            b = BitSet(13)
            self.assertEqual(b.as_binstr(), "1101")
            self.assertEqual(b.as_binstr(8), "00001101")

        def test_as_binstr_invalid_width(self):
            b = BitSet(5)
            with self.assertRaises(ValueError):
                b.as_binstr(0)
            with self.assertRaises(ValueError):
                b.as_binstr(-3)

        def test_bitwise_and_or_xor(self):
            a = BitSet(0b1011)
            b = BitSet(0b1100)
            self.assertEqual(int(a & b), 0b1000)
            self.assertEqual(int(a | b), 0b1111)
            self.assertEqual(int(a ^ b), 0b0111)

        def test_invert(self):
            b = BitSet(5)
            self.assertEqual(int(~b), ~5)

        def test_shifts(self):
            b = BitSet(0b1011)
            self.assertEqual(int(b << 2), 0b101100)
            self.assertEqual(int(b >> 1), 0b101)

        def test_eq_hash(self):
            a = BitSet(13)
            b = BitSet("1101")
            c = BitSet(5)

            self.assertTrue(a == b)
            self.assertFalse(a == c)
            self.assertEqual(hash(a), hash(b))

            self.assertTrue(a == 13)
            self.assertFalse(a == 12)

        def test_str_repr(self):
            b = BitSet(13)
            self.assertEqual(str(b), "0b1101")
            self.assertIn("0b1101", repr(b))

        def test_negative_index_get(self):
            b = BitSet(5)
            with self.assertRaises(IndexError):
                _ = b[-1]

        def test_negative_index_set(self):
            b = BitSet(5)
            with self.assertRaises(IndexError):
                b[-3] = 1

        def test_non_integer_index_get(self):
            b = BitSet(5)
            with self.assertRaises(TypeError):
                _ = b["x"]

        def test_non_integer_index_set(self):
            b = BitSet(5)
            with self.assertRaises(TypeError):
                b[1.5] = True

        def test_toggle_errors(self):
            b = BitSet(5)
            with self.assertRaises(IndexError):
                b.toggle(-1)
            with self.assertRaises(TypeError):
                b.toggle("a")

        def test_shift_invalid_type(self):
            b = BitSet(3)
            with self.assertRaises(TypeError):
                _ = b << "x"
            with self.assertRaises(TypeError):
                _ = b >> 2.7

    unittest.main()
