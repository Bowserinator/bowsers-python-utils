#!/bin/python3

from collections import Counter
from collections.abc import Callable, Iterable, Sequence


def count_if(iterable: Iterable[any], predicate: Callable[[any], bool]) -> int:
    """Return number of elements satisfying predicate(x)."""
    return sum(1 for x in iterable if predicate(x))


def find_if(iterable: Iterable[any], predicate: Callable[[any], bool]) -> any | None:
    """Return the first element satisfying predicate(x), else None."""
    for x in iterable:
        if predicate(x):
            return x
    return None


def find_where(iterable: Iterable[any], predicate: Callable[[any], bool]) -> int | None:
    """Return the first index satisfying predicate(x), else None."""
    for i, x in enumerate(iterable):
        if predicate(x):
            return i
    return None


def is_sorted(sequence: Sequence[any], key: Callable = lambda x: x) -> bool:
    """Check if sequence is non-decreasing."""
    return all(key(sequence[i]) <= key(sequence[i + 1]) for i in range(len(sequence) - 1))


def is_sorted_until(sequence: Sequence[any], key: Callable = lambda x: x) -> int:
    """
    Returns index of first element violating sortedness.
    If sorted, returns len(sequence).
    """
    for i in range(len(sequence) - 1):
        if key(sequence[i]) > key(sequence[i + 1]):
            return i + 1
    return len(sequence)


def next_permutation(seq: list[any]) -> bool:
    """
    Transform seq into the next lexicographical permutation. (In-place)
    Returns True if such permutation exists, otherwise False and seq is reversed.
    """
    if len(seq) <= 1:
        return False

    i = len(seq) - 2
    while i >= 0 and seq[i] >= seq[i + 1]:
        i -= 1

    if i < 0:
        seq.reverse()
        return False

    j = len(seq) - 1
    while seq[j] <= seq[i]:
        j -= 1

    seq[i], seq[j] = seq[j], seq[i]
    seq[i + 1 :] = reversed(seq[i + 1 :])
    return True


def prev_permutation(seq: list[any]) -> bool:
    """
    Transform seq into previous lexicographical permutation (in place)
    Returns True if such permutation exists, else False and seq becomes max permutation.
    """

    if len(seq) <= 1:
        return False

    i = len(seq) - 2
    while i >= 0 and seq[i] <= seq[i + 1]:
        i -= 1

    if i < 0:
        seq.reverse()  # becomes max permutation
        return False

    j = len(seq) - 1
    while seq[j] >= seq[i]:
        j -= 1

    seq[i], seq[j] = seq[j], seq[i]
    seq[i + 1 :] = reversed(seq[i + 1 :])
    return True


def inplace_merge(seq: list[any], mid: int, key=lambda x: x) -> None:
    """
    Merge seq[:mid] and seq[mid:] which are assumed already sorted.
    Performs stable in-place merge.
    """
    merged = []
    i, j = 0, mid

    while i < mid and j < len(seq):
        if key(seq[i]) <= key(seq[j]):
            merged.append(seq[i])
            i += 1
        else:
            merged.append(seq[j])
            j += 1

    merged.extend(seq[i:mid])
    merged.extend(seq[j:])
    seq[:] = merged


def is_permutation(a: Iterable[any], b: Iterable[any]) -> bool:
    """Returns True if two iterables contain the same elements with the same frequency."""
    return Counter(a) == Counter(b)
