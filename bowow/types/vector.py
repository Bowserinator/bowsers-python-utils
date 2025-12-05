#!/bin/python3

import abc
import math
from collections.abc import Callable
from typing import Union


class AbstractVector(abc.ABC):
    """Abstract base class for a mathematical vector."""

    def __init__(self, components: int):
        self.components = components

    @abc.abstractmethod
    def almost_equals(self, other: "AbstractVector") -> bool:
        """Return True if this vector is approximately equal to another."""
        raise NotImplementedError

    @abc.abstractmethod
    def length(self) -> float:
        """Return the magnitude (length) of the vector."""
        raise NotImplementedError

    @abc.abstractmethod
    def length_sqr(self) -> float:
        """Return the squared length of the vector (avoids sqrt)."""
        raise NotImplementedError

    @abc.abstractmethod
    def distance(self, other: "AbstractVector") -> float:
        """Return the Euclidean distance between this vector and another."""
        raise NotImplementedError

    @abc.abstractmethod
    def distance_sqr(self, other: "AbstractVector") -> float:
        """Return the squared distance to another vector (avoids sqrt)."""
        raise NotImplementedError

    @abc.abstractmethod
    def dot(self, other: "AbstractVector") -> float:
        """Return the dot product of this vector with another."""
        raise NotImplementedError

    @abc.abstractmethod
    def angle(self, other: "AbstractVector") -> float:
        """Return the angle in radians between this vector and another."""
        raise NotImplementedError

    @abc.abstractmethod
    def clamp(self, a: Union["AbstractVector", float], b: Union["AbstractVector", float]) -> "AbstractVector":
        """Clamp each component between values or vectors a and b."""
        raise NotImplementedError

    @abc.abstractmethod
    def clamp_length(self, a: float, b: float) -> "AbstractVector":
        """Clamp the vector's length between a minimum and maximum value. If vector is zero, returns Vec(0)"""
        raise NotImplementedError

    @abc.abstractmethod
    def apply(self, func: Callable[[float], float]) -> "AbstractVector":
        """Apply a function to each component of the vector."""
        raise NotImplementedError

    @abc.abstractmethod
    def normalize(self) -> "AbstractVector":
        """Return a unit vector in the same direction. If vector is zero, returns Vec(0)"""
        raise NotImplementedError

    def reflect(self, normal: "AbstractVector") -> "AbstractVector":
        """Reflect this vector across the given normal vector (normal must be unit length)."""
        dot = self.dot(normal)
        return self - 2 * dot * normal

    @abc.abstractmethod
    def move_torwards(self, target: "AbstractVector", distance: float) -> "AbstractVector":
        """
        Return a vector moved towards target by the specified distance. If distance
        exceeds distance to target, clamps to target
        """
        raise NotImplementedError

    @abc.abstractmethod
    def lerp(self, target: "AbstractVector", amt: float) -> "AbstractVector":
        """Return a vector linearly interpolated towards target by amt (0..1)."""
        raise NotImplementedError

    @abc.abstractmethod
    def __add__(self, other: "AbstractVector") -> "AbstractVector":
        raise NotImplementedError

    @abc.abstractmethod
    def __sub__(self, other: "AbstractVector") -> "AbstractVector":
        raise NotImplementedError

    @abc.abstractmethod
    def __mul__(self, scalar: float | int) -> "AbstractVector":
        raise NotImplementedError

    def __rmul__(self, scalar: float | int) -> "AbstractVector":
        return self.__mul__(scalar)

    @abc.abstractmethod
    def __truediv__(self, scalar: float | int) -> "AbstractVector":
        raise NotImplementedError

    def __neg__(self) -> "Vec2":
        return -1 * self

    def __eq__(self, other: object) -> bool:
        return not (self != other)

    def __ne__(self, other: object) -> bool:
        if len(other) != len(self):
            return True
        return any(self[i] != val for i, val in enumerate(other))

    @abc.abstractmethod
    def __setitem__(self, index: int, val: float | int):
        raise NotImplementedError

    @abc.abstractmethod
    def __getitem__(self, index: int) -> float | int:
        raise NotImplementedError

    @abc.abstractmethod
    def __hash__(self) -> int:
        raise NotImplementedError

    def __iter__(self):
        """Iterate over the vector's components in order."""
        for i in range(len(self)):
            yield self[i]

    def __len__(self) -> int:
        """Return the number of components in the vector (e.g., 2 for Vec2, 3 for Vec3)."""
        return self.components


class Vec2(AbstractVector):
    __slots__ = ("x", "y")

    def __init__(self, x: Union["Vec2", float, int] = 0.0, y: None | float | int = None):
        AbstractVector.__init__(self, 2)
        if isinstance(x, Vec2):
            self.x, self.y = x.x, x.y
        else:
            self.x = x
            self.y = y if y is not None else self.x

    @classmethod
    def from_angle(cls, angle: float) -> "Vec2":
        """Return a unit vector pointing in the direction of `angle` radians."""
        return cls(math.cos(angle), math.sin(angle))

    # Vec2 specific functions
    # ---------------------------------------------------

    def rotate(self, angle: float, origin: "Vec2" = None) -> "Vec2":
        """
        Return this vector rotated counter-clockwise by `angle` radians.
        If `origin` is given, rotate around that point; otherwise rotate around the origin (0,0).
        """
        ox, oy = (origin.x, origin.y) if origin else (0.0, 0.0)

        x, y = self.x - ox, self.y - oy
        cos_a = math.cos(-angle)
        sin_a = math.sin(-angle)

        rx = x * cos_a + y * sin_a
        ry = -x * sin_a + y * cos_a

        return Vec2(rx + ox, ry + oy)

    # ---------------------------------------------------

    def almost_equals(self, other: "AbstractVector") -> bool:
        return math.isclose(self.x, other.x) and math.isclose(self.y, other.y)

    def length(self) -> float:
        return math.hypot(self.x, self.y)

    def length_sqr(self) -> float:
        return self.x * self.x + self.y * self.y

    def distance(self, other: "AbstractVector") -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        return math.hypot(dx, dy)

    def distance_sqr(self, other: "AbstractVector") -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        return dx * dx + dy * dy

    def dot(self, other: "AbstractVector") -> float | int:
        return self.x * other.x + self.y * other.y

    def angle(self, other: "AbstractVector") -> float:
        dot = self.dot(other)
        len_self = self.length()
        len_other = other.length()
        if len_self == 0 or len_other == 0:
            return 0.0
        # Clamp to avoid numeric issues
        cos_theta = max(min(dot / (len_self * len_other), 1.0), -1.0)
        return math.acos(cos_theta)

    def clamp(self, a: Union["AbstractVector", float | int], b: Union["AbstractVector", float | int]) -> "Vec2":
        x = self.x
        y = self.y
        if isinstance(a, Vec2) and isinstance(b, Vec2):
            x = max(a.x, min(x, b.x))
            y = max(a.y, min(y, b.y))
        else:
            x = max(a, min(x, b))
            y = max(a, min(y, b))
        return Vec2(x, y)

    def clamp_length(self, a: float | int, b: float | int) -> "Vec2":
        l = self.length()
        if l == 0:
            return Vec2(0, 0)
        scale = max(a, min(l, b)) / l
        return self * scale

    def apply(self, func: Callable[[float | int], float | int]) -> "Vec2":
        return Vec2(func(self.x), func(self.y))

    def normalize(self) -> "Vec2":
        l = self.length()
        if l == 0:
            return Vec2(0, 0)
        return self / l

    def move_torwards(self, target: "Vec2", distance: float | int) -> "Vec2":
        direction = target - self
        d_len = direction.length()
        if d_len == 0 or distance <= 0:
            return Vec2(self.x, self.y)
        if distance >= d_len:
            return Vec2(target.x, target.y)
        return self + direction * (distance / d_len)

    def lerp(self, target: "Vec2", amt: float) -> "Vec2":
        return Vec2(self.x + (target.x - self.x) * amt, self.y + (target.y - self.y) * amt)

    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float | int) -> "Vec2":
        return Vec2(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar: float | int) -> "Vec2":
        if scalar == 0:
            raise ZeroDivisionError("division by zero in Vec2")
        return Vec2(self.x / scalar, self.y / scalar)

    def __repr__(self) -> str:
        return f"Vec2(x={self.x}, y={self.y})"

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __getitem__(self, index: int) -> float | int:
        if index in (0, -2):
            return self.x
        if index in (1, -1):
            return self.y
        raise IndexError("Vec2 index out of range")

    def __setitem__(self, index: int, val: float | int):
        if index in (0, -2):
            self.x = val
        elif index in (1, -1):
            self.y = val
        else:
            raise IndexError("Vec2 index out of range")


class Vec3(AbstractVector):
    __slots__ = ("x", "y", "z")

    def __init__(self, a: Union[float, int, "Vec2", "Vec3"] = 0.0, b: Union[float, int, "Vec2", None] | None = None, c: float | int | None | None = None):
        super().__init__(3)

        if isinstance(a, Vec3):
            self.x, self.y, self.z = a.x, a.y, a.z
        elif isinstance(a, Vec2):
            self.x = a.x
            if b is None:
                self.y = a.y
                self.z = a.y
            elif isinstance(b, (float, int)):
                self.y = a.y
                self.z = b
            else:
                raise TypeError("Invalid Vec3 construction")
        elif isinstance(b, Vec2):
            if not isinstance(a, (float, int)):
                raise TypeError("Invalid Vec3 construction")
            self.x = a
            self.y = b.x
            self.z = b.y
        elif isinstance(a, (float, int)) and b is None and c is None:
            self.x = self.y = self.z = a
        elif isinstance(a, (float, int)) and isinstance(b, (float, int)) and c is None:
            self.x = a
            self.y = b
            self.z = b
        elif all(isinstance(v, (float, int)) for v in (a, b, c)):
            self.x, self.y, self.z = a, b, c
        else:
            raise TypeError("Invalid Vec3 constructor arguments")

    @classmethod
    def from_polar(cls, pitch: float | "Vec2", yaw: None | float = None) -> "Vec3":  # noqa: TC010
        """
        Create a unit vector from polar angles.
        Usage:
            Vec3.from_polar(pitch, yaw)
            Vec3.from_polar(Vec2(pitch, yaw))
        pitch: up/down angle in radians
        yaw: left/right angle in radians
        """
        if isinstance(pitch, Vec2):
            pitch, yaw = pitch.x, pitch.y
        cp = math.cos(pitch)
        sp = math.sin(pitch)
        cy = math.cos(yaw)
        sy = math.sin(yaw)
        return cls(cp * cy, sp, cp * sy)

    # Vec3 specific functions
    # ---------------------------------------------------

    def cross(self, other: "Vec3") -> "Vec3":
        return Vec3(self.y * other.z - self.z * other.y, self.z * other.x - self.x * other.z, self.x * other.y - self.y * other.x)

    def rotate_by_axis(self, axis: "Vec3", angle: float) -> "Vec3":
        """
        Rotate this vector around the given axis by `angle` radians.
        Axis must be a normalized vector.
        """
        k = axis.normalize()
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return self * cos_a + k.cross(self) * sin_a + k * (k.dot(self)) * (1 - cos_a)

    def refract(self, normal: "Vec3", r: float) -> "Vec3":
        """
        Refract this normalized vector given a normalized surface normal and
        index of refraction r (n1/n2). Returns Vec3(0,0,0) on total internal reflection.
        """
        cos_i = -self.dot(normal)
        sin_t2 = r * r * (1 - cos_i * cos_i)
        if sin_t2 > 1.0:  # Total internal reflection
            return Vec3(0, 0, 0)
        cos_t = math.sqrt(1.0 - sin_t2)
        return self * r + normal * (r * cos_i - cos_t)

    def rotate_x(self, angle: float) -> "Vec3":
        """Rotate this vector CCW around X-axis by `angle` radians."""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        y = self.y * cos_a - self.z * sin_a
        z = self.y * sin_a + self.z * cos_a
        return Vec3(self.x, y, z)

    def rotate_y(self, angle: float) -> "Vec3":
        """Rotate this vector CCW around Y-axis by `angle` radians."""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        x = self.z * sin_a + self.x * cos_a
        z = self.z * cos_a - self.x * sin_a
        return Vec3(x, self.y, z)

    def rotate_z(self, angle: float) -> "Vec3":
        """Rotate this vector CCW around Z-axis by `angle` radians."""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        x = self.x * cos_a - self.y * sin_a
        y = self.x * sin_a + self.y * cos_a
        return Vec3(x, y, self.z)

    # ---------------------------------------------------

    def almost_equals(self, other: "AbstractVector") -> bool:
        return math.isclose(self.x, other.x) and math.isclose(self.y, other.y) and math.isclose(self.z, other.z)

    def length(self) -> float:
        return math.hypot(self.x, self.y, self.z)

    def length_sqr(self) -> float:
        return self.x * self.x + self.y * self.y + self.z * self.z

    def distance(self, other: "AbstractVector") -> float:
        return math.hypot(self.x - other.x, self.y - other.y, self.z - other.z)

    def distance_sqr(self, other: "AbstractVector") -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return dx * dx + dy * dy + dz * dz

    def dot(self, other: "AbstractVector") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def angle(self, other: "AbstractVector") -> float:
        dot = self.dot(other)
        len_self = self.length()
        len_other = other.length()
        if len_self == 0 or len_other == 0:
            return 0.0
        cos_theta = max(min(dot / (len_self * len_other), 1.0), -1.0)
        return math.acos(cos_theta)

    def clamp(self, a: Union["AbstractVector", float], b: Union["AbstractVector", float]) -> "Vec3":
        x, y, z = self.x, self.y, self.z
        if isinstance(a, Vec3) and isinstance(b, Vec3):
            x = max(a.x, min(x, b.x))
            y = max(a.y, min(y, b.y))
            z = max(a.z, min(z, b.z))
        else:
            x = max(a, min(x, b))
            y = max(a, min(y, b))
            z = max(a, min(z, b))
        return Vec3(x, y, z)

    def clamp_length(self, a: float, b: float) -> "Vec3":
        l = self.length()
        if l == 0:
            return Vec3(0, 0, 0)
        scale = max(a, min(l, b)) / l
        return self * scale

    def apply(self, func: Callable[[float], float]) -> "Vec3":
        return Vec3(func(self.x), func(self.y), func(self.z))

    def normalize(self) -> "Vec3":
        l = self.length()
        if l == 0:
            return Vec3(0, 0, 0)
        return self / l

    def move_torwards(self, target: "Vec3", distance: float) -> "Vec3":
        direction = target - self
        d_len = direction.length()
        if d_len == 0 or distance <= 0:
            return Vec3(self.x, self.y, self.z)
        if distance >= d_len:
            return Vec3(target.x, target.y, target.z)
        return self + direction * (distance / d_len)

    def lerp(self, target: "Vec3", amt: float) -> "Vec3":
        return Vec3(self.x + (target.x - self.x) * amt, self.y + (target.y - self.y) * amt, self.z + (target.z - self.z) * amt)

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float | int) -> "Vec3":
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar: float | int) -> "Vec3":
        if scalar == 0:
            raise ZeroDivisionError("division by zero in Vec3")
        return Vec3(self.x / scalar, self.y / scalar, self.z / scalar)

    def __getitem__(self, index: int) -> float:
        if index in (0, -3):
            return self.x
        if index in (1, -2):
            return self.y
        if index in (2, -1):
            return self.z
        raise IndexError("Vec3 index out of range")

    def __getattr__(self, name: str) -> float:
        if name == "r":
            return self.x
        if name == "g":
            return self.y
        if name == "b":
            return self.z
        raise AttributeError(f"'Vec3' object has no attribute '{name}'")

    def __setitem__(self, index: int, val: float):
        if index in (0, -3):
            self.x = val
        elif index in (1, -2):
            self.y = val
        elif index in (2, -1):
            self.z = val
        else:
            raise IndexError("Vec3 index out of range")

    def __hash__(self) -> int:
        return hash((self.x, self.y, self.z))

    def __repr__(self) -> str:
        return f"Vec3(x={self.x}, y={self.y}, z={self.z})"

    def __str__(self) -> str:
        return f"({self.x}, {self.y}, {self.z})"


class Vec4(AbstractVector):
    __slots__ = ("x", "y", "z", "w")  # noqa: RUF023

    def __init__(
        self,
        a: Union[float, int, "Vec2", "Vec3", "Vec4"] = 0.0,
        b: Union[float, int, "Vec2", "Vec3", None] | None = None,
        c: Union[float, int, "Vec2", None] | None = None,
        d: float | int | None | None = None,
    ):
        super().__init__(4)

        # Copy constructor
        if isinstance(a, Vec4):
            self.x, self.y, self.z, self.w = a.x, a.y, a.z, a.w
        # Vec3 + scalar
        elif isinstance(a, Vec3) and isinstance(b, (float, int)):
            self.x, self.y, self.z, self.w = a.x, a.y, a.z, b
        elif isinstance(b, Vec3) and isinstance(a, (float, int)):
            self.x, self.y, self.z, self.w = a, b.x, b.y, b.z
        # Vec2 + scalar + scalar
        elif isinstance(a, Vec2) and isinstance(b, (float, int)) and isinstance(c, (float, int)):
            self.x, self.y, self.z, self.w = a.x, a.y, b, c
        elif isinstance(b, Vec2) and isinstance(a, (float, int)) and isinstance(c, (float, int)):
            self.x, self.y, self.z, self.w = a, b.x, b.y, c
        elif isinstance(c, Vec2) and isinstance(a, (float, int)) and isinstance(b, (float, int)):
            self.x, self.y, self.z, self.w = a, b, c.x, c.y
        # All scalars
        elif all(isinstance(v, (float, int)) for v in (a, b, c, d) if v is not None):
            vals = [v for v in (a, b, c, d) if v is not None]
            if len(vals) == 1:
                self.x = self.y = self.z = self.w = vals[0]
            else:
                while len(vals) < 4:
                    vals.append(vals[-1])
                self.x, self.y, self.z, self.w = vals
        else:
            raise TypeError("Invalid Vec4 constructor arguments")

    def almost_equals(self, other: "AbstractVector") -> bool:
        return all(math.isclose(a, b) for a, b in zip(self, other, strict=False))

    def length(self) -> float:
        return math.hypot(self.x, self.y, self.z, self.w)

    def length_sqr(self) -> float:
        return self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w

    def distance(self, other: "AbstractVector") -> float:
        return math.hypot(self.x - other.x, self.y - other.y, self.z - other.z, self.w - other.w)

    def distance_sqr(self, other: "AbstractVector") -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        dw = self.w - other.w
        return dx * dx + dy * dy + dz * dz + dw * dw

    def dot(self, other: "AbstractVector") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w

    def angle(self, other: "AbstractVector") -> float:
        len_self = self.length()
        len_other = other.length()
        if len_self == 0 or len_other == 0:
            return 0.0
        cos_theta = max(min(self.dot(other) / (len_self * len_other), 1.0), -1.0)
        return math.acos(cos_theta)

    def clamp(self, a: Union["AbstractVector", float], b: Union["AbstractVector", float]) -> "Vec4":
        x, y, z, w = self.x, self.y, self.z, self.w
        if isinstance(a, Vec4) and isinstance(b, Vec4):
            x = max(a.x, min(x, b.x))
            y = max(a.y, min(y, b.y))
            z = max(a.z, min(z, b.z))
            w = max(a.w, min(w, b.w))
        else:
            x = max(a, min(x, b))
            y = max(a, min(y, b))
            z = max(a, min(z, b))
            w = max(a, min(w, b))
        return Vec4(x, y, z, w)

    def clamp_length(self, a: float, b: float) -> "Vec4":
        l = self.length()
        if l == 0:
            return Vec4(0, 0, 0, 0)
        scale = max(a, min(l, b)) / l
        return self * scale

    def apply(self, func: Callable[[float], float]) -> "Vec4":
        return Vec4(func(self.x), func(self.y), func(self.z), func(self.w))

    def normalize(self) -> "Vec4":
        l = self.length()
        if l == 0:
            return Vec4(0, 0, 0, 0)
        return self / l

    def move_torwards(self, target: "Vec4", distance: float) -> "Vec4":
        direction = target - self
        d_len = direction.length()
        if d_len == 0 or distance <= 0:
            return Vec4(self.x, self.y, self.z, self.w)
        if distance >= d_len:
            return Vec4(target.x, target.y, target.z, target.w)
        return self + direction * (distance / d_len)

    def lerp(self, target: "Vec4", amt: float) -> "Vec4":
        return Vec4(self.x + (target.x - self.x) * amt, self.y + (target.y - self.y) * amt, self.z + (target.z - self.z) * amt, self.w + (target.w - self.w) * amt)

    def __add__(self, other: "Vec4") -> "Vec4":
        return Vec4(self.x + other.x, self.y + other.y, self.z + other.z, self.w + other.w)

    def __sub__(self, other: "Vec4") -> "Vec4":
        return Vec4(self.x - other.x, self.y - other.y, self.z - other.z, self.w - other.w)

    def __mul__(self, scalar: float | int) -> "Vec4":
        return Vec4(self.x * scalar, self.y * scalar, self.z * scalar, self.w * scalar)

    def __truediv__(self, scalar: float | int) -> "Vec4":
        if scalar == 0:
            raise ZeroDivisionError("division by zero in Vec4")
        return Vec4(self.x / scalar, self.y / scalar, self.z / scalar, self.w / scalar)

    def __getitem__(self, index: int) -> float:
        if index in (0, -4):
            return self.x
        if index in (1, -3):
            return self.y
        if index in (2, -2):
            return self.z
        if index in (3, -1):
            return self.w
        raise IndexError("Vec4 index out of range")

    def __getattr__(self, name: str) -> float:
        if name == "r":
            return self.x
        if name == "g":
            return self.y
        if name == "b":
            return self.z
        if name == "a":
            return self.w
        raise AttributeError(f"'Vec4' object has no attribute '{name}'")

    def __setitem__(self, index: int, val: float):
        if index in (0, -4):
            self.x = val
        elif index in (1, -3):
            self.y = val
        elif index in (2, -2):
            self.z = val
        elif index in (3, -1):
            self.w = val
        else:
            raise IndexError("Vec4 index out of range")

    def __hash__(self) -> int:
        return hash((self.x, self.y, self.z, self.w))

    def __repr__(self) -> str:
        return f"Vec4(x={self.x}, y={self.y}, z={self.z}, w={self.w})"

    def __str__(self) -> str:
        return f"({self.x}, {self.y}, {self.z}, {self.w})"


if __name__ == "__main__":
    import unittest

    class TestVectors(unittest.TestCase):
        def test_vec2_construction(self):
            v = Vec2(1)
            self.assertEqual((v.x, v.y), (1, 1))
            v = Vec2(1, 2)
            self.assertEqual((v.x, v.y), (1, 2))
            v2 = Vec2(v)
            self.assertEqual((v2.x, v2.y), (1, 2))

        def test_vec2_arithmetic(self):
            v1 = Vec2(1, 2)
            v2 = Vec2(3, 4)
            self.assertEqual(v1 + v2, Vec2(4, 6))
            self.assertEqual(v2 - v1, Vec2(2, 2))
            self.assertEqual(v1 * 2, Vec2(2, 4))
            self.assertEqual(2 * v1, Vec2(2, 4))
            self.assertEqual(v2 / 2, Vec2(1.5, 2))

        def test_vec2_indexing_and_iter(self):
            v = Vec2(1, 2)
            self.assertEqual(v[0], 1)
            self.assertEqual(v[1], 2)
            v[0] = 5
            self.assertEqual(v.x, 5)
            self.assertEqual(list(v), [5, 2])
            self.assertEqual(len(v), 2)

        def test_vec2_length_distance_dot(self):
            v1 = Vec2(3, 4)
            v2 = Vec2(0, 0)
            self.assertAlmostEqual(v1.length(), 5)
            self.assertAlmostEqual(v1.distance(v2), 5)
            self.assertEqual(v1.dot(v2), 0)
            self.assertEqual(v1.dot(Vec2(1, 0)), 3)

        def test_vec2_normalize_lerp(self):
            v = Vec2(3, 4)
            n = v.normalize()
            self.assertAlmostEqual(n.length(), 1)
            lerp_v = Vec2(0, 0).lerp(Vec2(2, 2), 0.5)
            self.assertEqual(lerp_v, Vec2(1, 1))

        def test_vec2_clamp(self):
            v = Vec2(5, -1)
            clamped = v.clamp(0, 3)
            self.assertEqual(clamped, Vec2(3, 0))
            clamped_vec = v.clamp(Vec2(1, -2), Vec2(4, 2))
            self.assertEqual(clamped_vec, Vec2(4, -1))

        def test_vec2_clamp_length(self):
            v = Vec2(3, 4)  # length = 5
            clamped = v.clamp_length(2, 4)
            self.assertAlmostEqual(clamped.length(), 4)
            clamped = v.clamp_length(6, 10)
            self.assertAlmostEqual(clamped.length(), 6)

        def test_vec2_move_torwards(self):
            v = Vec2(0, 0)
            target = Vec2(3, 4)
            moved = v.move_torwards(target, 2)
            self.assertAlmostEqual(moved.length(), 2)
            moved_full = v.move_torwards(target, 10)
            self.assertEqual(moved_full, target)

        def test_vec3_construction(self):
            v = Vec3(1)
            self.assertEqual((v.x, v.y, v.z), (1, 1, 1))
            v = Vec3(Vec2(1, 2), 3)
            self.assertEqual((v.x, v.y, v.z), (1, 2, 3))
            v = Vec3(0, Vec2(1, 2))
            self.assertEqual((v.x, v.y, v.z), (0, 1, 2))
            v = Vec3(1, 2, 3)
            self.assertEqual((v.x, v.y, v.z), (1, 2, 3))

        def test_vec3_arithmetic(self):
            v1 = Vec3(1, 2, 3)
            v2 = Vec3(4, 5, 6)
            self.assertEqual(v1 + v2, Vec3(5, 7, 9))
            self.assertEqual(v2 - v1, Vec3(3, 3, 3))
            self.assertEqual(v1 * 2, Vec3(2, 4, 6))
            self.assertEqual(2 * v1, Vec3(2, 4, 6))
            self.assertEqual(v2 / 2, Vec3(2, 2.5, 3))

        def test_vec3_indexing_iter_length(self):
            v = Vec3(1, 2, 3)
            self.assertEqual(v[0], 1)
            self.assertEqual(v[1], 2)
            self.assertEqual(v[2], 3)
            v[0] = 10
            self.assertEqual(v.x, 10)
            self.assertEqual(list(v), [10, 2, 3])
            self.assertEqual(len(v), 3)

        def test_vec3_length_distance_dot(self):
            v1 = Vec3(1, 2, 2)
            v2 = Vec3(0, 0, 0)
            self.assertAlmostEqual(v1.length(), 3)
            self.assertAlmostEqual(v1.distance(v2), 3)
            self.assertEqual(v1.dot(Vec3(1, 0, 0)), 1)

        def test_vec3_normalize_lerp(self):
            v = Vec3(0, 3, 4)
            n = v.normalize()
            self.assertAlmostEqual(n.length(), 1)
            lerp_v = Vec3(0, 0, 0).lerp(Vec3(2, 2, 2), 0.5)
            self.assertEqual(lerp_v, Vec3(1, 1, 1))

        def test_vec3_clamp(self):
            v = Vec3(5, -1, 2)
            clamped = v.clamp(0, 3)
            self.assertEqual(clamped, Vec3(3, 0, 2))
            clamped_vec = v.clamp(Vec3(1, -2, 0), Vec3(4, 2, 3))
            self.assertEqual(clamped_vec, Vec3(4, -1, 2))

        def test_vec3_clamp_length(self):
            v = Vec3(3, 4, 12)  # length = 13
            clamped = v.clamp_length(5, 10)
            self.assertAlmostEqual(clamped.length(), 10)
            clamped = v.clamp_length(15, 20)
            self.assertAlmostEqual(clamped.length(), 15)

        def test_vec3_move_torwards(self):
            v = Vec3(0, 0, 0)
            target = Vec3(1, 2, 2)
            moved = v.move_torwards(target, 2)
            self.assertAlmostEqual(moved.length(), 2)
            moved_full = v.move_torwards(target, 10)
            self.assertEqual(moved_full, target)

        def test_vec4_construction(self):
            v = Vec4(1)
            self.assertEqual((v.x, v.y, v.z, v.w), (1, 1, 1, 1))
            v = Vec4(Vec3(1, 2, 3), 4)
            self.assertEqual((v.x, v.y, v.z, v.w), (1, 2, 3, 4))
            v = Vec4(0, Vec3(1, 2, 3))
            self.assertEqual((v.x, v.y, v.z, v.w), (0, 1, 2, 3))
            v = Vec4(Vec2(1, 2), 3, 4)
            self.assertEqual((v.x, v.y, v.z, v.w), (1, 2, 3, 4))
            v = Vec4(0, Vec2(1, 2), 3)
            self.assertEqual((v.x, v.y, v.z, v.w), (0, 1, 2, 3))
            v = Vec4(0, 1, Vec2(2, 3))
            self.assertEqual((v.x, v.y, v.z, v.w), (0, 1, 2, 3))

        def test_vec4_arithmetic(self):
            v1 = Vec4(1, 2, 3, 4)
            v2 = Vec4(4, 3, 2, 1)
            self.assertEqual(v1 + v2, Vec4(5, 5, 5, 5))
            self.assertEqual(v2 - v1, Vec4(3, 1, -1, -3))
            self.assertEqual(v1 * 2, Vec4(2, 4, 6, 8))
            self.assertEqual(2 * v1, Vec4(2, 4, 6, 8))
            self.assertEqual(v2 / 2, Vec4(2, 1.5, 1, 0.5))

        def test_vec4_indexing_iter_length(self):
            v = Vec4(1, 2, 3, 4)
            self.assertEqual(v[0], 1)
            self.assertEqual(v[1], 2)
            self.assertEqual(v[2], 3)
            self.assertEqual(v[3], 4)
            v[0] = 10
            self.assertEqual(v.x, 10)
            self.assertEqual(list(v), [10, 2, 3, 4])
            self.assertEqual(len(v), 4)

        def test_vec4_length_distance_dot(self):
            v1 = Vec4(1, 2, 2, 1)
            v2 = Vec4(0, 0, 0, 0)
            self.assertAlmostEqual(v1.length(), 3.1622776601683794)
            self.assertAlmostEqual(v1.distance(v2), 3.1622776601683794)
            self.assertEqual(v1.dot(Vec4(1, 0, 0, 0)), 1)

        def test_vec4_normalize_lerp(self):
            v = Vec4(0, 3, 4, 0)
            n = v.normalize()
            self.assertAlmostEqual(n.length(), 1)
            lerp_v = Vec4(0, 0, 0, 0).lerp(Vec4(2, 2, 2, 2), 0.5)
            self.assertEqual(lerp_v, Vec4(1, 1, 1, 1))

        def test_vec4_clamp(self):
            v = Vec4(5, -1, 2, 10)
            clamped = v.clamp(0, 3)
            self.assertEqual(clamped, Vec4(3, 0, 2, 3))
            clamped_vec = v.clamp(Vec4(1, -2, 0, 5), Vec4(4, 2, 3, 7))
            self.assertEqual(clamped_vec, Vec4(4, -1, 2, 7))

        def test_vec4_clamp_length(self):
            v = Vec4(3, 4, 12, 0)  # length = 13
            clamped = v.clamp_length(5, 10)
            self.assertAlmostEqual(clamped.length(), 10)
            clamped = v.clamp_length(15, 20)
            self.assertAlmostEqual(clamped.length(), 15)

        def test_vec4_move_torwards(self):
            v = Vec4(0, 0, 0, 0)
            target = Vec4(1, 2, 2, 1)
            moved = v.move_torwards(target, 2)
            self.assertAlmostEqual(moved.length(), 2)
            moved_full = v.move_torwards(target, 10)
            self.assertEqual(moved_full, target)

        def test_vec2_from_angle(self):
            v = Vec2.from_angle(0)
            self.assertAlmostEqual(v.x, 1)
            self.assertAlmostEqual(v.y, 0)

            v = Vec2.from_angle(math.pi/2)
            self.assertAlmostEqual(v.x, 0, places=7)
            self.assertAlmostEqual(v.y, 1, places=7)

        def test_vec2_rotate_origin(self):
            v = Vec2(1, 0)
            r = v.rotate(math.pi/2)  # rotate 90° CCW around origin
            self.assertAlmostEqual(r.x, 0, places=7)
            self.assertAlmostEqual(r.y, 1, places=7)

        def test_vec2_rotate_custom_origin(self):
            v = Vec2(2, 1)
            origin = Vec2(1, 1)
            r = v.rotate(math.pi/2, origin)
            self.assertAlmostEqual(r.x, 1, places=7)
            self.assertAlmostEqual(r.y, 2, places=7)

        def test_vec3_from_polar(self):
            v = Vec3.from_polar(0, 0)
            self.assertAlmostEqual(v.x, 1)
            self.assertAlmostEqual(v.y, 0)
            self.assertAlmostEqual(v.z, 0)

            v = Vec3.from_polar(math.pi/2, 0)  # pitch 90°
            self.assertAlmostEqual(v.x, 0, places=7)
            self.assertAlmostEqual(v.y, 1, places=7)
            self.assertAlmostEqual(v.z, 0, places=7)

            # Using Vec2 input
            v = Vec3.from_polar(Vec2(math.pi/2, math.pi/2))
            self.assertAlmostEqual(v.x, 0, places=7)
            self.assertAlmostEqual(v.y, 1, places=7)
            self.assertAlmostEqual(v.z, 0, places=7)

        def test_vec3_rotate_x(self):
            v = Vec3(0, 1, 0)
            r = v.rotate_x(math.pi/2)
            self.assertAlmostEqual(r.x, 0, places=7)
            self.assertAlmostEqual(r.y, 0, places=7)
            self.assertAlmostEqual(r.z, 1, places=7)

        def test_vec3_rotate_y(self):
            v = Vec3(1, 0, 0)
            r = v.rotate_y(math.pi/2)
            self.assertAlmostEqual(r.x, 0, places=7)
            self.assertAlmostEqual(r.y, 0, places=7)
            self.assertAlmostEqual(r.z, -1, places=7)

        def test_vec3_rotate_z(self):
            v = Vec3(1, 0, 0)
            r = v.rotate_z(math.pi/2)
            self.assertAlmostEqual(r.x, 0, places=7)
            self.assertAlmostEqual(r.y, 1, places=7)
            self.assertAlmostEqual(r.z, 0, places=7)

        def test_vec3_rgba_attributes(self):
            v3 = Vec3(1, 2, 3)
            self.assertEqual(v3.r, v3.x)
            self.assertEqual(v3.g, v3.y)
            self.assertEqual(v3.b, v3.z)
            with self.assertRaises(AttributeError):
                _ = v3.a

        def test_vec4_rgba_attributes(self):
            v4 = Vec4(1, 2, 3, 4)
            self.assertEqual(v4.r, v4.x)
            self.assertEqual(v4.g, v4.y)
            self.assertEqual(v4.b, v4.z)
            self.assertEqual(v4.a, v4.w)
            with self.assertRaises(AttributeError):
                _ = v4.q  # invalid attribute

    unittest.main()
