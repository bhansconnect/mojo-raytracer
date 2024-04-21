from math import rsqrt, sqrt
from pcg import Pcg32

@register_passable("trivial")
struct Vec3f(Stringable):
    var data: SIMD[DType.float32, 4]

    @always_inline
    fn __init__(x: Float32, y: Float32, z: Float32) -> Self:
        return Vec3f {data: SIMD[DType.float32, 4](x, y, z, 0)}

    @always_inline
    fn __init__(data: SIMD[DType.float32, 4]) -> Self:
        return Vec3f {data: data}

    fn __str__(self) -> String:
        return (
            "("
            + str(self.x())
            + ", "
            + str(self.y())
            + ", "
            + str(self.z())
            + ")"
        )

    @always_inline
    fn x(self) -> Float32:
        return self.data[0]

    @always_inline
    fn y(self) -> Float32:
        return self.data[1]

    @always_inline
    fn z(self) -> Float32:
        return self.data[2]

    @always_inline
    @staticmethod
    fn zero() -> Vec3f:
        return Vec3f(0, 0, 0)

    @always_inline
    @staticmethod
    fn one() -> Vec3f:
        return Vec3f(1, 1, 1)

    @always_inline
    fn sqrt(self) -> Vec3f:
        return sqrt(self.data)

    @always_inline
    fn __sub__(self, other: Vec3f) -> Vec3f:
        return self.data - other.data

    @always_inline
    fn __add__(self, other: Vec3f) -> Vec3f:
        return self.data + other.data

    @always_inline
    fn __add__(self, other: Float32) -> Vec3f:
        return self.data + other

    @always_inline
    fn __matmul__(self, other: Vec3f) -> Float32:
        return (self.data * other.data).reduce_add()

    @always_inline
    fn __mul__(self, k: Float32) -> Vec3f:
        return self.data * k

    @always_inline
    fn __mul__(self, other: Vec3f) -> Vec3f:
        return self.data * other.data

    @always_inline
    fn __imul__(inout self, other: Vec3f):
        self.data *= other.data

    @always_inline
    fn __neg__(self) -> Vec3f:
        return self.data * -1.0

    @always_inline
    fn __getitem__(self, idx: Int) -> SIMD[DType.float32, 1]:
        return self.data[idx]

    @always_inline
    fn cross(self, other: Vec3f) -> Vec3f:
        var self_zxy = self.data.shuffle[2, 0, 1, 3]()
        var other_zxy = other.data.shuffle[2, 0, 1, 3]()
        return (self_zxy * other.data - self.data * other_zxy).shuffle[
            2, 0, 1, 3
        ]()

    @always_inline
    fn normalize(self) -> Vec3f:
        return self.data * rsqrt(self @ self)

    @always_inline
    fn length(self) -> Float32:
        return sqrt(self @ self)


@value
@register_passable("trivial")
struct Ray(Stringable):
    var origin: Vec3f
    var direction: Vec3f

    fn __str__(self) -> String:
        return "(" + str(self.origin) + ", " + str(self.direction) + ")"

    fn point_at_parameter(self, t: Float32) -> Vec3f:
        return self.origin + self.direction * t


# TODO: test picking random polar coord and avoiding loop.
fn random_in_unit_disk(inout pcg: Pcg32) -> Vec3f:
    var p = Vec3f(pcg.random_f32(), pcg.random_f32(), 0) * 2.0 - Vec3f(1, 1, 0)
    while p @ p >= 1.0:
        p = Vec3f(pcg.random_f32(), pcg.random_f32(), 0) * 2.0 - Vec3f(1, 1, 0)
    return p


# TODO: test picking random polar coord and avoiding loop.
fn random_in_unit_sphere(inout pcg: Pcg32) -> Vec3f:
    var p = Vec3f(
        pcg.random_f32(), pcg.random_f32(), pcg.random_f32()
    ) * 2.0 - Vec3f.one()
    while p @ p >= 1.0:
        p = (
            Vec3f(pcg.random_f32(), pcg.random_f32(), pcg.random_f32()) * 2.0
            - Vec3f.one()
        )
    return p


