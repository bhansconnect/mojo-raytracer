from algorithm import parallelize
from math import sqrt, rsqrt, pow, tan
from pcg import Pcg32
from python import Python
from python.object import PythonObject
from time import time_function


struct Image:
    # reference count used to make the object efficiently copyable
    var rc: Pointer[Int]
    # the two dimensional image is represented as a flat array
    var pixels: Pointer[Vec3f]
    var height: Int
    var width: Int

    fn __init__(inout self, height: Int, width: Int):
        self.height = height
        self.width = width
        self.pixels = Pointer[Vec3f].alloc(self.height * self.width)
        self.rc = Pointer[Int].alloc(1)
        self.rc.store(1)

    fn __copyinit__(inout self, other: Self):
        other._inc_rc()
        self.pixels = other.pixels
        self.rc = other.rc
        self.height = other.height
        self.width = other.width

    fn __del__(owned self):
        self._dec_rc()

    fn _get_rc(self) -> Int:
        return self.rc.load()

    fn _dec_rc(self):
        var rc = self._get_rc()
        if rc > 1:
            self.rc.store(rc - 1)
            return
        self._free()

    fn _inc_rc(self):
        var rc = self._get_rc()
        self.rc.store(rc + 1)

    fn _free(self):
        self.rc.free()
        self.pixels.free()

    @always_inline
    fn set(self, row: Int, col: Int, value: Vec3f) -> None:
        self.pixels.store(self._pos_to_index(row, col), value)

    @always_inline
    fn get(self, row: Int, col: Int) -> Vec3f:
        return self.pixels.load(self._pos_to_index(row, col))

    @always_inline
    fn _pos_to_index(self, row: Int, col: Int) -> Int:
        # Convert a (rol, col) position into an index in the underlying linear storage
        return row * self.width + col

    def to_numpy_image(self) -> PythonObject:
        var np = Python.import_module("numpy")
        var plt = Python.import_module("matplotlib.pyplot")

        var np_image = np.zeros((self.height, self.width, 3), np.float32)

        # We use raw pointers to efficiently copy the pixels to the numpy array
        var out_pointer = Pointer(
            __mlir_op.`pop.index_to_pointer`[
                _type = __mlir_type[`!kgen.pointer<scalar<f32>>`]
            ](
                SIMD[DType.index, 1](
                    np_image.__array_interface__["data"][0].__index__()
                ).value
            )
        )
        var in_pointer = Pointer(
            __mlir_op.`pop.index_to_pointer`[
                _type = __mlir_type[`!kgen.pointer<scalar<f32>>`]
            ](SIMD[DType.index, 1](int(self.pixels)).value)
        )

        for row in range(self.height):
            for col in range(self.width):
                var index = self._pos_to_index(row, col)
                for dim in range(3):
                    out_pointer.store(
                        index * 3 + dim, in_pointer[index * 4 + dim]
                    )

        return np_image


def load_image(fname: String) -> Image:
    var np = Python.import_module("numpy")
    var plt = Python.import_module("matplotlib.pyplot")

    var np_image = plt.imread(fname)
    var rows = np_image.shape[0].__index__()
    var cols = np_image.shape[1].__index__()
    var image = Image(rows, cols)

    var in_pointer = Pointer(
        __mlir_op.`pop.index_to_pointer`[
            _type = __mlir_type[`!kgen.pointer<scalar<f32>>`]
        ](
            SIMD[DType.index, 1](
                np_image.__array_interface__["data"][0].__index__()
            ).value
        )
    )
    var out_pointer = Pointer(
        __mlir_op.`pop.index_to_pointer`[
            _type = __mlir_type[`!kgen.pointer<scalar<f32>>`]
        ](SIMD[DType.index, 1](int(image.pixels)).value)
    )
    for row in range(rows):
        for col in range(cols):
            var index = image._pos_to_index(row, col)
            for dim in range(3):
                out_pointer.store(index * 4 + dim, in_pointer[index * 3 + dim])
    return image


def render(image: Image):
    np = Python.import_module("numpy")
    plt = Python.import_module("matplotlib.pyplot")
    colors = Python.import_module("matplotlib.colors")

    plt.imshow(image.to_numpy_image())
    plt.axis("off")
    plt.show()


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


var pi: Float32 = 3.14159265358979323846264338327950288


@value
struct Camera(Stringable):
    var lower_left_corner: Vec3f
    var horizontal: Vec3f
    var vertical: Vec3f
    var origin: Vec3f
    var u: Vec3f
    var v: Vec3f
    var w: Vec3f
    var lens_radius: Float32

    fn __init__(
        inout self,
        lookfrom: Vec3f,
        lookat: Vec3f,
        vup: Vec3f,
        vfov: Float32,
        aspect: Float32,
        aperature: Float32,
        focus_dist: Float32,
    ):
        self.lens_radius = aperature / 2
        var theta = vfov * pi / 180
        var half_height = tan(theta / 2)
        var half_width = aspect * half_height
        self.origin = lookfrom
        self.w = (lookfrom - lookat).normalize()
        self.u = vup.cross(self.w).normalize()
        self.v = self.w.cross(self.u)
        self.lower_left_corner = (
            self.origin
            - self.u * focus_dist * half_width
            - self.v * focus_dist * half_height
            - self.w * focus_dist
        )
        self.horizontal = self.u * 2 * half_width * focus_dist
        self.vertical = self.v * 2 * half_height * focus_dist

    fn get_ray(self, s: Float32, t: Float32, inout rng: Pcg32) -> Ray:
        var rd = random_in_unit_sphere(rng) * self.lens_radius
        var offset = self.u * rd.x() + self.v * rd.y()
        return Ray(
            self.origin + offset,
            self.lower_left_corner
            + self.horizontal * s
            + self.vertical * t
            - self.origin
            - offset,
        )

    fn __str__(self) -> String:
        return (
            "("
            + str(self.lower_left_corner)
            + ", "
            + str(self.horizontal)
            + ", "
            + str(self.vertical)
            + ", "
            + str(self.origin)
            + ")"
        )


@value
@register_passable("trivial")
struct Ray(Stringable):
    var origin: Vec3f
    var direction: Vec3f

    fn __str__(self) -> String:
        return "(" + str(self.origin) + ", " + str(self.direction) + ")"

    fn point_at_parameter(self, t: Float32) -> Vec3f:
        return self.origin + self.direction * t


@value
@register_passable("trivial")
struct HitRecord:
    var t: Float32
    var p: Vec3f
    var normal: Vec3f
    var material_index: Int

    fn __str__(self) -> String:
        return (
            "("
            + str(self.t)
            + ", "
            + str(self.p)
            + ", "
            + str(self.normal)
            + ", "
            + str(self.material_index)
            + ")"
        )


trait Hitable:
    fn hit(
        self, ray: Ray, t_min: Float32, t_max: Float32, inout rec: HitRecord
    ) -> Bool:
        ...


@value
@register_passable("trivial")
struct MaterialKind(Stringable):
    var value: UInt8

    alias metal = MaterialKind(0)
    alias dielectric = MaterialKind(1)
    alias lambertian = MaterialKind(2)

    fn __init__(inout self, i: UInt8):
        self.value = i

    fn __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    fn __str__(self) -> String:
        if self == Self.metal:
            return "metal"
        elif self == Self.dielectric:
            return "dielectric"
        elif self == Self.lambertian:
            return "lambertian"
        else:
            return abort[String]("Unreachable: unknown MaterialKind")


fn reflect(v: Vec3f, n: Vec3f) -> Vec3f:
    return v - n * (v @ n * 2.0)


fn refract(
    v: Vec3f, n: Vec3f, ni_over_nt: Float32, inout refracted: Vec3f
) -> Bool:
    var uv = v.normalize()
    var dt = uv @ n
    var disc = 1 - ni_over_nt * ni_over_nt * (1 - dt * dt)
    if disc > 0:
        refracted = (uv - n * dt) * ni_over_nt - n * sqrt(disc)
        return True
    return False


fn schlick(cosine: Float32, ref_idx: Float32) -> Float32:
    var r0 = (1 - ref_idx) / (1 + ref_idx)
    r0 = r0 * r0
    return r0 + (1 - r0) * pow(1 - cosine, 5)


@value
@register_passable("trivial")
struct Material(Stringable):
    var kind: MaterialKind
    var albedo: Vec3f
    var data: Float32

    fn __str__(self) -> String:
        return (
            "("
            + str(self.kind)
            + ", "
            + str(self.albedo)
            + ", "
            + str(self.data)
            + ")"
        )

    fn scatter(
        self,
        r_in: Ray,
        inout rec: HitRecord,
        inout attenuation: Vec3f,
        inout scattered: Ray,
        inout rng: Pcg32,
    ) -> Bool:
        var rand = random_in_unit_sphere(rng)
        if self.kind == MaterialKind.metal:
            var reflected = reflect(r_in.direction.normalize(), rec.normal)
            scattered = Ray(rec.p, reflected + rand * self.data)
            attenuation = self.albedo
            return scattered.direction @ rec.normal > 0
        if self.kind == MaterialKind.dielectric:
            var reflected = reflect(r_in.direction, rec.normal)
            attenuation = Vec3f.one()
            var outward_normal: Vec3f
            var ni_over_nt: Float32
            var refracted = Vec3f.zero()
            var reflect_prob: Float32
            var cosine: Float32
            if r_in.direction @ rec.normal > 0:
                outward_normal = -rec.normal
                ni_over_nt = self.data
                cosine = (r_in.direction @ rec.normal) * rsqrt(
                    r_in.direction @ r_in.direction
                )
                cosine = sqrt(1 - self.data * self.data * (1 - cosine * cosine))
            else:
                outward_normal = rec.normal
                ni_over_nt = 1 / self.data
                cosine = -(r_in.direction @ rec.normal) * rsqrt(
                    r_in.direction @ r_in.direction
                )
            if refract(r_in.direction, outward_normal, ni_over_nt, refracted):
                reflect_prob = schlick(cosine, self.data)
            else:
                reflect_prob = 1
            if rng.random_f32() < reflect_prob:
                scattered = Ray(rec.p, reflected)
            else:
                scattered = Ray(rec.p, refracted)
            return True
        if self.kind == MaterialKind.lambertian:
            var target = rec.p + rec.normal + rand
            scattered = Ray(rec.p, target - rec.p)
            attenuation = self.albedo
            return True
        else:
            return abort[Bool]("Unreachable: unknown MaterialKind")


@value
@register_passable("trivial")
struct Sphere(Hitable, Stringable):
    var center: Vec3f
    var radius: Float32
    var material_index: Int

    fn __str__(self) -> String:
        return (
            "("
            + str(self.center)
            + ", "
            + str(self.radius)
            + ", "
            + str(self.material_index)
            + ")"
        )

    fn hit(
        self, ray: Ray, t_min: Float32, t_max: Float32, inout rec: HitRecord
    ) -> Bool:
        var oc = ray.origin - self.center
        var a = ray.direction @ ray.direction
        var b = oc @ ray.direction
        var c = oc @ oc - self.radius * self.radius
        var disc = b * b - a * c
        if disc > 0:
            var tmp = (-b - sqrt(disc)) / a
            if tmp < t_max and tmp > t_min:
                rec.t = tmp
                rec.p = ray.point_at_parameter(tmp)
                rec.normal = (rec.p - self.center) * (1 / self.radius)
                rec.material_index = self.material_index
                return True
            tmp = (-b + sqrt(disc)) / a
            if tmp < t_max and tmp > t_min:
                rec.t = tmp
                rec.p = ray.point_at_parameter(tmp)
                rec.normal = (rec.p - self.center) * (1 / self.radius)
                rec.material_index = self.material_index
                return True
        return False


fn hit_any(
    ray: Ray,
    world: List[Sphere],
    t_min: Float32,
    t_max: Float32,
    inout rec: HitRecord,
) -> Bool:
    var tmp_rec = HitRecord(0, Vec3f.zero(), Vec3f.zero(), 0)
    var hit_anything = False
    var closest_yet = t_max
    for i in range(len(world)):
        if world[i].hit(ray, t_min, closest_yet, tmp_rec):
            hit_anything = True
            closest_yet = tmp_rec.t
            rec = tmp_rec
    return hit_anything


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


fn color(
    ray: Ray, world: List[Sphere], materials: List[Material], inout rng: Pcg32
) -> Vec3f:
    var current_ray = ray
    var current_attenuation = Vec3f.one()
    for _ in range(50):
        var rec = HitRecord(0, Vec3f.zero(), Vec3f.zero(), 0)
        if hit_any(current_ray, world, 0.001, Float32.MAX_FINITE, rec):
            var scattered = Ray(Vec3f.zero(), Vec3f.zero())
            var attenuation = Vec3f.zero()
            var mat = materials[rec.material_index]
            if mat.scatter(current_ray, rec, attenuation, scattered, rng):
                current_attenuation *= attenuation
                current_ray = scattered
            else:
                return Vec3f.zero()
        else:
            var unit_dir = ray.direction.normalize()
            var t = 0.5 * (unit_dir.y() + 1.0)
            var c = Vec3f.one() * (1.0 - t) + Vec3f(0.5, 0.7, 1.0) * t
            return current_attenuation * c
    return Vec3f.zero()


def main():
    width = 1200
    height = 800
    samples = 10

    image = Image(height, width)

    lookfrom = Vec3f(13, 2, 3)
    lookat = Vec3f(0, 0, 0)
    cam = Camera(
        lookfrom,
        lookat,
        vup=Vec3f(0, 1, 0),
        vfov=30.0,
        aspect=Float32(width) / height,
        aperature=0.1,
        focus_dist=10.0,
        # focus_dist = (lookfrom - lookat).length(),
    )
    materials = List[Material](
        Material(MaterialKind.lambertian, Vec3f(0.5, 0.5, 0.5), 0),
        Material(MaterialKind.dielectric, Vec3f.zero(), 1.5),
    )
    world = List[Sphere](Sphere(Vec3f(0, -1000, -1), 1000, 0))

    var items = 22 * 22 + 1 + 3
    materials.reserve(items)
    world.reserve(items)

    seed = 42
    var world_rng = Pcg32(seed)

    for a in range(-11, 11):
        for b in range(-11, 11):
            var choose_mat = world_rng.random_f32()
            var center = Vec3f(
                a + world_rng.random_f32(), 0.2, b + world_rng.random_f32()
            )
            if choose_mat < 0.8:
                var r = world_rng.random_f32() * world_rng.random_f32()
                var g = world_rng.random_f32() * world_rng.random_f32()
                var b = world_rng.random_f32() * world_rng.random_f32()
                materials.append(
                    Material(MaterialKind.lambertian, Vec3f(r, g, b), 0)
                )
                world.append(Sphere(center, 0.2, len(materials) - 1))
            if choose_mat < 0.95:
                var r = 0.5 * (1.0 + world_rng.random_f32())
                var g = world_rng.random_f32() * world_rng.random_f32()
                var b = world_rng.random_f32() * world_rng.random_f32()
                materials.append(
                    Material(
                        MaterialKind.metal,
                        Vec3f(r, g, b),
                        0.5 * world_rng.random_f32(),
                    )
                )
                world.append(Sphere(center, 0.2, len(materials) - 1))
            else:
                world.append(Sphere(center, 0.2, 1))

    world.append(Sphere(Vec3f(0, 1, 0), 1.0, 1))
    materials.append(Material(MaterialKind.lambertian, Vec3f(0.4, 0.2, 0.1), 0))
    world.append(Sphere(Vec3f(-4, 1, 0), 1.0, len(materials) - 1))
    materials.append(Material(MaterialKind.metal, Vec3f(0.7, 0.6, 0.5), 0))
    world.append(Sphere(Vec3f(4, 1, 0), 1.0, len(materials) - 1))

    @parameter
    fn _process_row(row: Int):
        var rng = Pcg32(seed, row)
        for col in range(image.width):
            var c = Vec3f.zero()
            for _ in range(samples):
                var v = (Float32(row) + rng.random_f32()) / image.height
                var u = (Float32(col) + rng.random_f32()) / image.width
                var ray = cam.get_ray(u, v, rng)
                c = c + color(ray, world, materials, rng)

            image.set(
                image.height - row - 1,
                col,
                (c * (1.0 / samples)).sqrt(),
            )

    @parameter
    fn _process():
        parallelize[_process_row](image.height)
        # for row in range(image.height):
        #     _process_row(row)

    var nanos = time_function[_process]()

    print("Rendering took " + str(nanos // 1_000_000) + "ms")
    _ = world[0]
    _ = materials[0]
    render(image)
