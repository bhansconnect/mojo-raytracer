from math import sqrt, pow, rsqrt
from primitives import Vec3f, Ray, random_in_unit_sphere
from pcg import Pcg32
from hittable import HitRecord


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
