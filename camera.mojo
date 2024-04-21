from math import tan
from pcg import Pcg32
from primitives import Vec3f, Ray,random_in_unit_sphere

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


