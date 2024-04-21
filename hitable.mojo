from math import sqrt
from primitives import Vec3f, Ray

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
    # TODO: can mojo make this a dynamic list of Hitables?
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


