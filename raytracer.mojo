from algorithm import parallelize
from math import sqrt, rsqrt, pow, tan
from pcg import Pcg32
from primitives import Vec3f, Ray, random_in_unit_disk, random_in_unit_sphere
from camera import Camera
from image import Image
from time import time_function
from hitable import HitRecord, Hitable, Sphere, hit_any
from material import Material, MaterialKind


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

    img = Image(height, width)

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
        for col in range(img.width):
            var c = Vec3f.zero()
            for _ in range(samples):
                var v = (Float32(row) + rng.random_f32()) / img.height
                var u = (Float32(col) + rng.random_f32()) / img.width
                var ray = cam.get_ray(u, v, rng)
                c = c + color(ray, world, materials, rng)

            img.set(
                img.height - row - 1,
                col,
                (c * (1.0 / samples)).sqrt(),
            )

    @parameter
    fn _process():
        parallelize[_process_row](img.height)
        # for row in range(img.height):
        #     _process_row(row)

    var nanos = time_function[_process]()

    print("Rendering took " + str(nanos // 1_000_000) + "ms")
    _ = world[0]
    _ = materials[0]
    img.render()
