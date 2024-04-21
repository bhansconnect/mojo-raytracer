# [Raytracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html) in [Mojo](https://github.com/modularml/mojo)

This project is specifically based on the [Accelerated Ray Tracing in One Weekend in CUDA tutorial](https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/) ([Repo](https://github.com/rogerallen/raytracinginoneweekendincuda)).

You need to have `matplotlib` and `numpy` installed in your python environment to run the project.
Then simply `mojo raytracer.mojo`.

The program directly creates the image in memory and displays it instead of printing to stdout.
It also processes rows in parallel and uses a simd vec3f type for performance.
Otherwise, it has not changed much from the original c/cuda versions.
