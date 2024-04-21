from primitives import Vec3f
from python import Python
from python.object import PythonObject

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


    def render(self: Image):
        np = Python.import_module("numpy")
        plt = Python.import_module("matplotlib.pyplot")
        colors = Python.import_module("matplotlib.colors")

        plt.imshow(self.to_numpy_image())
        plt.axis("off")
        plt.show()
