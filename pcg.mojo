from math import ldexp


@register_passable("trivial")
struct Pcg32:
    var _state: UInt64
    var _inc: UInt64

    fn __init__(inout self, init_state: UInt64, init_seq: UInt64):
        self._state = 0
        self._inc = (init_seq << 1) | 1

        _ = self.random_u32()
        self._state += init_state
        _ = self.random_u32()

    fn __init__(inout self, init_state: UInt64):
        self._state = 0
        self._inc = 2891336453

        _ = self.random_u32()
        self._state += init_state
        _ = self.random_u32()

    fn random_u32(inout self) -> UInt32:
        var old_state = self._state
        self._state = old_state * 6364136223846793005 + self._inc
        var xorshifted: UInt32 = (((old_state >> 18) ^ old_state) >> 27).cast[
            DType.uint32
        ]()
        var rot: UInt32 = (old_state >> 59).cast[DType.uint32]()
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31))

    fn random_f32(inout self) -> Float32:
        var u32 = self.random_u32()
        return ldexp(u32.cast[DType.float32](), -32)
