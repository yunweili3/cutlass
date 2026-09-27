# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Use of this software is governed by the terms and conditions of the
# NVIDIA End User License Agreement (EULA), available at:
# https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/license.html
#
# Any use, reproduction, disclosure, or distribution of this software
# and related documentation outside the scope permitted by the EULA
# is strictly prohibited.

"""
Unit tests for `//` and `%` on dynamic signed integers and floats.

`//` floors, so `%` must take the divisor's sign for `(a // b) * b + a % b == a`
to hold, as it does in Python and for constant-folded operands. arith.remsi and
arith.remf take the dividend's sign, so `ArithValue.__mod__` corrects them.
Results are computed on the GPU, per scalar and per TensorSSA, and compared
against Python for every sign combination.
"""

import math
import unittest

import cutlass.cute as cute
from cutlass import Float32, Int32

try:
    import torch
    from cutlass.cute.runtime import from_dlpack

    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

N = 16

INT_PAIRS = [
    (7, 2), (-7, 2), (7, -2), (-7, -2),
    (6, 3), (-6, 3), (6, -3), (-6, -3),
    (0, 4), (0, -4), (-1, 4), (1, -4),
    (3, 5), (-3, 5), (3, -5), (-3, -5),
]  # fmt: skip

FLOAT_PAIRS = [
    (7.0, 2.0), (-7.0, 2.0), (7.0, -2.0), (-7.0, -2.0),
    (6.0, 3.0), (-6.0, 3.0), (6.0, -3.0), (-6.0, -3.0),
    (0.0, 4.0), (-0.0, 4.0), (0.0, -4.0), (-1.5, 4.0),
    (5.5, -2.0), (-5.5, -2.0), (-0.5, 3.0), (0.5, -3.0),
]  # fmt: skip


@cute.kernel
def _scalar_kernel(a: cute.Tensor, b: cute.Tensor, q: cute.Tensor, r: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    q[tidx] = a[tidx] // b[tidx]
    r[tidx] = a[tidx] % b[tidx]


@cute.jit
def _scalar(a: cute.Tensor, b: cute.Tensor, q: cute.Tensor, r: cute.Tensor):
    _scalar_kernel(a, b, q, r).launch(grid=(1, 1, 1), block=(N, 1, 1))


@cute.kernel
def _vector_kernel(a: cute.Tensor, b: cute.Tensor, q: cute.Tensor, r: cute.Tensor):
    va = a.load()
    vb = b.load()
    q.store(va // vb)
    r.store(va % vb)


@cute.jit
def _vector(a: cute.Tensor, b: cute.Tensor, q: cute.Tensor, r: cute.Tensor):
    _vector_kernel(a, b, q, r).launch(grid=(1, 1, 1), block=(1, 1, 1))


@unittest.skipUnless(HAS_CUDA, "requires torch and a CUDA device")
class TestFloordivModSign(unittest.TestCase):
    def _run(self, fn, pairs, dtype):
        a = torch.tensor([p[0] for p in pairs], dtype=dtype, device="cuda")
        b = torch.tensor([p[1] for p in pairs], dtype=dtype, device="cuda")
        q = torch.zeros_like(a)
        r = torch.zeros_like(a)
        fn(from_dlpack(a), from_dlpack(b), from_dlpack(q), from_dlpack(r))
        torch.cuda.synchronize()
        return q.tolist(), r.tolist()

    def _check(self, pairs, q, r, is_float):
        for (x, y), qi, ri in zip(pairs, q, r):
            with self.subTest(a=x, b=y):
                self.assertEqual(qi, x // y)
                self.assertEqual(ri, x % y)
                if is_float:
                    # Zero results take the divisor's sign as well.
                    self.assertEqual(math.copysign(1.0, ri), math.copysign(1.0, x % y))
                else:
                    self.assertEqual(qi * y + ri, x)

    def test_scalar(self):
        for dtype, pairs in [
            (torch.int32, INT_PAIRS),
            (torch.int64, INT_PAIRS),
            (torch.float32, FLOAT_PAIRS),
        ]:
            with self.subTest(dtype=dtype):
                q, r = self._run(_scalar, pairs, dtype)
                self._check(pairs, q, r, dtype.is_floating_point)

    def test_tensor_ssa(self):
        for dtype, pairs in [(torch.int32, INT_PAIRS), (torch.float32, FLOAT_PAIRS)]:
            with self.subTest(dtype=dtype):
                q, r = self._run(_vector, pairs, dtype)
                self._check(pairs, q, r, dtype.is_floating_point)

    def test_dynamic_matches_constant(self):
        """The issue's wrap-around index: `(i - 1) % n` with i = 0, n = 4."""

        @cute.kernel
        def k(out: cute.Tensor, i: Int32, n: Int32, x: Float32, y: Float32):
            out[0] = (i - 1) % n
            out[1] = (Int32(0) - 1) % Int32(4)
            out[2] = Int32(x % y)
            out[3] = Int32(Float32(-7.0) % Float32(2.0))

        @cute.jit
        def entry(out: cute.Tensor, i: Int32, n: Int32, x: Float32, y: Float32):
            k(out, i, n, x, y).launch(grid=(1, 1, 1), block=(1, 1, 1))

        out = torch.zeros(4, dtype=torch.int32, device="cuda")
        entry(from_dlpack(out), 0, 4, -7.0, 2.0)
        torch.cuda.synchronize()
        self.assertEqual(out.tolist(), [3, 3, 1, 1])


if __name__ == "__main__":
    unittest.main()
