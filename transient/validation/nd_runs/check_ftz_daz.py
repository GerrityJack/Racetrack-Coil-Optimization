"""
check_ftz_daz.py -- report the MXCSR flush-to-zero (FTZ, bit 15) and
denormals-are-zero (DAZ, bit 6) flags after importing the same stack
first_step_diagnostic.py uses, to see whether that state itself is stable
or varies run to run (candidate 5 in the nondeterminism investigation).

Uses ctypes + a tiny inline ASM read via the `struct`/`array` module is not
directly possible in pure Python without a C extension; the practical way
is via numpy's floating point control if exposed, or via the `fpectl`-
adjacent behavior. Simpler and portable: use ctypes to call the C library's
fegetenv-family is also not exposed for MXCSR specifically in standard
libc. Instead, use a minimal behavioural probe: whether a known denormal
float survives a multiply-by-one unchanged (FTZ/DAZ off) or is flushed to
0.0 (FTZ/DAZ on) after importing numpy/dolfinx/petsc4py, matching this
project's own eps_reg floor regime (very small ratios near the smoothed
j/jc floor).
"""
import struct
import sys


def denormal_probe():
    # smallest positive subnormal double
    tiny = struct.unpack('d', struct.pack('Q', 1))[0]
    x = tiny * 1.0
    y = tiny / 1.0
    return dict(tiny=tiny, tiny_times_one=x, tiny_div_one=y,
               flushed_mul=(x == 0.0), flushed_div=(y == 0.0))


print("BEFORE any imports:", denormal_probe(), flush=True)

import numpy as np  # noqa: E402
print("AFTER numpy:", denormal_probe(), flush=True)

from mpi4py import MPI  # noqa: E402
print("AFTER mpi4py:", denormal_probe(), flush=True)

from petsc4py import PETSc  # noqa: E402
print("AFTER petsc4py:", denormal_probe(), flush=True)

import dolfinx  # noqa: E402
print("AFTER dolfinx:", denormal_probe(), flush=True)

# also do a tiny numpy-level reduction with subnormal inputs to see if
# numpy's own compiled kernels (which may run through the OpenBLAS/OpenMP
# path for large arrays, and a different scalar path for small ones)
# treat them differently.
arr = np.full(8, struct.unpack('d', struct.pack('Q', 1))[0])
print("numpy sum of 8 subnormals:", arr.sum(),
     "(0.0 means flushed under this codepath)", flush=True)
