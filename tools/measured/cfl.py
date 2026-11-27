"""Reader of the BART ``.cfl`` / ``.hdr`` pair: a text header with the dimensions and a file of complex64 numbers in
column-major order (BART toolbox, doi:10.5281/zenodo.592960). It maps the file; nothing in it is executed."""
import numpy as np


def read(base: str) -> np.ndarray:
    """The array of ``base.hdr`` / ``base.cfl``, trailing dimensions of length 1 dropped, memory-mapped."""
    with open(base + ".hdr", encoding="ascii") as f:
        dims = [int(v) for v in f.read().split("\n")[1].split()]
    while len(dims) > 1 and dims[-1] == 1:
        dims.pop()
    return np.memmap(base + ".cfl", dtype="<c8", mode="r", shape=tuple(dims), order="F")
