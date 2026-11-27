"""The example of the package docstring runs as written and gives the inverted lactate doublet."""
import re

import numpy as np

import mrgym.spins


def test_the_example_of_the_package_docstring():
    code = "\n".join(line[4:] for line in mrgym.spins.__doc__.splitlines() if re.match(r"    (from|lines|ppm)", line))
    scope = {}
    exec(code, scope)                                            # noqa: S102 - the package's own docstring
    ppm, spec = scope["ppm"], scope["spec"]
    doublet = (ppm > 1.2) & (ppm < 1.43)
    k = np.argmax(np.abs(spec.real * doublet))
    assert abs(ppm[k] - 1.3142) < 0.05 and spec.real[k] < 0          # at TE = 144 ms the doublet points down
    assert len(code.splitlines()) == 3
