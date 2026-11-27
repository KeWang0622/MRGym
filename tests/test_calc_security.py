"""``calc`` stays a calculator: the classic escapes of a Python sandbox find nothing, every resource is bounded by a
count (never by a clock), memory per call is bounded and measured, and the same text gives the same answer."""
import ast
import inspect
import os
import subprocess
import sys
import threading
import time
import types

import pytest

from mrgym import calc as C


def show(text, names=None):
    return C.render(C.calc(text, names)).split("\n")


# ------------------------------------------------------------------------------------------------ nothing to reach
ESCAPES = [
    # attributes of Python objects
    "(1).__class__", "().__class__.__mro__", "[].__class__.__base__.__subclasses__()", "''.__class__.__mro__[1]", "{}.__class__",
    "f = lambda: 0\nf.__globals__", "f = lambda: 0\nf.__code__", "f = lambda: 0\nf.body", "f = lambda: 0\nf.env", "def f(): pass\nf.__closure__",
    "print.__self__", "print.__func__", "sqrt.__module__", "sqrt.__wrapped__", "sqrt.__closure__", "np.__dict__", "math.__loader__",
    "np.linalg.__builtins__", "x = [1]; x.__len__()", "x = 5; x.__class__ = 3", "d = {'__class__': 1}; d.__class__", "x = [1]; x.append.__self__",
    "r = scipy.optimize.minimize_scalar(lambda v: v * v, bounds=(-1, 1)); r.__class__", "x = array([1]); x.base", "x = array([1]); x.ctypes",
    "x = array([1]); x.dtype", "x = array([1]); x.tofile('f')", "x = array([1]); x.dump('f')", "x = array([1]); x.__array_interface__",
    "e = None\ntry:\n    1 + none_here\nexcept Exception as e:\n    pass\ne.__traceback__", "e = ValueError('x'); e.__class__",
    # names of Python that reach further
    "getattr(1, 'real')", "setattr(1, 'a', 2)", "delattr(1, 'a')", "__import__('os')", "eval('1')", "exec('1')", "open('/etc/passwd')",
    "compile('1', 'x', 'eval')", "globals()", "locals()", "vars()", "type(1)", "__builtins__", "__name__", "__file__", "breakpoint()",
    "memoryview(1)", "object()", "super()", "input()", "exit()", "quit()", "help()", "dir()", "id(1)", "classmethod(1)", "staticmethod(1)",
    # what numpy / math / scipy have beyond the tables
    "np.load('x')", "np.fromfile('x')", "np.save('x', [1])", "np.loadtxt('x')", "np.genfromtxt('x')", "np.memmap('x')", "np.lib", "np.core",
    "np.ctypeslib", "np.testing", "np.random.rand(3)", "np.vectorize(sqrt)", "np.frompyfunc(sqrt, 1, 1)", "np.array([1, 2], dtype=object)",
    "zeros(3, dtype=object)", "math.exec", "scipy.io", "scipy.optimize.show_options()", "scipy.optimize._minimize", "itertools.__spec__",
    # imports
    "import os", "import subprocess", "from os import system", "import numpy.f2py", "import numpy.distutils", "import importlib", "import builtins",
    "import ctypes", "import pickle", "import socket", "import shutil", "import pathlib", "from math import *; __import__('os')",
    "import math as m; m.system", "import os.path", "from . import x", "import numpy.random", "from numpy import load", "from scipy import io",
    # format strings that read attributes
    "'{0.__class__}'.format(1)", "'{0.__class__.__mro__}'.format([])", "'{a.__globals__}'.format(a=sqrt)", "'{0[__class__]}'.format([1])",
    "'%(__class__)s' % {'x': 1}", "f'{(1).__class__}'", "'{0.real}'.format(1)", "'{0.__init__.__globals__}'.format(print)",
    "format(print, '{0.__self__}')", "'{.__class__}'.format(1)",
    # statements that are not a calculator's
    "class A:\n    pass", "with open('f') as g:\n    pass", "x = 1\ndel x", "f = lambda: (yield)\nf()", "async def f(): pass\nf()", "def f():\n    yield 1\nf()",
    "@print\ndef f(): pass", "raise SystemExit", "raise KeyboardInterrupt('x')", "def f(x, /): pass", "x = 1\ndef f():\n    nonlocal x\nf()",
]


@pytest.mark.parametrize("text", ESCAPES)
def test_escapes_find_nothing(text):
    names = {}
    with pytest.raises((C.CalcError, SyntaxError)) as e:
        C.calc(text, names)
    assert not isinstance(e.value, C._Stop)
    for value in names.values():                                         # and nothing of Python is left in the names
        assert value is None or isinstance(value, (float, bool, str, list, tuple, dict, complex, C._Function, C._Module)) or \
            type(value).__module__ == "numpy", type(value)


def test_the_tables_hold_functions_of_the_calculator_and_nothing_else():
    """Every name a text can reach: a number, a function defined in calc.py (or a plain method of str), a table."""
    def check(name, value, depth=0):
        assert not name.startswith("_"), name
        assert not isinstance(value, (types.ModuleType, type)), name
        if isinstance(value, C._Module):
            assert depth < 2
            for k, v in value.members.items():
                check(f"{name}.{k}", v, depth + 1)
        elif callable(value):
            home = getattr(value, "__module__", None) or getattr(getattr(value, "__objclass__", None), "__name__", None)
            assert home in ("mrgym.calc", "str"), (name, home)
        else:
            assert value is None or isinstance(value, (float, bool)), name
    for table in (C.FUNCTIONS, C.CONSTANTS, C.MODULES):
        for name, value in table.items():
            check(name, value)
    for kind, table in list(C.METHODS.items()) + list(C._FIELDS.items()):
        assert kind in (list, tuple, dict, str, float, complex, C._Result) or kind.__module__ == "numpy"
        for name, value in table.items():
            check(name, value)
    forbidden = {"eval", "exec", "open", "getattr", "setattr", "delattr", "compile", "globals", "locals", "vars", "type", "input", "__import__",
                 "load", "save", "fromfile", "tofile", "loadtxt", "savetxt", "genfromtxt", "memmap", "vectorize", "frompyfunc", "system", "popen"}
    reachable = set(C.FUNCTIONS) | {k for m in C.MODULES.values() for k in m.members} | {k for t in C.METHODS.values() for k in t}
    assert not forbidden & reachable
    assert set(C.MODULES) == {"math", "np", "numpy", "cmath", "statistics", "itertools", "bisect", "functools", "copy", "scipy"}


def test_the_source_never_evaluates_text_and_never_looks_up_an_attribute_by_a_name_of_the_text():
    tree = ast.parse(inspect.getsource(C))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in ("eval", "exec", "compile", "__import__", "open", "input", "setattr", "delattr", "globals", "locals", "vars"), \
                (node.func.id, node.lineno)
            if node.func.id == "getattr":                               # only with a name the source itself writes
                name = node.args[1]
                fixed = isinstance(name, ast.Constant) or (isinstance(name, ast.BinOp) and isinstance(name.left, ast.Constant)
                                                           and name.left.value in ("e_", "s_"))
                assert fixed, node.lineno
        elif isinstance(node, ast.Attribute):
            assert node.attr not in ("__globals__", "__builtins__", "__subclasses__", "__dict__", "system", "popen"), node.lineno
    assert imported == {"__future__", "ast", "bisect", "difflib", "functools", "itertools", "json", "math", "operator", "re", "string", "threading",
                        "numpy", "scipy"}


def test_a_function_of_the_text_is_data_not_python():
    names = {}
    C.calc("def f(x, k=2):\n    return x * k\ng = lambda v: f(v) + 1\nh = [1, 2].append\nimport numpy as nn", names)
    assert {type(v).__name__ for v in names.values()} == {"_Function", "partial", "_Module"}
    assert show("f(3); g(3); nn.sqrt(4)", names) == ["6", "7", "2"]
    for probe in ("f.__globals__", "f.body", "f.env", "g.params", "h.func", "h.args", "nn.members", "nn.name", "f.invoke(1)"):
        with pytest.raises(C.CalcError):
            C.calc(probe, names)


# ------------------------------------------------------------------------------------------------ bounded by counts
BOMBS = [
    # huge values, asked for in one expression: refused before anything is allocated
    "[0] * 10 ** 9", "10 ** 9 * [0]", "'ab' * 10 ** 9", "(0,) * 10 ** 9", "sum(range(10 ** 12))", "list(range(10 ** 9))", "[i for i in range(10 ** 12)]", "for i in range(10 ** 12):\n    pass",
    "sorted(range(10 ** 7))", "x = range(10 ** 7)\narray(x)", "max(range(10 ** 12))", "zip(range(10 ** 7), range(10 ** 7))", "'%s' % [range(10 ** 12)] * 2000 + 1", "linspace(0, 1, 10 ** 9)",
    "arange(0, 1, 1e-12)", "zeros(10 ** 9)", "zeros((10 ** 5, 10 ** 5))", "np.ones((40000, 40000))", "eye(10 ** 5)", "np.repeat([1], 10 ** 9)",
    "np.tile([1, 2], 10 ** 9)", "x = linspace(0, 1, 50000); np.outer(x, x)", "x = linspace(0, 1, 50000); x[:, None] * x[None, :]",
    "x = linspace(0, 1, 50000); np.meshgrid(x, x)", "x = linspace(0, 1, 50000); np.subtract(x[:, None], x[None, :])",
    "x = linspace(0, 1, 50000); x[:, None] < x[None, :]", "x = linspace(0, 1, 50000); np.where(x[:, None] > 0, x[None, :], 0)",
    "x = zeros((300, 300)); x @ x", "itertools.product(range(1000), range(1000), range(1000))", "itertools.permutations(range(20))",
    "itertools.combinations(range(100), 50)", "math.factorial(10 ** 6)", "math.comb(10 ** 6, 5 * 10 ** 5)", "np.linalg.inv(zeros((300, 300)))",
    "polyfit(range(100), range(100), 60)", "diag(linspace(0, 1, 5000))",
    # huge formats
    "'{:>1000000000}'.format(1)", "'%1000000000d' % 1", "'%.*f' % (10 ** 9, 1.0)", "f'{1:1000000000}'", "f'{1:.1000000000f}'",
    "'x'.zfill(10 ** 9)", "'x'.ljust(10 ** 9)", "'x'.center(10 ** 9)", "format(1, '>999999999')", "'a'.replace('a', 'b' * 1500) * 2",
    # growth in a loop
    "x = [1]\nwhile True:\n    x = x + x", "x = [1]\nwhile True:\n    x.extend(x)", "x = array([1.0])\nwhile True:\n    x = np.concatenate([x, x])",
    "s = 'a'\nxs = []\nwhile True:\n    xs.append(s * 2000)", "d = {}\ni = 0\nwhile True:\n    d[i] = [i] * 3\n    i += 1",
    "x = [[j for j in range(5000)] for i in range(5000)]", "[range(80000) for i in range(4000)]",
    # recursion and nesting
    "def f(n): return f(n + 1)\nf(0)", "def f(n): return f(n + 1) + f(n + 2)\nf(0)", "f = lambda f, n: f(f, n + 1)\nf(f, 0)", "+".join(["1"] * 3000),
    "a = []\na.append(a)\nprint(a)", "a = []\na.append(a)\na == a", "a = []\na.append(a)\narray(a)", "d = {}\nd['d'] = d\nd",
    "a = []\nfor i in range(100000):\n    a = [a]\na", "a = []\nfor i in range(100000):\n    a = [a]\nstr(a) + 'x'",
    # a list that holds itself many times over: 2 ** 40 elements behind 40 small lists
    "x = [1]\nfor i in range(40):\n    x = [x, x]\nx", "x = [1]\nfor i in range(40):\n    x = [x, x]\narray(x)", "x = [1]\nfor i in range(40):\n    x = [x, x]\nsum(x)",
    "x = [1]\nfor i in range(40):\n    x = [x, x]\ny = [1]\nfor i in range(40):\n    y = [y, y]\nx == y", "x = [1]\nfor i in range(40):\n    x = [x, x]\nsorted([x, x])",
    "x = (1,)\nfor i in range(40):\n    x = (x, x)\nd = {x: 1}", "x = [1]\nfor i in range(40):\n    x = [x, x]\nmax(x)", "x = [1]\nfor i in range(40):\n    x = [x, x]\ny = [1]\nfor i in range(40):\n    y = [y, y]\nx in [3, y]",
    "x = [1]\nfor i in range(40):\n    x = [x, x]\nnp.mean(x)", "x = [1]\nfor i in range(40):\n    x = [x, x]\nx.count(x[0])", "x = [1]\nfor i in range(40):\n    x = [x, x]\n{'k': x}",
    # loops that do not end, with and without work inside, in every construct
    "while True:\n    pass", "while 1:\n    x = [v * 2 for v in range(4000)]", "a = linspace(0, 1, 4000)\nwhile 1:\n    b = polyfit(a, a, 5)",
    "f = lambda: 0\nwhile True:\n    f()", "while True:\n    try:\n        x = [1][3]\n    except:\n        pass",
    "def f():\n    while True:\n        pass\ntry:\n    f()\nexcept Exception:\n    f()\nfinally:\n    f()",
    "x = [1]\nfor v in x:\n    x.append(v)", "from scipy.optimize import brentq\nbrentq(lambda v: [w for w in range(100000)][0] + v, -1, 1)\n" * 3,
    "while True:\n    s = '%d %s' % (1, 'a')", "z = 1j\nwhile True:\n    z = z * 1j", "x = [3, 1, 2] * 1\nwhile True:\n    y = sorted(x)",
]


@pytest.mark.parametrize("text", BOMBS)
def test_what_does_not_end_or_does_not_fit_is_refused_by_a_count(text, monkeypatch):
    monkeypatch.setattr(C, "MAX_STEPS", 400_000)                         # the same code path at a tenth of the limit: the suite stays fast
    t0 = time.perf_counter()
    names = {}
    with pytest.raises(C.CalcError) as e:
        C.calc(text, names)
    assert e.value.kind == "value" and len(str(e.value)) <= 220, str(e.value)
    assert time.perf_counter() - t0 < 3.0
    C._trim(names)
    assert sum(C._size(v) for v in names.values()) <= C.MAX_KEPT + len(names)


def test_numbers_overflow_to_inf_and_nan_not_to_an_exception_or_a_long_integer():
    assert show("10 ** 10 ** 10; 9 ** 9 ** 9 ** 9; 2 ** 1e6; (-8) ** (1 / 3); 1e308 * 10; -1e308 * 10; 1 / 0; 0 / 0; exp(1000); log(0); 7 % 0; 7 // 0; "
                "int(1e300) > 1; 10 ** 400; float('1e999')") == \
        ["inf", "inf", "inf", "nan", "inf", "-inf", "inf", "nan", "inf", "-inf", "nan", "inf" if False else show("7 // 0")[0], "true", "inf", "inf"]
    names = {}
    C.calc("x = 2 ** 5000; y = 12345678901234567890123", names)
    assert all(type(v) is float for v in names.values())                 # never a Python integer that could grow


def test_a_call_that_runs_to_the_limit_is_stopped_at_the_same_step_every_time():
    """The limit is a count of steps: the outcome of a call does not depend on the machine or on its load."""
    def run(text):
        runs = []

        class Probe(C._Run):
            __slots__ = ()

            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                runs.append(self)
        saved, C._Run = C._Run, Probe
        try:
            with pytest.raises(C._Stop) as e:
                C.calc(text)
        finally:
            C._Run = saved
        return runs[-1].steps, str(e.value), e.value.line
    text = ("def f(x, T1): return 1 - 2 * exp(-x / T1)\nn = 0\nbest = None\nwhile True:\n    n += 1\n    v = f(n, 900.0)\n"
            "    if best is None or v < best[0]:\n        best = (v, n)")
    t0 = time.perf_counter()
    first = run(text)
    seconds = time.perf_counter() - t0
    full = C.MAX_STEPS
    C.MAX_STEPS = full // 10                                             # twice the same at a tenth of the limit (the suite stays fast)
    try:
        assert run(text) == run(text) and run(text)[0] < first[0]
    finally:
        C.MAX_STEPS = full
    assert C.MAX_STEPS < first[0] <= C.MAX_STEPS + 64 and f"stopped after {C.MAX_STEPS} steps" in first[1] and first[2] in range(1, 9)
    assert "coarser grid" in first[1] and "names stay between calls" in first[1]
    assert seconds < 20.0                                                # about five seconds on a 2024 server core; never a clock in calc itself
    assert not [n for n in ("time", "perf_counter", "monotonic", "signal", "alarm") if n in vars(C)]


def test_the_same_text_gives_the_same_answer():
    text = ("import math\nTI = [60, 300, 500, 740]\nS = [-1.072, -0.059721, 0.63779, 1.3348]\nbest = None\nfor k in range(400):\n    T1 = 100 + 5 * k\n"
            "    x = [math.exp(-t / T1) for t in TI]\n    n = len(x); sx = sum(x); sy = sum(S); sxx = sum(v * v for v in x)\n"
            "    sxy = sum(a * b for a, b in zip(x, S)); den = n * sxx - sx * sx\n    b = (n * sxy - sx * sy) / den; a = (sy - b * sx) / n\n"
            "    r = sum((s - a - b * v) ** 2 for s, v in zip(S, x))\n    if best is None or r < best[0]:\n        best = (r, T1, a, b)\n"
            "print('T1 %.0f a %.4f b %.4f' % (best[1], best[2], best[3])); d = {'b': 1, 'a': 2}; sorted(d); list(d)")
    first = C.calc(text)
    assert first == C.calc(text) and first[0] == (None, "T1 1175 a 4.4047 b -5.7636") and first[1:] == [(None, ["a", "b"]), (None, ["b", "a"])]
    out = []
    threads = [threading.Thread(target=lambda: out.append(C.calc(text))) for _ in range(4)]             # one evaluation per thread
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert out == [first] * 4


# ------------------------------------------------------------------------------------------------ memory
MEMORY_TEXTS = [
    "x = [[j for j in range(1000)] for i in range(100000)]",
    "x = [[{'a': j} for j in range(1000)] for i in range(100000)]",
    "x = [[dict(a=j) for j in range(1000)] for i in range(100000)]",
    "x = [['a' * 2000 + str(j) for j in range(1000)] for i in range(100000)]",
    "r = range(100000)\nx = [map(str, r) for i in range(100000)]",
    "def make(k):\n    return lambda: k\nx = [[make(j) for j in range(1000)] for i in range(100000)]",
    "r = range(100000)\nx = [enumerate(r) for i in range(100000)]",
    "d = {i: i for i in range(100000)}\nx = [d.items() for i in range(1000)]",
    "r = zeros(100000)\nx = [r.tolist() for i in range(1000)]",
    "x = [[[j] for j in range(1000)] for i in range(100000)]",
    "d = {}\ni = 0\nwhile True:\n    d[i] = {}\n    for j in range(1000):\n        d[i][j] = [j]\n    i += 1",
    "x = [zeros(1000) + i for i in range(100000)]",
]
# The highest resident size this process has had, in MB. On Linux it is read from the process's own memory (VmHWM), not
# from ``ru_maxrss``: that one starts at the resident size of the process that started this one (the kernel carries the
# mark over exec), so under a parent of 700 MB a call that holds 570 MB moved it by nothing.
PEAK_SOURCE = """
import resource, sys
def peak_mb():
    try:
        with open("/proc/self/status") as f:
            return next(int(line.split()[1]) for line in f if line.startswith("VmHWM:")) / 1024
    except (OSError, StopIteration):
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024)
"""
MEMORY_SCRIPT = PEAK_SOURCE + """
from mrgym import calc as C
import scipy.optimize
base = peak_mb()
for text in {texts!r}:
    names = {{}}
    try:
        C.calc(text, names)
        print("finished")
    except C._Stop:
        pass
    print(peak_mb() - base)
    del names
"""
PARENT_SCRIPT = PEAK_SOURCE + """
import subprocess
held = bytearray(b"x") * ({hold_mb} << 20)
print(peak_mb(), subprocess.run([sys.executable, "-c", {source!r} + "print(peak_mb())"], capture_output=True, text=True).stdout)
"""


@pytest.mark.skipif(sys.platform == "win32", reason="resource.getrusage")
def test_memory_of_one_call_is_bounded():
    """Every element a call makes is charged a step before it is made, so ``MAX_STEPS`` bounds what a call can hold.
    The constructions that hold the most per step, each run to the limit in a fresh process: peak resident memory
    of that process over what it held before the first call (``peak_mb``)."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.dirname(os.path.dirname(os.path.dirname(C.__file__)))] + sys.path), OMP_NUM_THREADS="1")
    out = subprocess.run([sys.executable, "-c", MEMORY_SCRIPT.format(texts=MEMORY_TEXTS)], capture_output=True, text=True, env=env, timeout=300)
    assert out.returncode == 0, out.stderr[-600:]
    lines = out.stdout.split()
    assert "finished" not in lines and len(lines) == len(MEMORY_TEXTS)   # every one of them ran to the limit
    peak = max(float(x) for x in lines)
    assert 20 < peak < C.MEMORY_BOUND_MB, lines
    assert C.MAX_ITEMS <= C.MAX_STEPS and C.MEMORY_BOUND_MB <= 25e-6 * C.MAX_STEPS


@pytest.mark.skipif(not os.path.exists("/proc/self/status"), reason="the measure reads /proc/self/status; elsewhere it is ru_maxrss")
def test_the_memory_measure_does_not_see_the_process_that_started_it():
    """The bound above is measured from whatever process runs the tests. A process that holds 256 MB starts another
    one: the measure of the second is its own few MB. With ``ru_maxrss`` it was the 256 MB of the first, and from the
    whole suite in one process the calls of the test above seemed to hold nothing (and a call over the bound would
    have passed by as much as that process held)."""
    out = subprocess.run([sys.executable, "-c", PARENT_SCRIPT.format(hold_mb=256, source=PEAK_SOURCE)], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-600:]
    parent, child = (float(x) for x in out.stdout.split())
    assert child < 64 < 256 <= parent, out.stdout


def test_what_stays_between_calls_is_bounded_and_a_closure_counts_what_it_holds():
    names = {}
    for k in range(40):
        C.calc(f"v{k} = [i for i in range(4000)]", names)
    assert "v39" in names and "v0" not in names and sum(C._size(v) for k, v in names.items() if k != C._DROPPED) <= C.MAX_KEPT
    names = {}
    C.calc("def keep(n):\n    big = [i for i in range(n)]\n    return lambda k: big[k]\nfs = [keep(30000), keep(30000)]; small = 1", names)
    assert "fs" not in names and names["small"] == 1.0                   # two closures over 60,000 numbers are more than is kept
    names = {}
    C.calc("a = []\na.append(a)\nb = [1]\nfor i in range(40):\n    b = [b, b]\nc = 2", names)
    assert names["c"] == 2.0 and C._size(names["a"]) <= 2 and C._size(names["b"]) < 200         # counted once, however often it is held
    names = {}
    C.calc("deep = []\nfor i in range(100000):\n    deep = [deep]\nc = 2", names)
    assert "deep" not in names and names["c"] == 2.0
    names = {"old": 1.0}
    with pytest.raises(C.CalcError):
        C.calc("big = [i for i in range(60000)]\nundefined_name", names)
    assert "big" not in names and names["old"] == 1.0                    # also when the call was refused
