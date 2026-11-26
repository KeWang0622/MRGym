"""``calc``: the calculator of every scanner environment. It reads the Python a model writes when it wants a number,
reaches nothing outside its own values, and stops whatever does not end."""
import time

import pytest

import mrgym
from mrgym import calc as C

NOWRAP = "Scanner/NoWrap-v1"


def show(text, names=None):
    return C.render(C.calc(text, names)).split("\n")


def test_calc_arithmetic_lists_and_functions():
    assert C.calc("a = [1, 2, 3]; mean(a) * 2; a[-1] ^ 2; max(1, 5)") == [(None, 4.0), (None, 9.0), (None, 5.0)]
    assert show("2 * ceil(104.8 / 4); floor(2.7); round(2.345, 2); abs(-3); 7 // 2; 7 % 4; sqrt(2) ** 2") == ["54", "2", "2.35", "3", "3", "3", "2"]
    assert show("x = [1, 2, 4]; x * 2 + 1; sum(x); len(x); min(x); x[1:]; x[-1]") == ["[3, 5, 9]", "7", "3", "1", "[2, 4]", "4"]
    assert show("degrees(arctan2(1, 1)); radians(180) / pi; log(e); exp(0); log10(1000)") == ["45", "1", "1", "1", "3"]
    assert show("np.exp(np.array([0, 1]))[0]; math.sqrt(16); math.pi > 3; numpy.cos(0)") == ["1", "4", "true", "1"]
    assert show("polyfit([0, 1, 2], [1, 3, 5], 1); interp(0.5, [0, 1], [10, 20]); linspace(0, 1, 3)") == ["[2, 1]", "15", "[0, 0.5, 1]"]
    assert show("1 / 0; -1 / 0; sqrt(-1)") == ["inf", "-inf", "nan"]                                  # numbers, not exceptions


def test_calc_shows_what_the_text_asks_to_see():
    # bare expressions and prints; a bare name comes with its name; assignments are not echoed next to them
    assert show("Ey = 110; Ex = 174; lines = 2 * (Ey / 4); lines; Ex; lines / Ex") == ["lines = 55", "Ex = 174", "0.316092"]
    assert show("a = 3\nprint('a is', a, 'and twice', 2 * a)\nprint(f'{a / 7:.3f}')") == ["a is 3 and twice 6", "0.429"]
    assert show("rlen = 172; rwidth = 105\nrlen, rwidth") == ["[172, 105]"]
    assert show('"label"; "x = " + 1.5; "a", 2') == ["label", "x = 1.5", "[a, 2]"]
    # a text that asks for nothing gets the names it assigned (the last ones), also when they were assigned in a loop
    assert show("a = 1; b = 2; a = 3") == ["a = 3", "b = 2"]
    assert show("for i in range(3):\n    x = i * 2") == ["i = 2", "x = 4"]
    assert C.calc("") == [] and C.calc("# only a comment") == [] and C.calc("def f(x): return x") == []
    assert len(C.calc("\n".join(f"v{k} = {k}" for k in range(60)))) == C.MAX_ECHO
    rows = C.calc("for i in range(500):\n    print(i)")
    assert len(rows) == C.MAX_ROWS + 1 and "print fewer" in rows[-1][1]


def test_calc_reads_the_python_a_model_writes():
    fit = """
TI = [60, 300, 500, 740]
S = [-1.072, -0.059721, 0.63779, 1.3348]
def fit(S):
    best = None
    T1 = 50.0
    while T1 < 3000:
        T1 += 25
        n = len(S)
        sE = sum(exp(-t / T1) for t in TI)
        sE2 = sum(exp(-t / T1) ** 2 for t in TI)
        sES = sum(exp(-t / T1) * s for t, s in zip(TI, S))
        den = n * sE2 - sE ** 2
        a = (sE2 * sum(S) - sE * sES) / den
        b = (n * sES - sE * sum(S)) / den
        res = sum((s - a - b * exp(-t / T1)) ** 2 for t, s in zip(TI, S))
        if best is None or res < best[0]:
            best = (res, T1, a, b)
    return best
fit(S)[1]"""
    assert show(fit) == ["1175"]
    assert show("import math\ndef cosA(r): return (1 - r) / (5 * r - 1)\nfor name, s1, s2 in [('R1', 0.8, 0.7), ('R2', 0.9, 0.76)]:\n"
                "    print(name, round(math.degrees(math.acos(cosA(s2 / s1)))))") == ["R1 88", "R2 87"]
    assert show("f = lambda a, T1: sin(a) * (1 - exp(-30 / T1)); [round(f(0.3, t), 4) for t in range(500, 1501, 500)]") == ["[0.0172, 0.0087, 0.0059]"]
    assert show("min([(abs(t - 730), t) for t in range(700, 760, 7)]); sorted([3, 1, 2], reverse=True); max([1, 5, 2], key=lambda v: -v)") == \
        ["[2, 728]", "[3, 2, 1]", "1"]
    assert show("S = dict(R3=0.5, R4=1.0); S['R4'] / S['R3']; [k for k in S]; {k: v * 2 for k, v in S.items()}; 'R3' in S") == \
        ["2", "[R3, R4]", "{R3: 1, R4: 2}", "true"]
    assert show("R4, R3 = 0.2, 0.1; R4 / R3; x = 3; x += 2; x; 'big' if x > 4 and not x > 9 else 'small'; 1 < x <= 5") == ["2", "x = 5", "big", "true"]
    assert show("T = {}\nfor r in ['a', 'bb']:\n    T[r] = len(r)\nT; xs = []; xs.append(4); xs") == ["T = {a: 1, bb: 2}", "xs = [4]"]
    assert show("k = 0\nwhile True:\n    k += 1\n    if k >= 5:\n        break\nk") == ["k = 5"]
    assert show("from math import exp, log as ln2\nln2(exp(2))") == ["2"]


@pytest.mark.parametrize("written, read", [
    ("// a note\na = 2; a * 3  // twice the comment sign", ["6"]),                                  # comments of other languages
    ("# S = k*sqrt(N); N=(req*c/S)^2\nb = 3 ^ 2; b", ["b = 9"]),                                     # a ';' inside a comment; ^ for **
    ("px = 240 / 104; print px; print 'shift mm', 2 * px", ["2.307692", "shift mm 4.615385"]),      # print without brackets
    ("y_lo = -92.7; y_hi = 84.6\\nE = y_hi - y_lo + 4\\n[E]", ["[181.3]"]),                           # backslash-n for line breaks
    ('"a = 2\nprint(\'a:\', a)\n"', ["a: 2"]),                                                        # the whole text in quotation marks
    ('half = 3.2237 / 2; half"', ["half = 1.61185"]),                                                # one stray quotation mark
    ("  a = 1\n   b = a + 1\nb", ["b = 2"]),                                                         # stray indentation
    ("a = 2; f(T1, T2) = (1 - exp(-a / T1)) * T2; f(2, 10)", ["6.321206"]),                         # a function as mathematics writes it
    ("def S(tr, t1) = 3 * (1 - exp(-tr / t1))\nS(1, 1)", ["1.896362"]),
])
def test_calc_repairs_the_spellings_that_have_one_reading(written, read):
    assert show(written) == read


@pytest.mark.parametrize("text, kind", [
    ("__import__('os').system('id')", "unknown"), ("open('x')", "unknown"), ("exec('1')", "unknown"), ("eval('1')", "unknown"),
    ("getattr(1, 'real')", "unknown"), ("a.b", "unknown"), ("zz + 1", "unknown"),
    ("import os", "unsupported"), ("from scipy.stats import norm", "unsupported"), ("import math as m; m.system", "unsupported"),
    ("(1).__class__", "unsupported"), ("x = [1]; x.__class__", "unsupported"), ("''.encode()", "unsupported"), ("math.exec", "unsupported"),
    ("class A:\n    pass", "unsupported"), ("def f():\n    yield 1\nf()", "unsupported"), ("with 1 as x:\n    pass", "unsupported"),
    ("x = 1\ndef f():\n    nonlocal x\nf()", "unsupported"), ("x = 1; del x", "unsupported"), ("1 << 2", "unsupported"), ("a = {1, 2}", "unsupported"),
    ("f = lambda *a: 1", "unsupported"), ("return 1", "unsupported"), ("break", "unsupported"), ("1 & 2", "unsupported"), ("~1", "unsupported"),
    ("x = [1, 2]; x[5]", "value"), ("len(3)", "value"), ("[1, 2] - [1, 2, 3] * 2", "value"), ("'ab' * 1001", "value"), ("d = {}; d['k']", "value"),
    ("a, b = 1, 2, 3", "value"), ("def f(x): return f(x)\nf(1)", "value"), ("list(range(10 ** 9))", "value"), ("3()", "value"),
])
def test_calc_refuses_what_it_does_not_have(text, kind):
    with pytest.raises(C.CalcError) as e:
        C.calc(text)
    assert e.value.kind == kind, str(e.value)


@pytest.mark.parametrize("text", ["function f(S){ return 1 }", "a = = 2", "for i in range(3)\n    i", "S(3) = 1 - E", "1 +", "print = [", "a = 1 +* 2"])
def test_calc_text_that_is_not_python_is_a_syntax_error(text):
    with pytest.raises(SyntaxError):
        C.calc(text)


@pytest.mark.parametrize("text", [
    "while True:\n    pass", "x = [i for i in range(4000)]\nwhile True:\n    y = [v * 2 for v in x]", "a = linspace(0, 1, 4000)\nwhile 1:\n    b = polyfit(a, a, 5)",
    "[[j for j in range(5000)] for i in range(5000)]", "x = [1]\nwhile True:\n    x = x + x", "s = 'a'\nwhile True:\n    s = s + s",
    "def f(n): return f(n + 1) + f(n + 2)\nf(0)", "9 ** 9 ** 9 ** 9", "[range(80000) for i in range(4000)]", "d = {}\nfor i in range(10 ** 6):\n    d[i] = [i] * 3",
])
def test_calc_stops_what_does_not_end_and_holds_nothing_large(text, monkeypatch):
    monkeypatch.setattr(C, "MAX_STEPS", 1_000_000)        # a tenth of the limit (tests/test_calc_security.py runs one call to the full limit)
    t0 = time.perf_counter()
    try:
        rows = C.calc(text)
        assert len(rows) <= C.MAX_ROWS + 1
    except C.CalcError as e:
        assert e.kind == "value"
    assert time.perf_counter() - t0 < 2.0
    with pytest.raises(C.CalcError):
        C.calc("a = 1\n" * (C.MAX_CALC_STATEMENTS + 1))
    with pytest.raises(C.CalcError):
        C.calc("1 + " * C.MAX_CALC_CHARS + "1")


def test_calc_names_stay_for_the_calls_of_one_episode():
    names = {}
    assert C.calc("E_y = 110; f = lambda x: x / 4", names) == [("E_y", 110.0)]
    assert show("lines = 2 * ceil(f(E_y)); lines", names) == ["lines = 56"]                         # a value and a function of the call before
    assert set(names) == {"E_y", "f", "lines"}
    with pytest.raises(C.CalcError):
        C.calc("lines")                                                                              # another episode knows nothing of it
    big = {}
    for k in range(40):                                                                              # what is kept is bounded: the oldest names go first
        C.calc(f"v{k} = [i for i in range(4000)]", big)
    assert "v39" in big and "v0" not in big and sum(len(v) for v in big.values()) <= C.MAX_KEPT
    # in an environment: the names of an episode, gone at reset; a refusal says what kind of slip it was
    env = mrgym.make(NOWRAP, level=1, seed=0)
    env.reset()
    call = lambda text: env.step({"name": "calc", "arguments": {"expressions": text}}).observation     # noqa: E731
    assert call("E = 104.8 + 4").text.split("\n")[0] == "E = 108.8" and call("2 * ceil(E / 4)").text.split("\n")[0] == "56"
    for text, kind, cls in (("for i in range(3)\n  i", "malformed_call", "shape"), ("import os", "invalid_value", "shape"),
                            ("never_assigned * 2", "invalid_value", "task"), ("[1, 2][7]", "invalid_value", "task")):
        obs = call(text)
        assert obs.is_error and (obs.data["error"]["type"], obs.data["error"]["class"]) == (kind, cls) and len(obs.text) < 260, obs.text
    assert "line 1" in call("a = (1").text and "never_assigned" in call("never_assigned").text
    assert env.state()["tool_calls"] == 5 and env.state()["rejected_calls"] == 3      # the three run-time errors are tool calls, the three the calculator cannot read are rejected
    env.reset()
    assert call("E").is_error


def test_calc_reaches_nothing_but_its_own_values():
    """Every name a text can reach is a number, a list, text, a dictionary, or one of the listed functions."""
    assert {"math", "np", "numpy"} <= set(C.MODULES) and all(callable(f) for f in C.FUNCTIONS.values())
    assert not [k for k in C.FUNCTIONS if k.startswith("_") or k in ("eval", "exec", "open", "getattr", "setattr", "compile", "globals", "locals", "vars", "type")]
    names = {}
    C.calc("f = lambda x: x; d = {'a': [1, 2]}; t = (1, 'b')", names)
    assert {type(v).__name__ for v in names.values()} == {"_Function", "dict", "tuple"}
    for probe in ("f.__globals__", "f.run", "d.__class__", "t.__class__", "print.__self__", "sqrt.__module__", "pi.__class__", "[].__class__"):
        with pytest.raises((C.CalcError, SyntaxError)):
            C.calc(probe, names)


def test_the_solvers_the_tool_description_names_work_without_an_import_and_in_a_later_call():
    names = {}
    assert abs(C.calc("r = brentq(lambda t: t * t - 2, 0, 2)\nr", names)[-1][1] - 2 ** 0.5) < 1e-6
    C.calc("def f(t, a, T1):\n    return a * (1 - 2 * exp(-t / T1))\npopt, pcov = curve_fit(f, [100, 200, 400, 800], [-0.4, 0.0, 0.45, 0.85], p0=[1.0, 300.0])", names)
    assert abs(names["popt"][1] - 292.87) < 0.5                              # the fitted T1 of an inversion recovery
    assert C.calc("m = minimize(lambda x: (x[0] - 2) ** 2, [0.0])\nround(m['x'][0], 3)", names)[-1][1] == 2
    assert C.calc("s = minimize_scalar(lambda x: (x - 3) ** 2, bounds=(0, 10))\nround(s['x'], 3)", names)[-1][1] == 3
    with pytest.raises(C.CalcError):
        C.calc("np.brentq(lambda t: t - 1, 0, 2)")                           # numpy has no brentq: the prefix stays what numpy is
    assert C.calc("def minimize(a):\n    return a + 1\nminimize(1)")[-1][1] == 2   # a name of the text hides calc's
