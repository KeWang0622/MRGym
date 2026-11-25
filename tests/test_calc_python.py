"""``calc`` runs the Python an open model writes: methods of lists / text / dictionaries, formatting, unpacking,
``try``, ``:=``, closures, Python's rules for lists and ``None``, the small numerical helpers; and what it still
refuses says in one line what to write instead, with the line it happened on."""
import pytest

from mrgym import calc as C


def show(text, names=None):
    return C.render(C.calc(text, names)).split("\n")


def refusal(text, names=None):
    with pytest.raises(C.CalcError) as e:
        C.calc(text, names)
    return e.value


def test_list_methods():
    assert show("x = [3, 1, 2]; x.sort(); x; x.index(2); x.count(1); x.reverse(); x; x.pop(); x.insert(0, 9); x; x.extend([7, 8]); x") == \
        ["x = [1, 2, 3]", "1", "1", "x = [3, 2, 1]", "1", "x = [9, 3, 2]", "x = [9, 3, 2, 7, 8]"]
    assert show("x = [9, 3]; y = x.copy(); y.remove(9); y; x; y.clear(); len(y); x.pop(0); x") == ["y = [3]", "x = [9, 3]", "0", "9", "x = [3]"]
    assert show("c = [(3, 'a'), (1, 'b')]; c.sort(); c[0]; c.sort(key=lambda t: t[1], reverse=True); c[0][1]; c.sort(reverse=True); c[0][0]") == \
        ["[1, b]", "b", "3"]
    assert show("cand = [(abs(t - 730), t) for t in range(700, 760, 7)]; cand.sort(); cand[0][1]; (1, 2).index(2); (1, 1, 2).count(1)") == ["728", "1", "2"]


def test_text_methods_and_formatting():
    assert show("', '.join(['a', 'b']); ' | '.join(str(v) for v in [1, 2.5]); 'a b  c'.split(); 'a,b'.split(','); ' x '.strip(); 'ab'.upper(); "
                "'AB'.lower(); 'R1'.startswith('R'); 'R1'.endswith('2'); 'a-b'.replace('-', '+'); 'x'.rjust(3) + '|'; 'x'.ljust(3) + '|'") == \
        ["a, b", "1 | 2.5", "[a, b, c]", "[a, b]", "x", "AB", "ab", "true", "false", "a+b", "  x|", "x  |"]
    assert show("'T1={:.1f} k={}'.format(1234.567, 0.5); '{0:>6.2f}|{1}|{0:.0f}'.format(3.14159, 'x'); '{a}-{b:03d}'.format(a='R', b=7); "
                "'{}'.format([1, 2]); '{:{w}.{p}f}'.format(3.14159, w=8, p=2); '{0[1]}'.format([5, 6])") == \
        ["T1=1234.6 k=0.5", "  3.14|x|3", "R-007", "[1, 2]", "    3.14", "6"]
    # % formatting as Python has it: %d cuts the fraction of a float, %s shows any value, a list is formatted element by element
    assert show("'%.2f %d %s %5.1f|%-6s|%g' % (3.14159, 7, [1, 2], 2.25, 'ab', 1e-5); '%d' % 3.7; '%s' % None; '%.3f' % [1, 2]; "
                "'%5d|%05.1f|%+d|%e' % (42, 2.5, 3, 1234.5); '%d%%' % 50; '%(a)s=%(b).1f' % {'a': 'x', 'b': 2}") == \
        ["3.14 7 [1.0, 2.0]   2.2|ab    |1e-05", "3", "None", "[1.000, 2.000]", "   42|002.5|+3|1.234500e+03", "50%", "x=2.0"]
    assert show("'%s %s %s %s %s' % (2.0, 2.5, True, 'a', (1, 'b')); '%5s|%-5s|' % (1.5, 'x'); 'n=%s' % 1234567.891") == \
        ["2.0 2.5 True a [1.0, 'b']", "  1.5|x    |", "n=1234568.0"]
    assert show("x = 3.14159; f'{x:.2f} {x:8.3f}|{x:>10}|{x:<10}|{7:d} {7:03d} {x:e} {x:.3g} {1234567:,} {0.25:.0%} {7:5}|'") == \
        ["3.14    3.142|   3.14159|3.14159   |7 007 3.141590e+00 3.14 1,234,567 25%     7|"]
    assert show("x = 2.5; f'{[1.234, 2.345]:.1f} {None} {\"ab\":>5} {x!r} {x=} {x:{6}.{2}f}'; format(x, '.3f'); str(x) + 'ms'; '-' * 5 + '>'") == \
        ["[1.2, 2.3] none    ab 2.5 x=2.5   2.50", "2.500", "2.5ms", "----->"]
    assert show("r = 3; print('R%d signed:' % r, list(zip([1, 2], [round(s, 4) for s in [0.12345, 0.5]])))") == ["R3 signed: [[1, 0.1235], [2, 0.5]]"]
    assert show("x = float('nan'); '%s %s' % (x, float('inf'))") == ["nan inf"]


def test_dictionary_methods():
    assert show("d = {'a': 1}; d.get('b', 0); d.get('a'); d.setdefault('c', []).append(2); d; d.update({'z': 5}); list(d.keys()); list(d.values()); "
                "d.items(); d.pop('a'); 'a' in d; len(d); sorted(d)") == \
        ["0", "1", "d = {a: 1, c: [2]}", "[a, c, z]", "[1, [2], 5]", "[[a, 1], [c, [2]], [z, 5]]", "1", "false", "2", "[c, z]"]
    assert show("best = {}\nfor T1 in (400, 800):\n    key = T1\n    if key not in best or T1 / 2 < best[key][0]:\n        best[key] = (T1 / 2, T1)\n"
                "for k, v in sorted(best.items()):\n    print(k, v[0])\n{**best, 1: 2}[1]; {(1, 2): 'p'}[(1, 2)]") == ["400 200", "800 400", "2", "p"]


def test_unpacking_walrus_and_try():
    assert show("def f(a, b, c): return a + b * c\nargs = (1, 2, 3); f(*args); print(*[1, 2, 3]); a, *b = [1, 2, 3]; b; [*args, 4]; (*args, 5)[-1]; "
                "pairs = [(1, 2), (3, 4)]; xs, ys = zip(*pairs); xs; f(**{'a': 1, 'b': 2, 'c': 3}); *h, t = 'abc'; h; t") == \
        ["7", "1 2 3", "b = [2, 3]", "[1, 2, 3, 4]", "5", "xs = [1, 3]", "7", "h = [a, b]", "t = c"]
    assert show("def g(*values, scale=1, **more): return sum(values) * scale + len(more)\ng(1, 2, 3); g(1, scale=2, k=0)") == ["6", "3"]
    assert show("if (n := len([1, 2, 3])) > 2: print(n)\n[y for v in [1, 2, 3] if (y := v * 2) > 2]; n") == ["3", "[4, 6]", "n = 3"]
    assert show("try:\n    x = [1][5]\nexcept IndexError as e:\n    print('caught', e)\ntry:\n    y = undefined_name\nexcept Exception:\n    y = -1\n"
                "finally:\n    print('done')\ny") == ["caught index 5 is out of range: the list has 1 element(s) (0 to 0)", "done", "y = -1"]
    assert show("def g(v):\n    if v < 0:\n        raise ValueError('negative')\n    return sqrt(v)\nout = []\nfor v in (4, -1):\n    try:\n"
                "        out.append(g(v))\n    except ValueError as e:\n        out.append(str(e))\n    else:\n        out.append('ok')\nout") == \
        ["out = [2, ok, negative]"]
    assert show("def first(xs):\n    for x in xs:\n        try:\n            return 1 / x if x else none_here\n        except:\n            continue\n"
                "    return -1\nfirst([0, 4]); first([0])") == ["0.25", "-1"]
    assert refusal("try:\n    assert 1 > 2, 'one is small'\nexcept NameError:\n    pass").line == 2             # another error: not caught
    assert refusal("try:\n    zz\nexcept NameError:\n    raise").kind == "unknown"                              # a bare raise raises it again
    assert show("def f(k):\n    try:\n        return [1, 2][k] + {'a': 1}['a']\n    except IndexError:\n        return 'index'\n"
                "    except (KeyError, TypeError) as e:\n        return 'key'\n    finally:\n        print('left', k)\nf(0); f(5)\n"
                "try:\n    try:\n        raise RuntimeError('inner')\n    finally:\n        print('cleanup')\nexcept RuntimeError as e:\n    print(e)") == \
        ["left 0", "2", "left 5", "index", "cleanup", "inner"]
    e = refusal("try:\n    while True:\n        pass\nexcept:\n    print('never')")                           # the step limit is not caught
    assert "stopped after" in str(e) and e.line in (2, 3)


def test_lists_and_none_as_in_python():
    assert show("a = []; b = [1]; 'empty' if not a else 'full'; 'full' if b else 'empty'; bool({}); bool('x'); n = None; n is None; n == None; "
                "not n; n is not None; (1, 2) < (1, 3); [1, 2] == [1, 2]; (2, 'a') > (1, 'b'); [1, 2] != [1, 3]; 'a' < 'b'; None in [1, None]") == \
        ["empty", "full", "false", "true", "true", "true", "true", "false", "true", "true", "true", "true", "true", "true"]
    # + joins and a written list times a whole number repeats, as in Python; what Python has no meaning for is element-wise
    assert show("[0.0] * 3; [[0] * 2 for _ in range(2)]; [1, 2] + [3]; (1,) + (2, 3); 2 * [1, 2]; z = [None] * 3; z; r = (1, 2, 3); (0.5,) + r") == \
        ["[0, 0, 0]", "[[0, 0], [0, 0]]", "[1, 2, 3]", "[1, 2, 3]", "[1, 2, 1, 2]", "z = [none, none, none]", "[0.5, 1, 2, 3]"]
    # a list written in place times a whole number repeats; a list from a name or a call times a number is scaled
    assert show("x = [1, 2]; x * 2; 2 * x; 2.0 * x; [1, 2] * 2; 2 * [1, 2]; zeros = [0] * 4; zeros[2] = 1; zeros; list(x) * 2; x * 2 + 1; "
                "[v for v in x] * 2; (0, 1) * 2; x *= 2; x") == \
        ["[2, 4]", "[2, 4]", "[2, 4]", "[1, 2, 1, 2]", "[1, 2, 1, 2]", "zeros = [0, 0, 1, 0]", "[2, 4]", "[3, 5]", "[1, 2, 1, 2]",
         "[0, 1, 0, 1]", "x = [2, 4]"]
    assert show("x = [1, 2, 4]; x * 2.5; x / 2; x - [1, 1, 1]; x * x; -x; x > 1; exp([0, 0]); x + 1; array(x) * 2 + 1") == \
        ["[2.5, 5, 10]", "[0.5, 1, 2]", "[0, 1, 3]", "[1, 4, 16]", "[-1, -2, -4]", "[false, true, true]", "[1, 1]", "[2, 3, 5]", "[3, 5, 9]"]
    assert show("out = []; alias = out; out += [1]; out += (2, 3); out = out + [4]; alias; out; n = 3; n += 1; n; n *= 2; n; t = (1,); "
                "t += (2,); t; v = array([1, 2]); v += 1; v; w = [1, 2]; w += 1; w") == \
        ["alias = [1, 2, 3]", "out = [1, 2, 3, 4]", "n = 4", "n = 8", "t = [1, 2]", "v = [2, 3]", "w = [2, 3]"]
    assert show("best = None\nfor t in (3, 1, 2):\n    if best is None or t < best[0]:\n        best = (t, t * 2)\nbest; bt = None; "
                "bt = (2, 5) if bt is None or (2, 5) < (bt[0], bt[1]) else bt; bt") == ["best = [1, 2]", "bt = [2, 5]"]
    assert show("def nothing(): pass\nv = nothing(); v is None; [v]; str(v); print(v); min([], default=None) is None; float('inf') > 1e300; "
                "int('12') + int(3.9); float('nan') != float('nan'); sum([[1], [2]], []); max([3, 1], key=lambda v: -v)") == \
        ["true", "[none]", "none", "none", "true", "true", "15", "true", "[1, 2]", "1"]


def test_a_name_of_calc_that_the_text_hid_comes_back_with_an_import():
    names = {}
    C.calc("exp = lambda x: exp(x); log = 3", names)
    assert "calls itself without an end" in str(refusal("exp(1)", names)) and "hides calc's log" in str(refusal("log(2)", names))
    assert show("from math import exp, log\nexp(0); log(1); import copy\na = [[1], [2]]; b = copy.deepcopy(a); b[0][0] = 9; a; b", names) == \
        ["1", "0", "a = [[1], [2]]", "b = [[9], [2]]"]


def test_closures_recursion_and_global():
    assert show("def outer(T1):\n    def inner(t): return exp(-t / T1)\n    return inner\nf = outer(100.0); f(100); h = (lambda k: (lambda v: v * k))(3); "
                "h(2); fs = [lambda v, k=k: v + k for k in range(3)]; fs[2](10)") == ["0.3678794", "6", "12"]
    assert show("def fact(n): return 1 if n <= 1 else n * fact(n - 1)\nfact(20); T1 = 5\ndef g(): return T1 * 2\ndef k():\n    T1 = 7\n    return g()\nk()") == \
        ["2.432902e+18", "10"]                                                                           # a function sees where it was defined
    assert show("count = 0\ndef bump():\n    global count\n    count += 1\nbump(); bump(); count") == ["count = 2"]
    names = {}
    C.calc("def make(T1):\n    return lambda t: exp(-t / T1)\ndecay = make(200.0)", names)
    assert show("decay(200)", names) == ["0.3678794"]                                                   # a closure stays for the next call


def test_numerical_helpers():
    assert show("A = [[2, 1], [1, 3]]; b = [3, 5]; np.linalg.solve(A, b); np.linalg.det(A); np.linalg.inv(A); "
                "np.linalg.lstsq([[1, 0], [1, 1], [1, 2]], [1, 3, 5], rcond=None)[0]; np.array(A) @ b; np.dot(A, b); np.linalg.norm([3, 4])") == \
        ["[0.8, 1.4]", "5", "[[0.6, -0.2], [-0.2, 0.4]]", "[1, 2]", "[11, 18]", "[11, 18]", "5"]
    assert show("p = polyfit([0, 1, 2], [1, 3, 5], 1); polyval(p, 3); X, Y = np.meshgrid([1, 2], [10, 20, 30]); X.shape; Y[2, 0]; "
                "np.column_stack(([1, 2], [3, 4])); np.vstack([[1, 2], [3, 4]]); np.hstack([[1, 2], [3]]); np.concatenate([[1], [2, 3]])") == \
        ["7", "[3, 2]", "30", "[[1, 3], [2, 4]]", "[[1, 2], [3, 4]]", "[1, 2, 3]", "[1, 2, 3]"]
    assert show("import statistics, itertools\nstatistics.mean([1, 2, 4]); statistics.median([1, 2, 4]); statistics.stdev([1, 2, 4]); "
                "list(itertools.product([1, 2], [3, 4])); list(itertools.combinations([1, 2, 3], 2)); from itertools import product\n"
                "len(product(range(3), repeat=2))") == ["2.333333", "2", "1.527525", "[[1, 3], [1, 4], [2, 3], [2, 4]]", "[[1, 2], [1, 3], [2, 3]]", "9"]
    assert show("x = np.linspace(0, 1, 5); x[x > 0.4]; x[-1]; x.tolist(); x.mean(); np.where(x > 0.4, 1, 0); x[np.argmin(abs(x - 0.3))]; x.shape; "
                "M = np.zeros((2, 3)); M[0, 1] = 5; M; M.T.shape; M[:, 1]; np.maximum(x, 0.5)[0]; np.arange(0, 1, 0.25); x[:, None].shape") == \
        ["[0.5, 0.75, 1]", "1", "[0, 0.25, 0.5, 0.75, 1]", "0.5", "[0, 0, 1, 1, 1]", "0.25", "[5]", "M = [[0, 5, 0], [0, 0, 0]]", "[3, 2]",
         "[5, 0]", "0.5", "[0, 0.25, 0.5, 0.75]", "[5, 1]"]
    assert show("z = 2 * exp(1j * pi / 2); abs(z); round(z.imag, 6); np.angle(z); (1 + 2j) * (1 - 2j); s = 1.6461 * math.exp(-2.993j); "
                "abs(s - 1.857 * math.exp(-3.0j)); import cmath; cmath.phase(z); abs(np.array([3 + 4j, 1j]))") == \
        ["2", "2", "1.570796", "5+0j", "0.2112548", "1.570796", "[5, 1]"]
    assert show("isinstance(3, (int, float)); isinstance([1], list); round(2.5); round(3.14159, 2); divmod(7, 2); sorted({'b': 1, 'a': 2}); "
                "list(map(lambda v: v * 2, [1, 2])); any(v > 1 for v in [1, 2]); all([]); list(reversed([1, 2])); math.isclose(0.1 + 0.2, 0.3); "
                "import bisect; bisect.bisect_left([1, 3, 5], 4); math.factorial(5); math.log(8, 2); math.hypot(3, 4)") == \
        ["true", "true", "2", "3.14", "[3, 1]", "[a, b]", "[2, 4]", "true", "true", "[2, 1]", "true", "2", "120", "3", "5"]


def test_root_finding_and_fitting_on_functions_of_the_text():
    """Generic numerics on a function the text defines; the function's own steps are charged, so the step limit holds."""
    fit = ("from scipy.optimize import brentq, curve_fit, minimize, least_squares, minimize_scalar, fsolve\n"
           "def model(t, a, T1): return a * (1 - 2 * exp(-t / T1))\n"
           "t = [100, 300, 600, 1200, 2400]; y = [model(v, 2.0, 800.0) for v in t]\n")
    assert show(fit + "brentq(lambda x: x * x - 2, 0, 2); brentq(lambda ti: model(ti, 2.0, 800.0), 100, 2400) / 800") == ["1.414214", "0.6931472"]
    assert show(fit + "popt, pcov = curve_fit(model, t, y, p0=[1, 500]); popt; pcov.shape; curve_fit(model, np.array(t), y, [1, 500], "
                "bounds=([0, 100], [10, 5000]))[0]") == ["popt = [2, 800]", "[2, 2]", "[2, 800]"]
    assert show(fit + "r = minimize(lambda p: sum((model(array(t), p[0], p[1]) - array(y)) ** 2), [1, 500]); r.x; r.fun < 1e-6; r.success; "
                "ls = least_squares(lambda p: model(array(t), p[0], p[1]) - array(y), [1, 500]); ls.x; ls['cost'] < 1e-9; "
                "minimize_scalar(lambda x: (x - 3) ** 2, bounds=(0, 10)).x; fsolve(lambda x: x ** 3 - 8, 1.0)[0]; import scipy.optimize as opt\n"
                "opt.brentq(lambda x: cos(x), 0, 3)") == ["[2, 800]", "true", "true", "[2, 800]", "true", "3", "2", "1.570796"]
    assert "opposite sign" in str(refusal("from scipy.optimize import brentq\nbrentq(lambda x: x * x + 1, 0, 2)"))
    assert "must work on the whole list" in str(refusal(fit + "curve_fit(lambda t, a: 3.0, t, y, p0=[1])"))
    e = refusal("from scipy.optimize import brentq\ndef f(x):\n    while True:\n        x += 1\nbrentq(f, 0, 1)")
    assert "stopped after" in str(e) and e.line in (3, 4)                                               # a function that does not end is stopped


@pytest.mark.parametrize("text, kind, says, line", [
    ("x = [1, 2, 3]\nx[5]", "value", "index 5 is out of range: the list has 3 element(s) (0 to 2)", 2),
    ("d = {'a': 1}\n\nd['b']", "value", "no key 'b'; keys: a", 3),
    ("s = None\ns[0]", "value", "none has no elements: it was never set", 2),
    ("def f(): pass\nf() + 1", "value", "arithmetic with none: it was never set (a function that ends without return gives none", 2),
    ("best = None\nfor v in best:\n    pass", "value", "cannot loop over none", 2),
    ("a, b = None", "value", "2 names on the left and one value on the right (none)", 1),
    ("'%.2f' % None", "value", "value 1 for %.2f is none, not a number", 1),
    ("'T1=%.1f a=%.3f' % (3, 'x')", "value", "value 2 for %.3f is 'x', not a number (%s shows any value)", 1),
    ("'%d %d' % (1, 2, 3)", "value", "the text has 2 % place(s) and 3 value(s) are given", 1),
    ("f'{None:.2f}'", "value", "the format '.2f' needs a number, here none", 1),
    ("'{:.1f}'.format('x')", "value", "the format '.1f' is for a number and this value is text", 1),
    ("'x' * 2.5", "value", "text * n repeats the text", 1),
    ("'a' - 1", "value", "float(t) reads a number from text", 1),
    ("import os", "unsupported", "calc reaches no files", 1),
    ("import random", "unsupported", "use a grid", 1),
    ("import scipy.stats", "unsupported", "calc has scipy.optimize only: brentq(f, a, b)", 1),
    ("from scipy.optimize import root", "unsupported", "scipy.optimize has: bisect brenth brentq curve_fit", 1),
    ("x = {1, 2}", "unsupported", "no sets in calc: use a list", 1),
    ("set([1, 2])", "unsupported", "no sets in calc: use a list", 1),
    ("next(iter([1]))", "unsupported", "index the list (x[0])", 1),
    ("class A:\n    pass", "unsupported", "keep the values in a dictionary", 1),
    ("x = 1\ndel x", "unsupported", "assign a new value instead", 2),
    ("with open('f') as g:\n    pass", "unsupported", "no with in calc", 1),
    ("np.foo(1)", "unsupported", "no np.foo in calc; nearest: floor", 1),
    ("np.linalg.eig([[1, 2], [3, 4]])", "unsupported", "np.linalg has: det inv lstsq norm pinv solve", 1),
    ("[1, 2].foo()", "unsupported", "no .foo on a list; it has: append clear copy count extend index insert pop remove reverse sort", 1),
    ("d = {'x': 1}\nd.x", "unsupported", "read with brackets: d['x']", 2),
    ("def f(x):\n    y = x + 1\n    return undefined + y\nfor i in range(3):\n    f(i)", "unknown", "unknown name 'undefined'", 3),
    ("def f(x):\n    for k in range(3):\n        if k == 2:\n            return [1, 2][k]\nf(0)", "value", "index 2 is out of range", 4),
    ("f = lambda x: f(x)\nf(1)", "value", "f calls itself without an end (as in f = lambda x: f(x)); write the repetition as a loop", 2),
    ("exp = lambda x: exp(x)\nexp(1)", "value", "`from math import exp` brings calc's exp back", 2),
    ("print('SNR: %.1f' % 2.5 * sqrt(2))", "value", "write '%.1f' % (a * b)", 1),
    ("x = 5\nx(2)", "value", "x is a number, not a function: it cannot be called (x * (...) multiplies)", 2),
    ("sum = 3\nsum([1, 2])", "value", "a name of the text hides calc's sum: rename it", 2),
    ("range(2.5)", "value", "int(x) cuts a fraction", 1),
    ("x = [1, 2, 3]; x[1.5]", "value", "an index is a whole number, not 1.5", 1),
    ("t = (1, 2)\nt[0] = 5", "value", "make it a list first", 2),
    ("x = [1, 2]\nx[2] = 3", "value", "x.append(v) adds one", 2),
    ("[1, 2] * [1, 2, 3]", "value", "arithmetic needs lists of one length: here 2 and 3 values", 1),
    ("[1, 'a'] * 2.5", "value", "a list of numbers of one length (this one holds text)", 1),
    ("x = array([1, 2, 3])\nif x > 1:\n    pass", "value", "use any(...) or all(...)", 2),
    ("def f(a, b): return a\nf(1)", "value", "f takes 2 argument(s) (a, b) and b is not given", 2),
    ("def f(a): return a\nf(1, k=2)", "value", "f has no argument 'k': it takes a", 2),
    ("list(range(10 ** 9))", "value", "a range of more than 100000 numbers can only be looped over (for i in range(n): ...)", 1),
    ("[0] * 10 ** 9", "value", "use a coarser grid", 1),
    ("x = [1, 2, 3]\nif False:\n    import os\nx.foo", "unsupported", "no .foo on a list", 4),
    ("1 << 3", "unsupported", "2 ** n is a power of two", 1),
    ("~3", "unsupported", "`not x` negates one test, -x a number", 1),
    ("a, b, c = (1, 2), 3", "value", "3 names on the left, 2 value(s) on the right", 1),
    ("linspace(0, 1, 10 ** 7)", "value", "use a coarser grid", 1),
    ("np.linalg.solve([[1, 2], [2, 4]], [1, 2])", "value", "the matrix is singular", 1),
    ("'{0.real}'.format(1)", "unsupported", "no attributes", 1),
    ("f'{3.5:d}'", "value", "is for a whole number and this value is 3.5: use .0f", 1),
    ("lambda *a: 1", "unsupported", "def f(*values)", 1),
    ("return 1", "unsupported", "end the text with the value", 1),
    ("1 & 2", "unsupported", "between numbers write `and` / `or` (a power is **)", 1),
])
def test_refusals_say_what_to_write_instead_and_where(text, kind, says, line):
    e = refusal(text)
    assert e.kind == kind and says in str(e) and e.line == line, (e.kind, str(e), e.line)
    assert len(str(e)) <= 200 and "\n" not in str(e)                    # one line; the environment shows the first 200 characters


def test_numbers_that_are_not_finite_are_formatted_not_refused():
    """A formula that went wrong gives nan or inf; printing it must show that, not refuse the print."""
    assert show("x = log(0.0)\nprint('%.1f' % x)") == ["-inf"]
    assert show("x = sqrt(-1.0)\nprint('v=%.2f' % x)") == ["v=nan"]
    assert show("x = 1e400\nprint('%.1f' % x)") == ["inf"]
    assert show("x = float('nan'); y = float('inf'); z = float('-inf'); x; y; z; round(x, 2); round(y); print(x, y, z)") == \
        ["x = nan", "y = inf", "z = -inf", "nan", "inf", "nan inf -inf"]
    assert show("x = float('nan'); y = -float('inf'); '%.1f|%5.2f|%d|%5d|%g|%s|%e' % (x, y, x, y, x, y, x)") == ["nan| -inf|nan| -inf|nan|-inf|nan"]
    assert show("x = float('nan'); y = float('inf'); f'{x:.1f}|{y:8.2f}|{x:d}|{y:>6d}|{x}|{y:g}'; '{:.2f} {:d} {}'.format(x, y, x)") == \
        ["nan|     inf|nan|   inf|nan|inf", "nan inf nan"]
    assert show("TI = 300 * log(-0.5)\nprint('TI=%.1f ms' % TI, 'ok' if TI == TI else 'not a number: the argument of log was negative')") == \
        ["TI=nan ms not a number: the argument of log was negative"]
    assert show("isnan(float('nan')); math.isnan(sqrt(-1)); math.isinf(1e400); math.isfinite(1.0); max(1, float('nan')); min([2, 1, float('inf')])") == \
        ["true", "true", "true", "true", "nan", "1"]


def test_a_written_list_times_a_number_repeats_and_its_elements_can_be_assigned():
    assert show("x = [0.0] * 3; x[1] = 2.0; print(x); y = 3 * [1]; y[0] = 5; y; g = [[0] * 3 for _ in range(2)]; g[1][2] = 7; g; g[0] is g[1]; "
                "[1, 2] + [3]; sol = [0.0] * 3; sol = sol + [1.5]; len(sol)") == \
        ["[0, 2, 0]", "y = [5, 1, 1]", "g = [[0, 0, 0], [0, 0, 7]]", "false", "[1, 2, 3]", "4"]
    assert show("n = 4; z = [None] * n; z[2] = 1; z; t = (0,) * 2 + (1,); t; s = [1, 2] * 2 + [9]; len(s); e = [[]] * 2; len(e)") == \
        ["z = [none, none, 1, none]", "t = [0, 0, 1]", "5", "2"]
    assert show("a = zeros(3); a[1] = 2.0; a; m = zeros((2, 2)); m[0][1] = 3; m[1, 0] = 4; m") == ["a = [0, 2, 0]", "m = [[0, 3], [4, 0]]"]
    assert show("def fit(): return (1.5, 2.5), 0.1\n(a, b), err = fit(); a; b; err; [(p, q), r] = fit(); q\nfor (u, v), w in [fit()]:\n    print(u, v, w)") == \
        ["a = 1.5", "b = 2.5", "err = 0.1", "q = 2.5", "1.5 2.5 0.1"]                                  # a tuple inside a tuple unpacks


def test_tests_of_arrays_are_joined_with_and_or_signs_and_a_long_range_is_looped_over():
    assert show("a = array([1, 2, 3, 4]); (a > 1) & (a < 4); (a < 2) | (a > 3); ~(a > 2); a[(a > 1) & (a < 4)]; (2 > 1) & (1 > 2); 3 ^ 2") == \
        ["[false, true, true, false]", "[true, false, false, true]", "[true, true, false, false]", "[2, 3]", "false", "9"]
    assert show("n = 0\nfor i in range(300000):\n    n += 1\nn; s = 0\nfor k in range(20000, 600000):\n    if k > 20010: break\n    s += k\ns; "
                "len([i for i in range(200000) if i % 50000 == 0]); range(5); range(10 ** 6)") == \
        ["n = 300000", "s = 220055", "4", "[0, 1, 2, 3, 4]", "range(0, 1000000, 1)"]
    e = refusal("x = range(10 ** 6)\nsum(x)")
    assert "can only be looped over" in str(e) and e.line == 2


def test_a_stopped_loop_says_how_far_it_was(monkeypatch):
    monkeypatch.setattr(C, "MAX_STEPS", 2_000_000)                       # the same message at a fraction of the limit
    e = refusal("def cost(a, b): return (a - 3) ** 2 + (b - 4) ** 2\nbest = None\nfor a in range(1000):\n    for b in range(100000):\n"
                "        c = cost(a, b)\n        if best is None or c < best[0]:\n            best = (c, a, b)")
    assert isinstance(e, C._Stop) and len(str(e)) <= 200
    turn, inner = e.loops[1][1], e.loops[0][1]
    assert str(e) == (f"stopped after {C.MAX_STEPS} steps in the loop of line 3 (turn {turn} of 1000) and of line 4 (turn {inner} of 100000): "
                      "use a coarser grid, then a fine one around the best, or minimize(f, x0); names stay")
    assert [line for line, _, _ in e.loops] == [4, 3] and 1 <= turn <= 5 and 1 <= inner <= 100000
    e = refusal("n = 0\nwhile True:\n    n += 1")
    assert str(e) == (f"stopped after {C.MAX_STEPS} steps: a loop that does not end, or too much work for one call: use a coarser grid, "
                      "then a fine one around the best; names stay between calls")


def test_interp_takes_numpy_s_arguments_and_says_them_when_they_are_wrong():
    assert show("interp(2.5, [1, 2, 3], [10, 20, 30]); interp([0, 1.5, 9], [1, 2, 3], [10, 20, 30]); np.interp(0, [1, 2], [5, 6], left=-1); "
                "interp(9, [1, 2], [5, 6], right=0); interp(9, [1, 2], [5, 6], -1, 7); interp(0.5, [0, 1], [10, 20])") == \
        ["25", "[10, 15, 30]", "-1", "0", "7", "15"]                    # outside xp: the end value of fp, as numpy (or left / right)
    assert show("interp(0, [1.2, 0.4, -0.3], [100, 300, 500])") == ["414.2857"]            # xp falling throughout: the same points
    usage = "interp(x, xp, fp, left=None, right=None): x a number or a list, xp and fp two lists of one length, xp rising"
    for text, says in (("interp(1, 2, 3, 4, 5, 6)", usage + "; here 6 argument(s)"), ("interp(2, [1, 2])", usage + "; here 2 argument(s)"),
                       ("interp([1, 2], 3, [4, 5])", usage + "; here xp is 3 and fp is a list of 2"),
                       ("interp(2.5, 2, 3, 20, 30)", usage + "; here xp is 2 and fp is 3"),
                       ("interp(1, [1, 2, 3], [4, 5])", usage + "; here xp is a list of 3 and fp is a list of 2"),
                       ("interp(1, [1, 3, 2], [4, 5, 6])", "xp goes up and down: sort the points by xp first")):
        e = refusal(text)
        assert says in str(e) and len(str(e)) <= 200 and e.kind == "value", str(e)


def test_a_name_that_was_dropped_says_so():
    names = {}
    C.calc("grid = [i for i in range(60000)]; best = 3", names)        # more than MAX_KEPT numbers: the oldest name is not kept
    assert "grid" not in names and names["best"] == 3.0
    e = refusal("len(grid)", names)
    assert "grid was not kept" in str(e) and "keep the result, not the whole grid" in str(e) and e.kind == "value"
    C.calc("grid = [1, 2]", names)
    assert show("len(grid)", names) == ["2"] and C._DROPPED not in names
    assert show("x = 1", names) == ["x = 1"]                             # the note about dropped names is never shown as a name


def test_line_numbers_are_those_of_the_text_as_written():
    assert refusal("\n\n# a comment\nx = 1\ny = zz + x").line == 5
    assert refusal("a = 1; b = [a]\nfor i in range(3):\n    c = b[i]").line == 3
    assert refusal("def f(v):\n    return v[\n        3]\nprint(1)\nf([1])").line == 2            # a statement over several lines: its first line
    env_text = "x = [1]\nfor k in range(2):\n    print(x[k])"
    assert refusal(env_text).line == 3
