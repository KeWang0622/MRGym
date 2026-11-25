"""``calc``: a calculator for the agent, and nothing else. A tool of every scanner environment.

The text is parsed to a syntax tree; the tree is turned once into Python closures (one per node) and those are run.
Nothing is ever handed to ``eval`` / ``exec`` / ``compile``, no attribute of a Python object is ever looked up by a
name the text supplies (methods come from the tables below, by the kind of the value), and nothing is imported for
the text. It knows nothing about the scanner or the object, and it can reach nothing outside its own values.

What it reads is the subset of Python a model writes when it wants a number:

    values       numbers (floats; ``1j`` complex), text, ``None``, true / false, lists, tuples, dictionaries, arrays
    lists        ``a + b`` joins and ``a += b`` extends, as in Python; a list written in place (a display or a
                 comprehension) times a whole number repeats (``[0] * n``, ``n * [1, 2]``, ``[[0] * 3 for _ in range(2)]``),
                 while a list from a name, a call or any other expression times a number is scaled (``x * 2``: calc has no
                 integers, so what repeats is told by how it is written); ``a < b`` orders like words in a dictionary,
                 ``if a:`` asks whether it is empty; arithmetic Python does not have for lists
                 (``a * 2.5``, ``a - b``, ``a * b``, ``a + 1``, ``exp(a)``) is element-wise and gives an array
    arrays       ``array([...])``, ``linspace``, ``arange``, ``zeros`` ...: element-wise throughout, as numpy
    operators    + - * / ** // %  (``^`` is read as ``**``), comparisons, and / or / not, ``a if c else b``, ``:=``,
                 indexing and slices, ``%`` and ``.format`` and f-string formatting, ``*`` unpacking
    statements   assignment (also ``a, *b = ...`` and ``x += 1``), an expression, ``print(...)``, ``for`` / ``while`` /
                 ``if``, ``def`` and ``lambda`` (closures as in Python), comprehensions, ``try`` / ``except`` (it
                 catches what the calculator refuses, never the step limit), ``raise``, ``assert``, ``global``;
                 ``import math`` / ``numpy`` / ``cmath`` / ``statistics`` / ``itertools`` / ``bisect`` /
                 ``scipy.optimize`` bind the tables below and import nothing
    functions    ``FUNCTIONS`` (math and numpy names, with or without the prefix) and ``MODULES``
    methods      ``METHODS``: of lists, tuples, text, dictionaries and arrays
    comments     ``# ...`` and a line that starts with ``// ...``

Bounds (all of them counts, none of them a clock: the outcome of a call does not depend on the machine):
``MAX_CALC_CHARS`` characters; ``MAX_STEPS`` steps (a step is an evaluated syntax node, or one element of a list /
array / text the call makes or hands to a function; a loop that does not end is stopped there); ``MAX_ITEMS``
elements per list / array / dictionary; ``MAX_TEXT`` characters per text; ``MAX_DEPTH`` nested calls;
``MAX_ROWS`` printed values. Memory: every element a call makes is charged a step first, so a call holds at most
``MAX_STEPS`` elements of its own (``MEMORY_BOUND_MB``, measured by tests/test_calc_security.py), and at most
``MAX_KEPT`` numbers stay between calls. Numbers are floats: ``7 / 2`` is 3.5, a whole number may index a list.

What a call shows: every ``print(...)`` and every bare expression of the top level (a bare name with its name). A
text that asks to see nothing gets the names it assigned. Names, values and functions alike, stay from one call to
the next when the caller passes a dictionary (``calc(text, names)``): the environment keeps one per episode.

Spellings that have one reading are repaired before parsing and never change a value: a leading ``//`` comment, a
trailing ``// comment`` on a line that does not parse with it, ``print a, b`` without brackets, ``f(x) = x + 1`` for
a function, the two characters backslash-n where line breaks were meant, quotation marks around the whole text.
"""
from __future__ import annotations

import ast
import bisect as _bisect
import difflib
import functools
import itertools as _itertools
import json
import math
import operator
import re
import string
import threading

import numpy as np

MAX_CALC_CHARS, MAX_CALC_STATEMENTS, MAX_CALC_ARRAY = 8000, 400, 4096
MAX_ITEMS = 100_000             # elements of one list / array / dictionary
MAX_STEPS = 30_000_000          # steps per call (see the module text)
MAX_ROWS = 120                  # values a call prints
MAX_TEXT = 2000                 # characters of one text value
MAX_DEPTH = 40                  # nested calls of functions defined in the text
MAX_KEPT = 50_000               # numbers kept between the calls of one episode (the oldest names go first)
MAX_ECHO = 24                   # assignments echoed by a text that prints nothing
MAX_MATRIX = 64                 # rows / columns of a matrix handed to linalg (cost grows with the cube)
MAX_PARAMETERS = 12             # unknowns of curve_fit / least_squares / minimize
MEMORY_BOUND_MB = 750           # what one call can hold at most (25 bytes per step; the worst constructions found hold 20)
_CURRENT = threading.local()    # the evaluation that is running in this thread (a function defined in an earlier call runs in it)
_DROPPED = "__dropped__"        # key of the names dictionary: names that were dropped because too much was kept


class CalcError(ValueError):
    """What the calculator refuses. ``kind``: ``unknown`` (a name that was never assigned: the agent's slip, not the
    interface's), ``unsupported`` (Python the calculator does not have) or ``value`` (an operation that has no result)."""

    def __init__(self, message: str, kind: str = "unsupported", line: int | None = None):
        super().__init__(message)
        self.kind, self.line = kind, line


class _Stop(CalcError):
    """The step limit. ``try`` / ``except`` in the text never catches it. ``loops``: (line, turn, turns) of the ``for``
    loops that were running, innermost first."""

    def __init__(self, message, kind="value", line=None):
        super().__init__(message, kind, line)
        self.loops = []


_PY_ERRORS = (TypeError, ValueError, IndexError, KeyError, ZeroDivisionError, OverflowError, ArithmeticError, AttributeError)
_F64 = np.float64
_BOOL = np.dtype(bool)
_REAL = (float, int, bool, np.floating, np.integer, np.bool_)
_NUMBER = _REAL + (complex, np.complexfloating)
_FLAT = frozenset((float, np.float64, bool, int, np.bool_))
_FLAT_C = _FLAT | frozenset((complex, np.complex128))
_ATOMS = _FLAT_C | frozenset((str, type(None)))


# ------------------------------------------------------------------------------------------------ steps
def _run():
    return getattr(_CURRENT, "run", None) or _IDLE


def _charge(n) -> None:
    """``n`` steps of work done outside the syntax tree (the elements a function reads or makes)."""
    run = _run()
    run.steps += n
    if run.steps > run.limit:
        run.stop(None)


def _room(n, what="a list") -> int:
    """``n`` elements are about to be made: refuse more than ``MAX_ITEMS`` before anything is allocated."""
    if not n <= MAX_ITEMS:
        raise CalcError(f"{what} of more than {MAX_ITEMS} values: use a coarser grid, or two passes (coarse, then fine around the best)", "value")
    n = int(n)
    _charge(n)
    return n


# ------------------------------------------------------------------------------------------------ values
def _whole(x) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, (bool, np.bool_)) and math.isfinite(x) and float(x) == int(x)


def _int(x, what="this") -> int:
    if isinstance(x, np.ndarray) and x.ndim == 0:
        x = x.item()
    if isinstance(x, (bool, np.bool_)):
        return int(x)
    if not _whole(x):
        raise CalcError(f"{what} is a whole number, not {_brief(x)}", "value")
    return int(x)


def _kind(x) -> str:
    if isinstance(x, str):
        return "text"
    if x is None:
        return "none"
    if isinstance(x, (bool, np.bool_)):
        return "true / false"
    if isinstance(x, _NUMBER):
        return "a number"
    if isinstance(x, dict):
        return "a dictionary"
    if isinstance(x, (list, tuple, np.ndarray)):
        return "a list"
    if isinstance(x, _Module):
        return f"the module {x.name}"
    if isinstance(x, _Lazy):
        return f"a range of more than {MAX_ITEMS} numbers (it can only be looped over)"
    return "a function" if callable(x) else type(x).__name__


def _brief(x) -> str:
    """A value in an error message: short."""
    if isinstance(x, _NUMBER) or x is None or isinstance(x, (bool, np.bool_)):
        return _text(x)
    if isinstance(x, str):
        return repr(x[:24])
    if isinstance(x, (list, tuple, np.ndarray, dict)):
        return f"{_kind(x)} of {len(x) if not isinstance(x, np.ndarray) else x.size}"
    return _kind(x)


_NONE_HINT = ": it was never set (a function that ends without return gives none; test with `x is not None`)"


def _need_number(x):
    hint = _NONE_HINT if x is None else ": float('1.5') reads a number from text" if isinstance(x, str) else ""
    raise CalcError(f"a number is needed here, not {_kind(x)}{hint}", "value")


def _bad_list(x=None):
    held = ""
    if x is not None:
        atoms = [e for e in x if not isinstance(e, (list, tuple, np.ndarray))][:50]
        if any(isinstance(e, str) for e in atoms):
            held = " (this one holds text)"
        elif any(e is None for e in atoms):
            held = " (this one holds none)"
        elif len({len(e) for e in x if isinstance(e, (list, tuple, np.ndarray))} | ({1} if atoms else set())) > 1:
            held = " (its rows differ in length)"
    raise CalcError(f"arithmetic needs a list of numbers of one length{held}", "value")


def _count(x, limit) -> int:
    """Numbers in a nested list, counted before numpy sees it (a list that holds itself many times is refused here)."""
    n = 0
    for e in x:
        if isinstance(e, (list, tuple)):
            n += _count(e, limit - n)
        elif isinstance(e, np.ndarray):
            n += e.size
        else:
            n += 1
        if n > limit:
            raise CalcError(f"a list of more than {MAX_ITEMS} values", "value")
    return n


def _array(x):
    """A value as a float (or complex) array; a number as a numpy scalar. Refuses text and ragged lists."""
    if isinstance(x, np.ndarray):
        kind = x.dtype.kind
        if kind == "f" or kind == "c" or kind == "b":
            return x
        return x.astype(float) if kind in "iu" else _bad_list()
    if isinstance(x, _REAL):
        return _F64(x)
    if isinstance(x, (list, tuple)):
        n = len(x)
        if n > MAX_ITEMS:
            raise CalcError(f"a list of more than {MAX_ITEMS} values", "value")
        kinds = set(map(type, x))
        if kinds <= _FLAT:
            _charge(n)
            return np.array(x, float)
        if not kinds <= _FLAT_C:
            _charge(_count(x, MAX_ITEMS))
        try:
            a = np.array(x)
        except ValueError:
            _bad_list(x)
        kind = a.dtype.kind
        return a if kind == "f" or kind == "c" else a.astype(float) if kind in "iub" else _bad_list(x)
    if isinstance(x, (complex, np.complexfloating)):
        return np.complex128(x)
    _need_number(x)


def _out(v):
    """What a function gives back, as a value of the calculator (numpy scalars become numbers, whole-number arrays floats)."""
    t = type(v)
    if t is float or t is str or t is bool or v is None or t is list:
        return v
    if t is np.ndarray:
        if v.ndim == 0:
            return _out(v.item())
        if v.size > MAX_ITEMS:
            raise CalcError(f"a list of more than {MAX_ITEMS} values", "value")
        return v.astype(float) if v.dtype.kind in "iu" else v
    if t is _F64 or t is int or isinstance(v, (np.integer, np.floating)):
        try:
            return float(v)
        except OverflowError:
            return math.inf if v > 0 else -math.inf
    if t is np.bool_:
        return bool(v)
    if isinstance(v, np.complexfloating):
        return complex(v)
    return v


def _cells(v) -> int:
    t = type(v)
    if t is float or v is None or t is bool:
        return 0
    if t is np.ndarray:
        return v.size
    if t is list:                                                        # a list of tuples (zip, enumerate, items): the tuples count too
        return len(v) * (2 + len(v[0])) if v and type(v[0]) is tuple else len(v)
    if t is tuple or t is dict or isinstance(v, dict):
        return len(v)
    return len(v) >> 3 if t is str else 0


def _is_true(x) -> bool:
    if isinstance(x, np.ndarray):
        if x.size != 1:
            raise CalcError("a list of comparisons is neither true nor false: use any(...) or all(...); "
                            "tuple(a) < tuple(b) orders two lists as Python does", "value")
        return bool(x.ravel()[0])
    return bool(x)


def _g(x) -> str:
    return f"{x:.7g}"


def _gc(z) -> str:
    z = complex(z)
    return f"{_g(z.real)}{'+' if z.imag >= 0 or z.imag != z.imag else '-'}{_g(abs(z.imag))}j"


def _text(v) -> str:
    """A value as the agent reads it: seven significant digits, no trailing ``.0``."""
    return _show(v, [MAX_CALC_ARRAY])


def _show(v, room) -> str:
    t = type(v)
    if t is float or t is _F64:
        return f"{v:.7g}"
    if t is str:
        return v
    if isinstance(v, (bool, np.bool_)):
        return "true" if v else "false"
    if v is None:
        return "none"
    if isinstance(v, _REAL):
        return _g(float(v))
    if isinstance(v, (complex, np.complexfloating)):
        return _gc(v)
    if isinstance(v, np.ndarray) and v.ndim == 0:
        return _show(v.item(), room)
    if isinstance(v, (list, tuple, np.ndarray)):
        parts = []
        for x in (v.tolist() if isinstance(v, np.ndarray) and v.ndim == 1 else v):
            if room[0] <= 0:
                parts.append(f"... ({len(v)} values)")
                break
            room[0] -= 1
            parts.append(_show(x, room))
        return "[" + ", ".join(parts) + "]"
    if isinstance(v, dict):
        parts = []
        for k, x in v.items():
            if room[0] <= 0:
                parts.append("...")
                break
            room[0] -= 1
            parts.append(f"{_show(k, room)}: {_show(x, room)}")
        return "{" + ", ".join(parts) + "}"
    if isinstance(v, _Lazy):
        return f"range({v.r.start}, {v.r.stop}, {v.r.step})"
    return f"<module {v.name}>" if isinstance(v, _Module) else "<function>"


def _num(v, room=None):
    """A value as JSON for ``Observation.data``: numbers at seven significant digits, lists, text."""
    room = [MAX_CALC_ARRAY] if room is None else room
    if isinstance(v, str):
        room[0] -= len(v) // 12
        return v
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if v is None or callable(v) or isinstance(v, _Module):
        return None
    if isinstance(v, _Lazy):
        return _text(v)
    if isinstance(v, _REAL):
        x = float(v)
        return float(f"{x:.7g}") if math.isfinite(x) else str(x)
    if isinstance(v, (complex, np.complexfloating)):
        return _gc(v)
    if isinstance(v, np.ndarray) and v.ndim == 0:
        return _num(v.item(), room)
    if isinstance(v, dict):
        items = [(_text(k), _num(x, room)) for k, x in v.items() if _spend(room, 1)]
        return dict(items)
    if isinstance(v, np.ndarray):
        _spend(room, v.size)
        v = v.tolist()
    else:
        _spend(room, len(v))
    return [_num(x, room) for x in v]


def _spend(room, n) -> bool:
    room[0] -= n
    if room[0] < 0:
        raise CalcError(f"the result has more than {MAX_CALC_ARRAY} values: show a part of it (x[:20], len(x), min(x), max(x))", "value")
    return True


def _shown(v):
    """``_num``, or one line about a value that is too large to show (a name a text assigned and did not ask to see)."""
    try:
        return _num(v)
    except CalcError:
        return f"{_cells(v)} values (not shown)"
    except RecursionError:
        return "a deeply nested list (not shown)"


# ------------------------------------------------------------------------------------------------ comparing
def _order(a, b) -> int:
    """-1, 0 or 1, as Python orders numbers, text, and lists / tuples element by element."""
    if isinstance(a, _REAL) and isinstance(b, _REAL):
        return (a > b) - (a < b)
    if isinstance(a, str) and isinstance(b, str):
        return (a > b) - (a < b)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        _charge(4 + 2 * min(len(a), len(b)))
        for x, y in zip(a, b):
            c = _order(x, y)
            if c:
                return c
        return (len(a) > len(b)) - (len(a) < len(b))
    if a is None or b is None:
        raise CalcError("none has no order" + _NONE_HINT, "value")
    raise CalcError(f"{_kind(a)} and {_kind(b)} cannot be ordered: compare numbers, or lists of numbers", "value")


def _same(a, b) -> bool:
    """``a == b`` as Python has it for numbers, text, none, lists, tuples and dictionaries."""
    if type(a) in _ATOMS and type(b) in _ATOMS:
        return a == b
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if a is None or b is None or isinstance(a, (str, dict)) or isinstance(b, (str, dict)):
            return False
        x, y = _array(a), _array(b)
        return x.shape == y.shape and bool(np.all(x == y))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        _charge(4 + 2 * len(a))
        return all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        _charge(1 + len(a))
        return a.keys() == b.keys() and all(_same(v, b[k]) for k, v in a.items())
    try:
        return bool(a == b)
    except (TypeError, ValueError):
        return False


def _simple(items) -> bool:
    """Numbers, text, or flat lists / tuples of them: what Python's own sort can order without a deep walk."""
    kinds = set(map(type, items))
    if kinds <= _FLAT or kinds == {str}:
        return True
    if not kinds <= {tuple, list}:
        return False
    _charge(sum(map(len, items)))
    return all(set(map(type, x)) <= _ATOMS for x in items)


def _sort_key(items, key):
    """The list of sort keys, and whether Python's own comparison may be used on them."""
    keys = list(items) if key is None else [key(x) for x in items]
    n = len(keys)
    _charge(n * max(1, n.bit_length()))
    return keys, _simple(keys)


def _sorted(x, key=None, reverse=False):
    items = list(_iterable(x))
    keys, simple = _sort_key(items, key)
    order = range(len(items))
    try:
        if simple:
            order = sorted(order, key=keys.__getitem__, reverse=bool(reverse))
        else:
            order = sorted(order, key=functools.cmp_to_key(lambda i, j: _order(keys[i], keys[j])), reverse=bool(reverse))
    except TypeError:
        raise CalcError("these values cannot be ordered: sort numbers, text, or lists of the same shape (none and text do not mix with numbers)",
                        "value") from None
    return [items[i] for i in order]


def _extreme(np_fn, sign):
    """``max(1, 2)``, ``max([1, 2])``, and ``min(pairs, key=...)`` over tuples (compared like Python compares them)."""
    def call(*a, key=None, default=_MISSING, axis=None):
        if len(a) == 2 and key is None:
            x, y = a
            if type(x) is float and type(y) is float:
                return math.nan if x != x or y != y else x if (x - y) * sign >= 0 else y
        _charge(4)
        if axis is not None:
            return np_fn(_array(a[0]), axis=_int(axis, "axis"))
        if len(a) == 1 and isinstance(a[0], np.ndarray) and key is None:
            if not a[0].size:
                raise CalcError("min / max of an empty list", "value")
            return np_fn(a[0])
        items = list(_iterable(a[0])) if len(a) == 1 and not isinstance(a[0], _NUMBER) else list(a)
        if not items:
            if default is _MISSING:
                raise CalcError("min / max of an empty list: test len(x) first, or give default=", "value")
            return default
        keys = items if key is None else [key(x) for x in items]
        kinds = set(map(type, keys))
        if kinds <= _FLAT:
            _charge(len(keys))
            best = int((np.argmax if sign > 0 else np.argmin)(np.array(keys, float)))
            return items[best] if key is not None or type(items[best]) is float else float(np_fn(np.array(keys, float)))
        if any(isinstance(k, np.ndarray) for k in keys) and key is None:
            return np_fn(np.array([_array(k) for k in keys]))
        best = 0
        for i in range(1, len(keys)):
            if _order(keys[i], keys[best]) * sign > 0:
                best = i
        return items[best]
    return call


def right_has(container, item) -> bool:
    if isinstance(container, dict):
        return _key(item) in container
    if isinstance(container, str):
        return isinstance(item, str) and item in container
    if isinstance(container, np.ndarray):
        return bool(np.any(container == _array(item))) if isinstance(item, _NUMBER) else False
    if isinstance(container, (list, tuple)):
        _charge(len(container))
        if type(item) in _ATOMS and set(map(type, container)) <= _ATOMS:
            return item in container
        return any(_same(x, item) for x in container)
    raise CalcError(f"`in` needs a list, a dictionary or text on the right, not {_kind(container)}", "value")


def _key(k):
    """A dictionary key: text, a number, or a tuple of them."""
    t = type(k)
    if t is str or t is float:
        return k
    if isinstance(k, _REAL) and not isinstance(k, (bool, np.bool_)):
        return float(k)
    if t is tuple:
        _charge(6 + 2 * len(k))
        return tuple(_key(x) for x in k)
    if isinstance(k, np.ndarray) and k.ndim == 0:
        return float(k)
    raise CalcError(f"a key is text or a number (or a tuple of them), not {_kind(k)}", "value")


# ------------------------------------------------------------------------------------------------ formatting
_PERCENT = re.compile(r"%(?:\((\w+)\))?([-+ #0]*)(\*|\d+)?(?:\.(\*|\d+))?[hlL]?(.)?")
_SPEC = re.compile(r"^(?:(.)?([<>=^]))?([-+ ])?(#)?(0)?(\d+)?([_,])?(?:\.(\d+))?([bcdeEfFgGnosxX%])?$")
_FIELD = re.compile(r"^(\w*)((?:\[[^\]\[]+\])*)$")
MAX_WIDTH = 200                 # width and precision of one formatted value


def _narrow(width, what="width"):
    if width is not None and int(width) > MAX_WIDTH:
        raise CalcError(f"a format {what} of more than {MAX_WIDTH}", "value")


def _format(v, spec: str) -> str:
    """``format(v, spec)`` as Python has it; a list is formatted element by element, a whole float fits ``d``."""
    if not spec:
        return _text(v)
    if type(v) is float and _spec_plan(spec):
        return format(v, spec)
    _charge(4)
    m = _SPEC.match(spec)
    if m is None:
        raise CalcError(f"the format {spec!r} is not a Python format: write for example .2f, 8.3f, .3g, >6 or d", "value")
    _narrow(m.group(6)), _narrow(m.group(8), "precision")
    kind, precision = m.group(9), m.group(8)
    if isinstance(v, np.ndarray) and v.ndim == 0:
        v = v.item()
    if isinstance(v, str):
        if kind not in (None, "s"):
            raise CalcError(f"the format {spec!r} is for a number and this value is text", "value")
        return format(v, spec)
    plain = kind in (None, "s") and (precision is None or kind == "s")
    if isinstance(v, (list, tuple, np.ndarray)) and not plain:
        _charge(len(v))
        return "[" + ", ".join(_format(x, spec) for x in (v.tolist() if isinstance(v, np.ndarray) and v.ndim == 1 else v)) + "]"
    if plain or v is None or isinstance(v, dict) or callable(v):
        if not plain:
            raise CalcError(f"the format {spec!r} needs a number, here {_kind(v)}" + (_NONE_HINT if v is None else ""), "value")
        body = spec[:-1] if kind == "s" else spec
        if isinstance(v, _NUMBER) and not isinstance(v, (bool, np.bool_)) and m.group(2) is None and m.group(5) is None:
            body = (m.group(1) or "") + ">" + body[len(m.group(1) or ""):]              # a number is set to the right, as in Python
        if m.group(3) or m.group(4) or m.group(5) or m.group(7):
            if not isinstance(v, _REAL) or isinstance(v, (bool, np.bool_)):
                raise CalcError(f"the format {spec!r} needs a number, here {_kind(v)}", "value")
            x = float(v)
            return format(int(x) if math.isfinite(x) and x == int(x) and abs(x) < 1e15 else x, spec)
        return format(_text(v), body)
    if isinstance(v, (complex, np.complexfloating)):
        if kind in "bcdnosxX":
            raise CalcError(f"the format {spec!r} does not fit a complex number: format abs(z) or z.real", "value")
        return format(complex(v), spec)
    x = float(v)
    if kind is None and precision is None and math.isfinite(x) and x == int(x) and abs(x) < 1e15:
        return format(int(x), spec)                                      # a whole number is shown without .0, as calc prints it
    if kind in "bcdoxX":
        if not math.isfinite(x):                                         # nan and inf are shown as such, as .2f shows them
            return format(_g(x), (m.group(1) or "") + (m.group(2) or ">") + (m.group(6) or ""))
        if x != int(x):
            raise CalcError(f"the format {spec!r} is for a whole number and this value is {_g(x)}: use .0f, or round it first", "value")
        return format(int(x), spec)
    return format(x, spec)


@functools.lru_cache(maxsize=512)
def _percent_plan(template: str) -> int:
    """The number of places of a template whose places are all plain number formats (``%.2f``, ``%5d``, ``%g``), else -1."""
    places = 0
    for m in _PERCENT.finditer(template):
        name, _, width, precision, conv = m.groups()
        if conv == "%" and not (name or width or precision):
            continue
        if name is not None or conv is None or conv not in "dieEfFgG" or width == "*" or precision == "*" or \
                int(width or 0) > MAX_WIDTH or int(precision or 0) > MAX_WIDTH:
            return -1
        places += 1
    return places


@functools.lru_cache(maxsize=512)
def _spec_plan(spec: str) -> bool:
    """Whether a format spec is a plain float format within the width bound (``.2f``, ``8.3f``, ``.3g``, ``.1%``, ``e``)."""
    m = _SPEC.match(spec)
    return m is not None and m.group(9) is not None and m.group(9) in "eEfFgG%" and int(m.group(6) or 0) <= MAX_WIDTH and \
        int(m.group(8) or 0) <= MAX_WIDTH


@functools.lru_cache(maxsize=512)
def _format_plan(template: str):
    """The fields of a ``str.format`` template (the tokens only: no value is touched), or the error text."""
    try:
        return tuple(string.Formatter().parse(template))
    except ValueError as e:
        return str(e)


def _percent(template: str, values) -> str:
    """``template % values`` as Python has it (``%d`` of a float cuts its fraction, ``%s`` shows a value as calc prints it)."""
    places = _percent_plan(template)
    if places >= 0:
        if type(values) is float:
            if places == 1 and values - values == 0.0:
                return template % values
        elif type(values) is tuple and len(values) == places and set(map(type, values)) == {float}:
            try:
                return template % values
            except (ValueError, OverflowError):                          # %d of inf or nan: the slow path says so
                pass
    _charge(8)
    if isinstance(values, dict):
        named, args = values, []
    else:
        named, args = None, list(values) if isinstance(values, tuple) else [values]
    places = sum(1 for m in _PERCENT.finditer(template) if m.group(5) != "%" and m.group(1) is None)
    if named is None and len(args) == 1 and places > 1 and isinstance(args[0], (list, np.ndarray)) and len(args[0]) == places:
        args = list(args[0])                                             # '%.1f %.1f' % [a, b]: the list is the values
    used = [0]

    def take():
        if used[0] >= len(args):
            raise CalcError(f"the text has {places} % place(s) and {len(args)} value(s) are given", "value")
        used[0] += 1
        return args[used[0] - 1]

    def one(m):
        name, flags, width, precision, conv = m.groups()
        if conv == "%":
            return "%"
        if conv is None or conv not in "diouxXeEfFgGcrsa":
            raise CalcError(f"%{conv or ''} is not a format: %d %.2f %g %s are (a literal percent sign is written %%)", "value")
        if name is not None:
            if named is None or name not in named:
                raise CalcError(f"%({name})... needs a dictionary with the key {name!r} on the right", "value")
            v = named[name]
        else:
            width = str(_int(take(), "a * width")) if width == "*" else width
            precision = str(_int(take(), "a * precision")) if precision == "*" else precision
            v = take()
        _narrow(width), _narrow(precision, "precision")
        return _percent_one(v, flags, width or "", precision, conv, used[0])

    out = _PERCENT.sub(one, template)
    if named is None and used[0] != len(args):
        raise CalcError(f"the text has {places} % place(s) and {len(args)} value(s) are given", "value")
    return out


def _percent_one(v, flags, width, precision, conv, place=0) -> str:
    dot = "" if precision is None else "." + precision
    if isinstance(v, np.ndarray) and v.ndim == 0:
        v = v.item()
    if conv in "sra":                                                    # as Python's str(): 2.0, True, None, [1.0, 'a']
        _charge(2 * _cells(v))
        try:
            shown = v if isinstance(v, str) else str(_num(v))
        except CalcError:
            shown = _text(v)
        return ("%" + flags.replace("0", "").replace("#", "") + width + dot + "s") % shown
    if isinstance(v, (list, tuple, np.ndarray)):
        _charge(len(v))
        return "[" + ", ".join(_percent_one(x, flags, width, precision, conv, place) for x in (v.tolist() if isinstance(v, np.ndarray) else v)) + "]"
    if not isinstance(v, _REAL):
        which = f"value {place} for " if place else ""
        raise CalcError(f"{which}%{width}{dot}{conv} is {_brief(v)}, not a number" + (_NONE_HINT if v is None else " (%s shows any value)"), "value")
    x = float(v)
    if conv in "diouxXc":
        if not math.isfinite(x):                                         # nan and inf are shown as such, as %f shows them
            return ("%" + flags.replace("0", "") + width + "s") % _g(x)
        return ("%" + flags + width + dot + ("d" if conv in "iu" else conv)) % int(x)
    return ("%" + flags + width + dot + conv) % x


def _str_format(template, *args, **kw) -> str:
    """``template.format(...)``: fields are ``{}``, ``{0}``, ``{name}``, with an index ``{0[1]}`` and a format ``{:.2f}``;
    no attribute of a value is ever read (``{0.real}`` is refused)."""
    out, auto = [], [0]

    def field(name):
        m = _FIELD.match(name)
        if m is None:
            raise CalcError("a format field is {} or {0} or {name}, with a format after a colon ({0:.2f}): no attributes, no calls", "unsupported")
        head, rest = m.groups()
        try:
            if head == "":
                v = args[auto[0]]
                auto[0] += 1
            elif head.isdigit():
                v = args[int(head)]
            else:
                v = kw[head]
        except (IndexError, KeyError):
            raise CalcError(f"format: the text asks for {{{name}}} and {len(args)} value(s) are given", "value") from None
        for k in re.findall(r"\[([^\]\[]+)\]", rest):
            v = _item(v, float(k) if re.fullmatch(r"-?\d+", k) else k)
        return v

    parsed = _format_plan(template)
    if isinstance(parsed, str):
        raise CalcError(f"format: {parsed} (a literal brace is written {{{{ or }}}})", "value")
    _charge(4 + 2 * len(parsed))
    for literal, name, spec, conversion in parsed:
        out.append(literal)
        if name is None:
            continue
        v = field(name)
        if spec and "{" in spec:                                         # a width or a precision given as a value: {:{w}.{p}f}
            spec = _str_format(spec, *args[auto[0]:], **kw) if not re.search(r"\{\d", spec) else _str_format(spec, *args, **kw)
            auto[0] += spec.count("\0")
        out.append(_text(v) if conversion and not spec else _format(v, spec or ""))
    return "".join(out)[:MAX_TEXT]


# ------------------------------------------------------------------------------------------------ functions
_MISSING = object()


def _iterable(value):
    """What a loop runs over: the elements of a list, the keys of a dictionary, the characters of text."""
    t = type(value)
    if t is list or t is tuple:
        return value
    if t is np.ndarray:
        if value.ndim == 0:
            raise CalcError("cannot loop over a number: use range(n)", "value")
        return value.tolist() if value.ndim == 1 else list(value)
    if isinstance(value, dict):
        return list(value)
    if t is str:
        return list(value)
    if value is None:
        raise CalcError("cannot loop over none" + _NONE_HINT, "value")
    if isinstance(value, _NUMBER):
        raise CalcError("cannot loop over a number: use range(n)", "value")
    if t is _Lazy:
        raise CalcError(f"a range of more than {MAX_ITEMS} numbers can only be looped over (for i in range(n): ...): use a coarser step, "
                        "or linspace(a, b, n)", "value")
    raise CalcError(f"cannot loop over {_kind(value)}", "value")


def _item(value, index):
    """``value[index]`` for one index (no slice)."""
    t = type(value)
    if t is list or t is tuple or t is str:
        try:
            k = int(index)
            if k != index or type(index) is str:
                raise ValueError
            return value[k]
        except IndexError:
            n = len(value)
            raise CalcError(f"index {k} is out of range: the list has {n} element(s)" + (f" (0 to {n - 1})" if n else ""), "value") from None
        except (ValueError, TypeError, OverflowError):
            pass
        if isinstance(index, (list, np.ndarray)) and t is not str:
            return _item(_array(value), index)
        raise CalcError(f"an index is a whole number, not {_brief(index)}" + (": int(x) cuts a fraction" if isinstance(index, float) else ""), "value")
    if isinstance(value, dict):
        key = _key(index)
        if key not in value:
            raise CalcError(f"no key {_text(key)!r}; keys: {', '.join(_text(k) for k in list(value)[:12])}", "value")
        return value[key]
    if t is np.ndarray:
        try:
            if isinstance(index, (list, np.ndarray)):
                index = _array(index)
                index = index if index.dtype.kind == "b" else index.astype(int)
                _charge(index.size)
            else:
                index = _int(index, "an index")
            out = value[index]
        except IndexError:
            raise CalcError(f"index {_brief(index)} is out of range: the list has {len(value)} element(s)", "value") from None
        return out.item() if out.ndim == 0 else out
    if value is None:
        raise CalcError("none has no elements" + _NONE_HINT, "value")
    raise CalcError(f"{_kind(value)} has no elements" + (": it is one number here, not a list" if isinstance(value, _NUMBER) else ""), "value")


def _reducer(fn, what):
    """``mean([1, 2, 3])`` and ``mean(1, 2, 3)`` both work; ``axis=`` as numpy has it."""
    def call(*a, axis=None, **kw):
        _only(kw, what, "dtype", "ddof", "keepdims")
        if not a:
            raise CalcError(f"{what} needs a list", "value")
        x = _array(a[0]) if len(a) == 1 else np.array([_array(v) for v in a])
        if not x.size:
            raise CalcError(f"{what} of an empty list: test len(x) first", "value")
        _charge(6 + x.size)
        more = {} if "ddof" not in kw else dict(ddof=_int(kw["ddof"], "ddof"))
        return fn(x, **more) if axis is None else fn(x, axis=_int(axis, "axis"), **more)
    return call


def _only(kw, what, *allowed):
    for k in kw:
        if k not in allowed:
            raise CalcError(f"{what} has no {k}= here", "value")


def _one(m, n):
    """A function of one number: Python's ``math`` on a float, numpy on a list (and where math raises: inf / nan)."""
    def call(x, **kw):
        if type(x) is float:
            try:
                return m(x)
            except (ValueError, OverflowError):
                pass
        elif type(x) is complex:
            _charge(4)
            return complex(n(x))
        return n(_array(x))
    return call


def _pair(a, b):
    """Two values as arrays that numpy may combine: the size of what comes out is checked first."""
    x, y = _array(a), _array(b)
    if x.ndim and y.ndim and x.shape != y.shape:
        try:
            shape = np.broadcast_shapes(x.shape, y.shape)
        except ValueError:
            raise CalcError(f"arithmetic needs lists of one length: here {_shape(x)} and {_shape(y)} values", "value") from None
        _room(math.prod(shape))
    return x, y


def _shape(x) -> str:
    return str(x.shape[0]) if x.ndim == 1 else " x ".join(str(n) for n in x.shape)


def _two(fn, m=None):
    def call(a, b, **kw):
        if m is not None and type(a) is float and type(b) is float:
            try:
                return m(a, b)
            except (ValueError, OverflowError, ZeroDivisionError):
                pass
        x, y = _pair(a, b)
        _charge(4)
        return fn(x, y)
    return call


def _max2(a, b):
    return math.nan if a != a or b != b else a if a >= b else b


def _min2(a, b):
    return math.nan if a != a or b != b else a if a <= b else b


def _log(x, base=None):
    if base is None:
        if type(x) is float and x > 0:
            return math.log(x)
        return np.log(_array(x))
    return np.log(_array(x)) / np.log(_array(base))


def _abs(x):
    return abs(x) if type(x) is float or type(x) is complex else np.abs(_array(x))


def _sum(x, start=0.0, axis=None, **kw):
    if axis is not None:
        return np.sum(_array(x), axis=_int(axis, "axis"))
    if isinstance(x, np.ndarray):
        if type(start) is float and not start:
            return x.sum()
        return np.sum(x) + start if x.ndim == 1 or type(start) is float else np.sum(x, axis=0) + _array(start)
    items = _iterable(x)
    if isinstance(start, (list, tuple)):                               # sum(lists, []): the lists joined
        out = list(start)
        for part in items:
            out.extend(_iterable(part))
            _room(len(out))
        return out if isinstance(start, list) else tuple(out)
    if set(map(type, items)) <= _FLAT:
        _charge(len(items))
        return sum(items, start) if isinstance(start, _NUMBER) else float(sum(items)) + _array(start)
    _charge(6)
    return np.sum(_array(items), axis=0) + _array(start)


def _round(x, ndigits=0, **kw):
    if type(x) is float and type(ndigits) is float and ndigits - ndigits == 0.0 and ndigits == int(ndigits) and x - x == 0.0:
        return round(x, int(ndigits))
    digits = _int(0 if ndigits is None else ndigits, "the number of digits")
    if type(x) is float:                                                 # Python's rounding of a number: round(0.12345, 4) is 0.1235
        return round(x, digits) if x - x == 0.0 else x
    return np.round(_array(x), digits)


def _range(*a):
    if not 1 <= len(a) <= 3 or any(not _whole(x) for x in a):
        raise CalcError("range takes one to three whole numbers: range(n), range(a, b), range(a, b, step); int(x) cuts a fraction, "
                        "linspace(a, b, n) and arange(a, b, step) take any numbers", "value")
    if len(a) == 3 and a[2] == 0:
        raise CalcError("range with a step of 0", "value")
    r = range(*[int(x) for x in a])
    if len(r) > MAX_ITEMS:                                               # a loop needs no list: the step limit bounds it
        return _Lazy(r)
    _charge(2 * len(r))
    return [float(x) for x in r]


def _linspace(a, b, n=50, endpoint=True, **kw):
    if not _whole(n) or n < 0:
        raise CalcError("linspace(a, b, n) takes a whole number of points", "value")
    _room(n, "a linspace")
    return np.linspace(_array(a), _array(b), int(n), endpoint=bool(endpoint))


def _arange(*a, **kw):
    if not 1 <= len(a) <= 3 or not all(isinstance(x, _REAL) for x in a):
        raise CalcError("arange takes one to three numbers: arange(n), arange(a, b), arange(a, b, step)", "value")
    start, stop, step = (0.0, a[0], 1.0) if len(a) == 1 else (a[0], a[1], 1.0) if len(a) == 2 else a
    if not step or not all(math.isfinite(v) for v in (start, stop, step)):
        raise CalcError("arange needs finite numbers and a step that is not 0", "value")
    _room(max(0.0, math.ceil((stop - start) / step)), "an arange")
    return np.arange(float(start), float(stop), float(step))


def _logspace(a, b, n=50, base=10.0, **kw):
    return np.power(float(base), _linspace(a, b, n))


def _geomspace(a, b, n=50, **kw):
    if not (a > 0 and b > 0):
        raise CalcError("geomspace needs two positive ends", "value")
    return np.exp(_linspace(math.log(a), math.log(b), n))


def _filled(value):
    def call(shape, fill=None, **kw):
        dims = tuple(_int(n, "a size") for n in (shape if isinstance(shape, (list, tuple)) else [shape]))
        if any(n < 0 for n in dims):
            raise CalcError("a size is not negative", "value")
        _room(math.prod(dims), "an array")
        return np.full(dims, value if value is not None else _array(fill))
    return call


def _like(value):
    return lambda x, **kw: np.full(_array(x).shape, value)


def _eye(n, **kw):
    n = _int(n, "a size")
    _room(n * n, "a matrix")
    return np.eye(n)


def _float(x=0.0):
    if isinstance(x, str):
        try:
            return float(x.strip())
        except ValueError:
            raise CalcError(f"float: {x[:24]!r} is not a number", "value") from None
    return _array(x)


def _trunc(x=0.0, base=None):
    if type(x) is float and x == x and abs(x) < 1e300:
        return float(int(x))
    if isinstance(x, str):
        return np.trunc(_float(x))
    return np.trunc(_array(x))


def _str(x=""):
    _charge(2)
    return _text(x)


def _len(x):
    if isinstance(x, (list, tuple, dict, str, np.ndarray)):
        return float(len(x))
    raise CalcError(f"len of {_kind(x)}" + (_NONE_HINT if x is None else ": it is one number here, not a list" if isinstance(x, _NUMBER) else ""), "value")


def _list(x=()):
    if isinstance(x, np.ndarray):
        return x.tolist() if x.ndim == 1 else [row for row in x]
    return list(_iterable(x))


def _zip(*a, strict=False):
    return [tuple(t) for t in zip(*[_iterable(x) for x in a])]


def _enumerate(x, start=0):
    return [(float(i), v) for i, v in enumerate(_iterable(x), _int(start, "start"))]


def _dict(*a, **kw):
    out = {}
    _charge(8)
    for x in a:
        for pair in (x.items() if isinstance(x, dict) else _iterable(x)):
            k, v = pair
            out[_key(k)] = v
    out.update(kw)
    return out


def _map(fn, *lists):
    out = [fn(*t) for t in zip(*[_iterable(x) for x in lists])]
    _charge(len(out))
    return out


def _filter(fn, x):
    return [v for v in _iterable(x) if _is_true(v if fn is None else fn(v))]


def _reversed(x):
    return list(_iterable(x))[::-1]


def _any(x, **kw):
    return bool(np.any(x)) if isinstance(x, np.ndarray) else any(_is_true(v) for v in _iterable(x))


def _all(x, **kw):
    return bool(np.all(x)) if isinstance(x, np.ndarray) else all(_is_true(v) for v in _iterable(x))


def _isinstance(x, kinds):
    kinds = kinds if isinstance(kinds, tuple) else (kinds,)
    _charge(4)
    names = {next((k for k in _KINDS if FUNCTIONS[k] is f), None) for f in kinds}
    if None in names:
        raise CalcError("isinstance tests for int, float, complex, bool, str, list, tuple, dict or array", "value")
    return any(_KINDS[n](x) for n in names)


_KINDS = dict(
    int=_whole, float=lambda x: isinstance(x, _REAL) and not isinstance(x, (bool, np.bool_)), bool=lambda x: isinstance(x, (bool, np.bool_)),
    complex=lambda x: isinstance(x, (complex, np.complexfloating)), str=lambda x: isinstance(x, str), list=lambda x: isinstance(x, list),
    tuple=lambda x: isinstance(x, tuple), dict=lambda x: isinstance(x, dict), array=lambda x: isinstance(x, np.ndarray))


def _finite(what, *arrays):
    for a in arrays:
        if not np.all(np.isfinite(a)):
            raise CalcError(f"{what} needs finite numbers (there is an inf or a nan in them)", "value")


def _polyfit(x, y, deg, **kw):
    x, y, deg = _array(x), _array(y), _int(deg, "the degree")
    if x.ndim != 1 or x.shape != y.shape:
        raise CalcError(f"polyfit(x, y, degree) needs two lists of one length: here {_shape(x)} and {_shape(y)} values", "value")
    if not 0 <= deg < x.size or deg > 20:
        raise CalcError(f"polyfit: a degree of {deg} needs at least {deg + 1} points (and at most degree 20)", "value")
    _finite("polyfit", x, y)
    _charge(120 + x.size * (deg + 1) ** 2)
    lhs = np.vander(x.astype(float), deg + 1)                           # numpy's polyfit: scaled columns, least squares
    scale = np.sqrt((lhs * lhs).sum(axis=0))
    scale[scale == 0] = 1.0
    try:
        c = np.linalg.lstsq(lhs / scale, y, rcond=x.size * np.finfo(float).eps)[0]
    except np.linalg.LinAlgError:
        raise CalcError("polyfit: these points do not determine the polynomial", "value") from None
    return (c.T / scale).T


def _polyval(p, x):
    p, x = _array(p), _array(x)
    _charge(p.size * max(1, x.size))
    return np.polyval(p, x)


def _roots(p):
    p = _array(p)
    if p.ndim != 1 or not 2 <= p.size <= 21:
        raise CalcError("roots takes the coefficients of a polynomial of degree 1 to 20, highest power first", "value")
    _finite("roots", p)
    _charge(p.size ** 3)
    r = np.roots(p)
    return r.real if np.all(np.abs(r.imag) < 1e-12 * (1 + np.abs(r.real))) else r


def _interp(*a, left=None, right=None, period=None, **kw):
    """numpy's ``interp(x, xp, fp, left=None, right=None)``: outside ``xp`` it gives the end values of ``fp`` (or left / right)."""
    usage = "interp(x, xp, fp, left=None, right=None): x a number or a list, xp and fp two lists of one length, xp rising"
    if not 3 <= len(a) <= 5 or kw or period is not None:
        raise CalcError(f"{usage}; here {len(a)} argument(s)" + (f" and {next(iter(kw))}=" if kw else ""), "value")
    left, right = (a[3] if len(a) > 3 else left), (a[4] if len(a) > 4 else right)
    if not all(v is None or isinstance(v, _REAL) for v in (left, right)):
        raise CalcError(f"{usage}; here left / right is {_brief(left if not isinstance(left, _REAL) else right)}", "value")
    x, xp, fp = (_array(v) for v in a[:3])
    if xp.ndim != 1 or xp.shape != fp.shape or not xp.size:
        raise CalcError(f"{usage}; here xp is {_brief(a[1])} and fp is {_brief(a[2])}", "value")
    if xp.dtype.kind == "c" or x.dtype.kind == "c":
        raise CalcError(f"{usage}; x and xp are real numbers", "value")
    _charge(4 + x.size + 2 * xp.size)
    step = np.diff(xp)
    if xp.size > 1 and np.all(step < 0):                               # falling throughout: the same points read from the other end
        xp, fp = xp[::-1], fp[::-1]
    elif not np.all(step >= 0):
        raise CalcError(f"{usage}; here xp goes up and down: sort the points by xp first (sorted(zip(xp, fp)))", "value")
    return np.interp(x, xp, fp, None if left is None else float(_array(left)), None if right is None else float(_array(right)))


def _where(c, a=_MISSING, b=_MISSING):
    if a is _MISSING or b is _MISSING:
        idx = np.nonzero(_array(c))
        return idx[0].astype(float) if len(idx) == 1 else [i.astype(float) for i in idx]
    c = _array(c)
    x, y = _pair(a, b)
    _pair(c, x if x.ndim >= y.ndim else y)
    return np.where(c, x, y)


def _clip(x, lo=None, hi=None, **kw):
    x = _array(x)
    lo, hi = kw.get("a_min", kw.get("min", lo)), kw.get("a_max", kw.get("max", hi))
    return np.clip(x, None if lo is None else _array(lo), None if hi is None else _array(hi))


def _dot(a, b):
    x, y = _array(a), _array(b)
    if x.ndim > 2 or y.ndim > 2:
        raise CalcError("dot takes lists and matrices", "value")
    if x.ndim == 2 and y.ndim == 2:
        _room(x.shape[0] * y.shape[1], "a product")
    _charge(x.size * (y.shape[-1] if y.ndim == 2 else 1) + y.size)
    try:
        return np.dot(x, y)
    except ValueError:
        raise CalcError(f"dot: the shapes {_shape(x) if x.ndim else 1} and {_shape(y) if y.ndim else 1} do not fit", "value") from None


def _outer(a, b):
    x, y = _array(a).ravel(), _array(b).ravel()
    _room(x.size * y.size, "an outer product")
    return np.outer(x, y)


def _cross(a, b):
    x, y = _array(a), _array(b)
    if x.shape != y.shape or x.shape not in ((3,), (2,)):
        raise CalcError("cross takes two vectors of 3 (or of 2) numbers", "value")
    return x[0] * y[1] - x[1] * y[0] if x.shape == (2,) else np.array([x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]])


def _stack(how):
    def call(*a, axis=0, **kw):
        parts = a[0] if len(a) == 1 and isinstance(a[0], (list, tuple)) else a
        if isinstance(parts, np.ndarray):
            parts = list(parts)
        arrays = [np.atleast_1d(_array(p)) for p in parts]
        if not arrays:
            raise CalcError("nothing to stack", "value")
        _room(sum(x.size for x in arrays), "an array")
        try:
            if how == "column":
                return np.column_stack(arrays)
            if how == "v":
                return np.vstack(arrays)
            if how == "h":
                return np.hstack(arrays)
            if how == "stack":
                return np.stack(arrays, axis=_int(axis, "axis"))
            return np.concatenate(arrays, axis=_int(axis, "axis"))
        except ValueError:
            raise CalcError(f"these lists do not fit together: their shapes are {', '.join(_shape(x) for x in arrays[:6])}", "value") from None
    return call


def _meshgrid(*a, indexing="xy", **kw):
    arrays = [_array(x).ravel() for x in a]
    if not 1 <= len(arrays) <= 3 or indexing not in ("xy", "ij"):
        raise CalcError("meshgrid takes one to three lists", "value")
    _room(len(arrays) * math.prod(x.size for x in arrays), "a grid")
    return [g for g in np.meshgrid(*arrays, indexing=indexing)]


def _reshape(x, *shape, **kw):
    x = _array(x)
    dims = shape[0] if len(shape) == 1 and isinstance(shape[0], (list, tuple)) else shape
    try:
        return x.reshape(tuple(_int(n, "a size") for n in dims))
    except ValueError:
        raise CalcError(f"reshape: {x.size} values do not fill the shape {' x '.join(_text(n) for n in dims)}", "value") from None


def _diff(x, n=1, **kw):
    return np.diff(_array(x), _int(n, "n"))


def _trapz(y, x=None, dx=1.0, **kw):
    y = _array(y)
    if x is None:
        return float(dx) * (np.sum(y) - 0.5 * (y[0] + y[-1])) if y.size else 0.0
    x = _array(x)
    if x.shape != y.shape or x.ndim != 1:
        raise CalcError("trapz(y, x) needs two lists of one length", "value")
    return np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x))


def _percentile(scale):
    def call(x, q, **kw):
        x, q = _array(x), _array(q)
        if not x.size:
            raise CalcError("percentile of an empty list", "value")
        _charge(x.size * max(1, x.size.bit_length()))
        return np.quantile(x.ravel(), np.clip(q / scale, 0.0, 1.0))
    return call


def _arg(fn, x):
    x = _array(x)
    if not x.size:
        raise CalcError("argmin / argmax of an empty list", "value")
    _charge(4)
    return float(fn(x))


def _argsort(x, **kw):
    x = _array(x)
    _charge(x.size * max(1, x.size.bit_length()))
    return np.argsort(x, kind="stable").astype(float)


def _np_sort(x, **kw):
    x = _array(x)
    _charge(x.size * max(1, x.size.bit_length()))
    return np.sort(x)


def _unique(x, **kw):
    x = _array(x)
    _charge(x.size * max(1, x.size.bit_length()))
    return np.unique(x)


def _np_append(x, values, **kw):
    return np.concatenate([np.atleast_1d(_array(x)).ravel(), np.atleast_1d(_array(values)).ravel()])


def _repeat(x, n, **kw):
    x, n = np.atleast_1d(_array(x)), _int(n, "the number of repeats")
    _room(x.size * max(n, 0), "an array")
    return np.repeat(x, n)


def _tile(x, n, **kw):
    x, n = np.atleast_1d(_array(x)), _int(n, "the number of repeats")
    _room(x.size * max(n, 0), "an array")
    return np.tile(x, n)


def _isclose(a, b, rel_tol=1e-9, abs_tol=0.0, rtol=None, atol=None, **kw):
    x, y = _pair(a, b)
    if rtol is not None or atol is not None:                           # numpy's rule
        return np.abs(x - y) <= (1e-8 if atol is None else atol) + (1e-5 if rtol is None else rtol) * np.abs(y)
    return np.abs(x - y) <= np.maximum(rel_tol * np.maximum(np.abs(x), np.abs(y)), abs_tol)


def _allclose(a, b, rtol=1e-5, atol=1e-8, **kw):
    return bool(np.all(_isclose(a, b, rtol=rtol, atol=atol)))


def _whole_fn(fn, what):
    def call(*a):
        return float(fn(*[_small(x, what) for x in a]))
    return call


def _small(x, what):
    n = _int(x, what)
    if not 0 <= n <= 170 and what != "gcd":
        raise CalcError(f"{what} takes whole numbers from 0 to 170", "value")
    return n


def _divmod(a, b):
    _charge(8)
    x, y = _pair(a, b)
    return (np.floor_divide(x, y), np.mod(x, y))


def _angle(z, deg=False):
    return np.angle(_array(z), deg=bool(deg))


def _complex(re=0.0, im=0.0):
    return complex(float(_array(re)), float(_array(im))) if not isinstance(re, (complex, np.complexfloating)) else complex(re) + 1j * float(im)


def _format_fn(v, spec=""):
    return _format(v, _text(spec))


def _norm(x, ord=None, **kw):
    x = _array(x)
    if ord not in (None, 1, 2, math.inf) or x.ndim > 2:
        raise CalcError("norm(x) is the root of the sum of squares; ord may be 1, 2 or inf for a list", "value")
    v = np.abs(x.ravel())
    return np.sqrt(np.sum(v * v)) if ord in (None, 2) else np.sum(v) if ord == 1 else np.max(v)


def _matrix(a, what, square=True):
    a = _array(a)
    if a.ndim != 2 or (square and a.shape[0] != a.shape[1]) or not a.size:
        raise CalcError(f"{what} needs a {'square ' if square else ''}matrix, written as a list of rows: [[a, b], [c, d]]" +
                        (f" (this has the shape {_shape(a)})" if a.ndim else ""), "value")
    if max(a.shape) > (MAX_MATRIX if square else MAX_ITEMS) or min(a.shape) > MAX_MATRIX:
        raise CalcError(f"{what} takes at most {MAX_MATRIX} unknowns", "value")
    _finite(what, a)
    _charge(60 + max(a.shape) * min(a.shape) ** 2)
    return a


def _solve(a, b):
    a, b = _matrix(a, "solve(A, b)"), _array(b)
    if not b.ndim or b.shape[0] != a.shape[0]:
        raise CalcError(f"solve(A, b): A has {a.shape[0]} rows and b has {b.shape[0] if b.ndim else 1} value(s)", "value")
    _finite("solve(A, b)", b)
    try:
        return np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        raise CalcError("solve(A, b): the matrix is singular (its rows are not independent): lstsq(A, b)[0] gives the least-squares answer",
                        "value") from None


def _inv(a):
    a = _matrix(a, "inv(A)")
    try:
        return np.linalg.inv(a)
    except np.linalg.LinAlgError:
        raise CalcError("inv(A): the matrix is singular (its rows are not independent)", "value") from None


def _det(a):
    return np.linalg.det(_matrix(a, "det(A)"))


def _lstsq(a, b, rcond=None, **kw):
    a, b = _matrix(a, "lstsq(A, b)", square=False), _array(b)
    if not b.ndim or b.shape[0] != a.shape[0]:
        raise CalcError(f"lstsq(A, b): A has {a.shape[0]} rows and b has {b.shape[0] if b.ndim else 1} value(s)", "value")
    _finite("lstsq(A, b)", b)
    x, residuals, rank, sv = np.linalg.lstsq(a, b, rcond=None)
    return (x, residuals, float(rank), sv)


def _pinv(a, **kw):
    return np.linalg.pinv(_matrix(a, "pinv(A)", square=False))


def _transpose(x, **kw):
    return _array(x).T


def _diag(x, **kw):
    x = _array(x)
    if x.ndim == 1:
        _room(x.size * x.size, "a matrix")
    return np.diag(x)


def _stat(fn, least, what):
    def call(x, *more):
        x = _array(x).ravel() if not more else np.array([_array(x)] + [_array(m) for m in more]).ravel()
        if x.size < least:
            raise CalcError(f"{what} needs at least {least} value(s)", "value")
        _charge(20 + x.size)
        return fn(x)
    return call


def _product(*lists, repeat=1):
    pools = [list(_iterable(x)) for x in lists] * _int(repeat, "repeat")
    _room(math.prod(len(p) for p in pools) * max(1, len(pools)), "a product")
    return [tuple(t) for t in _itertools.product(*pools)]


def _choose(fn, what):
    def call(x, r=None):
        pool = list(_iterable(x))
        r = len(pool) if r is None else _int(r, "the size of a choice")
        if r < 0:
            raise CalcError(f"{what}: a size is not negative", "value")
        count = math.comb(len(pool), r) if what == "combinations" else math.perm(len(pool), r) if what == "permutations" else \
            math.comb(len(pool) + r - 1, r) if pool or not r else 0
        _room(count * max(1, r), what)
        return [tuple(t) for t in fn(pool, r)]
    return call


def _chain(*lists):
    out = []
    for x in lists:
        out.extend(_iterable(x))
        _room(len(out))
    return out


def _accumulate(x, func=None, initial=_MISSING):
    items = list(_iterable(x)) if initial is _MISSING else [initial] + list(_iterable(x))
    out = []
    for v in items:
        out.append(v if not out else (out[-1] + v if func is None else func(out[-1], v)))
    return out


def _bisect_fn(fn):
    def call(a, x, lo=0, hi=None, key=None):
        items = a.tolist() if isinstance(a, np.ndarray) else _iterable(a)
        if not set(map(type, items)) <= _FLAT or not isinstance(x, _REAL):
            raise CalcError("bisect searches a sorted list of numbers for a number", "value")
        return float(fn(items, x, _int(lo, "lo"), len(items) if hi is None else _int(hi, "hi")))
    return call


def _insort(a, x):
    if not isinstance(a, list) or len(a) >= MAX_ITEMS:
        raise CalcError("insort puts a number into a sorted list", "value")
    a.insert(int(_bisect_fn(_bisect.bisect_right)(a, x)), x)


def _deepcopy(x):
    """A copy of a list / dictionary and of what it holds (``copy.deepcopy``)."""
    if isinstance(x, (list, tuple)):
        _charge(1 + len(x))
        return type(x)(_deepcopy(v) for v in x)
    if isinstance(x, dict):
        _charge(1 + len(x))
        return type(x)((k, _deepcopy(v)) for k, v in x.items())
    return x.copy() if isinstance(x, np.ndarray) else x


def _copy(x):
    return list(x) if isinstance(x, list) else dict(x) if type(x) is dict else x.copy() if isinstance(x, np.ndarray) else x


def _reduce(fn, x, initial=_MISSING):
    items = list(_iterable(x))
    if initial is _MISSING:
        if not items:
            raise CalcError("reduce of an empty list", "value")
        initial, items = items[0], items[1:]
    for v in items:
        initial = fn(initial, v)
    return initial


# ---- root finding, minimising and least squares on a function defined in the text. The work is scipy's; every evaluation
# ---- of the function is charged its steps, so the step limit bounds them as it bounds a loop.
class _Result(dict):
    """What ``minimize`` / ``least_squares`` / ``minimize_scalar`` give back: ``res.x``, ``res.fun``, ``res.success`` ... (also ``res['x']``)."""


def _optimize():
    _charge(400)                                                         # the set-up of a solver, apart from its evaluations
    from scipy import optimize  # noqa: PLC0415  (loaded when a text first asks for it)
    return optimize


def _callable(f, what):
    if not callable(f):
        raise CalcError(f"{what} takes a function first: def f(x): ... or lambda x: ...", "value")
    return f


def _scalar_of(f, args=()):
    def call(x):
        _charge(12)
        v = f(float(x), *args)
        if isinstance(v, np.ndarray) and v.size == 1:
            v = v.item()
        if not isinstance(v, _REAL):
            raise CalcError(f"the function must give one number, it gave {_kind(v)}", "value")
        return float(v)
    return call


def _vector_of(f, args=()):
    def call(x, *more):
        _charge(60 + x.size)
        v = f(x, *more, *args)
        return np.atleast_1d(_array(v)).astype(float).ravel()
    return call


def _unknowns(x0, what):
    x = np.atleast_1d(_array(x0)).astype(float).ravel()
    if not 1 <= x.size <= MAX_PARAMETERS:
        raise CalcError(f"{what} takes 1 to {MAX_PARAMETERS} unknowns", "value")
    _finite(what + ": the start", x)
    return x


def _bounds_of(bounds, n, what):
    if bounds is None:
        return None
    try:
        if len(bounds) == 2 and not all(isinstance(b, (list, tuple, np.ndarray)) and len(b) == 2 for b in bounds):
            lo, hi = (np.broadcast_to(_array(b).astype(float), (n,)) for b in bounds)       # (lows, highs)
        else:
            lo, hi = np.array([[float(-math.inf if a is None else a), float(math.inf if b is None else b)] for a, b in bounds]).T
    except (TypeError, ValueError):
        raise CalcError(f"{what}: bounds are (lows, highs), or one (low, high) per unknown", "value") from None
    if lo.shape != (n,) or np.any(lo >= hi):
        raise CalcError(f"{what}: {n} unknown(s) need {n} bounds with low < high", "value")
    return lo, hi


def _brentq(f, a, b, args=(), xtol=2e-12, rtol=8.9e-16, maxiter=100, **kw):
    f = _scalar_of(_callable(f, "brentq(f, a, b)"), tuple(args))
    a, b = float(_array(a)), float(_array(b))
    fa, fb = f(a), f(b)
    if not (math.isfinite(fa) and math.isfinite(fb)) or fa * fb > 0:
        raise CalcError(f"brentq(f, a, b) needs f(a) and f(b) of opposite sign: f({_g(a)}) = {_g(fa)}, f({_g(b)}) = {_g(fb)}; "
                        "tabulate f over a grid to find a bracket", "value")
    if fa == 0 or fb == 0:
        return a if fa == 0 else b
    return float(_optimize().brentq(f, a, b, xtol=float(xtol), rtol=max(float(rtol), 8.9e-16), maxiter=min(_int(maxiter, "maxiter"), 200)))


def _fsolve(f, x0, args=(), **kw):
    x0 = _unknowns(x0, "fsolve(f, x0)")
    g = _vector_of(_callable(f, "fsolve(f, x0)"), tuple(args))
    one = lambda x: g(x[0] if x0.size == 1 else x)                     # noqa: E731
    x, _, ok, _ = _optimize().fsolve(one, x0, full_output=True, maxfev=400 * (x0.size + 1))
    if ok != 1:
        raise CalcError("fsolve did not converge from this start: bracket the root and use brentq(f, a, b), or try another start", "value")
    return x


def _minimize_scalar(f, bracket=None, bounds=None, args=(), method=None, **kw):
    f = _scalar_of(_callable(f, "minimize_scalar(f, bounds=(a, b))"), tuple(args))
    if bounds is None and bracket is not None and len(bracket) == 2:
        bounds = bracket
    if bounds is None:
        raise CalcError("minimize_scalar needs bounds=(a, b): the interval to search", "value")
    a, b = (float(_array(v)) for v in bounds)
    if not (math.isfinite(a) and math.isfinite(b) and a < b):
        raise CalcError("minimize_scalar: bounds=(a, b) with a < b, both finite", "value")
    r = _optimize().minimize_scalar(f, bounds=(a, b), method="bounded", options=dict(maxiter=500, xatol=1e-10 * (1 + abs(a) + abs(b))))
    return _Result(x=float(r.x), fun=float(r.fun), success=bool(r.success), nfev=float(r.nfev))


def _minimize(f, x0, args=(), method=None, bounds=None, **kw):
    x0 = _unknowns(x0, "minimize(f, x0)")
    g = _vector_of(_callable(f, "minimize(f, x0)"), tuple(args))

    def cost(x):
        v = g(x)
        if v.size != 1:
            raise CalcError(f"minimize: the function must give one number (a sum of squares), it gave {v.size}", "value")
        return float(v[0])
    box = _bounds_of(bounds, x0.size, "minimize")
    r = _optimize().minimize(cost, x0, method="Nelder-Mead", bounds=None if box is None else list(zip(*box)),
                             options=dict(maxiter=400 * x0.size, maxfev=600 * x0.size, xatol=1e-9, fatol=1e-12))
    return _Result(x=r.x, fun=float(r.fun), success=bool(r.success), nfev=float(r.nfev), message=str(r.message)[:80])


def _least_squares(f, x0, bounds=None, args=(), **kw):
    x0 = _unknowns(x0, "least_squares(f, x0)")
    g = _vector_of(_callable(f, "least_squares(f, x0)"), tuple(args))
    box = _bounds_of(bounds, x0.size, "least_squares")
    if g(x0).size < x0.size:
        raise CalcError(f"least_squares: {x0.size} unknowns need at least {x0.size} residuals", "value")
    inner = g

    def g(x):                                                            # noqa: F811  the solver's own work per evaluation
        _charge(200)
        return inner(x)
    try:
        r = _optimize().least_squares(g, x0, bounds=(-np.inf, np.inf) if box is None else box, max_nfev=200 * (x0.size + 1))
    except ValueError as e:
        raise CalcError(f"least_squares: {str(e)[:100]}", "value") from None
    return _Result(x=r.x, cost=float(r.cost), fun=r.fun, success=bool(r.success), nfev=float(r.nfev), message=str(r.message)[:80])


def _curve_fit(f, xdata, ydata, p0=None, sigma=None, bounds=None, **kw):
    f = _callable(f, "curve_fit(f, x, y, p0)")
    x, y = _array(xdata), np.atleast_1d(_array(ydata)).astype(float)
    if p0 is None:
        n = len(f.params) - 1 if isinstance(f, _Function) else 0
        if n < 1:
            raise CalcError("curve_fit(f, x, y, p0) needs the start values p0, one per unknown of f(x, a, b, ...)", "value")
        p0 = [1.0] * n
    p0 = _unknowns(p0, "curve_fit(f, x, y, p0)")
    if y.size < p0.size:
        raise CalcError(f"curve_fit: {p0.size} unknowns need at least {p0.size} points, here {y.size}", "value")
    _finite("curve_fit", y)
    weight = 1.0 if sigma is None else 1.0 / np.broadcast_to(_array(sigma).astype(float), y.shape)

    def residual(p):
        _charge(250 + y.size)
        v = f(x, *p.tolist())
        v = np.atleast_1d(_array(v)).astype(float)
        if v.shape != y.shape:
            if v.size == 1 and y.size > 1:
                raise CalcError("curve_fit: f(x, ...) must work on the whole list x (use exp, not math.exp of one number at a time: "
                                "calc's exp takes lists)", "value")
            raise CalcError(f"curve_fit: f(x, ...) gave {v.size} values for {y.size} points", "value")
        return (v - y) * weight
    box = _bounds_of(bounds, p0.size, "curve_fit")
    if box is not None:
        p0 = np.clip(p0, np.where(np.isfinite(box[0]), box[0], p0), np.where(np.isfinite(box[1]), box[1], p0))
    try:
        r = _optimize().least_squares(residual, p0, bounds=(-np.inf, np.inf) if box is None else box, max_nfev=200 * (p0.size + 1))
    except ValueError as e:
        raise CalcError(f"curve_fit: {str(e)[:100]}", "value") from None
    jac = np.atleast_2d(r.jac)
    dof = max(1, y.size - p0.size)
    with np.errstate(all="ignore"):
        cov = np.linalg.pinv(jac.T @ jac) * (1.0 if sigma is not None and kw.get("absolute_sigma") else 2.0 * r.cost / dof)
    return (r.x, cov)


class _Lazy:
    """``range(...)`` of more than ``MAX_ITEMS`` numbers: a ``for`` loop or a comprehension may run over it, nothing holds it as a list."""
    __slots__ = ("r",)

    def __init__(self, r):
        self.r = r

    def __len__(self):
        return len(self.r)

    def __iter__(self):
        return map(float, self.r)


def _loop_over(value):
    return value if type(value) is _Lazy else _iterable(value)


class _Module:
    """A prefix a text may write: ``math.exp``, ``np.linalg.solve``. A table of the calculator's own functions."""
    __slots__ = ("name", "members")

    def __init__(self, name, members):
        self.name, self.members = name, members


def _mean_of(x, *a, **kw):
    return FUNCTIONS["mean"](x, *a, **kw)


FUNCTIONS = dict(
    sin=_one(math.sin, np.sin), cos=_one(math.cos, np.cos), tan=_one(math.tan, np.tan),
    arcsin=_one(math.asin, np.arcsin), arccos=_one(math.acos, np.arccos), arctan=_one(math.atan, np.arctan),
    asin=_one(math.asin, np.arcsin), acos=_one(math.acos, np.arccos), atan=_one(math.atan, np.arctan),
    arctan2=_two(np.arctan2, math.atan2), atan2=_two(np.arctan2, math.atan2), sinh=_one(math.sinh, np.sinh), cosh=_one(math.cosh, np.cosh), tanh=_one(math.tanh, np.tanh),
    arcsinh=_one(math.asinh, np.arcsinh), arccosh=_one(math.acosh, np.arccosh), arctanh=_one(math.atanh, np.arctanh),
    asinh=_one(math.asinh, np.arcsinh), acosh=_one(math.acosh, np.arccosh), atanh=_one(math.atanh, np.arctanh),
    exp=_one(math.exp, np.exp), expm1=_one(math.expm1, np.expm1), exp2=_one(lambda x: 2.0 ** x, np.exp2),
    log=_log, ln=_log, log10=_one(math.log10, np.log10), log2=_one(math.log2, np.log2), log1p=_one(math.log1p, np.log1p),
    sqrt=_one(math.sqrt, np.sqrt), cbrt=_one(lambda x: math.copysign(abs(x) ** (1 / 3), x), np.cbrt), square=lambda x: np.square(_array(x)),
    abs=_abs, fabs=_abs, absolute=_abs, degrees=_one(math.degrees, np.degrees), radians=_one(math.radians, np.radians),
    deg2rad=_one(math.radians, np.radians), rad2deg=_one(math.degrees, np.degrees), hypot=_two(np.hypot, math.hypot),
    ceil=_one(lambda x: float(math.ceil(x)), np.ceil), floor=_one(lambda x: float(math.floor(x)), np.floor), trunc=_trunc, fix=_trunc,
    rint=lambda x: np.rint(_array(x)), round=_round, around=_round, sign=lambda x: np.sign(_array(x)), copysign=_two(np.copysign),
    clip=_clip, maximum=_two(np.maximum, _max2), minimum=_two(np.minimum, _min2), fmax=_two(np.fmax), fmin=_two(np.fmin),
    mod=_two(np.mod), fmod=_two(np.fmod), remainder=_two(np.mod), power=_two(np.power), pow=_two(np.power), divmod=_divmod,
    add=_two(np.add), subtract=_two(np.subtract), multiply=_two(np.multiply), divide=_two(np.divide), negative=lambda x: np.negative(_array(x)),
    erf=_one(math.erf, lambda x: np.vectorize(math.erf, otypes=[float])(x)), erfc=_one(math.erfc, lambda x: np.vectorize(math.erfc, otypes=[float])(x)),
    gamma=_one(math.gamma, lambda x: np.vectorize(math.gamma, otypes=[float])(x)), sinc=lambda x: np.sinc(_array(x)),
    factorial=_whole_fn(math.factorial, "factorial"), comb=_whole_fn(math.comb, "comb"), perm=_whole_fn(math.perm, "perm"),
    gcd=_whole_fn(math.gcd, "gcd"),
    isnan=lambda x: np.isnan(_array(x)), isinf=lambda x: np.isinf(_array(x)), isfinite=lambda x: np.isfinite(_array(x)),
    isclose=_isclose, allclose=_allclose,
    mean=_reducer(np.mean, "mean"), average=_reducer(np.mean, "average"), median=_reducer(np.median, "median"),
    std=_reducer(np.std, "std"), var=_reducer(np.var, "var"), prod=_reducer(np.prod, "prod"), ptp=_reducer(np.ptp, "ptp"),
    nanmean=_reducer(np.nanmean, "nanmean"), nanmin=_reducer(np.nanmin, "nanmin"), nanmax=_reducer(np.nanmax, "nanmax"),
    nansum=_reducer(np.nansum, "nansum"), nanstd=_reducer(np.nanstd, "nanstd"), nanmedian=_reducer(np.nanmedian, "nanmedian"),
    amin=_reducer(np.min, "amin"), amax=_reducer(np.max, "amax"),
    sum=_sum, fsum=_sum, min=_extreme(np.min, -1), max=_extreme(np.max, 1),
    cumsum=lambda x, **kw: np.cumsum(_array(x)), cumprod=lambda x, **kw: np.cumprod(_array(x)), diff=_diff, trapz=_trapz, trapezoid=_trapz,
    argmin=lambda x, **kw: _arg(np.argmin, x), argmax=lambda x, **kw: _arg(np.argmax, x),
    argsort=_argsort, sort=_np_sort, unique=_unique, percentile=_percentile(100.0), quantile=_percentile(1.0),
    count_nonzero=lambda x, **kw: float(np.count_nonzero(_array(x))), nonzero=lambda x: _where(x), flatnonzero=lambda x: _where(np.ravel(_array(x))),
    any=_any, all=_all, int=_trunc, float=_float, bool=_is_true, str=_str, repr=_str, complex=_complex, len=_len, range=_range,
    list=_list, tuple=lambda x=(): tuple(_list(x)), dict=_dict, sorted=_sorted, reversed=_reversed, zip=_zip, enumerate=_enumerate,
    map=_map, filter=_filter, isinstance=_isinstance, format=_format_fn,
    array=lambda x, *a, **kw: _array(x), asarray=lambda x, *a, **kw: _array(x), float64=_float, atleast_1d=lambda x: np.atleast_1d(_array(x)),
    zeros=_filled(0.0), ones=_filled(1.0), empty=_filled(0.0), full=_filled(None), zeros_like=_like(0.0), ones_like=_like(1.0),
    eye=_eye, identity=_eye, diag=_diag,
    linspace=_linspace, arange=_arange, logspace=_logspace, geomspace=_geomspace, meshgrid=_meshgrid,
    column_stack=_stack("column"), vstack=_stack("v"), hstack=_stack("h"), stack=_stack("stack"), concatenate=_stack("concatenate"),
    append=_np_append, repeat=_repeat, tile=_tile, reshape=_reshape, ravel=lambda x: np.ravel(_array(x)), transpose=_transpose,
    interp=_interp, polyfit=_polyfit, polyval=_polyval, roots=_roots, where=_where,
    dot=_dot, matmul=_dot, inner=_dot, outer=_outer, cross=_cross, norm=_norm,
    real=lambda z: np.real(_array(z)), imag=lambda z: np.imag(_array(z)), conj=lambda z: np.conj(_array(z)),
    conjugate=lambda z: np.conj(_array(z)), angle=_angle, phase=_angle)
_FUNCS = FUNCTIONS                                                        # the name before the calculator read statements
CONSTANTS = dict(pi=math.pi, e=math.e, inf=math.inf, nan=math.nan, tau=math.tau, true=True, false=False, Inf=math.inf, NaN=math.nan,
                 infty=math.inf, newaxis=None)
_LINALG = _Module("np.linalg", dict(solve=_solve, inv=_inv, det=_det, lstsq=_lstsq, pinv=_pinv, norm=_norm))
_OPTIMIZE = _Module("scipy.optimize", dict(brentq=_brentq, bisect=_brentq, brenth=_brentq, ridder=_brentq, fsolve=_fsolve,
                                           minimize_scalar=_minimize_scalar, fminbound=lambda f, a, b, **kw: _minimize_scalar(f, bounds=(a, b))["x"],
                                           minimize=_minimize, least_squares=_least_squares, curve_fit=_curve_fit))
_NUMPY = dict(FUNCTIONS, linalg=_LINALG, **CONSTANTS)
_MATH = dict(FUNCTIONS, **CONSTANTS)
_STATISTICS = dict(
    mean=_stat(np.mean, 1, "mean"), fmean=_stat(np.mean, 1, "fmean"), median=_stat(np.median, 1, "median"),
    stdev=_stat(lambda x: np.std(x, ddof=1), 2, "stdev"), pstdev=_stat(np.std, 1, "pstdev"),
    variance=_stat(lambda x: np.var(x, ddof=1), 2, "variance"), pvariance=_stat(np.var, 1, "pvariance"))
_ITERTOOLS = dict(product=_product, combinations=_choose(_itertools.combinations, "combinations"),
                  permutations=_choose(_itertools.permutations, "permutations"),
                  combinations_with_replacement=_choose(_itertools.combinations_with_replacement, "combinations_with_replacement"),
                  chain=_chain, accumulate=_accumulate)
_CMATH = dict(exp=FUNCTIONS["exp"], sqrt=lambda z: np.sqrt(np.complex128(_array(z))), log=lambda z: np.log(np.complex128(_array(z))),
              phase=_angle, polar=lambda z: (abs(complex(_array(z))), float(np.angle(_array(z)))),
              rect=lambda r, phi: complex(r * math.cos(phi), r * math.sin(phi)), cos=FUNCTIONS["cos"], sin=FUNCTIONS["sin"],
              pi=math.pi, e=math.e, inf=math.inf, nan=math.nan, tau=math.tau)
MODULES = dict(
    math=_Module("math", _MATH), numpy=_Module("np", _NUMPY), np=_Module("np", _NUMPY), cmath=_Module("cmath", _CMATH),
    statistics=_Module("statistics", _STATISTICS), itertools=_Module("itertools", _ITERTOOLS),
    bisect=_Module("bisect", dict(bisect_left=_bisect_fn(_bisect.bisect_left), bisect_right=_bisect_fn(_bisect.bisect_right),
                                  bisect=_bisect_fn(_bisect.bisect_right), insort=_insort)),
    functools=_Module("functools", dict(reduce=_reduce)), copy=_Module("copy", dict(copy=_copy, deepcopy=_deepcopy)),
    scipy=_Module("scipy", dict(optimize=_OPTIMIZE)))
_SUBMODULES = {"numpy.linalg": _LINALG, "np.linalg": _LINALG, "scipy.optimize": _OPTIMIZE}
# What a text may ask for that the calculator does not have, and what to write instead.
_NO_MODULE = dict(
    random="no random numbers in calc (a call must give the same answer every time): use a grid, e.g. linspace(a, b, n)",
    scipy="calc has scipy.optimize only: brentq(f, a, b), minimize_scalar(f, bounds=(a, b)), minimize(f, x0), curve_fit(f, x, y, p0), "
          "least_squares(f, x0), fsolve(f, x0)",
    sympy="no symbolic algebra in calc: solve numerically with brentq(f, a, b) or a grid",
    mpmath="no mpmath in calc: floats have 15 digits; brentq(f, a, b) finds a root, minimize_scalar(f, bounds=(a, b)) a minimum",
    pandas="no tables in calc: use lists and dictionaries", matplotlib="calc draws nothing: print the numbers",
    sys="no sys in calc: a text ends at its last line (inf is the largest number)", os="calc reaches no files and no processes",
    time="no clock in calc", json="no json in calc: write the values as lists and dictionaries",
    collections="no collections in calc: a dictionary with d.get(k, 0) or d.setdefault(k, []) does it",
    decimal="no decimal in calc: floats have 15 digits", fractions="no fractions in calc: floats have 15 digits")
_INERT = ("sys", "warnings", "typing", "__future__")                     # imports that do nothing and bind nothing
_INSTEAD = dict(
    ClassDef="no classes in calc: keep the values in a dictionary or a tuple, the steps in functions", With="no with in calc",
    Nonlocal="no nonlocal in calc: return the value from the inner function", Delete="no del in calc: assign a new value instead",
    Await="no await in calc", Yield="no yield in calc: build a list and return it", YieldFrom="no yield in calc: build a list and return it",
    Set="no sets in calc: use a list (sorted(x) orders it, `v not in x` keeps it free of repeats)",
    SetComp="no sets in calc: use a list (sorted(x) orders it, `v not in x` keeps it free of repeats)",
    AsyncFor="no async in calc", AsyncWith="no async in calc", AsyncFunctionDef="no async in calc", Match="no match in calc: use if / elif",
    Bytes="no bytes in calc", Ellipsis="no ... in calc")
_NOT_HERE = dict(
    set="no sets in calc: use a list (sorted(x) orders it, `v not in x` keeps it free of repeats)", frozenset="no sets in calc: use a list",
    next="no iterators in calc: index the list (x[0]), or loop over it", iter="no iterators in calc: index the list (x[0]), or loop over it",
    input="calc reads nothing: write the numbers into the text", exit="a text ends at its last line", quit="a text ends at its last line",
    id="no object identities in calc", hash="no hashes in calc", chr="no character codes in calc", ord="no character codes in calc",
    hex="no hex in calc", bin="no binary in calc", oct="no octal in calc", slice="write the slice in brackets: x[a:b]",
    callable="calc has no callable(...)", super="no classes in calc", object="no classes in calc", bytes="no bytes in calc",
    help="the tool description lists what calc has", dir="the tool description lists what calc has")


# ------------------------------------------------------------------------------------------------ methods
def _grow(v, n=1):
    if len(v) + n > MAX_ITEMS:
        raise CalcError(f"a list of more than {MAX_ITEMS} values", "value")


def _l_append(v, x):
    _grow(v)
    v.append(x)


def _l_extend(v, x):
    items = _list(x)
    _grow(v, len(items))
    v.extend(items)


def _l_insert(v, i, x):
    _grow(v)
    v.insert(_int(i, "a position"), x)


def _l_pop(v, i=-1):
    if not v:
        raise CalcError("pop from an empty list: test len(x) first", "value")
    try:
        return v.pop(_int(i, "a position"))
    except IndexError:
        raise CalcError(f"pop: index {_text(i)} is out of range: the list has {len(v)} element(s)", "value") from None


def _find(v, x, start=0, stop=None):
    items = v.tolist() if isinstance(v, np.ndarray) else v
    stop = len(items) if stop is None else _int(stop, "a position")
    _charge(len(items))
    for i in range(*slice(_int(start, "a position"), stop).indices(len(items))):
        if _same(items[i], x):
            return i
    return -1


def _l_index(v, x, start=0, stop=None):
    i = _find(v, x, start, stop)
    if i < 0:
        raise CalcError(f"index: {_brief(x)} is not in the list" + (": for the place of the smallest value write argmin(x)"
                                                                    if isinstance(x, float) else ""), "value")
    return float(i)


def _l_remove(v, x):
    i = _find(v, x)
    if i < 0:
        raise CalcError(f"remove: {_brief(x)} is not in the list", "value")
    del v[i]


def _l_count(v, x):
    _charge(len(v))
    return float(sum(1 for e in v if _same(e, x)))


def _l_sort(v, key=None, reverse=False):
    v[:] = _sorted(v, key=key, reverse=reverse)


def _s_join(sep, parts):
    out, size = [], 0
    for p in _iterable(parts):
        p = _text(p)
        out.append(p)
        size += len(p) + len(sep)
        if size > MAX_TEXT:
            break
    return sep.join(out)[:MAX_TEXT]


def _s_replace(s, old, new, count=-1):
    old, new = _text(old), _text(new)
    if len(s) + s.count(old) * max(0, len(new) - len(old)) > MAX_TEXT:
        raise CalcError(f"a text of more than {MAX_TEXT} characters", "value")
    return s.replace(old, new, _int(count, "a count"))


def _s_pad(fn):
    def call(s, width, fill=" "):
        _narrow(_int(width, "a width"))
        return fn(s, int(width), fill) if fn is not str.zfill else s.zfill(int(width))
    return call


def _s_text(fn):
    return lambda s, *a: fn(s, *[x if isinstance(x, str) else _need_text(x) for x in a])


def _need_text(x):
    if isinstance(x, tuple) and all(isinstance(e, str) for e in x):
        return x
    if x is None or _whole(x):
        return x if x is None else int(x)
    raise CalcError(f"text is needed here, not {_kind(x)}", "value")


def _s_split(fn):
    def call(s, sep=None, maxsplit=-1):
        if sep is not None and (not isinstance(sep, str) or not sep):
            raise CalcError("split takes a separator text, or nothing (blanks)", "value")
        return fn(s, sep, _int(maxsplit, "maxsplit"))
    return call


def _d_get(d, k, default=None):
    return d.get(_key(k), default)


def _d_setdefault(d, k, default=None):
    k = _key(k)
    if k not in d:
        _grow(d)
    return d.setdefault(k, default)


def _d_update(d, other=(), **kw):
    for k, v in list(_dict(other, **kw).items()):
        if k not in d:
            _grow(d)
        d[k] = v


def _d_pop(d, k, default=_MISSING):
    k = _key(k)
    if k not in d:
        if default is _MISSING:
            raise CalcError(f"pop: no key {_text(k)!r}", "value")
        return default
    return d.pop(k)


def _a_fn(name):
    def method(v, *a, **kw):
        _charge(6)
        return FUNCTIONS[name](v, *a, **kw)
    return method


def _a_sort(v, **kw):
    _charge(v.size * max(1, v.size.bit_length()))
    v.sort()


def _a_fill(v, x):
    v.fill(float(_array(x)))


_VECTOR = {name: _a_fn(name) for name in (
    "sum", "mean", "min", "max", "std", "var", "prod", "argmin", "argmax", "argsort", "cumsum", "cumprod", "round", "clip", "dot",
    "ravel", "reshape", "transpose", "conj", "conjugate", "any", "all", "repeat", "nonzero")}
METHODS = {
    list: dict(_VECTOR, append=_l_append, extend=_l_extend, insert=_l_insert, pop=_l_pop, remove=_l_remove, clear=lambda v: v.clear(),
               index=_l_index, count=_l_count, sort=_l_sort, reverse=lambda v: v.reverse(), copy=lambda v: list(v)),
    tuple: dict(_VECTOR, index=_l_index, count=_l_count),
    np.ndarray: dict(_VECTOR, tolist=lambda v: v.tolist(), flatten=lambda v, **kw: v.ravel().copy(), copy=lambda v: v.copy(),
                     astype=lambda v, *a, **kw: v.astype(float), item=lambda v, *a: _item(v.ravel(), a[0] if a else 0.0),
                     sort=_a_sort, fill=_a_fill, index=_l_index, count=lambda v, x: float(np.count_nonzero(v == _array(x)))),
    str: dict(join=_s_join, format=_str_format, split=_s_split(str.split), rsplit=_s_split(str.rsplit), splitlines=lambda s: s.splitlines(),
              strip=_s_text(str.strip), lstrip=_s_text(str.lstrip), rstrip=_s_text(str.rstrip), upper=str.upper, lower=str.lower,
              title=str.title, capitalize=str.capitalize, startswith=_s_text(str.startswith), endswith=_s_text(str.endswith),
              replace=_s_replace, find=lambda s, x, *a: float(s.find(_text(x))), count=lambda s, x: float(s.count(_text(x))),
              index=lambda s, x: float(s.find(_text(x))), ljust=_s_pad(str.ljust), rjust=_s_pad(str.rjust), center=_s_pad(str.center),
              zfill=_s_pad(str.zfill), isdigit=str.isdigit, isalpha=str.isalpha, isnumeric=str.isnumeric, isspace=str.isspace),
    dict: dict(get=_d_get, setdefault=_d_setdefault, update=_d_update, pop=_d_pop, copy=lambda d: dict(d), clear=lambda d: d.clear(),
               items=lambda d: [tuple(kv) for kv in d.items()], keys=lambda d: list(d), values=lambda d: list(d.values())),
    float: dict(is_integer=lambda v: float(v) == int(v) if math.isfinite(v) else False, conjugate=lambda v: v),
    complex: dict(conjugate=lambda v: v.conjugate()),
}
METHODS[_Result] = METHODS[dict]
METHODS[_F64] = METHODS[float]
METHODS[np.complex128] = METHODS[complex]
# Attributes that are values, not calls.
_FIELDS = {
    np.ndarray: dict(T=lambda v: v.T, shape=lambda v: tuple(float(n) for n in v.shape), size=lambda v: float(v.size),
                     ndim=lambda v: float(v.ndim), real=lambda v: v.real, imag=lambda v: v.imag),
    float: dict(real=lambda v: v, imag=lambda v: 0.0), complex: dict(real=lambda v: v.real, imag=lambda v: v.imag),
}
_FIELDS[_F64] = _FIELDS[float]
_FIELDS[np.complex128] = _FIELDS[complex]


def _no_attribute(value, attr):
    if attr.startswith("_"):
        raise CalcError(f"no .{attr} in calc: calc has numbers, lists, text and dictionaries, and no objects behind them", "unsupported")
    if isinstance(value, _Module):
        names = [k for k, v in value.members.items() if not k.startswith("_")]
        near = difflib.get_close_matches(attr, names, 3, 0.6) or difflib.get_close_matches(attr.lower(), names, 3, 0.45)
        tail = f"; nearest: {', '.join(near)}" if near else f"; {value.name} has: {' '.join(sorted(names)[:40])}" if len(names) <= 40 else \
            ": write it with the functions of the tool description"
        raise CalcError(f"no {value.name}.{attr} in calc{tail}", "unsupported")
    table = METHODS.get(type(value))
    if value is None:
        raise CalcError(f"none has no .{attr}" + _NONE_HINT, "value")
    if table is None:
        raise CalcError(f"no .{attr} on {_kind(value)}: calc has numbers, lists, text and dictionaries (no objects)", "unsupported")
    if isinstance(value, dict) and attr in value:
        raise CalcError(f"a dictionary's value is read with brackets: d[{attr!r}]", "unsupported")
    names = sorted(k for k in table if k not in _VECTOR) + sorted(_FIELDS.get(type(value), ()))
    raise CalcError(f"no .{attr} on {_kind(value)}; it has: {' '.join(names)}", "unsupported")


def _attribute(value, attr):
    """``value.attr``: a method or a field from the tables, by the kind of the value. Never an attribute of the Python object."""
    t = type(value)
    table = METHODS.get(t)
    if table is not None:
        method = table.get(attr)
        if method is not None:
            _charge(2)
            return functools.partial(method, value)
    fields = _FIELDS.get(t)
    if fields is not None and attr in fields:
        return fields[attr](value)
    if t is _Result and attr in value:
        return value[attr]
    if t is _Module:
        if attr in value.members and not attr.startswith("_"):
            return value.members[attr]
    _no_attribute(value, attr)


# ------------------------------------------------------------------------------------------------ arithmetic
def _arith(ufunc, a, b):
    """Element-wise arithmetic on anything that is not two plain floats."""
    ta, tb = type(a), type(b)
    if ta is np.ndarray:                                                 # an array with a number, or with an array of its shape
        if (tb is float or (tb is np.ndarray and a.shape == b.shape and b.dtype is not _BOOL)) and a.dtype is not _BOOL:
            out = ufunc(a, b)
            _charge(4 + out.size)
            return out
    elif ta is float and tb is np.ndarray and b.dtype is not _BOOL:
        out = ufunc(a, b)
        _charge(4 + out.size)
        return out
    if a is None or b is None:
        raise CalcError("arithmetic with none" + _NONE_HINT, "value")
    if isinstance(a, str) or isinstance(b, str):
        raise CalcError("text takes no arithmetic: float(t) reads a number from text, + joins text, '%.2f' % x formats a number", "value")
    x, y = _array(a), _array(b)
    if x.ndim and y.ndim and x.shape != y.shape:
        _pair(x, y)
    if x.dtype.kind == "b":
        x = x.astype(float)
    if y.dtype.kind == "b":
        y = y.astype(float)
    out = ufunc(x, y)
    if out.ndim:
        _charge(4 + out.size)
        return out
    _charge(2)
    return out.item()


def _add(a, b):
    ta, tb = type(a), type(b)
    if ta is str or tb is str:                                           # a label next to a value
        out = (_text(a) + _text(b))[:MAX_TEXT]
        _charge(3 + (len(out) >> 3))
        return out
    if (ta is list or ta is tuple) and (tb is list or tb is tuple):      # as Python: the two joined
        _room(len(a) + len(b))
        return a + b if ta is tb else list(a) + list(b)
    if (ta is complex or ta is float) and (tb is complex or tb is float):
        return a + b
    return _arith(np.add, a, b)


def _mul(a, b, a_written=False, b_written=False):
    ta, tb = type(a), type(b)
    if (ta is complex or ta is float) and (tb is complex or tb is float):
        return a * b
    if ta is str or tb is str:
        s, n = (a, b) if ta is str else (b, a)
        if not _whole(n):
            raise CalcError("text * n repeats the text n times (a whole number); '%.1f' % a * b formats a and then multiplies the text: "
                            "write '%.1f' % (a * b)", "value")
        if len(s) * int(n) > MAX_TEXT:
            raise CalcError(f"a text of more than {MAX_TEXT} characters", "value")
        _charge(len(s) * max(0, int(n)) >> 3)
        return s * int(n)
    if a_written and (ta is list or ta is tuple) and _whole(b):          # [0] * n: a list written in place is repeated, as in Python
        _room(len(a) * max(0, int(b)))
        return a * int(b)
    if b_written and (tb is list or tb is tuple) and _whole(a):
        _room(len(b) * max(0, int(a)))
        return b * int(a)
    return _arith(np.multiply, a, b)                                     # x * 2 on a named list: every element doubled


def _mod(a, b):
    if isinstance(a, str):
        out = _percent(a, b)[:MAX_TEXT]
        _charge(len(out) >> 3)
        return out
    return _arith(np.mod, a, b)


def _pow(a, b):
    ta, tb = type(a), type(b)
    if ta is float and tb is float:
        return float(np.power(_F64(a), _F64(b)))
    if (ta is complex or ta is float) and (tb is complex or tb is float):
        try:
            return a ** b
        except (OverflowError, ZeroDivisionError):
            pass
    return _arith(np.power, a, b)


def _div(a, b):
    ta, tb = type(a), type(b)
    if (ta is complex or ta is float) and (tb is complex or tb is float) and b:
        return a / b
    return _arith(np.divide, a, b)


def _sub(a, b):
    ta, tb = type(a), type(b)
    if (ta is complex or ta is float) and (tb is complex or tb is float):
        return a - b
    return _arith(np.subtract, a, b)


def _floordiv(a, b):
    return _arith(np.floor_divide, a, b)


def _is_test(v) -> bool:
    return isinstance(v, (bool, np.bool_)) or (isinstance(v, np.ndarray) and v.dtype is _BOOL)


def _logic(fn, sign, a, b):
    """``&`` and ``|`` between tests: true / false, or the element-wise tests of arrays, as numpy has them."""
    if not (_is_test(a) and _is_test(b)):
        raise CalcError(f"{sign} joins two tests of arrays, (a > 1) {sign} (a < 3); between numbers write `and` / `or` (a power is **)",
                        "unsupported")
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        x, y = _pair(a, b)
        _charge(4 + max(x.size, y.size))
        return fn(x, y)
    return bool(fn(a, b))


_LOGIC = {ast.BitAnd: (np.logical_and, "&"), ast.BitOr: (np.logical_or, "|"), ast.BitXor: (np.logical_xor, "^")}
_CMP = {ast.Eq: np.equal, ast.NotEq: np.not_equal, ast.Lt: np.less, ast.LtE: np.less_equal, ast.Gt: np.greater, ast.GtE: np.greater_equal}
_PYCMP = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge}
_ORDER = {ast.Lt: lambda c: c < 0, ast.LtE: lambda c: c <= 0, ast.Gt: lambda c: c > 0, ast.GtE: lambda c: c >= 0}


def _compare(op, a, b):
    """One comparison of two values that are not both plain floats."""
    if op is ast.In or op is ast.NotIn:
        return right_has(b, a) is (op is ast.In)
    if op is ast.Is or op is ast.IsNot:
        same = a is b or (type(a) in _ATOMS and type(b) in _ATOMS and type(a) is type(b) and a == b)
        return same is (op is ast.Is)
    seq_a, seq_b = isinstance(a, (list, tuple)), isinstance(b, (list, tuple))
    if op is ast.Eq or op is ast.NotEq:
        if a is None or b is None or isinstance(a, (str, dict)) or isinstance(b, (str, dict)) or (seq_a and seq_b):
            return _same(a, b) is (op is ast.Eq)
    elif seq_a and seq_b:
        return _ORDER[op](_order(a, b))
    elif isinstance(a, str) and isinstance(b, str):
        return _PYCMP[op](a, b)
    elif a is None or b is None:
        raise CalcError("none has no order" + _NONE_HINT, "value")
    elif isinstance(a, (str, dict)) or isinstance(b, (str, dict)):
        raise CalcError(f"{_kind(a)} and {_kind(b)} cannot be ordered: compare two numbers or two texts", "value")
    x, y = _array(a), _array(b)
    if x.dtype.kind == "c" or y.dtype.kind == "c":
        if op is not ast.Eq and op is not ast.NotEq:
            raise CalcError("complex numbers have no order: compare abs(z)", "value")
    if x.ndim and y.ndim and x.shape != y.shape:
        _pair(x, y)
    out = _CMP[op](x, y)
    if out.ndim:
        _charge(out.size)
        return out
    return bool(out)


# ------------------------------------------------------------------------------------------------ the tree as closures
_RETURN, _BREAK, _CONTINUE = object(), object(), object()
_NO_DEFAULTS = {}                                                        # shared by every function without default values (never written)
_PRINT = object()                                                        # row name of a printed line
_EXCEPTIONS = ("Exception", "BaseException", "ValueError", "TypeError", "ZeroDivisionError", "IndexError", "KeyError", "ArithmeticError",
               "OverflowError", "RuntimeError", "NameError", "AssertionError", "FloatingPointError", "StopIteration", "LookupError",
               "AttributeError", "RecursionError", "NotImplementedError")


def _weight(node) -> int:
    """The syntax nodes of an expression that are evaluated when it runs once (the body of a lambda and the turns of a
    comprehension are charged when they run)."""
    n, stack = 0, [node]
    while stack:
        x = stack.pop()
        if isinstance(x, (ast.expr_context, ast.operator, ast.unaryop, ast.cmpop, ast.boolop)):
            continue
        n += 8 if isinstance(x, (ast.Dict, ast.DictComp)) else 1         # a dictionary is the largest small object
        if isinstance(x, ast.Lambda):
            continue
        if isinstance(x, (ast.ListComp, ast.GeneratorExp, ast.DictComp, ast.SetComp)):
            stack.append(x.generators[0].iter)
            continue
        stack.extend(ast.iter_child_nodes(x))
    return n


class _Frame:
    """The names of one call of a function (or of one comprehension), and the frame the function was defined in."""
    __slots__ = ("v", "up")

    def __init__(self, v, up):
        self.v, self.up = v, up


class _Function:
    """A function defined in the text. It holds closures over its own syntax tree and the frame it was defined in."""
    __slots__ = ("name", "params", "defaults", "vararg", "kwarg", "kwonly", "body", "is_lambda", "env", "cost")

    def __init__(self, name, params, defaults, vararg, kwarg, kwonly, body, is_lambda, env, cost):
        self.name, self.params, self.defaults, self.vararg, self.kwarg, self.kwonly = name, params, defaults, vararg, kwarg, kwonly
        self.body, self.is_lambda, self.env, self.cost = body, is_lambda, env, cost

    def __call__(self, *values, **keywords):                             # called by a function of the calculator (sorted, map, brentq ...)
        return self.invoke(_run(), [_out(v) for v in values], keywords)

    def invoke(self, run, values, keywords):
        run.steps += self.cost
        if run.steps > run.limit:
            run.stop(None)
        params = self.params
        if len(values) == len(params) and not keywords and not self.kwonly:
            local = dict(zip(params, values))
            if self.vararg:
                local[self.vararg] = ()
            if self.kwarg:
                local[self.kwarg] = {}
        else:
            local = self._bind(values, keywords or {})
        if run.depth >= MAX_DEPTH:
            hint = f"`from math import {self.name}` brings calc's {self.name} back" if self.name in FUNCTIONS else \
                "write the repetition as a loop"
            raise CalcError(f"functions nested more than {MAX_DEPTH} deep: {self.name} calls itself without an end "
                            f"(as in f = lambda x: f(x)); {hint}", "value")
        saved = run.frame
        run.frame = _Frame(local, self.env if self.env is not None else run.ground)
        run.depth += 1
        try:
            if self.is_lambda:
                return self.body(run)
            out = self.body(run)
            return out[1] if out is not None else None
        finally:
            run.frame = saved
            run.depth -= 1

    def _bind(self, values, keywords):
        params, name = self.params, self.name
        local = dict(zip(params, values))
        if len(values) > len(params):
            if not self.vararg:
                raise CalcError(f"{name} takes {len(params)} argument(s) ({', '.join(params)}) and {len(values)} are given", "value")
        if self.vararg:
            local[self.vararg] = tuple(values[len(params):])
        extra = {}
        for k, v in keywords.items():
            if k in local and k in params:
                raise CalcError(f"{name}: {k} is given twice", "value")
            if k in params or k in self.kwonly:
                local[k] = v
            elif self.kwarg:
                extra[k] = v
            else:
                raise CalcError(f"{name} has no argument {k!r}: it takes {', '.join(params + self.kwonly) or 'none'}", "value")
        if self.kwarg:
            local[self.kwarg] = extra
        for k in params + self.kwonly:
            if k not in local:
                if k not in self.defaults:
                    raise CalcError(f"{name} takes {len(params)} argument(s) ({', '.join(params)}) and {k} is not given", "value")
                local[k] = self.defaults[k]
        return local


def _explain(e) -> str:
    """A Python error inside a function of the calculator, in one line."""
    if isinstance(e, KeyError):
        return f"no key {str(e)[:60]}"
    text = re.sub(r"^(?:[\w.]*<locals>\.)?(?:<lambda>|\w+)\(\) ", "", str(e)[:140])
    return text or type(e).__name__


class _Compiler:
    """Turns the syntax tree into closures ``f(run)``. Expressions give their value; statements give ``None``, or a
    signal (``_BREAK``, ``_CONTINUE``, ``(_RETURN, value)``) that the enclosing loop or function reads."""

    def __init__(self):
        self.functions = 0                                               # depth of ``def`` / ``lambda`` being compiled
        self.loops = 0
        self.declared = [set()]                                          # names declared ``global`` in the function being compiled

    # ---------------------------------------------------------------- expressions
    def expr(self, node):
        """The closure of an expression. What calc does not have is refused when it is reached, not when it is read
        (``x if ok else <something calc lacks>`` runs as long as ``ok`` holds, as it would in Python)."""
        method = getattr(self, "e_" + type(node).__name__, None)
        try:
            if method is None:
                raise CalcError(_INSTEAD.get(type(node).__name__, f"no {type(node).__name__} in calc"), "unsupported")
            return method(node)
        except CalcError as e:
            return _refuse(e, getattr(node, "lineno", None))

    def e_Constant(self, node):
        v = node.value
        if isinstance(v, bool) or v is None:
            pass
        elif isinstance(v, (int, float)):
            try:
                v = float(v)
            except OverflowError:
                v = math.inf
        elif isinstance(v, str):
            v = v[:MAX_TEXT]
        elif not isinstance(v, complex):
            raise CalcError(_INSTEAD["Bytes" if isinstance(v, bytes) else "Ellipsis"], "unsupported", node.lineno)
        return lambda run: v

    def e_Name(self, node):
        name, line = node.id, node.lineno
        if name in FUNCTIONS:
            builtin = FUNCTIONS[name]
        elif name in CONSTANTS:
            builtin = CONSTANTS[name]
        elif name in MODULES:
            builtin = MODULES[name]
        else:
            builtin = _MISSING

        def load(run):
            frame = run.frame
            while frame is not None:
                v = frame.v
                if name in v:
                    return v[name]
                frame = frame.up
            if builtin is not _MISSING:
                return builtin
            if name == "print":
                return run.printer
            return run.unknown(name, line)
        return load

    def e_List(self, node):
        if any(isinstance(e, ast.Starred) for e in node.elts):
            parts = [(isinstance(e, ast.Starred), self.expr(e.value if isinstance(e, ast.Starred) else e)) for e in node.elts]

            def spread(run):
                out = []
                for starred, part in parts:
                    if starred:
                        out.extend(_iterable(part(run)))
                        _room(len(out))
                    else:
                        out.append(part(run))
                return out
            return spread
        parts = [self.expr(e) for e in node.elts]
        return lambda run: [p(run) for p in parts]

    def e_Tuple(self, node):
        if len(node.elts) == 2 and not any(isinstance(e, ast.Starred) for e in node.elts):
            a, b = self.expr(node.elts[0]), self.expr(node.elts[1])
            return lambda run: (a(run), b(run))
        items = self.e_List(node)
        return lambda run: tuple(items(run))

    def e_Dict(self, node):
        pairs = [(None if k is None else self.expr(k), self.expr(v)) for k, v in zip(node.keys, node.values)]

        def build(run):
            out = {}
            for k, v in pairs:
                if k is None:                                           # {**other}
                    other = v(run)
                    if not isinstance(other, dict):
                        raise CalcError("** takes a dictionary", "value")
                    out.update(other)
                else:
                    out[_key(k(run))] = v(run)
            return out
        return build

    def e_BinOp(self, node):
        left, right, op = self.expr(node.left), self.expr(node.right), type(node.op)
        if op is ast.Add:
            def add(run):
                a, b = left(run), right(run)
                if type(a) is float and type(b) is float:
                    return a + b
                return _add(a, b)
            return add
        if op is ast.Sub:
            def sub(run):
                a, b = left(run), right(run)
                if type(a) is float and type(b) is float:
                    return a - b
                return _sub(a, b)
            return sub
        if op is ast.Mult:
            written = (ast.List, ast.Tuple, ast.ListComp)               # calc has no integers: what repeats is told by how it is written
            a_written, b_written = isinstance(node.left, written), isinstance(node.right, written)

            def mul(run):
                a, b = left(run), right(run)
                if type(a) is float and type(b) is float:
                    return a * b
                return _mul(a, b, a_written, b_written)
            return mul
        if op is ast.Div:
            def div(run):
                a, b = left(run), right(run)
                if type(a) is float and type(b) is float and b:
                    return a / b
                return _div(a, b)
            return div
        if op is ast.Pow:
            def power(run):
                a, b = left(run), right(run)
                if type(a) is float and type(b) is float:
                    try:
                        v = a ** b
                        if type(v) is float:
                            return v
                    except (OverflowError, ZeroDivisionError):
                        pass
                return _pow(a, b)
            return power
        if op is ast.Mod:
            def mod(run):
                a, b = left(run), right(run)
                if type(a) is float and type(b) is float and b and b - b == 0.0 and a - a == 0.0:
                    return a % b
                return _mod(a, b)
            return mod
        if op is ast.FloorDiv:
            def floordiv(run):
                a, b = left(run), right(run)
                if type(a) is float and type(b) is float and b and b - b == 0.0 and a - a == 0.0:
                    return a // b
                return _floordiv(a, b)
            return floordiv
        if op is ast.MatMult:
            return lambda run: _dot(left(run), right(run))
        if op in _LOGIC:
            fn, sign = _LOGIC[op]
            return lambda run: _logic(fn, sign, left(run), right(run))
        raise CalcError("no bit shifts in calc: 2 ** n is a power of two", "unsupported", node.lineno)

    def e_UnaryOp(self, node):
        operand, op = self.expr(node.operand), type(node.op)
        if op is ast.Not:
            return lambda run: not _is_true(operand(run))
        if op is ast.USub:
            def minus(run):
                v = operand(run)
                if type(v) is float or type(v) is complex:
                    return -v
                if type(v) is np.ndarray and v.dtype is not _BOOL:
                    _charge(2 + v.size)
                    return -v
                if v is None or isinstance(v, str):
                    _need_number(v)
                out = np.negative(_array(v).astype(float) if isinstance(v, (bool, np.bool_)) or
                                  (isinstance(v, np.ndarray) and v.dtype.kind == "b") else _array(v))
                return out if out.ndim else out.item()
            return minus
        if op is ast.UAdd:
            return lambda run: _out(_array(operand(run)))

        def invert(run):
            v = operand(run)
            if _is_test(v):
                return not v if isinstance(v, (bool, np.bool_)) else np.logical_not(v)
            raise CalcError("~ negates a test of an array, ~(a > 1); `not x` negates one test, -x a number", "unsupported")
        return invert

    def e_BoolOp(self, node):
        parts, is_and = [self.expr(v) for v in node.values], isinstance(node.op, ast.And)

        def test(run):
            value = None
            for part in parts:
                value = part(run)
                if (value is True or (value is not False and _is_true(value))) != is_and:
                    return value
            return value
        return test

    def e_Compare(self, node):
        left = self.expr(node.left)
        ops = [(type(op), self.expr(right)) for op, right in zip(node.ops, node.comparators)]
        if len(ops) == 1 and ops[0][0] in _PYCMP:
            op, right = ops[0]
            py = _PYCMP[op]

            def compare(run):
                a, b = left(run), right(run)
                if type(a) is float and type(b) is float:
                    return py(a, b)
                return _compare(op, a, b)
            return compare

        def chain(run):
            a = left(run)
            result = True
            for op, right in ops:
                b = right(run)
                one = _compare(op, a, b)
                result = one if result is True else np.logical_and(result, one)
                if result is False:
                    return False
                a = b
            return bool(result) if isinstance(result, (np.bool_, bool)) or (isinstance(result, np.ndarray) and result.ndim == 0) else result
        return chain

    def e_IfExp(self, node):
        test, body, orelse = self.expr(node.test), self.expr(node.body), self.expr(node.orelse)
        return lambda run: body(run) if _is_true(test(run)) else orelse(run)

    def e_NamedExpr(self, node):
        value, store = self.expr(node.value), self.store(node.target)

        def walrus(run):
            v = value(run)
            store(run, v)
            return v
        return walrus

    def e_Subscript(self, node):
        value = self.expr(node.value)
        if not isinstance(node.slice, (ast.Slice, ast.Tuple)):
            index = self.expr(node.slice)

            def item(run):
                v, i = value(run), index(run)
                if type(i) is float:
                    tv = type(v)
                    if tv is list or tv is tuple:
                        try:
                            k = int(i)
                            if k == i:
                                return v[k]
                        except (IndexError, ValueError, OverflowError):
                            pass
                    elif tv is np.ndarray and v.ndim == 1:
                        try:
                            k = int(i)
                            if k == i:
                                return v.item(k)
                        except (IndexError, ValueError, OverflowError):
                            pass
                return _item(v, i)
            return item
        index = self.index(node.slice)

        def cut(run):
            v, i = value(run), index(run)
            if isinstance(v, dict):
                return _item(v, i)
            if not isinstance(v, (list, tuple, str, np.ndarray)):
                return _item(v, 0.0)
            try:
                out = _array(v)[i] if isinstance(i, tuple) and not isinstance(v, np.ndarray) else v[i]
            except (IndexError, TypeError):
                raise CalcError(f"index out of range: the list has {len(v)} element(s) (0 to {len(v) - 1})", "value") from None
            if isinstance(out, (np.ndarray, np.generic)):
                _charge(3)
                return out if out.ndim else out.item()
            _charge(2 * len(out))
            return out
        return cut

    def index(self, node):
        """A slice or a tuple of indices (``a[1:3]``, ``m[i, j]``, ``m[:, 0]``) as a closure that gives numpy's index."""
        if isinstance(node, ast.Slice):
            parts = [None if p is None else self.expr(p) for p in (node.lower, node.upper, node.step)]
            return lambda run: slice(*[None if p is None else _slice_int(p(run)) for p in parts])
        if isinstance(node, ast.Tuple):
            parts = [self.index(e) for e in node.elts]
            return lambda run: tuple(p(run) for p in parts)
        one = self.expr(node)

        def single(run):
            v = one(run)
            if isinstance(v, (list, np.ndarray)):
                v = _array(v)
                return v if v.dtype.kind == "b" else v.astype(int)
            return None if v is None else _int(v, "an index")
        return single

    def e_Attribute(self, node):
        value, attr = self.expr(node.value), node.attr
        return lambda run: _attribute(value(run), attr)

    def arguments(self, node):
        """The arguments of a call with ``*`` / ``**`` / keywords as one closure ``run -> (values, keywords or None)``."""
        args = [(isinstance(a, ast.Starred), self.expr(a.value if isinstance(a, ast.Starred) else a)) for a in node.args]
        keywords = [(k.arg, self.expr(k.value)) for k in node.keywords]

        def build(run):
            values = []
            for starred, a in args:
                if starred:
                    values.extend(_iterable(a(run)))
                    _room(len(values))
                else:
                    values.append(a(run))
            kw = {}
            for name, v in keywords:
                if name is None:
                    more = v(run)
                    if not isinstance(more, dict) or not all(isinstance(k, str) for k in more):
                        raise CalcError("** takes a dictionary with names as keys", "value")
                    kw.update(more)
                else:
                    kw[name] = v(run)
            return values, kw or None
        return build

    def e_Call(self, node):
        line, func = node.lineno, node.func
        if isinstance(func, ast.Name) and func.id in _EXCEPTIONS:        # ValueError("...") is its message
            arguments = self.arguments(node)
            return lambda run: _raised(func.id, arguments(run)[0])
        method = func.attr if isinstance(func, ast.Attribute) else None
        target = self.expr(func.value) if method else self.expr(func)
        shown = ast.unparse(func)[:40]

        def finish(run, fn, values, keywords):
            """A function of the calculator (not one of the text): call it, and charge what it read and made."""
            if not callable(fn):
                raise CalcError(f"{shown} is {_kind(fn)}, not a function: it cannot be called" +
                                (f" (a name of the text hides calc's {shown}: rename it)" if shown in FUNCTIONS else
                                 f" ({shown} * (...) multiplies)" if isinstance(fn, _NUMBER) else ""), "value")
            try:
                out = fn(*values) if keywords is None else fn(*values, **keywords)
            except (CalcError, RecursionError):
                raise
            except _PY_ERRORS as e:
                raise CalcError(f"{shown}: {_explain(e)}", "value", line) from None
            if type(out) is float or out is None:
                return out
            out = _out(out)
            used = _cells(out)
            if type(out) is list:
                used += len(out)
            for v in values:
                if type(v) is not float:
                    used += _cells(v)
            if used:
                run.steps += used
                if run.steps > run.limit:
                    run.stop(line)
            return out

        plain = not node.keywords and not any(isinstance(a, ast.Starred) for a in node.args)
        if plain and len(node.args) == 1:                                # f(x): the call a loop makes most often
            arg = self.expr(node.args[0])
            is_append = method == "append"

            def call_one(run):
                if method is None:
                    fn = target(run)
                else:
                    owner = target(run)
                    if type(owner) is _Module:
                        fn = owner.members.get(method)
                        if fn is None:
                            _no_attribute(owner, method)
                    elif is_append and type(owner) is list:
                        v = arg(run)
                        if len(owner) >= MAX_ITEMS:
                            _grow(owner)
                        owner.append(v)
                        return None
                    else:
                        fn = _attribute(owner, method)
                v = arg(run)
                if type(fn) is _Function:
                    return fn.invoke(run, [v], None)
                if type(v) is float:
                    try:
                        out = fn(v)
                    except (CalcError, RecursionError):
                        raise
                    except _PY_ERRORS as e:
                        if not callable(fn):
                            return finish(run, fn, (v,), None)
                        raise CalcError(f"{shown}: {_explain(e)}", "value", line) from None
                    if type(out) is float:
                        return out
                    return finish(run, lambda x: out, (v,), None)
                return finish(run, fn, (v,), None)
            return call_one

        if plain:
            args = [self.expr(a) for a in node.args]

            def call_plain(run):
                if method is None:
                    fn = target(run)
                else:
                    owner = target(run)
                    if type(owner) is _Module:
                        fn = owner.members.get(method)
                        if fn is None:
                            _no_attribute(owner, method)
                    else:
                        fn = _attribute(owner, method)
                values = [a(run) for a in args]
                if type(fn) is _Function:
                    return fn.invoke(run, values, None)
                return finish(run, fn, values, None)
            return call_plain

        arguments = self.arguments(node)

        def call(run):
            fn = target(run) if method is None else _attribute(target(run), method)
            values, keywords = arguments(run)
            if type(fn) is _Function:
                return fn.invoke(run, values, keywords)
            return finish(run, fn, values, keywords)
        return call

    def function(self, node, name, is_lambda):
        a = node.args
        if a.posonlyargs or (not is_lambda and node.decorator_list):
            raise CalcError("a function takes plain arguments: def f(x, y=1): ...", "unsupported", node.lineno)
        if is_lambda and (a.vararg or a.kwarg):
            raise CalcError("a lambda takes plain arguments: lambda x, y: ... (def f(*values): ... takes any number)", "unsupported", node.lineno)
        params, kwonly = [x.arg for x in a.args], [x.arg for x in a.kwonlyargs]
        defaults = [self.expr(d) for d in a.defaults]
        later = [(x.arg, self.expr(d)) for x, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None]
        self.functions += 1
        self.declared.append(set())
        loops, self.loops = self.loops, 0
        try:
            body = self.expr(node.body) if is_lambda else self.block(node.body, False)
        finally:
            self.functions -= 1
            self.declared.pop()
            self.loops = loops
        cost = 1 + (_weight(node.body) if is_lambda else 0)
        vararg, kwarg = a.vararg.arg if a.vararg else None, a.kwarg.arg if a.kwarg else None

        def define(run):
            run.steps += 12                                              # a function is an object: making one is not free
            values = [d(run) for d in defaults]
            env = None if run.frame is run.ground else run.frame
            given = dict(zip(params[len(params) - len(values):], values)) if values or later else _NO_DEFAULTS
            given.update((k, d(run)) for k, d in later)
            return _Function(name, params, given, vararg, kwarg, kwonly, body, is_lambda, env, cost)
        return define

    def e_Lambda(self, node):
        return self.function(node, "the function", True)

    def comprehension(self, node, element, cost):
        gens = [(self.store(g.target), self.expr(g.iter), [self.expr(c) for c in g.ifs], 1 + sum(_weight(c) for c in g.ifs) +
                 (_weight(node.generators[k + 1].iter) if k + 1 < len(node.generators) else cost))
                for k, g in enumerate(node.generators)]
        if len(gens) == 1 and not gens[0][2]:                            # [f(x) for x in xs]: the turns are charged at once
            store, source, _, per = gens[0]
            target = node.generators[0].target
            name = target.id if isinstance(target, ast.Name) else None

            def simple(run):
                items = _loop_over(source(run))
                run.steps += per * len(items)
                if run.steps > run.limit:
                    run.stop(None)
                saved = run.frame
                names = {}
                run.frame = _Frame(names, saved)
                try:
                    if name is not None:
                        out = []
                        for item in items:
                            names[name] = item
                            out.append(element(run))
                        return out
                    out = []
                    for item in items:
                        store(run, item)
                        out.append(element(run))
                    return out
                finally:
                    run.frame = saved
            return simple

        def general(run):
            out = []

            def loop(k):
                store, source, tests, per = gens[k]
                last = k + 1 == len(gens)
                items = _loop_over(source(run))
                run.steps += per * len(items)
                if run.steps > run.limit:
                    run.stop(None)
                for item in items:
                    store(run, item)
                    for test in tests:
                        if not _is_true(test(run)):
                            break
                    else:
                        if last:
                            out.append(element(run))
                            if len(out) > MAX_ITEMS:
                                raise CalcError(f"a list of more than {MAX_ITEMS} values: use a coarser grid, or keep only the best "
                                                "value in a loop", "value")
                        else:
                            loop(k + 1)
            saved = run.frame
            run.frame = _Frame({}, saved)
            try:
                loop(0)
            finally:
                run.frame = saved
            return out
        return general

    def e_ListComp(self, node):
        return self.comprehension(node, self.expr(node.elt), _weight(node.elt))

    e_GeneratorExp = e_ListComp

    def e_DictComp(self, node):
        key, value = self.expr(node.key), self.expr(node.value)
        pairs = self.comprehension(node, lambda run: (_key(key(run)), value(run)), _weight(node.key) + _weight(node.value))
        return lambda run: dict(pairs(run))

    def e_JoinedStr(self, node):
        parts = [self.expr(v) if isinstance(v, ast.Constant) else self.e_FormattedValue(v) for v in node.values]

        def joined(run):
            out = "".join([p(run) for p in parts])[:MAX_TEXT]
            if len(out) > 64:
                _charge(len(out) >> 3)
            return out
        return joined

    def e_FormattedValue(self, node):
        value = self.expr(node.value)
        spec = self.expr(node.format_spec) if node.format_spec is not None else None
        plain = node.conversion in (ord("r"), ord("s"), ord("a"))

        def formatted(run):
            v = value(run)
            if spec is None:
                return _text(v)
            return _format(_text(v) if plain else v, spec(run))[:MAX_TEXT]
        return formatted

    def e_Starred(self, node):
        raise CalcError("* unpacks inside a call or a list: f(*values), [*a, *b], first, *rest = values", "unsupported", node.lineno)

    # ---------------------------------------------------------------- assignment
    def store(self, target):
        """A closure ``(run, value)`` that assigns to a name, to names (``a, b`` / ``a, *b``), or to an element."""
        if isinstance(target, ast.Name):
            name = target.id
            if name in self.declared[-1]:
                def store_global(run, value):
                    run.ground.v[name] = value
                return store_global

            def store_name(run, value):
                run.frame.v[name] = value
            return store_name
        if isinstance(target, (ast.Tuple, ast.List)):
            star = [k for k, e in enumerate(target.elts) if isinstance(e, ast.Starred)]
            if len(star) > 1:
                raise CalcError("one * on the left at most: first, *rest = values", "unsupported", target.lineno)
            stores = [self.store(e.value if isinstance(e, ast.Starred) else e) for e in target.elts]
            n, line = len(stores), target.lineno
            plain = [e.id for e in target.elts if isinstance(e, ast.Name) and e.id not in self.declared[-1]]

            def unpack(run, value):
                if (type(value) is tuple or type(value) is list) and len(value) == n == len(plain):
                    names = run.frame.v
                    for name, item in zip(plain, value):
                        names[name] = item
                    return
                if isinstance(value, _NUMBER) or value is None or callable(value):
                    raise CalcError(f"{n} names on the left and one value on the right ({_kind(value)})" + (_NONE_HINT if value is None else ""),
                                    "value", line)
                items = _iterable(value)
                if star:
                    if len(items) < n - 1:
                        raise CalcError(f"{n - 1} names (and a *) on the left, {len(items)} value(s) on the right", "value", line)
                    k, tail = star[0], n - 1 - star[0]
                    items = list(items[:k]) + [list(items[k:len(items) - tail])] + list(items[len(items) - tail:])
                elif len(items) != n:
                    raise CalcError(f"{n} names on the left, {len(items)} value(s) on the right", "value", line)
                for one, item in zip(stores, items):
                    one(run, item)
            return unpack
        if isinstance(target, ast.Subscript):
            holder = self.expr(target.value)
            sliced = isinstance(target.slice, (ast.Slice, ast.Tuple))
            index = self.index(target.slice) if sliced else self.expr(target.slice)

            def store_item(run, value):
                h, i = holder(run), index(run)
                if isinstance(h, dict):
                    k = _key(i)
                    if k not in h and len(h) >= MAX_ITEMS:
                        raise CalcError(f"a dictionary of more than {MAX_ITEMS} entries", "value")
                    h[k] = value
                elif isinstance(h, list) and not isinstance(i, tuple):
                    try:
                        if sliced:
                            items = _list(value)
                            _grow(h, len(items))
                            h[i] = items
                        else:
                            h[_int(i, "an index")] = value
                    except IndexError:
                        raise CalcError(f"index {_text(i)} is out of range: the list has {len(h)} element(s); x.append(v) adds one", "value") from None
                elif isinstance(h, np.ndarray):
                    try:
                        if not sliced:
                            i = _array(i) if isinstance(i, (list, np.ndarray)) else _int(i, "an index")
                            i = (i if i.dtype.kind == "b" else i.astype(int)) if isinstance(i, np.ndarray) else i
                        v = _array(value)
                        if v.dtype.kind == "c" and h.dtype.kind != "c":
                            raise CalcError("a complex number does not go into an array of real numbers: make it with zeros(n) * 1j", "value")
                        h[i] = v
                    except (IndexError, ValueError):
                        raise CalcError(f"index out of range or wrong length: the array has the shape {_shape(h)}", "value") from None
                elif isinstance(h, tuple):
                    raise CalcError("a tuple cannot be changed: make it a list first (x = list(x))", "value")
                else:
                    raise CalcError(f"only an element of a list, an array or a dictionary can be assigned, not of {_kind(h)}" +
                                    (_NONE_HINT if h is None else ""), "value")
            return store_item
        if isinstance(target, ast.Attribute):
            raise CalcError("no attributes in calc: assign to a name, or to a key of a dictionary (d['x'] = 1)", "unsupported", target.lineno)
        raise CalcError("assign to a name: name = value", "unsupported", getattr(target, "lineno", None))

    # ---------------------------------------------------------------- statements
    def block(self, body, top):
        statements = [self.statement(st, top) for st in body]
        if len(statements) == 1:
            return statements[0]

        def run_block(run):
            for statement in statements:
                signal = statement(run)
                if signal is not None:
                    return signal
            return None
        return run_block

    def statement(self, st, top):
        method = getattr(self, "s_" + type(st).__name__, None)
        line = st.lineno
        try:
            if method is None:
                raise CalcError(_INSTEAD.get(type(st).__name__, f"no {type(st).__name__} statement in calc"), "unsupported")
            inner, cost = method(st, top)
        except CalcError as e:
            inner, cost = _refuse(e, line), 1

        def run_statement(run):
            run.steps += cost
            if run.steps > run.limit:
                run.stop(line)
            try:
                return inner(run)
            except CalcError as e:
                if e.line is None:
                    e.line = line
                raise
            except _PY_ERRORS as e:
                raise CalcError(_explain(e), "value", line) from None
        return run_statement

    def s_Assign(self, st, top):
        named = isinstance(st.value, ast.Lambda) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name)
        try:                                                             # f = lambda x: ...: the function is called f in what calc says
            value = self.function(st.value, st.targets[0].id, True) if named else self.expr(st.value)
        except CalcError as e:
            value = _refuse(e, st.lineno)
        if len(st.targets) == 1 and isinstance(st.targets[0], ast.Name) and st.targets[0].id not in self.declared[-1]:
            name = st.targets[0].id

            def assign_name(run):
                run.frame.v[name] = value(run)
            return assign_name, 1 + _weight(st.value)
        stores = [self.store(t) for t in st.targets]

        def assign(run):
            v = value(run)
            for store in stores:
                store(run, v)
        return assign, 1 + _weight(st.value) + sum(_weight(t) for t in st.targets)

    def s_AnnAssign(self, st, top):
        if st.value is None:
            return (lambda run: None), 1
        value, store = self.expr(st.value), self.store(st.target)
        return (lambda run: store(run, value(run))), 1 + _weight(st.value)

    def s_AugAssign(self, st, top):
        if not isinstance(st.target, (ast.Name, ast.Subscript)):
            raise CalcError("assign to a name: name += value", "unsupported", st.lineno)
        load = ast.copy_location(ast.Name(id=st.target.id, ctx=ast.Load()), st) if isinstance(st.target, ast.Name) else \
            ast.copy_location(ast.Subscript(value=st.target.value, slice=st.target.slice, ctx=ast.Load()), st)
        value = self.expr(ast.copy_location(ast.BinOp(left=load, op=st.op, right=st.value), st))
        store = self.store(st.target)
        cost = 2 + _weight(st.target) + _weight(st.value)
        if not isinstance(st.op, ast.Add):
            return (lambda run: store(run, value(run))), cost
        current, more = self.expr(load), self.expr(st.value)

        def in_place(run):
            held = current(run)
            if type(held) is not list:
                return store(run, value(run))
            other = more(run)                                           # x += [v] changes the list itself, as in Python
            if isinstance(other, (list, tuple)):
                _grow(held, len(other))
                _charge(len(other))
                held.extend(other)
            else:
                held = _add(held, other)
            store(run, held)
        return in_place, cost

    def s_Expr(self, st, top):
        value = self.expr(st.value)
        if not top:
            return value_only(value), 1 + _weight(st.value)
        name = st.value.id if isinstance(st.value, ast.Name) else None   # a bare name is shown with its name
        is_call = isinstance(st.value, ast.Call)

        def show(run):
            v = value(run)
            if not (is_call and v is None) and not callable(v) and not isinstance(v, _Module):
                run.row(name, v)
        return show, 1 + _weight(st.value)

    def s_Import(self, st, top):
        binds = []
        for a in st.names:
            head = a.name.split(".")[0]
            if a.name in _INERT:
                continue
            if head not in MODULES or (("." in a.name) and a.name not in _SUBMODULES):
                raise CalcError(_no_module(a.name), "unsupported", st.lineno)
            if a.asname:                                                 # import numpy as np: np stands for the table
                binds.append((a.asname, _SUBMODULES.get(a.name) or MODULES[head]))

        def bind(run):
            for name, module in binds:
                run.frame.v[name] = module
        return bind, 1

    def s_ImportFrom(self, st, top):
        module = st.module or ""
        if module in _INERT:
            return (lambda run: None), 1
        table = _SUBMODULES.get(module) or (MODULES.get(module) if "." not in module else None)
        if table is None or st.level:
            raise CalcError(_no_module(module), "unsupported", st.lineno)
        binds = []
        for a in st.names:
            if a.name == "*":
                if module not in ("math", "numpy"):                      # from math import *: the names are there already
                    binds.extend((k, v) for k, v in table.members.items())
                continue
            if a.name not in table.members:
                _no_attribute(table, a.name)
            binds.append((a.asname or a.name, table.members[a.name]))    # also `from math import exp`: it brings calc's exp back

        def bind(run):
            for name, value in binds:
                run.frame.v[name] = value
        return bind, 1

    def s_FunctionDef(self, st, top):
        define, name = self.function(st, st.name, False), st.name

        def bind(run):
            run.frame.v[name] = define(run)
        return bind, 1

    def s_Return(self, st, top):
        if not self.functions:
            raise CalcError("return outside a function: end the text with the value (or print it)", "unsupported", st.lineno)
        if st.value is None:
            return (lambda run: (_RETURN, None)), 1
        value = self.expr(st.value)
        return (lambda run: (_RETURN, value(run))), 1 + _weight(st.value)

    def s_If(self, st, top):
        test, body = self.expr(st.test), self.block(st.body, top)
        orelse = self.block(st.orelse, top) if st.orelse else None

        def branch(run):
            v = test(run)
            if v is True or (v is not False and _is_true(v)):
                return body(run)
            return orelse(run) if orelse is not None else None
        return branch, 1 + _weight(st.test)

    def s_For(self, st, top):
        source, store = self.expr(st.iter), self.store(st.target)
        self.loops += 1
        try:
            body = self.block(st.body, top and "expr")
        finally:
            self.loops -= 1
        orelse = self.block(st.orelse, top and "expr") if st.orelse else None
        name = st.target.id if isinstance(st.target, ast.Name) and st.target.id not in self.declared[-1] else None

        line = st.lineno

        def loop(run):
            names = run.frame.v
            items = _loop_over(source(run))
            turn = 0
            try:
                for item in items:
                    turn += 1
                    if name is not None:
                        names[name] = item
                    else:
                        store(run, item)
                    signal = body(run)
                    if signal is not None:
                        if signal is _BREAK:
                            break
                        if signal is not _CONTINUE:
                            return signal
                else:
                    return orelse(run) if orelse is not None else None
            except _Stop as e:                                           # the step limit: say how far this loop was
                e.loops.append((line, turn, len(items)))
                raise
            return None
        return loop, 1 + _weight(st.iter)

    def s_While(self, st, top):
        test = self.expr(st.test)
        self.loops += 1
        try:
            body = self.block(st.body, top and "expr")
        finally:
            self.loops -= 1
        orelse = self.block(st.orelse, top and "expr") if st.orelse else None
        cost, line = 1 + _weight(st.test), st.lineno

        def loop(run):
            while True:
                run.steps += cost
                if run.steps > run.limit:
                    run.stop(line)
                v = test(run)
                if not (v is True or (v is not False and _is_true(v))):
                    return orelse(run) if orelse is not None else None
                signal = body(run)
                if signal is not None:
                    if signal is _BREAK:
                        return None
                    if signal is not _CONTINUE:
                        return signal
        return loop, 1

    def s_Break(self, st, top):
        if not self.loops:
            raise CalcError("break outside a loop", "unsupported", st.lineno)
        return (lambda run: _BREAK), 1

    def s_Continue(self, st, top):
        if not self.loops:
            raise CalcError("continue outside a loop", "unsupported", st.lineno)
        return (lambda run: _CONTINUE), 1

    def s_Pass(self, st, top):
        return (lambda run: None), 1

    def s_Global(self, st, top):
        if self.functions:
            self.declared[-1].update(st.names)
        return (lambda run: None), 1

    def s_Try(self, st, top):
        body = self.block(st.body, top)
        handlers = [(_handles(h.type), h.name, self.block(h.body, top)) for h in st.handlers]
        orelse = self.block(st.orelse, top) if st.orelse else None
        final = self.block(st.finalbody, top) if st.finalbody else None

        def attempt(run):
            frame, depth = run.frame, run.depth
            try:
                try:
                    signal = body(run)
                    if signal is None and orelse is not None:
                        signal = orelse(run)
                except _Stop:
                    raise
                except (CalcError,) + _PY_ERRORS as e:                   # what the calculator refuses, as the error Python would raise
                    run.frame, run.depth = frame, depth
                    run.steps += 6
                    raised = _raised_as(e)
                    handler = next((h for h in handlers if h[0] is None or raised is None or raised in h[0]), None)
                    if handler is None:
                        raise
                    if handler[1]:
                        text = str(e)[:200]
                        run.frame.v[handler[1]] = text[len(raised) + 2:] if raised and text.startswith(raised + ": ") else text
                    saved, run.caught = run.caught, e
                    try:
                        signal = handler[2](run)
                    finally:
                        run.caught = saved
            except _Stop:
                raise
            except (CalcError,) + _PY_ERRORS:
                if final is not None:
                    final(run)
                raise
            if final is not None:
                last = final(run)
                return last if last is not None else signal
            return signal
        return attempt, 1

    def s_Raise(self, st, top):
        value = self.expr(st.exc) if st.exc is not None else None

        def throw(run):
            if value is None:
                if run.caught is not None:
                    raise run.caught
                raise CalcError("raise without an error to raise again", "value")
            v = value(run)
            raise CalcError(_text(v)[:200] if not isinstance(v, CalcError) else str(v), "value")
        return throw, 1 + (_weight(st.exc) if st.exc is not None else 0)

    def s_Assert(self, st, top):
        test = self.expr(st.test)
        message = self.expr(st.msg) if st.msg is not None else None
        shown = ast.unparse(st.test)[:80]

        def check(run):
            if not _is_true(test(run)):
                raise CalcError(f"assert failed: {_text(message(run))[:160] if message is not None else shown}", "value")
        return check, 1 + _weight(st.test)


def _refuse(error, line):
    """A closure that raises ``error`` (a fresh copy each time) when it runs."""
    message, kind, line = str(error), error.kind, error.line or line

    def refuse(run, *value):
        raise CalcError(message, kind, line)
    return refuse


def value_only(value):
    def discard(run):
        value(run)
    return discard


def _handles(node):
    """The error names of an ``except`` clause, or None when it catches everything."""
    if node is None:
        return None
    names = {e.id for e in (node.elts if isinstance(node, ast.Tuple) else [node]) if isinstance(e, ast.Name)}
    if not names or names & {"Exception", "BaseException"} or not names <= set(_EXCEPTIONS):
        return None
    if names & {"ArithmeticError"}:
        names |= {"ZeroDivisionError", "OverflowError", "FloatingPointError"}
    if names & {"LookupError"}:
        names |= {"IndexError", "KeyError"}
    return names


def _raised_as(e):
    """The Python error a refusal stands for, where that is clear (None: any ``except`` clause catches it)."""
    if not isinstance(e, CalcError):
        return type(e).__name__
    text = str(e)
    head = text.split(":", 1)[0]
    if head in _EXCEPTIONS:
        return head
    if e.kind == "unknown":
        return "NameError"
    if text.startswith("index ") or text.startswith("pop"):
        return "IndexError"
    if text.startswith("no key"):
        return "KeyError"
    return "AssertionError" if text.startswith("assert failed") else None


def _raised(name, values):
    """``ValueError("...")`` in the text: the message (``raise`` makes a refusal of it, ``except ... as e`` reads it)."""
    return f"{name}: {' '.join(_text(v) for v in values)}"[:200] if values else name


def _slice_int(v):
    if isinstance(v, np.ndarray) and v.ndim == 0:
        v = v.item()
    if v is None:
        return None
    if not _whole(v):
        raise CalcError(f"a slice takes whole numbers, not {_brief(v)}: int(x) cuts a fraction", "value")
    return int(v)


def _no_module(name) -> str:
    head = name.split(".")[0]
    if head in _NO_MODULE and name not in _SUBMODULES:
        return _NO_MODULE[head]
    if head in MODULES:
        return f"no {name} in calc: {head} has {' '.join(sorted(k for k, v in MODULES[head].members.items() if isinstance(v, _Module))) or 'no parts'}"
    return f"no {head} in calc: it has math, numpy, cmath, statistics, itertools, bisect, copy and scipy.optimize (their names work without import)"


class _Run:
    """One evaluation: the names, the printed rows and the step count."""
    __slots__ = ("ground", "frame", "rows", "steps", "limit", "depth", "printer", "caught")

    def __init__(self, names: dict | None = None, limit: int | None = None):
        self.ground = self.frame = _Frame({} if names is None else names, None)
        self.rows, self.steps, self.depth, self.caught = [], 0, 0, None
        self.limit = MAX_STEPS if limit is None else limit
        self.printer = self.print                                        # one object, so that a call can be recognised

    def stop(self, line):
        raise _Stop(f"stopped after {self.limit} steps", "value", line)

    def unknown(self, name, line):
        dropped = self.ground.v.get(_DROPPED) or ()
        if name in dropped:
            raise CalcError(f"{name} was not kept: an earlier call left more than {MAX_KEPT} numbers in its names; compute it again, "
                            "and keep the result, not the whole grid", "value", line)
        if name in _NOT_HERE:
            raise CalcError(_NOT_HERE[name], "unsupported", line)
        if name in _NO_MODULE:
            raise CalcError(_NO_MODULE[name], "unsupported", line)
        raise CalcError(f"unknown name {name!r}", "unknown", line)

    def print(self, *values, sep=" ", **_):
        self.row(_PRINT, (sep if isinstance(sep, str) else " ").join(_text(v) for v in values)[:MAX_TEXT])

    def row(self, name, value):
        """One line of the answer: a bare expression or a print."""
        if len(self.rows) >= MAX_ROWS:
            if len(self.rows) == MAX_ROWS:
                self.rows.append((_PRINT, f"... more than {MAX_ROWS} values: print fewer"))
            return
        self.rows.append((name, value if name is _PRINT else _num(value)))   # the value as it is now: a list may change later


_IDLE = _Run(limit=1 << 62)                                              # a function of the calculator called outside a text


# ------------------------------------------------------------------------------------------------ reading the text
_PRINT_STATEMENT = re.compile(r"^(\s*)print\s+(?![=(])(.+?)\s*$")
_MATH_FUNCTION = re.compile(r"^(\s*)(?:def\s+)?([A-Za-z_]\w*)\(\s*([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s*\)\s*=(?!=)\s*(.+?)\s*$")
_CARET = re.compile(r"\^")


def _repair_line(line: str) -> str:
    """One line with the spellings that have one reading repaired (see the module text)."""
    if line.lstrip().startswith("//"):
        return line.replace("//", "#", 1)
    return line


def _split_code(line: str) -> tuple[str, str]:
    """(code, comment) of a line: the comment starts at the first ``#`` outside a quoted text."""
    quote = None
    for i, ch in enumerate(line):
        if quote:
            if ch == quote and line[i - 1] != "\\":
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "#":
            return line[:i], line[i:]
    return line, ""


def _statements(code: str) -> list:
    """The pieces of a line between its semicolons (outside quoted text and brackets)."""
    out, depth, quote, start = [], 0, None, 0
    for i, ch in enumerate(code):
        if quote:
            quote = None if ch == quote and code[i - 1] != "\\" else quote
        elif ch in "\"'":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == ";" and depth <= 0:
            out.append(code[start:i])
            start = i + 1
    return out + [code[start:]]


def _prepare(text: str) -> str:
    text = "\n" * text[:len(text) - len(text.lstrip())].count("\n") + text.strip()     # the lines keep their numbers
    if "\\n" in text and "\n" not in text.strip():
        text = text.replace("\\n", "\n")                              # backslash-n written where line breaks were meant
    lines = []
    for raw in text.split("\n"):
        code, comment = _split_code(_repair_line(raw))
        lines.append(_CARET.sub("**", code) + comment)
    return "\n".join(lines)


def _parse(text: str):
    """The syntax tree of the text. When it does not parse, the line the error points at is repaired if it carries one
    of the spellings of the module text, and the text is parsed again; the first error is raised when none applies."""
    src = _prepare(text)
    first = None
    for _ in range(60):
        try:
            return ast.parse(src, mode="exec")
        except SyntaxError as e:
            first = first or e
            lines = src.split("\n")
            n = (e.lineno or 1) - 1
            if not 0 <= n < len(lines):
                break
            line = lines[n]
            code, comment = _split_code(line)
            fixed = None
            m = _PRINT_STATEMENT.match(code)
            if "unterminated string" in (e.msg or ""):
                quote = next((q for q in "\"'" if src.startswith(q)), None)
                if quote and n == 0:                                    # the whole text in quotation marks
                    src = src[1:-1] if src.endswith(quote) else src[1:]
                    continue
                if src[-1:] in "\"'" and src.count(src[-1]) % 2 == 1:    # one stray quotation mark at the end
                    src = src[:-1]
                    continue
            elif any(_MATH_FUNCTION.match(part) for part in _statements(code)):          # f(x, y) = expression, as mathematics writes it
                parts = [_MATH_FUNCTION.sub(r"\1\2 = lambda \3: (\4)", part) for part in _statements(code)]
                fixed = ";".join(parts) + comment
            elif m or "Missing parentheses in call to 'print'" in (e.msg or ""):
                parts = [_PRINT_STATEMENT.sub(r"\1print(\2)", part) for part in _statements(code)]     # print a, b
                fixed = ";".join(parts) + comment
            elif "//" in code and not comment:
                fixed = code[:code.index("//")].rstrip() + "  #" + code[code.index("//") + 2:]   # a trailing // comment
            elif isinstance(e, IndentationError) and "unexpected indent" in (e.msg or "") and line[:1] in " \t":
                fixed = line.lstrip()
            if fixed is None or fixed == line:
                break
            lines[n] = fixed
            src = "\n".join(lines)
    raise first


def _size(value, seen=None, left=None) -> int:
    """Numbers a value holds (what ``MAX_KEPT`` counts). A list that holds itself is counted once; the count stops
    one past the bound."""
    left = [MAX_KEPT + 1] if left is None else left
    seen = set() if seen is None else seen
    if isinstance(value, np.ndarray):
        left[0] -= value.size
        return int(value.size)
    if isinstance(value, str):
        left[0] -= 1 + len(value) // 8
        return 1 + len(value) // 8
    if isinstance(value, _Function):
        items, frame = [d for d in value.defaults.values()], value.env
        while frame is not None and frame.up is not None:              # the frames a closure keeps alive (not the ground)
            items.append(frame.v)
            frame = frame.up
    elif isinstance(value, dict):
        items = list(value.values())
    elif isinstance(value, (list, tuple)):
        items = value
    else:
        left[0] -= 1
        return 1
    if id(value) in seen:
        return 0
    seen.add(id(value))
    total = 1
    for x in items:
        if left[0] < 0:
            break
        total += _size(x, seen, left)
    left[0] -= 1
    return total


def _trim(names: dict) -> None:
    """Keep at most ``MAX_KEPT`` numbers between calls: the names assigned longest ago are dropped first (and
    remembered, so that a later text is told why the name is gone)."""
    dropped = names.pop(_DROPPED, ())
    sizes = {}
    for key, value in names.items():
        try:
            sizes[key] = _size(value)
        except RecursionError:
            sizes[key] = MAX_KEPT + 1
    total = sum(sizes.values())
    gone = []
    for key in [k for k in names if sizes[k] > MAX_KEPT] + list(names):   # first what can never be kept, then the oldest
        if total <= MAX_KEPT:
            break
        if key in names:
            total -= sizes[key]
            del names[key]
            gone.append(key)
    dropped = tuple(k for k in tuple(dropped) + tuple(gone) if k not in names)[-64:]
    if dropped:
        names[_DROPPED] = dropped


def _stopped(e: _Stop) -> _Stop:
    """The refusal of a call that reached the step limit: how far its outer loops were, and what to do instead (in
    the 200 characters the environment shows)."""
    head = str(e)
    if e.loops:                                                         # the two outermost loops say what share of the work was done
        where = [f"of line {line} (turn {turn} of {turns})" for line, turn, turns in reversed(e.loops[-2:])]
        head += " in the loop " + " and ".join(where)
        advice = ("use a coarser grid, then a fine one around the best", ", or minimize(f, x0)", "; names stay")
    else:
        advice = ("a loop that does not end, or too much work for one call", ": use a coarser grid, then a fine one around the best",
                  "; names stay between calls")
    text = head + ": " + advice[0]
    for part in advice[1:]:
        if len(text) + len(part) <= 200:
            text += part
    out = _Stop(text[:200], "value", e.line)
    out.loops = e.loops
    return out


def calc(text: str, names: dict | None = None) -> list[tuple[str | None, object]]:
    """Evaluate the statements. Returns ``[(name or None, value)]``: one entry per bare expression at the top level
    (with its name when the expression is one name) and one per ``print`` (its name is None and its value the printed
    text). A text that asks to see nothing gets the names it assigned, the last ``MAX_ECHO`` of them.

    ``names``: a dictionary that holds the assigned names from one call to the next (the environment keeps one per
    episode, so that a value computed in one call can be used in a later one). It is changed in place.

    Raises ``SyntaxError`` for text that is not Python, ``CalcError`` (a ``ValueError``) for what the calculator does
    not have, an unknown name, or an operation without a result.
    """
    if not isinstance(text, str) or len(text) > MAX_CALC_CHARS:
        raise CalcError(f"expressions must be a string of at most {MAX_CALC_CHARS} characters", "value")
    try:
        tree = _parse(text)
        if len(tree.body) > MAX_CALC_STATEMENTS:
            raise CalcError(f"more than {MAX_CALC_STATEMENTS} statements: split the text over calls (names stay)", "value")
        program = _Compiler().block(tree.body, True) if tree.body else None
    except (RecursionError, MemoryError):
        raise CalcError("an expression nested too deeply: split it over several lines", "value") from None
    if program is None:
        return []
    run = _Run(names)
    ground = run.ground.v
    had = dict(ground)
    before = getattr(_CURRENT, "run", None)
    _CURRENT.run = run
    with np.errstate(all="ignore"):
        try:
            program(run)
        except _Stop as e:
            raise _stopped(e) from None
        except RecursionError:
            raise CalcError("nested too deeply (a function that calls itself, or a list that holds itself or is nested hundreds deep): "
                            "write the repetition as a loop", "value") from None
        except MemoryError:
            raise CalcError("too large for calc", "value") from None
        finally:
            _CURRENT.run = before
            run.frame = run.ground
            if names is not None:
                _trim(names)
    new = [(k, _shown(v)) for k, v in ground.items()
           if not callable(v) and not isinstance(v, _Module) and k != _DROPPED and (k not in had or had[k] is not v)]
    asked = [(None if name is _PRINT else name, value) for name, value in run.rows]
    if asked:                                                           # what the text asks to see: its bare expressions and prints
        return asked
    return new[-MAX_ECHO:]                                              # nothing asked for: the names it assigned (the last ones)


def render(rows) -> str:
    """The rows as the agent reads them: ``name = value`` or the value, one per line."""
    return "\n".join(f"{name} = {_text(v)}" if name else _text(v) for name, v in rows)


def calc_call(raw_arguments: str) -> tuple[str, bool]:
    """The ``calc`` tool for a harness without an environment: raw argument string -> (text for the agent, valid)."""
    try:
        args = json.loads(raw_arguments) if (raw_arguments or "").strip() else {}
        if not isinstance(args, dict) or set(args) != {"expressions"}:
            raise ValueError('arguments must be {"expressions": "<statements>"}')
        rows = calc(args["expressions"])
    except _CALC_ERRORS as e:
        return f"error: invalid_tool_call: calc: {e}\ncalc is the arithmetic helper of the session, not a scanner tool", False
    if not rows:
        return "calc: nothing to evaluate", True
    return "calc\n" + render(rows), True


_CALC_ERRORS = (ValueError, SyntaxError, TypeError, IndexError, KeyError, ZeroDivisionError, OverflowError, RecursionError, MemoryError)
