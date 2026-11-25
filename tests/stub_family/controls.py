"""Policies of ``Test/Stub``, through the scanner interface only. What each passes is a function of the number."""
from . import task

REFERENCE = "reference"
MUST_FAIL = ("prior", "fixed_recipe", "brute_force", "wrong_units")
ABLATIONS = {"even_only": (1,)}
REPORTED = ("late",)
DESCRIPTIONS = dict(reference="one scan, then the number", prior="the number without a scan", fixed_recipe="one scan, a constant",
                    brute_force="one scan, minus one", wrong_units="one scan, the number over 1000",
                    even_only="one scan, the number when it is even", late="three scans, then the number")


def _number(sc):
    return sc.info()["task"]["number"]


def _scan(sc, n=1):
    for _ in range(n):
        sc.scan(task.CHEAP_SCAN)


def reference(sc):
    _scan(sc)
    return dict(value=_number(sc))


def prior(sc):
    return dict(value=_number(sc))


def fixed_recipe(recipe):
    def policy(sc):
        _scan(sc)
        return dict(value=recipe["value"])
    return policy


def brute_force(sc):
    _scan(sc)
    return dict(value=-1)                       # refused by the task's check: a failed episode


def wrong_units(sc):
    _scan(sc)
    return dict(value=_number(sc) / 1000.0 + 0.0005)


def even_only(sc):
    _scan(sc)
    n = _number(sc)
    return dict(value=n if n % 2 == 0 else n + 1)


def late(sc):
    _scan(sc, 3)
    return dict(value=_number(sc))


DEFAULT_RECIPE = dict(value=0)


def policies(recipe=None):
    return dict(reference=reference, prior=prior, fixed_recipe=fixed_recipe(recipe or DEFAULT_RECIPE), brute_force=brute_force,
                wrong_units=wrong_units, even_only=even_only, late=late)


def tune(cfgs):
    """The most frequent number of the tuning worlds (the smallest on a tie)."""
    numbers = [c["task"]["number"] for c in cfgs]
    best = min(set(numbers), key=lambda v: (-numbers.count(v), v))
    return dict(value=best, tuned_on=len(cfgs))
