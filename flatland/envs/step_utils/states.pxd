# Cython pure-Python-mode type augmentation for states.py.
# See https://cython.readthedocs.io/en/latest/src/tutorial/pure.html#augmenting-pxd
#
# `TrainState` is intentionally NOT declared here: it is a `enum.IntEnum`, and IntEnum's metaclass/singleton
# machinery (`_value2member_map_`, member identity, etc.) is not compatible with promoting the class to a
# `cdef class` extension type. Leaving it undeclared keeps it a plain Python object - the surrounding module
# code still compiles to C.
#
# `StateTransitionSignals` IS declared as a `cdef class` below (dropped `@dataclass` - its class-body annotated
# field defaults are incompatible with `.pxd`-based `cdef` attribute declarations - replaced with a manual
# __init__/__repr__/__eq__, confirmed to break no tests on its own). Promoting it to a `cdef class` changes its
# default pickle format (Cython's tuple-based auto-reduce) away from the plain dict-based one that
# pre-existing pickled fixture files (tests/env_data/tests/service_test/*.pkl) use - this would normally break
# unpickling them (as it did for a first attempt, and for TrainStateMachine before this fix). Fixed here (and
# in TrainStateMachine) by manually defining `__getstate__`/`__setstate__` using a dict, which disables
# Cython's auto-generated tuple-based reduce (see
# https://cython.readthedocs.io/en/latest/src/userguide/extension_types.html#pickling, and
# https://github.com/cython/cython/pull/3400 for cdef dataclass support generally - note PR #3400 only affects
# __init__/__repr__/__eq__ codegen, not pickling, so it alone would not have fixed this) and lets the class
# read both old (dict-shaped `__dict__`-based) and new pickles transparently. Verified against the real
# fixture files: all `test_regression_forward`/`test_regression_random` tests in
# test_flatland_envs_persistence.py pass with this fix in place.
cdef class StateTransitionSignals:
    cdef public bint in_malfunction
    cdef public bint earliest_departure_reached
    cdef public bint stop_action_given
    cdef public bint movement_action_given
    # NOT bint: rail_env.py assigns `target_reached = None` as a "not yet known" sentinel (set properly later
    # once the motion check has run). Cython bint would silently coerce None -> False rather than reject it,
    # which happens to be safe here (target_reached is only ever read in a truthy context, never `is None`
    # checked - verified by grep) but keeping it `object` is more honest about the actual value domain.
    cdef public object target_reached
    cdef public bint movement_allowed
    cdef public bint new_speed_zero
