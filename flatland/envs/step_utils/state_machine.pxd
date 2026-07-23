# Cython pure-Python-mode type augmentation for state_machine.py.
# See https://cython.readthedocs.io/en/latest/src/tutorial/pure.html#augmenting-pxd
#
# Promoting TrainStateMachine to a `cdef class` changes its default pickle format away from the plain
# dict-based one pre-existing pickled fixture files use (see states.pxd for the full explanation and links) -
# fixed by TrainStateMachine.__getstate__/__setstate__ in state_machine.py, which disables Cython's
# auto-generated tuple-based reduce. Verified against the real fixture files: all
# test_regression_forward/test_regression_random tests in test_flatland_envs_persistence.py pass with this
# fix in place.

from flatland.envs.step_utils.states cimport StateTransitionSignals

# module-level C int constants mirroring TrainState's values, hoisted once in state_machine.py so
# calculate_next_state's hot dispatch can compare a `cython.int` local against pure C ints instead of
# repeatedly doing a Python attribute lookup on `TrainState.WAITING` etc. per branch.
cdef int _WAITING
cdef int _READY_TO_DEPART
cdef int _MALFUNCTION_OFF_MAP
cdef int _MOVING
cdef int _STOPPED
cdef int _MALFUNCTION
cdef int _DONE

cdef class TrainStateMachine:
    # `TrainState` is a plain enum.IntEnum (see states.pxd) - not a cdef class - so these stay
    # generic `object` references rather than a specific Cython extension type.
    cdef public object _initial_state
    cdef public object _state
    cdef public object next_state
    cdef public object previous_state
    cdef public StateTransitionSignals st_signals

    # internal dispatch helpers: cdef (not cpdef) since they're only ever called from
    # calculate_next_state within this module, never from outside Python code.
    cdef void _handle_waiting(self)
    cdef void _handle_ready_to_depart(self)
    cdef void _handle_malfunction_off_map(self)
    cdef void _handle_moving(self)
    cdef void _handle_stopped(self)
    cdef void _handle_malfunction(self)
    cdef void _handle_done(self)

    cpdef void calculate_next_state(self, current_state)
    cpdef void step(self)
    cpdef void clear_next_state(self)
    cpdef void set_state(self, state)
    cpdef void reset(self)
    cpdef void update_if_reached(self, configuration, targets)
    cpdef void set_transition_signals(self, state_transition_signals)
    cpdef void state_position_sync_check(self, configuration, i_agent, bint remove_agents_at_target)

    # `can_get_moving_independent` is intentionally NOT declared here: this Cython version does not
    # support `@staticmethod` combined with `cpdef` ("static cpdef methods not yet supported"), and since
    # its only caller (rail_env.py) is plain, uncompiled Python, a `cdef`-only declaration would make it
    # uncallable from there. It stays a plain Python staticmethod - no pxd entry needed.
