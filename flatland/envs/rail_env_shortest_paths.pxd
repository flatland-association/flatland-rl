# Cython pure-Python-mode type augmentation for rail_env_shortest_paths.py.
# See https://cython.readthedocs.io/en/latest/src/tutorial/pure.html#augmenting-pxd
#
# Only _k_shortest_paths_search is declared here: it's the hot loop extracted from get_k_shortest_paths
# (the public API, left as plain Python). Every parameter here is either a real C primitive/array, or - where
# it structurally can't be (heap, shortest_paths) - `object`: libcpp/C++ containers can't be used in
# rail_env_shortest_paths.py since `cython.cimports.libcpp.*` isn't importable when this module isn't compiled,
# which would break the ext-modules `optional = true` pure-Python fallback (see pyproject.toml / tox.ini's
# verify-build-no-cython env) - so heap/shortest_paths stay plain Python containers (defaultdict(deque) / list)
# rather than approximating them with disallowed types.
#
# target_direction / cutoff are `int`, not `object`+None: both use a -1 sentinel for "unconstrained" (real
# values are always >= 0), since a C int has no None - same reasoning states.pxd documents for why
# target_reached stays `object` there (that field's None sentinel genuinely can't be replaced this way).
cdef void _k_shortest_paths_search(unsigned short[:, :] rail_grid, int height, int width, int k, bint debug,
                                    int target_row, int target_col, int target_direction, int cutoff,
                                    unsigned char[:, :] forbidden_mask, object shortest_paths, int[:] count,
                                    object heap)
