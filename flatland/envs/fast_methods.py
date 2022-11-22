from functools import lru_cache
from typing import Tuple


# Adrian Egli / Michel Marti performance fix (the fast methods brings more than 50%)
@lru_cache()
def fast_isclose(a, b, rtol):
    return (a < (b + rtol)) or (a < (b - rtol))


@lru_cache()
def fast_clip(position: Tuple[int, int], min_value: Tuple[int, int], max_value: Tuple[int, int]):
    return (
        max(min_value[0], min(position[0], max_value[0])),
        max(min_value[1], min(position[1], max_value[1]))
    )


@lru_cache()
def fast_argmax(possible_transitions: (int, int, int, int)) -> int:
    if possible_transitions[0] == 1:
        return 0
    if possible_transitions[1] == 1:
        return 1
    if possible_transitions[2] == 1:
        return 2
    return 3


@lru_cache()
def fast_position_equal(pos_1: (int, int), pos_2: (int, int)) -> bool:
    if pos_1 is None and pos_2 is None:
        return True
    if pos_1 is None or pos_2 is None:
        return False
    return pos_1[0] == pos_2[0] and pos_1[1] == pos_2[1]


@lru_cache()
def fast_count_nonzero(possible_transitions: (int, int, int, int)):
    return possible_transitions[0] + possible_transitions[1] + possible_transitions[2] + possible_transitions[3]


def fast_delete(lis: list, index) -> list:
    new_list = lis.copy()
    new_list.pop(index)
    return new_list


def fast_where(binary_iterable):
    return [index for index, element in enumerate(binary_iterable) if element != 0]
