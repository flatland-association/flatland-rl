from flatland.core.grid.grid4 import Grid4TransitionsEnum


def get_direction(pos1, pos2) -> Grid4TransitionsEnum:
    """
    Assumes pos1 and pos2 are adjacent location on grid.
    Returns direction (int) that can be used with transitions.
    """
    diff_0 = pos2[0] - pos1[0]
    diff_1 = pos2[1] - pos1[1]
    if diff_0 < 0:
        return 0
    if diff_0 > 0:
        return 2
    if diff_1 > 0:
        return 1
    if diff_1 < 0:
        return 3
    raise Exception("Could not determine direction {}->{}".format(pos1, pos2))


def mirror(dir):
    return (dir + 2) % 4


def validate_new_transition(rail_trans, rail_array, prev_pos, current_pos, new_pos, end_pos):
    # start by getting direction used to get to current node
    # and direction from current node to possible child node
    new_dir = get_direction(current_pos, new_pos)
    if prev_pos is not None:
        current_dir = get_direction(prev_pos, current_pos)
    else:
        current_dir = new_dir
    # create new transition that would go to child
    new_trans = rail_array[current_pos]
    if prev_pos is None:
        if new_trans == 0:
            # need to flip direction because of how end points are defined
            new_trans = rail_trans.set_transition(new_trans, mirror(current_dir), new_dir, 1)
        else:
            # check if matches existing layout
            new_trans = rail_trans.set_transition(new_trans, current_dir, new_dir, 1)
    else:
        # set the forward path
        new_trans = rail_trans.set_transition(new_trans, current_dir, new_dir, 1)
        # set the backwards path
        new_trans = rail_trans.set_transition(new_trans, mirror(new_dir), mirror(current_dir), 1)
    if new_pos == end_pos:
        # need to validate end pos setup as well
        new_trans_e = rail_array[end_pos]
        if new_trans_e == 0:
            # need to flip direction because of how end points are defined
            new_trans_e = rail_trans.set_transition(new_trans_e, new_dir, mirror(new_dir), 1)
        else:
            # check if matches existing layout
            new_trans_e = rail_trans.set_transition(new_trans_e, new_dir, new_dir, 1)

        if not rail_trans.is_valid(new_trans_e):
            return False

    # is transition is valid?
    return rail_trans.is_valid(new_trans)


def get_new_position(position, movement):
    """ Utility function that converts a compass movement over a 2D grid to new positions (r, c). """
    if movement == Grid4TransitionsEnum.NORTH:
        return (position[0] - 1, position[1])
    elif movement == Grid4TransitionsEnum.EAST:
        return (position[0], position[1] + 1)
    elif movement == Grid4TransitionsEnum.SOUTH:
        return (position[0] + 1, position[1])
    elif movement == Grid4TransitionsEnum.WEST:
        return (position[0], position[1] - 1)
