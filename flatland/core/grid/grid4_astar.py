from flatland.core.grid.grid4_utils import validate_new_transition


class AStarNode():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, pos=None):
        self.parent = parent
        self.pos = pos
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.pos == other.pos

    def __hash__(self):
        return hash(self.pos)

    def update_if_better(self, other):
        if other.g < self.g:
            self.parent = other.parent
            self.g = other.g
            self.h = other.h
            self.f = other.f


def a_star(rail_trans, rail_array, start, end):
    """
    Returns a list of tuples as a path from the given start to end.
    If no path is found, returns path to closest point to end.
    """
    rail_shape = rail_array.shape
    start_node = AStarNode(None, start)
    end_node = AStarNode(None, end)
    open_nodes = set()
    closed_nodes = set()
    open_nodes.add(start_node)

    while len(open_nodes) > 0:
        # get node with current shortest est. path (lowest f)
        current_node = None
        for item in open_nodes:
            if current_node is None:
                current_node = item
                continue
            if item.f < current_node.f:
                current_node = item

        # pop current off open list, add to closed list
        open_nodes.remove(current_node)
        closed_nodes.add(current_node)

        # found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.pos)
                current = current.parent
            # return reversed path
            return path[::-1]

        # generate children
        children = []
        if current_node.parent is not None:
            prev_pos = current_node.parent.pos
        else:
            prev_pos = None
        for new_pos in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            node_pos = (current_node.pos[0] + new_pos[0], current_node.pos[1] + new_pos[1])
            if node_pos[0] >= rail_shape[0] or node_pos[0] < 0 or node_pos[1] >= rail_shape[1] or node_pos[1] < 0:
                continue

            # validate positions
            if not validate_new_transition(rail_trans, rail_array, prev_pos, current_node.pos, node_pos, end_node.pos):
                continue

            # create new node
            new_node = AStarNode(current_node, node_pos)
            children.append(new_node)

        # loop through children
        for child in children:
            # already in closed list?
            if child in closed_nodes:
                continue

            # create the f, g, and h values
            child.g = current_node.g + 1
            # this heuristic favors diagonal paths:
            # child.h = ((child.pos[0] - end_node.pos[0]) ** 2) + ((child.pos[1] - end_node.pos[1]) ** 2) \#  noqa: E800
            # this heuristic avoids diagonal paths
            child.h = abs(child.pos[0] - end_node.pos[0]) + abs(child.pos[1] - end_node.pos[1])
            child.f = child.g + child.h

            # already in the open list?
            if child in open_nodes:
                continue

            # add the child to the open list
            open_nodes.add(child)

        # no full path found
        if len(open_nodes) == 0:
            return []
