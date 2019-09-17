from flatland.core.grid.grid_utils import IntVector2D
from flatland.core.grid.grid_utils import IntVector2DArrayType
from flatland.core.grid.grid_utils import Vec2dOperations as Vec2d
from flatland.core.grid.rail_env_grid import RailEnvTransitions
from flatland.core.transition_map import GridTransitionMap


class AStarNode:
    """A node class for A* Pathfinding"""

    def __init__(self, parent: IntVector2D = None, pos: IntVector2D = None):
        self.parent: IntVector2D = parent
        self.pos: IntVector2D = pos
        self.g = 0.0
        self.h = 0.0
        self.f = 0.0

    def __eq__(self, other: IntVector2D):
        return self.pos == other.pos

    def __hash__(self):
        return hash(self.pos)

    def update_if_better(self, other):
        if other.g < self.g:
            self.parent = other.parent
            self.g = other.g
            self.h = other.h
            self.f = other.f


def a_star(rail_trans: RailEnvTransitions,
           grid_map: GridTransitionMap,
           start: IntVector2D, end: IntVector2D) -> IntVector2DArrayType:
    """
    Returns a list of tuples as a path from the given start to end.
    If no path is found, returns path to closest point to end.
    """
    rail_shape = grid_map.grid.shape
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
            # update the "current" pos
            node_pos = Vec2d.add(current_node.pos, new_pos)

            # is node_pos inside the grid?
            if node_pos[0] >= rail_shape[0] or node_pos[0] < 0 or node_pos[1] >= rail_shape[1] or node_pos[1] < 0:
                continue

            # validate positions
            if not grid_map.validate_new_transition(prev_pos, current_node.pos, node_pos, end_node.pos):
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
            child.g = current_node.g + 1.0
            # this heuristic avoids diagonal paths
            child.h = Vec2d.get_manhattan_distance(child.pos, end_node.pos)
            child.f = child.g + child.h

            # already in the open list?
            if child in open_nodes:
                continue

            # add the child to the open list
            open_nodes.add(child)

        # no full path found
        if len(open_nodes) == 0:
            return []
