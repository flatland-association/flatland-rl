from typing import NamedTuple, Tuple, List, Dict

# A way point is the entry into a cell defined by
# - the row and column coordinates of the cell entered
# - direction, in which the agent is facing to enter the cell.
# This induces a graph on top of the FLATland cells:
# - four possible way points per cell
# - edges are the possible transitions in the cell.
Waypoint = NamedTuple('Waypoint', [('position', Tuple[int, int]), ('direction', int)])

# A train run is represented by the waypoints traversed and the times of traversal
# The terminology follows https://github.com/crowdAI/train-schedule-optimisation-challenge-starter-kit/blob/master/documentation/output_data_model.md
TrainrunWaypoint = NamedTuple('TrainrunWaypoint', [
    ('scheduled_at', int),
    ('waypoint', Waypoint)
])
# A train run is the list of an agent's way points and their scheduled time
Trainrun = List[TrainrunWaypoint]
TrainrunDict = Dict[int, Trainrun]
