from typing import NamedTuple, Tuple, List

WayPoint = NamedTuple('WayPoint', [('position', Tuple[int, int]), ('direction', int)])


# A train run is represented by the waypoints traversed and the times of traversal
# The terminology follows https://github.com/crowdAI/train-schedule-optimisation-challenge-starter-kit/blob/master/documentation/output_data_model.md
TrainRunWayPoint = NamedTuple('TrainRunWayPoint', [
    ('scheduled_at', int),
    ('way_point', WayPoint)
])
# A path schedule is the list of an agent's cell pin entries
TrainRun = List[TrainRunWayPoint]
