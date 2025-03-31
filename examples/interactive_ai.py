import os
import pickle
from pathlib import Path

from flatland.evaluators.trajectory_evaluator import TrajectoryEvaluator
from flatland.integrations.interactiveai.interactiveai import FlatlandInteractiveAICallbacks
from flatland.trajectories.trajectories import Trajectory

if __name__ == '__main__':
    # scenario Olten has step every 3 seconds for an hour
    STEPS_ONE_HOUR = 1300  # 1h + additional time for agents to leave the map
    # how many ms per step if replaying in real-time
    REALTIME_STEP_TO_MILLIS = 3600 / STEPS_ONE_HOUR * 1000
    # run faster... limiting factor becomes environment stepping time and blocking requests InteractiveAI platform
    SPEEDUP = 1000

    scenario = "olten_partially_closed"

    # https://github.com/flatland-association/flatland-scenarios/raw/refs/heads/scenario-olten-fix/scenario_olten/data/OLTEN_PARTIALLY_CLOSED_v1.zip

    _dir = os.getenv("SCENARIOS_FOLDER", "../scenarios")
    data_dir = Path(f"{_dir}/scenario_olten/data/{scenario}")

    with (data_dir / "position_to_latlon.pkl").resolve().open("rb") as file_in:
        position_to_latlon_olten = pickle.loads(file_in.read())

    trajectory = Trajectory(data_dir=data_dir, ep_id=scenario)

    # see above for configuration options, use collect_only=False for live POSTing
    cb = FlatlandInteractiveAICallbacks(position_to_latlon_olten, collect_only=True, step_to_millis=REALTIME_STEP_TO_MILLIS / SPEEDUP)
    TrajectoryEvaluator(trajectory, cb).evaluate(end_step=150)
    print(cb.events)
    print(cb.contexts)
