import os.path
import tempfile
import uuid
from pathlib import Path

import numpy as np

from flatland.env_generation.env_generator import env_generator
from flatland.envs.line_generators import FileLineGenerator
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.envs.timetable_utils import Line


def test_generate():
    agent_positions = np.random.randint(30, size=(10, 2)).tolist()
    agent_targets = np.random.randint(30, size=(10, 2)).tolist()
    agent_directions = np.random.randint(4, size=(10)).tolist()
    agent_waypoints = {
        i: [[Waypoint(position, direction)], [Waypoint(target, None)]]
        for i, (position, direction, target) in enumerate(zip(agent_positions, agent_directions, agent_targets))
    }
    expected = Line(agent_waypoints=agent_waypoints, agent_speeds=[1 / c for c in np.random.randint(1, 10, size=10).tolist()], )

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        pkl = Path(tmp_dir_name) / f"{uuid.uuid4()}.pkl"
        FileLineGenerator.save(filename=pkl, line=expected)

        actual = FileLineGenerator(filename=pkl).generate(None, None)
        assert expected == actual


def test_line_timetable_generator_from_file_behaves_in_reset_behaves_same_as_load_new():
    env, _, _ = env_generator(seed=42)
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        RailEnvPersister.save(env, os.path.join(tmp_dir_name, "env.pkl"))
        env2, _ = RailEnvPersister.load_new(os.path.join(tmp_dir_name, "env.pkl"))

    # same from file
    assert np.array_equal(env.rail.grid, env2.rail.grid)
    assert env.agents == env2.agents

    # same from file even after reset
    env2.reset()
    assert np.array_equal(env.rail.grid, env2.rail.grid)
    assert env.agents == env2.agents
