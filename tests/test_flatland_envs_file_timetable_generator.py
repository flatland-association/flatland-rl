import tempfile
import uuid
from pathlib import Path

import numpy as np

from flatland.envs.timetable_generators import FileTimetableGenerator
from flatland.envs.timetable_utils import Timetable


def test_generate():
    expected = Timetable(earliest_departures=np.random.randint(2, size=10).tolist(),
                         latest_arrivals=np.random.randint(5, size=10).tolist(),
                         max_episode_steps=55)

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        pkl = Path(tmp_dir_name) / f"{uuid.uuid4()}.pkl"
        FileTimetableGenerator.save(filename=pkl, tt=expected)

        actual = FileTimetableGenerator(filename=pkl).generate()
        assert expected.earliest_departures == actual.earliest_departures
        assert expected.latest_arrivals == actual.latest_arrivals
        assert expected.max_episode_steps == actual.max_episode_steps
