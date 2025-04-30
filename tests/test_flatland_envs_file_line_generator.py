import tempfile
import uuid
from pathlib import Path

import numpy as np

from flatland.envs.line_generators import FileLineGenerator
from flatland.envs.timetable_utils import Line


def test_generate():
    expected = Line(agent_positions=np.random.randint(30, size=(10, 2)).tolist(),
                    agent_targets=np.random.randint(30, size=(10, 2)).tolist(),
                    agent_directions=np.random.randint(4, size=(10, 2)).tolist(),
                    agent_speeds=[1 / c for c in np.random.randint(1, 10, size=10).tolist()],
                    )

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        pkl = Path(tmp_dir_name) / f"{uuid.uuid4()}.pkl"
        FileLineGenerator.save(filename=pkl, line=expected)

        actual = FileLineGenerator(filename=pkl).generate(None, None)
        assert expected == actual
