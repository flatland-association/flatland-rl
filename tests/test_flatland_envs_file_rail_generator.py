import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest

from flatland.envs.rail_generators import sparse_rail_generator, FileRailFromGridGen, RailGeneratorProduct


@pytest.mark.parametrize("w, h, na, w2, h2, na2, warn", [
    (30, 32, 33, 30, 32, 33, False),
    (30, 32, 33, 40, 42, 33, True),
])
def test_generate(w: int, h: int, na: int, w2: int, h2: int, na2: int, warn: bool):
    expected: RailGeneratorProduct = sparse_rail_generator()(w, h, na, 0)

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        pkl = Path(tmp_dir_name) / f"{uuid.uuid4()}.pkl"
        FileRailFromGridGen.save(filename=pkl, prod=expected)

        if warn:
            with pytest.warns(UserWarning) as record:
                actual = FileRailFromGridGen(filename=pkl).generate(width=w2, height=h2, num_agents=na2)
                assert 2 == len(record)
                assert str(record[0].message) == f"Expected height {h2}, found {h}."
                assert str(record[1].message) == f"Expected width {w2}, found {w}."
        else:
            actual = FileRailFromGridGen(filename=pkl).generate(width=w2, height=h2, num_agents=na2)
        assert expected[1] == actual[1]
        assert expected[0].grid.shape == actual[0].grid.shape
        assert np.array_equal(expected[0].grid, actual[0].grid)
