import tempfile
import uuid
from pathlib import Path

from flatland.env_generation.env_generator import env_generator
from flatland.envs.line_generators import SparseLineGen
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import FileRailFromGridGen
from flatland.envs.timetable_generators import FileTimetableGenerator


def test_rail_env():
    rail_env = env_generator()

    # TODO refactor reset instead
    prod = rail_env.rail_generator(rail_env.width, rail_env.height, rail_env.number_of_agents, rail_env.num_resets, rail_env.np_random)
    rail, optionals = prod
    agents_hints = optionals["agents_hints"]
    line = rail_env.line_generator(rail_env.rail, rail_env.number_of_agents, agents_hints,
                                   rail_env.num_resets, rail_env.np_random)
    tt = rail_env.timetable_generator(rail_env.agents, rail_env.distance_map, agents_hints, rail_env.np_random)

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        tt_pkl = Path(tmp_dir_name) / f"{uuid.uuid4()}.pkl"
        rail_pkl = Path(tmp_dir_name) / f"{uuid.uuid4()}.pkl"
        FileRailFromGridGen.save_rail_generator_product(rail_pkl, prod)
        FileTimetableGenerator.save(tt_pkl, tt)

        rail_env_loaded = RailEnv(
            width=rail_env.width, height=rail_env.height,
            number_of_agents=rail_env.number_of_agents,
            rail_generator=FileRailFromGridGen(rail_pkl),
            # TODO FileLineGen
            line_generator=SparseLineGen(),
            timetable_generator=FileTimetableGenerator(tt_pkl)
        )
        rail_env_loaded.reset()

        assert rail_env.rail.grid.tolist() == rail_env_loaded.rail.grid.tolist()
        # TODO will not be the same as the two resets do not pass through same generators - how to harmonize?
        # assert rail_env.np_random.get_state() == rail_env_loaded.np_random.get_state()
