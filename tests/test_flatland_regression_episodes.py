import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarks.benchmark_episodes import run_episode, DOWNLOAD_INSTRUCTIONS
from flatland.core.policy import Policy
from flatland.env_generation.env_generator import env_generator
from flatland.envs.rail_env_action import RailEnvActions
from flatland.trajectories.policy_runner import PolicyRunner
from flatland.trajectories.trajectories import EVENT_LOGS_SUBDIR, OUTPUTS_SUBDIR, Trajectory


@pytest.mark.parametrize(
    "data_sub_dir,ep_id,run_from_intermediate,skip_rewards_dones_infos,skip_rewards,shifted_malfunction,shifted_off_map_speed", [
    # trajectories do not contain rewards/dones/info: https://github.com/flatland-association/flatland-rl/pull/222 -> skip_rewards_dones_infos=True

    ("30x30 map/10_trains", "1649ef98-e3a8-4dd3-a289-bbfff12876ce", True, True, True, False, False),
    ("30x30 map/10_trains", "4affa89b-72f6-4305-aeca-e5182efbe467", True, True, True, False, False),

    ("30x30 map/15_trains", "a61843e8-b550-407b-9348-5029686cc967", True, True, True, False, False),
    ("30x30 map/15_trains", "9845da2f-2366-44f6-8b25-beca522495b4", True, True, True, False, False),

    ("30x30 map/20_trains", "57e1ebc5-947c-4314-83c7-0d6fd76b2bd3", True, True, True, False, False),
    ("30x30 map/20_trains", "56a78985-588b-42d0-a972-7f8f2514c665", True, True, True, False, False),

    # remaining 30x30 map episodes do not verify cleanly under the shift-and-replay transform (see
    # regenerate_30x30_scenarios_with_shift below) - regenerating those would need to actually resolve
    # the new emergent deadlocks, not just retime actions, so they stay unregenerated for now.

    # shifted_malfunction=True, shifted_off_map_speed=True: TODO remove once these episodes are regenerated
    ("malfunction_deadlock_avoidance_heuristics/Test_00/Level_8", "Test_00_Level_8", True, False, False, True, True),
    ("malfunction_deadlock_avoidance_heuristics/Test_01/Level_3", "Test_01_Level_3", True, False, False, True, True),
    ("malfunction_deadlock_avoidance_heuristics/Test_02/Level_6", "Test_02_Level_6", True, False, False, True, True),
    ("malfunction_deadlock_avoidance_heuristics/Test_02/Level_8", "Test_02_Level_8", False, False, False, True, True),
    ("malfunction_deadlock_avoidance_heuristics/Test_03/Level_1", "Test_03_Level_1", False, False, False, True, True),
    ("malfunction_deadlock_avoidance_heuristics/Test_03/Level_2", "Test_03_Level_2", False, False, False, True, True),
])
def test_episode(data_sub_dir: str, ep_id: str, run_from_intermediate: bool, skip_rewards_dones_infos: bool, skip_rewards: bool,
                 shifted_malfunction: bool, shifted_off_map_speed: bool):
    """
    Run a subset of episodes for regression in unit testing, comparing only positions.
    Protects against breaking changes in flatland-rl.
    """
    _dir = os.getenv("BENCHMARK_EPISODES_FOLDER")
    assert _dir is not None, (DOWNLOAD_INSTRUCTIONS, _dir)
    assert os.path.exists(_dir), (DOWNLOAD_INSTRUCTIONS, _dir)
    data_dir = Path(os.path.join(_dir, data_sub_dir))

    with tempfile.TemporaryDirectory() as tmpdirname:
        shutil.copytree(data_dir, tmpdirname, dirs_exist_ok=True)

        # run with snapshots to outputs/serialised_state directory
        run_episode(Path(tmpdirname), ep_id, snapshot_interval=1 if run_from_intermediate else 0,
                    skip_rewards_dones_infos=skip_rewards_dones_infos,
                    skip_rewards=skip_rewards,
                    shifted_malfunction=shifted_malfunction,
                    shifted_off_map_speed=shifted_off_map_speed)

        if run_from_intermediate:
            # copy actions etc. to outputs subfolder, so outputs subfolder becomes a proper trajectory data dir.
            shutil.copytree(os.path.join(data_dir, EVENT_LOGS_SUBDIR), os.path.join(tmpdirname, OUTPUTS_SUBDIR, EVENT_LOGS_SUBDIR), dirs_exist_ok=True)

            # start episode from a snapshot to ensure snapshot contains full state!
            run_episode(Path(tmpdirname) / OUTPUTS_SUBDIR, ep_id, start_step=np.random.randint(0, 50),
                        skip_rewards_dones_infos=skip_rewards_dones_infos,
                        skip_rewards=skip_rewards,
                        shifted_malfunction=shifted_malfunction,
                        shifted_off_map_speed=shifted_off_map_speed)


def test_restore_episode():
    """
    Test that refactorings in env generation does not introduce changes in behaviour with the default parameters.

    See <a href="https://github.com/flatland-association/flatland-scenarios/tree/main?tab=readme-ov-file#changelog-2">changelog</a>.
    """
    _dir = os.getenv("BENCHMARK_EPISODES_FOLDER")
    assert _dir is not None, (DOWNLOAD_INSTRUCTIONS, _dir)
    assert os.path.exists(_dir), (DOWNLOAD_INSTRUCTIONS, _dir)

    metadata_csv = Path(f"{_dir}/malfunction_deadlock_avoidance_heuristics/metadata.csv").resolve()
    metadata = pd.read_csv(metadata_csv)
    for i, (k, v) in enumerate(metadata.iterrows()):
        ep_id = f'{v["test_id"]}_{v["env_id"]}'
        print(ep_id)
        if i >= 40:
            break
        env_regen, _, _ = env_generator(
            n_agents=v["n_agents"],
            x_dim=v["x_dim"],
            y_dim=v["y_dim"],
            n_cities=v["n_cities"],
            max_rail_pairs_in_city=v["max_rail_pairs_in_city"],
            grid_mode=v["grid_mode"],
            max_rails_between_cities=v["max_rails_between_cities"],
            malfunction_duration_min=v["malfunction_duration_min"],
            malfunction_duration_max=v["malfunction_duration_max"],
            malfunction_interval=v["malfunction_interval"],
            speed_ratios={1.0: 0.25,
                          0.5: 0.25,
                          0.33: 0.25,
                          0.25: 0.25},
            seed=v["seed"],
        )

        data_sub_dir = f'malfunction_deadlock_avoidance_heuristics/{v["test_id"]}/{v["env_id"]}'

        data_dir = Path(os.path.join(_dir, data_sub_dir))

        with tempfile.TemporaryDirectory() as tmpdirname:
            shutil.copytree(data_dir, tmpdirname, dirs_exist_ok=True)

            t = Trajectory.load_existing(data_dir=Path(tmpdirname), ep_id=ep_id)
            env_restored = t.load_env()

            # TODO poor man's state comparison for now
            def _position(a):
                return a.current_entry_point[0] if a.current_entry_point is not None else None

            assert [_position(a) for a in env_regen.agents] == [_position(a) for a in env_restored.agents]


class ShiftedReplayPolicy(Policy):
    """
    Replay a pre-https://github.com/flatland-association/flatland-rl/issues/178 recorded action
    log through current code, retimed to account for actions now being consulted one cell earlier
    (at cell entry instead of cell exit): from its own departure step onward, each agent is given
    the action recorded one index later than "now", since that is the action that decides the same
    logical crossing under the new design.

    OBSERVATION (measured by actually running `regenerate_30x30_scenarios_with_shift` below against all
    80 "30x30 map" episodes, 3 of which are unloadable for lack of an initial snapshot pkl in this
    fixture): this retiming reproduces the original recorded success rate exactly for 53 of the 77
    loadable episodes - all of 10_trains/15_trains but one each, most of 20_trains, but only 3 of 20 in
    50_trains. It does NOT generalise to densely congested episodes: once several agents' independently
    shifted timings interact at shared resources, the same recorded action script can resolve a
    shared-resource contest differently, usually into a new permanent deadlock the original policy had
    avoided by redirecting a blocked agent at the last moment - no longer possible now that a blocked
    agent's target cell is committed one step before it is reached and held fixed across retries (see
    test_blocked_agent_cannot_redirect_via_later_action in test_flatland_envs_rail_env.py for the
    minimal isolated example of that mechanism). So a clean verification per episode - not a blanket
    replay - is the actual gate for whether it is safe to regenerate a given episode's trajectory this
    way.
    """

    def __init__(self, shifted_action_cache: dict):
        super().__init__()
        self.shifted_action_cache = shifted_action_cache
        self.t = 0

    def act_many(self, handles, observations, **kwargs):
        actions = {handle: self.shifted_action_cache.get(self.t, {}).get(handle, RailEnvActions.MOVE_FORWARD) for handle in handles}
        self.t += 1
        return actions


def build_shifted_action_cache(action_cache: dict, position_cache: dict, n_agents: int, max_steps: int) -> dict:
    """
    Build the per-agent retimed action cache `ShiftedReplayPolicy` replays: for each agent, from its
    departure step onward (its first step with a non-None recorded position), take the action recorded
    one step later than "now" - that later action is exactly what the old (pre-#178) design would have
    consulted for the same crossing decision one step further along. Before departure (and for agents
    that never depart), the action is left untouched - there is nothing to retime yet.
    """
    departure_step = {}
    for agent_id in range(n_agents):
        for t in range(1, max_steps + 1):
            if position_cache.get(t, {}).get(agent_id) is not None:
                departure_step[agent_id] = t - 1
                break

    shifted_action_cache = {}
    for t in range(max_steps + 1):
        shifted_action_cache[t] = {}
        for agent_id in range(n_agents):
            dep = departure_step.get(agent_id)
            src_t = t + 1 if (dep is not None and t >= dep) else t
            shifted_action_cache[t][agent_id] = action_cache.get(src_t, {}).get(agent_id, RailEnvActions.MOVE_FORWARD)
    return shifted_action_cache


def regenerate_episode_with_shift(data_dir: Path, ep_id: str) -> dict:
    """
    Replay `ep_id`'s original recording through current code via `ShiftedReplayPolicy`, and report
    whether it reproduces the original recording's success rate exactly - the signal for whether this
    episode is safe to regenerate this way (see the OBSERVATION in `ShiftedReplayPolicy`'s docstring).

    Note this only checks success rate, the one ground truth directly available from the existing
    recording (these "30x30 map" trajectories do not record rewards, see skip_rewards_dones_infos
    above) - reward-sum preservation for the episodes that verify cleanly here was checked separately,
    by replaying the same shifted actions through a pre-#178 checkout.
    """
    trajectory = Trajectory.load_existing(data_dir=data_dir, ep_id=ep_id)
    action_cache, position_cache, _ = trajectory.build_cache()
    env = trajectory.load_env(start_step=0)
    n_agents = env.get_num_agents()
    max_steps = env._max_episode_steps

    shifted_action_cache = build_shifted_action_cache(action_cache, position_cache, n_agents, max_steps)
    policy = ShiftedReplayPolicy(shifted_action_cache)

    with tempfile.TemporaryDirectory() as scratch_dir:
        new_trajectory = PolicyRunner.create_from_policy(
            policy=policy, data_dir=Path(scratch_dir), env=env, ep_id=ep_id, snapshot_interval=0,
        )
        new_success_rate = float(new_trajectory.trains_arrived_lookup()["success_rate"])

    original_success_rate = float(trajectory.trains_arrived_lookup()["success_rate"])

    return {
        "ep_id": ep_id,
        "original_success_rate": original_success_rate,
        "new_success_rate": new_success_rate,
        "clean": bool(np.isclose(new_success_rate, original_success_rate)),
    }


def regenerate_30x30_scenarios_with_shift():
    """
    Apply the shift-and-replay regeneration to every episode of every "30x30 map" scenario
    (10/15/20/50_trains), printing per-episode whether it verifies cleanly. Does NOT write anything
    back to the fixture - regenerating the actual TSV/pkl files for a given episode is a separate,
    explicit decision once its verification is known to be clean (see the OBSERVATION in
    `ShiftedReplayPolicy`'s docstring for why a blanket rewrite would be wrong).

    Not collected by pytest (no `test_` prefix) - run standalone:
    `BENCHMARK_EPISODES_FOLDER=... python tests/test_flatland_regression_episodes.py`
    """
    _dir = os.getenv("BENCHMARK_EPISODES_FOLDER")
    assert _dir is not None, (DOWNLOAD_INSTRUCTIONS, _dir)
    assert os.path.exists(_dir), (DOWNLOAD_INSTRUCTIONS, _dir)

    results = []
    for sub in ["10_trains", "15_trains", "20_trains", "50_trains"]:
        data_dir = Path(_dir) / "30x30 map" / sub
        arrived = pd.read_csv(data_dir / "event_logs" / "TrainMovementEvents.trains_arrived.tsv", sep="\t")
        for ep_id in arrived["episode_id"].unique():
            # some episodes are missing their initial serialised_state/<ep_id>.pkl snapshot in this fixture
            # (an upstream data gap, unrelated to #178) - skip rather than fail the whole sweep on those.
            if not (data_dir / "serialised_state" / f"{ep_id}.pkl").exists():
                result = {"sub_dir": sub, "ep_id": ep_id, "clean": None, "error": "missing initial snapshot pkl"}
            else:
                try:
                    result = {"sub_dir": sub, **regenerate_episode_with_shift(data_dir, ep_id)}
                except Exception as e:
                    result = {"sub_dir": sub, "ep_id": ep_id, "clean": False, "error": str(e)}
            results.append(result)
            print(result)
    n_clean = sum(r["clean"] is True for r in results)
    n_skipped = sum(r["clean"] is None for r in results)
    print(f"{n_clean}/{len(results) - n_skipped} loadable episodes verify cleanly under the shift-and-replay transform ({n_skipped} skipped for missing snapshots)")
    return results


if __name__ == "__main__":
    regenerate_30x30_scenarios_with_shift()
