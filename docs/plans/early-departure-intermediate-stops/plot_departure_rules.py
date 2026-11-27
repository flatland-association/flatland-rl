"""#554 Q-1 (plan docs/plans/early-departure-intermediate-stops.md): the departure from an intermediate stop B
after halt, creep, halt and start, dated by main, a) and b).

Reuses plot_trajectory() of flatland-book's environment/environment/speed_profiles.ipynb (cell 4), unchanged except:
it returns the figure instead of plt.show(), STOPPED/DONE get colours, and the title is set here.
Run: PYTHONPATH=<flatland-rl> python plot_q1.py <flatland-book>/environment/environment/speed_profiles.ipynb <out dir>
"""
import json
import sys
from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

out = Path(sys.argv[2])
cell_src = next(src for src in ("".join(c["source"]) for c in json.loads(Path(sys.argv[1]).read_text())["cells"])
                if "def plot_trajectory(" in src)
# increments as fractions (the book's .1f shows 1/4 as 0.2)
cell_src = cell_src.replace('f"+{values[k] - values[k - 1]:.1f}"',
                            '("+" if values[k] > values[k - 1] else "") + str(Fraction(values[k] - values[k - 1]).limit_denominator(8))')
cell_src = cell_src.replace("    plt.show()\n\n\ndef record_trajectory", "    return fig, ax_s, ax_d, ax_states\n\n\ndef record_trajectory")
ns = {}
exec(compile(cell_src, "speed_profiles_cell4", "exec"), ns)
ns["STATE_COLORS"].update({"STOPPED": ("#fecaca", "#dc2626"), "DONE": ("#ddd6fe", "#7c3aed")})

import numpy as np  # noqa: E402
from flatland.envs.grid.rail_env_grid import RailEnvTransitions  # noqa: E402
from flatland.envs.observations import FullEnvObservation  # noqa: E402
from flatland.envs.rail_env import RailEnv  # noqa: E402
from flatland.envs.rail_env_action import RailEnvActions  # noqa: E402
from flatland.envs.rail_generators import rail_from_grid_transition_map  # noqa: E402
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap  # noqa: E402
from flatland.envs.rail_trainrun_data_structures import Waypoint  # noqa: E402
from flatland.envs.rewards import DefaultRewards  # noqa: E402
from flatland.envs.timetable_utils import Line, Timetable  # noqa: E402

ED_A, ED_B = 1, 12  # earliest departure at A and at the stop B
FWD, STOP = RailEnvActions.MOVE_FORWARD, RailEnvActions.STOP_MOVING
X_MAX = 18

n = 10
transitions = RailEnvTransitions()
cells = transitions.transition_list
straight = transitions.rotate_transition(cells[1], 90)
rail = RailGridTransitionMap(width=n, height=1, transitions=transitions)
rail.grid = np.array([[transitions.rotate_transition(cells[7], 270)] + [straight] * (n - 2) + [transitions.rotate_transition(cells[7], 90)]],
                     dtype=np.uint16)
A, B, C = ((0, 1), 1), ((0, 2), 1), ((0, 8), 1)
waypoints = [[Waypoint(*A)], [Waypoint(*B)], [Waypoint(*C)]]
rewards = DefaultRewards()
env = RailEnv(width=n, height=1, rail_generator=rail_from_grid_transition_map(rail),
              line_generator=lambda *a, **k: Line(agent_waypoints={0: waypoints}, agent_speeds=[Fraction(1, 4)]),
              timetable_generator=lambda *a, **k: Timetable(earliest_departures=[[ED_A, ED_B, None]], latest_arrivals=[[None, 40, 60]],
                                                            max_episode_steps=60),
              number_of_agents=1, obs_builder_object=FullEnvObservation(), rewards=rewards)
env.reset()
agent = env.agents[0]
rows = []  # (elapsed_steps, state, entry point, speed, travelled)


def record():
    sc, ep = agent.speed_counter, agent.current_entry_point
    travelled = None if ep is None else (ep[0][1] - 1) + sc.distance
    rows.append((env._elapsed_steps, agent.state.name, ep, sc.speed, travelled))


record()
early = {}
# phases at B: brake on entry (first halt), creep one step, brake (second halt), wait, then leave 2 steps before ED_B
phase, actions = "approach", {}
for k in range(1, X_MAX + 1):
    sc, ep = agent.speed_counter, agent.current_entry_point
    if phase == "approach" and ep == B:
        action, phase = STOP, "halt1"
    elif phase == "halt1" and sc.speed == 0:
        action, phase = FWD, "creep"
    elif phase == "creep":
        action, phase = STOP, "halt2"
    elif phase == "halt2" and k < ED_B - 2:
        action = STOP
    elif phase == "halt2":
        action, phase = FWD, "leave"
    elif phase == "halt1":
        action = STOP
    else:
        action = FWD
    actions[k] = action
    _, r, dones, _ = env.step({0: action})
    record()
    if r[0]:
        early[k] = r[0]
for row in rows:
    print(row)

# departure from B by each rule
on_b = [row for row in rows if row[2] == B]
halts = [row[0] for row in on_b if row[3] == 0]
starts_after_halt = [row[0] for row in on_b if row[3] > 0 and row[0] > halts[0]]
# main before #554: the departure is the entry into the next cell (computed here, the rewards on the branch date it as a))
dep_main = next(row[0] for row in rows if row[0] > on_b[0][0] and row[2] is not None and row[2] != B)
dep_a = halts[-1] + 1
dep_b = starts_after_halt[0]
print("actions", {k: a.name for k, a in actions.items()})
print("halts at B", halts, "departure main", dep_main, "a)", dep_a, "b)", dep_b, "rewards", early)

speed = [float(row[3] or 0) for row in rows]
travelled = [float(row[4] or 0) for row in rows]


def spans(states):
    starts = [k for k in range(len(states)) if k == 0 or states[k] != states[k - 1]]
    return [(k, ([*starts[1:], X_MAX])[i], states[k]) for i, k in enumerate(starts)]


ns["T"], ns["X_MAX"], ns["DISTANCE_MAX"] = ED_B, X_MAX, 4.5
ns["SPEED_MAX"] = 1.5
action_name = {FWD: "MOVE_FORWARD", STOP: "STOP_MOVING"}
# labels at elapsed_steps = k for the action into the step k -> k+1
first_halt, creep, second_halt = halts[0], dep_b, halts[1]
labels = [(first_halt - 1, "STOP_MOVING: halts in B", 0, 0.62, "speed"),
          (creep - 1, "MOVE_FORWARD: creeps 1/4 cell in B", 0.25, 0.86, "speed"),
          (second_halt - 1, "STOP_MOVING: halts again in B", 0, 1.1, "speed"),
          (dep_a - 1, "MOVE_FORWARD: starts, leaves B", 0.25, 0.4, "speed")]
penalty = DefaultRewards().intermediate_early_departure_penalty_factor
for key, dep, text in [
    ("main", dep_main, "main today: departure = entry into the next cell"),
    ("a", dep_a, "a) last start: departure = last halt + 1"),
    ("b", dep_b, "b) first start: departure = first start after the first halt"),
]:
    fig, ax_s, ax_d, ax_states = ns["plot_trajectory"](Fraction(1), speed, travelled, {"state": spans([row[1] for row in rows])}, labels)
    ax_d.set_ylim(0, 4.5)
    ax_d.set_yticks([0, 1, 2, 3])
    ax_s.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax_s.set_ylim(0, 1.5)
    ax_d.axhspan(1, 2, color="#ede9fe", alpha=0.6, zorder=0)
    ax_d.text(X_MAX - 0.1, 1.5, "cell B", ha="right", va="center", fontsize=8, color="#6b7280")
    ax_s.axvline(dep, color="#7c3aed", linewidth=3, alpha=0.8)
    ax_states.axvline(dep, color="#7c3aed", linewidth=3, alpha=0.8)
    lateness = ED_B - dep
    verdict = f"early by {lateness} → penalty {-penalty * lateness:g}" if lateness > 0 else "on time → no penalty"
    ax_s.text(dep + 0.1, 1.42, f"departure from B = {dep}\n{verdict}", color="#7c3aed", fontsize=9, va="top", fontweight="bold")
    ax_s.set_title(f"{text}\nmax speed 1/4, B is the cell at travelled 1-2, earliest departure at B = {ED_B}", color="#1f2937", fontsize=10,
                   loc="left")
    fig.savefig(out / f"departure-{key}.png", dpi=110, bbox_inches="tight")
    print("wrote", out / f"departure-{key}.png", dep, verdict)
