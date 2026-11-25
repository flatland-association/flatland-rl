"""Builds the tutorial scenario (flatland-ui-hub plan-b `tutorial`, drawn with the drawing tool) as RailEnv and
GraphRailEnv, steps it, and writes the proposed exchange-format scenario documents."""
import json, sys, ast
from fractions import Fraction
import numpy as np
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.grid.rail_env_grid import RailEnvTransitions
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.timetable_utils import Line, Timetable
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.envs.graph_rail_env import GraphRailEnv
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.envs.persistence import RailEnvPersister

UIHUB, BOOK, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
gen = json.load(open(f"{UIHUB}/generator.json"))
sl_grid = json.load(open(f"{BOOK}/example-ui-hub-plan-b.json"))
H, W = gen["gridDimensions"]["rows"], gen["gridDimensions"]["cols"]
STEPS = 10

def make_env():
    tm = RailGridTransitionMap(width=W, height=H, transitions=RailEnvTransitions())
    tm.grid = np.array(gen["grid"], dtype=np.uint16)
    fl, ft = gen["flatlandLine"], gen["flatlandTimetable"]
    wps = {i: [[Waypoint((2, 3), 1)], [Waypoint((2, 21), None)]] if i == 0 else [[Waypoint((2, 21), 3)], [Waypoint((2, 3), None)]] for i in range(2)}
    line = Line(agent_waypoints=wps, agent_speeds=fl["agent_speeds"])
    tt = Timetable(earliest_departures=[[0, None], [2, None]], latest_arrivals=ft["latest_arrivals"], max_episode_steps=ft["max_episode_steps"])
    env = RailEnv(width=W, height=H, rail_generator=rail_from_grid_transition_map(tm), line_generator=lambda *a, **k: line,
                  timetable_generator=lambda *a, **k: tt, number_of_agents=2, obs_builder_object=GlobalObsForRailEnv())
    env.reset(random_seed=42)
    return env

def frac(x):
    return None if x is None else str(Fraction(x))

def pos(p):
    return None if p is None else [int(v) for v in p]

def ep(e, graph):
    if e is None: return None
    if graph: return e
    (r, c), d = e
    return {"position": [int(r), int(c)], "direction": None if d is None else int(d)}

def rng(env):
    name, key, p, hg, cg = env.np_random.get_state()
    return {"bit_generator": name, "key": [int(k) for k in key], "pos": int(p), "has_gauss": int(hg), "cached_gaussian": float(cg)}

def lines(env, graph):
    out = []
    for a in env.agents:
        wps = [[(str(w) if graph and not hasattr(w, "position") else {"position": pos(w.position), "direction": None if w.direction is None else int(w.direction)}) for w in g] for g in a.waypoints]
        out.append({"handle": a.handle, "waypoints": wps, "earliest_departures": a.waypoints_earliest_departure,
                    "latest_arrivals": a.waypoints_latest_arrival, "max_speed": frac(a.speed_counter.max_speed)})
    return out

def state(env, graph):
    ag = []
    for a in env.agents:
        sm = a.state_machine
        ag.append({"handle": a.handle, "entry_point": ep(a.current_entry_point, graph), "old_entry_point": ep(a.old_entry_point, graph),
                   "target_entry_point": ep(a.target_entry_point, graph), "moving": bool(a.moving),
                   "speed": frac(a.speed_counter.speed), "distance": frac(a.speed_counter.distance), "arrival_time": a.arrival_time,
                   "malfunction_down_counter": a.malfunction_handler.malfunction_down_counter,
                   "num_malfunctions": a.malfunction_handler.num_malfunctions})
    return {"elapsed_steps": env._elapsed_steps, "random": {"seed": env.random_seed, "seed_history": [int(s) for s in env.seed_history], "np_random_state": rng(env)},
            "agents": ag, "dones": {str(k): bool(v) for k, v in env.dones.items()},
            "effects": env.effects_generator.__getstate__()}

def run(env):
    for _ in range(STEPS):
        env.step({a.handle: RailEnvActions.MOVE_FORWARD for a in env.agents})

def doc(env, sl, binding, graph, rail, generators, extensions=None, with_state=True):
    d = {"format": "flatland-scenario", "version": "1.0.0", "binding": binding, "extensions_used": {k: None for k in (extensions or {})}}
    d["configuration"] = {"rail": rail, "stations_links": sl, "lines": lines(env, graph), "max_episode_steps": env._max_episode_steps,
                          "seed": env.random_seed, "generators": generators,
                          "skip_state_machine_update": bool(getattr(env, "skip_state_machine_update", True))}
    if extensions: d["configuration"]["extensions"] = extensions
    d["state"] = state(env, graph) if with_state else {}
    return d

GENS = {"rail": {"cls": "flatland.envs.rail_generators.RailFromGridGen"}, "line": {"cls": "flatland.envs.line_generators.FileLineGenerator"},
        "timetable": {"cls": "flatland.envs.timetable_generators.FileTimetableGenerator"}}

# grid: what scenario.pkl holds today, at step 10
env = make_env()
grail = {"width": W, "height": H, "transitions": env.rail.grid.tolist(), "level_free_positions": sorted(pos(p) for p in env.resource_map.level_free_positions)}
genv = GraphRailEnv.from_rail_env(env, DummyObservationBuilder(), seed=42)
run(env)
gens = dict(GENS, effects={"cls": type(env.effects_generator).__module__ + "." + type(env.effects_generator).__name__})
json.dump(doc(env, sl_grid, "grid", False, grail, gens), open(f"{OUT}/example-scenario-grid.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else repr(o))

# graph: the same scenario converted with from_rail_env, at step 10
rmap = genv.resource_map.resource_map if hasattr(genv.resource_map, "resource_map") else GraphRailEnv._grid_resource_map(make_env(), genv.rail.g)
g = genv.rail.g
nodes = {n: {"resource": rmap[n], "prohibited_actions": sorted(a.name for a in g.nodes[n].get("prohibited_actions", set()))} for n in sorted(g.nodes)}
edges = [{"from": u, "to": v, "actions": sorted(a.name for a in g[u][v].get("actions", set()))} for u, v in sorted(g.edges)]
def res(x):
    if isinstance(x, list) and len(x) == 2 and all(isinstance(i, int) for i in x): return str(tuple(x))
    if isinstance(x, list): return [res(i) for i in x]
    if isinstance(x, dict): return {k: (v if k in ("placement",) else res(v)) for k, v in x.items()}
    return x
sl_graph = res(sl_grid); sl_graph["binding"] = "graph"
run(genv)
json.dump(doc(genv, sl_graph, "graph", True, {"nodes": nodes, "edges": edges}, {k: v for k, v in gens.items() if k != "rail"}),
          open(f"{OUT}/example-scenario-graph.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else repr(o))

# ui-hub tutorial: the plan-b folder's draw.json, generator.json, config.json and scenario.pkl in one document, not started
env0 = make_env()
sl_ui = json.loads(json.dumps(sl_grid)); sl_ui["extensions_used"] = {"drawing-tool": None}
ext = {"drawing-tool": {"cellSize": gen["gridDimensions"]["cellSize"], "nextStationId": gen["nextStationId"],
                        "lines": gen["lines"], "timetables": gen["timetables"], "trainCategories": gen["trainCategories"],
                        "trains": {"0": "AB", "1": "BA"}},
       "flatland-scenarios": {"malfunctions": gen["malfunctions"]}}
json.dump(doc(env0, sl_ui, "grid", False, {"width": W, "height": H, "transitions": env0.rail.grid.tolist(), "level_free_positions": gen["overpasses"]},
              dict(GENS, effects={"cls": "flatland.envs.malfunction_effects_generators.MalfunctionEffectsGenerator"}), ext, with_state=False),
          open(f"{OUT}/example-scenario-ui-hub-tutorial.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else repr(o))
print("ok", env._elapsed_steps, [a.state_machine.state.name for a in env.agents], [a.current_entry_point for a in env.agents], [a.current_entry_point for a in genv.agents])
