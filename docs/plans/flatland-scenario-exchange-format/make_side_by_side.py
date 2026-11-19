"""Side-by-side: flatland-ui-hub scenario files vs. the exchange format, boxes around 1:1 parts.

Usage: python make_side_by_side.py <out.html> <flatland-ui-hub plan-b/tutorial/generator.json> <example-scenario-ui-hub-tutorial.json>
The other excerpts are written out below, taken from flatland-ui-hub's fixtures and flatland-book's example-ui-hub-*.json.
"""
import json, html, sys
OUT = sys.argv[1]

class B:  # boxed value, number n
    def __init__(s, n, v): s.n, s.v = n, v
class E:  # elision
    def __init__(s, t="…"): s.t = t
class C:  # comment after a value
    def __init__(s, v, t): s.v, s.t = v, t

COLORS = ["#e8590c", "#1c7ed6", "#2f9e44", "#ae3ec9", "#f08c00", "#0c8599", "#c2255c", "#5c940d", "#3b5bdb", "#d9480f", "#087f5b", "#862e9c"]

def flat(v):
    if isinstance(v, B): return flat(v.v)
    if isinstance(v, C): return flat(v.v)
    if isinstance(v, E): return v.t
    if isinstance(v, dict): return "{" + ", ".join(json.dumps(k, ensure_ascii=False) + ": " + flat(x) for k, x in v.items()) + "}"
    if isinstance(v, list): return "[" + ", ".join(flat(x) for x in v) + "]"
    return json.dumps(v, ensure_ascii=False)

def has_box(v):
    if isinstance(v, B): return True
    if isinstance(v, C): return has_box(v.v)
    if isinstance(v, dict): return any(has_box(x) for x in v.values())
    if isinstance(v, list): return any(has_box(x) for x in v)
    return False

def J(items):
    out = []
    for i, it in enumerate(items):
        c = "," if i < len(items) - 1 else ""
        if "\x00" in it:
            a, b = it.rsplit("\x00", 1); out.append(a + c + b)
        else:
            out.append(it + c)
    return "\n".join(out)

def r(v, ind=0, width=78):
    """returns html"""
    if isinstance(v, B):
        c = COLORS[(v.n - 1) % len(COLORS)]
        return f'<span class="box" style="border-color:{c}"><sup style="background:{c}">{v.n}</sup>{r(v.v, ind, 10**6)}</span>'
    if isinstance(v, C):
        return r(v.v, ind, width) + f'\x00<span class="cm">  // {html.escape(v.t)}</span>'
    if isinstance(v, E):
        return f'<span class="el">{html.escape(v.t)}</span>'
    if isinstance(v, (dict, list)):
        f = flat(v)
        if len(f) + ind <= width and not any(isinstance(x, C) for x in (v.values() if isinstance(v, dict) else v)):
            if isinstance(v, dict):
                return "{" + ", ".join(html.escape(json.dumps(k, ensure_ascii=False)) + ": " + r(x, ind, width) for k, x in v.items()) + "}"
            return "[" + ", ".join(r(x, ind, width) for x in v) + "]"
        pad = " " * (ind + 1)
        if isinstance(v, dict):
            items = [pad + html.escape(json.dumps(k, ensure_ascii=False)) + ": " + r(x, ind + 1, width) for k, x in v.items()]
            return "{\n" + J(items) + "\n" + " " * ind + "}"
        items = [pad + r(x, ind + 1, width) for x in v]
        return "[\n" + J(items) + "\n" + " " * ind + "]"
    return html.escape(json.dumps(v, ensure_ascii=False))

def panel(title, left, right, notes):
    def col(files):
        return "".join(f'<div class="file"><div class="fn">{html.escape(n)}</div><pre>{r(v)}</pre></div>' for n, v in files)
    return (f'<section><h2>{html.escape(title)}</h2><div class="cols"><div><h3>flatland-ui-hub</h3>{col(left)}</div>'
            f'<div><h3>exchange format</h3>{col(right)}</div></div><ul class="notes">' + "".join(f"<li>{html.escape(n)}</li>" for n in notes) + "</ul></section>")

P = []
# ---------------- Olten
P.append(panel("Olten network (fixtures/olten/)",
 [("olten.geography.json", {"layout": C("network", "derived from the links"), "stations": [
    {"code": B(1, "OL"), "name": B(2, "Olten"), "track": B(3, 1), "cell": B(4, [37, 7]), "kind": C("platform", "→ station: kind absent"), "basis": B(5, "notebook variable olten1")},
    {"code": C(B(1, "OL"), "same station, next row"), "name": "Olten", "track": B(6, 2), "cell": B(7, [37, 8]), "kind": "platform", "basis": B(8, "notebook variable olten2")}, E("… 30 more rows"),
    {"code": B(9, "P-BASEL"), "name": B(10, "Basel"), "track": None, "cell": B(11, [0, 23]), "kind": B(12, "portal"), "via": B(13, "Hauenstein-Basistunnel"), "basis": B(14, "notebook variable end10north; …")}, E()],
   "locations": [], "single_track": C([], "derived from the links"), "_comment": B(15, "Names for the Olten network …")}),
  ("olten.pkl", C(E("RailEnv, pickled"), "→ configuration.rail and state: see the plan-b panel"))],
 [("stations_links (book example-ui-hub-olten.json)", {"version": "2.0.0", "binding": "grid", "extensions_used": {"flatland-ui-hub": None}, "stations": {
    "OL": {"name": B(1, "OL"), "display_name": B(2, "Olten"), "gates": {}, "stopping_points": [
       {"name": C("OL.0", "derived: <station>.<index>"), "node": B(4, [37, 7]), "extensions": {"flatland-ui-hub": {"track": B(3, 1), "basis": B(5, "notebook variable olten1")}}},
       {"name": "OL.1", "node": B(7, [37, 8]), "extensions": {"flatland-ui-hub": {"track": B(6, 2), "basis": B(8, "notebook variable olten2")}}}, E()]},
    "P-BASEL": {"name": B(9, "P-BASEL"), "display_name": B(10, "Basel"), "kind": B(12, "portal"), "gates": {}, "stopping_points": [
       {"name": "P-BASEL.0", "node": B(11, [0, 23]), "extensions": {"flatland-ui-hub": {"basis": B(14, "notebook variable end10north; …")}}}, E()],
       "extensions": {"flatland-ui-hub": {"via": B(13, "Hauenstein-Basistunnel")}}}, "…": E()},
   "links": C([], "from olten.pkl's rail, once derived"), "gate_links": [], "extensions": {"flatland-ui-hub": {"comment": B(15, "Names for the Olten network …")}}})],
 ["Rows are grouped by code: one station per code, one stopping point per row, in row order.",
  "Not 1:1: kind platform (station, the default), layout and single_track (derived from the links), stopping point names (derived)."]))
# ---------------- PF-CH
P.append(panel("PF–CH corridor (fixtures/pf_ch/*.scene.json, *.plan.json)",
 [("*.scene.json", {"id": "scene_pf_ch_corridor_stops", "grid": {"width": B(1, 191), "height": B(2, 9), "cellSizePx": B(3, 28)},
   "cells": C([{"id": "cell_87_0", "x": 87, "y": 0, "kind": "station", "connections": ["E", "W"], "stationId": "st_WN_1"}, E("… 548 cells")], "→ rail transitions: converted, not 1:1"),
   "graph": C({"nodes": [E("548")], "edges": [E("617")]}, "a second form of the same rail: dropped"),
   "stations": [{"id": C("st_WN_1", "derived: st_<code>_<track>"), "name": C("WN 1", "derived: <code> <track>"), "x": B(5, 87), "y": B(4, 0), "metadata": {"station": C("Weesen", "spelling; display name from STATION_NAMES[code]"), "code": B(6, "WN"), "track": B(7, 1)}},
                {"id": "st_WN_2", "name": "WN 2", "x": B(9, 87), "y": B(8, 2), "metadata": {"station": "Weesen", "code": C(B(6, "WN"), "same station"), "track": B(10, 2)}}, E()],
   "agents": [{"id": "train_1", "name": B(11, "T01"), "start": {"x": B(13, 17), "y": B(12, 0)}, "target": {"x": B(15, 71), "y": B(14, 8)}, "speed": B(16, 1.0),
               "metadata": {"from": "SIB", "fromTrack": 1, "to": "ZB", "toTrack": 5}, "via": [{"x": B(18, 28), "y": B(17, 2), "station": "Schuebelbach-Butt.", "code": "SCBU"}, E()]}, E()],
   "metadata": {"locationOrder": C(["…", "WN", "MH", "TIEF", "MG", "…"], "order along the line: derived from positions"), "locationColumns": {"MH": B(19, 95), "…": E()},
                "tracksPerStation": C({"WN": 3, "…": E()}, "derived: number of stopping points"), "singleTrackSection": C(["MH", "TIEF"], "derived from the links")}}),
  ("*.plan.json", C(E("each train's route and timing"), "→ lines and timetables: per train, not shown"))],
 [("scenario document (shape of Q-16)", {"format": "flatland-scenario", "version": "1.0.0", "binding": "grid", "configuration": {
   "rail": {"width": B(1, 191), "height": B(2, 9), "transitions": C([E("9 rows of 191 numbers")], "from cells[].connections")},
   "stations_links": {"version": "2.0.0", "binding": "grid", "stations": {
      "WN": {"name": B(6, "WN"), "display_name": C("Weesen", "STATION_NAMES['WN']"), "gates": {}, "stopping_points": [
          {"name": "WN.0", "node": [B(4, 0), B(5, 87)], "extensions": {"flatland-ui-hub": {"track": B(7, 1)}}},
          {"name": "WN.1", "node": [B(8, 2), B(9, 87)], "extensions": {"flatland-ui-hub": {"track": B(10, 2)}}}, E()],
          "placement": C({"position": [2, 87], "orientation": 90.0}, "derived: middle platform row, column")},
      "MH": {"name": "MH", "display_name": "Mühlehorn", "gates": {}, "stopping_points": [], "placement": {"position": [C(1, "middle row"), B(19, 95)], "orientation": 90.0}}, "…": E()}},
   "lines": [{"handle": 0, "waypoints": [[{"position": [B(12, 0), B(13, 17)], "direction": C(1, "derived from the rail")}], [{"position": [B(17, 2), B(18, 28)], "direction": E()}], E(), [{"position": [B(14, 8), B(15, 71)], "direction": None}]],
              "max_speed": B(16, "1"), "earliest_departures": C(E(), "from *.plan.json"), "latest_arrivals": E()}, E()],
   "extensions": {"flatland-ui-hub": {"cellSizePx": B(3, 28), "trains": {"0": B(11, "T01")}}}}})],
 ["Scene x, y become [row, column] = [y, x]: the same numbers, swapped.",
  "Not 1:1: the rail (cells and graph are converted to transitions), display names (looked up), placement, the order along the line, tracks per station and the single-track section (derived)."]))
# ---------------- Gotthard
P.append(panel("Gotthard (fixtures/gotthard/gotthard.topology.json, gotthard.traffic.json)",
 [("gotthard.topology.json", {"id": "gotthard", "version": 1, "generated": "2026-10-04", "scope": "stage 1: Arth-Goldau .. Bellinzona",
   "portals": C({"north": "Arth-Goldau", "south": "Bellinzona"}, "rest of the scenario"), "speedAssumptions": C(E(), "rest of the scenario"),
   "operatingPoints": [{"id": B(1, "GD"), "name": B(2, "Arth-Goldau"), "kind": C("station", "→ kind absent"), "kmBG": B(3, 8.864), "lat": B(4, 47.04904), "lon": B(5, 8.54945), "basis": B(6, "sbb-open-data:linie-mit-betriebspunkten"), "route": B(7, "north")},
                       {"id": B(8, "STNS"), "name": B(9, "Steinen Nord (Spw)"), "kind": B(10, "crossover"), "kmBG": B(11, 13.223), "lat": B(12, 47.05045), "lon": B(13, 8.60009), "basis": "sbb-open-data:…", "route": "north"}, E("… 44 more")],
   "sections": C([{"id": "GD-STNS", "from": "GD", "to": "STNS", "km": 4.359, "tracks": 2}, E()], "→ links with 2 fibres, once there is rail"),
   "routes": C(E(), "rest of the scenario"), "crossovers": C(E(), "rest of the scenario")}),
  ("gotthard.traffic.json", C(E("trips of one service day, synthetic freight"), "→ lines and timetables, once there is rail"))],
 [("stations_links (book example-ui-hub-gotthard.json)", {"version": "2.0.0", "binding": "graph-geo", "extensions_used": {"flatland-ui-hub": None}, "stations": {
   "GD": {"name": B(1, "GD"), "display_name": B(2, "Arth-Goldau"), "gates": {}, "stopping_points": [], "placement": {"position": [B(4, 47.04904), B(5, 8.54945)], "orientation": C(87.6, "derived: bearing along its route")},
          "extensions": {"flatland-ui-hub": {"kmBG": B(3, 8.864), "route": B(7, "north"), "basis": B(6, "sbb-open-data:linie-mit-betriebspunkten")}}},
   "STNS": {"name": B(8, "STNS"), "display_name": B(9, "Steinen Nord (Spw)"), "kind": B(10, "crossover"), "gates": {}, "stopping_points": [],
          "placement": {"position": [B(12, 47.05045), B(13, 8.60009)], "orientation": 92.0}, "extensions": {"flatland-ui-hub": {"kmBG": B(11, 13.223), "route": "north", "basis": "sbb-open-data:…"}}}, "…": E()},
   "links": C([], "none until a graph is built"), "gate_links": []})],
 ["No rail yet: stations without stopping points, area or gates; kind loop becomes flatland-ui-hub:loop.",
  "Not 1:1: orientation (derived), sections, routes, portals, speed assumptions and the traffic file (rest of the scenario, flatland-ui-hub's until there is rail)."]))
# ---------------- plan-b tutorial
gen = json.load(open(sys.argv[2]))
doc = json.load(open(sys.argv[3]))
row2 = gen["grid"][2]
P.append(panel("plan-b experiment folder (fixtures/plan-b/tutorial/)",
 [("generator.json (= draw.json + malfunctions)", {"gridDimensions": {"rows": B(1, 6), "cols": B(2, 25), "cellSize": B(3, 30)},
   "grid": [E("row 0"), E("row 1"), B(4, row2), E("rows 3–5")], "overpasses": B(5, []),
   "stations": [{"id": B(6, 1), "name": B(7, "A"), "city_position": B(8, [2, 3]), "city_orientation": C(1, "× 90 → orientation"),
      "stopping_points": B(9, [{"name": "A.0", "node": [2, 3]}]), "edges": B(10, [[2, 2], [2, 3], [2, 4]]),
      "gates": B(11, {"E": {"name": "A.E", "pins": {"0": {"name": "A.E.0", "node": [2, 4]}}}, "W": {"name": "A.W", "pins": {"0": {"name": "A.W.0", "node": [2, 2]}}}}),
      "area": C([1, 2, 3, 4], "→ outline: its corners")}, E("station B")], "nextStationId": B(12, 3),
   "lines": B(13, [{"id": 1794084461089, "name": "A-B", "stationIds": [1, 2], "…": E()}, E()]),
   "timetables": B(14, [{"id": 1794084502192, "name": "AB", "lineId": 1794084461089, "…": E()}, E()]),
   "trainCategories": B(15, {"IC": {"id": 1, "speed": 1}, "…": E()}),
   "flatlandLine": {"agent_positions": [[[B(16, [2, 3])], [B(17, [2, 21])]], E()], "agent_directions": [[[B(18, 1)]], E()], "agent_targets": C(E(), "= last waypoints"), "agent_speeds": [B(19, 1), E()]},
   "flatlandTimetable": {"earliest_departures": [[B(20, 0), C(22, "at the target: none")], E()], "latest_arrivals": [B(21, [None, 18]), E()], "max_episode_steps": B(22, 40)},
   "malfunctions": B(23, {"scheduled": [{"train": "AB", "step": 8, "duration": 4}]})}),
  ("config.json (where a folder has one, e.g. example-sm)", {"malfunctions": C({"scheduled": [{"train": "IC 1.1", "step": 12, "duration": 15}]}, "→ extensions.flatland-scenarios.malfunctions"), "seed": C(42, "→ configuration.seed")}),
  ("scenario.pkl", C(E("RailEnv, pickled"), "the same configuration, plus the state of a started env")),
  ("scenario.json, scenario.schema.json, scenario.png", C({"id": "tutorial", "title": "Tutorial", "description": "…", "seconds_per_step": 5, "locked": True}, "kept: flatland-ui-hub's card, not in the format"))],
 [("example-scenario-ui-hub-tutorial.json", {"format": "flatland-scenario", "version": "1.0.0", "binding": "grid", "extensions_used": {"drawing-tool": None, "flatland-scenarios": None}, "configuration": {
   "rail": {"width": B(2, 25), "height": B(1, 6), "transitions": [E("row 0"), E("row 1"), B(4, row2), E("rows 3–5")], "level_free_positions": []},
   "stations_links": {"version": "2.0.0", "binding": "grid", "extensions_used": {"drawing-tool": None}, "stations": {
      "A": {"name": B(7, "A"), "id": B(6, 1), "area": B(10, [[2, 2], [2, 3], [2, 4]]), "gates": B(11, {"E": {"name": "A.E", "pins": {"0": {"name": "A.E.0", "node": [2, 4]}}}, "W": {"name": "A.W", "pins": {"0": {"name": "A.W.0", "node": [2, 2]}}}}),
            "stopping_points": B(9, [{"name": "A.0", "node": [2, 3]}]), "placement": {"position": B(8, [2, 3]), "orientation": 90.0, "outline": [[1, 2], [1, 4], [3, 4], [3, 2]]}}, "B": E()},
      "links": C([E("A.E → B.W")], "derived from the rail"), "gate_links": C([E()], "derived from the rail")},
   "lines": [{"handle": 0, "waypoints": [[{"position": B(16, [2, 3]), "direction": B(18, 1)}], [{"position": B(17, [2, 21]), "direction": 1}, {"position": [2, 21], "direction": 3}]],
              "earliest_departures": [B(20, 0), None], "latest_arrivals": B(21, [None, 18]), "max_speed": B(19, "1")}, E()],
   "max_episode_steps": B(22, 40), "seed": C(42, "config.json's seed"), "generators": E(),
   "extensions": {"drawing-tool": {"cellSize": B(3, 30), "nextStationId": B(12, 3), "overpasses": B(5, []), "lines": B(13, E("as in generator.json")), "timetables": B(14, E("as in generator.json")),
                  "trainCategories": B(15, E("as in generator.json")), "trains": {"0": "AB", "1": "BA"}}, "flatland-scenarios": {"malfunctions": B(23, {"scheduled": [{"train": "AB", "step": 8, "duration": 4}]})}}},
   "state": C({}, "not started; a started env (scenario.pkl) fills it, see example-scenario-grid.json")})],
 ["draw.json, generator.json, config.json and scenario.pkl become one document; the intro card stays.",
  "Not 1:1: orientation (× 90), the rectangle (→ outline corners), links and gate links (derived), the target's waypoint directions (derived)."]))
# ---------------- derived geography
P.append(panel("Derived geography (_stations_links_geography, from scenario.pkl; replaced by the adapter)",
 [("GET /hmi/geography for plan-b/tutorial", {"layout": C("corridor", "derived from the links"), "stations": [
   {"code": B(1, "A"), "name": C(B(2, "A"), "display_name, else name"), "track": C(1, "index + 1"), "cell": B(3, [2, 3]), "kind": C("platform", "kind absent: station")},
   {"code": B(4, "B"), "name": B(5, "B"), "track": 1, "cell": B(6, [2, 21]), "kind": "platform"}], "locations": [], "single_track": C([], "derived from the links")})],
 [("stations_links", {"stations": {"A": {"name": B(1, "A"), "display_name": C(B(2, None), "absent"), "stopping_points": [{"name": "A.0", "node": B(3, [2, 3])}], "id": 1, "area": E(), "gates": E(), "placement": E()},
   "B": {"name": B(4, "B"), "stopping_points": [{"name": "B.0", "node": B(6, [2, 21])}], "…": E()}}, "links": E(), "gate_links": E()})],
 ["The adapter computes the geography from the stored document (Q-7 b)); everything unboxed on the right stays in the document.",
  "ECML2026 (fixtures/ecml2026/*.pkl): converted like any .pkl (plan-b panel); not opened here."]))

CSS = """body{font-family:system-ui,sans-serif;margin:16px;background:#fff;color:#222}h1{font-size:20px}h2{font-size:17px;margin:28px 0 6px;border-top:2px solid #ccc;padding-top:10px}
h3{font-size:13px;color:#666;margin:4px 0}.cols{display:grid;grid-template-columns:max-content max-content;gap:14px}.file{margin-bottom:10px}.fn{font:12px monospace;background:#f1f3f5;padding:3px 6px;border-radius:3px 3px 0 0}
pre{font:12px/1.55 ui-monospace,Menlo,monospace;margin:0;padding:6px;border:1px solid #e9ecef;white-space:pre}
.box{border:2px solid;border-radius:3px;padding:0 2px;position:relative}.box sup{color:#fff;font:bold 9px sans-serif;padding:0 3px;border-radius:6px;margin-right:2px;vertical-align:top}
.cm{color:#868e96;font-style:italic}.el{color:#adb5bd}.notes{font-size:13px;color:#444}"""
open(OUT, "w").write((f"<!doctype html><meta charset=utf-8><title>flatland-ui-hub vs exchange format</title><style>{CSS}</style><h1>flatland-ui-hub scenario files and the exchange format</h1>"
 "<p>Boxes with the same number hold the same value on both sides (1:1, possibly reshaped: moved, grouped, or renamed as a key). Grey comments mark what is derived, converted or not in the format.</p>" + "".join(P)).replace("\x00", ""))
print("ok")
