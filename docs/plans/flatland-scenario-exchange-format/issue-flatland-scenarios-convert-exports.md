<!-- Draft of a flatland-scenarios issue for step S-7 of docs/plans/flatland-scenario-exchange-format.md in flatland-rl.
Not created on GitHub yet. Title on the first heading line; the rest is the issue body. -->

# Convert older drawing tool exports to flatland-rl's exchange format

**Motivation**

The exchange format replaces the drawing tool's export format. The tool and `scenario_generator` then read only the
new format; older exports need a one-off conversion, outside both.

**What changes**

- [ ] A command-line script (e.g. `scenario_generator/convert_export.py`): reads an export of any older generation
  (flat per-cell `{id, r, c}`, `station_cells`, PR #39's `stations` list) and writes a scenario document of the
  exchange format, through flatland-rl's reader and writer: the grid with level-free positions, lines and timetables
  as configuration; for the stations the rectangle becomes `placement.outline`, `edges` becomes `area`
  (through tracks added), the colour a `drawing-tool` extension.
- [ ] The stored fixtures converted with it, in one PR with the whole export issue, this issue's commits first.
- [ ] Tests: every older fixture converts, validates against the scenario schema for the grid binding (`scenario.grid.schema.json`) and builds the same env as before.

**Depends on**

flatland-rl release with whole scenarios (#129); the `env.flatland-rl-ref` bump.
