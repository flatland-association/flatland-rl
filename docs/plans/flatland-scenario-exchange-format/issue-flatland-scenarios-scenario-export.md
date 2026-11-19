<!-- Draft of a flatland-scenarios issue for step S-8 of docs/plans/flatland-scenario-exchange-format.md in flatland-rl.
Not created on GitHub yet. Title on the first heading line; the rest is the issue body. -->

# Drawing tool: export whole scenarios in flatland-rl's exchange format

**Motivation**

The drawing tool writes its scenarios in its own shape, and `scenario_generator` converts them per export generation
(`stations_to_stations_links`). The exchange format covers whole scenarios (flatland-rl #129), so the export becomes
one scenario document, in one switch with no in-between shape.

**What changes**

- [ ] The drawing tool writes and reads a scenario document, valid against flatland-rl's scenario schema for the grid binding (`scenario.grid.schema.json`): grid with
  level-free positions, lines and timetables as configuration, and `stations_links` with `id`, `name`, `placement`
  with the rectangle's corners as `outline`, `area` (the rail inside the rectangle), stopping points and the gates as
  the tool forms them today; links and gate links empty. Gates by the corridor rule, complete pins and links follow
  with flatland-book's draft ["Drawing tool: gates by the corridor rule, complete pins,
  links"](https://github.com/flatland-association/flatland-book/blob/22f74b8/docs/plans/stations-links-logical-model/issue-flatland-scenarios-drawing-tool-gates.md).
- [ ] `scenario_generator` builds envs from it with flatland-rl's reader and derives `agents_hints`;
  `stations_to_stations_links` and its per-format branches go away.
- [ ] The fixtures, converted by the converter issue's script, in the same PR.
- [ ] Tests: export → load → export is unchanged; converted fixtures build the same envs (the assertions on `edges`
  move to `area`, since flatland-book's station fields issue removes `edges`).
- [ ] Bump `env.flatland-rl-ref`.

**Depends on**

flatland-rl release with whole scenarios (#129); the converter for older exports (one PR, the converter first).
