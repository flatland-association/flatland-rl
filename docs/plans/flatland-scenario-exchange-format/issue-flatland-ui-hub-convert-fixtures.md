<!-- Draft of a flatland-ui-hub issue for step S-12 of flatland-rl's docs/plans/flatland-scenario-exchange-format.md.
Not created on GitHub yet. Title on the first heading line; the rest is the issue body. -->

# Convert the stored scenarios to flatland-rl's exchange format

**Motivation**

flatland-ui-hub keeps scenarios in several formats of its own. Once it loads the exchange format, each stored scenario
is converted once, and the conversion is checked to lose nothing before the old files go.

**What changes**

One converter per stored format (`backend/scripts/`), each with a check file → exchange → file that gives the same
file, apart from the fields that are derived in the exchange format:

- [ ] Experiment folders (`backend/app/fixtures/plan-b/*/`): `draw.json`, `generator.json`, `config.json` and
  `scenario.pkl` → one scenario document; the drawing tool's own data (lines and timetables by name, train categories,
  cell size, overpasses) as `drawing-tool` extension, scheduled malfunctions as `flatland-scenarios` extension.
  Derived: the station's orientation (× 90), its rectangle (→ outline corners), links and gate links.
- [ ] The Olten network (`backend/app/fixtures/olten/`): `olten.geography.json` and `olten.pkl` → one scenario
  document; rows grouped by `code`, `track`, `basis` and `via` as `flatland-ui-hub` extension. Derived: `layout`,
  `single_track`, stopping point names.
- [ ] The PF–CH scenes (`backend/app/fixtures/pf_ch/`): `*.scene.json` and `*.plan.json` → scenario documents; scene
  `x, y` as `[row, column]`; `track` as extension; places without tracks as stations without stopping points. Derived:
  the display name (`STATION_NAMES`), placement, the order along the line, tracks per station, the single-track
  section; the rail from `cells` (the scene's `graph` is the same rail).
- [ ] The Gotthard topology (`backend/app/fixtures/gotthard/`): operating points → a `stations_links` document of
  binding `graph-geo` (no rail yet); `kmBG`, `kmGBT`, `route`, `basis` as extension. `sections`, `routes`, `portals`,
  `crossovers` and `gotthard.traffic.json` stay flatland-ui-hub's until there is rail.
- [ ] ECML2026 (`backend/app/fixtures/ecml2026/`): the pickles → scenario documents, as any `.pkl`.
- [ ] The old files removed once their check passes; `.pkl` written again as a cache.

**Depends on**

Loading scenarios from the exchange format.

**Related**

- The side-by-side of these files and the format: flatland-rl's `docs/plans/flatland-scenario-exchange-format/side-by-side.html`.
