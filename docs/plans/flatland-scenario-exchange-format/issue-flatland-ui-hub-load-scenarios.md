<!-- Draft of a flatland-ui-hub issue for step S-11 of flatland-rl's docs/plans/flatland-scenario-exchange-format.md.
Not created on GitHub yet. Title on the first heading line; the rest is the issue body. -->

# Load scenarios from flatland-rl's exchange format, `.pkl` as a cache

**Motivation**

flatland-ui-hub builds a session's env from `.pkl`s (`scenario.pkl`, `olten.pkl`, the ECML2026 pickles) and the
drawing tool's outputs (`draw.json`, `generator.json`, `config.json`) with its own code
(`app/core/experiment_scenarios.py`, `app/core/env_factory.py`). flatland-rl's exchange format holds a whole scenario
(configuration and state) in one document, read and written by flatland-rl.

**What changes**

- [ ] A scenario folder may hold a scenario document (`scenario.flatland.json`); the env is built from it with
  flatland-rl's reader, the geography computed from its `stations_links` by the geography adapter.
- [ ] `.pkl` read only as a cache: written next to the document after the first load, rebuilt when it is missing or
  older than the document.
- [ ] Folders without a document keep loading as today until they are converted (the conversion issue).
- [ ] The intro card (`scenario.json`, `scenario.schema.json`, its image) stays as it is: presentation data, not part
  of a scenario.
- [ ] Tests: a session built from the tutorial's document equals one built from its `scenario.pkl` (same rail,
  agents, timetable and stations; replaying actions gives the same trajectory); the geography equals today's.

**Depends on**

flatland-rl release with whole scenarios (#129); the geography adapter (draft: flatland-book's
[`issue-flatland-ui-hub-geography-adapter.md`](https://github.com/flatland-association/flatland-book/blob/22f74b8/docs/plans/stations-links-logical-model/issue-flatland-ui-hub-geography-adapter.md)).
