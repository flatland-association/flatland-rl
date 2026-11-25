<!-- Draft of a flatland-rl issue for step S-5 of docs/plans/flatland-scenario-exchange-format.md in flatland-rl.
Not created on GitHub yet. Title on the first heading line; the rest is the issue body. -->

# Scenario configuration in the exchange format: rail, stations, lines, timetables, generators

**Motivation**

A scenario's configuration (what builds the env) has no portable form: `RailEnvPersister` pickles objects. #129 asks
for configuration and state apart; generators need a generic serialization by class name and state first (#242, on
the base classes of #145).

**What changes**

- [ ] Rail, line, timetable and effects generators serialize by class name and state (#242).
- [ ] The configuration part of the exchange format: the rail (grid with level-free positions, or graph),
  `stations_links`, lines and timetables, the generators by class name (with their parameters when the scenario is to be regenerated), `max_episode_steps`,
  the seed and `skip_state_machine_update`; the schema of the configuration, reusing the stations and links schema by
  `$ref`.
- [ ] Tests: configuration → env → configuration is unchanged for `sparse_rail_generator` envs and the ECML2026
  scenarios.

**Depends on**

#145, #242; the JSON form of `StationsLinks`.

**Related**

#129 (the state part and the hops `RailEnv`, `GraphRailEnv`, `.pkl` ↔ format follow there).
