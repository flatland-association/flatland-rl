<!-- Draft of a flatland-ui-hub issue for step S-10 of flatland-rl's docs/plans/flatland-scenario-exchange-format.md.
Not created on GitHub yet. Title on the first heading line; the rest is the issue body. -->

# TypeScript types for the exchange format, generated from flatland-rl's schemas

**Motivation**

The frontend reads stations by hand-written types (`models.ts`). flatland-rl's JSON schemas are the contract of the
exchange format; types generated from them stay in step.

**What changes**

- [ ] Generate TypeScript types from the schemas of the pinned flatland-rl version (e.g. `json-schema-to-typescript`),
  in the frontend build or as a checked-in file with a check that it is current.
- [ ] Use them where the frontend reads stations.

**Depends on**

flatland-rl release with the JSON schemas; the scenario types once flatland-rl #129 is released.
