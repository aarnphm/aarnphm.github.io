# alembic-ornaments, vendored

The source of [lauriparonen/alembic-ornaments](https://github.com/lauriparonen/alembic-ornaments)
(`@lauriparonen/alembic-ornaments`) by Lauri Paronen, MIT (see LICENSE), at commit
`ec6f48f574e8a1b43b184bfeddd0c4cea140c455` (2026-10-30).

Taken from `src/`: `index.ts` and `internal/*.ts`, unchanged except that `vine()` and
`composePreset()` in `internal/presets.ts` take an optional band width (upstream fixes it at 600).
The React components (`src/react.tsx`) are not used. The garden calls the internals directly for
the chapter rule (`../../index.ts`): `composePreset('vine', rng, width)` through `render()`.

To refresh: copy the same files from a newer commit, reapply the width parameter, update the hash here.
