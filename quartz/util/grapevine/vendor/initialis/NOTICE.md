# INITIĀLIS, vendored

The ornament engine of [azlen/initialis](https://github.com/azlen/initialis) by Azlen Elza, MIT
(see LICENSE), at commit `6cf93f7d3d2922efdbeb86cafa27f39758be9398` (2026-10-30).

Taken from `render/src/`: `ornament.js`, `field.js`, `shapes.js`, `util.js`, and the `kinds/leaf`
and `kinds/flower` families, unchanged except that the `?v=a8136959` cache stamp is stripped from
their imports and `ornament.js` reads its 4000-unit stem cap from an optional `maxLen` option. `kinds.js` is a static registry written for this build in place of the upstream
on-demand loader (it imports `params.js` and fetches modules by URL). The letter raster, the
settings and the SVG output around the engine are the garden's own (`../../*.ts`); the upstream
ink pass (WebGL2) is not used.

To refresh: copy the same files from a newer commit, strip the stamp again, update the hash here.
