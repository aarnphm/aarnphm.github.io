# Local Obsidian plugin work

Use each plugin's existing package and manifest. The local packages use npm with esbuild and a `package-lock.json`; this is a scoped exception to Garden's root pnpm workflow. Use existing dependencies and the Node version supported by the package. Do not install global lint tools.

`npm run build` performs the package's TypeScript check and emits its plugin bundle. It does not build Quartz. These folders are inside a vault, so a bundle write may affect the installed plugin; inspect existing watch processes and the requested validation scope first. `npm run dev` starts a watcher. Avoid starting a duplicate or reloading the app without need.

Keep the entrypoint focused on lifecycle management and use the existing module layout. Register event, DOM, and interval cleanup with Obsidian's registration helpers, and clean up views and resources on unload. Test repeated load/unload when lifecycle behavior changes.

Preserve released plugin IDs, command IDs, settings compatibility, and an accurate `minAppVersion`. Check `isDesktopOnly` before using desktop APIs. Keep vault data local unless an explicitly requested feature requires a disclosed external operation.

Release artifacts are `main.js`, `manifest.json`, and optional `styles.css` at the plugin's root. Follow the repository's existing tracking policy. Do not run `npm run version` as a validation command: it changes version files and stages them.

Use the [Obsidian API documentation](https://docs.obsidian.md) for uncertain interfaces and the [developer policies](https://docs.obsidian.md/Developer+policies) for a release review.
