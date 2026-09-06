# obsidian-metadata

This directory is a local Obsidian plugin. Read [plugin workflow](../../../../docs/agent-obsidian-plugins.md) when changing its code, lifecycle, or release configuration.

- Use this package's npm scripts and existing esbuild configuration. `npm run build` is the scoped TypeScript-and-plugin-bundle check; it is distinct from a Quartz site build.
- Preserve existing watchers and inspect the effect of writing the installed plugin bundle before testing it in the vault.
- Keep plugin and command IDs stable. Register resources for unload cleanup and preserve the manifest's platform and API-version requirements.
- `npm run version` mutates and stages release files; use it only for an authorized release task.
