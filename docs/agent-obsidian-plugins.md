# Local Obsidian plugin work

Use each plugin's existing package and manifest. The local packages use npm with esbuild and a `package-lock.json`; this is a scoped exception to Garden's root pnpm workflow. Use existing dependencies and the Node version supported by the package. Do not install global lint tools.

`npm run build` performs the package's TypeScript check and emits its plugin bundle. It does not build Quartz. These folders are inside a vault, so a bundle write may affect the installed plugin; inspect existing watch processes and the requested validation scope first. `npm run dev` starts a watcher. Avoid starting a duplicate or reloading the app without need.

Keep the entrypoint focused on lifecycle management and use the existing module layout. Register event, DOM, and interval cleanup with Obsidian's registration helpers, and clean up views and resources on unload. Test repeated load/unload when lifecycle behavior changes.

Preserve released plugin IDs, command IDs, settings compatibility, and an accurate `minAppVersion`. Check `isDesktopOnly` before using desktop APIs. Keep vault data local unless an explicitly requested feature requires a disclosed external operation.

Release artifacts are `main.js`, `manifest.json`, and optional `styles.css` at the plugin's root. Follow the repository's existing tracking policy. Do not run `npm run version` as a validation command: it changes version files and stages them.

Use the [Obsidian API documentation](https://docs.obsidian.md) for uncertain interfaces and the [developer policies](https://docs.obsidian.md/Developer+policies) for a release review.

## Garden publishing and authoring

Garden's published Markdown behavior follows the [current compatibility profile](../quartz/compatibility/README.md). Local Obsidian plugin UI and Garden page output have separate acceptance. The original native observations use Obsidian 1.15.0 and remain limited to the experiments in [the native record](../quartz/compatibility/native-observations.md).

Preserve authored wiki-reference bytes and source paths. Ordinary references, embeds, heading paths, block IDs, aliases and settings enter shared syntax. Deliberate edits use `editReference`/`editWikilink`; derived route or provider interpretation must leave the authored export intact. In GFM tables, escaped label/settings separators retain their source coordinates. Code, escaped literals and default block HTML retain their syntax owners. See [the syntax contract](../quartz/compatibility/phase-1-contract.md) and [context/source contract](../quartz/compatibility/phase-2-contract.md).

YAML metadata is parsed as typed YAML before reference interpretation. Quote a wiki-looking scalar, for example `image: "[[cover.png]]"`; an unquoted flow sequence is diagnosed instead of reconstructed. Arrays and nested objects retain their authored shapes. Rich labels/descriptions use safe inline phrasing and plain projections; authored HTML in a label remains text. Garden formatted labels intentionally differ from the native literal-label example.

Tags are case-insensitive lowercase NFC identities while visible labels retain authored spelling. Nested tags share membership with their parents. Canonical tag pages and authored tag descriptions publish lowercase paths; historical spellings currently present in source receive redirect rules. A removed spelling has no persistent route-history guarantee. [The tag contract](../quartz/compatibility/tag-identity-contract.md) records Unicode, inline boundaries and search behavior.

Highlights support formatted content and the six leading color markers `🔴`, `🟠`, `🟡`, `🟢`, `🔵`, `🟣`. Tasks retain arbitrary single-character completion markers as source metadata. Garden uses strict CommonMark soft/hard breaks, KaTeX's supported TeX subset and Pele's documented diagram profile. Comments opened inline across blank paragraphs retain Garden's paired suppression. [The dialect contract](../quartz/compatibility/dialect-capabilities.md) records exact behavior, engine limits and intentional native differences.

Published query fences use the [declared public-source query profile](../quartz/compatibility/search-capabilities.md). Public Markdown/Canvas source can be searched; protected/private/draft/noindex/unpublished/global-excluded pages are filtered first. Unsupported operators and bounds produce visible errors. Asset-only filesystem search and native interactive sort/export controls are outside this profile. Base formulas/functions/views retain their [own capability inventory](../quartz/compatibility/base-capabilities.md); Canvas keeps Garden's richer previews.

Repeated embeds render separate occurrences. Ordered/loose list selections retain their structure, footnotes/citations travel with selections and occurrence IDs keep local relationships distinct. Cycles and work limits yield source-link fallbacks. Heading/block source paths and published notebook/Base/deck routes remain separate identities. [The renderer contract](../quartz/compatibility/phase-4-contract.md) and [architecture](obsidian-compatibility/architecture.md) define those boundaries.

Actual NotebookViewer code-cell links retain generated one-based IDs such as `#cell-1`, counting code cells independently of Markdown cells. Raw nbformat `cell.id` is not the published cell address. Generic flat-parser native-ID fixtures are a separate boundary; no renamed IDs or aliases are added.

Reference changes need real source-to-output fixtures and relevant existing domain tests. Served acceptance additionally requires the owning watcher's matching `build:ready`, exact-route HTTP success and fresh browser observations. Parser success does not establish native UI, media playback, provider behavior, hydration or deployment. Preserve main-checkout watchers and use the isolated [fixture preview](../quartz/compatibility/fixture-preview.md) when verifying this profile.
