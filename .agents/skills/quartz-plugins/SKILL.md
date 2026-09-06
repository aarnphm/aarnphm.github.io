---
name: quartz-plugins
description: Build Quartz plugins and Preact components with the correct transform, emit, and browser lifecycle boundaries.
---

# Quartz plugins and components

Transformers in `quartz/plugins/transformers` operate on mdast/hast trees. Keep filesystem access out of transformers; use the emitter phase for filesystem output. Reuse Quartz's existing processor rather than creating a nested unist processor inside `markdownPlugins` or `htmlPlugins`.

Components live in `quartz/components` and use Preact with `PascalCase.tsx` names. Utilities and inline script basenames use kebab-case; scripts live in `quartz/components/scripts/<name>.inline.ts`. Follow existing ES modules and two-space indentation.

In inline scripts, initialize page-scoped state inside the `nav` handler and register cleanup there. Keep transient state local to the current page. Use storage only for state intended to persist, with the owning parser and versioned contract where applicable.

Use native `<button type="button">` for actions and links for navigation. Preserve accessible names, focus, and keyboard behavior. Complex controls follow their actual tab, radio, or toggle semantics.

Trace changes through the affected data model, serialization, SSR, hydration, styles, and interactions. Verify the relevant boundaries and wait for the matching watcher build before treating a browser page as evidence. Read the [Quartz plugin documentation](https://quartz.jzhao.xyz/advanced/making-plugins) when an extension API needs confirmation.
