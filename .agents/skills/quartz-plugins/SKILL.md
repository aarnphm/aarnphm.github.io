---
name: quartz-plugins
description: Extend Garden's Quartz transforms, emitters, and Preact or browser integration.
---

# Quartz plugins and components

Transformers in `quartz/plugins/transformers` operate on mdast/hast trees. Keep filesystem access out of transformers; use the emitter phase for filesystem output. Reuse Quartz's existing processor rather than creating a nested unist processor inside `markdownPlugins` or `htmlPlugins`.

Components live in `quartz/components` and use Preact with `PascalCase.tsx` names. Utilities and inline script basenames use kebab-case; scripts live in `quartz/components/scripts/<name>.inline.ts` or an existing feature module. Registered note figures have their own `interactive-diagrams` skill; use it when figure conventions are relevant.

In inline scripts, initialize page-scoped state inside the `nav` handler and register cleanup there. Keep transient state local to the current page. Use storage only for state intended to persist, with the owning parser and versioned contract where applicable.

Use native `<button type="button">` for actions and links for navigation. Preserve accessible names, focus, and keyboard behavior. Complex controls follow their actual tab, radio, or toggle semantics.

Follow the boundaries the change actually crosses. For data-bearing components, check the model, serialized payload, SSR, and browser consumer together; retain per-field provenance when merging provider data. A style-only edit can stay with its component and styles.

Complete the requested behavior and fix failures caused by the change using focused checks. Browser claims require the matching watcher `build:ready`, HTTP availability, and inspection of the served implementation. Report a blocked boundary precisely if verification cannot finish. Read the [Quartz plugin documentation](https://quartz.jzhao.xyz/advanced/making-plugins) when an extension API needs confirmation, with this fork's types and callers as the local contract.
