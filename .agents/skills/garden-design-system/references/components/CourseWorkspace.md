# CourseWorkspace

The lecture workspace beside an authored lecture note: source, scratchpad and notes tabs.

**Source:** `quartz/components/course-workspace/` (`app.tsx`, `Header.tsx`, `editor.tsx`, `selection.tsx`, `store.ts`, `anchors.ts`, `split-divider.tsx`) · styles: `quartz/components/styles/course-workspace.scss` · notes: `docs/course-workspace.md`

**Corrections:** in the source, the license line (`.course-source-license`) and the unselected tab labels are `gray`. This rendition uses `darkgray`.

## Preview

A copy of the server-rendered markup from `/courses/18.905-fall-2016/lectures/01` (selector `.course-workspace`), styled by `site/garden.css`, the site's compiled CSS pruned to the rules these previews use. Links point to `#`. The PDF reader iframe and the CodeMirror editor are grey placeholders, and the panel sits in the page flow instead of fixed to the screen edge. The preview is static because the app is a Preact tree that reads browser storage.

## Rules

- The workspace is not a modal. Above 1400px it is a fixed side panel in the free space right of the reading column, with a drag divider. At 1400px and below the same panel sits inline under the note (`min(48rem, 100dvh - space-6)` high).
- The header is a tablist of three `tri-map-tab` tabs. Each shows its shortcut key while unselected (`S`, `P`, `N`) and swaps to its label (`source`, `scratchpad`, `notes 0`) when selected. Actions follow: `export .md`, `open reader ↗`, `pdf ↗`. The license line sits under the head.
- Source embeds the PDF reader with `?embed=course`, so marks, notes and reading position keep their existing PDF identities.
- Scratchpad uses the shared CodeMirror Markdown editor. Tab leaves the editor; Escape and Mod-Enter return focus to the workspace tab. The status line reads `private · saved in this browser`; notes read `private · saved to your account`.
- Selecting lecture text offers mark, question, disagree, note and insert. Passage highlights use the CSS Custom Highlight API (`::highlight(...)`), so the lecture DOM stays intact.
- Drafts live in browser storage under `course-workspace:v1:<slug>`. A failed write stays in the editor, the panel names the storage error, and export still works.
- Export Markdown keeps note IDs, lecture and PDF citations, and LaTeX source in place of KaTeX output.
- Focus: tabs draw the 2px `focus` ring outside the tab (see `focus.md`, Tabs).
