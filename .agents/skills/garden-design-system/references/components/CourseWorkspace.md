# CourseWorkspace

The lecture workspace beside an authored lecture note: source, scratchpad and notes tabs.

**Source:** `quartz/components/course-workspace/` (`app.tsx`, `editor.tsx`, `selection.tsx`, `store.ts`, `anchors.ts`, `split-divider.tsx`) · styles: `quartz/components/styles/course-workspace.scss` · notes: `docs/course-workspace.md`

## Preview

A copy of the server-rendered markup from `/courses/18.905-fall-2016/lectures/01` (selector `.course-workspace`), styled by `site/garden.css`, the site's compiled CSS pruned to the rules these previews use. Links point to `#`. The PDF reader iframe and the CodeMirror editor are grey placeholders. The preview is static because the app is a Preact tree that reads browser storage.

## Rules

- Above 1400px the workspace is a side panel beside the reading column, with a drag divider. Below that, the same workspace opens in a native modal `<dialog>`.
- Source embeds the PDF reader with `?embed=course`, so marks, notes and reading position keep their existing PDF identities.
- Scratchpad uses the shared CodeMirror Markdown editor. Tab leaves the editor; Escape and Mod-Enter return focus to the workspace tab.
- Selecting lecture text offers mark, question, disagree, note and insert. Passage highlights use the CSS Custom Highlight API (`::highlight(...)`), so the lecture DOM stays intact.
- Drafts live in browser storage under `course-workspace:v1:<slug>`. A failed write stays in the editor, the panel names the storage error, and export still works.
- Export Markdown keeps note IDs, lecture and PDF citations, and LaTeX source in place of KaTeX output.
