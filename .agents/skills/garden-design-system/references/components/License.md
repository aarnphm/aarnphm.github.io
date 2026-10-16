# License

The licence line at the foot of a page that sets `license` in frontmatter.

**Source:** `quartz/components/License.tsx` · icons: `quartz/util/cc-icons.ts` · styles: `quartz/components/styles/license.scss`

**Debt:** the label sets in `gray` (2.00:1 on paper). Use `darkgray` in new work.

## Preview

A copy of the server-rendered markup from `/courses/18-156-projection-theory-spring-2025/notes/16` (selector `.license-meta`), styled by `site/garden.css`, the site's compiled CSS pruned to the rules these previews use. Links point to `#`. The preview is static; the component has no script.

## Rules

- A Creative Commons URL in `license_url` renders the CC mark and one icon per module, then the code: `CC BY-NC-SA 4.0`. Icons are inline SVG in `currentColor`, `aria-hidden`.
- Another URL renders the `license` text as the link. No URL renders plain text.
- The course mirrors set it from their OCW source; any note can set it.
