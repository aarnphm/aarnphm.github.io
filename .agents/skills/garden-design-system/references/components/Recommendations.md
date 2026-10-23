# Recommendations

Up to nine notes picked for the end of a page. The picks come from the build's recommendation pool (the current page is left out), shuffled with a seed taken from the page path, so a page always shows the same list.

**Source:** `quartz/components/Recommendations.tsx` · styles: `quartz/styles/custom.scss` (`.recommendations`, `#recommendations-label`), `quartz/styles/pages/technical.scss`, `quartz/components/styles/graph.scss`

## Preview

Last-known markup, copied from `/thoughts/Attention` when a layout still placed the component (selector `section.recommendations`), styled by `site/garden.css`, the site's compiled CSS pruned to the rules these previews use. Links point to `#`. No layout places the component now, so no page renders it and the preview cannot be refreshed from the site. The component has no script; the markup is the whole component. The heading is French (`vous pourriez aimer ce qui suit`) and sets `lang="en"`.

## Rules

- Same row grammar as Backlinks. The list is computed at build time.
- Lectures (course units of kind `lectures`) render nothing.
