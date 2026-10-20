import { Features, transform } from 'lightningcss'

// Component rules name system cursors; the site draws its own (styles/native-cursor.scss), so
// every keyword cursor reads its custom property, which the triathlon pages reset to the keyword.
const SYSTEM_CURSOR =
  /(?<![-\w])cursor\s*:\s*(?!(?:auto|none|inherit|initial|unset|revert|revert-layer)\b)([a-z]+(?:-[a-z]+)*)(?=\s*[;}!])/g

export function minifyStylesheet(filename: string, stylesheet: string): string {
  return transform({
    filename,
    code: Buffer.from(stylesheet.replace(SYSTEM_CURSOR, 'cursor: var(--cursor-$1, $1)')),
    minify: true,
    targets: {
      safari: (15 << 16) | (6 << 8),
      ios_saf: (15 << 16) | (6 << 8),
      edge: 115 << 16,
      firefox: 102 << 16,
      chrome: 109 << 16,
    },
    include: Features.MediaQueries,
  }).code.toString()
}
