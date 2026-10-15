import { Features, transform } from 'lightningcss'

export function minifyStylesheet(filename: string, stylesheet: string): string {
  return transform({
    filename,
    code: Buffer.from(stylesheet),
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
