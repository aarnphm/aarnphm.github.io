"""Subset the site's web fonts to the characters, weights, axes and features the site uses.

Run by hand after a build (the character census reads the rendered HTML in public/):

    pnpm fonts:subset                 # subset everything, write SCSS + manifest, push changed
                                      # files to the FONTS KV (the site reads them there)
    pnpm fonts:subset -- --no-upload  # local only; the generated CSS then points at missing keys

Outputs
  quartz/.quartz-cache/fonts/out/<stem>.{core,ext}.woff2   every face; pushed to the FONTS KV when
                                                            its content changed, never committed
  quartz/styles/fonts.scss                        generated @font-face rules with unicode-range
  quartz/styles/fonts.preload.json                core files Head.tsx preloads
  quartz/styles/fonts.manifest.json               per-file bytes, glyphs, ranges (committed)

Output is byte-for-byte reproducible (fontTools pinned, head timestamps kept), so a rerun that
changes nothing leaves the tree clean.

Rerun when: content adds characters outside the kept set (a page shows a system-font glyph where
the web font should be), a stylesheet starts requesting a weight or style that FACES does not
enable, a new OpenType feature is turned on (KEEP_FEATURES), or a source font is replaced.
"""
