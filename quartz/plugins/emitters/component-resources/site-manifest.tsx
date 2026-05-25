import type { BuildCtx } from '../../../util/ctx'
import type { FilePath, FullSlug } from '../../../util/path'
import { write } from '../helpers'

export async function writeSiteManifest(ctx: BuildCtx): Promise<FilePath> {
  const cfg = ctx.cfg.configuration
  const manifest = {
    name: cfg.pageTitle,
    short_name: cfg.baseUrl,
    icons: [
      { src: '/android-chrome-192x192.png', sizes: '192x192', type: 'image/png' },
      { src: '/android-chrome-512x512.png', sizes: '512x512', type: 'image/png' },
    ],
    theme_color: cfg.theme.colors['lightMode'].light,
    background_color: cfg.theme.colors['lightMode'].light,
    display: 'standalone',
    lang: cfg.locale,
    dir: 'auto',
  }
  return write({
    ctx,
    slug: 'site' as FullSlug,
    ext: '.webmanifest',
    content: JSON.stringify(manifest),
  })
}

export function externalResources({ cfg }: BuildCtx) {
  return {
    additionalHead: [
      <link rel="manifest" href={`https://${cfg.configuration.baseUrl}/site.webmanifest`} />,
      <link rel="shortcut icon" href={`https://${cfg.configuration.baseUrl}/favicon.ico`} />,
      <link
        rel="icon"
        type="image/png"
        sizes="32x32"
        href={`https://${cfg.configuration.baseUrl}/favicon-32x32.png`}
      />,
      <link
        rel="icon"
        type="image/png"
        sizes="16x16"
        href={`https://${cfg.configuration.baseUrl}/favicon-16x16.png`}
      />,
      <link
        rel="apple-touch-icon"
        sizes="180x180"
        href={`https://${cfg.configuration.baseUrl}/apple-touch-icon.png`}
      />,
      <link
        rel="android-chrome"
        sizes="192x192"
        href={`https://${cfg.configuration.baseUrl}/android-chrome-192x192.png`}
      />,
      <link
        rel="android-chrome"
        sizes="512x512"
        href={`https://${cfg.configuration.baseUrl}/android-chrome-512x512.png`}
      />,
    ],
  }
}
