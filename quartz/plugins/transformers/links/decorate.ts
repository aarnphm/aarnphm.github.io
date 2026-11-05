import type { Element } from 'hast'
import { h, s } from 'hastscript'
import isAbsoluteUrl from 'is-absolute-url'
import {
  anthropicSvg,
  bentomlHoverSvg,
  bentomlSvg,
  ycSvg,
  bskySvg,
  doiSvg,
  githubSvg,
  githubWhiteSvg,
  substackSvg,
  svgOptions,
  twitterSvg,
  openaiSvg,
  hfSvg,
  obsidianSvg,
  youtubeSvg,
  gwernSvg,
  modularSvg,
  nvidiaSvg,
} from '../../../components/svg'
import { parseLessWrongTarget } from '../../../util/lesswrong'
import { sectionOfLink } from '../../../util/section-icon'
import { parseSepTarget } from '../../../util/sep'
import { hostnameMatches, parseExternalUrl } from '../../../util/url'
import { parseWikipediaTarget } from '../../../util/wikipedia'
import { extractArxivId } from '../../stores/citations'
import { filterEmbedTwitter } from '../twitter'
import { metadataDisablesPopover } from './metadata'

const ALLOWED_EXTENSIONS = [
  '.py',
  '.go',
  '.java',
  '.c',
  '.cpp',
  '.cxx',
  '.cu',
  '.cuh',
  '.h',
  '.hpp',
  '.ts',
  '.tsx',
  '.yaml',
  '.yml',
  '.rs',
  '.m',
  '.sql',
  '.sh',
  '.txt',
]

interface LinkContext {
  classes: string[]
  dest: string
  ext: string
  isExternal: boolean
  node: Element
  metadata?: Record<string, unknown>
}

interface DecorationOptions {
  enableArxivEmbed: boolean
  enableRawEmbed: boolean
  externalLinkIcon: boolean
  baseUrl?: string
  fileSlug?: string
  email?: boolean
  basePages(): Set<string>
}

export function decorateLink(
  node: Element,
  dest: string,
  ext: string,
  metadata: Record<string, unknown> | undefined,
  options: DecorationOptions,
): { destination: string; isExternal: boolean } {
  const opts = options
  const classes = Array.isArray(node.properties.className)
    ? node.properties.className.filter(value => typeof value === 'string')
    : []
  const externalUrl = parseExternalUrl(dest)
  const matchesHost = (hostname: string) => hostnameMatches(externalUrl, hostname)
  const configuredBaseUrl = options.baseUrl
  const apexUrl = configuredBaseUrl
    ? parseExternalUrl(
        configuredBaseUrl.includes('://') ? configuredBaseUrl : `https://${configuredBaseUrl}`,
      )
    : undefined
  const ctx: LinkContext = {
    node,
    classes,
    dest,
    ext,
    metadata,
    isExternal:
      options.enableRawEmbed && ALLOWED_EXTENSIONS.includes(ext)
        ? true
        : isAbsoluteUrl(dest, { httpOnly: false }) || /^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(dest),
  }
  const linkTypes = {
    isApexDomain: apexUrl !== undefined && hostnameMatches(externalUrl, apexUrl.hostname),
    isCslNode: classes.includes('csl-external-link'),
    isEmbedTwitter: filterEmbedTwitter(node),
    isArxiv: matchesHost('arxiv.org'),
    isWikipedia: matchesHost('wikipedia.org'),
    isLessWrong: matchesHost('lesswrong.com'),
    isBentoml: matchesHost('bentoml.com'),
    isModular: matchesHost('modular.com'),
    isNvidia: matchesHost('nvidia.com') || matchesHost('nvidia.cn'),
    isSep: matchesHost('plato.stanford.edu'),
    isYoutube: matchesHost('youtube.com') || matchesHost('youtu.be'),
    isGwern: matchesHost('gwern.net'),
    isNeovim: matchesHost('neovim.io'),
    isQuartz: matchesHost('quartz.jzhao.xyz'),
    isObsidian: matchesHost('obsidian.md'),
    isGithub: matchesHost('github.com'),
    isSubstack: matchesHost('substack.com'),
    isTwitter: matchesHost('x.com') || matchesHost('twitter.com'),
    isBsky: matchesHost('bsky.app'),
    isDoi: matchesHost('doi.org'),
    isOpenai: matchesHost('openai.com'),
    isHf: matchesHost('huggingface.co'),
    isYC: matchesHost('ycombinator.com'),
    isAnthropic: matchesHost('transformer-circuits.pub') || matchesHost('anthropic.com'),
    isGoogleDocs: matchesHost('docs.google.com'),
    isGoogleDrive: matchesHost('drive.google.com'),
  }
  const wikipediaTarget = linkTypes.isWikipedia ? parseWikipediaTarget(ctx.dest) : undefined
  const lessWrongTarget = linkTypes.isLessWrong ? parseLessWrongTarget(ctx.dest) : undefined
  const sepTarget = linkTypes.isSep ? parseSepTarget(ctx.dest) : undefined

  if (linkTypes.isBentoml) {
    if (!classes.includes('bentoml-link')) {
      classes.push('bentoml-link')
    }
    ctx.node.properties.dataLinkVendor = 'bentoml'
  }

  if (linkTypes.isModular) {
    if (!classes.includes('modular-link')) {
      classes.push('modular-link')
    }
    ctx.node.properties.dataLinkVendor = 'modular'
  }

  if (linkTypes.isNvidia) {
    if (!classes.includes('nvidia-link')) {
      classes.push('nvidia-link')
    }
    ctx.node.properties.dataLinkVendor = 'nvidia'
  }

  if (linkTypes.isGithub) {
    if (!classes.includes('github-link')) {
      classes.push('github-link')
    }
    ctx.node.properties.dataLinkVendor = 'github'
  }

  if (
    wikipediaTarget &&
    node.children.length === 1 &&
    node.children[0].type === 'text' &&
    node.children[0].value === dest
  ) {
    const langPrefix = wikipediaTarget.lang !== 'simple' ? `${wikipediaTarget.lang}/` : ''
    node.children[0].value = `wikipedia/${langPrefix}${wikipediaTarget.title}`
  }

  if (
    lessWrongTarget &&
    node.children.length === 1 &&
    node.children[0].type === 'text' &&
    node.children[0].value === dest
  ) {
    node.children[0].value = `lesswrong/${lessWrongTarget.slug ?? lessWrongTarget.postId}`
  }

  if (
    sepTarget &&
    node.children.length === 1 &&
    node.children[0].type === 'text' &&
    node.children[0].value === dest
  ) {
    node.children[0].value = `sep/${sepTarget.entry}`
  }

  if (
    linkTypes.isYoutube &&
    node.children.length === 1 &&
    node.children[0].type === 'text' &&
    node.children[0].value === dest
  ) {
    try {
      const u = new URL(dest)
      const vid = u.searchParams.get('v')
      if (vid) {
        node.children = [{ type: 'text', value: `youtube/v=${vid}` }]
      }
    } catch {}
  }

  if (
    linkTypes.isGoogleDocs &&
    node.children.length === 1 &&
    node.children[0].type === 'text' &&
    node.children[0].value === dest
  ) {
    try {
      const u = new URL(dest)
      const m = u.pathname.match(/\/d\/([^/]+)/)
      if (m) {
        const id = m[1]
        let displayId = id
        if (id.length > 10) {
          displayId = `${id.slice(0, 3)}[...]${id.slice(-3)}`
        }
        node.children[0].value = `docs.google.com/${displayId}`
      }
    } catch {}
  }

  if (
    linkTypes.isGoogleDrive &&
    node.children.length === 1 &&
    node.children[0].type === 'text' &&
    node.children[0].value === dest
  ) {
    try {
      const u = new URL(dest)
      const m = u.pathname.match(/\/d\/([^/]+)/)
      if (m) {
        const id = m[1]
        node.children[0].value = `drive.google.com/${id}`
      }
    } catch {}
  }

  // Handle special link types
  const handleArxiv = (ctx: LinkContext) => {
    if (opts.enableArxivEmbed && linkTypes.isArxiv) {
      ctx.classes.push('internal')
      ctx.node.properties.dataArxivId = extractArxivId(ctx.dest)
      return true
    }
    return false
  }

  const handleWikipedia = (ctx: LinkContext) => {
    if (!wikipediaTarget || metadataDisablesPopover(ctx.metadata)) return false
    ctx.classes.push('internal')
    ctx.node.properties.dataWikipediaLang = wikipediaTarget.lang
    ctx.node.properties.dataWikipediaTitle = wikipediaTarget.title
    return true
  }

  const handleLessWrong = (ctx: LinkContext) => {
    if (!lessWrongTarget || metadataDisablesPopover(ctx.metadata)) return false
    ctx.classes.push('internal')
    ctx.node.properties.dataLesswrongPostId = lessWrongTarget.postId
    if (lessWrongTarget.slug) {
      ctx.node.properties.dataLesswrongSlug = lessWrongTarget.slug
    }
    return true
  }

  const handleSep = (ctx: LinkContext) => {
    if (!sepTarget || metadataDisablesPopover(ctx.metadata)) return false
    ctx.classes.push('internal')
    ctx.node.properties.dataSepEntry = sepTarget.entry
    if (sepTarget.archive) {
      ctx.node.properties.dataSepArchive = sepTarget.archive
    }
    return true
  }

  const handleCdnLinks = (ctx: LinkContext) => {
    if (ctx.isExternal && opts.enableRawEmbed) {
      if (ALLOWED_EXTENSIONS.includes(ctx.ext) && !isAbsoluteUrl(ctx.dest, { httpOnly: false })) {
        ctx.classes.push('cdn-links')
        ctx.dest = ctx.node.properties.href = `https://aarnphm.xyz/${ctx.dest}`
      }
    }
  }

  const createIconElement = (src: string, alt: string) =>
    h(
      'span',
      { style: 'white-space: nowrap;' },
      // The word joiner keeps the icon on the line of the link's last word.
      '\u2060',
      h('img.inline-icons', {
        src,
        alt,
        style: 'height: 8px; width: 8px; margin-left: 3px; bottom: 2px; position: relative;',
      }),
    )

  // Add appropriate icons based on link type
  if (
    !handleArxiv(ctx) &&
    !handleWikipedia(ctx) &&
    !handleLessWrong(ctx) &&
    !handleSep(ctx) &&
    !linkTypes.isEmbedTwitter
  ) {
    ctx.classes.push(ctx.isExternal ? 'external' : 'internal')
  }

  handleCdnLinks(ctx)

  // Add appropriate icons (skip if data-skip-icons is present)
  const skipIcons =
    ctx.node.properties.dataSkipIcons === true ||
    ctx.node.properties.dataSkipIcons === 'true' ||
    options.email === true
  if (!skipIcons) {
    if (linkTypes.isWikipedia) {
      ctx.node.children.push(createIconElement('/static/favicons/wikipedia.svg', 'Wikipedia'))
    } else if (linkTypes.isApexDomain && options.fileSlug !== 'index') {
      // The icon of the page's section, as its own tab shows (util/section-icon.ts).
      const section = sectionOfLink(externalUrl!, apexUrl!.hostname, options.basePages())
      ctx.node.children.push(
        section
          ? createIconElement(`/static/icon-${section}.png`, section)
          : createIconElement('/static/icon.webp', 'apex'),
      )
    } else if (linkTypes.isArxiv) {
      ctx.node.children.push(createIconElement('/static/favicons/arxiv.avif', 'arXiv'))
    } else if (linkTypes.isLessWrong) {
      ctx.node.children.push(createIconElement('/static/favicons/lesswrong.avif', 'LessWrong'))
    } else if (linkTypes.isQuartz) {
      ctx.node.children.push(createIconElement('/static/favicons/quartz.png', 'Quartz'))
    } else if (linkTypes.isNeovim) {
      ctx.node.children.push(createIconElement('/static/favicons/neovim.svg', 'Neovim'))
    } else if (linkTypes.isBentoml) {
      ctx.node.children.push(bentomlSvg, bentomlHoverSvg)
    } else if (linkTypes.isModular) {
      ctx.node.children.push(modularSvg)
    } else if (linkTypes.isNvidia) {
      ctx.node.children.push(nvidiaSvg)
    } else if (linkTypes.isSep) {
      ctx.node.children.push(createIconElement('/static/favicons/sep-man-red.png', 'SEP'))
    } else if (linkTypes.isYoutube) {
      ctx.node.children.push(youtubeSvg)
    } else if (linkTypes.isGwern) {
      ctx.node.children.push(gwernSvg)
    } else if (linkTypes.isObsidian) {
      ctx.node.children.push(obsidianSvg)
    } else if (linkTypes.isYC) {
      ctx.node.children.push(ycSvg)
    } else if (linkTypes.isDoi) {
      ctx.node.children.push(doiSvg)
    } else if (linkTypes.isHf) {
      ctx.node.children.push(hfSvg)
    } else if (linkTypes.isAnthropic) {
      ctx.node.children.push(anthropicSvg)
    } else if (linkTypes.isOpenai) {
      ctx.node.children.push(openaiSvg)
    } else if (linkTypes.isGithub) {
      ctx.node.children.push(githubSvg, githubWhiteSvg)
    } else if (linkTypes.isSubstack) {
      ctx.node.children.push(substackSvg)
    } else if (linkTypes.isTwitter) {
      ctx.node.children.push(twitterSvg)
    } else if (linkTypes.isBsky) {
      ctx.node.children.push(bskySvg)
    } else if (
      !linkTypes.isEmbedTwitter &&
      !linkTypes.isCslNode &&
      !linkTypes.isArxiv &&
      ctx.isExternal &&
      opts.externalLinkIcon
    ) {
      ctx.node.children.push(
        s(
          'svg',
          {
            ...svgOptions,
            ariaHidden: true,
            class: 'external-icon',
            viewbox: '0 -12 24 24',
            fill: 'none',
            stroke: 'currentColor',
            strokewidth: 1.5,
          },
          [s('use', { href: '#arrow-ne' })],
        ),
      )
    }
  }

  // Check if the link has alias text
  if (
    node.children.length === 1 &&
    node.children[0].type === 'text' &&
    node.children[0].value !== dest
  ) {
    // Add the 'alias' class if the text content is not the same as the href
    classes.push('alias')
  }
  node.properties.className = classes
  return { destination: ctx.dest, isExternal: ctx.isExternal }
}
