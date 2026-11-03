import type { Element, ElementContent, Root } from 'hast'
import { toString } from 'hast-util-to-string'
import { h } from 'hastscript'
import { imageSize } from 'image-size'
import isAbsoluteUrl from 'is-absolute-url'
import { createHash } from 'node:crypto'
import { readFileSync, statSync } from 'node:fs'
import path from 'node:path'
import { SKIP, visit } from 'unist-util-visit'
import type { RuleKind } from '../../util/grapevine'
// @ts-ignore
import script from '../../components/scripts/profile-notes.inline'
import { QuartzTransformerPlugin } from '../../types/plugin'
import { dropcapOf, initials, separatorOf } from './profile/initials'

export const NEWSPAPER = 'newspaper' as const

export const PROFILE_WIDTHS = [480, 800, 1200] as const
export const PROFILE_DIR = 'static/profile'

// Portrait filename patterns; other images are references.
const portrait = /^(me-|selfie)/

const SIZES = {
  hero: '(min-width: 60rem) 26rem, 46vw',
  me: '(min-width: 48rem) 24rem, min(20rem, 92vw)',
  ref: '(min-width: 48rem) 15rem, min(14rem, 70vw)',
}

type Kind = keyof typeof SIZES

interface Source {
  rel: string
  width: number
  height: number
  key: string
}

function readSource(directory: string, rel: string): Source | null {
  const abs = path.join(directory, rel)
  try {
    const stat = statSync(abs)
    const { width, height } = imageSize(readFileSync(abs))
    if (!width || !height) return null
    const key = createHash('sha1')
      .update(`${rel}:${stat.size}:${stat.mtimeMs}`)
      .digest('hex')
      .slice(0, 12)
    return { rel, width, height, key }
  } catch {
    return null
  }
}

function localSource(slug: string, src: unknown): string | null {
  if (typeof src !== 'string' || src.length === 0 || isAbsoluteUrl(src) || src.startsWith('/'))
    return null
  const clean = decodeURI(src.split(/[?#]/)[0])
  return path.posix.normalize(path.posix.join(path.posix.dirname(slug), clean))
}

function rootPrefix(slug: string): string {
  const depth = slug.split('/').length - 1
  return depth === 0 ? './' : '../'.repeat(depth)
}

function isQuote(node: Element): boolean {
  return (
    node.tagName === 'blockquote' &&
    ((node.properties.className as string[] | undefined)?.includes('quotes') ?? false)
  )
}

function frame(
  img: Element,
  kind: Kind,
  index: number | null,
  caption?: ElementContent[],
): Element {
  const label = index === null ? [] : [h('span.profile-frame-no', String(index).padStart(2, '0'))]
  const content = [...label, ...(caption ?? [])]
  return h('figure.profile-frame', { dataKind: kind, dataCaptioned: caption ? '' : undefined }, [
    img,
    ...(content.length > 0 ? [h('figcaption', content)] : []),
  ])
}

function hasStyledImage(node: Element): boolean {
  return (
    (node.tagName === 'img' &&
      typeof node.properties.style === 'string' &&
      node.properties.style.trim().length > 0) ||
    node.children.some(child => child.type === 'element' && hasStyledImage(child))
  )
}

export const Profile: QuartzTransformerPlugin = () => {
  return {
    name: 'Profile',
    htmlPlugins(ctx) {
      return [
        () => (tree: Root, file) => {
          const slug = file.data.slug
          const frontmatter = file.data.frontmatter
          if (!slug || !frontmatter) return
          if (frontmatter.pageLayout !== NEWSPAPER) {
            if (frontmatter.dropcap != null) initials(tree, dropcapOf(frontmatter.dropcap), slug)
            return
          }
          file.data.profile = true

          const prefix = rootPrefix(slug)
          let count = 0
          let hero = true

          const dress = (img: Element, kind: Kind, eager: boolean) => {
            const rel = localSource(slug, img.properties.src)
            const source = rel ? readSource(ctx.argv.directory, rel) : null
            if (source) {
              img.properties.width = source.width
              img.properties.height = source.height
              const widths = PROFILE_WIDTHS.filter(w => w < source.width)
              if (widths.length > 0) {
                img.properties.srcSet = [
                  ...widths.map(w => `${prefix}${PROFILE_DIR}/${source.key}-${w}.webp ${w}w`),
                  `${img.properties.src} ${source.width}w`,
                ].join(', ')
                img.properties.sizes = SIZES[kind]
                img.properties.dataProfileSource = `${source.rel}#${source.key}#${widths.join(',')}`
              }
            }
            img.properties.decoding = 'async'
            if (eager) {
              img.properties.loading = 'eager'
              if (count === 1) img.properties.fetchPriority = 'high'
            }
          }

          const isPhoto = (node: ElementContent | undefined): node is Element =>
            node?.type === 'element' &&
            node.tagName === 'img' &&
            !(node.properties.className as string[] | undefined)?.includes('inline-icons') &&
            localSource(slug, node.properties.src) !== null

          const kindOf = (img: Element): 'me' | 'ref' => {
            const base = path.posix.basename(String(img.properties.src))
            return portrait.test(base) ? 'me' : 'ref'
          }

          visit(tree, 'element', (node, index, parent) => {
            if (!parent || index === undefined) return

            if (node.tagName === 'blockquote' && isQuote(node)) {
              const run: Element[] = [node]
              let end = index + 1
              for (; end < parent.children.length; end++) {
                const next = parent.children[end]
                if (next.type === 'text' && !next.value.trim()) continue
                if (next.type === 'element' && isQuote(next)) run.push(next)
                else break
              }
              if (run.length > 1) {
                parent.children.splice(index, end - index, h('div.profile-pulls', run))
                return SKIP
              }
              return
            }

            if (node.tagName === 'h2' || node.tagName === 'section') hero = false

            if (
              node.tagName === 'div' &&
              (node.properties.className as string[] | undefined)?.includes('image-grid')
            ) {
              if (hasStyledImage(node)) return SKIP
              const cells = node.children.flatMap(c => {
                if (c.type !== 'element') return []
                if (isPhoto(c)) return [{ img: c, caption: undefined }]
                const fig = c as Element
                if (fig.tagName !== 'figure') return []
                const img = fig.children.find(isPhoto)
                const caption = fig.children.find(
                  (f): f is Element => f.type === 'element' && f.tagName === 'figcaption',
                )
                return img ? [{ img, caption: caption?.children }] : []
              })
              const asHero = hero && cells.every(({ img }) => kindOf(img) === 'me')
              node.properties.className = asHero
                ? ['image-grid', 'profile-strip', 'profile-hero']
                : ['image-grid', 'profile-strip']
              node.children = cells.map(({ img, caption }) => {
                const kind = kindOf(img)
                const n = kind === 'me' ? ++count : null
                dress(img, asHero ? 'hero' : kind, asHero)
                return frame(img, kind, n, caption)
              })
              hero = false
              return SKIP
            }

            if (node.tagName === 'figure' && node.properties.dataImgWCaption !== undefined) {
              const img = node.children.find(isPhoto)
              if (!img || hasStyledImage(img)) return SKIP
              const kind = kindOf(img)
              const n = kind === 'me' ? ++count : null
              dress(img, kind, false)
              const caption = node.children.find(
                (c): c is Element => c.type === 'element' && c.tagName === 'figcaption',
              )
              const framed = frame(img, kind, n, caption?.children)
              // Preserve the side of a hand-placed float.
              if (
                parent.type === 'element' &&
                parent.tagName === 'figure' &&
                parent.children.filter(c => c.type === 'element').length === 1
              ) {
                const side = (parent.properties.className as string[] | undefined)?.filter(c =>
                  c.startsWith('float-'),
                )
                if (side?.length) framed.properties.className = ['profile-frame', ...side]
                Object.assign(parent, framed)
                return SKIP
              }
              parent.children[index] = framed
              return SKIP
            }

            if (isPhoto(node)) {
              // Authored CSS controls placement; a profile frame would contain floats and override sizing.
              if (hasStyledImage(node)) return SKIP
              const kind = kindOf(node)
              const n = kind === 'me' ? ++count : null
              dress(node, kind, false)
              const framed = frame({ ...node }, kind, n)
              if (
                parent.type === 'element' &&
                parent.tagName === 'p' &&
                parent.children.every(c => c === node || (c.type === 'text' && !c.value.trim()))
              ) {
                Object.assign(parent, framed)
                return SKIP
              }
              parent.children[index] = framed
              return SKIP
            }
          })

          visit(tree, 'element', node => {
            if (node.tagName !== 'figure') return
            const side = classes(node).filter(c => c.startsWith('float-'))
            const inner = node.children.filter(c => c.type === 'element')
            const only = inner.length === 1 ? inner[0] : undefined
            if (side.length === 0 || only?.tagName !== 'figure') return
            if (!classes(only).includes('profile-frame')) return
            only.properties.className = ['profile-frame', ...side]
            Object.assign(node, only)
          })

          const face = dropcapOf(frontmatter.dropcap)
          file.data.separator = separatorOf(frontmatter.separator, slug)
          initials(tree, face, slug)
          marginalia(tree)
        },
      ]
    },
    externalResources() {
      return { js: [{ script, contentType: 'inline', loadTime: 'afterDOMReady' }] }
    },
  }
}

const BLOCKS = new Set(['p', 'li', 'blockquote', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'dd', 'dt'])
const classes = (node: Element) => (node.properties.className as string[] | undefined) ?? []

// Long notes, lists and maths are boxed instead of using the 13rem rail.
const BOXED_WORDS = 70

function note(id: string, kind: 'footnote' | 'sidenote', mark: string, body: ElementContent[]) {
  let boxed = toString({ type: 'root', children: body } as Root).split(/\s+/).length > BOXED_WORDS
  visit({ type: 'root', children: body } as Root, 'element', el => {
    if (['ul', 'ol', 'figure', 'table'].includes(el.tagName) || classes(el).includes('katex'))
      boxed = true
  })
  return h(
    'aside',
    {
      id,
      dataKind: kind,
      className: boxed ? ['profile-note', 'profile-note-boxed'] : ['profile-note'],
    },
    [h('span.profile-note-mark', mark), ...body],
  )
}

/** Move notes after their citing block for CSS placement; keep them inline on narrow screens. */
function marginalia(tree: Root) {
  const footnotes = new Map<string, Element>()
  let section: { node: Element; parent: Root | Element; index: number } | undefined
  visit(tree, 'element', (node, index, parent) => {
    if (node.properties.dataFootnotes === undefined || !parent || index === undefined) return
    section = { node, parent: parent as Root | Element, index }
    visit(node, 'element', li => {
      if (li.tagName === 'li' && typeof li.properties.id === 'string')
        footnotes.set(li.properties.id, li)
    })
    return SKIP
  })

  // Wait for each block to close; asides cannot sit inside paragraphs.
  const pending = new Map<Element, Element[]>()
  const moved = new Set<string>()
  // Notes whose mark (label word or footnote square) carries an anchor, so a hovered note can
  // box it from the rail.
  const marked = new Set<string>()
  const mark = (el: Element, id: string) => {
    el.properties.style = `anchor-name:--${anchorId(id)}-mark`
    marked.add(id)
  }
  const walk = (node: Element | Root, block: Element | null) => {
    for (let i = 0; i < node.children.length; i++) {
      const child = node.children[i]
      if (child.type !== 'element') continue
      if (child === section?.node) continue
      const here = BLOCKS.has(child.tagName) ? child : block

      if (classes(child).includes('sidenote-content') && here) {
        const label = node.children[i - 1]
        const word =
          label?.type === 'element' && classes(label).includes('sidenote')
            ? toString(label).trim()
            : ''
        const id = `${String(child.properties.id ?? 'note')}-margin`
        const aside = note(id, 'sidenote', word, child.children)
        pending.set(here, [...(pending.get(here) ?? []), aside])
        if (label?.type === 'element') {
          label.properties.dataNote = id
          mark(label, id)
          // Insert the cite before the label; the moved content then sits at i + 1.
          node.children.splice(i - 1, 0, cite(aside))
          node.children.splice(i + 1, 1)
        } else {
          node.children.splice(i, 1)
          i--
        }
        continue
      }

      if (child.tagName === 'sup' && here) {
        const ref = child.children.find(
          (c): c is Element => c.type === 'element' && c.properties.dataFootnoteRef !== undefined,
        )
        const target = String(ref?.properties.href ?? '').replace(/^#/, '')
        const item = footnotes.get(target)
        if (ref && item && !moved.has(target)) {
          moved.add(target)
          const id = `${target}-margin`
          ref.properties.href = `#${id}`
          ref.properties.dataNote = id
          mark(ref, id)
          const body = item.children.filter(
            c => !(c.type === 'text' && !c.value.trim()),
          ) as ElementContent[]
          visit({ type: 'root', children: body } as Root, 'element', (el, at, owner) => {
            if (el.properties.dataFootnoteBackref === undefined || !owner || at === undefined)
              return
            owner.children.splice(at, 1)
            return [SKIP, at]
          })
          const aside = note(id, 'footnote', toString(ref).trim(), body)
          pending.set(here, [...(pending.get(here) ?? []), aside])
          node.children.splice(i, 0, cite(aside))
          i++
          continue
        }
      }

      walk(child, here)
      const asides = pending.get(child)
      if (!asides) continue
      pending.delete(child)
      if (child.tagName === 'li') child.children.push(...asides)
      else {
        node.children.splice(i + 1, 0, ...asides)
        i += asides.length
      }
    }
  }
  walk(tree, null)

  // Collect TLDR notes at the box end, as margin or inline footnotes.
  visit(tree, 'element', box => {
    if (box.properties.id !== 'tldr') return
    const collected: Element[] = []
    visit(box, 'element', (el, at, owner) => {
      if (!classes(el).includes('profile-note') || !owner || at === undefined) return
      if (owner === box) return SKIP
      owner.children.splice(at, 1)
      collected.push(el)
      return [SKIP, at]
    })
    box.children.push(...collected)
    return SKIP
  })

  if (section && moved.size === footnotes.size) section.parent.children.splice(section.index, 1)

  chain(tree, marked)
}

const anchorId = (id: unknown) => String(id).replace(/[^\w-]/g, '-')

/** Zero-width citation anchor; the word joiner shares its line and prevents a break. */
function cite(aside: Element): Element {
  return h(
    'span.profile-note-cite',
    { ariaHidden: 'true', style: `anchor-name:--${anchorId(aside.properties.id)}-cite` },
    '\u2060',
  )
}

/** Link citation and prior-note anchors. Anchors target earlier notes in this or an enclosing
 * block, so TLDR notes chain within their box.
 */
function chain(tree: Root, marked: Set<string>) {
  const last = { any: '', left: '', right: '', box: '' }
  let sections = false
  let alternate = 0
  const link = (box: Element | null) => (el: Element) => {
    if (el.tagName === 'h2') sections = true
    if (el.properties.id === 'tldr' && el !== box) {
      visit(el, 'element', link(el))
      return SKIP
    }
    if (!classes(el).includes('profile-note') || classes(el).includes('profile-note-boxed')) return
    const name = `--${anchorId(el.properties.id)}`
    const vars = [`anchor-name:${name}`, `--cite:${name}-cite`]
    if (marked.has(String(el.properties.id))) vars.push(`--mark:${name}-mark`)
    if (box) {
      if (last.box) vars.push(`--prev:${last.box}`, `--prev-side:${last.box}`)
      // Right-rail notes after the box must still clear its notes.
      last.box = last.any = last.right = name
    } else {
      const side = sections && alternate++ % 2 === 1 ? 'left' : 'right'
      if (side === 'left') el.properties.dataSide = 'left'
      if (last.any) vars.push(`--prev:${last.any}`)
      if (last[side]) vars.push(`--prev-side:${last[side]}`)
      last.any = name
      last[side] = name
    }
    el.properties.style = vars.join(';')
    return SKIP
  }
  visit(tree, 'element', link(null))
}

declare module 'vfile' {
  interface DataMap {
    profile: boolean
    /** `separator:` chapter ornament and the page seed it is grown from; unset means none. */
    separator?: { kind: RuleKind; seed: string }
  }
}
