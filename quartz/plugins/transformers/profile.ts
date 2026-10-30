import type { Element, ElementContent, Root } from 'hast'
import { toString } from 'hast-util-to-string'
import { h } from 'hastscript'
import { imageSize } from 'image-size'
import isAbsoluteUrl from 'is-absolute-url'
import { createHash } from 'node:crypto'
import { readFileSync, statSync } from 'node:fs'
import path from 'node:path'
import { SKIP, visit } from 'unist-util-visit'
// @ts-ignore
import script from '../../components/scripts/profile-notes.inline'
import { QuartzTransformerPlugin } from '../../types/plugin'

/** A page opts in with `layout: newspaper` in its frontmatter. */
export const NEWSPAPER = 'newspaper' as const

/** Widths emitted by the ProfileFrames emitter; the original stays the largest candidate. */
export const PROFILE_WIDTHS = [480, 800, 1200] as const
export const PROFILE_DIR = 'static/profile'

// A photo of the author, by filename. Everything else (stills, art, memes) is a reference.
const portrait = /^(me-|selfie)/

// Column frames cap at 24rem, the opening pair splits a 52rem spread, references stay small.
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
  // Portraits are numbered plates; references carry only their caption, if any.
  const label = index === null ? [] : [h('span.profile-frame-no', String(index).padStart(2, '0'))]
  const content = [...label, ...(caption ?? [])]
  return h('figure.profile-frame', { dataKind: kind, dataCaptioned: caption ? '' : undefined }, [
    img,
    ...(content.length > 0 ? [h('figcaption', content)] : []),
  ])
}

/**
 * Lays a profile page out as a proof sheet: every local photo gets its intrinsic size, a resized
 * srcset, and a numbered frame. Portraits of the author are numbered; references are not.
 */
export const Profile: QuartzTransformerPlugin = () => {
  return {
    name: 'Profile',
    htmlPlugins(ctx) {
      return [
        () => (tree: Root, file) => {
          const slug = file.data.slug
          const frontmatter = file.data.frontmatter
          if (!slug || frontmatter?.pageLayout !== NEWSPAPER) return
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

            // Consecutive attestations read as one band of testimonials.
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

            // The first run of photos, before any heading, is the hero strip.
            if (node.tagName === 'h2' || node.tagName === 'section') hero = false

            if (
              node.tagName === 'div' &&
              (node.properties.className as string[] | undefined)?.includes('image-grid')
            ) {
              // Cells are bare embeds or captioned ones (figure > img + figcaption).
              const cells = node.children.flatMap(c => {
                if (c.type !== 'element') return []
                if (isPhoto(c)) return [{ img: c, caption: undefined }]
                // isPhoto's guard narrows c to never here; it is still an element.
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

            // Captioned embeds already arrive as figure > img + figcaption.
            if (node.tagName === 'figure' && node.properties.dataImgWCaption !== undefined) {
              const img = node.children.find(isPhoto)
              if (!img) return SKIP
              const kind = kindOf(img)
              const n = kind === 'me' ? ++count : null
              dress(img, kind, false)
              const caption = node.children.find(
                (c): c is Element => c.type === 'element' && c.tagName === 'figcaption',
              )
              const framed = frame(img, kind, n, caption?.children)
              // A hand-placed float wrapper keeps its side; the frame takes its place.
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
              const kind = kindOf(node)
              const n = kind === 'me' ? ++count : null
              dress(node, kind, false)
              const framed = frame({ ...node }, kind, n)
              // A lone photo inside a paragraph lifts the paragraph to a frame.
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

          // A float wrapper around a bare embed: the frame takes the wrapper's place and side.
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

// A rail is 13rem: anything longer than a short paragraph, or with a list or maths, reads as a
// boxed note in the body instead.
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

/**
 * Lifts sidenotes and footnotes out of the running text into `aside.profile-note`, placed right
 * after the block that cites them. CSS hangs them in the margin; on narrow screens they stay
 * inline. The sidenote script skips them because their content no longer follows the label.
 */
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

  // Asides cannot live inside a paragraph, so each one waits for its block to close.
  const pending = new Map<Element, Element[]>()
  const moved = new Set<string>()
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
        if (label?.type === 'element') label.properties.dataNote = id
        const aside = note(id, 'sidenote', word, child.children)
        pending.set(here, [...(pending.get(here) ?? []), aside])
        node.children.splice(i, 1)
        i--
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

  // The tldr is a boxed sidebar in two columns: its notes collect at the end of the box, where
  // they hang beside it when there is a margin and read as the box's own footnotes when not.
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
}

declare module 'vfile' {
  interface DataMap {
    profile: boolean
  }
}
