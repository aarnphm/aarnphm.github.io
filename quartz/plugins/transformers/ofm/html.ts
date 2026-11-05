import { Element, Literal, Root as HtmlRoot, Parent, Properties } from 'hast'
import { phrasing } from 'hast-util-phrasing'
import { toText } from 'hast-util-to-text'
import { whitespace } from 'hast-util-whitespace'
import { h, s } from 'hastscript'
import { PluggableList } from 'unified'
import { remove } from 'unist-util-remove'
import { SKIP, visit } from 'unist-util-visit'
import type { Options } from '../ofm'
import { youtubeTranscriptEmbed } from '../../../components/YouTubeTranscript'
import { AUDIO_ICON_PATHS, AUDIO_BAR_COUNT, audioBarHeight } from '../../../util/audio'
import { buildExternalEmbed } from '../../../util/embed'
import { labelToHast, labelToText, parseReferenceLabel } from '../../../util/references/label'
import { buildYouTubeEmbed } from '../../../util/youtube'
import { diagramMath } from '../pele-math'
import { groupImageGridHast } from './image-grid'
export function enableMarkdownTaskCheckboxes(tree: HtmlRoot): void {
  visit(tree, 'element', node => {
    if (
      node.tagName !== 'li' ||
      typeof (node.properties['data-task-marker'] ?? node.properties.dataTaskMarker) !== 'string'
    )
      return
    const head = node.children.find(child => child.type === 'element')
    const input = head?.tagName === 'p' ? head.children[0] : head
    if (
      input?.type === 'element' &&
      input.tagName === 'input' &&
      input.properties.type === 'checkbox'
    )
      input.properties.disabled = false
  })
}

const parseBooleanAttr = (value: unknown, fallback = false) => {
  if (typeof value === 'boolean') {
    return value
  }
  if (typeof value === 'string') {
    return value === '' || value.toLowerCase() === 'true' || value.toLowerCase() === 'checked'
  }
  return fallback
}

const sanitizeForId = (value: string) =>
  value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/(^-+)|(-+$)/g, '')

const collectInlineLabel = (parent: Parent | undefined, startIndex: number | null | undefined) => {
  if (!parent || typeof startIndex !== 'number') {
    return undefined
  }

  for (let i = startIndex + 1; i < parent.children.length; i++) {
    const sibling = parent.children[i]
    if (!sibling) {
      break
    }
    if (sibling.type === 'text') {
      const trimmed = sibling.value.trim()
      if (trimmed.length === 0) {
        continue
      }
      return trimmed
    }
    if (phrasing(sibling)) {
      const label = toText({ type: 'root', children: [sibling] }).trim()
      if (label.length > 0) {
        return label
      }
      continue
    }
    if (sibling.type === 'element' && sibling.tagName === 'br') {
      continue
    }
    break
  }
  return undefined
}

export const checkMermaidCode = ({ tagName, properties }: Element) =>
  tagName === 'code' &&
  Boolean(properties.className) &&
  (properties.className as string[]).includes('mermaid')

const blockReferenceRegex = new RegExp(/\^([-_A-Za-z0-9]+)$/g)
export const externalLinkRegex = /^https?:\/\//i
let stravaEmbedSeq = 0
export function createHtmlPlugins(opts: Options): PluggableList {
  const plugins: PluggableList = []

  if (opts.enableImageGrid) plugins.push(() => groupImageGridHast)

  if (opts.parseBlockReferences) {
    plugins.push(() => {
      const inlineTagTypes = new Set(['p', 'li'])
      const blockTagTypes = new Set(['blockquote'])
      return (tree: HtmlRoot, file) => {
        file.data.blocks = {}

        visit(tree, 'element', (node, index, parent) => {
          if (blockTagTypes.has(node.tagName)) {
            const nextChild = parent?.children.at(index! + 2) as Element
            if (nextChild && nextChild.tagName === 'p') {
              const text = nextChild.children.at(0) as Literal
              if (text && text.value && text.type === 'text') {
                const matches = text.value.match(blockReferenceRegex)
                if (matches && matches.length >= 1) {
                  parent!.children.splice(index! + 2, 1)
                  const block = matches[0].slice(1)

                  if (!Object.keys(file.data.blocks!).includes(block)) {
                    node.properties = { ...node.properties, id: block }
                    file.data.blocks![block] = node
                  }
                }
              }
            }
          } else if (inlineTagTypes.has(node.tagName)) {
            const last = node.children.at(-1) as Literal
            if (last && last.value && typeof last.value === 'string') {
              const matches = last.value.match(blockReferenceRegex)
              if (matches && matches.length >= 1) {
                last.value = last.value.slice(0, -matches[0].length)
                const block = matches[0].slice(1)

                if (last.value === '') {
                  // this is an inline block ref but the actual block
                  // is the previous element above it
                  let idx = (index ?? 1) - 1
                  while (idx >= 0) {
                    const element = parent?.children.at(idx)
                    if (!element) break
                    if (element.type !== 'element') {
                      idx -= 1
                    } else {
                      if (!Object.keys(file.data.blocks!).includes(block)) {
                        element.properties = { ...element.properties, id: block }
                        file.data.blocks![block] = element
                      }
                      return
                    }
                  }
                } else {
                  // normal paragraph transclude
                  if (!Object.keys(file.data.blocks!).includes(block)) {
                    node.properties = { ...node.properties, id: block }
                    file.data.blocks![block] = node
                  }
                }
              }
            }
          }
        })

        file.data.htmlAst = tree
      }
    })
  }

  if (opts.enableCheckbox) {
    plugins.push(() => {
      return (tree: HtmlRoot, file) => {
        let checkboxCounter = 0
        const slugValue =
          typeof file?.data?.slug === 'string' && file.data.slug.length > 0
            ? sanitizeForId(file.data.slug)
            : ''
        const slug = slugValue.length > 0 ? slugValue : 'global'
        const checkboxPrefix = `ofm-checkbox-${slug}`

        visit(tree, 'element', (node, index, parent) => {
          const typeProp = node.properties?.type
          const inputType = typeof typeProp === 'string' ? typeProp.toLowerCase() : typeProp

          if (node.tagName !== 'input' || inputType !== 'checkbox') {
            return
          }

          const properties: Properties = { ...node.properties }

          const existingId =
            typeof properties.id === 'string' && properties.id.trim().length > 0
              ? properties.id.trim()
              : undefined
          const checkboxId = existingId ?? `${checkboxPrefix}-${checkboxCounter++}`

          const classSet = new Set<string>()
          const registerClass = (value: unknown) => {
            if (typeof value === 'string') {
              value
                .split(/\s+/)
                .filter(Boolean)
                .forEach(entry => classSet.add(entry))
            } else if (Array.isArray(value)) {
              value
                .flatMap(entry =>
                  typeof entry === 'string' ? entry.split(/\s+/) : String(entry).split(/\s+/),
                )
                .filter(Boolean)
                .forEach(entry => classSet.add(entry))
            }
          }

          registerClass(properties.class)
          registerClass((properties as Record<string, unknown>).className)
          classSet.add('checkbox-toggle')

          const name =
            typeof properties.name === 'string' && properties.name.trim().length > 0
              ? properties.name.trim()
              : checkboxId

          const checked = parseBooleanAttr(properties.checked)
          const disabled = parseBooleanAttr(properties.disabled, false)

          node.properties = {
            ...properties,
            type: 'checkbox',
            id: checkboxId,
            name,
            checked,
            disabled,
            class: Array.from(classSet).join(' '),
            className: Array.from(classSet),
          }

          if (node.properties['aria-label'] == null && parent) {
            const parentNode = parent as Parent
            const position =
              typeof index === 'number'
                ? index
                : Array.isArray(parentNode.children)
                  ? parentNode.children.indexOf(node)
                  : -1
            const label = position >= 0 ? collectInlineLabel(parentNode, position) : undefined
            const finalLabel = label && label.length > 0 ? label : checkboxId
            if (finalLabel.length > 0) {
              node.properties['aria-label'] = finalLabel
            }
          }
        })
      }
    })
  }

  if (opts.enableYouTubeEmbed) {
    const checkEmbed = ({ tagName, properties }: Element) =>
      tagName === 'img' && Boolean(properties.src) && typeof properties.src === 'string'

    plugins.push(() => {
      return tree => {
        visit(tree, 'element', (node, _index, parent) => {
          if (!checkEmbed(node)) return

          const src = (node.properties.src ?? '') as string

          const youtube = buildYouTubeEmbed(src)
          if (youtube) {
            if (youtube.transcripts) {
              const transcript = youtubeTranscriptEmbed(youtube)
              if (transcript?.type === 'element') {
                Object.assign(node, transcript)
                if (parent?.type === 'element' && parent.tagName === 'p') parent.tagName = 'div'
              }
              return
            }
            node.tagName = 'iframe'
            node.properties = {
              class: 'external-embed youtube',
              allow:
                'accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share',
              frameborder: 0,
              width: '600px',
              referrerpolicy: 'strict-origin-when-cross-origin',
              allowfullscreen: true,
              src: youtube.src,
            }
            return
          }

          const embed = buildExternalEmbed(src)
          const marked =
            Array.isArray(node.properties.className) &&
            node.properties.className.includes('external-embed')
          if (!embed || !(embed.provider || marked)) return

          const { width, height } = node.properties
          const usable = (v: unknown) =>
            (typeof v === 'string' || typeof v === 'number') && v !== 'auto'

          node.tagName = 'iframe'
          node.properties = {
            class: embed.provider ? `external-embed ${embed.provider}` : 'external-embed',
            frameborder: 0,
            loading: 'lazy',
            referrerpolicy: 'no-referrer',
            src:
              embed.provider === 'strava'
                ? `${embed.src}#ns=garden-${(stravaEmbedSeq++).toString(36)}`
                : embed.src,
            ...(usable(width) ? { width } : {}),
            ...(usable(height) ? { height } : {}),
          }
        })
      }
    })
  }

  if (opts.mermaid) {
    plugins.push(() => {
      return (tree, file) => {
        const control = (
          className: string,
          label: string,
          paths: readonly string[],
          properties: Properties = {},
        ) =>
          h(
            `button.${className}`,
            { type: 'button', ariaLabel: label, title: label, ...properties },
            [
              s(
                'svg',
                { viewBox: '0 0 24 24', ariaHidden: true },
                paths.map(d => s('path', { d })),
              ),
            ],
          )
        visit(tree, 'element', (node: Element, _, parent) => {
          if (!checkMermaidCode(node) || parent?.type !== 'element' || parent.tagName !== 'pre')
            return
          parent.tagName = 'figure'
          parent.properties = { ...parent.properties, className: ['pele-figure'] }
          const source = node.children
            .map(child => (child.type === 'text' ? child.value : toText(child)))
            .join('')
          const math = diagramMath(source)
          if (math) file.data.hasMath = true
          parent.children = [
            control('pele-trigger', 'expand diagram', ['M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7'], {
              ariaHasPopup: 'dialog',
              disabled: true,
            }),
            control('pele-copy.pele-inline-copy', 'copy source', ['M9 9h11v11H9zM5 15H3V3h12v2']),
            h(
              '.pele-diagram',
              { 'data-clipboard': node.properties['data-clipboard'] },
              node.children,
            ),
            h('dialog.pele-dialog', { ariaLabel: 'expanded diagram' }, [
              h('.pele-controls', { role: 'group', ariaLabel: 'diagram controls' }, [
                control('pele-close', 'close diagram', ['M6 6l12 12M6 18 18 6'], {
                  autofocus: true,
                }),
                control('pele-fold', 'collapse diagram controls', ['M6 9l6 6 6-6'], {
                  ariaExpanded: 'true',
                }),
                h('.pele-controls-body', [
                  control('pele-zoom-in', 'zoom in', ['M12 5v14M5 12h14']),
                  control('pele-zoom-out', 'zoom out', ['M5 12h14']),
                  control('pele-reset', 'fit diagram', ['M8 3H3v5M16 3h5v5M3 16v5h5M21 16v5h-5']),
                  control('pele-copy', 'copy source', ['M9 9h11v11H9zM5 15H3V3h12v2']),
                ]),
              ]),
              h('.pele-dialog-content'),
            ]),
            ...(math ? [math] : []),
          ]
        })
      }
    })
  }

  plugins.push(() => {
    return (tree, file) => {
      const onlyImage = ({ children }: Element) =>
        children.every(child => (child as Element).tagName === 'img' || whitespace(child))
      const withAlt = ({ tagName, properties }: Element) =>
        tagName === 'img' && Boolean(properties.alt) && Boolean(properties.src)
      const withCaption = ({ tagName, children }: Element) => {
        return (
          tagName === 'figure' &&
          children.some(child => (child as Element).tagName === 'figcaption')
        )
      }

      // support better image captions
      visit(tree, { tagName: 'p' }, (node, idx, parent) => {
        if (!onlyImage(node)) return
        remove(node, 'text')
        if (node.properties.id) {
          node.tagName = 'div'
          return
        }
        parent?.children.splice(idx!, 1, ...node.children)
        return idx
      })

      file.data.images = {}
      let counter = 0

      visit(
        tree,
        node => withAlt(node as Element),
        (node, idx, parent) => {
          const caption = parseReferenceLabel(String(node.properties.alt ?? ''), { links: true })
          node.properties.alt = labelToText(caption)
          if (withCaption(parent as Element) || (parent as Element)!.tagName === 'a') {
            return
          }

          counter++
          const captionChildren = labelToHast(caption)
          parent?.children.splice(
            idx!,
            1,
            h('figure', { 'data-img-w-caption': true }, [
              h('img', { ...node.properties }),
              h('figcaption', [h('span', { class: 'figure-caption' }, ...captionChildren)]),
            ]),
          )
        },
      )
    }
  })

  plugins.push(() => {
    return tree => {
      const isAudioEmbed = (node: Element) =>
        node.tagName === 'audio' && typeof node.properties?.src === 'string'
      const onlyAudio = (node: Element) =>
        node.children.some(child => isAudioEmbed(child as Element)) &&
        node.children.every(child => isAudioEmbed(child as Element) || whitespace(child))
      const icon = (path: string) =>
        s('svg', { viewBox: '0 0 24 24', 'aria-hidden': 'true' }, [s('path', { d: path })])
      const bars = Array.from({ length: AUDIO_BAR_COUNT }, (_, i) =>
        h('span', { style: `height:${audioBarHeight(i)}%` }),
      )

      visit(tree, { tagName: 'p' }, (node, idx, parent) => {
        if (!onlyAudio(node as Element)) return
        remove(node, 'text')
        if (node.properties.id) {
          node.tagName = 'div'
          return
        }
        parent?.children.splice(idx!, 1, ...node.children)
        return idx
      })

      visit(
        tree,
        node => isAudioEmbed(node as Element),
        (node, idx, parent) => {
          const el = node as Element
          if ((parent as Element)?.tagName === 'figure') return
          const props = { ...el.properties }
          const caption =
            (props['data-caption'] as string | undefined) ??
            (props.dataCaption as string | undefined)
          delete props['data-caption']
          delete props.dataCaption
          parent?.children.splice(
            idx!,
            1,
            h('figure.audio-player', { 'data-audio-embed': true }, [
              h('div.ap-row', [
                h('button.ap-play', { type: 'button', 'aria-label': 'Play' }, [
                  icon(AUDIO_ICON_PATHS.play),
                ]),
                h(
                  'button.ap-repeat',
                  { type: 'button', 'aria-label': 'Repeat', 'aria-pressed': 'false' },
                  [icon(AUDIO_ICON_PATHS.repeat)],
                ),
                h('div.ap-bars', bars),
              ]),
              h('div.ap-time', '0:00 / 0:00'),
              ...(caption
                ? [
                    h(
                      'figcaption.ap-caption',
                      labelToHast(parseReferenceLabel(caption, { links: true })),
                    ),
                  ]
                : []),
              h('audio', props),
            ]),
          )
          return SKIP
        },
      )
    }
  })

  return plugins
}
