import { Element, ElementContent, Root, Text } from 'hast'
import { h, s } from 'hastscript'
import { EXIT, visit } from 'unist-util-visit'
import type { RenderTreeFeatures } from './transclude-scope'
import { checkBib, checkBibSection } from '../../plugins/transformers/citations'
import { checkFootnoteRef, checkFootnoteSection } from '../../plugins/transformers/gfm'
import { clone } from '../../util/clone'
import { svgOptions } from '../svg'

export function mergeReferences(root: Root, appendSuffix?: string | undefined): void {
  const finalRefs: Element[] = []
  const toRemove: Element[] = []

  // visit all references with bib to update suffix
  visit(
    root,
    //@ts-ignore
    (node: Element) => checkBib(node as Element),
    (node: Element) => {
      node.properties.href = `${(node as Element).properties.href}${appendSuffix !== undefined ? '-' + appendSuffix : ''}`
    },
  )

  // Find all reference divs and collect their entries
  visit(root, 'element', (node: Element) => {
    if (
      node.type === 'element' &&
      node.tagName === 'section' &&
      node.properties.dataReferences == ''
    ) {
      toRemove.push(node)
      const items = (node.children as Element[]).filter(val => val.tagName === 'ul')[0] // The ul is in here
      finalRefs.push(
        ...(items.children as Element[]).map(li => {
          li.properties.id = `${li.properties?.id}${appendSuffix ? '-' + appendSuffix : ''}`
          return li
        }),
      )
    }
  })

  // we don't want to remove the last nodes
  toRemove.pop()
  if (toRemove.length === 0) return

  // Remove all reference divs except the last one
  visit(root, 'element', (node: Element, index, parent) => {
    if (toRemove.includes(node)) {
      parent!.children.splice(index!, 1)
    }
  })

  // finally, update the final position
  visit(root, { tagName: 'section' }, (node: Element, index, parent) => {
    if (node.properties.dataReferences == '') {
      // @ts-ignore
      node.children[1].children = finalRefs
      parent!.children.splice(index as number, 1, node)
    }
  })
}

export const getFootnotesList = (node: Element) =>
  (node.children as Element[]).filter(val => val.tagName === 'ol')[0]

export const getBibList = (node: Element) =>
  (node.children as Element[]).filter(val => val.tagName === 'ul')[0]

export const isTrailingNoteSection = (node: ElementContent): boolean =>
  node.type === 'element' &&
  node.tagName === 'section' &&
  (node.properties?.dataFootnotes != null || node.properties?.dataReferences != null)

type FootnoteInfo = {
  originalHref: string
  index: number
  footnoteId: string
  referenceIds: string[]
}

export function mergeFootnotes(root: Root, appendSuffix?: string | undefined): void {
  const notesByHref = new Map<string, FootnoteInfo>()
  const noteOrder: FootnoteInfo[] = []
  const finalRefs: Element[] = []
  const toRemove: Element[] = []
  const suffixFragment = appendSuffix ? `-${appendSuffix}` : ''

  visit(
    root,
    // @ts-ignore
    (node: Element) => {
      if (checkFootnoteRef(node)) {
        const originalHref = node.properties.href as string
        let info = notesByHref.get(originalHref)
        if (!info) {
          const index = notesByHref.size + 1
          info = {
            originalHref,
            index,
            footnoteId: `fn-${index}${suffixFragment}`,
            referenceIds: [],
          }
          notesByHref.set(originalHref, info)
          noteOrder.push(info)
        }

        const refId = `fnref-${info.index}-${info.referenceIds.length + 1}${suffixFragment}`
        info.referenceIds.push(refId)

        node.properties.href = `#${info.footnoteId}`
        node.properties.id = refId

        const current = info
        if (current) {
          visit(node, 'text', (textNode: Text) => {
            textNode.value = `${current.index}`
          })
        }
      }
    },
  )

  visit(root, function (node) {
    if (checkFootnoteSection(node as Element)) {
      toRemove.push(node as Element)
      finalRefs.push(...(getFootnotesList(node as Element).children as Element[]))
    }
  })

  // we don't want to remove the last nodes
  toRemove.pop()
  if (noteOrder.length === 0) return

  // Remove all reference divs except the last one
  visit(root, { tagName: 'section' }, (node: Element, index, parent) => {
    if (toRemove.includes(node)) {
      parent!.children.splice(index as number, 1)
    }
  })

  const sortedRefs: Element[] = []
  const seenOriginal = new Set<string>()

  for (const note of noteOrder) {
    const originalId = note.originalHref.replace('#', '')
    if (seenOriginal.has(originalId)) {
      continue
    }
    const refIdx = finalRefs.findIndex(ref => ref.properties?.id === originalId)
    if (refIdx === -1) {
      continue
    }
    const ref = finalRefs[refIdx]
    seenOriginal.add(originalId)

    ref.properties = ref.properties ?? {}
    ref.properties.id = note.footnoteId

    const anchorsToRemove: { parent: Element; index: number }[] = []
    visit(ref, 'element', (child: Element, index, parent) => {
      if (child.tagName === 'a' && child.properties?.dataFootnoteBackref === '') {
        anchorsToRemove.push({ parent: parent as Element, index: index as number })
      }
    })

    anchorsToRemove.sort((a, b) => b.index - a.index)
    for (const { parent, index } of anchorsToRemove) {
      parent.children.splice(index, 1)
      const maybeText = parent.children[index - 1] as Text | undefined
      if (maybeText && maybeText.type === 'text' && maybeText.value.trim() === '') {
        parent.children.splice(index - 1, 1)
      }
    }

    let container: Element = ref
    for (let i = ref.children.length - 1; i >= 0; i--) {
      const child = ref.children[i]
      if (child.type === 'element') {
        container = child as Element
        break
      }
    }

    note.referenceIds.forEach((refId, ordinal) => {
      if (container.children.length > 0) {
        container.children.push({ type: 'text', value: ' ' } as Text)
      }
      container.children.push(
        h(
          'a',
          { href: `#${refId}`, dataFootnoteBackref: '', ariaLabel: 'Back to content' },
          `↩︎${ordinal === 0 ? '' : ordinal + 1}`,
        ) as Element,
      )
    })

    sortedRefs.push(ref)
  }

  // finally, update the final position
  visit(root, { tagName: 'section' }, (node: Element) => {
    if (node.properties.dataFootnotes == '') {
      // HACK: The node.children will have length 4, and ol is the 3rd items
      const ol = node.children[2] as Element
      ol.children = sortedRefs
    }
  })
}

/** support transcluding footnote and bib data */
export function appendTranscludedNotes(
  root: Root,
  features: RenderTreeFeatures,
  node: Element,
  pageTree: Root,
): void {
  let footnoteSection: Element | undefined = undefined
  let bibSection: Element | undefined = undefined
  visit(root, node => {
    if (checkFootnoteSection(node as Element)) {
      footnoteSection = node as Element
      return EXIT
    } else if (checkBibSection(node as Element)) {
      bibSection = node as Element
      return EXIT
    }
  })

  const transcludeFootnoteBlock: Element[] = []
  const transcludeBibBlock: Element[] = []

  visit(node, function (el) {
    const node = el as Element
    const { properties } = node
    if (checkFootnoteRef(node)) {
      visit(pageTree, { tagName: 'section' }, node => {
        if (node.properties?.dataFootnotes != null) {
          const noteId = (properties.href! as string).replace('#', '')
          transcludeFootnoteBlock.push(
            getFootnotesList(node).children.find(
              ref => (ref as Element).properties?.id === noteId,
            ) as Element,
          )
        }
      })
    } else if (node.tagName === 'cite' && node.children) {
      const linkId = (
        (node.children as Element[]).find(v => v.tagName === 'a')!.properties.href as string
      ).replace('#', '')
      visit(pageTree, { tagName: 'section' }, node => {
        if (node.properties.dataReferences == '') {
          transcludeBibBlock.push(
            getBibList(node).children.find(
              ref => (ref as Element).properties?.id === linkId,
            ) as Element,
          )
        }
      })
    }
  })

  // Footer packing rewrites IDs and backlinks, so each host needs private nodes.
  const filteredFootnotes = transcludeFootnoteBlock
    .filter((ref): ref is Element => ref !== undefined && ref !== null)
    .map(ref => clone(ref))
  const filteredBibs = transcludeBibBlock
    .filter((ref): ref is Element => ref !== undefined && ref !== null)
    .map(ref => clone(ref))

  if (filteredFootnotes.length !== 0) {
    if (!footnoteSection) {
      footnoteSection = h(
        'section.footnotes.main-col',
        { dataFootnotes: '', dataTransclude: '' },
        h(
          'h2.sr-only#footnote-label',
          { dir: 'auto' },
          h('span.highlight-span', [{ type: 'text', value: 'remarque' }]),
          h(
            'a.internal#footnote-label',
            { 'data-role': 'anchor', 'data-no-popover': 'true' },
            s(
              'svg',
              { ...svgOptions, fill: 'none', stroke: 'currentColor', strokeWidth: '2' },
              s('use', { href: '#github-anchor' }),
            ),
          ),
        ),
        { type: 'text', value: '\n' },
        h('ol', { dir: 'auto' }, [...filteredFootnotes]),
        { type: 'text', value: '\n' },
      )
      root.children.push(footnoteSection)
      features.hasFootnotes = true
    } else {
      visit(footnoteSection, { tagName: 'ol' }, (node: Element) => {
        node.children.push(...filteredFootnotes)
      })
    }
  }
  if (filteredBibs.length !== 0) {
    if (!bibSection) {
      bibSection = h(
        'section.bibliography.main-col',
        { dataReferences: '', dataTransclude: '' },
        h(
          'h2#reference-label',
          { dir: 'auto' },
          h('span.highlight-span', [{ type: 'text', value: 'bibliographie' }]),
        ),
        { type: 'text', value: '\n' },
        h('ul', { dir: 'auto' }, [...filteredBibs]),
        { type: 'text', value: '\n' },
      )
      root.children.push(bibSection)
      features.hasReferences = true
    } else {
      visit(bibSection, { tagName: 'ul' }, (node: Element) => {
        node.children.push(...filteredBibs)
      })
    }
  }
}
