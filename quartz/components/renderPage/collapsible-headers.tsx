import { Element, ElementContent } from 'hast'
import { headingRank } from 'hast-util-heading-rank'
import { h, s } from 'hastscript'
import { QuartzPluginData } from '../../plugins/vfile'
import { FullSlug } from '../../util/path'
import { svgOptions } from '../svg'

function headerElement(
  node: Element,
  content: ElementContent[],
  idx: number,
  endHr: boolean,
  hasNextLevelChild: boolean = false,
): Element {
  const buttonId = `collapsible-header-${node.properties?.id ?? idx}`

  const id = node.properties?.id ?? idx
  const rank = headingRank(node) as number

  const originalChildren = [...node.children]

  const anchorIndex = originalChildren.findIndex(
    child => child.type === 'element' && (child as Element).tagName === 'a',
  )
  const anchorChild =
    anchorIndex >= 0 ? (originalChildren.splice(anchorIndex, 1)[0] as ElementContent) : undefined

  const headingContent = originalChildren.filter(child => {
    if (child.type !== 'element') {
      return true
    }
    const element = child as Element
    const classes = element.properties?.className
    const normalized = Array.isArray(classes) ? classes : classes ? [classes] : []
    return !normalized.includes('toggle-button') && !normalized.includes('collapsed-dots')
  })

  const toggleButton = h(
    `span.toggle-button.collapse-toggle#${buttonId}-toggle`,
    {
      role: 'button',
      ariaExpanded: true,
      ariaLabel: 'Toggle content visibility',
      ariaControls: `${buttonId}-content`,
      type: 'button',
      ['data-collapse-toggle']: '',
    },
    [
      h('span.collapse-rail', { ariaHidden: true }, [
        h('span.collapse-line.collapse-line--before'),
        h('button.toggle-icons', [
          s(
            'svg',
            { ...svgOptions, fill: 'var(--dark)', stroke: 'var(--dark)', class: 'circle-icon' },
            [s('use', { href: '#circle-icon' })],
          ),
          s(
            'svg',
            { ...svgOptions, fill: 'var(--iris)', stroke: 'var(--iris)', class: 'expand-icon' },
            [s('use', { href: '#arrow-down' })],
          ),
          s(
            'svg',
            { ...svgOptions, fill: 'var(--foam)', stroke: 'var(--foam)', class: 'collapse-icon' },
            [s('use', { href: '#arrow-up' })],
          ),
        ]),
        h('span.collapse-line.collapse-line--after'),
      ]),
      h('span.collapse-title', hasNextLevelChild ? { class: 'has-next-level-child' } : {}, [
        ...headingContent,
        s('svg', { ...svgOptions, class: 'collapsed-dots' }, [s('use', { href: '#triple-dots' })]),
        ...(anchorChild ? [anchorChild] : []),
      ]),
    ],
  )

  node.children = [toggleButton]

  let className = ['collapsible-header']
  if (endHr) {
    className.push('end-hr')
  }

  return h(`section.${className.join('.')}#${id}`, { 'data-level': rank }, [
    h(
      'div.collapse-shell.is-open',
      { ['data-collapse-shell']: '', ['data-initial-open']: 'true' },
      [
        node,
        h(
          'div.collapse-body',
          {
            id: `${buttonId}-content`,
            role: 'region',
            ariaLabelledby: `${buttonId}-toggle`,
            ['data-collapse-body']: '',
          },
          [
            h('span.collapse-rail.collapse-rail--body', { ariaHidden: true }, [
              h('span.collapse-line.collapse-line--body'),
            ]),
            h('div.collapse-body-content', [
              h(
                '.collapsible-header-content',
                {
                  ['data-level']: `${rank}`,
                  ['data-heading-id']: node.properties.id, // HACK: This assumes that rehype-slug already runs this target
                },
                content,
              ),
            ]),
          ],
        ),
      ],
    ),
  ])
}

function spacerElement(): Element {
  return h('div.collapsible-header-spacer', { ariaHidden: true }, [
    h('span.collapse-rail', { ariaHidden: true }, [h('span.collapse-line.collapse-line--spacer')]),
  ])
}

function hrElement(): Element {
  return h('div.collapsible-header-hr', { ariaHidden: true }, [
    h('span.collapse-rail', { ariaHidden: true }, [h('span.collapse-line.collapse-line--hr')]),
    h('hr'),
  ])
}

function isCollapsibleHeader(node: ElementContent): boolean {
  if (node.type !== 'element') return false
  const element = node as Element
  if (element.tagName !== 'section') return false
  const classNames = element.properties?.className
  const normalized = Array.isArray(classNames) ? classNames : classNames ? [classNames] : []
  return normalized.includes('collapsible-header')
}

function shouldStopWrapping(node: ElementContent) {
  if (node.type === 'element') {
    if (
      node.properties?.dataReferences === '' ||
      node.properties?.dataFootnotes === '' ||
      node.properties?.dataBacklinks === ''
    ) {
      return true
    }
    if (node.tagName === 'hr') {
      return true
    }
  }
  return false
}

interface StackElement {
  level: number
  element: Element
  content: ElementContent[]
  hasNextLevelChild?: boolean
}

export function processHeaders(nodes: ElementContent[]): ElementContent[] {
  let result: ElementContent[] = []

  let stack: StackElement[] = []
  for (const node of nodes) {
    if (shouldStopWrapping(node)) {
      const endHr = (node as Element).tagName === 'hr'
      while (stack.length > 0) {
        const completedSection = stack.pop()!
        const wrappedElement = headerElement(
          completedSection.element,
          completedSection.content,
          0,
          endHr,
          completedSection.hasNextLevelChild ?? false,
        )
        if (stack.length > 0) {
          const parentContent = stack[stack.length - 1].content
          if (
            parentContent.length > 0 &&
            isCollapsibleHeader(parentContent[parentContent.length - 1])
          ) {
            parentContent.push(spacerElement())
          }
          parentContent.push(wrappedElement)
        } else {
          if (result.length > 0 && isCollapsibleHeader(result[result.length - 1])) {
            result.push(spacerElement())
          }
          result.push(wrappedElement)
        }
      }
      if (endHr) {
        if (result.length > 0 && isCollapsibleHeader(result[result.length - 1])) {
          result.push(spacerElement())
        }
        result.push(hrElement())
      } else {
        result.push(node)
      }
    } else if (node.type === 'element' && headingRank(node)) {
      const level = headingRank(node) as number

      while (stack.length > 0 && stack[stack.length - 1].level >= level) {
        const completedSection = stack.pop()!
        const wrappedElement = headerElement(
          completedSection.element,
          completedSection.content,
          0,
          false,
          completedSection.hasNextLevelChild ?? false,
        )
        if (stack.length > 0) {
          const parentContent = stack[stack.length - 1].content
          if (
            parentContent.length > 0 &&
            isCollapsibleHeader(parentContent[parentContent.length - 1])
          ) {
            parentContent.push(spacerElement())
          }
          parentContent.push(wrappedElement)
        } else {
          if (result.length > 0 && isCollapsibleHeader(result[result.length - 1])) {
            result.push(spacerElement())
          }
          result.push(wrappedElement)
        }
      }

      if (stack.length > 0 && stack[stack.length - 1].level === level - 1) {
        stack[stack.length - 1].hasNextLevelChild = true
      }

      stack.push({ level, element: node as Element, content: [] })
    } else {
      if (stack.length > 0) {
        stack[stack.length - 1].content.push(node)
      } else {
        result.push(node)
      }
    }
  }

  while (stack.length > 0) {
    const completedSection = stack.pop()!
    const wrappedElement = headerElement(
      completedSection.element,
      completedSection.content,
      0,
      false,
      completedSection.hasNextLevelChild ?? false,
    )
    if (stack.length > 0) {
      const parentContent = stack[stack.length - 1].content
      if (
        parentContent.length > 0 &&
        isCollapsibleHeader(parentContent[parentContent.length - 1])
      ) {
        parentContent.push(spacerElement())
      }
      parentContent.push(wrappedElement)
    } else {
      if (result.length > 0 && isCollapsibleHeader(result[result.length - 1])) {
        result.push(spacerElement())
      }
      result.push(wrappedElement)
    }
  }

  return result
}

function hasCssClass(fileData: QuartzPluginData, className: string): boolean {
  return fileData.frontmatter?.cssclasses?.includes(className) ?? false
}

export function shouldProcessHeaders(
  fileData: QuartzPluginData,
  dynalist: boolean,
  slug: FullSlug,
): boolean {
  if (!dynalist || slug.includes('posts')) return false
  if (fileData.frontmatter?.collapseHeadings === false) return false
  return !hasCssClass(fileData, 'notebook-page')
}
