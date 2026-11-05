import type { Element } from 'hast'
import { h } from 'hastscript'
import type { QuartzPluginData } from '../../plugins/vfile'
import type { ReferenceDependency } from '../../util/references/dependencies'
import type { TranscludeScope } from './transclude-scope'
import { noteStackQuery } from '../../util/note-stack'
import { isFullSlug, resolveRelative } from '../../util/path'
import { referenceElement } from '../../util/references/render'
import { referenceHref } from '../../util/references/route'

export function transcludePageStatus(
  page: QuartzPluginData,
): 'protected' | 'unpublished' | undefined {
  if (page.protectedPassword !== undefined || page.protected || page.frontmatter?.protected)
    return 'protected'
  if (page.frontmatter?.private || page.frontmatter?.draft || page.frontmatter?.publish === false)
    return 'unpublished'
}

export function transcludeNoteStack(
  scope: TranscludeScope,
  node: Element,
  request: ReferenceDependency,
): boolean {
  const selector = request.selector
  if (selector?.kind !== 'note-stack') return false
  node.tagName = 'section'
  node.properties = {
    className: ['note-stack'],
    ariaLabel: request.label || 'note trail',
    ariaRoleDescription: ['note trail'],
  }
  const members = request.members ?? []
  if (!members.length) {
    node.children = [h('p.note-stack-unavailable', 'invalid note trail')]
    return true
  }
  const prefix = `note-stack-${scope.expansion.occurrence}`
  const entries = members.map((member, index) => {
    let id = `${prefix}-${index}`
    while (scope.expansion.ids.has(id)) id += '-1'
    scope.expansion.ids.add(id)
    const slug = member.resolvedSlug ?? selector.slugs[index]
    const page = isFullSlug(slug) ? scope.renderData.bySlug.get(slug) : undefined
    const unavailable = member.status !== 'found' || !page || transcludePageStatus(page)
    const title = unavailable ? 'unavailable note' : (page.frontmatter?.title ?? slug)
    const href = member.route ? referenceHref(scope.slug, member.route) : undefined
    const content = unavailable
      ? h('p.note-stack-unavailable', 'This note is unavailable.')
      : referenceElement(member)
    if (!unavailable) {
      content.data = { position: {}, ...content.data, noteStackMember: true }
      content.properties.className = ['transclude', 'note-stack-content']
      content.children.push(h('p.note-stack-unavailable', 'This note could not be expanded.'))
    }
    return {
      title,
      button: h(
        'button.note-stack-step',
        {
          type: 'button',
          'data-note-step': index,
          ariaControls: id,
          ariaLabel: `${index + 1} of ${members.length}: ${title}`,
          ...(index === 0 ? { ariaCurrent: 'step' } : {}),
        },
        title,
      ),
      slide: h(
        'section.note-stack-slide',
        {
          id,
          'data-note-slug': selector.slugs[index],
          role: 'group',
          ariaRoleDescription: 'note',
          ariaLabel: `${index + 1} of ${members.length}: ${title}`,
        },
        [
          h(
            'div.note-stack-heading',
            href && !unavailable ? [h('a.internal', { href }, title)] : [title],
          ),
          content,
        ],
      ),
    }
  })
  const first = members[0]
  const root = selector.slugs[0]
  const trailSlugs = members.map((member, index) => member.route?.slug ?? selector.slugs[index])
  const rootHref = first.route
    ? referenceHref(scope.slug, first.route)
    : isFullSlug(root)
      ? resolveRelative(scope.slug, root)
      : undefined
  node.children = [
    h('div.note-stack-toolbar', [
      h(
        'div.note-stack-steps',
        { role: 'group', ariaLabel: 'notes in this trail' },
        entries.map(entry => entry.button),
      ),
      h('div.note-stack-controls', [
        h(
          'button',
          {
            type: 'button',
            'data-note-previous': true,
            ariaLabel: 'previous note',
            ariaDisabled: 'true',
          },
          '←',
        ),
        h('span.note-stack-position', { 'data-note-position': true }, `1 / ${members.length}`),
        h(
          'button',
          {
            type: 'button',
            'data-note-next': true,
            ariaLabel: 'next note',
            ariaDisabled: String(members.length === 1),
          },
          '→',
        ),
        ...(rootHref
          ? [
              h(
                'a.internal.note-stack-open',
                {
                  href: `${rootHref}${noteStackQuery(trailSlugs)}`,
                  ariaLabel: 'open full note trail',
                },
                'open',
              ),
            ]
          : []),
      ]),
    ]),
    h(
      'div.note-stack-viewport',
      entries.map(entry => entry.slide),
    ),
    h('span.note-stack-status', { ariaLive: 'polite', ariaAtomic: 'true' }),
  ]
  return true
}

declare module 'hast' {
  interface Data {
    noteStackMember?: boolean
  }
}
