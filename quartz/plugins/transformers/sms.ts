import { Root as HtmlRoot, Element as HtmlElement } from 'hast'
import { fromHtml } from 'hast-util-from-html'
import { phrasing } from 'hast-util-phrasing'

export type SmsHtmlFragment = { wrapperTagName: 'p' | 'div'; children: HtmlElement['children'] }

const escapeInvalidHtmlStarts = (value: string) =>
  value.replace(/<(?!\/?[a-z][a-z0-9-]*(?:\s|\/?>))/gi, '&lt;')

export const parseSmsHtmlFragment = (value: string): SmsHtmlFragment => {
  const safeHtml = escapeInvalidHtmlStarts(value)
  const fragment = fromHtml(safeHtml, { fragment: true }) as HtmlRoot
  const children = fragment.children as HtmlElement['children']
  const wrapperTagName = children.some(child => child.type === 'element' && !phrasing(child))
    ? 'div'
    : 'p'
  return { wrapperTagName, children }
}
