import { Element as HtmlElement } from 'hast'
import { toHtml } from 'hast-util-to-html'
import assert from 'node:assert'
import test, { describe } from 'node:test'
import { parseSmsHtmlFragment } from './sms'

describe('sms html fragment parsing', () => {
  test('renders inline html', () => {
    const { wrapperTagName, children } = parseSmsHtmlFragment('10<sup>15</sup> ops')
    assert.strictEqual(wrapperTagName, 'p')

    const html = toHtml(
      { type: 'element', tagName: wrapperTagName, properties: {}, children } as HtmlElement,
      { allowDangerousHtml: true },
    )
    assert.ok(html.includes('<sup>15</sup>'))
    assert.ok(!html.includes('&lt;sup&gt;'))
  })

  test('does not misparse angle-bracket autolinks as html tags', () => {
    const { wrapperTagName, children } = parseSmsHtmlFragment('<https://example.com>')
    const html = toHtml(
      { type: 'element', tagName: wrapperTagName, properties: {}, children } as HtmlElement,
      { allowDangerousHtml: true },
    )
    assert.ok(html.includes('https://example.com'))
    assert.ok(!html.includes('<https:'))
  })

  test('uses a div wrapper when a block element appears', () => {
    const { wrapperTagName, children } = parseSmsHtmlFragment('<div>one</div><div>two</div>')
    assert.strictEqual(wrapperTagName, 'div')

    const html = toHtml(
      { type: 'element', tagName: wrapperTagName, properties: {}, children } as HtmlElement,
      { allowDangerousHtml: true },
    )
    assert.ok(html.startsWith('<div'))
    assert.ok(html.includes('<div>one</div>'))
    assert.ok(html.includes('<div>two</div>'))
  })
})
