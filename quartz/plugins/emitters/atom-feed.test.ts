import { XMLParser, XMLValidator } from 'fast-xml-parser'
import assert from 'node:assert/strict'
import test from 'node:test'
import { isJsonObject } from '../../util/type-guards'
import {
  atomFeedDocument,
  atomFeedStylesheet,
  atomFeedStylesheetHref,
  xhtmlNamespace,
} from './atom-feed'

const parser = new XMLParser({ ignoreAttributes: false })
const stylesheetParser = new XMLParser({ ignoreAttributes: false, removeNSPrefix: true })

test('atom feed document embeds the browser polyfill as a foreign child', () => {
  const document = atomFeedDocument('  <title>notes</title>', {
    stylesheetHref: atomFeedStylesheetHref,
    polyfillSrc: '/static/scripts/xslt-polyfill-deadbeef.js',
  })

  assert.equal(XMLValidator.validate(document), true)
  const parsed: unknown = parser.parse(document)
  assert.ok(isJsonObject(parsed))
  const feed = parsed.feed
  assert.ok(isJsonObject(feed))
  const script = feed.script
  assert.ok(isJsonObject(script))
  assert.equal(script['@_xmlns'], xhtmlNamespace)
  assert.equal(script['@_src'], '/static/scripts/xslt-polyfill-deadbeef.js')
  assert.equal(feed.title, 'notes')
})

test('atom feed stylesheet references the emitted site styles', () => {
  const stylesheetXml = atomFeedStylesheet({
    indexStylesheetHref: '/index-deadbeef.css',
    componentStylesheetHref: '/static/component-deadbeef.css',
  })

  assert.equal(XMLValidator.validate(stylesheetXml), true)
  const parsed: unknown = stylesheetParser.parse(stylesheetXml)
  assert.ok(isJsonObject(parsed))
  const stylesheet = parsed.stylesheet
  assert.ok(isJsonObject(stylesheet))
  assert.equal(stylesheet['@_version'], '1.0')
  assert.ok(Array.isArray(stylesheet.template))
  const rootTemplate: unknown = stylesheet.template[0]
  assert.ok(isJsonObject(rootTemplate))
  assert.ok(isJsonObject(rootTemplate.html))
  assert.ok(isJsonObject(rootTemplate.html.head))
  assert.ok(Array.isArray(rootTemplate.html.head.link))
  const hrefs = rootTemplate.html.head.link.map((link: unknown) => {
    assert.ok(isJsonObject(link))
    return link['@_href']
  })
  assert.deepEqual(hrefs, [
    '/static/icon.webp',
    '/index-deadbeef.css',
    '/static/component-deadbeef.css',
  ])
})
