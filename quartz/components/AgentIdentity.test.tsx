import assert from 'node:assert/strict'
import test from 'node:test'
import renderToString from 'preact-render-to-string'
import {
  HomepageIdentityHeading,
  PersonStructuredData,
  personStructuredData,
} from './AgentIdentity'

test('renders a homepage H1 without changing its visible layout contract', () => {
  assert.equal(
    renderToString(<HomepageIdentityHeading />),
    '<h1 class="agent-identity-heading">Aaron Pham\'s notes</h1>',
  )
})

test('publishes complete Person JSON-LD for the site identity', () => {
  const data = personStructuredData('aarnphm.xyz')
  assert.equal(data['@context'], 'https://schema.org')
  assert.equal(data['@type'], 'Person')
  assert.equal(data.name, 'Aaron Pham')
  assert.equal(data.url, 'https://aarnphm.xyz')
  assert.equal(data.contactPoint.email, 'contact@aarnphm.xyz')
  assert.ok(data.sameAs.includes('https://github.com/aarnphm'))

  const html = renderToString(<PersonStructuredData baseUrl="aarnphm.xyz" />)
  assert.match(html, /^<script type="application\/ld\+json">/)
  assert.match(html, /"@type":"Person"/)
  assert.match(html, /"contactPoint"/)
})
