import assert from 'node:assert/strict'
import test from 'node:test'
import { llmsIndex } from './llm'

test('publishes a proposal-compliant llms.txt with use cases and discovery links', () => {
  const content = llmsIndex('aarnphm.xyz')
  const lines = content.split('\n')

  assert.equal(lines[0], '# aarnphm.xyz')
  assert.ok(lines[2]?.startsWith('> '))
  assert.match(content, /## When to use this site/)
  assert.match(content, /Use the read-only MCP tools/)
  assert.match(content, /https:\/\/aarnphm\.xyz\/api\/docs/)
  assert.match(content, /https:\/\/aarnphm\.xyz\/openapi\.json/)
  assert.match(content, /https:\/\/aarnphm\.xyz\/\.well-known\/api-catalog/)
  assert.match(content, /https:\/\/aarnphm\.xyz\/about\.md/)

  for (const line of lines.filter(line => line.startsWith('- '))) {
    assert.match(line, /^- \[[^\]]+\]\(https:\/\/aarnphm\.xyz\/[^)]+\): .+/)
  }
})
