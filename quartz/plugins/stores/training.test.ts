import type { Root as HtmlRoot } from 'hast'
import assert from 'node:assert/strict'
import test from 'node:test'
import { parseTrainingPlans } from './training'

function tree(): HtmlRoot {
  return {
    type: 'root',
    children: [
      {
        type: 'element',
        tagName: 'h2',
        properties: {},
        children: [{ type: 'text', value: 'before' }],
      },
      {
        type: 'comment',
        value:
          ' training plan start\nmeta: supertri toronto 2026\ndistance: olympic\ndate: 2026-07-26\ntarget: sub-3\n',
      },
      {
        type: 'element',
        tagName: 'p',
        properties: {},
        children: [{ type: 'text', value: 'body para' }],
      },
      {
        type: 'element',
        tagName: 'h3',
        properties: {},
        children: [{ type: 'text', value: 'weekly build' }],
      },
      {
        type: 'element',
        tagName: 'p',
        properties: {},
        children: [
          { type: 'text', value: 'most stay easy' },
          {
            type: 'element',
            tagName: 'sup',
            properties: {},
            children: [
              {
                type: 'element',
                tagName: 'a',
                properties: { href: '#user-content-fn-seiler', dataFootnoteRef: true },
                children: [{ type: 'text', value: '1' }],
              },
            ],
          },
        ],
      },
      { type: 'comment', value: ' training plan end ' },
      {
        type: 'element',
        tagName: 'p',
        properties: {},
        children: [{ type: 'text', value: 'after' }],
      },
      {
        type: 'element',
        tagName: 'section',
        properties: { dataFootnotes: '', className: ['footnotes'] },
        children: [
          {
            type: 'element',
            tagName: 'ol',
            properties: {},
            children: [
              {
                type: 'element',
                tagName: 'li',
                properties: { id: 'user-content-fn-seiler' },
                children: [{ type: 'text', value: 'Seiler 2010' }],
              },
            ],
          },
        ],
      },
    ],
  } as HtmlRoot
}

test('parseTrainingPlans extracts meta and renders the body slice', () => {
  const plans = parseTrainingPlans(tree())
  assert.equal(plans.length, 1)
  const p = plans[0]
  assert.equal(p.id, 'plan-0')
  assert.equal(p.meta, 'supertri toronto 2026')
  assert.equal(p.distance, 'olympic')
  assert.equal(p.date, '2026-07-26')
  assert.equal(p.target, 'sub-3')
  assert.match(p.html, /<p>body para<\/p>/)
  assert.match(p.html, /<h3>weekly build<\/h3>/)
})

test('parseTrainingPlans excludes content outside the markers', () => {
  const p = parseTrainingPlans(tree())[0]
  assert.doesNotMatch(p.html, /before/)
  assert.doesNotMatch(p.html, />after</)
})

test('parseTrainingPlans appends the footnotes section so refs resolve', () => {
  const p = parseTrainingPlans(tree())[0]
  assert.match(p.html, /href="#user-content-fn-seiler"/)
  assert.match(p.html, /id="user-content-fn-seiler"/)
  assert.match(p.html, /Seiler 2010/)
})

test('parseTrainingPlans returns empty when no markers present', () => {
  const t: HtmlRoot = {
    type: 'root',
    children: [
      { type: 'element', tagName: 'p', properties: {}, children: [{ type: 'text', value: 'x' }] },
    ],
  }
  assert.deepEqual(parseTrainingPlans(t), [])
})
