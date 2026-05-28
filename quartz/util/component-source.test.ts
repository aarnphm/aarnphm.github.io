import assert from 'node:assert/strict'
import test from 'node:test'
import type { QuartzComponent } from '../types/component'
import { componentSourceNames, inheritComponentSourceNames } from './component-source'

function component(
  name: string,
  sourceNames: readonly string[] = [],
  displayName = name,
): QuartzComponent {
  const Component: QuartzComponent = () => null
  Component.displayName = displayName
  Component.sourceNames = sourceNames
  return Component
}

test('component source names include display name and explicit aliases', () => {
  assert.deepEqual(componentSourceNames(component('ContentMeta', ['Date', 'Date'])), [
    'ContentMeta',
    'Date',
  ])
})

test('component source names skip anonymous wrapper defaults', () => {
  assert.deepEqual(componentSourceNames(component('Component', ['Nested'], 'Component')), [
    'Nested',
  ])
})

test('component source names include composed children', () => {
  assert.deepEqual(inheritComponentSourceNames('Byline', [component('ContentMeta')]), [
    'Byline',
    'ContentMeta',
  ])
})
