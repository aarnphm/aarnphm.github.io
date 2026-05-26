import assert from 'node:assert/strict'
import test from 'node:test'
import type { QuartzComponent } from '../types/component'
import type { ChangeEvent, QuartzEmitterPluginInstance } from '../types/plugin'
import type { BuildCtx } from './ctx'
import {
  affectedComponentPageEmitters,
  componentSourceNameCandidates,
  inheritComponentSourceNames,
  isComponentRenderSourcePath,
} from './component-source'

function component(name: string, sourceNames: readonly string[] = []): QuartzComponent {
  const Component: QuartzComponent = () => null
  Component.displayName = name
  Component.sourceNames = sourceNames
  return Component
}

function emitter(
  name: string,
  components: readonly QuartzComponent[],
): QuartzEmitterPluginInstance {
  return {
    name,
    async emit() {
      return []
    },
    getQuartzComponents() {
      return [...components]
    },
  }
}

function change(path: string): ChangeEvent {
  return { path, type: 'change' } as ChangeEvent
}

const ctx = {} as BuildCtx

test('component source names include composed children', () => {
  assert.deepEqual(inheritComponentSourceNames('Byline', [component('ContentMeta')]), [
    'Byline',
    'ContentMeta',
  ])
})

test('component source paths distinguish render changes from resource-only changes', () => {
  assert.equal(isComponentRenderSourcePath('quartz/components/ArticleTitle.tsx'), true)
  assert.equal(isComponentRenderSourcePath('quartz/components/scripts/search.inline.ts'), false)
  assert.deepEqual(componentSourceNameCandidates('quartz/components/pages/404.tsx'), [
    '404',
    'NotFound',
  ])
})

test('component source changes select emitters that use matching components', () => {
  const plan = affectedComponentPageEmitters(
    ctx,
    [
      emitter('ContentPage', [component('Byline', ['ContentMeta'])]),
      emitter('FolderPage', [component('FolderContent')]),
    ],
    [change('quartz/components/ContentMeta.tsx')],
  )

  assert.equal(plan.all, false)
  assert.deepEqual([...plan.names], ['ContentPage'])
})

test('global render components select all page emitters', () => {
  const plan = affectedComponentPageEmitters(
    ctx,
    [emitter('ContentPage', [component('Content')])],
    [change('quartz/components/Header.tsx')],
  )

  assert.equal(plan.all, true)
  assert.deepEqual([...plan.names], [])
})
