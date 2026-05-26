import type { QuartzComponent } from '../types/component'
import type { ChangeEvent, QuartzEmitterPluginInstance } from '../types/plugin'
import type { BuildCtx } from './ctx'

const componentPathPrefix = 'quartz/components/'
const componentResourcePathPrefixes = [
  'quartz/components/scripts/',
  'quartz/components/styles/',
  'quartz/components/multiplayer/',
]
const broadComponentPaths = new Set([
  'quartz/components/index.ts',
  'quartz/components/renderPage.tsx',
])
const globalRenderComponentNames = new Set(['Header', 'Headings'])
const componentSourceAliases = new Map<string, readonly string[]>([
  ['quartz/components/pages/404.tsx', ['NotFound']],
])

export type ComponentPageEmitterPlan = { all: boolean; names: Set<string> }

const unique = (names: Iterable<string>): string[] => [...new Set(names)]

const normalizeSourcePath = (sourcePath: string): string => sourcePath.replaceAll('\\', '/')

const presentName = (name: string | undefined): name is string =>
  typeof name === 'string' && name.length > 0 && name !== 'Component'

const sourceBaseName = (sourcePath: string): string => {
  const base = sourcePath.slice(sourcePath.lastIndexOf('/') + 1)
  return base.replace(/\.(tsx|ts|jsx|js|scss)$/, '').replace(/\.inline$/, '')
}

export function componentSourceNames(component: QuartzComponent): string[] {
  return unique(
    [component.displayName, component.name, ...(component.sourceNames ?? [])].filter(presentName),
  )
}

export function inheritComponentSourceNames(
  owner: string,
  components: readonly QuartzComponent[],
): string[] {
  return unique([owner, ...components.flatMap(componentSourceNames)])
}

export function isComponentSourcePath(sourcePath: string): boolean {
  const normalized = normalizeSourcePath(sourcePath)
  return (
    normalized.startsWith(componentPathPrefix) &&
    !normalized.endsWith('.test.ts') &&
    !normalized.endsWith('.test.tsx')
  )
}

export function isComponentRenderSourcePath(sourcePath: string): boolean {
  const normalized = normalizeSourcePath(sourcePath)
  return (
    isComponentSourcePath(normalized) &&
    !componentResourcePathPrefixes.some(prefix => normalized.startsWith(prefix))
  )
}

export function componentSourceNameCandidates(sourcePath: string): string[] {
  const normalized = normalizeSourcePath(sourcePath)
  if (!isComponentRenderSourcePath(normalized)) return []
  return unique([sourceBaseName(normalized), ...(componentSourceAliases.get(normalized) ?? [])])
}

export function affectedComponentPageEmitters(
  ctx: BuildCtx,
  emitters: readonly QuartzEmitterPluginInstance[],
  changeEvents: readonly ChangeEvent[],
): ComponentPageEmitterPlan {
  const renderChanges = changeEvents.filter(changeEvent =>
    isComponentRenderSourcePath(changeEvent.path),
  )
  if (renderChanges.length === 0) return { all: false, names: new Set() }

  if (
    renderChanges.some(changeEvent =>
      broadComponentPaths.has(normalizeSourcePath(changeEvent.path)),
    )
  ) {
    return { all: true, names: new Set() }
  }

  const candidates = new Set(
    renderChanges.flatMap(changeEvent => componentSourceNameCandidates(changeEvent.path)),
  )
  if ([...candidates].some(name => globalRenderComponentNames.has(name))) {
    return { all: true, names: new Set() }
  }
  if (candidates.size === 0) return { all: true, names: new Set() }

  const names = new Set<string>()
  for (const emitter of emitters) {
    const emitterComponentNames = new Set(
      (emitter.getQuartzComponents?.(ctx) ?? []).flatMap(componentSourceNames),
    )
    if ([...candidates].some(name => emitterComponentNames.has(name))) {
      names.add(emitter.name)
    }
  }

  return names.size > 0 ? { all: false, names } : { all: true, names: new Set() }
}
