import type { ComponentType } from "preact"
import type { StringResource } from "../../util/resources"

export type QuartzMdxComponent<Props = any> = ComponentType<Props> & {
  css?: StringResource
  beforeDOMLoaded?: StringResource
  afterDOMLoaded?: StringResource
}

export type QuartzMdxConstructor<Options extends object | undefined = undefined, Props = any> = (
  opts: Options,
) => QuartzMdxComponent<Props>

const registry = new Map<string, QuartzMdxComponent>()

export function registerMdxComponent<Props = any>(
  names: string | string[],
  component: QuartzMdxComponent<Props>,
): QuartzMdxComponent<Props> {
  const resolved = Array.isArray(names) ? names : [names]
  for (const name of resolved) {
    registry.set(name, component as QuartzMdxComponent)
  }
  return component
}

export function getMdxComponent(name: string): QuartzMdxComponent | undefined {
  return registry.get(name)
}

export function getMdxComponentEntries(): [string, QuartzMdxComponent][] {
  return Array.from(registry.entries())
}

export function getMdxComponents(): QuartzMdxComponent[] {
  return Array.from(new Set(registry.values()))
}
