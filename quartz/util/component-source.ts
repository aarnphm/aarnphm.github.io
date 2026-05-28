import type { QuartzComponent } from '../types/component'

const unique = (names: Iterable<string>): string[] => [...new Set(names)]

const presentName = (name: string | undefined): name is string =>
  typeof name === 'string' && name.length > 0 && name !== 'Component'

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
