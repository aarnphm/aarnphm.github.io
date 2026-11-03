import type { QuartzComponent } from './types/component'
import type { Argv } from './util/ctx'
import { isDevelopmentBuild } from './util/build-mode'

export const devFeatures = { arena: false, slides: false, courses: false } as const

export type DevFeature = keyof typeof devFeatures

export function devFeatureEnabled(
  argv: Pick<Argv, 'watch' | 'force'>,
  feature: DevFeature,
): boolean {
  return devFeatures[feature] || !isDevelopmentBuild(argv)
}

export function argvFromProcess(
  argv: readonly string[] = process.argv,
): Pick<Argv, 'watch' | 'force'> {
  return {
    watch: argv.includes('--watch') || argv.includes('--serve'),
    force: argv.includes('--force'),
  }
}

export function coursesBundleEnabled(): boolean {
  return devFeatureEnabled(argvFromProcess(), 'courses')
}

export function isCourseContentPath(fp: string): boolean {
  const normalized = fp.replaceAll('\\', '/')
  const relative = normalized.startsWith('./') ? normalized.slice(2) : normalized
  return relative === 'courses' || relative.startsWith('courses/')
}

export function omitDisabledCourseContent<T extends string>(
  argv: Pick<Argv, 'watch' | 'force'>,
  files: T[],
): T[] {
  if (devFeatureEnabled(argv, 'courses')) return files
  return files.filter(file => !isCourseContentPath(file))
}

export function gateCourseComponent<T extends QuartzComponent>(component: T): T {
  if (coursesBundleEnabled()) return component
  const hidden: QuartzComponent = () => null
  hidden.displayName = component.displayName
  return hidden as T
}
