import type { Argv } from './ctx'

export function isDevelopmentBuild(argv: Pick<Argv, 'watch' | 'force'>): boolean {
  return argv.watch && !argv.force
}
