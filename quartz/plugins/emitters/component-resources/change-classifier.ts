export type SourceRebuildScope =
  | { kind: 'full' }
  | { kind: 'partial'; staticFiles: boolean; styles: boolean; scripts: boolean }
  | { kind: 'courses' }
  | { kind: 'emitters'; emitters: string[] }

const clientScriptRoots = [
  'quartz/components/',
  'quartz/util/',
  'quartz/workers/',
  'quartz/runtime/',
]
const inlineScriptPattern = /\.inline\.[jt]s$/
const scriptPattern = /\.[jt]sx?$/

function isClientScriptChange(changePath: string, serverInputs: ReadonlySet<string>): boolean {
  // The build bundle loads an inline script as text, so its edits only change page chunks.
  if (inlineScriptPattern.test(changePath)) return true
  return (
    !serverInputs.has(changePath) &&
    scriptPattern.test(changePath) &&
    clientScriptRoots.some(root => changePath.startsWith(root))
  )
}

/**
 * Decides how much of the site a source edit invalidates. `serverInputs` is the esbuild metafile
 * closure of the build bundle (before and after the edit). A module outside it cannot change rendered
 * HTML, so it only needs the client bundles (ComponentResources, LazyScripts). A server module in
 * `courseInputs` renders only course pages, so it re-renders the courses. Anything unrecognised is
 * a full rebuild.
 */
export function sourceRebuildScope(
  changedPaths: readonly string[],
  serverInputs: ReadonlySet<string>,
  courseInputs: ReadonlySet<string> = new Set(),
  emitterInputs: ReadonlyMap<string, ReadonlySet<string>> = new Map(),
): SourceRebuildScope {
  if (changedPaths.length === 0) return { kind: 'full' }
  const scope = { kind: 'partial' as const, staticFiles: false, styles: false, scripts: false }
  let courses = false
  const emitters = new Set<string>()
  for (const changePath of changedPaths) {
    if (changePath.startsWith('quartz/static/')) {
      scope.staticFiles = true
    } else if (changePath.startsWith('quartz/') && changePath.endsWith('.scss')) {
      scope.styles = true
    } else if (isClientScriptChange(changePath, serverInputs)) {
      scope.scripts = true
    } else if (courseInputs.has(changePath)) {
      courses = true
    } else if (emitterInputs.get(changePath)?.size) {
      for (const name of emitterInputs.get(changePath) ?? []) emitters.add(name)
    } else {
      return { kind: 'full' }
    }
  }
  if (emitters.size > 0) {
    if (courses || scope.staticFiles || scope.styles || scope.scripts) return { kind: 'full' }
    return { kind: 'emitters', emitters: [...emitters].sort() }
  }
  // The course rebuild regenerates the client bundles, but not the static files.
  if (courses) return scope.staticFiles ? { kind: 'full' } : { kind: 'courses' }
  return scope
}
