import type { Metafile } from 'esbuild'

// These are factory boundaries exported by the emitter registry. Imports through any other caller
// remain unscoped, including a build helper imported directly from an emitter module.
const emitterModules: Readonly<Record<string, readonly string[]>> = {
  '404.tsx': ['404Page'],
  'agent-skills.ts': ['AgentSkills'],
  'aliases.ts': ['AliasRedirects'],
  'arenaPage.tsx': ['ArenaPage'],
  'assets.ts': ['Assets'],
  'base.tsx': ['BaseViewPage'],
  'canvas.ts': ['CanvasPage'],
  'citations.ts': ['Bibliography'],
  'cname.ts': ['CNAME'],
  'componentResources.tsx': ['ComponentResources'],
  'contentIndex.tsx': ['ContentIndex'],
  'contentPage.tsx': ['ContentPage'],
  'courseReviewPage.tsx': ['CourseReviewPage'],
  'disabled-page.tsx': ['DisabledPage'],
  'email.tsx': ['EmailEmitter'],
  'favicon.ts': ['Favicon'],
  'flashcardsPage.tsx': ['FlashcardsPage'],
  'folderPage.tsx': ['FolderPage'],
  'lazy-scripts.ts': ['LazyScripts'],
  'llm.tsx': ['LLMText'],
  'masonry.ts': ['Masonry'],
  'notebook.ts': ['NotebookViewer'],
  'notebookManifest.ts': ['NotebookPagesManifest'],
  'ogImage.tsx': ['CustomOgImages', 'CustomOgImagesFallback'],
  'pdfReader.tsx': ['PdfReader'],
  'semantic.ts': ['SemanticIndex'],
  'slidesPage.tsx': ['SlidesPage'],
  'static.ts': ['Static'],
  'strava.tsx': ['Strava'],
  'streamIndex.ts': ['StreamIndex'],
  'streamPage.tsx': ['StreamPage'],
  'tagPage.tsx': ['TagPage'],
}
const registry = 'quartz/plugins/emitters/index.ts'
const prefix = 'quartz/plugins/emitters/'
export type EmitterSourceInputs = Map<string, Set<string>>

export function emitterSourceInputs(metafile: Pick<Metafile, 'inputs'>): EmitterSourceInputs {
  const { inputs } = metafile
  const owners: EmitterSourceInputs = new Map()
  const walk = (seeds: readonly string[], visit: (file: string) => void) => {
    const queue = [...seeds],
      visited = new Set<string>()
    while (queue.length > 0) {
      const file = queue.pop()
      if (!file || visited.has(file)) continue
      visited.add(file)
      visit(file)
      for (const imported of inputs[file]?.imports ?? []) {
        if (imported.external || !inputs[imported.path]) continue
        if (file === registry && emitterModules[imported.path.slice(prefix.length)]) continue
        queue.push(imported.path)
      }
    }
  }
  for (const [module, names] of Object.entries(emitterModules)) {
    const root = prefix + module
    if (!inputs[root]) continue
    walk([root], file => {
      const namesForFile = owners.get(file) ?? new Set<string>()
      for (const name of names) namesForFile.add(name)
      owners.set(file, namesForFile)
    })
  }
  // Everything reachable outside the registered factory imports can affect parsing, publication,
  // configuration, resources or orchestration. An empty owner set forces the full path.
  walk(['quartz/build.ts'], file => owners.set(file, new Set()))
  return owners
}

export function mergeEmitterSourceInputs(
  previous: ReadonlyMap<string, ReadonlySet<string>>,
  current: ReadonlyMap<string, ReadonlySet<string>>,
): EmitterSourceInputs {
  const merged: EmitterSourceInputs = new Map()
  for (const file of new Set([...previous.keys(), ...current.keys()])) {
    const before = previous.get(file),
      after = current.get(file)
    merged.set(
      file,
      before?.size === 0 || after?.size === 0
        ? new Set()
        : new Set([...(before ?? []), ...(after ?? [])]),
    )
  }
  return merged
}
