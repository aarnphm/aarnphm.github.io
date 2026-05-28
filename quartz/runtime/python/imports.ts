const ignoredImports = new Set([
  'import_ipynb',
  'ipython',
  'jax',
  'js',
  'nbimporter',
  'pyodide',
  'torch',
])

export function extractPythonImports(source: string): string[] {
  const names = new Set<string>()
  for (const line of source.split(/\r?\n/)) {
    const withoutComment = line.replace(/#.*/, '')
    for (const importMatch of withoutComment.matchAll(/(?:^|[;:])\s*import\s+([^;]+)/g)) {
      for (const part of importMatch[1].split(',')) {
        const name = part
          .trim()
          .split(/\s+|\./)[0]
          ?.replace(/\W+$/, '')
        if (name && /^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) names.add(name)
      }
    }
    for (const fromMatch of withoutComment.matchAll(
      /(?:^|[;:])\s*from\s+([A-Za-z_][A-Za-z0-9_]*)\b/g,
    )) {
      names.add(fromMatch[1])
    }
  }
  for (const name of names) {
    if (ignoredImports.has(name.toLowerCase())) names.delete(name)
  }
  return [...names]
}
