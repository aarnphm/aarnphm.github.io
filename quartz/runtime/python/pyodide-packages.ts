const pyodideStdlibExtensionPackages = new Map<string, string>([
  ['hashlib', 'hashlib'],
  ['ssl', 'ssl'],
])

export function pyodideStdlibExtensionPackageForImport(name: string): string | undefined {
  const root = name.split('.')[0]
  if (!root) return undefined
  return pyodideStdlibExtensionPackages.get(root.toLowerCase())
}
