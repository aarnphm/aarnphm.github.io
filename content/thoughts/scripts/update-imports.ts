import * as path from 'path'
import { Project } from 'ts-morph'

/**
 * Codemod: replace "@/xyz" style imports and other `@` path aliases with a file-relative path.
 *
 * Path mappings are taken from tsconfig.json in this repo:
 *   "@/*"           -> "./quartz/*"
 *   "@config"       -> "./quartz.config.ts"
 *   "@layout"       -> "./quartz.layout.ts"
 *   "@package.json" -> "./package.json"
 *
 * Usage
 *   pnpm dlx tsx scripts/update-imports.ts
 */

const ROOT = process.cwd()

const pathMap: Record<string, string> = {
  '/': path.join(ROOT, 'quartz'), // handles "@/..."
  '@config': path.join(ROOT, 'quartz.config.ts'),
  '@layout': path.join(ROOT, 'quartz.layout.ts'),
  '@package.json': path.join(ROOT, 'package.json'),
}

function toPosix(p: string) {
  return p.split(path.sep).join('/')
}

function resolveAlias(moduleSpecifier: string, containingFileDir: string): string | null {
  if (moduleSpecifier.startsWith('@/')) {
    const rest = moduleSpecifier.slice(2) // drop "@/"
    const absTargetNoExt = path.join(pathMap['/'], rest)

    // Accept target without extension; TypeScript module resolution will pick correct file
    return makeRelative(absTargetNoExt, containingFileDir)
  }

  if (pathMap[moduleSpecifier]) {
    const absTarget = pathMap[moduleSpecifier]
    const absNoExt = absTarget.replace(/\.[jt]sx?$/, '')
    return makeRelative(absNoExt, containingFileDir)
  }

  return null
}

function makeRelative(absTargetNoExt: string, fromDir: string) {
  let rel = path.relative(fromDir, absTargetNoExt)
  if (!rel.startsWith('.')) {
    rel = './' + rel
  }
  return toPosix(rel)
}

function main() {
  const project = new Project({ tsConfigFilePath: path.join(ROOT, 'tsconfig.json') })
  const sourceFiles = project.getSourceFiles(['**/*.ts', '**/*.tsx', '**/*.mts', '**/*.cts'])

  const modified: string[] = []

  for (const sf of sourceFiles) {
    let changed = false
    const dir = path.dirname(sf.getFilePath())

    sf.getImportDeclarations().forEach(decl => {
      const spec = decl.getModuleSpecifierValue()
      const replacement = resolveAlias(spec, dir)
      if (replacement) {
        decl.setModuleSpecifier(replacement)
        changed = true
      }
    })

    sf.getExportDeclarations().forEach(decl => {
      if (!decl.getModuleSpecifierValue) return
      const spec = decl.getModuleSpecifierValue()
      if (!spec) return
      const replacement = resolveAlias(spec, dir)
      if (replacement) {
        decl.setModuleSpecifier(replacement)
        changed = true
      }
    })

    if (changed) {
      modified.push(toPosix(path.relative(ROOT, sf.getFilePath())))
    }
  }

  project.saveSync()

  console.log(`Updated ${modified.length} files:`)
  for (const file of modified) {
    console.log('  -', file)
  }
}

main()
