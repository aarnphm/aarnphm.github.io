import { build as bundle, type Plugin } from 'esbuild'
import fs from 'node:fs/promises'
import path from 'path'

export const clientHighlighterEntry = 'quartz/util/highlight-source.ts'
const clientHighlighterSpecifier = 'quartz:client-highlighter'

/** Keep the client highlighter lazy across both stages of inline-script bundling. */
export function externalizeClientHighlighter(url = clientHighlighterSpecifier): Plugin {
  return {
    name: 'shared-client-highlighter',
    setup(build) {
      build.onResolve(
        { filter: /(?:^quartz:client-highlighter$|(?:^|\/)highlight-source(?:\.ts)?$)/ },
        args => {
          const source = args.path.endsWith('.ts') ? args.path : `${args.path}.ts`
          if (
            args.path !== clientHighlighterSpecifier &&
            path.resolve(args.resolveDir, source) !== path.resolve(clientHighlighterEntry)
          )
            return
          if (args.kind !== 'dynamic-import')
            throw new Error('Client highlighting must be loaded through import()')
          return { path: url, external: true }
        },
      )
    },
  }
}

export async function bundleInlineScript(scriptPath: string): Promise<string> {
  let text = await fs.readFile(scriptPath, 'utf8')
  text = text.replace('export default', '')
  text = text.replace('export', '')

  const sourcefile = path.relative(path.resolve('.'), scriptPath)
  const transpiled = await bundle({
    stdin: {
      contents: text,
      loader: path.extname(scriptPath) === '.js' ? 'js' : 'ts',
      resolveDir: path.dirname(sourcefile),
      sourcefile,
    },
    write: false,
    bundle: true,
    minify: true,
    platform: 'browser',
    format: 'esm',
    // npm imports stay bare; script-assets.ts bundles them so page chunks share one copy
    packages: 'external',
    loader: { '.py': 'text' },
    plugins: [externalizeClientHighlighter()],
  })
  return transpiled.outputFiles[0].text
}
