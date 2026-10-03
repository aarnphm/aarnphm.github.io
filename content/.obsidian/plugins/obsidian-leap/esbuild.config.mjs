import esbuild from 'esbuild'
import { builtinModules } from 'node:module'

const production = process.argv[2] === 'production'
const context = await esbuild.context({
  entryPoints: ['main.ts'],
  bundle: true,
  external: ['obsidian', '@codemirror/*', 'electron', ...builtinModules],
  format: 'cjs',
  target: 'es2022',
  outfile: 'main.js',
  minify: production,
  sourcemap: production ? false : 'inline',
  logLevel: 'info',
})

if (production) {
  await context.rebuild()
  await context.dispose()
} else {
  await context.watch()
}
