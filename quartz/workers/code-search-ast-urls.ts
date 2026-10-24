import javascript from 'tree-sitter-javascript/tree-sitter-javascript.wasm'
import python from 'tree-sitter-python/tree-sitter-python.wasm'
import tsx from 'tree-sitter-typescript/tree-sitter-tsx.wasm'
import typescript from 'tree-sitter-typescript/tree-sitter-typescript.wasm'
import runtime from 'web-tree-sitter/web-tree-sitter.wasm'

const grammars: Readonly<Record<string, string>> = { javascript, typescript, tsx, python }
export function grammarUrl(language: string): string | undefined {
  const asset = grammars[language === 'jsx' ? 'javascript' : language]
  return asset ? new URL(asset, import.meta.url).href : undefined
}
export const runtimeUrl = new URL(runtime, import.meta.url).href
