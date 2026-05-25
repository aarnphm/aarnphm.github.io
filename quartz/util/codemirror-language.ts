import type { Extension } from '@codemirror/state'
import { go } from '@codemirror/lang-go'
import { javascript } from '@codemirror/lang-javascript'
import { python } from '@codemirror/lang-python'
import { rust } from '@codemirror/lang-rust'
import { StreamLanguage } from '@codemirror/language'
import { haskell } from '@codemirror/legacy-modes/mode/haskell'
import { oCaml } from '@codemirror/legacy-modes/mode/mllike'
import { shell } from '@codemirror/legacy-modes/mode/shell'
import { zig } from 'codemirror-lang-zig'

export function codemirrorCodeLanguage(language: string | undefined): Extension {
  const name = (language ?? '').trim().toLowerCase()
  if (name === 'bash' || name === 'sh' || name === 'shell' || name === 'zsh') {
    return StreamLanguage.define(shell)
  }
  if (name === 'javascript' || name === 'js') return javascript()
  if (name === 'jsx') return javascript({ jsx: true })
  if (name === 'typescript' || name === 'ts') return javascript({ typescript: true })
  if (name === 'tsx') return javascript({ typescript: true, jsx: true })
  if (name === 'go' || name === 'golang') return go()
  if (name === 'rust' || name === 'rs') return rust()
  if (
    name === 'haskell' ||
    name === 'hs' ||
    name === 'lhs' ||
    name === 'ghc' ||
    name === 'runghc'
  ) {
    return StreamLanguage.define(haskell)
  }
  if (name === 'ocaml' || name === 'ml' || name === 'mli' || name === 'utop') {
    return StreamLanguage.define(oCaml)
  }
  if (name === 'zig') return zig()
  if (name === 'python' || name === 'py' || name === 'python3' || name === 'mojo') return python()
  return python()
}
