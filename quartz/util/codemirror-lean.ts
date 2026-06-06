import type { StreamParser, StringStream } from '@codemirror/language'

type LeanState = { commentDepth: number }

const keywords = new Set([
  'theorem',
  'lemma',
  'def',
  'abbrev',
  'example',
  'instance',
  'class',
  'structure',
  'inductive',
  'mutual',
  'where',
  'deriving',
  'extends',
  'namespace',
  'section',
  'end',
  'open',
  'import',
  'variable',
  'variables',
  'universe',
  'axiom',
  'attribute',
  'macro',
  'macro_rules',
  'syntax',
  'notation',
  'infixl',
  'infixr',
  'infix',
  'prefix',
  'postfix',
  'set_option',
  'noncomputable',
  'partial',
  'unsafe',
  'private',
  'protected',
  'local',
  'scoped',
  'fun',
  'do',
  'let',
  'have',
  'show',
  'suffices',
  'match',
  'with',
  'if',
  'then',
  'else',
  'from',
  'by',
  'calc',
  'return',
  'try',
  'catch',
  'finally',
  'for',
  'in',
  'while',
  'continue',
  'break',
  'Type',
  'Prop',
  'Sort',
])

const tactics = new Set([
  'intro',
  'intros',
  'apply',
  'exact',
  'refine',
  'rw',
  'rwa',
  'simp',
  'simp_all',
  'omega',
  'decide',
  'rfl',
  'constructor',
  'cases',
  'rcases',
  'obtain',
  'induction',
  'exists',
  'use',
  'left',
  'right',
  'split',
  'contradiction',
  'assumption',
  'trivial',
  'ring',
  'linarith',
  'nlinarith',
  'norm_num',
  'field_simp',
  'ext',
  'funext',
  'subst',
  'unfold',
  'change',
  'specialize',
  'generalize',
  'gcongr',
  'positivity',
  'aesop',
  'tauto',
])

const identifier = /^[A-Za-z_À-ɏͰ-Ͽ℀-⅏][A-Za-z0-9_'!?.À-ɏͰ-Ͽ₀-₉℀-⅏]*/
const operators = /^[+\-*/=<>|&!~^%@:.,;∀∃→↦λ∧∨¬↔≤≥≠∈∉⊆⊂∪∩×∘⟨⟩⊢⊕⊗∑∏√∂∇·…]+/

function eatBlockComment(stream: StringStream, state: LeanState): void {
  while (!stream.eol()) {
    if (stream.match('/-')) {
      state.commentDepth++
      continue
    }
    if (stream.match('-/')) {
      state.commentDepth--
      if (state.commentDepth === 0) return
      continue
    }
    stream.next()
  }
}

export const leanStreamParser: StreamParser<LeanState> = {
  name: 'lean',
  startState: () => ({ commentDepth: 0 }),
  token(stream, state) {
    if (state.commentDepth > 0) {
      eatBlockComment(stream, state)
      return 'comment'
    }
    if (stream.eatSpace()) return null
    if (stream.match('/-')) {
      state.commentDepth = 1
      eatBlockComment(stream, state)
      return 'comment'
    }
    if (stream.match('--')) {
      stream.skipToEnd()
      return 'comment'
    }
    if (stream.match('"')) {
      let escaped = false
      while (!stream.eol()) {
        const ch = stream.next()
        if (ch === '"' && !escaped) break
        escaped = !escaped && ch === '\\'
      }
      return 'string'
    }
    if (stream.match(/^@\[/)) return 'meta'
    if (stream.match(/^0[xX][0-9a-fA-F]+/) || stream.match(/^[0-9]+(\.[0-9]+)?/)) return 'number'
    const word = stream.match(identifier)
    if (word) {
      const text = Array.isArray(word) ? word[0] : ''
      if (text === 'sorry' || text === 'admit') return 'invalid'
      if (keywords.has(text)) return 'keyword'
      if (tactics.has(text)) return 'typeName'
      if (/^[A-Z]/.test(text)) return 'typeName'
      return 'variableName'
    }
    if (stream.match(operators)) return 'operator'
    stream.next()
    return null
  },
  languageData: { commentTokens: { line: '--', block: { open: '/-', close: '-/' } } },
}
