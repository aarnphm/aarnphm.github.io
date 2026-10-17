import { CompletionContext, type CompletionResult } from '@codemirror/autocomplete'
import { EditorState, type Extension } from '@codemirror/state'
import assert from 'node:assert/strict'
import test, { describe } from 'node:test'
import { leanLanguageExtension } from '../../util/codemirror-lean'
import { leanBackend } from '../lean/backend'
import { getSharedLeanWebSocketSession } from '../lean/kernel'
import {
  extractLeanDocumentSymbols,
  leanCompletionSource,
  leanLspBridge,
  leanLspConfigFacet,
  renderLeanHoverHtml,
  resolveLeanHoverAt,
} from './lean'

async function runCompletion(
  doc: string,
  pos = doc.length,
  explicit = false,
  extensions: readonly Extension[] = [],
): Promise<CompletionResult | null> {
  const state = EditorState.create({ doc, extensions: [...extensions] })
  const context = new CompletionContext(state, pos, explicit)
  return await leanCompletionSource(context)
}

describe('Lean 4 CompletionSource & Mathlib/Tactic Index', () => {
  test('delegates backslash prefixes (\\alpha, \\to, \\sub) to Lean Unicode symbol completion', async () => {
    const alphaRes = await runCompletion('theorem t : \\alpha')
    assert.ok(alphaRes)
    assert.ok(alphaRes.options.some(opt => opt.label === '\\alpha' && opt.detail === 'α'))

    const toRes = await runCompletion('def f : α \\to')
    assert.ok(toRes)
    assert.ok(toRes.options.some(opt => opt.label === '\\to' && opt.detail === '→'))

    const subRes = await runCompletion('example (A B : Set α) : A \\sub')
    assert.ok(subRes)
    assert.ok(
      subRes.options.some(
        opt => (opt.label === '\\sub' || opt.label === '\\subset') && opt.detail === '⊆',
      ),
    )
  })

  test('autocompletes Lean 4 tactics and # commands with signatures and documentation', async () => {
    const rcasRes = await runCompletion('  rcas')
    assert.ok(rcasRes)
    const rcasesOpt = rcasRes.options.find(opt => opt.label === 'rcases')
    assert.ok(rcasesOpt)
    assert.match(rcasesOpt.detail ?? '', /rcases/)

    const obtRes = await runCompletion('  obt')
    assert.ok(obtRes)
    assert.ok(obtRes.options.some(opt => opt.label === 'obtain'))

    const rwRes = await runCompletion('  rw')
    assert.ok(rwRes)
    assert.equal(rwRes.from, 2)
    assert.ok(rwRes.options.some(opt => opt.label === 'rw'))
    assert.ok(rwRes.options.some(opt => opt.label === 'rwa'))

    const cmdRes = await runCompletion('#ch')
    assert.ok(cmdRes)
    assert.ok(cmdRes.options.some(opt => opt.label === '#check'))
  })

  test('autocompletes dotted Mathlib namespaces (Set.mem_* and Function.*)', async () => {
    const setMemRes = await runCompletion('  rw [Set.mem_')
    assert.ok(setMemRes)
    const labels = new Set(setMemRes.options.map(opt => opt.label))
    assert.ok(labels.has('Set.mem_union'))
    assert.ok(labels.has('Set.mem_inter_iff'))
    assert.ok(labels.has('Set.mem_diff'))
    assert.ok(labels.has('Set.mem_compl_iff'))
    assert.ok(labels.has('Set.mem_setOf'))
    assert.ok(!labels.has('rcases'))

    const fnRes = await runCompletion('example (f : α → β) : Function.')
    assert.ok(fnRes)
    const fnLabels = new Set(fnRes.options.map(opt => opt.label))
    assert.ok(fnLabels.has('Function.Injective'))
    assert.ok(fnLabels.has('Function.Surjective'))
    assert.ok(fnLabels.has('Function.Bijective'))
    assert.ok(fnLabels.has('Function.comp'))
  })

  test('extracts and autocompletes document-local theorems, definitions, and proof hypotheses across cells', async () => {
    const doc = [
      'theorem cantor_diag {α : Type*} (f : α → Set α) : ¬ Function.Surjective f := by',
      '  intro hsurj',
      '  let diagSet : Set α := {a | a ∉ f a}',
      '  obtain ⟨witness_d, h_witness⟩ := hsurj diagSet',
      '  have hiff_self : witness_d ∈ f witness_d ↔ witness_d ∉ f witness_d := sorry',
      '  exact ',
    ].join('\n')

    const extracted = extractLeanDocumentSymbols([doc])
    const extractedNames = new Set(extracted.map(item => item.label))
    assert.ok(extractedNames.has('cantor_diag'))
    assert.ok(extractedNames.has('hsurj'))
    assert.ok(extractedNames.has('diagSet'))
    assert.ok(extractedNames.has('witness_d'))
    assert.ok(extractedNames.has('h_witness'))
    assert.ok(extractedNames.has('hiff_self'))

    const hiffRes = await runCompletion(`${doc}hiff_`)
    assert.ok(hiffRes)
    assert.ok(hiffRes.options.some(opt => opt.label === 'hiff_self'))

    const witnessRes = await runCompletion(`${doc}witness_`)
    assert.ok(witnessRes)
    assert.ok(witnessRes.options.some(opt => opt.label === 'witness_d'))

    // Also test sibling Lean cells via leanLspConfigFacet
    const siblingExt = leanLspConfigFacet.of({
      enabled: true,
      runtimeId: 'rt-lean',
      sourcePath: 'notes/sets.md',
      cellId: 'cell-2',
      language: 'lean',
      cells: () => [
        {
          id: 'cell-1',
          language: 'lean',
          executionIndex: 1,
          source: 'lemma shared_set_lemma (A B : Set α) : A ∩ B = B ∩ A := sorry',
        },
      ],
    })
    const crossCellRes = await runCompletion('  apply shared_set_', undefined, false, [siblingExt])
    assert.ok(crossCellRes)
    assert.ok(crossCellRes.options.some(opt => opt.label === 'shared_set_lemma'))
  })

  test('merges live LeanWebSocketSession.complete results when warm and deduplicates by label', async () => {
    const session = getSharedLeanWebSocketSession()
    const origIsWarm = session.isWarm.bind(session)
    const origComplete = session.complete.bind(session)

    try {
      session.isWarm = () => true
      session.complete = async () => [
        {
          label: 'mem_singleton_iff',
          detail: 'x ∈ {a} ↔ x = a',
          documentation: 'Membership in a singleton set.',
          kind: 3,
        },
        { label: 'Set.mem_union', detail: 'x ∈ a ∪ b ↔ x ∈ a ∨ x ∈ b', kind: 3 },
      ]

      const res = await runCompletion('#check Set.mem_')
      assert.ok(res)
      const singletonMatches = res.options.filter(opt => opt.label === 'Set.mem_singleton_iff')
      assert.equal(singletonMatches.length, 1)
      assert.equal(singletonMatches[0]?.detail, 'x ∈ {a} ↔ x = a')

      const unionMatches = res.options.filter(opt => opt.label === 'Set.mem_union')
      assert.equal(unionMatches.length, 1)
    } finally {
      session.isWarm = origIsWarm
      session.complete = origComplete
    }
  })

  test('resolves hover tooltips for Unicode symbols, Mathlib entries, and live LSP goals', async () => {
    const line = 'theorem t (A B : Set α) : A ∩ B = B ∩ A := by rw [Set.mem_inter_iff]'
    const capCol = line.indexOf('∩')
    const capHover = await resolveLeanHoverAt(line, 0, line, capCol)
    assert.ok(capHover)
    assert.match(capHover.unicodeHint ?? '', /Type ∩ using/)
    assert.match(capHover.unicodeHint ?? '', /\\cap|\\inter/)

    const lemmaCol = line.indexOf('Set.mem_inter_iff') + 5
    const lemmaHover = await resolveLeanHoverAt(line, 0, line, lemmaCol)
    assert.ok(lemmaHover)
    assert.match(lemmaHover.signature ?? '', /Set\.mem_inter_iff/)
    assert.match(lemmaHover.documentation ?? '', /a ∩ b/)

    const session = getSharedLeanWebSocketSession()
    const origIsWarm = session.isWarm.bind(session)
    const origHover = session.hover.bind(session)
    try {
      session.isWarm = () => true
      session.hover = async () => ({
        contents:
          '```lean\nSet.mem_inter_iff (x : α) (a b : Set α) : x ∈ a ∩ b ↔ x ∈ a ∧ x ∈ b\n```',
        goals: ['A B : Set α\n⊢ A ∩ B = B ∩ A'],
      })

      const liveHover = await resolveLeanHoverAt(line, 0, line, lemmaCol)
      assert.ok(liveHover)
      assert.ok(liveHover.goals?.includes('A B : Set α\n⊢ A ∩ B = B ∩ A'))
      const html = renderLeanHoverHtml(liveHover)
      assert.match(html, /cm-lean-hover-goals/)
      assert.match(html, /⊢ A ∩ B = B ∩ A/)
    } finally {
      session.isWarm = origIsWarm
      session.hover = origHover
    }
  })

  test('wires leanLanguageExtension and 0ms leanLspBridge on leanBackend.editor', async () => {
    const ext = await leanBackend.editor?.languageExtension?.()
    assert.ok(ext)
    const state = EditorState.create({
      doc: '#check Set.ext',
      extensions: [leanLanguageExtension()],
    })
    assert.equal(state.doc.toString(), '#check Set.ext')

    const bridge = await leanBackend.editor?.lspBridge?.()
    assert.equal(bridge, leanLspBridge)
    const exts = await bridge!.extensions({
      enabled: true,
      runtimeId: 'lean-rt',
      sourcePath: 'notes/lean.md',
      cellId: 'cell-1',
      language: 'lean',
    })
    assert.equal(exts.length, 1)
  })
})
