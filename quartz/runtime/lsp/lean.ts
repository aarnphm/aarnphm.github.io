import {
  type Completion,
  type CompletionContext,
  type CompletionResult,
  type CompletionSource,
} from '@codemirror/autocomplete'
import { Facet, type Extension } from '@codemirror/state'
import { hoverTooltip, type Tooltip } from '@codemirror/view'
import type { LspBridge, LspConfig } from './bridge'
import { escapeHTML } from '../../util/escape'
import { getSharedLeanWebSocketSession, type LeanLspCompletionItem } from '../lean/kernel'
import { getLeanAbbreviationIndex, leanUnicodeCompletionSource } from '../lean/unicode-input'

export type LeanCatalogCategory = 'tactic' | 'keyword' | 'mathlib' | 'local'

export type LeanCatalogEntry = {
  readonly label: string
  readonly category: LeanCatalogCategory
  readonly type: string
  readonly detail: string
  readonly documentation: string
}

export const leanLspConfigFacet = Facet.define<LspConfig, LspConfig | undefined>({
  combine: values => values.at(-1),
})

const LEAN_IDENTIFIER_BEFORE_RE =
  /(?:#[A-Za-z_][A-Za-z0-9_']*|[A-Za-z_À-ɏͰ-Ͽ℀-⅏][A-Za-z0-9_.'!?\u00c0-\u024f\u0370-\u03ff\u2080-\u2089\u2100-\u214f]*)$/

const LEAN_TOKEN_CHAR_RE = /[A-Za-z0-9_.'!?#\u00c0-\u024f\u0370-\u03ff\u2080-\u2089\u2100-\u214f]/

const LEAN_STATIC_CATALOG: readonly LeanCatalogEntry[] = [
  // Tactics
  {
    label: 'intro',
    category: 'tactic',
    type: 'keyword',
    detail: 'tactic · intro x₁ x₂ …',
    documentation:
      'Introduce one or more hypotheses or variables from a ∀/→ binder into the local context.',
  },
  {
    label: 'intros',
    category: 'tactic',
    type: 'keyword',
    detail: 'tactic · intros x₁ …',
    documentation:
      'Introduce leading universally quantified variables and implications into the context.',
  },
  {
    label: 'apply',
    category: 'tactic',
    type: 'keyword',
    detail: 'tactic · apply e',
    documentation:
      'Unify the conclusion of term `e` with the current target and generate subgoals for its hypotheses.',
  },
  {
    label: 'exact',
    category: 'tactic',
    type: 'keyword',
    detail: 'tactic · exact e',
    documentation: 'Close the current goal with a term `e` of the target type.',
  },
  {
    label: 'refine',
    category: 'tactic',
    type: 'keyword',
    detail: 'tactic · refine e',
    documentation:
      'Behaves like `exact e`, converting any holes `?_` in `e` into new tactic subgoals.',
  },
  {
    label: 'rw',
    category: 'tactic',
    type: 'keyword',
    detail: 'tactic · rw [h₁, ← h₂] at loc',
    documentation:
      'Rewrite the target (or hypothesis `loc`) using equations or `↔` equivalences, then attempt `rfl`.',
  },
  {
    label: 'rwa',
    category: 'tactic',
    type: 'keyword',
    detail: 'tactic · rwa [h₁, h₂]',
    documentation: 'Rewrite with `rw` and close any remaining goal using `assumption`.',
  },
  {
    label: 'simp',
    category: 'tactic',
    type: 'keyword',
    detail: 'tactic · simp [h₁, h₂] at loc',
    documentation: 'Simplify the target or hypotheses using `@[simp]` lemmas and supplied rules.',
  },
  {
    label: 'simp only',
    category: 'tactic',
    type: 'keyword',
    detail: 'tactic · simp only [h₁, h₂]',
    documentation: 'Simplify using only the explicitly listed lemmas.',
  },
  {
    label: 'simp_all',
    category: 'tactic',
    type: 'keyword',
    detail: 'tactic · simp_all',
    documentation: 'Simplify all hypotheses and the goal mutually using `@[simp]` and local rules.',
  },
  {
    label: 'ext',
    category: 'tactic',
    type: 'keyword',
    detail: 'tactic · ext x',
    documentation:
      'Apply extensionality lemmas (`Set.ext`, `funext`, etc.) to reduce equality of sets, functions, or structures.',
  },
  {
    label: 'funext',
    category: 'tactic',
    type: 'keyword',
    detail: 'tactic · funext x',
    documentation: 'Apply function extensionality `(∀ x, f x = g x) → f = g`.',
  },
  {
    label: 'rcases',
    category: 'tactic',
    type: 'keyword',
    detail: 'tactic · rcases h with ⟨a, b⟩ | c',
    documentation: 'Recursively destructure hypotheses or expressions using tuple/or patterns.',
  },
  {
    label: 'obtain',
    category: 'tactic',
    type: 'keyword',
    detail: 'tactic · obtain ⟨x, hx⟩ := e',
    documentation: 'Destructure term `e` with `rcases` pattern syntax and bind new hypotheses.',
  },
  {
    label: 'constructor',
    category: 'tactic',
    type: 'keyword',
    detail: 'tactic · constructor',
    documentation:
      'Apply the first matching constructor of the target inductive type (e.g., `∧`, `↔`).',
  },
  {
    label: 'cases',
    category: 'tactic',
    type: 'keyword',
    detail: 'tactic · cases h',
    documentation: 'Deconstruct an inductive value or hypothesis into one subgoal per constructor.',
  },
  {
    label: 'induction',
    category: 'tactic',
    type: 'keyword',
    detail: 'tactic · induction n with …',
    documentation: 'Perform proof by induction on an inductive value in context.',
  },
  {
    label: 'have',
    category: 'tactic',
    type: 'keyword',
    detail: 'tactic · have h : P := proof',
    documentation: 'Introduce an intermediate subgoal or lemma `h : P` into the local context.',
  },
  {
    label: 'let',
    category: 'tactic',
    type: 'keyword',
    detail: 'tactic · let x : α := val',
    documentation: 'Introduce a local transparent definition `x : α := val`.',
  },
  {
    label: 'show',
    category: 'tactic',
    type: 'keyword',
    detail: 'tactic · show P',
    documentation: 'Select and restate a goal whose target is definitionally equal to `P`.',
  },
  {
    label: 'suffices',
    category: 'tactic',
    type: 'keyword',
    detail: 'tactic · suffices h : P by …',
    documentation:
      'Reduce the current goal to showing `P`, given a proof that `P` implies the goal.',
  },
  {
    label: 'calc',
    category: 'tactic',
    type: 'keyword',
    detail: 'tactic · calc a = b := h₁ _ = c := h₂',
    documentation: 'Enter structured calculation mode chaining transitive relations.',
  },
  {
    label: 'conv',
    category: 'tactic',
    type: 'keyword',
    detail: 'tactic · conv => lhs; rw [h]',
    documentation: 'Enter conversion mode for targeted navigation and rewriting inside subterms.',
  },
  {
    label: 'omega',
    category: 'tactic',
    type: 'keyword',
    detail: 'tactic · omega',
    documentation: 'Decision procedure for linear integer and natural number arithmetic.',
  },
  {
    label: 'ring',
    category: 'tactic',
    type: 'keyword',
    detail: 'tactic · ring',
    documentation: 'Prove polynomial equalities in commutative rings and semirings.',
  },
  {
    label: 'linarith',
    category: 'tactic',
    type: 'keyword',
    detail: 'tactic · linarith',
    documentation: 'Linear arithmetic solver over ordered rings and fields using local hypotheses.',
  },
  {
    label: 'nlinarith',
    category: 'tactic',
    type: 'keyword',
    detail: 'tactic · nlinarith',
    documentation: 'Nonlinear arithmetic extension of `linarith`.',
  },
  {
    label: 'norm_num',
    category: 'tactic',
    type: 'keyword',
    detail: 'tactic · norm_num',
    documentation: 'Normalize numerical expressions and inequalities.',
  },
  {
    label: 'decide',
    category: 'tactic',
    type: 'keyword',
    detail: 'tactic · decide',
    documentation: 'Prove a `Decidable` proposition by computation.',
  },
  {
    label: 'aesop',
    category: 'tactic',
    type: 'keyword',
    detail: 'tactic · aesop',
    documentation: 'Best-first automated proof search for Lean 4 and Mathlib.',
  },
  {
    label: 'tauto',
    category: 'tactic',
    type: 'keyword',
    detail: 'tactic · tauto',
    documentation: 'Decision procedure for propositional tautologies.',
  },
  {
    label: 'by_cases',
    category: 'tactic',
    type: 'keyword',
    detail: 'tactic · by_cases h : P',
    documentation: 'Split the goal into cases `h : P` and `h : ¬P` via excluded middle.',
  },
  {
    label: 'by_contra',
    category: 'tactic',
    type: 'keyword',
    detail: 'tactic · by_contra h',
    documentation: 'Assume the negation `h : ¬P` of the goal and change the target to `False`.',
  },
  {
    label: 'contradiction',
    category: 'tactic',
    type: 'keyword',
    detail: 'tactic · contradiction',
    documentation: 'Close the goal from contradictory hypotheses or `False` in the context.',
  },
  {
    label: 'assumption',
    category: 'tactic',
    type: 'keyword',
    detail: 'tactic · assumption',
    documentation: 'Close the goal using a matching hypothesis in the local context.',
  },
  {
    label: 'trivial',
    category: 'tactic',
    type: 'keyword',
    detail: 'tactic · trivial',
    documentation: 'Close simple goals using `True.intro`, `rfl`, or `contradiction`.',
  },
  {
    label: 'rfl',
    category: 'tactic',
    type: 'keyword',
    detail: 'tactic · rfl',
    documentation: 'Close a reflexive goal (`a = a` or `a ↔ a`) up to definitional equality.',
  },
  {
    label: 'unfold',
    category: 'tactic',
    type: 'keyword',
    detail: 'tactic · unfold f at loc',
    documentation: 'Unfold definitions of constants in the target or hypotheses.',
  },
  {
    label: 'change',
    category: 'tactic',
    type: 'keyword',
    detail: 'tactic · change P',
    documentation: 'Replace the target with a definitionally equal expression `P`.',
  },
  {
    label: 'specialize',
    category: 'tactic',
    type: 'keyword',
    detail: 'tactic · specialize h a₁ …',
    documentation: 'Instantiate a universal or implication hypothesis `h` in place.',
  },
  {
    label: 'generalize',
    category: 'tactic',
    type: 'keyword',
    detail: 'tactic · generalize h : e = x',
    documentation: 'Replace occurrences of subterm `e` in the goal with variable `x`.',
  },
  {
    label: 'congr',
    category: 'tactic',
    type: 'keyword',
    detail: 'tactic · congr',
    documentation: 'Reduce an equality of function applications to equalities of arguments.',
  },
  {
    label: 'subst',
    category: 'tactic',
    type: 'keyword',
    detail: 'tactic · subst x',
    documentation: 'Substitute an equality hypothesis `x = e` throughout the goal and context.',
  },
  {
    label: 'use',
    category: 'tactic',
    type: 'keyword',
    detail: 'tactic · use w',
    documentation: 'Instantiate existential quantifiers `∃ x, P x` with witness `w`.',
  },
  {
    label: 'left',
    category: 'tactic',
    type: 'keyword',
    detail: 'tactic · left',
    documentation: 'Apply the left constructor (`Or.inl`) of a binary inductive target.',
  },
  {
    label: 'right',
    category: 'tactic',
    type: 'keyword',
    detail: 'tactic · right',
    documentation: 'Apply the right constructor (`Or.inr`) of a binary inductive target.',
  },
  {
    label: 'exfalso',
    category: 'tactic',
    type: 'keyword',
    detail: 'tactic · exfalso',
    documentation: 'Change the current target to `False` via `False.elim`.',
  },

  // Declarations & Keywords
  {
    label: 'theorem',
    category: 'keyword',
    type: 'keyword',
    detail: 'declaration · theorem name : P := by …',
    documentation: 'Declare a named proposition and its proof.',
  },
  {
    label: 'lemma',
    category: 'keyword',
    type: 'keyword',
    detail: 'declaration · lemma name : P := by …',
    documentation: 'Declare a proof lemma (synonym for `theorem`).',
  },
  {
    label: 'def',
    category: 'keyword',
    type: 'keyword',
    detail: 'declaration · def name : α := …',
    documentation: 'Declare a computable definition or function.',
  },
  {
    label: 'abbrev',
    category: 'keyword',
    type: 'keyword',
    detail: 'declaration · abbrev name := …',
    documentation: 'Declare a reducible definition unfolded automatically during typechecking.',
  },
  {
    label: 'structure',
    category: 'keyword',
    type: 'keyword',
    detail: 'declaration · structure S where …',
    documentation: 'Declare a single-constructor inductive record type with named projections.',
  },
  {
    label: 'class',
    category: 'keyword',
    type: 'keyword',
    detail: 'declaration · class C (α : Type*) where …',
    documentation: 'Declare a typeclass participating in `[]` instance synthesis.',
  },
  {
    label: 'instance',
    category: 'keyword',
    type: 'keyword',
    detail: 'declaration · instance : C α where …',
    documentation: 'Register a typeclass instance for synthesis.',
  },
  {
    label: 'inductive',
    category: 'keyword',
    type: 'keyword',
    detail: 'declaration · inductive T where | mk : …',
    documentation: 'Declare an inductive data type or proposition family.',
  },
  {
    label: 'where',
    category: 'keyword',
    type: 'keyword',
    detail: 'keyword · where',
    documentation: 'Introduce structure/class fields or local auxiliary definitions.',
  },
  {
    label: 'namespace',
    category: 'keyword',
    type: 'keyword',
    detail: 'command · namespace N',
    documentation: 'Open a hierarchical declaration namespace `N`.',
  },
  {
    label: 'section',
    category: 'keyword',
    type: 'keyword',
    detail: 'command · section S',
    documentation: 'Open a lexical scope for `variable` and `open` declarations.',
  },
  {
    label: 'end',
    category: 'keyword',
    type: 'keyword',
    detail: 'command · end N',
    documentation: 'Close an enclosing `namespace` or `section`.',
  },
  {
    label: 'open',
    category: 'keyword',
    type: 'keyword',
    detail: 'command · open Set Function',
    documentation: 'Bring declarations from one or more namespaces into scope.',
  },
  {
    label: 'import',
    category: 'keyword',
    type: 'keyword',
    detail: 'command · import Mathlib',
    documentation: 'Import a module such as `Mathlib` at the top of a Lean document.',
  },
  {
    label: 'variable',
    category: 'keyword',
    type: 'keyword',
    detail: 'command · variable {α : Type*}',
    documentation: 'Declare section-scoped implicit or explicit parameters.',
  },
  {
    label: 'universe',
    category: 'keyword',
    type: 'keyword',
    detail: 'command · universe u v',
    documentation: 'Declare universe level variables.',
  },
  {
    label: 'noncomputable',
    category: 'keyword',
    type: 'keyword',
    detail: 'modifier · noncomputable',
    documentation:
      'Mark a definition or section as not generating executable VM code (e.g., using `Classical.choice`).',
  },
  {
    label: 'by',
    category: 'keyword',
    type: 'keyword',
    detail: 'keyword · by …',
    documentation: 'Enter tactic mode to construct a proof or term.',
  },
  {
    label: 'fun',
    category: 'keyword',
    type: 'keyword',
    detail: 'keyword · fun x => …',
    documentation: 'Lambda abstraction (`λ x => …`).',
  },
  {
    label: 'match',
    category: 'keyword',
    type: 'keyword',
    detail: 'keyword · match e with | pat => …',
    documentation: 'Pattern match on one or more expressions.',
  },
  {
    label: 'with',
    category: 'keyword',
    type: 'keyword',
    detail: 'keyword · with',
    documentation: 'Introduce pattern arms in `match`, `rcases`, `induction`, or `cases`.',
  },
  {
    label: 'if',
    category: 'keyword',
    type: 'keyword',
    detail: 'keyword · if P then a else b',
    documentation: 'Conditional expression (`ite` or dependent `if h : P then … else …`).',
  },
  {
    label: 'then',
    category: 'keyword',
    type: 'keyword',
    detail: 'keyword · then',
    documentation: 'True branch of a conditional `if` expression.',
  },
  {
    label: 'else',
    category: 'keyword',
    type: 'keyword',
    detail: 'keyword · else',
    documentation: 'False branch of a conditional `if` expression.',
  },
  {
    label: 'Type*',
    category: 'keyword',
    type: 'type',
    detail: 'sort · Type*',
    documentation: 'Type in an arbitrary fresh universe `Type u`.',
  },
  {
    label: 'Prop',
    category: 'keyword',
    type: 'type',
    detail: 'sort · Prop (Sort 0)',
    documentation: 'Impredicative sort of logical propositions.',
  },
  {
    label: 'Sort*',
    category: 'keyword',
    type: 'type',
    detail: 'sort · Sort*',
    documentation: 'Sort in an arbitrary fresh universe `Sort u`.',
  },
  {
    label: '#check',
    category: 'keyword',
    type: 'keyword',
    detail: 'command · #check expr',
    documentation: 'Display the inferred type or signature of an expression or declaration.',
  },
  {
    label: '#eval',
    category: 'keyword',
    type: 'keyword',
    detail: 'command · #eval expr',
    documentation: 'Evaluate an expression in the Lean VM and print the result.',
  },
  {
    label: '#print',
    category: 'keyword',
    type: 'keyword',
    detail: 'command · #print ident',
    documentation: 'Print the definition or axioms (`#print axioms`) of a declaration.',
  },
  {
    label: '#reduce',
    category: 'keyword',
    type: 'keyword',
    detail: 'command · #reduce expr',
    documentation: 'Reduce an expression to normal form in the Lean typechecker kernel.',
  },
  {
    label: '#synth',
    category: 'keyword',
    type: 'keyword',
    detail: 'command · #synth Class α',
    documentation: 'Run typeclass synthesis for the given class application.',
  },

  // Mathlib Set / Function / Logic / Topology API
  {
    label: 'Set',
    category: 'mathlib',
    type: 'type',
    detail: 'Set (α : Type u) : Type u',
    documentation: 'Type of sets of elements of `α`, defined as predicates `α → Prop`.',
  },
  {
    label: 'Set.ext',
    category: 'mathlib',
    type: 'function',
    detail: '∀ {s t : Set α}, (∀ x, x ∈ s ↔ x ∈ t) → s = t',
    documentation:
      'Set extensionality: two sets are equal if and only if they have the same elements.',
  },
  {
    label: 'Set.univ',
    category: 'mathlib',
    type: 'constant',
    detail: 'Set.univ : Set α',
    documentation: 'The universal set `{x : α | True}` on type `α`.',
  },
  {
    label: 'Set.empty',
    category: 'mathlib',
    type: 'constant',
    detail: 'Set.empty : Set α',
    documentation: 'The empty set `∅ : Set α`.',
  },
  {
    label: 'Set.Mem',
    category: 'mathlib',
    type: 'function',
    detail: 'Set.Mem (s : Set α) (a : α) : Prop',
    documentation: 'Set membership predicate `a ∈ s`, reducible to `s a`.',
  },
  {
    label: 'Set.Subset',
    category: 'mathlib',
    type: 'function',
    detail: 'Set.Subset (s₁ s₂ : Set α) : Prop',
    documentation: 'Subset relation `s₁ ⊆ s₂`, defined as `∀ ⦃a⦄, a ∈ s₁ → a ∈ s₂`.',
  },
  {
    label: 'Set.union',
    category: 'mathlib',
    type: 'function',
    detail: 'Set.union (s₁ s₂ : Set α) : Set α',
    documentation: 'Binary union `s₁ ∪ s₂ = {a | a ∈ s₁ ∨ a ∈ s₂}`.',
  },
  {
    label: 'Set.inter',
    category: 'mathlib',
    type: 'function',
    detail: 'Set.inter (s₁ s₂ : Set α) : Set α',
    documentation: 'Binary intersection `s₁ ∩ s₂ = {a | a ∈ s₁ ∧ a ∈ s₂}`.',
  },
  {
    label: 'Set.diff',
    category: 'mathlib',
    type: 'function',
    detail: 'Set.diff (s t : Set α) : Set α',
    documentation: 'Set difference `s \\ t = {a | a ∈ s ∧ a ∉ t}`.',
  },
  {
    label: 'Set.compl',
    category: 'mathlib',
    type: 'function',
    detail: 'Set.compl (s : Set α) : Set α',
    documentation: 'Set complement `sᶜ = {x | x ∉ s}`.',
  },
  {
    label: 'Set.symmDiff',
    category: 'mathlib',
    type: 'function',
    detail: 'Set.symmDiff (A B : Set α) : Set α',
    documentation: 'Symmetric difference `A ∆ B = (A \\ B) ∪ (B \\ A)`.',
  },
  {
    label: 'Set.iUnion',
    category: 'mathlib',
    type: 'function',
    detail: 'Set.iUnion (s : ι → Set α) : Set α',
    documentation: 'Indexed union `⋃ i, s i = {a | ∃ i, a ∈ s i}`.',
  },
  {
    label: 'Set.iInter',
    category: 'mathlib',
    type: 'function',
    detail: 'Set.iInter (s : ι → Set α) : Set α',
    documentation: 'Indexed intersection `⋂ i, s i = {a | ∀ i, a ∈ s i}`.',
  },
  {
    label: 'Set.preimage',
    category: 'mathlib',
    type: 'function',
    detail: 'Set.preimage (f : α → β) (s : Set β) : Set α',
    documentation: "Preimage `f ⁻¹' s = {x | f x ∈ s}` of set `s` under `f`.",
  },
  {
    label: 'Set.image',
    category: 'mathlib',
    type: 'function',
    detail: 'Set.image (f : α → β) (s : Set α) : Set β',
    documentation: "Direct image `f '' s = {y | ∃ x ∈ s, f x = y}` of set `s` under `f`.",
  },
  {
    label: 'Set.range',
    category: 'mathlib',
    type: 'function',
    detail: 'Set.range (f : ι → α) : Set α',
    documentation: 'Range `{x | ∃ y, f y = x}` of function `f`.',
  },
  {
    label: 'Set.mem_setOf',
    category: 'mathlib',
    type: 'function',
    detail: 'a ∈ {x | p x} ↔ p a',
    documentation: 'Membership in a set comprehension `{x | p x}` reduces to `p a`.',
  },
  {
    label: 'Set.mem_union',
    category: 'mathlib',
    type: 'function',
    detail: 'x ∈ a ∪ b ↔ x ∈ a ∨ x ∈ b',
    documentation: 'An element belongs to `a ∪ b` iff it belongs to `a` or `b`.',
  },
  {
    label: 'Set.mem_inter_iff',
    category: 'mathlib',
    type: 'function',
    detail: 'x ∈ a ∩ b ↔ x ∈ a ∧ x ∈ b',
    documentation: 'An element belongs to `a ∩ b` iff it belongs to both `a` and `b`.',
  },
  {
    label: 'Set.mem_diff',
    category: 'mathlib',
    type: 'function',
    detail: 'x ∈ s \\ t ↔ x ∈ s ∧ x ∉ t',
    documentation: 'An element belongs to `s \\ t` iff it is in `s` and not in `t`.',
  },
  {
    label: 'Set.mem_compl_iff',
    category: 'mathlib',
    type: 'function',
    detail: 'x ∈ sᶜ ↔ x ∉ s',
    documentation: 'An element belongs to the complement `sᶜ` iff it is not in `s`.',
  },
  {
    label: 'Set.mem_symmDiff',
    category: 'mathlib',
    type: 'function',
    detail: 'x ∈ A ∆ B ↔ x ∈ A ∧ x ∉ B ∨ x ∈ B ∧ x ∉ A',
    documentation: 'Membership characterization for symmetric difference `A ∆ B`.',
  },
  {
    label: 'Set.mem_iUnion',
    category: 'mathlib',
    type: 'function',
    detail: 'x ∈ ⋃ i, s i ↔ ∃ i, x ∈ s i',
    documentation: 'Membership in an indexed union `⋃ i, s i`.',
  },
  {
    label: 'Set.mem_iInter',
    category: 'mathlib',
    type: 'function',
    detail: 'x ∈ ⋂ i, s i ↔ ∀ i, x ∈ s i',
    documentation: 'Membership in an indexed intersection `⋂ i, s i`.',
  },
  {
    label: 'Set.mem_empty_iff_false',
    category: 'mathlib',
    type: 'function',
    detail: 'x ∈ (∅ : Set α) ↔ False',
    documentation: 'Membership in the empty set `∅` is equivalent to `False`.',
  },
  {
    label: 'Set.mem_univ',
    category: 'mathlib',
    type: 'function',
    detail: 'x ∈ (Set.univ : Set α)',
    documentation: 'Every element `x : α` is a member of `Set.univ`.',
  },
  {
    label: 'Set.mem_preimage',
    category: 'mathlib',
    type: 'function',
    detail: "a ∈ f ⁻¹' s ↔ f a ∈ s",
    documentation: "Membership in the preimage `f ⁻¹' s` reduces to `f a ∈ s`.",
  },
  {
    label: 'Set.mem_image',
    category: 'mathlib',
    type: 'function',
    detail: "y ∈ f '' s ↔ ∃ x ∈ s, f x = y",
    documentation: "Membership in the image `f '' s` of `s` under `f`.",
  },
  {
    label: 'Set.diff_eq',
    category: 'mathlib',
    type: 'function',
    detail: 's \\ t = s ∩ tᶜ',
    documentation: 'Set difference `s \\ t` equals intersection with the complement `s ∩ tᶜ`.',
  },
  {
    label: 'Set.compl_union',
    category: 'mathlib',
    type: 'function',
    detail: '(s ∪ t)ᶜ = sᶜ ∩ tᶜ',
    documentation: 'De Morgan law for the complement of a union.',
  },
  {
    label: 'Set.compl_inter',
    category: 'mathlib',
    type: 'function',
    detail: '(s ∩ t)ᶜ = sᶜ ∪ tᶜ',
    documentation: 'De Morgan law for the complement of an intersection.',
  },
  {
    label: 'Function.Injective',
    category: 'mathlib',
    type: 'function',
    detail: 'Function.Injective (f : α → β) : Prop',
    documentation: 'Injectivity predicate `∀ ⦃a₁ a₂⦄, f a₁ = f a₂ → a₁ = a₂`.',
  },
  {
    label: 'Function.Surjective',
    category: 'mathlib',
    type: 'function',
    detail: 'Function.Surjective (f : α → β) : Prop',
    documentation: 'Surjectivity predicate `∀ b, ∃ a, f a = b`.',
  },
  {
    label: 'Function.Bijective',
    category: 'mathlib',
    type: 'function',
    detail: 'Function.Bijective (f : α → β) : Prop',
    documentation: 'Bijectivity predicate `Injective f ∧ Surjective f`.',
  },
  {
    label: 'Function.comp',
    category: 'mathlib',
    type: 'function',
    detail: 'Function.comp (f : β → φ) (g : α → β) : α → φ',
    documentation: 'Function composition `(f ∘ g) x = f (g x)`.',
  },
  {
    label: 'Equivalence',
    category: 'mathlib',
    type: 'type',
    detail: 'Equivalence (r : α → α → Prop) : Prop',
    documentation:
      'Bundled proof that binary relation `r` is reflexive, symmetric, and transitive.',
  },
  {
    label: 'Equivalence.refl',
    category: 'mathlib',
    type: 'function',
    detail: 'Equivalence.refl (h : Equivalence r) : ∀ x, r x x',
    documentation: 'Reflexivity projection of an `Equivalence` relation.',
  },
  {
    label: 'Equivalence.symm',
    category: 'mathlib',
    type: 'function',
    detail: 'Equivalence.symm (h : Equivalence r) : r x y → r y x',
    documentation: 'Symmetry projection of an `Equivalence` relation.',
  },
  {
    label: 'Equivalence.trans',
    category: 'mathlib',
    type: 'function',
    detail: 'Equivalence.trans (h : Equivalence r) : r x y → r y z → r x z',
    documentation: 'Transitivity projection of an `Equivalence` relation.',
  },
  {
    label: 'Classical.em',
    category: 'mathlib',
    type: 'function',
    detail: 'Classical.em (p : Prop) : p ∨ ¬p',
    documentation: 'Law of the excluded middle for any proposition `p`.',
  },
  {
    label: 'Classical.byContradiction',
    category: 'mathlib',
    type: 'function',
    detail: 'Classical.byContradiction (h : ¬p → False) : p',
    documentation: 'Classical proof by contradiction (`(¬p → False) → p`).',
  },
  {
    label: 'iff_not_self',
    category: 'mathlib',
    type: 'function',
    detail: 'iff_not_self : ¬(a ↔ ¬a)',
    documentation: 'No proposition is equivalent to its own negation.',
  },
  {
    label: 'Or.elim',
    category: 'mathlib',
    type: 'function',
    detail: 'Or.elim (h : a ∨ b) (left : a → c) (right : b → c) : c',
    documentation: 'Case analysis on disjunction `a ∨ b`.',
  },
  {
    label: 'Or.inl',
    category: 'mathlib',
    type: 'function',
    detail: 'Or.inl (h : a) : a ∨ b',
    documentation: 'Left constructor of disjunction `a ∨ b`.',
  },
  {
    label: 'Or.inr',
    category: 'mathlib',
    type: 'function',
    detail: 'Or.inr (h : b) : a ∨ b',
    documentation: 'Right constructor of disjunction `a ∨ b`.',
  },
  {
    label: 'And.intro',
    category: 'mathlib',
    type: 'function',
    detail: 'And.intro (left : a) (right : b) : a ∧ b',
    documentation: 'Constructor `⟨left, right⟩` for conjunction `a ∧ b`.',
  },
  {
    label: 'And.left',
    category: 'mathlib',
    type: 'function',
    detail: 'And.left (h : a ∧ b) : a',
    documentation: 'Left projection of conjunction `a ∧ b`.',
  },
  {
    label: 'And.right',
    category: 'mathlib',
    type: 'function',
    detail: 'And.right (h : a ∧ b) : b',
    documentation: 'Right projection of conjunction `a ∧ b`.',
  },
  {
    label: 'Iff.intro',
    category: 'mathlib',
    type: 'function',
    detail: 'Iff.intro (mp : a → b) (mpr : b → a) : a ↔ b',
    documentation: 'Construct logical equivalence `a ↔ b` from forward and backward implications.',
  },
  {
    label: 'Iff.mp',
    category: 'mathlib',
    type: 'function',
    detail: 'Iff.mp (h : a ↔ b) : a → b',
    documentation: 'Forward implication projection of `a ↔ b`.',
  },
  {
    label: 'Iff.mpr',
    category: 'mathlib',
    type: 'function',
    detail: 'Iff.mpr (h : a ↔ b) : b → a',
    documentation: 'Reverse implication projection of `a ↔ b`.',
  },
  {
    label: 'False.elim',
    category: 'mathlib',
    type: 'function',
    detail: 'False.elim (h : False) : C',
    documentation: 'Ex falso quodlibet: derive any goal from `False`.',
  },
  {
    label: 'not_exists',
    category: 'mathlib',
    type: 'function',
    detail: 'not_exists : (¬∃ x, p x) ↔ ∀ x, ¬p x',
    documentation: 'Push negation across an existential quantifier.',
  },
  {
    label: 'not_forall',
    category: 'mathlib',
    type: 'function',
    detail: 'not_forall : (¬∀ x, p x) ↔ ∃ x, ¬p x',
    documentation: 'Push negation across a universal quantifier.',
  },
]

const LEAN_CATALOG_BY_LABEL = new Map<string, LeanCatalogEntry>(
  LEAN_STATIC_CATALOG.map(entry => [entry.label, entry]),
)

const RESERVED_SKIP_NAMES = new Set([
  'by',
  'in',
  'with',
  'where',
  'do',
  'fun',
  'match',
  'if',
  'then',
  'else',
  'this',
  'rfl',
  'sorry',
  'admit',
])

export function extractLeanDocumentSymbols(sources: readonly string[]): LeanCatalogEntry[] {
  const seen = new Set<string>(LEAN_CATALOG_BY_LABEL.keys())
  const out: LeanCatalogEntry[] = []

  const addSymbol = (label: string, type: string, detail: string, documentation: string) => {
    const clean = label.trim()
    if (
      !clean ||
      clean === '_' ||
      RESERVED_SKIP_NAMES.has(clean) ||
      seen.has(clean) ||
      !/^[A-Za-z_À-ɏͰ-Ͽ℀-⅏][A-Za-z0-9_.'!?\u00c0-\u024f\u0370-\u03ff\u2080-\u2089\u2100-\u214f]*$/.test(
        clean,
      )
    ) {
      return
    }
    seen.add(clean)
    out.push({ label: clean, category: 'local', type, detail, documentation })
  }

  for (const source of sources) {
    const lines = source.split(/\r?\n/)
    for (const rawLine of lines) {
      const line = rawLine.replace(/--.*$/, '').trim()
      if (!line) continue

      const declMatch = line.match(
        /^(?:(?:noncomputable|private|protected|scoped|local|partial|unsafe)\s+)*(theorem|lemma|def|abbrev|structure|class|inductive|axiom|opaque)\s+([A-Za-z_À-ɏͰ-Ͽ℀-⅏][A-Za-z0-9_.'!?\u00c0-\u024f\u0370-\u03ff\u2080-\u2089\u2100-\u214f]*)(.*)$/,
      )
      if (declMatch) {
        const kind = declMatch[1]
        const name = declMatch[2]
        const rest =
          declMatch[3]
            ?.replace(/:=\s*by\b.*$/, '')
            .replace(/:=.*$/, '')
            .trim() ?? ''
        const isType = kind === 'structure' || kind === 'class' || kind === 'inductive'
        addSymbol(
          name,
          isType ? 'type' : 'function',
          rest ? `${kind} ${name} ${rest}` : `${kind} ${name}`,
          `Document-local ${kind} \`${name}\`.`,
        )

        for (const binder of rest.matchAll(/[(⦃{]\s*([^():⦃⦄{}]+?)\s*:\s*([^()⦃⦄{}]+?)[)⦄}]/g)) {
          const namesPart = binder[1].trim()
          const typePart = binder[2].trim()
          for (const param of namesPart.split(/\s+/)) {
            addSymbol(
              param,
              'variable',
              `${param} : ${typePart}`,
              `Parameter \`${param} : ${typePart}\` of \`${name}\`.`,
            )
          }
        }
      }

      const haveLetMatch = line.match(
        /\b(have|let)\s+([A-Za-z_À-ɏͰ-Ͽ℀-⅏][A-Za-z0-9_'!?\u00c0-\u024f\u0370-\u03ff\u2080-\u2089\u2100-\u214f]*)\s*(?::\s*([^:=]+?))?\s*:=/,
      )
      if (haveLetMatch) {
        const kind = haveLetMatch[1]
        const name = haveLetMatch[2]
        const typeStr = haveLetMatch[3]?.trim()
        addSymbol(
          name,
          'variable',
          typeStr ? `${name} : ${typeStr}` : `${kind} ${name}`,
          `Local ${kind === 'have' ? 'hypothesis' : 'binding'} \`${name}\`${
            typeStr ? ` : \`${typeStr}\`` : ''
          }.`,
        )
      }

      for (const tupleMatch of line.matchAll(/\b(?:obtain|rcases)\b[^⟨]*⟨([^⟩]+)⟩/g)) {
        for (const part of tupleMatch[1].split(/[,\s|⟨⟩]+/)) {
          addSymbol(
            part,
            'variable',
            `hypothesis ${part}`,
            `Local destructured witness or hypothesis \`${part}\`.`,
          )
        }
      }

      const introMatch = line.match(
        /\b(?:intro|intros)\s+([A-Za-z0-9_'!?\s\u00c0-\u024f\u0370-\u03ff\u2080-\u2089]+)/,
      )
      if (introMatch) {
        for (const ident of introMatch[1].split(/\s+/)) {
          addSymbol(ident, 'variable', `hypothesis ${ident}`, `Local hypothesis \`${ident}\`.`)
        }
      }
    }
  }

  return out
}

function isRemoteBrowser(): boolean {
  if (typeof window === 'undefined' || typeof WebSocket === 'undefined') return false
  const host = window.location?.hostname
  return Boolean(host && host !== 'localhost' && host !== '127.0.0.1')
}

function lspKindToCompletionType(kind: number | undefined): string {
  switch (kind) {
    case 2:
    case 3:
      return 'function'
    case 6:
      return 'variable'
    case 7:
    case 8:
    case 22:
      return 'type'
    case 14:
      return 'keyword'
    case 21:
      return 'constant'
    default:
      return 'function'
  }
}

function matchCatalogCandidates(
  prefix: string,
  entries: readonly LeanCatalogEntry[],
): Completion[] {
  const query = prefix.trim()
  const lower = query.toLowerCase()
  const dotIndex = query.lastIndexOf('.')
  const hasDot = dotIndex !== -1
  const namespacePrefix = hasDot ? query.slice(0, dotIndex + 1) : ''
  const memberQuery = hasDot ? query.slice(dotIndex + 1).toLowerCase() : ''

  const results: Completion[] = []
  for (const entry of entries) {
    if (hasDot) {
      if (!entry.label.startsWith(namespacePrefix)) continue
      const member = entry.label.slice(namespacePrefix.length)
      const memberLower = member.toLowerCase()
      if (
        memberQuery.length > 0 &&
        !memberLower.startsWith(memberQuery) &&
        !memberLower.includes(memberQuery)
      ) {
        continue
      }
      const startsWith = memberLower.startsWith(memberQuery)
      results.push({
        label: entry.label,
        type: entry.type,
        detail: entry.detail,
        info: entry.documentation,
        boost: startsWith ? 25 : 10,
      })
      continue
    }

    if (query.startsWith('#')) {
      if (!entry.label.startsWith('#') || !entry.label.toLowerCase().startsWith(lower)) continue
      results.push({
        label: entry.label,
        type: entry.type,
        detail: entry.detail,
        info: entry.documentation,
        boost: entry.label === query ? 40 : 25,
      })
      continue
    }

    if (entry.label.startsWith('#')) continue

    const labelLower = entry.label.toLowerCase()
    const startsWith = labelLower.startsWith(lower)
    const includes = lower.length >= 2 && labelLower.includes(lower)
    if (query.length > 0 && !startsWith && !includes) continue

    let boost = 5
    if (entry.label === query) boost = 45
    else if (labelLower === lower) boost = 40
    else if (startsWith && entry.category === 'local') boost = 30
    else if (startsWith && !entry.label.includes('.')) boost = 25
    else if (startsWith) boost = 18

    results.push({
      label: entry.label,
      type: entry.type,
      detail: entry.detail,
      info: entry.documentation,
      boost,
    })
  }

  return results
}

function mergeRemoteCompletions(
  prefix: string,
  localOptions: readonly Completion[],
  remoteItems: readonly LeanLspCompletionItem[],
): Completion[] {
  if (remoteItems.length === 0) return [...localOptions]
  const dotIndex = prefix.lastIndexOf('.')
  const namespacePrefix = dotIndex !== -1 ? prefix.slice(0, dotIndex + 1) : ''
  const memberQuery =
    dotIndex !== -1 ? prefix.slice(dotIndex + 1).toLowerCase() : prefix.toLowerCase()

  const byLabel = new Map<string, Completion>()
  for (const option of localOptions) {
    byLabel.set(option.label, option)
  }

  for (const item of remoteItems) {
    const rawLabel = item.label.trim()
    if (!rawLabel) continue
    const normalizedLabel =
      namespacePrefix && !rawLabel.includes('.') && !rawLabel.startsWith(namespacePrefix)
        ? `${namespacePrefix}${rawLabel}`
        : rawLabel
    const compareTarget = namespacePrefix
      ? normalizedLabel.slice(namespacePrefix.length).toLowerCase()
      : normalizedLabel.toLowerCase()
    if (
      memberQuery &&
      !compareTarget.startsWith(memberQuery) &&
      !compareTarget.includes(memberQuery)
    ) {
      continue
    }

    const existing = byLabel.get(normalizedLabel)
    if (existing) {
      if (!existing.detail && item.detail) {
        byLabel.set(normalizedLabel, {
          ...existing,
          detail: item.detail,
          ...(item.documentation && !existing.info ? { info: item.documentation } : {}),
        })
      }
      continue
    }

    byLabel.set(normalizedLabel, {
      label: normalizedLabel,
      type: lspKindToCompletionType(item.kind),
      ...(item.detail ? { detail: item.detail } : {}),
      ...(item.documentation ? { info: item.documentation } : {}),
      boost: compareTarget.startsWith(memberQuery) ? 22 : 12,
    })
  }

  return Array.from(byLabel.values())
}

function collectSourcesForCompletion(context: CompletionContext): string[] {
  const currentDoc = context.state.doc.toString()
  const config = context.state.facet(leanLspConfigFacet)
  const siblingCells = config?.cells?.() ?? []
  const sources = [currentDoc]
  for (const cell of siblingCells) {
    const lang = cell.language.trim().toLowerCase()
    if ((lang === 'lean' || lang === 'lean4') && cell.source !== currentDoc) {
      sources.push(cell.source)
    }
  }
  return sources
}

export const leanCompletionSource: CompletionSource = context => {
  const backslashMatch = context.matchBefore(/\\[^\s\\]*/)
  if (backslashMatch) {
    return leanUnicodeCompletionSource(context)
  }

  const identMatch = context.matchBefore(LEAN_IDENTIFIER_BEFORE_RE)
  if (!identMatch && !context.explicit) return null

  const from = identMatch ? identMatch.from : context.pos
  const prefix = identMatch ? identMatch.text : ''
  const sources = collectSourcesForCompletion(context)
  const localEntries = extractLeanDocumentSymbols(sources)
  const localOptions = matchCatalogCandidates(prefix, [...localEntries, ...LEAN_STATIC_CATALOG])

  const session = getSharedLeanWebSocketSession()
  if (!session.isWarm()) {
    if (isRemoteBrowser()) {
      void session.warm()
    }
    if (localOptions.length === 0 && !context.explicit) return null
    return { from, options: localOptions } satisfies CompletionResult
  }

  const docText = sources[0] ?? ''
  const line = context.state.doc.lineAt(context.pos)
  const position = { line: line.number - 1, character: context.pos - line.from }

  return session.complete(docText, position).then(remoteItems => {
    const merged = mergeRemoteCompletions(prefix, localOptions, remoteItems)
    if (merged.length === 0 && !context.explicit) return null
    return { from, options: merged } satisfies CompletionResult
  })
}

export function extractLeanIdentAtColumn(
  lineText: string,
  col: number,
): { readonly token: string; readonly startCol: number; readonly endCol: number } | undefined {
  if (col < 0 || col > lineText.length || lineText.length === 0) return undefined
  let probe = col
  if (probe === lineText.length || !LEAN_TOKEN_CHAR_RE.test(lineText[probe] ?? '')) {
    if (probe > 0 && LEAN_TOKEN_CHAR_RE.test(lineText[probe - 1] ?? '')) {
      probe -= 1
    } else {
      return undefined
    }
  }
  let start = probe
  while (start > 0 && LEAN_TOKEN_CHAR_RE.test(lineText[start - 1] ?? '')) {
    start -= 1
  }
  let end = probe + 1
  while (end < lineText.length && LEAN_TOKEN_CHAR_RE.test(lineText[end] ?? '')) {
    end += 1
  }
  const token = lineText.slice(start, end).replace(/\.+$/, '')
  if (!token) return undefined
  return { token, startCol: start, endCol: start + token.length }
}

export type LeanHoverSummary = {
  readonly startCol: number
  readonly endCol: number
  readonly unicodeHint?: string
  readonly signature?: string
  readonly documentation?: string
  readonly goals?: readonly string[]
}

export async function resolveLeanHoverAt(
  docText: string,
  lineNumber0: number,
  lineText: string,
  col: number,
): Promise<LeanHoverSummary | undefined> {
  const symbolHover = getLeanAbbreviationIndex().hoverInfoForTextAt(lineText, col)
  const unicodeHint = symbolHover ? symbolHover.markdown.replaceAll('`', '') : undefined

  const ident = extractLeanIdentAtColumn(lineText, col)
  const localCatalog = ident
    ? (LEAN_CATALOG_BY_LABEL.get(ident.token) ??
      extractLeanDocumentSymbols([docText]).find(entry => entry.label === ident.token))
    : undefined

  const signature = localCatalog ? `${localCatalog.label} : ${localCatalog.detail}` : undefined
  let documentation = localCatalog?.documentation
  let goals: readonly string[] | undefined

  const session = getSharedLeanWebSocketSession()
  if (session.isWarm()) {
    const liveHover = await session.hover(docText, { line: lineNumber0, character: col })
    if (liveHover?.contents) {
      documentation = liveHover.contents
    }
    if (liveHover?.goals && liveHover.goals.length > 0) {
      goals = liveHover.goals
    }
  } else if (isRemoteBrowser()) {
    void session.warm()
  }

  if (!unicodeHint && !signature && !documentation && (!goals || goals.length === 0)) {
    return undefined
  }

  const startCol = ident
    ? ident.startCol
    : symbolHover
      ? symbolHover.fromCol
      : Math.min(col, lineText.length)
  const endCol = ident
    ? ident.endCol
    : symbolHover
      ? symbolHover.toCol
      : Math.min(col + 1, lineText.length)

  return {
    startCol,
    endCol,
    ...(unicodeHint ? { unicodeHint } : {}),
    ...(signature ? { signature } : {}),
    ...(documentation ? { documentation } : {}),
    ...(goals ? { goals } : {}),
  }
}

export function renderLeanHoverHtml(summary: LeanHoverSummary): string {
  const parts: string[] = []
  if (summary.unicodeHint) {
    parts.push(
      `<div class="cm-lean-hover-unicode"><code>${escapeHTML(summary.unicodeHint)}</code></div>`,
    )
  }
  if (summary.signature) {
    parts.push(
      `<pre class="cm-lean-hover-signature"><code>${escapeHTML(summary.signature)}</code></pre>`,
    )
  }
  if (summary.documentation) {
    parts.push(`<div class="cm-lean-hover-doc">${escapeHTML(summary.documentation)}</div>`)
  }
  if (summary.goals && summary.goals.length > 0) {
    parts.push(
      `<pre class="cm-lean-hover-goals"><code>${escapeHTML(summary.goals.join('\n\n'))}</code></pre>`,
    )
  }
  return parts.join('')
}

export function leanHoverTooltipExtension(hoverTime = 220): Extension {
  return hoverTooltip(
    async (view, pos): Promise<Tooltip | null> => {
      const line = view.state.doc.lineAt(pos)
      const col = pos - line.from
      const docText = view.state.doc.toString()
      const summary = await resolveLeanHoverAt(docText, line.number - 1, line.text, col)
      if (!summary) return null

      return {
        pos: line.from + summary.startCol,
        end: line.from + summary.endCol,
        above: true,
        create() {
          const element = document.createElement('div')
          element.className = 'cm-lsp-hover-tooltip cm-lsp-documentation'
          element.innerHTML = renderLeanHoverHtml(summary)
          return { dom: element }
        },
      }
    },
    { hideOn: transaction => transaction.docChanged, hoverTime },
  )
}

export const leanLspBridge: LspBridge = {
  extensions(config: LspConfig): Promise<readonly Extension[]> {
    if (isRemoteBrowser()) {
      void getSharedLeanWebSocketSession().warm()
    }
    return Promise.resolve(config.enabled ? [leanLspConfigFacet.of(config)] : [])
  },
}
