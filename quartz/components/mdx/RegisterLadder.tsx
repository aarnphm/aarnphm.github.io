//@ts-ignore
import script from '../scripts/fr-rewriter.inline'
import style from '../styles/frRewriter.scss'
import { FrRewriter, parseJson, type Axis, type RewriterProps, type Table } from './FrRewriter'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

const AXES: Axis[] = [
  {
    id: 'registre',
    label: 'registre',
    options: [
      { id: 'fam', label: 'familier' },
      { id: 'std', label: 'standard' },
      { id: 'sou', label: 'soutenu' },
    ],
  },
  {
    id: 'variete',
    label: 'variété',
    options: [
      { id: 'qc', label: 'Québec' },
      { id: 'fr', label: 'France' },
    ],
  },
]

/** Note data is `{ qc: { fam: [phrase, why], std, sou }, fr: { … } }`. */
type Pairs = Record<string, Record<string, [string, string]>>

const flatten = (pairs: Pairs): Table => {
  const table: Table = {}
  for (const [variety, levels] of Object.entries(pairs))
    for (const [level, pair] of Object.entries(levels)) table[`${level}.${variety}`] = pair
  return table
}

const RegisterLadderImpl: QuartzMdxComponent<RewriterProps> = props => {
  const pairs = parseJson<Pairs>(props.pairs) ?? {}
  return <FrRewriter {...props} cls="fr-register-ladder" axes={AXES} table={flatten(pairs)} />
}

RegisterLadderImpl.css = style
RegisterLadderImpl.afterDOMLoaded = script

export const RegisterLadder = registerMdxComponent('RegisterLadder', RegisterLadderImpl)
