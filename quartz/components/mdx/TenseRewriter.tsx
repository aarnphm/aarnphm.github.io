//@ts-ignore
import script from '../scripts/fr-rewriter.inline'
import style from '../styles/frRewriter.scss'
import { FrRewriter, parseJson, type Axis, type RewriterProps, type Table } from './FrRewriter'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

const AXES: Axis[] = [
  {
    id: 'temps',
    label: 'temps',
    options: [
      { id: 'rec', label: 'passé récent' },
      { id: 'pre', label: 'présent' },
      { id: 'fp', label: 'futur proche' },
      { id: 'pc', label: 'passé composé' },
    ],
  },
]

/** Note data is `{ rec: [phrase, why], pre, fp, pc }`. */
const TenseRewriterImpl: QuartzMdxComponent<RewriterProps> = props => {
  const pairs = parseJson<Record<string, [string, string]>>(props.pairs) ?? {}
  const table: Table = Object.fromEntries(Object.entries(pairs))
  return <FrRewriter {...props} cls="fr-tense-rewriter" axes={AXES} table={table} />
}

TenseRewriterImpl.css = style
TenseRewriterImpl.afterDOMLoaded = script

export const TenseRewriter = registerMdxComponent('TenseRewriter', TenseRewriterImpl)
