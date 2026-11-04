import { type Element } from 'hast'
import { h } from 'hastscript'
import { mark } from '../../../util/latent/markup'
import { truncate } from '../../../util/latent/rail'
import { wordsOf, KIND_ORDER } from './features-data'
import { ancestry, steerKey, featureName, share, amount, chip, timeline } from './features-markup'
import { type FeatureState, type FeatureDict } from './features-types'
import { type Runtime, previewOf } from './runtime'

export async function featureCard(
  rt: Runtime,
  st: FeatureState,
  dict: FeatureDict,
  f: number,
): Promise<Element> {
  const row = dict.table[f]
  const up = ancestry(dict, f)
  const kids = dict.children.get(f) ?? []
  const [cross, words] = await Promise.all([
    Promise.all(row.cross.slice(0, 4).map(([r]) => previewOf(rt, r))),
    dict.meta.name === 'text' ? wordsOf(rt, st) : Promise.resolve(null),
  ])
  const unit = st.tablesNow?.dicts[dict.meta.name]?.[f]
  const said = words?.features[String(f)]
  const inTags = (dict.meta.name === 'text' ? (st.tablesNow?.tags ?? []) : []).filter(t =>
    t.explain.some(([g]) => g === f),
  )
  const kinds = KIND_ORDER.map((kind, i) =>
    row.kinds[i] ? h('span', [mark(kind), ` ${row.kinds[i].toLocaleString('en-US')}`]) : null,
  ).filter(Boolean) as Element[]
  const key = steerKey(dict.meta.name, f)
  const steered = st.steer.get(key)?.w ?? 0
  return h('section.lat-feat-card', { dataDict: dict.meta.name, dataF: String(f) }, [
    h('header', [
      h('span.lat-feat-head', featureName(row)),
      h('span.lat-feat-card-tools', [
        h(
          'span.lat-chip',
          `${dict.meta.name} #${f}${row.level === undefined ? '' : ` · L${row.level}`} · ${share(dict, row)}`,
        ),
        // add this feature to the steering set (another press steps it a unit further)
        h(
          'button.lat-feat-add',
          {
            type: 'button',
            dataTerm: key,
            dataSteer: '1',
            title: steered
              ? `steer one unit further toward it (now ${amount(steered)})`
              : 'add it to steering',
          },
          '+',
        ),
      ]),
    ]),
    up.length
      ? h('p.lat-feat-path', [
          h('span', 'in'),
          ...up.map((g, i) =>
            chip(
              dict,
              g,
              i === up.length - 1 && row.share
                ? `fires in ${(row.share * 100).toFixed(0)} % of this feature's rows`
                : undefined,
            ),
          ),
        ])
      : null,
    // one line, cut with an ellipsis; hover scrolls it to its end (marquee.ts)
    row.about ? h('p.lat-feat-about', row.about) : null,
    // the label's score, and an arrow whose note lists the label's terms
    h('p.lat-feat-terms', [
      row.score === undefined
        ? null
        : h(
            'span.lat-feat-score',
            {
              dataTip:
                'detection score: a second prompt picks, from 5 held-out rows that fire the feature and 5 that do not, the ones the label fits. 0.5 is chance.',
              dataTipAt: 'up',
              tabIndex: 0,
            },
            `label ${row.score.toFixed(2)}`,
          ),
      row.terms.length
        ? h(
            'span.lat-feat-more',
            { dataTip: row.terms.join(' · '), tabIndex: 0, ariaLabel: 'all terms' },
            '→',
          )
        : null,
    ]),
    said?.w.length
      ? h(
          'p.lat-feat-near.lat-feat-words',
          {
            dataTip:
              'integrated gradients from the encoder’s input tokens to this feature, on its strongest rows: the words that make it fire, with their share of the attribution',
            dataTipAt: 'up',
            tabIndex: 0,
          },
          [
            h('span', 'words'),
            ...said.w.map(([w, s]) =>
              h('span.lat-feat-word', { style: `--a:${Math.min(1, s * 4).toFixed(2)}` }, w),
            ),
          ],
        )
      : null,
    h('p.lat-feat-kinds', kinds),
    timeline(dict, row),
    unit
      ? h(
          'p.lat-feat-near.lat-feat-unit',
          {
            dataTip:
              'steering ranks rows by cosine plus weight × this feature’s detector; one unit is the weight at which half of a focus’s links fire it',
            dataTipAt: 'up',
            tabIndex: 0,
          },
          [
            h('span', 'steers'),
            h(
              'span',
              unit[2]
                ? `weakly: under half the links fire it at ${st.tablesNow!.grid}`
                : `1 unit = ${unit[0]} · ${Math.round(unit[1] * 100)} % of links fire it`,
            ),
          ],
        )
      : null,
    inTags.length
      ? h('p.lat-feat-near', [
          h('span', 'explains'),
          ...inTags.map(t =>
            h(
              'button.lat-feat-chip',
              {
                type: 'button',
                dataTerm: `tag:${t.t}`,
                dataSteer: '1',
                title: `one of the 5 features of the #${t.t} sparse probe; click to steer toward #${t.t}`,
              },
              `#${t.t}`,
            ),
          ),
        ])
      : null,
    kids.length
      ? h('p.lat-feat-near', [
          h('span', 'finer'),
          ...kids
            .slice(0, 8)
            .map(g => chip(dict, g, `${dict.table[g].n.toLocaleString('en-US')} rows`)),
          kids.length > 8 ? h('span.lat-count', `+${kids.length - 8}`) : null,
        ])
      : null,
    row.near.length
      ? h('p.lat-feat-near', [
          h('span', 'near'),
          ...row.near.map(([g, c]) => chip(dict, g, `decoder cosine ${c.toFixed(2)}`)),
        ])
      : null,
    cross.length
      ? h('div.lat-feat-cross', [
          h('span', dict.meta.name === 'text' ? 'also lights' : 'in text'),
          h(
            'ol',
            cross.map(p =>
              h('li', [
                mark(p.s),
                h('a', { href: `?focus=${p.r}`, dataLatFocus: String(p.r) }, truncate(p.t, 40)),
              ]),
            ),
          ),
        ])
      : null,
  ])
}
