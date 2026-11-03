import type { Root } from 'hast'
import { h } from 'hastscript'
import type { LinkItem } from '../../util/latent/markup'
import { customMacros } from '../../cfg'
// @ts-ignore
import loader from '../../components/scripts/latent-loader.inline'
import { QuartzTransformerPlugin } from '../../types/plugin'
import cachedKatex from '../../util/cached-katex'
import {
  axisCard,
  ledger,
  linksPane,
  railSvg,
  readerDocument,
  scopeFlagsFor,
} from '../../util/latent/markup'
import { defaultFocus, docOfRow, openSidecar } from '../../util/latent/sidecar'
import { LATENT, inScope, scopePrefixes } from '../../util/latent/types'

/**
 * `layout: latent` turns a note into a reader of the embedding space under `directory:`
 * (an embeddings folder, `[[embeddings]]` by default). The first screen is rendered here from the
 * sidecar on disk: rail, reader, links and probe need no fetch and no script. `embeddings:` keeps
 * its harness meaning and doubles as the scope: folder links name which documents the page is
 * about; neighbours outside the scope are greyed, never dropped.
 */

function directoryOf(value: unknown): string {
  if (typeof value !== 'string') return 'embeddings'
  const inner = value.trim().replace(/^\[\[/, '').replace(/\]\]$/, '')
  return (
    inner
      .split('|')[0]
      .split('#')[0]
      .trim()
      .replace(/^\/+|\/+$/g, '') || 'embeddings'
  )
}

// html only: the reader can hold thousands of passages, and MathML doubles every formula's DOM
const katex = cachedKatex({ output: 'html', macros: customMacros, strict: 'ignore' })

export const Latent: QuartzTransformerPlugin = () => {
  return {
    name: 'Latent',
    htmlPlugins(ctx) {
      return [
        () => (tree: Root, file) => {
          const frontmatter = file.data.frontmatter
          if (frontmatter?.pageLayout !== LATENT) return
          const directory = directoryOf(frontmatter.directory)
          const prefixes = scopePrefixes(frontmatter.embeddings)
          const sidecar = openSidecar(ctx.argv.directory, directory)
          file.data.latent = { directory, scope: prefixes, hash: sidecar?.index.hash ?? null }

          if (!sidecar) {
            tree.children.unshift(
              h('div.latent.latent-missing', { dataDirectory: directory }, [
                h('p.lat-missing', [
                  `no latent sidecar for `,
                  h('code', directory),
                  ` matches the stores on disk; run `,
                  h('code', 'uv run quartz/scripts/latent/build.py'),
                  `.`,
                ]),
              ]),
            )
            return
          }

          const { index, docs, track, probe } = sidecar
          const flags = scopeFlagsFor(docs, prefixes)
          const focus = defaultFocus(sidecar, prefixes)
          const focusDoc = docOfRow(sidecar, focus)
          const focusPreview = sidecar.preview(focus)
          const links = sidecar.knn(focus)
          const items: LinkItem[] = links.neighbours.map(n => {
            const preview = sidecar.preview(n.r)
            const doc = docs[preview.d]
            return { preview, score: n.score, doc, inScope: inScope(doc.id, prefixes) }
          })
          const neighbourDocs = new Map<number, number>()
          for (const item of items) {
            const best = neighbourDocs.get(item.doc.d) ?? 0
            if (item.score > best) neighbourDocs.set(item.doc.d, item.score)
          }
          const head = sidecar.text(focusDoc.d)
          const focusChunkIndex = Math.max(0, focus - focusDoc.r0)
          const part = Math.floor(focusChunkIndex / head.part)
          const text = { ...head, chunks: sidecar.textPart(focusDoc.d, part) }
          const offset = part * head.part

          const rail = h('aside.lat-rail', [
            h('header.lat-pane-head', [
              h('span', 'time'),
              h('span.lat-count', `${index.datedDocs.toLocaleString('en-US')} dated`),
            ]),
            railSvg({
              docs: track,
              titles: d => docs[d]?.t ?? '',
              open: focusDoc.d,
              focus,
              neighbours: neighbourDocs,
              focusChunkIndex,
              scopeFlags: flags,
            }),
          ])

          const prev = focusDoc.d > 0 ? docs[focusDoc.d - 1] : null
          const next = focusDoc.d + 1 < docs.length ? docs[focusDoc.d + 1] : null
          const stub = (doc: typeof focusDoc) =>
            readerDocument(
              {
                d: doc.d,
                id: doc.id,
                store: doc.s,
                title: doc.t,
                date: doc.date,
                prec: doc.prec,
                total: doc.n,
                part: 0,
                chunks: [],
              },
              doc,
              -1,
              { stub: true },
            )
          const reader = h('section.lat-reader', { dataFocus: String(focus) }, [
            prev ? stub(prev) : null,
            readerDocument(text, focusDoc, focus, { offset }),
            next ? stub(next) : null,
          ])

          const linksEl = linksPane(focusPreview, items, index.kinds)

          const probeEl = h('aside.lat-probe', [
            h('header.lat-pane-head', [
              h('span', 'probe'),
              h('nav.lat-tabs', { role: 'tablist' }, [
                h(
                  'button.lat-tab',
                  { type: 'button', dataTab: 'axes', ariaSelected: 'true' },
                  'axes',
                ),
                h(
                  'button.lat-tab',
                  { type: 'button', dataTab: 'bench', ariaSelected: 'false' },
                  '3-d',
                ),
              ]),
            ]),
            h(
              'div.lat-panel',
              { dataPanel: 'axes' },
              probe.axes.slice(0, 4).map((axis, i) => axisCard(axis, links.axes[i] ?? null)),
            ),
            h('div.lat-panel.lat-bench', { dataPanel: 'bench', hidden: true }, [
              h('div.lat-bench-controls'),
              h('canvas.lat-bench-canvas', { width: 320, height: 320 }),
              h('p.lat-bench-status'),
            ]),
            ledger(index, prefixes),
          ])

          const root = h(
            'div.latent',
            {
              dataLatentBase: sidecar.base,
              dataLatentHash: index.hash,
              dataLatentFocus: String(focus),
              dataLatentDoc: String(focusDoc.d),
              dataLatentK: String(index.k),
              dataLatentBlock: String(index.block),
              dataLatentAxes: String(index.axes),
              dataLatentKinds: index.kinds.join(','),
              dataLatentScope: prefixes.join(','),
              dataLatentRows: String(index.rows),
              dataLatentDocs: String(index.docs),
              dataLatentMacros: JSON.stringify(customMacros),
            },
            [rail, reader, linksEl, probeEl],
          )
          // passages carry math as `.math-inline` / `.math-display` spans; the Latex plugin's
          // html pass ran before this one, so render them here with the notes' macros
          katex({ type: 'root', children: [root] }, file)
          tree.children.unshift(root)
        },
      ]
    },
    externalResources() {
      return { js: [{ script: loader, contentType: 'inline', loadTime: 'afterDOMReady' }] }
    },
  }
}

declare module 'vfile' {
  interface DataMap {
    latent?: { directory: string; scope: string[]; hash: string | null }
  }
}
