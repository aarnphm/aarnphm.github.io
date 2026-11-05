import type { CompileContext, Extension, Token } from 'mdast-util-from-markdown'
import type { ReferencePart, ReferenceSyntax } from '../../util/references/model'
import { referenceFields } from '../../util/references/request'
import { isEscapedAt } from '../../util/references/scan'
import { isWikilink, type Wikilink } from './node'
import './types'

export { isWikilink }
export type { Wikilink } from './node'

interface ActiveReferenceState {
  node: Wikilink
  decoded: { target: string; fragments: string[]; settings?: string; label?: string }
  base: number
  embed: boolean
  openingEnd: number
  closingStart: number
  targetStart: number
  targetEnd: number
  settingsStart: number
  settingsEnd: number
  aliasStart: number
  aliasEnd: number
  markers: number[]
}

function buildMarkdownReference(
  state: ActiveReferenceState,
  raw: string,
  end: number,
  table: boolean,
): ReferenceSyntax {
  const base = state.base
  const span = Object.freeze({ start: base, end })
  const part = (from: number, to: number): ReferencePart =>
    Object.freeze({
      raw: raw.slice(from - base, to - base),
      span: Object.freeze({ start: from, end: to }),
    })
  const separatorStart =
    state.aliasStart >= 0
      ? state.aliasStart - (isEscapedAt(raw, state.aliasStart - base) ? 1 : 0)
      : state.closingStart
  const fragments: ReferencePart[] = []
  for (let index = 0; index < state.markers.length; index++) {
    const marker = state.markers[index]
    if (marker + 1 !== state.settingsStart)
      fragments.push(part(marker + 1, state.markers[index + 1] ?? separatorStart))
  }
  return Object.freeze({
    version: 1,
    context: 'markdown',
    source: 'authored',
    table,
    raw,
    span,
    embed: state.embed,
    target:
      state.targetStart >= 0
        ? part(state.targetStart, Math.min(state.targetEnd, separatorStart))
        : part(state.openingEnd, state.openingEnd),
    fragments: Object.freeze(fragments),
    ...(state.settingsStart >= 0
      ? {
          settings: part(
            state.settingsStart,
            state.settingsEnd >= 0 ? state.settingsEnd : state.settingsStart + 1,
          ),
        }
      : {}),
    ...(state.aliasStart >= 0
      ? {
          label: part(state.aliasEnd, state.closingStart),
          labelSeparator: part(separatorStart, state.aliasEnd),
        }
      : {}),
  })
}

export function wikilinkFromMarkdown(): Extension {
  let active: ActiveReferenceState | undefined
  function state(): ActiveReferenceState {
    if (!active) throw new Error('Wikilink compiler has no active reference')
    return active
  }
  return {
    enter: {
      wikilink(this: CompileContext, token: Token) {
        const node: Wikilink = { type: 'wikilink', value: '' }
        const start = token.start.offset
        active = {
          node,
          decoded: { target: '', fragments: [] },
          base: start,
          embed: false,
          openingEnd: start,
          closingStart: start,
          targetStart: -1,
          targetEnd: -1,
          settingsStart: -1,
          settingsEnd: -1,
          aliasStart: -1,
          aliasEnd: -1,
          markers: [],
        }
        this.enter(node, token)
      },
      wikilinkEmbedMarker() {
        state().embed = true
      },
      wikilinkTarget(this: CompileContext, token: Token) {
        state().targetStart = token.start.offset
        this.buffer()
      },
      wikilinkAnchorMarker(_token: Token) {
        state().markers.push(_token.start.offset)
      },
      wikilinkAnchor(this: CompileContext) {
        this.buffer()
      },
      wikilinkMetadataMarker(token: Token) {
        state().settingsStart = token.start.offset
      },
      wikilinkMetadata(this: CompileContext) {
        this.buffer()
      },
      wikilinkAliasMarker(token: Token) {
        state().aliasStart = token.start.offset
      },
      wikilinkAlias(this: CompileContext) {
        this.buffer()
      },
      wikilinkCloseMarker(token: Token) {
        state().closingStart = token.start.offset
      },
    },
    exit: {
      wikilinkOpenMarker(token: Token) {
        state().openingEnd = token.end.offset
      },
      wikilinkTarget(this: CompileContext, token: Token) {
        const current = state()
        current.targetEnd = token.end.offset
        current.decoded.target = this.resume()
      },
      wikilinkAnchor(this: CompileContext) {
        state().decoded.fragments.push(this.resume())
      },
      wikilinkMetadata(this: CompileContext) {
        state().decoded.settings = this.resume()
      },
      wikilinkMetadataCloseMarker(token: Token) {
        state().settingsEnd = token.end.offset
      },
      wikilinkAliasMarker(token: Token) {
        state().aliasEnd = token.end.offset
      },
      wikilinkAlias(this: CompileContext) {
        state().decoded.label = this.resume()
      },
      wikilink(this: CompileContext, token: Token) {
        const current = state()
        active = undefined
        const reference = buildMarkdownReference(
          current,
          this.sliceSerialize(token),
          token.end.offset,
          this.stack.some(parent => parent.type === 'tableCell'),
        )
        current.node.value = reference.raw
        current.node.data = {
          reference,
          referenceFields: referenceFields(reference, {
            ...current.decoded,
            ...(current.decoded.settings !== undefined
              ? { settings: `{${current.decoded.settings}}` }
              : {}),
          }),
        }
        this.exit(token)
      },
    },
  }
}
