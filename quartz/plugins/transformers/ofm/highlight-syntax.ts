import type { Code, Construct, Event, Resolver, State, Token } from 'micromark-util-types'
import {
  markdownLineEndingOrSpace,
  unicodePunctuation,
  unicodeWhitespace,
} from 'micromark-util-character'

function classify(code: Code) {
  if (code === null || markdownLineEndingOrSpace(code) || unicodeWhitespace(code)) return 1
  if (unicodePunctuation(code)) return 2
}

// Use GFM's attention-delimiter model so highlights share Markdown span boundaries.
const resolveHighlights: Resolver = (events, context) => {
  const available = context.parser.constructs.text[42]
  const constructs = Array.isArray(available) ? available : available ? [available] : []
  const attention = constructs.find(construct => construct.name === 'attention')
  if (attention?.resolveAll) events = attention.resolveAll(events, context)
  for (let index = 0; index < events.length; index++) {
    const closer = events[index]
    if (closer[0] !== 'enter' || closer[1].type !== 'ofmHighlightTemporary' || !closer[1]._close)
      continue
    for (let open = index - 1; open >= 0; open--) {
      const opener = events[open]
      if (opener[0] !== 'exit' || opener[1].type !== 'ofmHighlightTemporary' || !opener[1]._open)
        continue
      closer[1].type = 'ofmHighlightMarker'
      opener[1].type = 'ofmHighlightMarker'
      const highlight: Token = {
        type: 'ofmHighlight',
        start: { ...opener[1].start },
        end: { ...closer[1].end },
      }
      const text: Token = {
        type: 'ofmHighlightText',
        start: { ...opener[1].end },
        end: { ...closer[1].start },
      }
      let inside = events.slice(open + 1, index)
      const constructs = context.parser.constructs.insideSpan.null ?? []
      const called = new Set<Resolver>()
      for (const construct of constructs) {
        if (construct.resolveAll && !called.has(construct.resolveAll)) {
          inside = construct.resolveAll(inside, context)
          called.add(construct.resolveAll)
        }
      }
      const replacement: Event[] = [
        ['enter', highlight, context],
        ['enter', opener[1], context],
        ['exit', opener[1], context],
        ['enter', text, context],
        ...inside,
        ['exit', text, context],
        ['enter', closer[1], context],
        ['exit', closer[1], context],
        ['exit', highlight, context],
      ]
      events.splice(open - 1, index - open + 3)
      for (let cursor = 0; cursor < replacement.length; cursor += 1000) {
        events.splice(open - 1 + cursor, 0, ...replacement.slice(cursor, cursor + 1000))
      }
      index = open + replacement.length - 2
      break
    }
  }
  for (const event of events) if (event[1].type === 'ofmHighlightTemporary') event[1].type = 'data'
  return events
}

export const highlightConstruct: Construct = {
  name: 'ofmHighlight',
  resolveAll: resolveHighlights,
  tokenize(effects, ok, nok) {
    const previous = this.previous
    const events = this.events
    let size = 0
    return start
    function start(code: Code): State | undefined {
      if (previous === 61 && events.at(-1)?.[1].type !== 'characterEscape') return nok(code)
      effects.enter('ofmHighlightTemporary')
      return more(code)
    }
    function more(code: Code): State | undefined {
      if (code === 61) {
        if (size === 2) return nok(code)
        effects.consume(code)
        size++
        return more
      }
      if (size !== 2) return nok(code)
      const token = effects.exit('ofmHighlightTemporary')
      const before = classify(previous)
      const after = classify(code)
      token._open = !after || (after === 2 && Boolean(before))
      token._close = !before || (before === 2 && Boolean(after))
      return ok(code)
    }
  },
}
