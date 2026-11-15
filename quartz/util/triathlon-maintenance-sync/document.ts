import { JSON_SCHEMA, load } from 'js-yaml'
import { isDeepStrictEqual } from 'node:util'
import { isRecord, type UnknownRecord } from '../type-guards'

interface Node {
  start: number
  end: number
  value: unknown
  children: Node[]
}

interface Edit {
  start: number
  end: number
  text: string
}

// Keep source spans from the existing YAML parser so only owned distance values change.
export function maintenanceDocument(source: string) {
  const match = /^(?:\uFEFF)?---\r?\n([\s\S]*?)\r?\n(?:---|\.\.\.)(?:\r?\n|$)/.exec(source)
  if (!match) return null
  const yaml = match[1]
  const offset = source.indexOf(yaml)
  const stack: Node[] = []
  const mappings = new Map<UnknownRecord, Node>()
  const edits = new Map<UnknownRecord, Edit>()
  const document: unknown = load(yaml, {
    schema: JSON_SCHEMA,
    listener(event, state) {
      if (event === 'open') {
        stack.push({ start: state.position, end: state.position, value: undefined, children: [] })
        return
      }
      const node = stack.pop()
      if (!node) throw new Error('Unbalanced maintenance YAML parse')
      node.end = Math.min(state.position, yaml.length)
      node.value = state.result
      stack.at(-1)?.children.push(node)
      if (state.kind === 'mapping' && isRecord(node.value) && !mappings.has(node.value))
        mappings.set(node.value, node)
    },
  })
  if (!isRecord(document) || !isRecord(document.maintenance)) return null
  return {
    maintenance: document.maintenance,
    setDistance(target: UnknownRecord, distance: number | null) {
      const node = mappings.get(target)
      if (!node) throw new Error('Maintenance distance has no editable YAML mapping')
      const text = distance === null ? 'null' : distance.toFixed(2)
      const keyIndex = node.children.findIndex(
        (child, index) => index % 2 === 0 && child.value === 'distance',
      )
      let edit: Edit
      if (keyIndex >= 0) {
        const value = node.children[keyIndex + 1]
        if (!value || (value.value !== null && typeof value.value !== 'number'))
          throw new Error('Maintenance distance must be a number or null')
        const raw = yaml.slice(value.start, value.end)
        const trimmed = raw.trim()
        const start = value.start + (trimmed ? raw.length - raw.trimStart().length : 0)
        edit = { start, end: start + trimmed.length, text: trimmed ? text : ` ${text}` }
      } else {
        const raw = yaml.slice(node.start, node.end)
        const end = node.start + raw.trimEnd().length
        if (/^\s*(?:&\S+\s+)?\{/.test(raw)) {
          const insertion = node.start + yaml.slice(node.start, end - 1).trimEnd().length
          edit = {
            start: insertion,
            end: insertion,
            text: `${node.children.length ? ', ' : ''}distance: ${text}`,
          }
        } else {
          const firstKey = node.children[0]
          if (!firstKey) throw new Error('Cannot add a distance to an empty block mapping')
          const indent = firstKey.start - yaml.lastIndexOf('\n', firstKey.start - 1) - 1
          const newline = source.includes('\r\n') ? '\r\n' : '\n'
          edit = { start: end, end, text: `${newline}${' '.repeat(indent)}distance: ${text}` }
        }
      }
      const previous = edits.get(target)
      if (previous && previous.text !== edit.text)
        throw new Error('A shared YAML mapping has conflicting maintenance distances')
      edits.set(target, edit)
      target.distance = distance
    },
    finish() {
      let updated = yaml
      for (const edit of [...edits.values()].sort((left, right) => right.start - left.start))
        updated = updated.slice(0, edit.start) + edit.text + updated.slice(edit.end)
      if (!isDeepStrictEqual(load(updated, { schema: JSON_SCHEMA }), document))
        throw new Error('Maintenance edit changed fields beyond the calculated distances')
      return source.slice(0, offset) + updated + source.slice(offset + yaml.length)
    },
  }
}
