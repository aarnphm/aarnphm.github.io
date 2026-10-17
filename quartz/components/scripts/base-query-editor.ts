import {
  autocompletion,
  closeCompletion,
  completionKeymap,
  snippetCompletion,
  type Completion,
  type CompletionContext,
} from '@codemirror/autocomplete'
import { defaultKeymap, history, historyKeymap } from '@codemirror/commands'
import { javascript } from '@codemirror/lang-javascript'
import { defaultHighlightStyle, HighlightStyle, syntaxHighlighting } from '@codemirror/language'
import { EditorState, Prec } from '@codemirror/state'
import { EditorView, keymap, placeholder } from '@codemirror/view'
import type { QuartzPluginData } from '../../plugins/vfile'
import type { BaseFile } from '../../util/base/types'
import { buildPropertyExpressionSource } from '../../util/base/compiler/properties'
import { completionTheme } from '../../util/codemirror-completion-theme'
import { isRecord } from '../../util/type-guards'

export interface BaseQueryCatalog {
  properties: string[]
  values: Map<string, string[]>
  tags: string[]
  folders: string[]
}

const fileProperties = [
  'name',
  'basename',
  'title',
  'path',
  'folder',
  'ext',
  'size',
  'ctime',
  'mtime',
  'tags',
  'aliases',
  'links',
  'backlinks',
  'embeds',
].map(property => `file.${property}`)

const tokenColors: Record<string, string> = {
  '#404740': '--darkgray',
  '#708': '--code-keyword',
  '#219': '--code-value',
  '#164': '--code-value',
  '#a11': '--code-string',
  '#e40': '--code-important',
  '#00f': '--code-function',
  '#30a': '--code-normal',
  '#085': '--code-property',
  '#167': '--code-property',
  '#256': '--code-property',
  '#00c': '--code-property',
  '#940': '--darkgray',
  '#f00': '--code-operator',
}

const queryHighlightStyle = HighlightStyle.define(
  defaultHighlightStyle.specs.map(style => ({
    ...style,
    ...(typeof style.color === 'string'
      ? { color: `var(${tokenColors[style.color] ?? '--dark'})` }
      : {}),
  })),
)

export function createBaseQueryCatalog(
  files: QuartzPluginData[],
  config: BaseFile,
): BaseQueryCatalog {
  const properties = new Set([
    ...fileProperties,
    ...fileProperties.map(property => `this.${property}`),
  ])
  const values = new Map<string, Set<string>>()
  const tags = new Set<string>()
  const folders = new Set<string>()
  const collect = (property: string, value: unknown): void => {
    properties.add(property)
    if (Array.isArray(value)) {
      for (const item of value) collect(property, item)
      return
    }
    if (isRecord(value) && !(value instanceof Date)) {
      if (property.split('.').length < 4) {
        for (const [key, item] of Object.entries(value)) collect(`${property}.${key}`, item)
      }
      return
    }
    const scalar = value instanceof Date ? value.toISOString().slice(0, 10) : value
    if (typeof scalar !== 'string' && typeof scalar !== 'number' && typeof scalar !== 'boolean')
      return
    const text = String(scalar)
    if (text.length === 0 || text.length > 100 || text.includes('\n')) return
    const observed = values.get(property) ?? new Set<string>()
    if (observed.size < 80) observed.add(text)
    values.set(property, observed)
  }

  for (const file of files) {
    for (const [key, value] of Object.entries(file.frontmatter ?? {})) collect(`note.${key}`, value)
    // hasTag("a") also matches a/b, so offer every parent of a nested tag.
    for (const tag of file.frontmatter?.tags ?? []) {
      const parts = String(tag).replace(/^#/, '').split('/')
      for (let index = 1; index <= parts.length; index++) tags.add(parts.slice(0, index).join('/'))
    }
    const path = String(file.filePath ?? file.slug ?? '')
    const segments = path.split('/')
    segments.pop()
    for (let index = 1; index <= segments.length; index++)
      folders.add(segments.slice(0, index).join('/'))
  }
  for (const key of Object.keys(config.properties ?? {})) {
    properties.add(/^(?:note|file|formula)\./.test(key) ? key : `note.${key}`)
  }
  for (const key of Object.keys(config.formulas ?? {})) properties.add(`formula.${key}`)
  return {
    properties: [...properties].sort(),
    values: new Map([...values].map(([key, entries]) => [key, [...entries].sort()])),
    tags: [...tags].sort(),
    folders: [...folders].filter(Boolean).sort(),
  }
}

const functions: Completion[] = [
  ['file.hasTag("${tag}")', 'file.hasTag()', 'Match a tag, including nested tags'],
  ['file.inFolder("${folder}")', 'file.inFolder()', 'Match a folder and its descendants'],
  ['file.hasProperty("${property}")', 'file.hasProperty()', 'Test whether a property exists'],
  ['file.hasLink("${path}")', 'file.hasLink()', 'Match a linked note'],
  ['date("${2026-01-01}")', 'date()', 'Convert a value to a date'],
  ['today()', 'today()', 'Current date'],
  ['now()', 'now()', 'Current date and time'],
  ['duration("${7d}")', 'duration()', 'A duration, such as 7d'],
  ['number(${value})', 'number()', 'Convert a value to a number'],
  ['list(${value})', 'list()', 'Convert a value to a list'],
  ['link("${path}")', 'link()', 'A link to a note'],
  ['file("${path}")', 'file()', 'Look up a file'],
  ['if(${condition}, ${then}, ${else})', 'if()', 'Choose a value from a condition'],
  ['min(${a}, ${b})', 'min()', 'Smallest value'],
  ['max(${a}, ${b})', 'max()', 'Largest value'],
  ['image(${path})', 'image()', 'Render an image'],
  ['icon("${name}")', 'icon()', 'Render an icon'],
].map(([template, label, detail]) =>
  snippetCompletion(template, { label, detail, type: 'function' }),
)

const methods: Completion[] = [
  ['contains("${value}")', 'contains()', 'Contains a value'],
  ['containsAny("${value}")', 'containsAny()', 'Contains any supplied value'],
  ['containsAll("${value}")', 'containsAll()', 'Contains every supplied value'],
  ['isEmpty()', 'isEmpty()', 'Value is empty or missing'],
  ['isTruthy()', 'isTruthy()', 'Value is truthy'],
  ['isType("${string}")', 'isType()', 'Check the value type'],
  ['startsWith("${prefix}")', 'startsWith()', 'Text starts with a prefix'],
  ['endsWith("${suffix}")', 'endsWith()', 'Text ends with a suffix'],
  ['lower()', 'lower()', 'Lowercase text'],
  ['trim()', 'trim()', 'Remove surrounding whitespace'],
  ['split("${separator}")', 'split()', 'Split text into a list'],
  ['join("${separator}")', 'join()', 'Join a list as text'],
  ['filter(${value})', 'filter()', 'Keep matching list members'],
  ['map(${value})', 'map()', 'Transform list members'],
  ['reduce(${1:acc + value}, ${2:0})', 'reduce()', 'Combine list members'],
  ['unique()', 'unique()', 'Distinct list members'],
  ['sort()', 'sort()', 'Sort list members'],
  ['format("${YYYY-MM-DD}")', 'format()', 'Format a date'],
  ['asFile()', 'asFile()', 'Resolve a link to a file'],
].map(([template, label, detail]) => snippetCompletion(template, { label, detail, type: 'method' }))

function completions(catalog: BaseQueryCatalog) {
  const propertyOptions: Completion[] = catalog.properties.map(property => ({
    label: buildPropertyExpressionSource(property) ?? property,
    detail: property.startsWith('formula.') ? 'formula' : 'property',
    type: 'property',
  }))
  return (context: CompletionContext) => {
    const line = context.state.doc.lineAt(context.pos)
    const before = context.state.sliceDoc(line.from, context.pos)
    const quote = before.match(/(["'])([^"']*)$/)
    if (quote) {
      const prefix = before.slice(0, quote.index)
      const property = prefix.match(
        /([\w.]+)\s*(?:[!=<>]=?|\.(?:contains(?:Any|All)?|startsWith)\()\s*$/,
      )?.[1]
      const observed = /file\.hasTag\((?:\s*(["'])[^"']*\1\s*,)*\s*$/.test(prefix)
        ? catalog.tags
        : /file\.inFolder\(\s*$/.test(prefix) || /(?:^|\.)file\.folder$/.test(property ?? '')
          ? catalog.folders
          : catalog.values.get(property?.startsWith('note.') ? property : `note.${property}`)
      if (observed?.length) {
        const delimiter = quote[1]
        const closing =
          context.state.sliceDoc(context.pos, context.pos + 1) === delimiter ? '' : delimiter
        return {
          from: context.pos - quote[2].length,
          options: observed.map(value => ({
            label: value,
            apply: value.replaceAll('\\', '\\\\').replaceAll(delimiter, `\\${delimiter}`) + closing,
            type: 'constant',
            detail: 'observed value',
          })),
        }
      }
    }

    const word = context.matchBefore(/[\w.]*/)
    if (!word || (!context.explicit && word.from === word.to)) return null
    const member = word.text.match(/^(.+)\.([\w]*)$/)
    if (member && !['file', 'note', 'formula', 'this'].includes(member[1])) {
      const prefix = `${member[1]}.`
      const descendants = propertyOptions
        .filter(option => option.label.startsWith(prefix))
        .map(option => ({ ...option, label: option.label.slice(prefix.length) }))
      return {
        from: context.pos - member[2].length,
        options: [...descendants, ...methods],
        validFor: /^\w*$/,
      }
    }
    const options = [
      ...propertyOptions,
      ...functions,
      ...['true', 'false', 'null'].map(label => ({ label, type: 'keyword' })),
    ]
    return { from: word.from, options, validFor: /^[\w.]*$/ }
  }
}

export interface BaseQueryEditor {
  value(): string
  destroy(): void
}

export function createBaseQueryEditor(options: {
  parent: HTMLElement
  source: string
  catalog: BaseQueryCatalog
  onChange: () => void
  onApply: () => void
  onClose: () => void
}): BaseQueryEditor {
  const view = new EditorView({
    parent: options.parent,
    state: EditorState.create({
      doc: options.source,
      extensions: [
        javascript(),
        syntaxHighlighting(queryHighlightStyle),
        history(),
        EditorView.lineWrapping,
        EditorState.tabSize.of(2),
        EditorView.contentAttributes.of({ 'aria-label': 'Filter expression' }),
        placeholder('true'),
        autocompletion({ override: [completions(options.catalog)], icons: false }),
        completionTheme,
        Prec.highest(
          keymap.of([
            {
              key: 'Mod-Enter',
              run: () => {
                options.onApply()
                return true
              },
            },
            {
              key: 'Escape',
              run: editor => {
                if (!closeCompletion(editor)) options.onClose()
                return true
              },
            },
          ]),
        ),
        keymap.of([...completionKeymap, ...defaultKeymap, ...historyKeymap]),
        EditorView.updateListener.of(update => {
          if (update.docChanged) options.onChange()
        }),
      ],
    }),
  })
  return { value: () => view.state.doc.toString(), destroy: () => view.destroy() }
}
