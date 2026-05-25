import {
  notebookSuccessOutputLabel,
  type NotebookRuntimeCell,
  type NotebookRuntimeData,
} from '../../runtime/notebook/types'
import { escapeHTML } from '../escape'
import { notebookIconSvg, notebookLanguageIconSvg } from './render/icons'

type NotebookIcon = 'run' | 'stop' | 'reset' | 'debug' | 'edit' | 'save' | 'revert' | 'vim'

type NotebookLanguageInfo = { token: string; label: string; glyph: string }

export type NotebookRenderedOutput = { label: string; html: string }

const notebookLanguageAliases = new Map<string, NotebookLanguageInfo>([
  ['python', { token: 'python', label: 'Python', glyph: 'Py' }],
  ['py', { token: 'python', label: 'Python', glyph: 'Py' }],
  ['ipython', { token: 'python', label: 'Python', glyph: 'Py' }],
  ['javascript', { token: 'javascript', label: 'JavaScript', glyph: 'JS' }],
  ['js', { token: 'javascript', label: 'JavaScript', glyph: 'JS' }],
  ['typescript', { token: 'typescript', label: 'TypeScript', glyph: 'TS' }],
  ['ts', { token: 'typescript', label: 'TypeScript', glyph: 'TS' }],
  ['tsx', { token: 'typescript', label: 'TSX', glyph: 'TSX' }],
  ['jsx', { token: 'javascript', label: 'JSX', glyph: 'JSX' }],
  ['java', { token: 'java', label: 'Java', glyph: 'Ja' }],
  ['go', { token: 'go', label: 'Go', glyph: 'Go' }],
  ['golang', { token: 'go', label: 'Go', glyph: 'Go' }],
  ['rust', { token: 'rust', label: 'Rust', glyph: 'Rs' }],
  ['rs', { token: 'rust', label: 'Rust', glyph: 'Rs' }],
  ['c', { token: 'c', label: 'C', glyph: 'C' }],
  ['cpp', { token: 'cpp', label: 'C++', glyph: 'C++' }],
  ['c++', { token: 'cpp', label: 'C++', glyph: 'C++' }],
  ['cxx', { token: 'cpp', label: 'C++', glyph: 'C++' }],
  ['bash', { token: 'bash', label: 'Bash', glyph: 'sh' }],
  ['sh', { token: 'bash', label: 'Shell', glyph: 'sh' }],
  ['shell', { token: 'bash', label: 'Shell', glyph: 'sh' }],
  ['shellscript', { token: 'bash', label: 'Shell', glyph: 'sh' }],
  ['wasm', { token: 'wasm', label: 'WASM', glyph: 'Wasm' }],
  ['wat', { token: 'wasm', label: 'WAT', glyph: 'WAT' }],
  ['html', { token: 'html', label: 'HTML', glyph: 'HTML' }],
  ['css', { token: 'css', label: 'CSS', glyph: 'CSS' }],
  ['json', { token: 'json', label: 'JSON', glyph: '{}' }],
  ['csv', { token: 'csv', label: 'CSV', glyph: 'CSV' }],
  ['markdown', { token: 'markdown', label: 'Markdown', glyph: 'md' }],
  ['md', { token: 'markdown', label: 'Markdown', glyph: 'md' }],
  ['zig', { token: 'zig', label: 'Zig', glyph: 'Zig' }],
  ['ocaml', { token: 'ocaml', label: 'OCaml', glyph: 'ML' }],
  ['ml', { token: 'ocaml', label: 'OCaml', glyph: 'ML' }],
  ['text', { token: 'text', label: 'Text', glyph: 'txt' }],
])

function classToken(value: string): string {
  return value.replace(/[^A-Za-z0-9_-]/g, '-') || 'output'
}

function outputTabId(label: string): string {
  return classToken(label.toLowerCase())
}

function notebookLanguageGlyph(label: string): string {
  const compact = label.replace(/[^A-Za-z0-9+#]/g, '')
  if (compact.length <= 4) return compact || 'code'
  const uppercase = compact.replace(/[^A-Z0-9]/g, '')
  return uppercase.length >= 2 ? uppercase.slice(0, 4) : compact.slice(0, 2)
}

function notebookLanguageInfo(language: string): NotebookLanguageInfo {
  const normalized = language.trim().toLowerCase()
  const alias = notebookLanguageAliases.get(normalized)
  if (alias) return alias
  const label = language.trim() || 'Code'
  const token = classToken(normalized || label.toLowerCase())
  return { token, label, glyph: notebookLanguageGlyph(label) }
}

function notebookExecutionLabel(count: number | null): string {
  return count === null ? 'In [ ]:' : `In [${count}]:`
}

function notebookIconButton(
  cellId: string,
  icon: NotebookIcon,
  dataAttribute: string,
  label: string,
  hidden = false,
): string {
  const escapedId = escapeHTML(cellId)
  const escapedLabel = escapeHTML(label)
  return `<button type="button" class="notebook-icon-button" ${dataAttribute}="${escapedId}" aria-label="${escapedLabel}" title="${escapedLabel}"${
    hidden ? ' hidden' : ''
  }>${notebookIconSvg[icon]}</button>`
}

function notebookToolbarButton(
  icon: NotebookIcon,
  dataAttribute: string,
  label: string,
  options: { disabled?: boolean; pressed?: boolean } = {},
): string {
  const escapedLabel = escapeHTML(label)
  const disabled = options.disabled === true ? ' disabled' : ''
  const pressed = options.pressed === undefined ? '' : ` aria-pressed="${String(options.pressed)}"`
  return `<button type="button" class="notebook-icon-button notebook-toolbar-button" ${dataAttribute} aria-label="${escapedLabel}" title="${escapedLabel}" data-notebook-tooltip="${escapedLabel}"${pressed}${disabled}>${notebookIconSvg[icon]}</button>`
}

export function notebookRuntimeJson(value: NotebookRuntimeData): string {
  return JSON.stringify(value)
    .replace(/</g, '\\u003c')
    .replace(/>/g, '\\u003e')
    .replace(/&/g, '\\u0026')
    .replace(/\u2028/g, '\\u2028')
    .replace(/\u2029/g, '\\u2029')
}

export function notebookRuntimeControls(data: NotebookRuntimeData): string[] {
  const root = `<div class="notebook-runtime" data-notebook-runtime="${escapeHTML(data.id)}">`
  if (data.toolbar === false) return [`${root}</div>`]
  return [
    [
      root,
      '<div class="notebook-runtime-toolbar" data-notebook-runtime-toolbar role="toolbar" aria-label="Notebook runtime">',
      notebookToolbarButton('run', 'data-notebook-run-all', 'Run all'),
      notebookToolbarButton('stop', 'data-notebook-stop', 'Stop execution', { disabled: true }),
      notebookToolbarButton('reset', 'data-notebook-reset', 'Reset runtime'),
      notebookToolbarButton(
        'debug',
        'data-notebook-debug',
        data.debug === true ? 'Disable debug output' : 'Enable debug output',
        { pressed: data.debug === true },
      ),
      notebookToolbarButton(
        'vim',
        'data-notebook-vim-mode',
        data.vimMode === true ? 'Disable Vim mode' : 'Enable Vim mode',
        { pressed: data.vimMode === true },
      ),
      '<span class="notebook-runtime-status" data-notebook-status aria-live="polite">idle</span>',
      '</div>',
      '</div>',
    ].join('\n'),
  ]
}

export function notebookRuntimeDataScript(data: NotebookRuntimeData): string {
  return `<script type="application/json" data-notebook-runtime-data>${notebookRuntimeJson(data)}</script>`
}

export function notebookCellLanguageBadge(language: string): string {
  const info = notebookLanguageInfo(language)
  const escapedToken = escapeHTML(info.token)
  const escapedLabel = escapeHTML(info.label)
  const escapedGlyph = escapeHTML(info.glyph)
  const visual =
    notebookLanguageIconSvg[info.token] ??
    `<span class="notebook-language-text">${escapedGlyph}</span>`
  return `<span class="notebook-language-badge notebook-language-badge-${escapedToken}" data-notebook-language="${escapedToken}" title="${escapedLabel} cell"><span class="notebook-language-icon" aria-hidden="true">${visual}</span><span class="notebook-language-label">${escapedLabel} cell</span></span>`
}

export function notebookStaticCellActions(cellId: string, language: string): string {
  const escaped = escapeHTML(cellId)
  return [
    `<div class="notebook-cell-actions" data-notebook-cell-actions="${escaped}">`,
    notebookCellLanguageBadge(language),
    '</div>',
  ].join('\n')
}

export function notebookCellControls(cell: NotebookRuntimeCell): string[] {
  const cellId = cell.id
  const escaped = escapeHTML(cellId)
  return [
    `<div class="notebook-runtime-cell" data-notebook-cell="${escaped}" data-notebook-execution-count="${cell.executionIndex ?? ''}"><span class="notebook-execution-prompt" data-notebook-execution-label="${escaped}" aria-live="polite">${escapeHTML(
      notebookExecutionLabel(cell.executionIndex),
    )}</span></div>`,
  ]
}

export function notebookCellFrameOpen(cellId: string, language?: string): string {
  const escaped = escapeHTML(cellId)
  const languageAttr =
    language === undefined
      ? ''
      : ` data-notebook-language="${escapeHTML(notebookLanguageInfo(language).token)}"`
  return `<div class="notebook-code-cell" data-notebook-cell-frame="${escaped}" id="${escaped}"${languageAttr}>`
}

export function notebookCellActions(cell: NotebookRuntimeCell): string {
  const escaped = escapeHTML(cell.id)
  const language = cell.displayLanguage ?? cell.language
  return [
    `<div class="notebook-cell-actions" data-notebook-cell-actions="${escaped}">`,
    notebookCellLanguageBadge(language),
    notebookIconButton(cell.id, 'run', 'data-notebook-run-cell', `Run ${cell.id}`),
    notebookIconButton(cell.id, 'edit', 'data-notebook-edit-cell', `Edit ${cell.id}`),
    notebookIconButton(cell.id, 'save', 'data-notebook-save-cell', `Save ${cell.id} locally`, true),
    notebookIconButton(
      cell.id,
      'revert',
      'data-notebook-revert-cell',
      `Revert ${cell.id} local edit`,
      true,
    ),
    notebookIconButton(cell.id, 'vim', 'data-notebook-vim-cell', 'Enable Vim mode', true),
    `<span class="notebook-local-source-status" data-notebook-local-source-status="${escaped}" hidden></span>`,
    '</div>',
  ].join('\n')
}

export function notebookSourceEditor(cellId: string): string {
  return `<div class="notebook-source-editor" data-notebook-source-editor="${escapeHTML(cellId)}" hidden></div>`
}

export function notebookCellRuntimeOutput(cellId: string): string {
  return `<div class="notebook-runtime-output" data-notebook-output="${escapeHTML(cellId)}" hidden></div>`
}

export function notebookStaticOutputTabs(
  cellId: string,
  outputs: readonly NotebookRenderedOutput[],
): string {
  if (outputs.length === 0) return ''
  const groups = new Map<string, { id: string; label: string; outputs: string[] }>()
  for (const output of outputs) {
    const label = output.label.trim() || 'output'
    const id = outputTabId(label)
    const group = groups.get(id) ?? { id, label, outputs: [] }
    group.outputs.push(output.html)
    groups.set(id, group)
  }
  const orderedGroups = [...groups.values()]
  const activeId = orderedGroups.find(group => group.label !== notebookSuccessOutputLabel)?.id
  const outputId = classToken(cellId)
  const collapsed = activeId === undefined
  const tablist = orderedGroups
    .map((group, index) => {
      const tabId = `notebook-output-${outputId}-${index}-tab`
      const panelId = `notebook-output-${outputId}-${index}-panel`
      const disabled = group.label === notebookSuccessOutputLabel
      const active = activeId === group.id && !disabled
      const selected = active ? 'true' : 'false'
      const expanded = active ? 'true' : 'false'
      const tabindex = active ? '0' : '-1'
      return `<button type="button" class="notebook-output-tab" data-notebook-output-tab="${escapeHTML(group.id)}" id="${escapeHTML(
        tabId,
      )}" role="tab" aria-controls="${escapeHTML(panelId)}"${disabled ? ' disabled data-notebook-output-disabled aria-disabled="true"' : ''} aria-selected="${selected}" aria-expanded="${expanded}" tabindex="${tabindex}">${escapeHTML(
        group.label,
      )}</button>`
    })
    .join('')
  const actions = orderedGroups
    .map(group => {
      const active = activeId === group.id && group.label !== notebookSuccessOutputLabel
      const escapedId = escapeHTML(group.id)
      return `<button type="button" class="notebook-output-expand-button notebook-icon-button" aria-label="Expand output" title="Expand output" aria-expanded="false" data-notebook-output-action="${escapedId}" data-notebook-output-expand-action="${escapedId}"${
        active ? '' : ' hidden'
      }><span class="notebook-output-expand-icon">${notebookIconSvg.expand}</span></button><button type="button" class="notebook-output-copy-button notebook-icon-button" aria-label="Copy output" title="Copy output" data-notebook-output-action="${escapedId}"${
        active ? '' : ' hidden'
      }><span class="notebook-output-copy-icon">${notebookIconSvg.copy}</span><span class="notebook-output-check-icon">${notebookIconSvg.check}</span></button>`
    })
    .join('')
  const panels = orderedGroups
    .map((group, index) => {
      const tabId = `notebook-output-${outputId}-${index}-tab`
      const panelId = `notebook-output-${outputId}-${index}-panel`
      const active = activeId === group.id && group.label !== notebookSuccessOutputLabel
      return `<div class="notebook-output-panel-frame" data-notebook-output-panel="${escapeHTML(
        group.id,
      )}" id="${escapeHTML(panelId)}" role="tabpanel" aria-labelledby="${escapeHTML(tabId)}"${
        active ? '' : ' hidden'
      }><div class="notebook-output-panel">${group.outputs.join('')}</div></div>`
    })
    .join('')

  return `<div class="notebook-static-output" data-notebook-static-output="${escapeHTML(
    cellId,
  )}" data-notebook-output-tabbed><div class="notebook-output-tabs" data-notebook-output-tabs${
    collapsed ? ' data-notebook-output-collapsed' : ''
  }><div class="notebook-output-tablist" role="tablist" aria-orientation="horizontal">${tablist}</div><div class="notebook-output-actions">${actions}</div><div class="notebook-output-panels">${panels}</div></div></div>`
}
