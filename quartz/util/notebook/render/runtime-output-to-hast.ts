import type { ElementContent } from 'hast'
import { toHtml } from 'hast-util-to-html'
import { h } from 'hastscript'
import type {
  NotebookRuntimeDebugOutput,
  NotebookRuntimeOutput,
} from '../../../runtime/notebook/types'
import type { Output } from '../types'
import { renderOutput, renderSuccessMarker } from './output-to-hast'

function streamName(name: string): 'stdout' | 'stderr' {
  return name.trim() === 'stderr' ? 'stderr' : 'stdout'
}

export function runtimeOutputToTypedOutput(output: NotebookRuntimeOutput): Output | undefined {
  if (output.type === 'stream') {
    return { kind: 'stream', name: streamName(output.name), text: output.text }
  }
  if (output.type === 'error') {
    return {
      kind: 'error',
      ename: output.ename,
      evalue: output.evalue,
      traceback: output.traceback ? output.traceback.split('\n') : [],
    }
  }
  if (output.type === 'text') {
    return {
      kind: 'execute_result',
      data: { 'text/plain': output.text },
      metadata: {},
      executionCount: null,
    }
  }
  if (output.type === 'json') {
    return {
      kind: 'execute_result',
      data: { 'application/json': output.text },
      metadata: {},
      executionCount: null,
    }
  }
  if (output.type === 'html') {
    return { kind: 'display_data', data: { 'text/html': output.html }, metadata: {} }
  }
  return undefined
}

function renderDebugMarker(debug: NotebookRuntimeDebugOutput): ElementContent {
  const lines: string[] = []
  const fields: [string, string | undefined][] = [
    ['phase', debug.phase],
    ['cell', debug.cellId],
    ['error', debug.errorName],
    ['message', debug.errorMessage],
    ['stack', debug.stack],
  ]
  for (const [key, value] of fields) {
    if (value !== undefined && value.length > 0) lines.push(`${key}: ${value}`)
  }
  return h(
    'pre',
    { className: ['notebook-output', 'notebook-output-debug'], 'data-output-name': 'debug' },
    h('samp', {}, lines.join('\n')),
  )
}

export function renderRuntimeOutput(
  output: NotebookRuntimeOutput,
  options: { debug?: boolean } = {},
): ElementContent[] {
  if (output.type === 'success') return [renderSuccessMarker()]
  const typed = runtimeOutputToTypedOutput(output)
  const nodes: ElementContent[] = typed ? renderOutput(typed) : []
  if (output.type === 'error' && options.debug === true && output.debug) {
    nodes.push(renderDebugMarker(output.debug))
  }
  return nodes
}

export function renderRuntimeOutputHtml(
  output: NotebookRuntimeOutput,
  options: { debug?: boolean } = {},
): string {
  return renderRuntimeOutput(output, options)
    .map(node => toHtml(node))
    .join('')
}
