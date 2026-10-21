import type {
  NotebookKernelCommandDetail,
  NotebookKernelRequestDetail,
  NotebookKernelSnapshot,
  NotebookKernelStatus,
} from '../../../util/notebook-kernel-events'
import type { ExecutableLanguageBackend } from '../backend'
import type { Kernel, RuntimeEvent } from '../kernel'
import type { KernelPresence, NotebookRuntimeState, RuntimeCell } from './runtime-state'
import { supportsEagerRuntimePreload } from '../../../util/runtime-preload'
import { backendFor } from '../registry'
import { queryCell } from './cell-navigation'
import {
  clearOutput,
  debugOutput,
  renderKernelOutput,
  renderOutput,
  renderSuccessOutput,
  restoreSavedOutput,
  type RuntimeErrorOutput,
} from './output-render'
import {
  notebookRuntimeKernelLanguages,
  notebookRuntimePreloadLanguages,
  notebookRuntimeWasDisposed,
  scheduleNotebookRuntimePreload,
} from './payload-and-warmup'
import { downloadRuntimeFile, notebookModulesFor, resolveAsset } from './runtime-assets'
import { setRunningControls, setStatus } from './runtime-chrome'
import { sourceForCell } from './source-text'

export type CellRunResult = { failed: boolean }

export function handleKernelRequest(state: NotebookRuntimeState, event: Event) {
  const detail = (event as CustomEvent<NotebookKernelRequestDetail>).detail
  if (!detail || typeof detail.respond !== 'function') return
  for (const snapshot of kernelSnapshots(state)) detail.respond(snapshot)
}

export function handleKernelCommand(state: NotebookRuntimeState, event: Event) {
  const detail = (event as CustomEvent<NotebookKernelCommandDetail>).detail
  if (!detail || detail.runtimeId !== state.payload.id) return
  void applyKernelCommand(state, detail).catch(error => {
    setStatus(state, error instanceof Error ? error.message : 'kernel command failed')
  })
}

export function handleKernelRunAll(state: NotebookRuntimeState) {
  void runAll(state)
}

export function kernelSnapshots(state: NotebookRuntimeState): NotebookKernelSnapshot[] {
  return notebookRuntimeKernelLanguages(state.payload).map(language => {
    const backend = backendFor(language)
    const backendName = backend?.name ?? language
    const presence = kernelPresence(state, backendName)
    const snapshot: NotebookKernelSnapshot = {
      runtimeId: state.payload.id,
      sourcePath: state.payload.sourcePath,
      language: backendName,
      status: presence.status,
    }
    if (snapshot.status === 'running' && state.runningCellId) {
      return { ...snapshot, runningCellId: state.runningCellId, statusDetail: state.runningCellId }
    }
    if (presence.statusDetail !== undefined) {
      return { ...snapshot, statusDetail: presence.statusDetail }
    }
    return snapshot
  })
}

export function kernelPresence(state: NotebookRuntimeState, backendName: string): KernelPresence {
  const kernel = state.kernels.get(backendName)
  const presence = state.kernelPresences.get(backendName)
  if (kernel && state.running && state.runningKernel === kernel) {
    if (presence?.status === 'interrupting') return presence
    return { status: 'running', statusDetail: state.runningCellId }
  }
  if (state.kernelRequests.has(backendName)) return { status: 'warming' }
  if (presence) return presence
  return { status: kernel ? 'ready' : 'available' }
}

export function setKernelPresence(
  state: NotebookRuntimeState,
  backendName: string,
  status: NotebookKernelStatus,
  statusDetail?: string,
) {
  state.kernelPresences.set(
    backendName,
    statusDetail === undefined ? { status } : { status, statusDetail },
  )
}

export async function applyKernelCommand(
  state: NotebookRuntimeState,
  detail: NotebookKernelCommandDetail,
): Promise<void> {
  const backend = backendFor(detail.language)
  if (!backend) return
  if (detail.command === 'interrupt') {
    interruptBackendKernel(state, backend)
    return
  }
  disposeBackendKernel(state, backend)
  if (detail.command === 'kill') {
    setKernelPresence(state, backend.name, 'killed')
    setStatus(state, `killed ${backend.name}`)
    return
  }
  setKernelPresence(state, backend.name, 'warming')
  setStatus(state, `restarting ${backend.name}`)
  try {
    await ensureBackendKernel(state, backend)
    setKernelPresence(state, backend.name, 'ready')
    if (!state.running) setStatus(state, `${backend.name} ready`)
  } catch (error) {
    setKernelPresence(state, backend.name, 'failed')
    setStatus(state, error instanceof Error ? error.message : `${backend.name} restart failed`)
  }
}

export function interruptBackendKernel(
  state: NotebookRuntimeState,
  backend: ExecutableLanguageBackend,
) {
  const kernel = state.kernels.get(backend.name)
  if (kernel && state.running && state.runningKernel === kernel) {
    state.stopped = true
    setKernelPresence(
      state,
      backend.name,
      'interrupting',
      state.runningCellId === undefined ? undefined : state.runningCellId,
    )
    kernel.interrupt()
    setStatus(state, `interrupting ${backend.name}`)
    return
  }
  setStatus(state, `${backend.name} kernel is idle`)
}

export async function runAll(state: NotebookRuntimeState) {
  if (state.running) return
  state.stopped = false
  for (const cell of state.payload.cells) {
    if (state.stopped) break
    const succeeded = await runCell(state, cell)
    if (!succeeded) {
      if (!state.stopped) {
        state.stopped = true
        setStatus(state, `stopped after ${cell.id}`)
      }
      break
    }
  }
}

export async function runCell(state: NotebookRuntimeState, cell: RuntimeCell): Promise<boolean> {
  if (state.running) return false
  state.stopped = false
  const source = sourceForCell(state, cell)
  const executionCount = nextExecutionCount(state, cell.id)
  state.running = true
  state.runningCellId = cell.id
  setStatus(state, `running ${cell.id}`)
  setExecutionLabel(state, cell.id, '*')
  setRunningControls(state, true)
  clearOutput(state, cell.id)
  try {
    const backend = backendFor(cell.language)
    const check = backend?.canExecute(source) ?? {
      ok: false as const,
      reason: `${cell.language} notebook cells are not executable in this runtime.`,
    }
    if (!check.ok) {
      if (backend) setKernelPresence(state, backend.name, 'failed', cell.id)
      renderOutput(state, cell.id, {
        type: 'error',
        ename: 'UnsupportedRuntimeFeature',
        evalue: check.reason,
        traceback: check.reason,
      })
      return false
    }
    const result = await runKernelCell(state, cell, source)
    if (!result.failed) renderSuccessOutput(state, cell.id)
    return !result.failed
  } catch (error) {
    if (state.stopped) return false
    const text = error instanceof Error ? error.message : String(error)
    const output: RuntimeErrorOutput = {
      type: 'error',
      ename: 'RuntimeError',
      evalue: text,
      traceback: text,
    }
    if (state.debug) output.debug = debugOutput('client-runtime', cell.id, error)
    renderOutput(state, cell.id, output)
    return false
  } finally {
    state.running = false
    state.runningCellId = undefined
    setExecutionLabel(state, cell.id, executionCount)
    setRunningControls(state, false)
    setStatus(state, state.stopped ? 'stopped' : 'idle')
  }
}

export function stop(state: NotebookRuntimeState) {
  if (state.running && state.runningKernel) {
    state.stopped = true
    if (state.runningBackendName) {
      setKernelPresence(
        state,
        state.runningBackendName,
        'interrupting',
        state.runningCellId === undefined ? undefined : state.runningCellId,
      )
    }
    state.runningKernel.interrupt()
    setStatus(state, 'interrupting')
    return
  }
  state.stopped = true
  destroyRuntime(state)
  state.running = false
  state.runningCellId = undefined
  for (const language of notebookRuntimeKernelLanguages(state.payload)) {
    const backend = backendFor(language)
    setKernelPresence(state, backend?.name ?? language, 'stopped')
  }
  setRunningControls(state, false)
  setStatus(state, 'stopped')
}

export function reset(state: NotebookRuntimeState) {
  state.stopped = false
  destroyRuntime(state)
  state.kernelPresences.clear()
  state.running = false
  state.runningCellId = undefined
  state.payload.cells.forEach(cell => {
    clearOutput(state, cell.id)
    restoreSavedOutput(state, cell.id)
    setExecutionLabel(state, cell.id, cell.executionIndex)
  })
  state.executionCounter = Math.max(0, ...state.payload.cells.map(cell => cell.executionIndex ?? 0))
  setRunningControls(state, false)
  setStatus(state, 'idle')
  scheduleRuntimePreload(state)
}

export function nextExecutionCount(state: NotebookRuntimeState, cellId: string): number {
  state.executionCounter += 1
  const controls = queryCell<HTMLElement>(state, `[data-notebook-cell="${CSS.escape(cellId)}"]`)
  if (controls) controls.dataset.notebookExecutionCount = String(state.executionCounter)
  return state.executionCounter
}

export function setExecutionLabel(
  state: NotebookRuntimeState,
  cellId: string,
  count: number | '*' | null,
) {
  const label = queryCell<HTMLElement>(
    state,
    `[data-notebook-execution-label="${CSS.escape(cellId)}"]`,
  )
  if (!label) return
  label.textContent = count === null ? 'In [ ]:' : `In [${count}]:`
}

export async function ensureKernel(
  state: NotebookRuntimeState,
  cell: RuntimeCell,
): Promise<{ kernel: Kernel; backend: ExecutableLanguageBackend }> {
  const backend = backendFor(cell.language)
  if (!backend) {
    throw new Error(`${cell.language} notebook cells are not executable here.`)
  }
  return { kernel: await ensureBackendKernel(state, backend), backend }
}

export async function ensureBackendKernel(
  state: NotebookRuntimeState,
  backend: ExecutableLanguageBackend,
): Promise<Kernel> {
  const existing = state.kernels.get(backend.name)
  if (existing) return existing
  const pending = state.kernelRequests.get(backend.name)
  if (pending) return await pending
  const epoch = state.runtimeEpoch
  const kernelEpochBinding = kernelEpoch(state, backend.name)
  const request = createBackendKernel(state, backend, epoch, kernelEpochBinding)
  state.kernelRequests.set(backend.name, request)
  try {
    return await request
  } finally {
    if (state.kernelRequests.get(backend.name) === request)
      state.kernelRequests.delete(backend.name)
  }
}

export async function createBackendKernel(
  state: NotebookRuntimeState,
  backend: ExecutableLanguageBackend,
  epoch: number,
  kernelEpochBinding: number,
): Promise<Kernel> {
  const kernel = await backend.kernelFactory({
    runtimeId: state.payload.id,
    sourcePath: state.payload.sourcePath,
    indexUrl: state.payload.indexUrl || backend.defaultIndexUrl,
    workerUrl: backend.workerAssetKey
      ? state.assets[backend.workerAssetKey]
      : state.assets.workerUrl,
    resolveAsset: (request, runtimeFile) => resolveAsset(request, runtimeFile),
    status: text => {
      if (
        state.disposed ||
        state.runtimeEpoch !== epoch ||
        kernelEpoch(state, backend.name) !== kernelEpochBinding
      ) {
        return
      }
      setKernelPresence(state, backend.name, 'warming', text)
      setStatus(state, text)
    },
  })
  try {
    await kernel.init({
      signal: new AbortController().signal,
      indexUrl: state.payload.indexUrl || backend.defaultIndexUrl,
    })
  } catch (error) {
    await kernel.dispose()
    throw error
  }
  if (
    state.disposed ||
    state.runtimeEpoch !== epoch ||
    kernelEpoch(state, backend.name) !== kernelEpochBinding
  ) {
    await kernel.dispose()
    throw new Error('notebook runtime was disposed')
  }
  state.kernels.set(backend.name, kernel)
  setKernelPresence(state, backend.name, 'ready')
  return kernel
}

export function kernelEpoch(state: NotebookRuntimeState, backendName: string): number {
  return state.kernelEpochs.get(backendName) ?? 0
}

export function invalidateBackendKernel(state: NotebookRuntimeState, backendName: string) {
  state.kernelEpochs.set(backendName, kernelEpoch(state, backendName) + 1)
}

export function disposeBackendKernel(
  state: NotebookRuntimeState,
  backend: ExecutableLanguageBackend,
) {
  invalidateBackendKernel(state, backend.name)
  state.kernelRequests.delete(backend.name)
  const kernel = state.kernels.get(backend.name)
  state.kernels.delete(backend.name)
  if (kernel && state.running && state.runningKernel === kernel) {
    state.stopped = true
    setKernelPresence(
      state,
      backend.name,
      'interrupting',
      state.runningCellId === undefined ? undefined : state.runningCellId,
    )
    setStatus(state, `killing ${backend.name}`)
  }
  void kernel?.dispose()
}

export async function runKernelCell(
  state: NotebookRuntimeState,
  cell: RuntimeCell,
  source: string,
): Promise<CellRunResult> {
  const { kernel, backend } = await ensureKernel(state, cell)
  const modules = await notebookModulesFor(state, source, backend)
  let failed = false
  let threw = false
  let interrupted = false
  state.runningKernel = kernel
  state.runningBackendName = backend.name
  setKernelPresence(state, backend.name, 'running', cell.id)
  try {
    for await (const event of kernel.execute(cell.id, source, { debug: state.debug, modules })) {
      if (event.type === 'interrupted') interrupted = true
      failed = handleRuntimeEvent(state, event) || failed
    }
  } catch (error) {
    threw = true
    setKernelPresence(state, backend.name, 'failed', cell.id)
    throw error
  } finally {
    if (state.runningKernel === kernel) state.runningKernel = undefined
    if (state.runningBackendName === backend.name) state.runningBackendName = undefined
    const presence = state.kernelPresences.get(backend.name)
    if (!threw && presence?.status !== 'killed') {
      setKernelPresence(
        state,
        backend.name,
        interrupted || state.stopped ? 'interrupted' : failed ? 'failed' : 'ready',
        failed || interrupted || state.stopped ? cell.id : undefined,
      )
    }
  }
  return { failed }
}

export function scheduleRuntimePreload(state: NotebookRuntimeState) {
  if (state.runtimePreload) return
  if (!supportsEagerRuntimePreload()) return
  scheduleNotebookRuntimePreload(() => {
    if (state.disposed) return
    if (state.running) {
      state.runtimePreload = undefined
      scheduleRuntimePreload(state)
      return
    }
    state.runtimePreload = preloadRuntimeLanguages(state)
  })
}

export async function preloadRuntimeLanguages(state: NotebookRuntimeState): Promise<void> {
  for (const language of notebookRuntimePreloadLanguages(state.payload)) {
    if (state.disposed || state.running) {
      state.runtimePreload = undefined
      scheduleRuntimePreload(state)
      return
    }
    const backend = backendFor(language)
    if (!backend) continue
    try {
      await ensureBackendKernel(state, backend)
    } catch (error) {
      if (state.disposed || notebookRuntimeWasDisposed(error)) return
      console.warn(`failed to preload ${backend.name} notebook runtime`, error)
    }
  }
}

export function handleRuntimeEvent(state: NotebookRuntimeState, event: RuntimeEvent): boolean {
  if (event.type === 'started') return false
  if (event.type === 'stream') {
    renderOutput(state, event.cellId, { type: 'stream', name: event.name, text: event.text })
    return false
  }
  if (event.type === 'output') {
    if (
      'type' in event.output &&
      event.output.type === 'error' &&
      event.output.ename === 'UnsupportedRuntimeFeature'
    ) {
      restoreSavedOutput(state, event.cellId)
    }
    renderKernelOutput(state, event.cellId, event.output)
    return false
  }
  if (event.type === 'error') {
    renderKernelOutput(state, event.cellId, event.output)
    return true
  }
  if (event.type === 'download') {
    downloadRuntimeFile(state, event.download)
    return false
  }
  if (event.type === 'status') {
    setStatus(state, event.text)
    return false
  }
  if (event.type === 'interrupted') {
    return true
  }
  if (event.type === 'done') {
    return event.failed === true
  }
  return false
}

export function destroyRuntime(state: NotebookRuntimeState) {
  state.runtimeEpoch += 1
  state.runtimePreload = undefined
  const kernels = [...state.kernels.values()]
  state.kernels.clear()
  state.kernelRequests.clear()
  state.runningKernel = undefined
  state.runningBackendName = undefined
  for (const kernel of kernels) void kernel.dispose()
}
