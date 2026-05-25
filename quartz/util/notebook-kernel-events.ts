export const notebookKernelRequestEvent = 'notebookkernelrequest'
export const notebookKernelCommandEvent = 'notebookkernelcommand'

export type NotebookKernelStatus = 'available' | 'warming' | 'ready' | 'running'
export type NotebookKernelCommand = 'kill' | 'restart' | 'interrupt'

export type NotebookKernelSnapshot = {
  readonly runtimeId: string
  readonly sourcePath: string
  readonly language: string
  readonly status: NotebookKernelStatus
  readonly runningCellId?: string
}

export type NotebookKernelRequestDetail = {
  readonly respond: (snapshot: NotebookKernelSnapshot) => void
}

export type NotebookKernelCommandDetail = {
  readonly runtimeId: string
  readonly language: string
  readonly command: NotebookKernelCommand
}
