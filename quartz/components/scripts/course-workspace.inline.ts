import { currentNavSignal } from './nav-lifecycle'

type WorkspaceModule = typeof import('../course-workspace/app')
let workspaceModule: Promise<WorkspaceModule> | undefined

document.addEventListener('nav', () => {
  const root = document.querySelector<HTMLElement>(
    '.course-reading-split > .course-workspace > [data-course-workspace-mount]',
  )
  if (!root) return
  const signal = currentNavSignal()
  let unmount: (() => void) | undefined
  window.addCleanup(() => unmount?.())
  workspaceModule ??= import(new URL('course-workspace.js', import.meta.url).href)
  void workspaceModule
    .then(({ mountCourseWorkspace }) => {
      if (signal.aborted || !root.isConnected) return
      unmount = mountCourseWorkspace(root, signal)
    })
    .catch(error => {
      workspaceModule = undefined
      if (signal.aborted) return
      console.error(error)
      root
        .querySelector('[role="status"]')
        ?.replaceChildren('The workspace could not load. Reload to retry.')
    })
})
