let notebookRuntimeModule: Promise<typeof import('./notebook-runtime.client')> | undefined

function notebookRuntimeScriptUrl(name: string): string {
  return new URL(`static/scripts/${name}`, import.meta.url).href
}

async function mountNotebookRuntime() {
  const root = document.querySelector<HTMLElement>('[data-notebook-runtime]')
  const data = document.querySelector<HTMLScriptElement>('script[data-notebook-runtime-data]')
  if (!root || !data?.textContent || root.dataset.runtimeMounted === 'true') return
  notebookRuntimeModule ??= import(notebookRuntimeScriptUrl('notebook-runtime.client.js'))
  const runtime = await notebookRuntimeModule
  runtime.mountNotebookRuntime(root, data.textContent)
}

function scheduleNotebookRuntimeMount() {
  window.setTimeout(() => {
    void mountNotebookRuntime().catch(error => {
      console.error('failed to mount notebook runtime', error)
    })
  }, 0)
}

document.addEventListener('nav', scheduleNotebookRuntimeMount)
