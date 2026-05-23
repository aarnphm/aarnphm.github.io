let notebookRuntimeModule

function notebookRuntimeScriptUrl(name) {
  return new URL(`static/scripts/${name}`, import.meta.url).href
}

function isRecord(value) {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function runtimeIdFromJson(text) {
  try {
    const parsed = JSON.parse(text)
    return isRecord(parsed) && typeof parsed.id === 'string' ? parsed.id : undefined
  } catch {
    return undefined
  }
}

function runtimeDataById() {
  const data = new Map()
  document.querySelectorAll('script[data-notebook-runtime-data]').forEach(script => {
    const text = script.textContent
    if (!text) return
    const id = runtimeIdFromJson(text)
    if (id) data.set(id, text)
  })
  return data
}

async function mountNotebookRuntime() {
  const roots = document.querySelectorAll('[data-notebook-runtime]')
  const data = runtimeDataById()
  const targets = Array.from(roots)
    .map(root => {
      const id = root.dataset.notebookRuntime
      const text = id ? data.get(id) : undefined
      return text && root.dataset.runtimeMounted !== 'true' ? { root, text } : undefined
    })
    .filter(target => target !== undefined)
  if (targets.length === 0) return
  notebookRuntimeModule ??= import(notebookRuntimeScriptUrl('notebook-runtime.client.js'))
  const runtime = await notebookRuntimeModule
  for (const target of targets) {
    runtime.mountNotebookRuntime(target.root, target.text)
  }
}

function scheduleNotebookRuntimeMount() {
  window.setTimeout(() => {
    void mountNotebookRuntime().catch(error => {
      console.error('failed to mount notebook runtime', error)
    })
  }, 0)
}

document.addEventListener('nav', scheduleNotebookRuntimeMount)
