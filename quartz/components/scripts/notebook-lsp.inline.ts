let notebookLspModule

function notebookLspScriptUrl(name) {
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

async function mountNotebookLsp() {
  const roots = document.querySelectorAll('[data-notebook-runtime]')
  const data = runtimeDataById()
  const targets = Array.from(roots)
    .map(root => {
      const id = root.dataset.notebookRuntime
      const text = id ? data.get(id) : undefined
      return text && root.dataset.notebookLspMounted !== 'true' ? { root, text } : undefined
    })
    .filter(target => target !== undefined)
  if (targets.length === 0) return
  notebookLspModule ??= import(notebookLspScriptUrl('notebook-lsp.client.js'))
  const lsp = await notebookLspModule
  for (const target of targets) {
    lsp.mountNotebookLsp(target.root, target.text)
  }
}

function scheduleNotebookLspMount() {
  window.setTimeout(() => {
    void mountNotebookLsp().catch(error => {
      console.error('failed to mount notebook lsp', error)
    })
  }, 0)
}

document.addEventListener('nav', scheduleNotebookLspMount)
