const hostName = 'xyz.aarnphm.audiooutput'
const clients = new Set()
let nativePort
let reconnectTimer
let configuration = { enabled: false, sites: [] }
let bridgeError = 'Connecting to Audio Output'

function broadcast() {
  for (const client of clients) {
    client.port.postMessage({
      type: 'configuration',
      enabled: configuration.enabled,
      rule: configuration.sites.find(site => site.id === client.host),
      error: bridgeError,
    })
  }
}

function connect() {
  if (nativePort) return
  clearTimeout(reconnectTimer)
  const port = chrome.runtime.connectNative(hostName)
  nativePort = port
  port.onMessage.addListener(message => {
    if (message.configuration) {
      configuration = message.configuration
      bridgeError = ''
      broadcast()
    } else if (message.error) {
      bridgeError = message.error
    }
  })
  port.onDisconnect.addListener(() => {
    bridgeError = chrome.runtime.lastError?.message ?? 'Desktop bridge disconnected'
    nativePort = undefined
    configuration.enabled = false
    broadcast()
    reconnectTimer = setTimeout(connect, 3000)
  })
  port.postMessage({ type: 'snapshot' })
}

chrome.runtime.onConnect.addListener(port => {
  if (port.name !== 'audiooutput-site' || !port.sender?.url) return
  const url = new URL(port.sender.url)
  if (url.protocol !== 'https:') return
  const client = { port, host: url.hostname }
  clients.add(client)
  port.onDisconnect.addListener(() => clients.delete(client))
  port.onMessage.addListener(message => {
    if (message.type === 'status' && configuration.sites.some(site => site.id === client.host)) {
      nativePort?.postMessage({
        type: 'status',
        host: client.host,
        state: message.state,
        detail: String(message.detail).slice(0, 500),
        outputUID: message.outputUID,
      })
    }
  })
  connect()
  broadcast()
})

chrome.runtime.onMessage.addListener((message, sender, reply) => {
  if (sender.id !== chrome.runtime.id) return
  if (message.type === 'bridge-status') {
    reply({ error: bridgeError, configuration })
  } else if (message.type === 'subscribe' && !sender.tab) {
    nativePort?.postMessage({ type: 'subscribe', host: message.host })
    reply({ connected: Boolean(nativePort) })
  }
})

chrome.runtime.onInstalled.addListener(async () => {
  const tabs = await chrome.tabs.query({ url: 'https://music.youtube.com/*' })
  for (const tab of tabs) {
    if (tab.id === undefined) continue
    try {
      await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: ['content.js'] })
    } catch (error) {
      console.error('Audio Output could not attach to the open player:', error.message)
    }
  }
})

setInterval(() => nativePort?.postMessage({ type: 'snapshot' }), 2000)
connect()
