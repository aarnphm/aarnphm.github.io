import type { CanvasScene } from './scene'

export function buildControls(s: CanvasScene) {
  s.container.replaceChildren()

  s.width = s.container.clientWidth || 800

  s.height = s.container.clientHeight || 600

  s.toolbar = null

  s.helpModal = null

  if (!s.isEmbed) {
    s.toolbar = document.createElement('div')
    s.toolbar.className = 'canvas-controls'
    s.toolbar.innerHTML = `
        <div class="canvas-control-group">
          <button type="button" class="canvas-control-item" data-action="zoom-in" aria-label="Zoom in" title="Zoom in">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14"></path><path d="M12 5v14"></path></svg>
          </button>
          <button type="button" class="canvas-control-item" data-action="zoom-reset" aria-label="Reset zoom to 100%" title="Reset zoom to 100%">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8"></path><path d="M21 3v5h-5"></path></svg>
          </button>
          <button type="button" class="canvas-control-item" data-action="zoom-fit" aria-label="Zoom to fit" title="Zoom to fit">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 3H5a2 2 0 0 0-2 2v3"></path><path d="M21 8V5a2 2 0 0 0-2-2h-3"></path><path d="M3 16v3a2 2 0 0 0 2 2h3"></path><path d="M16 21h3a2 2 0 0 0 2-2v-3"></path></svg>
          </button>
          <button type="button" class="canvas-control-item" data-action="zoom-out" aria-label="Zoom out" title="Zoom out">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14"></path></svg>
          </button>
        </div>
        <div class="canvas-control-group">
          <button type="button" class="canvas-control-item" data-action="help" aria-label="Canvas help" title="Canvas help">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path><path d="M12 17h.01"></path></svg>
          </button>
        </div>
      `
    if (s.isLive) {
      s.toolbar.querySelector('[data-action="help"]')?.closest('.canvas-control-group')?.remove()
      const fullscreen = document.createElement('button')
      fullscreen.type = 'button'
      fullscreen.className = 'canvas-control-item'
      fullscreen.dataset.action = 'fullscreen'
      fullscreen.setAttribute('aria-label', 'Plein écran (f)')
      fullscreen.title = 'Plein écran (f)'
      fullscreen.innerHTML =
        '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 3h6v6"></path><path d="m21 3-7 7"></path><path d="M9 21H3v-6"></path><path d="m3 21 7-7"></path></svg>'
      s.toolbar.querySelector('.canvas-control-group')?.append(fullscreen)
    }
    s.container.appendChild(s.toolbar)

    s.helpModal = document.createElement('dialog')
    s.helpModal.className = 'canvas-help-modal'
    s.helpModal.setAttribute('aria-label', 'Canvas help')
    s.helpModal.innerHTML = `
        <div class="canvas-help-backdrop"></div>
        <div class="canvas-help-content">
          <div class="canvas-help-header">
            <h2>Canvas help</h2>
            <button type="button" class="canvas-help-close" aria-label="Close">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6 6 18"></path><path d="m6 6 12 12"></path></svg>
            </button>
          </div>
          <div class="canvas-help-body">
            <p>Drag cards to arrange this view. Reload restores the saved layout.</p>
            <div class="canvas-help-section">
              <h3>pan</h3>
              <div class="canvas-help-row">
                <span>Pan in either direction</span>
                <span class="canvas-help-keys"><kbd>Scroll</kbd> or drag background</span>
              </div>
              <div class="canvas-help-row">
                <span>Move a group</span>
                <span class="canvas-help-keys">Drag group title</span>
              </div>
              <div class="canvas-help-row">
                <span>Pan over a card</span>
                <span class="canvas-help-keys"><kbd>Space</kbd> + drag, or middle drag</span>
              </div>
              <div class="canvas-help-row">
                <span>Pan horizontally</span>
                <span class="canvas-help-keys"><kbd>Shift</kbd> <kbd>Scroll</kbd></span>
              </div>
            </div>
            <div class="canvas-help-section">
              <h3>zoom</h3>
              <div class="canvas-help-row">
                <span>Zoom</span>
                <span class="canvas-help-keys"><kbd>⌘/Ctrl</kbd> <kbd>Scroll</kbd></span>
              </div>
              <div class="canvas-help-row">
                <span>Zoom to fit</span>
                <span class="canvas-help-keys"><kbd>Shift</kbd> <kbd>1</kbd> or <kbd>0</kbd></span>
              </div>
              <div class="canvas-help-row">
                <span>Reset zoom to 100%</span>
                <span class="canvas-help-keys"><kbd>Shift</kbd> <kbd>0</kbd></span>
              </div>
              <div class="canvas-help-row">
                <span>Zoom to group</span>
                <span class="canvas-help-keys"><kbd>Double-click</kbd> group</span>
              </div>
            </div>
            <div class="canvas-help-section">
              <h3>navigation</h3>
              <div class="canvas-help-row">
                <span>Read card</span>
                <span class="canvas-help-keys"><kbd>Click</kbd> or <kbd>Enter</kbd></span>
              </div>
              <div class="canvas-help-row">
                <span>Open note</span>
                <span class="canvas-help-keys"><kbd>Click</kbd> card title</span>
              </div>
              <div class="canvas-help-row">
                <span>Open from content</span>
                <span class="canvas-help-keys"><kbd>⌘/Ctrl</kbd> <kbd>Click</kbd></span>
              </div>
              <div class="canvas-help-row">
                <span>Open in side panel</span>
                <span class="canvas-help-keys"><kbd>Alt</kbd> <kbd>Click</kbd></span>
              </div>
              <div class="canvas-help-row">
                <span>Scroll node content</span>
                <span class="canvas-help-keys">Focus node, then <kbd>Scroll</kbd></span>
              </div>
              <div class="canvas-help-row">
                <span>Defocus node</span>
                <span class="canvas-help-keys"><kbd>Esc</kbd></span>
              </div>
            </div>
          </div>
        </div>
      `
    s.container.appendChild(s.helpModal)
  }
}
