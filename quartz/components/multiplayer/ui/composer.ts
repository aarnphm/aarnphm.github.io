import type { MarkdownEditor } from '../../scripts/markdown-editor'
import type { MultiplayerComment } from '../model'
import type { UiContext } from './context'
import { computeStructuralAnchor, getRangeOffsets, hashText } from '../anchor'
import { getAuthor, getCommentPageId } from '../identity'
import {
  createMarkdownEditor,
  getTextNodeRects,
  keepModalInViewport,
  rangeIntersectsNotebookCellBoundary,
} from './dom'
import { submitNewComment } from './submit'

export const clearSelectionHighlight = (ctx: UiContext) => {
  if (ctx.selectionHighlightLayer) {
    ctx.selectionHighlightLayer.remove()
    ctx.selectionHighlightLayer = null
  }
}

export const hideComposer = (ctx: UiContext) => {
  ctx.composerVersion++
  ctx.activeComposerCleanup?.()
  ctx.activeComposerCleanup = null
  if (ctx.activeComposer) {
    document.body.removeChild(ctx.activeComposer)
    ctx.activeComposer = null
  }
  clearSelectionHighlight(ctx)
}

export const renderSelectionHighlight = (ctx: UiContext, range: Range) => {
  clearSelectionHighlight(ctx)
  const rects = getTextNodeRects(range)
  if (rects.length === 0) return
  const scrollTop = window.pageYOffset || document.documentElement.scrollTop
  const scrollLeft = window.pageXOffset || document.documentElement.scrollLeft
  const layer = document.createElement('div')
  layer.className = 'comment-selection-layer'
  for (const rect of rects) {
    const highlight = document.createElement('span')
    highlight.className = 'comment-selection-highlight'
    highlight.style.left = `${rect.left + scrollLeft}px`
    highlight.style.top = `${rect.top + scrollTop}px`
    highlight.style.width = `${rect.width}px`
    highlight.style.height = `${rect.height}px`
    layer.appendChild(highlight)
  }
  document.body.appendChild(layer)
  ctx.selectionHighlightLayer = layer
}

export const showComposer = async (ctx: UiContext, range: Range) => {
  const version = ++ctx.composerVersion
  ctx.activeComposerCleanup?.()
  ctx.activeComposerCleanup = null
  if (ctx.activeComposer) {
    document.body.removeChild(ctx.activeComposer)
    ctx.activeComposer = null
  }
  const article = document.querySelector('article.popover-hint')
  if (!article) return
  const offsets = getRangeOffsets(range, article)
  if (!offsets) return
  const anchorHash = await hashText(offsets.text)
  if (version !== ctx.composerVersion || !article.isConnected) return
  const structuralAnchor = computeStructuralAnchor(range, article)

  const composer = document.createElement('div')
  composer.className = 'comment-composer'
  ctx.activeComposer = composer

  const rects = Array.from(range.getClientRects()).filter(rect => rect.width && rect.height)
  const rect = rects[rects.length - 1] ?? range.getBoundingClientRect()

  const inputWrapper = document.createElement('div')
  inputWrapper.className = 'composer-input-wrapper composer-empty'

  const inputContainer = document.createElement('div')
  inputContainer.className = 'composer-input'
  inputContainer.setAttribute('role', 'textbox')
  inputContainer.setAttribute('aria-placeholder', 'Commentaire (compatible markdown)')

  const editorMount = document.createElement('div')
  editorMount.className = 'composer-editor-mount'

  const placeholderWrapper = document.createElement('div')
  placeholderWrapper.setAttribute('aria-hidden', 'true')
  const placeholderText = document.createElement('span')
  placeholderText.className = 'placeholder-text'
  placeholderText.textContent = 'Commentaire (compatible markdown)'
  placeholderWrapper.appendChild(placeholderText)

  let editor: MarkdownEditor | null = null
  ctx.activeComposerCleanup = () => {
    editor?.destroy()
    editor = null
  }

  const submitButton = document.createElement('button')
  submitButton.className = 'composer-submit'
  submitButton.setAttribute('aria-label', 'Submit comment')
  submitButton.disabled = true
  submitButton.innerHTML = `<span class="icon"><svg width="24" height="24" fill="none" viewBox="0 0 24 24"><path fill="currentColor" fill-rule="evenodd" d="M12 16a.5.5 0 0 1-.5-.5V8.707l-3.146 3.147a.5.5 0 0 1-.708-.708l4-4a.5.5 0 0 1 .708 0l4 4a.5.5 0 0 1-.708.708L12.5 8.707V15.5a.5.5 0 0 1-.5.5" clip-rule="evenodd"></path></svg></span>`

  submitButton.onclick = async () => {
    if (!editor) return

    const content = editor.getValue().trim()
    if (!content) return

    const comment: MultiplayerComment = {
      id: crypto.randomUUID(),
      pageId: getCommentPageId(),
      parentId: null,
      anchorHash,
      anchorStart: offsets.startOffset,
      anchorEnd: offsets.endOffset,
      anchorText: offsets.text,
      content,
      author: getAuthor(),
      createdAt: Date.now(),
      updatedAt: null,
      deletedAt: null,
      resolvedAt: null,
      anchor: structuralAnchor,
      orphaned: null,
      lastRecoveredAt: null,
    }

    submitNewComment(ctx, comment)

    hideComposer(ctx)
  }

  inputContainer.appendChild(editorMount)
  inputContainer.appendChild(placeholderWrapper)
  inputWrapper.appendChild(inputContainer)
  inputWrapper.appendChild(submitButton)
  composer.appendChild(inputWrapper)
  document.body.appendChild(composer)
  // Centered under the selection's last line, measured once mounted; the viewport clamp wins.
  composer.style.left = `${rect.left + rect.width / 2 - composer.offsetWidth / 2 + window.scrollX}px`
  composer.style.top = `${rect.bottom + window.scrollY + 8}px`
  keepModalInViewport(composer)

  editor = await createMarkdownEditor({
    parent: editorMount,
    ariaLabel: 'Comment',
    onEdited: state => {
      submitButton.disabled = state.empty
      if (state.empty) {
        inputWrapper.classList.add('composer-empty')
      } else {
        inputWrapper.classList.remove('composer-empty')
      }
    },
    onSubmit: () => submitButton.click(),
    onCancel: () => hideComposer(ctx),
  })

  if (composer.isConnected) {
    editor.focus()
  } else {
    editor.destroy()
    editor = null
  }
}

export const handleTextSelection = (ctx: UiContext) => {
  const selection = window.getSelection()
  if (!selection || selection.isCollapsed || selection.rangeCount === 0) {
    ctx.dispatch({ type: 'ui.selection.cleared' })
    return
  }

  const range = selection.getRangeAt(0)
  const article = document.querySelector('article.popover-hint')
  if (!article || !article.contains(range.commonAncestorContainer)) {
    ctx.dispatch({ type: 'ui.selection.cleared' })
    return
  }
  if (rangeIntersectsNotebookCellBoundary(range)) {
    ctx.dispatch({ type: 'ui.selection.cleared' })
    return
  }
  ctx.dispatch({ type: 'ui.selection.changed', range: range.cloneRange() })
}
