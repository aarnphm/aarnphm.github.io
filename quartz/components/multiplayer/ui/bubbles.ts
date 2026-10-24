import type { UiContext } from './context'
import { renderMarkdownInto } from '../../../util/markdown-renderer'
import { getFullSlug } from '../../../util/path'
import { computeStructuralAnchor, getArticleText, recoverFromStructuralAnchor } from '../anchor'
import { isStructuralAnchor } from '../model'
import { correctionOpId, getTextNodeRects } from './dom'
import { showCommentThread } from './modal'
import { avatarImg, peopleStack, timeElement } from './rows'
import { submitUpdateComment } from './submit'

export const renderComments = (ctx: UiContext) => {
  const { comments } = ctx.getState()
  const hiddenIds = comments
    .filter(comment => comment.deletedAt || comment.resolvedAt)
    .map(comment => comment.id)
  if (hiddenIds.length > 0) {
    ctx.dispatch({ type: 'ui.bubbleOffsets.prune', commentIds: hiddenIds })
  }

  document.querySelectorAll('.comment-highlight-layer').forEach(el => el.remove())
  document
    .querySelectorAll('.comment-highlight')
    .forEach(el => el.replaceWith(...Array.from(el.childNodes)))
  document.querySelectorAll('.comment-bubble').forEach(el => el.remove())

  const articleText = getArticleText()
  const article = document.querySelector('article.popover-hint')
  if (!article) return

  const topLevelComments = comments.filter(c => !c.parentId && !c.deletedAt && !c.resolvedAt)
  if (topLevelComments.length === 0) return

  const replyAuthors = new Map<string, string[]>()
  for (const comment of comments) {
    if (comment.parentId && !comment.deletedAt) {
      replyAuthors.set(comment.parentId, [
        ...(replyAuthors.get(comment.parentId) ?? []),
        comment.author,
      ])
    }
  }

  const textNodes: Array<{ node: Text; start: number; end: number }> = []
  const walker = document.createTreeWalker(article, NodeFilter.SHOW_TEXT)
  let offset = 0
  while (walker.nextNode()) {
    const node = walker.currentNode
    if (!(node instanceof Text)) continue
    if (node.length > 0) textNodes.push({ node, start: offset, end: offset + node.length })
    offset += node.length
  }
  const findTextNode = (position: number, isEnd: boolean) => {
    let low = 0
    let high = textNodes.length
    while (low < high) {
      const mid = (low + high) >>> 1
      const end = textNodes[mid].end
      if (isEnd ? end < position : end <= position) low = mid + 1
      else high = mid
    }
    return textNodes[low]
  }

  // Keep overlays detached while measuring ranges to avoid a layout per comment.
  const overlays = document.createDocumentFragment()
  const highlightLayer = document.createElement('div')
  highlightLayer.className = 'comment-highlight-layer'
  overlays.appendChild(highlightLayer)

  for (const comment of topLevelComments) {
    let startIdx = comment.anchorStart
    let endIdx = comment.anchorEnd

    const textAtOffsets = articleText.substring(startIdx, endIdx)
    const offsetsValid =
      startIdx >= 0 &&
      endIdx <= articleText.length &&
      startIdx < endIdx &&
      textAtOffsets === comment.anchorText

    if (!offsetsValid && comment.anchorText) {
      let recovered = false

      if (isStructuralAnchor(comment.anchor)) {
        const structuralResult = recoverFromStructuralAnchor(
          comment.anchor,
          comment.anchorText,
          article,
        )
        if (structuralResult) {
          startIdx = structuralResult.startIdx
          endIdx = structuralResult.endIdx
          recovered = true
        }
      }

      if (!recovered) {
        const matches: number[] = []
        let searchStart = 0
        while (true) {
          const idx = articleText.indexOf(comment.anchorText, searchStart)
          if (idx === -1) break
          matches.push(idx)
          searchStart = idx + 1
        }

        if (matches.length > 0) {
          const closest = matches.reduce((best, curr) =>
            Math.abs(curr - comment.anchorStart) < Math.abs(best - comment.anchorStart)
              ? curr
              : best,
          )
          startIdx = closest
          endIdx = closest + comment.anchorText.length
          recovered = true
        }
      }

      if (recovered && (startIdx !== comment.anchorStart || endIdx !== comment.anchorEnd)) {
        const opId = correctionOpId(comment.id, startIdx, endIdx)
        if (!ctx.getState().correctedAnchors.has(opId)) {
          ctx.dispatch({ type: 'ui.correctedAnchor.add', opId })
          submitUpdateComment(
            ctx,
            {
              ...comment,
              anchorStart: startIdx,
              anchorEnd: endIdx,
              lastRecoveredAt: Date.now(),
              updatedAt: Date.now(),
            },
            opId,
          )
        }
      }
    }

    if (startIdx === endIdx || startIdx < 0 || endIdx > articleText.length) {
      if (!comment.orphaned) {
        const opId = `orphan:${comment.id}`
        if (!ctx.getState().correctedAnchors.has(opId)) {
          ctx.dispatch({ type: 'ui.correctedAnchor.add', opId })
          submitUpdateComment(ctx, { ...comment, orphaned: true, updatedAt: Date.now() }, opId)
        }
      }
      continue
    }

    if (comment.orphaned) {
      const opId = `unorphan:${comment.id}`
      if (!ctx.getState().correctedAnchors.has(opId)) {
        ctx.dispatch({ type: 'ui.correctedAnchor.add', opId })
        submitUpdateComment(ctx, { ...comment, orphaned: false, updatedAt: Date.now() }, opId)
      }
    }

    const start = findTextNode(startIdx, false)
    const end = findTextNode(endIdx, true)

    if (start && end) {
      try {
        const range = document.createRange()
        range.setStart(start.node, startIdx - start.start)
        range.setEnd(end.node, endIdx - end.start)

        if (!comment.anchor) {
          const opId = `backfill-anchor:${comment.id}`
          if (!ctx.getState().correctedAnchors.has(opId)) {
            ctx.dispatch({ type: 'ui.correctedAnchor.add', opId })
            const newAnchor = computeStructuralAnchor(range, article)
            submitUpdateComment(ctx, { ...comment, anchor: newAnchor, updatedAt: Date.now() }, opId)
          }
        }

        const rects = getTextNodeRects(range)
        if (rects.length === 0) continue
        const anchorRect = rects[0]
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop
        const scrollLeft = window.pageXOffset || document.documentElement.scrollLeft

        for (const rect of rects) {
          const highlight = document.createElement('span')
          highlight.className = 'comment-highlight'
          highlight.dataset.commentId = comment.id
          highlight.style.left = `${rect.left + scrollLeft}px`
          highlight.style.top = `${rect.top + scrollTop}px`
          highlight.style.width = `${rect.width}px`
          highlight.style.height = `${rect.height}px`
          highlightLayer.appendChild(highlight)
        }

        const bubble = document.createElement('div')
        bubble.className = 'comment-bubble'
        if (ctx.getState().unreadCommentIds.has(comment.id)) {
          bubble.classList.add('comment-bubble-unread')
        }
        bubble.dataset.commentId = comment.id
        const bubbleSize = window.matchMedia('(pointer: coarse)').matches ? 44 : 32
        const baseLeft =
          Math.min(anchorRect.right + 8, document.documentElement.clientWidth - bubbleSize - 12) +
          scrollLeft
        const baseTop = anchorRect.top + scrollTop
        const offset = ctx.getState().bubbleOffsets.get(comment.id)
        const initialLeft = baseLeft + (offset?.x ?? 0)
        const initialTop = baseTop + (offset?.y ?? 0)
        bubble.style.top = `${initialTop}px`
        bubble.style.left = `${initialLeft}px`

        const preview = document.createElement('div')
        preview.className = 'reply-item'
        preview.appendChild(avatarImg(comment.author))

        const previewBody = document.createElement('div')
        previewBody.className = 'reply-body'
        const head = document.createElement('div')
        head.className = 'reply-head'
        const author = document.createElement('span')
        author.className = 'reply-author'
        author.textContent = comment.author
        head.appendChild(author)
        head.appendChild(timeElement(comment.createdAt))

        const text = document.createElement('div')
        text.className = 'reply-text markdown-content'
        const currentSlug = getFullSlug(window)
        renderMarkdownInto(text, comment.content, currentSlug)

        previewBody.appendChild(head)
        previewBody.appendChild(text)

        const repliers = replyAuthors.get(comment.id) ?? []
        if (repliers.length > 0) {
          const replies = document.createElement('div')
          replies.className = 'preview-replies'
          replies.appendChild(peopleStack(repliers, 3))
          const count = document.createElement('span')
          count.textContent = `${repliers.length} ${repliers.length === 1 ? 'reply' : 'replies'}`
          replies.appendChild(count)
          previewBody.appendChild(replies)
        }
        preview.appendChild(previewBody)
        bubble.appendChild(preview)

        bubble.onmouseenter = () => {
          if (ctx.activeModal && ctx.activeModal.dataset.commentId === comment.id) {
            bubble.classList.add('modal-active')
          }
        }

        bubble.onmouseleave = () => {
          bubble.classList.remove('modal-active')
        }

        bubble.onmousedown = (e: MouseEvent) => {
          if (e.button !== 0) return
          e.preventDefault()
          let isDragging = true
          let dragStartX = e.pageX
          let dragStartY = e.pageY
          let startLeft = parseFloat(bubble.style.left) || initialLeft
          let startTop = parseFloat(bubble.style.top) || initialTop

          const onMouseMove = (moveEvent: MouseEvent) => {
            if (!isDragging) return
            const deltaX = moveEvent.pageX - dragStartX
            const deltaY = moveEvent.pageY - dragStartY
            bubble.style.left = `${startLeft + deltaX}px`
            bubble.style.top = `${startTop + deltaY}px`
          }

          const onMouseUp = () => {
            isDragging = false
            document.removeEventListener('mousemove', onMouseMove)
            document.removeEventListener('mouseup', onMouseUp)
            const currentLeft = parseFloat(bubble.style.left) || startLeft
            const currentTop = parseFloat(bubble.style.top) || startTop
            ctx.dispatch({
              type: 'ui.bubble.offsetUpdated',
              commentId: comment.id,
              offset: { x: currentLeft - baseLeft, y: currentTop - baseTop },
            })
          }

          document.addEventListener('mousemove', onMouseMove)
          document.addEventListener('mouseup', onMouseUp)
        }

        bubble.onclick = () => {
          if (ctx.activeModal) {
            return
          }
          bubble.classList.add('modal-active')
          const durationRaw = getComputedStyle(bubble)
            .getPropertyValue('--expand-animation-time')
            .trim()
          let delay = 120
          if (durationRaw.endsWith('ms')) {
            const parsed = Number.parseFloat(durationRaw)
            if (!Number.isNaN(parsed)) delay = parsed
          } else if (durationRaw.endsWith('s')) {
            const parsed = Number.parseFloat(durationRaw)
            if (!Number.isNaN(parsed)) delay = parsed * 1000
          }
          window.clearTimeout(ctx.threadOpenTimer)
          ctx.threadOpenTimer = window.setTimeout(() => {
            ctx.threadOpenTimer = 0
            if (!bubble.isConnected) return
            const bubbleRect = bubble.getBoundingClientRect()
            const scrollTop = window.pageYOffset || document.documentElement.scrollTop
            const scrollLeft = window.pageXOffset || document.documentElement.scrollLeft
            showCommentThread(ctx, comment.id, {
              top: bubbleRect.top + scrollTop,
              left: bubbleRect.left + scrollLeft,
            })
          }, delay)
        }

        overlays.appendChild(bubble)
      } catch (err) {
        console.warn('failed to highlight comment', err)
      }
    }
  }
  document.body.appendChild(overlays)
}

export const renderAllComments = (ctx: UiContext) => {
  if (ctx.rendering) {
    ctx.renderAgain = true
    return
  }
  // Anchor recovery dispatches synchronous updates; finish this pass before rerendering.
  do {
    ctx.renderAgain = false
    ctx.rendering = true
    try {
      renderComments(ctx)
    } finally {
      ctx.rendering = false
    }
  } while (ctx.renderAgain)
}
