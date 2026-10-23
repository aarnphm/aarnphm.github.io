import type { MarkdownEditor } from '../../scripts/markdown-editor'
import type { MultiplayerComment } from '../model'
import type { UiContext } from './context'
import { renderMarkdown } from '../../../util/markdown-renderer'
import { getFullSlug } from '../../../util/path'
import { getAvatarUrl } from '../identity'
import { createMarkdownEditor, formatRelativeTime } from './dom'
import { showActionsPopover, showDeleteConfirmation } from './popovers'
import { submitUpdateComment } from './submit'

export const enterEditMode = async (
  ctx: UiContext,
  comment: MultiplayerComment,
  textElement: HTMLElement,
) => {
  const wrapper = document.createElement('div')
  wrapper.className = 'comment-edit-wrapper'

  const inputContent = document.createElement('div')
  inputContent.className = 'edit-input-content'

  const editorMount = document.createElement('div')
  editorMount.className = 'edit-input'

  let markdownEditor: MarkdownEditor | null = null

  const exitEditMode = () => {
    ctx.editCleanups.delete(exitEditMode)
    textElement.style.display = ''
    if (markdownEditor) {
      markdownEditor.destroy()
      markdownEditor = null
    }
    if (wrapper.parentNode) {
      wrapper.parentNode.removeChild(wrapper)
    }
  }

  const actions = document.createElement('div')
  actions.className = 'edit-actions'

  const cancelButton = document.createElement('button')
  cancelButton.innerHTML = `<span class="button-container"><span class="button-text"><span class="button-content">Cancel</span></span></span>`
  cancelButton.className = 'edit-button edit-button-cancel'
  cancelButton.onclick = () => {
    exitEditMode()
  }

  const saveButton = document.createElement('button')
  saveButton.innerHTML = `<span class="button-container"><span class="button-text"><span class="button-content">Save</span></span></span>`
  saveButton.className = 'edit-button edit-button-save'
  saveButton.onclick = async () => {
    if (!markdownEditor) return
    const newContent = markdownEditor.getValue().trim()
    if (!newContent || newContent === comment.content) {
      exitEditMode()
      return
    }

    const updatedAt = Date.now()
    submitUpdateComment(ctx, { ...comment, content: newContent, updatedAt })

    exitEditMode()
  }

  inputContent.appendChild(editorMount)
  actions.appendChild(cancelButton)
  actions.appendChild(saveButton)
  wrapper.appendChild(inputContent)
  wrapper.appendChild(actions)

  textElement.style.display = 'none'
  textElement.parentNode?.insertBefore(wrapper, textElement)
  ctx.editCleanups.add(exitEditMode)

  markdownEditor = await createMarkdownEditor({
    parent: editorMount,
    ariaLabel: 'Edit comment',
    initialContent: comment.content,
    onSubmit: () => saveButton.click(),
    onCancel: exitEditMode,
  })

  if (wrapper.isConnected) {
    markdownEditor.focus()
  } else {
    markdownEditor.destroy()
    markdownEditor = null
  }
}

export const avatarImg = (author: string) => {
  const avatar = document.createElement('img')
  avatar.className = 'reply-avatar'
  avatar.dataset.ignorePopup = 'true'
  avatar.alt = author
  getAvatarUrl(author).then(url => {
    avatar.src = url
  })
  return avatar
}

export const timeElement = (timestamp: number) => {
  const time = document.createElement('time')
  time.className = 'reply-time'
  time.dateTime = new Date(timestamp).toISOString()
  time.title = new Date(timestamp).toLocaleString()
  time.textContent = formatRelativeTime(timestamp)
  return time
}

// The thread's people as overlapping blobs, in the order they joined.
export const peopleStack = (authors: string[], limit = 4) => {
  const people = document.createElement('span')
  people.className = 'thread-people'
  const unique = [...new Set(authors)]
  people.title = unique.join(', ')
  for (const author of unique.slice(0, limit)) people.appendChild(avatarImg(author))
  if (unique.length > limit) {
    const more = document.createElement('span')
    more.className = 'thread-people-more'
    more.textContent = `+${unique.length - limit}`
    people.appendChild(more)
  }
  return people
}

// Avatar column, then name and time once per run: a message that follows the same author
// within ten minutes drops its head and sits under the one before.
export const sameRun = (prev: MultiplayerComment | undefined, next: MultiplayerComment) =>
  prev !== undefined &&
  prev.author === next.author &&
  next.createdAt - prev.createdAt < 10 * 60 * 1000

export const buildThreadItem = (
  ctx: UiContext,
  comment: MultiplayerComment,
  continued: boolean,
) => {
  const item = document.createElement('div')
  item.className = continued ? 'reply-item reply-continued' : 'reply-item'

  const body = document.createElement('div')
  body.className = 'reply-body'

  if (!continued) {
    item.appendChild(avatarImg(comment.author))
    const head = document.createElement('div')
    head.className = 'reply-head'
    const author = document.createElement('span')
    author.className = 'reply-author'
    author.textContent = comment.author
    head.appendChild(author)
    head.appendChild(timeElement(comment.createdAt))
    body.appendChild(head)
  }

  const text = document.createElement('div')
  text.className = 'reply-text markdown-content'
  const currentSlug = getFullSlug(window)
  text.innerHTML = renderMarkdown(comment.content, currentSlug)
  body.appendChild(text)

  const actions = document.createElement('button')
  actions.className = 'reply-actions'
  actions.setAttribute('aria-label', `Actions for ${comment.author}'s comment`)
  actions.setAttribute('aria-expanded', 'false')
  actions.innerHTML = `<svg width="24" height="24" fill="none" viewBox="0 0 24 24"><path fill="currentColor" fill-rule="evenodd" d="M7.5 12a1.5 1.5 0 1 1-3 0 1.5 1.5 0 0 1 3 0m6 0a1.5 1.5 0 1 1-3 0 1.5 1.5 0 0 1 3 0m4.5 1.5a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3" clip-rule="evenodd"></path></svg>`

  actions.onclick = (e: MouseEvent) => {
    e.stopPropagation()
    showActionsPopover(
      ctx,
      actions,
      () => {
        void enterEditMode(ctx, comment, text).catch(error => {
          console.error('failed to enter comment edit mode', error)
        })
      },
      () => showDeleteConfirmation(ctx, comment),
      comment.id,
      comment.parentId !== null,
    )
  }

  item.appendChild(body)
  item.appendChild(actions)
  return item
}

export const renderThreadContent = (
  ctx: UiContext,
  content: HTMLElement,
  comment: MultiplayerComment,
  replies: MultiplayerComment[],
) => {
  const items = [comment, ...replies]
  const previous = ctx.renderedThreads.get(content)
  if (previous?.length === items.length && items.every((item, i) => item === previous[i])) {
    return
  }
  for (const cleanup of ctx.editCleanups) cleanup()
  ctx.renderedThreads.set(content, items)
  content.replaceChildren()
  items.forEach((item, i) =>
    content.appendChild(buildThreadItem(ctx, item, sameRun(items[i - 1], item))),
  )
  content.parentElement
    ?.querySelector('.thread-people')
    ?.replaceWith(peopleStack(items.map(c => c.author)))
}
