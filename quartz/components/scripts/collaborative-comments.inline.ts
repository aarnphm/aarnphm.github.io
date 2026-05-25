import {
  commentRoomToggleEvent,
  readCommentRoomEnabled,
  writeCommentRoomEnabled,
} from '../../util/comment-room'

let collaborativeCommentsModule:
  | Promise<{ mountCollaborativeComments: () => () => void }>
  | undefined
let collaborativeCommentsCleanup: (() => void) | undefined

function scriptAssetUrl(name: string) {
  return new URL(name, import.meta.url).href
}

async function mountCollaborativeComments() {
  if (collaborativeCommentsCleanup) return
  collaborativeCommentsModule ??= import(scriptAssetUrl('collaborative-comments.client.js'))
  const comments = await collaborativeCommentsModule
  if (collaborativeCommentsCleanup || !readCommentRoomEnabled()) return
  const cleanup = comments.mountCollaborativeComments()
  collaborativeCommentsCleanup = () => {
    collaborativeCommentsCleanup = undefined
    cleanup()
  }
  window.addCleanup(collaborativeCommentsCleanup)
}

function unmountCollaborativeComments() {
  collaborativeCommentsCleanup?.()
}

document.addEventListener('nav', () => {
  if (!readCommentRoomEnabled()) return
  void mountCollaborativeComments().catch(error => {
    console.error('failed to mount collaborative comments', error)
  })
})

document.addEventListener(commentRoomToggleEvent, (event: CustomEventMap['commentsroomtoggle']) => {
  const enabled = event.detail.enabled ?? !readCommentRoomEnabled()
  writeCommentRoomEnabled(enabled)
  if (!enabled) {
    unmountCollaborativeComments()
    return
  }
  void mountCollaborativeComments().catch(error => {
    console.error('failed to mount collaborative comments', error)
  })
})
