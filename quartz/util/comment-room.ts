export const commentRoomEnabledKey = 'garden:comments-room-enabled:v1'
export const commentRoomToggleEvent = 'commentsroomtoggle'

export function readCommentRoomEnabled(storage: Storage = localStorage) {
  return storage.getItem(commentRoomEnabledKey) === 'true'
}

export function writeCommentRoomEnabled(enabled: boolean, storage: Storage = localStorage) {
  storage.setItem(commentRoomEnabledKey, enabled ? 'true' : 'false')
}
