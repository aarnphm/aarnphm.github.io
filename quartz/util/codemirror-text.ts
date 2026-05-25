import type { ChangeSet, Text } from '@codemirror/state'

export function codemirrorTextIsBlank(doc: Text): boolean {
  const cursor = doc.iter()
  while (!cursor.done) {
    if (/\S/.test(cursor.value)) return false
    cursor.next()
  }
  return true
}

export function codemirrorChangedTextIsBlank(
  previousBlank: boolean,
  doc: Text,
  changes: ChangeSet,
): boolean {
  if (changes.empty) return previousBlank
  if (previousBlank) {
    let blank = true
    changes.iterChanges((_fromA, _toA, _fromB, _toB, inserted) => {
      if (!codemirrorTextIsBlank(inserted)) blank = false
    })
    return blank
  }
  let deleted = false
  let insertedText = false
  changes.iterChanges((fromA, toA, _fromB, _toB, inserted) => {
    deleted ||= fromA !== toA
    insertedText ||= !codemirrorTextIsBlank(inserted)
  })
  if (insertedText || !deleted) return false
  return codemirrorTextIsBlank(doc)
}
