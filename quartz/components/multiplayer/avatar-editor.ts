import {
  BLOB_TRAITS,
  blobSvg,
  blobTraits,
  decodeBlob,
  encodeBlob,
  type BlobTrait,
  type BlobTraits,
} from '../../util/blob-avatar'
import { getAuthor, ownAvatarKey, setOwnAvatar } from './identity'

const labels: Record<BlobTrait, string> = {
  shape: 'shape',
  fill: 'colour',
  eyes: 'eyes',
  mouth: 'mouth',
  hat: 'hat',
  gaze: 'gaze',
}

const toast = (message: string) =>
  document.dispatchEvent(new CustomEvent('toast', { detail: { message } }))

// The blob is generated from trusted code, never from input.
const setSvg = (el: Element, svg: string) => {
  const template = document.createElement('template')
  template.innerHTML = svg
  el.replaceChildren(...template.content.childNodes)
}

// The avatar panel from the command palette: the blob the commenter's name picks, with every trait
// open to change. Saving stores the traits that differ from the name's own, so a blank code means
// "the one my name picks".
export function openAvatarEditor() {
  document.getElementById('avatar-editor')?.remove()
  const author = getAuthor()
  const base = blobTraits(author)
  let traits: BlobTraits = { ...base, ...decodeBlob(localStorage.getItem(ownAvatarKey) ?? '') }

  const dialog = document.createElement('dialog')
  dialog.id = 'avatar-editor'
  dialog.setAttribute('aria-labelledby', 'avatar-editor-title')

  const head = document.createElement('header')
  const title = document.createElement('h2')
  title.id = 'avatar-editor-title'
  title.textContent = 'your blob'
  const close = document.createElement('button')
  close.type = 'button'
  close.className = 'avatar-editor-close'
  close.textContent = 'close'
  close.addEventListener('click', () => dialog.close())
  head.append(title, close)

  const preview = document.createElement('div')
  preview.className = 'avatar-editor-preview'
  const name = document.createElement('p')
  name.className = 'avatar-editor-name'
  name.textContent = author

  const rows = document.createElement('div')
  rows.className = 'avatar-editor-rows'
  const options: [BlobTrait, number, HTMLButtonElement][] = []
  for (const trait of Object.keys(BLOB_TRAITS) as BlobTrait[]) {
    const group = document.createElement('fieldset')
    const legend = document.createElement('legend')
    legend.textContent = labels[trait]
    group.append(legend)
    BLOB_TRAITS[trait].forEach((value, i) => {
      const option = document.createElement('button')
      option.type = 'button'
      option.className = 'avatar-editor-option'
      option.setAttribute('aria-label', trait === 'fill' ? `colour ${i + 1}` : value)
      option.title = trait === 'fill' ? `colour ${i + 1}` : value
      option.addEventListener('click', () => {
        traits = { ...traits, [trait]: i }
        draw()
      })
      options.push([trait, i, option])
      group.append(option)
    })
    rows.append(group)
  }

  const actions = document.createElement('div')
  actions.className = 'avatar-editor-actions'
  const button = (label: string, onClick: () => void) => {
    const b = document.createElement('button')
    b.type = 'button'
    b.textContent = label
    b.addEventListener('click', onClick)
    actions.append(b)
    return b
  }
  button('shuffle', () => {
    traits = blobTraits(`${author}:${Math.random()}`)
    draw()
  })
  button('my name’s blob', () => {
    traits = { ...base }
    draw()
  })
  const save = button('save', async () => {
    const pinned = Object.fromEntries(
      (Object.keys(traits) as BlobTrait[])
        .filter(t => traits[t] !== base[t])
        .map(t => [t, traits[t]]),
    ) as Partial<BlobTraits>
    const code = Object.keys(pinned).length ? encodeBlob({ ...base, ...pinned }) : ''
    save.disabled = true
    const ok = await setOwnAvatar(code)
    save.disabled = false
    toast(ok ? 'blob saved' : 'could not save the blob, try again')
    if (ok) dialog.close()
  })
  save.className = 'avatar-editor-save'

  // Each option previews the blob as it would be with that one trait changed.
  const draw = () => {
    setSvg(preview, blobSvg(author, traits))
    for (const [trait, i, option] of options) {
      setSvg(option, blobSvg(author, { ...traits, [trait]: i }))
      option.setAttribute('aria-pressed', String(traits[trait] === i))
    }
  }

  dialog.append(head, preview, name, rows, actions)
  dialog.addEventListener('close', () => dialog.remove())
  document.body.append(dialog)
  draw()
  dialog.showModal()
}
