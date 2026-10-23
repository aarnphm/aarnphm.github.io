import { bindClipboardButton } from './clipboard-button'

function codeBlockClipboardSource(codeBlock: HTMLElement): string {
  return (codeBlock.dataset.clipboard ?? codeBlock.innerText).replace(/\n\n/g, '\n')
}

document.addEventListener('nav', () => {
  for (const pre of document.getElementsByTagName('pre')) {
    const codeBlock = pre.querySelector('code')
    const button = pre.querySelector<HTMLButtonElement>('button.clipboard-button')
    if (codeBlock && button) {
      window.addCleanup(bindClipboardButton(button, () => codeBlockClipboardSource(codeBlock)))
    }
  }
})
