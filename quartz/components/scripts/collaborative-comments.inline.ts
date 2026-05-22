let collaborativeCommentsModule: Promise<{ mountCollaborativeComments: () => void }> | undefined

function scriptAssetUrl(name: string) {
  return new URL(`static/scripts/${name}`, import.meta.url).href
}

async function mountCollaborativeComments() {
  collaborativeCommentsModule ??= import(scriptAssetUrl('collaborative-comments.client.js'))
  const comments = await collaborativeCommentsModule
  comments.mountCollaborativeComments()
}

document.addEventListener('nav', () => {
  void mountCollaborativeComments().catch(error => {
    console.error('failed to mount collaborative comments', error)
  })
})
