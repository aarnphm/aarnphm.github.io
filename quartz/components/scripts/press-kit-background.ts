/** Decode once, freeze animated sources, and retain at most the card's 4K resolution. */
export async function readCardBackground(file: File): Promise<HTMLImageElement> {
  if (!['image/png', 'image/jpeg', 'image/webp', 'image/avif', 'image/gif'].includes(file.type))
    throw new Error('choisir une image PNG, JPEG, WebP, AVIF ou GIF')
  if (file.size > 20 * 1024 * 1024) throw new Error('image limitée à 20 Mo')
  const bitmap = await createImageBitmap(file)
  const canvas = document.createElement('canvas')
  try {
    if (bitmap.width * bitmap.height > 40_000_000) throw new Error('image limitée à 40 mégapixels')
    const scale = Math.min(1, 3840 / Math.max(bitmap.width, bitmap.height))
    canvas.width = Math.max(1, Math.round(bitmap.width * scale))
    canvas.height = Math.max(1, Math.round(bitmap.height * scale))
    const ctx = canvas.getContext('2d')
    if (!ctx) throw new Error('Canvas is unavailable')
    ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height)
  } finally {
    bitmap.close()
  }
  const blob = await new Promise<Blob | null>(resolve => canvas.toBlob(resolve, 'image/png'))
  canvas.width = canvas.height = 1
  if (!blob) throw new Error('image illisible')
  const url = URL.createObjectURL(blob)
  try {
    const image = new Image()
    image.src = url
    await image.decode()
    return image
  } finally {
    URL.revokeObjectURL(url)
  }
}
