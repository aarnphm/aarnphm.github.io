// Injects a pixel verifier at the end of the compositor's region path. After each region-painted
// frame it reads the retained canvas, repaints every draw of the same frame into a fresh
// framebuffer, and records how many pixels differ by more than 3/255. Results accumulate in the
// trail frame's `__trailVerified`.
const anchor = `    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer)
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0)
  }`

const verifier = `    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer)
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0)
    const record = globalThis as unknown as { __trailVerify?: boolean; __trailVerified?: object[] }
    if (!record.__trailVerify) return
    const width = canvas.width
    const height = canvas.height
    const live = new Uint8Array(width * height * 4)
    gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, live)
    const texture = gl.createTexture()
    const framebuffer = gl.createFramebuffer()
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, texture)
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null)
    filter()
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer)
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0)
    gl.clear(gl.COLOR_BUFFER_BIT)
    state.unit = -1
    state.bound.fill(null)
    for (const draw of draws) paint(draw)
    shade(false)
    const fresh = new Uint8Array(width * height * 4)
    gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, fresh)
    gl.bindFramebuffer(gl.FRAMEBUFFER, null)
    gl.deleteFramebuffer(framebuffer)
    gl.deleteTexture(texture)
    let differing = 0
    let worst = 0
    for (let row = 0; row < height; row++)
      for (let x = 0; x < width; x++) {
        const index = (row * width + x) * 4
        let difference = 0
        for (let channel = 0; channel < 4; channel++)
          difference = Math.max(difference, Math.abs(live[index + channel] - fresh[index + channel]))
        worst = Math.max(worst, difference)
        if (difference > 3) differing++
      }
    ;(record.__trailVerified ??= []).push({ differing, worst, whole, regions: dirty.length })
  }`

export function injectVerifier(source) {
  const at = source.lastIndexOf(anchor)
  if (at < 0 || !source.includes('const spans: { unit: Unit; first: number; count: number }[]'))
    throw new Error('The compositor region path changed; update verify-patch.mjs')
  return source.slice(0, at) + verifier + source.slice(at + anchor.length)
}
