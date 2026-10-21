// Injects a pixel verifier into the compositor's retained paths. After each frame painted from
// retained regions or cached segments, it reads the canvas, repaints every draw of the same frame
// into a fresh framebuffer, and records how many pixels differ by more than 3/255. Results
// accumulate in the trail frame's `__trailVerified`.
const wholeAnchor = `    if (whole) {
      gl.clear(gl.COLOR_BUFFER_BIT)
      for (const unit of units)
        if ('segment' in unit) paintSegment(unit.segment)
        else paint(unit.draw)
      return
    }`
const regionAnchor = `    gl.bindBuffer(gl.ARRAY_BUFFER, gpu.vertexBuffer)
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0)
  }`

const verifier = `    const verifyFrame = () => {
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
      for (let index = 0; index < live.length; index += 4) {
        let difference = 0
        for (let channel = 0; channel < 4; channel++)
          difference = Math.max(difference, Math.abs(live[index + channel] - fresh[index + channel]))
        worst = Math.max(worst, difference)
        if (difference > 3) differing++
      }
      ;(record.__trailVerified ??= []).push({ differing, worst, whole, regions: dirty.length })
    }
`

export function injectVerifier(source) {
  const whole = source.indexOf(wholeAnchor)
  const region = source.lastIndexOf(regionAnchor)
  if (whole < 0 || region < whole)
    throw new Error('The compositor retained paths changed; update verify-patch.mjs')
  const wholeChecked = wholeAnchor.replace(
    '      return\n    }',
    '      verifyFrame()\n      return\n    }',
  )
  const regionChecked = regionAnchor.replace(/\n {2}\}$/, '\n    verifyFrame()\n  }')
  return (
    source.slice(0, whole) +
    verifier +
    wholeChecked +
    source.slice(whole + wholeAnchor.length, region) +
    regionChecked +
    source.slice(region + regionAnchor.length)
  )
}
