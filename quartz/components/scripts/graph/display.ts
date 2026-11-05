// workaround for pixijs webgpu issue: https://github.com/pixijs/pixijs/issues/11389
export async function determineGraphicsAPI(): Promise<'webgpu' | 'webgl'> {
  const adapter = await navigator.gpu?.requestAdapter().catch(() => null)
  const device = adapter && (await adapter.requestDevice().catch(() => null))
  if (!device) {
    return 'webgl'
  }

  const canvas = document.createElement('canvas')
  const gl =
    (canvas.getContext('webgl2') as WebGL2RenderingContext | null) ??
    (canvas.getContext('webgl') as WebGLRenderingContext | null)

  if (!gl) return 'webgl'

  const webglMaxTextures = gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS)
  const webgpuMaxTextures = device.limits.maxSampledTexturesPerShaderStage

  return webglMaxTextures === webgpuMaxTextures ? 'webgpu' : 'webgl'
}

export function readGraphTheme() {
  const read = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name)
  const colors = {
    secondary: read('--secondary'),
    tertiary: read('--tertiary'),
    gray: read('--gray'),
    lightgray: read('--lightgray'),
    dark: read('--dark'),
    font: read('--bodyFont'),
    foam: read('--foam'),
  }
  let colorParserCtx: CanvasRenderingContext2D | null = null
  const parseColorToHex = (value: string) => {
    if (!colorParserCtx) {
      colorParserCtx = document.createElement('canvas').getContext('2d')
    }
    if (!colorParserCtx) return 0xffffff
    colorParserCtx.fillStyle = value.trim()
    const resolved = colorParserCtx.fillStyle
    if (resolved.startsWith('#')) {
      const hex = resolved.slice(1)
      return parseInt(hex.length === 3 ? hex.replace(/(.)/g, '$1$1') : hex, 16)
    }
    const parts = resolved.match(/\d+/g)
    if (!parts) return 0xffffff
    const [r, g, b] = parts.map(part => parseInt(part, 10))
    return (r << 16) + (g << 8) + b
  }
  const labelTintBase = parseColorToHex(colors.dark)
  const labelTintDimmed = parseColorToHex(colors.lightgray)

  return { colors, labelTintBase, labelTintDimmed }
}
