import { IDENTITY_GRADE } from '../../../util/scene-grade'

const vertexShader = `#version 300 es
in vec2 corner;
// The drawing target in canvas pixels: the canvas itself, or a cached segment within it.
uniform vec4 target;
uniform mat3 place;
flat out mat3 unplace;
void main() {
  vec2 point = (place * vec3(corner, 1.0)).xy;
  unplace = inverse(place);
  vec2 local = (point - target.xy) / target.zw;
  gl_Position = vec4(local.x * 2.0 - 1.0, 1.0 - local.y * 2.0, 0.0, 1.0);
}`
const fragmentShader = `#version 300 es
precision highp float;
uniform sampler2D image;
uniform sampler2D maskImage;
uniform sampler2D clipImage;
uniform float alpha;
uniform vec2 viewport;
uniform vec3 reveal;
uniform vec3 sweep;
uniform vec4 themeReveal;
uniform sampler2D loadCoverage;
uniform float loadActive;
uniform mat3 gradeM;
uniform vec3 gradeB;
uniform vec4 target;
uniform mat3 maskPlace;
flat in mat3 unplace;
out vec4 color;
vec2 clipUv;
float trailAlpha(float progress, float cellSize) {
  if (progress <= 0.0) return 0.0;
  if (progress >= 1.0) return 1.0;
  vec2 point = (floor(clipUv * viewport / cellSize) + 0.5) * cellSize;
  return step(distance(point, sweep.xy), progress * (sweep.z + cellSize));
}
void main() {
  // Sample from the pixel position, not interpolated coordinates, so a quad and a region
  // drawn over it pick the same nearest texels.
  vec2 point = vec2(target.x + gl_FragCoord.x, target.y + target.w - gl_FragCoord.y);
  vec2 uv = (unplace * vec3(point, 1.0)).xy;
  // Region draws cover a rect around the quad; only the quad itself paints.
  if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) discard;
  vec2 maskUv = (maskPlace * vec3(point, 1.0)).xy;
  clipUv = point / viewport;
  // Masks hold premultiplied luminance, so this is luminance times alpha, as in SVG.
  float mask = texture(maskImage, maskUv).r;
  color = texture(image, uv) * alpha * mask * texture(clipImage, clipUv).r;
  // Textures and the blend target hold premultiplied RGB.
  color.rgb = clamp(gradeM * color.rgb + gradeB * color.a, vec3(0.0), vec3(color.a));
  float ready = 1.0;
  if (loadActive > 0.0 && themeReveal.x >= 0.0) {
    vec2 cell = floor(clipUv * viewport / themeReveal.w);
    ready = texture(loadCoverage, (cell + 0.5) / vec2(textureSize(loadCoverage, 0))).r;
  }
  if (themeReveal.x > 0.0) {
    float next = trailAlpha(themeReveal.y, themeReveal.w) * ready;
    color *= themeReveal.x < 1.5 ? next : 1.0 - next;
  } else color *= ready;
  if (reveal.x > 0.0) {
    vec2 cell = floor(clipUv * viewport / reveal.z);
    float noise = fract(dot(cell, vec2(0.754877666, 0.569840296)));
    float threshold = 0.7 * (1.0 - clipUv.y) + 0.3 * noise;
    float next = step(threshold, reveal.y);
    color *= reveal.x < 1.5 ? next : 1.0 - next;
  }
}`

// The WebGL program, its uniforms and the textures it allocates.
export function createGpu(gl: WebGL2RenderingContext) {
  const uniforms: Record<string, WebGLUniformLocation | null> = {}
  // One screen-space coverage texture per chain of clipped ancestors.
  const clipTextures = new Map<string, { texture: WebGLTexture | null; key: string }>()
  const textures = new Set<WebGLTexture>()
  const contents = new WeakMap<WebGLTexture, number>()
  const allocations = new WeakMap<
    WebGLTexture,
    { width: number; height: number; single: boolean }
  >()
  const setup = () => {
    const compile = (type: number, source: string) => {
      const shader = gl.createShader(type)
      if (!shader) throw new Error('WebGL could not create a shader')
      gl.shaderSource(shader, source)
      gl.compileShader(shader)
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS))
        throw new Error(gl.getShaderInfoLog(shader) ?? 'WebGL shader failed')
      return shader
    }
    const next = gl.createProgram()
    const vertex = compile(gl.VERTEX_SHADER, vertexShader)
    const fragment = compile(gl.FRAGMENT_SHADER, fragmentShader)
    gl.attachShader(next, vertex)
    gl.attachShader(next, fragment)
    gl.bindAttribLocation(next, 0, 'corner')
    gl.linkProgram(next)
    gl.deleteShader(vertex)
    gl.deleteShader(fragment)
    if (!gl.getProgramParameter(next, gl.LINK_STATUS))
      throw new Error(gl.getProgramInfoLog(next) ?? 'WebGL program failed')
    gpu.program = next
    gl.useProgram(gpu.program)
    Object.assign(
      uniforms,
      Object.fromEntries(
        [
          'viewport',
          'target',
          'place',
          'maskPlace',
          'alpha',
          'image',
          'maskImage',
          'clipImage',
          'reveal',
          'sweep',
          'themeReveal',
          'loadCoverage',
          'loadActive',
          'gradeM',
          'gradeB',
        ].map(name => [name, gl.getUniformLocation(next, name)]),
      ),
    )
    gl.uniform1i(uniforms.image, 0)
    gl.uniform1i(uniforms.maskImage, 1)
    gl.uniform1i(uniforms.clipImage, 2)
    gl.uniform1i(uniforms.loadCoverage, 3)
    gl.uniformMatrix3fv(uniforms.gradeM, true, IDENTITY_GRADE.m)
    gl.uniform3fv(uniforms.gradeB, IDENTITY_GRADE.b)
    gpu.vertexBuffer = gl.createBuffer()
    gl.bindBuffer(gl.ARRAY_BUFFER, gpu.vertexBuffer)
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), gl.STATIC_DRAW)
    gl.enableVertexAttribArray(0)
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0)
    gl.enable(gl.BLEND)
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA)
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true)
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1)
    gpu.blank = gl.createTexture()
    gl.bindTexture(gl.TEXTURE_2D, gpu.blank)
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA,
      1,
      1,
      0,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      new Uint8Array([255, 255, 255, 255]),
    )
    filter()
    gpu.maxTexture = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number
    clipTextures.clear()
  }
  const filter = () => {
    const sampling = gl.NEAREST
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, sampling)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, sampling)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
  }
  const upload = (
    texture: WebGLTexture | null,
    source: HTMLCanvasElement,
    single = false,
    width = source.width,
    height = source.height,
  ) => {
    const known = texture ? allocations.get(texture) : undefined
    const matching = known?.width === width && known.height === height && known.single === single
    if (texture && !matching) release(texture)
    const target = matching ? texture : gl.createTexture()
    if (!target) return null
    textures.add(target)
    gl.bindTexture(gl.TEXTURE_2D, target)
    const format = single ? gl.RED : gl.RGBA
    if (!matching) {
      gl.texStorage2D(gl.TEXTURE_2D, 1, single ? gl.R8 : gl.RGBA8, width, height)
      allocations.set(target, { width, height, single })
      filter()
    }
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, width, height, format, gl.UNSIGNED_BYTE, source)
    // A cached segment holding this texture's previous pixels must be rebuilt.
    contents.set(target, (contents.get(target) ?? 0) + 1)
    return target
  }
  const release = (texture: WebGLTexture | null) => {
    if (!texture) return
    gl.deleteTexture(texture)
    textures.delete(texture)
    allocations.delete(texture)
  }
  const gpu = {
    program: null as WebGLProgram | null,
    vertexBuffer: null as WebGLBuffer | null,
    blank: null as WebGLTexture | null,
    maxTexture: 4096,
    uniforms,
    clipTextures,
    textures,
    contents,
    setup,
    filter,
    upload,
    release,
  }
  return gpu
}
export type Gpu = ReturnType<typeof createGpu>
