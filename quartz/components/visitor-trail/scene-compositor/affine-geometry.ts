export type Matrix = [number, number, number, number, number, number]

export const identity: Matrix = [1, 0, 0, 1, 0, 0]
export const multiply = (m: Matrix, n: Matrix): Matrix => [
  m[0] * n[0] + m[2] * n[1],
  m[1] * n[0] + m[3] * n[1],
  m[0] * n[2] + m[2] * n[3],
  m[1] * n[2] + m[3] * n[3],
  m[0] * n[4] + m[2] * n[5] + m[4],
  m[1] * n[4] + m[3] * n[5] + m[5],
]
export const invert = (m: Matrix): Matrix => {
  const det = m[0] * m[3] - m[1] * m[2] || 1e-9
  return [
    m[3] / det,
    -m[1] / det,
    -m[2] / det,
    m[0] / det,
    (m[2] * m[5] - m[3] * m[4]) / det,
    (m[1] * m[4] - m[0] * m[5]) / det,
  ]
}
export const scaleOf = (m: Matrix) => Math.sqrt(Math.abs(m[0] * m[3] - m[1] * m[2])) || 1
export const scaling = (x: number, y = x): Matrix => [x, 0, 0, y, 0, 0]
export const translation = (x: number, y: number): Matrix => [1, 0, 0, 1, x, y]
export const ctm = (element: SVGGraphicsElement): Matrix => {
  const matrix = element.getCTM()
  return matrix ? [matrix.a, matrix.b, matrix.c, matrix.d, matrix.e, matrix.f] : identity
}
export const place = (matrix: Matrix) =>
  new Float32Array([matrix[0], matrix[1], 0, matrix[2], matrix[3], 0, matrix[4], matrix[5], 1])
export const none = new Float32Array([0, 0, 0, 0, 0, 0, 0, 0, 0])
