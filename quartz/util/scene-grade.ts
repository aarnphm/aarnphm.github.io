import type { Almanac, Sky } from '../components/scripts/404-almanac'
import type { RGB } from '../components/scripts/404-pixel'
import type { Season } from './visitor-world'
import { LIGHT, lighting } from './scene-palette'

export type GradeMatrix = [number, number, number, number, number, number, number, number, number]
export type GradeOffset = [number, number, number]

export type Grade = { m: GradeMatrix; b: GradeOffset }

export const IDENTITY_GRADE: Grade = { m: [1, 0, 0, 0, 1, 0, 0, 0, 1], b: [0, 0, 0] }

export const GRADE_STRENGTH = 0.45
const RIDGE = 1e-4

const round6 = (v: number) => Math.round(v * 1e6) / 1e6

// The homepage keeps its own colours and takes the 404's light as an affine RGB grade: the least
// squares fit of every LIGHT ramp colour onto the same colour of lighting(), pulled toward the
// identity by a small ridge, then blended toward the identity by strength. Rows of the augmented
// normal equations are [XᵀX | XᵀY] for inputs [r, g, b, 1].
export function sceneGrade(al: Almanac, s: Season, sky: Sky, strength = GRADE_STRENGTH): Grade {
  if (strength <= 0) return cloneGrade(IDENTITY_GRADE)
  const [target] = lighting(al, s, sky)
  const aug = Array.from({ length: 4 }, (_, row) =>
    Array.from({ length: 7 }, (_, col) => (col === row || col === row + 4 ? RIDGE : 0)),
  )
  // Ramps only: the single accent colours would weigh a handful of hues too heavily.
  for (const [key, ramp] of Object.entries(LIGHT)) {
    if (!Array.isArray(ramp) || !Array.isArray(ramp[0])) continue
    const from = ramp as RGB[]
    const to = target[key as keyof typeof LIGHT] as RGB[]
    for (let i = 0; i < from.length; i++) {
      const x = [from[i][0] / 255, from[i][1] / 255, from[i][2] / 255, 1]
      const y = [to[i][0] / 255, to[i][1] / 255, to[i][2] / 255]
      for (let row = 0; row < 4; row++) {
        for (let col = 0; col < 4; col++) aug[row][col] += x[row] * x[col]
        for (let c = 0; c < 3; c++) aug[row][4 + c] += x[row] * y[c]
      }
    }
  }
  // Gauss-Jordan with partial pivoting; aug[row][4 + c] ends as the coefficient of input row on
  // output channel c.
  for (let col = 0; col < 4; col++) {
    let pivot = col
    for (let row = col + 1; row < 4; row++)
      if (Math.abs(aug[row][col]) > Math.abs(aug[pivot][col])) pivot = row
    ;[aug[col], aug[pivot]] = [aug[pivot], aug[col]]
    const diag = aug[col][col]
    if (!Number.isFinite(diag) || Math.abs(diag) < 1e-12) return cloneGrade(IDENTITY_GRADE)
    for (let j = col; j < 7; j++) aug[col][j] /= diag
    for (let row = 0; row < 4; row++) {
      const factor = aug[row][col]
      if (row === col || factor === 0) continue
      for (let j = col; j < 7; j++) aug[row][j] -= factor * aug[col][j]
    }
  }
  const coefficient = (input: number, channel: number) =>
    round6(
      (input === channel ? 1 : 0) +
        (aug[input][4 + channel] - (input === channel ? 1 : 0)) * strength,
    )
  return {
    m: [0, 1, 2].flatMap(c => [0, 1, 2].map(i => coefficient(i, c))) as GradeMatrix,
    b: [0, 1, 2].map(c => round6(aug[3][4 + c] * strength)) as GradeOffset,
  }
}

export function cloneGrade(grade: Grade): Grade {
  return { m: [...grade.m] as GradeMatrix, b: [...grade.b] as GradeOffset }
}

export function equalGrade(a: Grade, b: Grade, eps = 1e-5): boolean {
  for (let i = 0; i < 9; i++) if (Math.abs(a.m[i] - b.m[i]) > eps) return false
  for (let i = 0; i < 3; i++) if (Math.abs(a.b[i] - b.b[i]) > eps) return false
  return true
}

export function isIdentityGrade(grade: Grade, eps = 1e-4): boolean {
  return equalGrade(grade, IDENTITY_GRADE, eps)
}

export function interpolateGrade(from: Grade, to: Grade, t: number): Grade {
  if (t <= 0) return cloneGrade(from)
  if (t >= 1) return cloneGrade(to)
  const m = from.m.map((v, i) => v + (to.m[i] - v) * t) as GradeMatrix
  const b = from.b.map((v, i) => v + (to.b[i] - v) * t) as GradeOffset
  return { m, b }
}

export function formatGrade(grade: Grade): string {
  return [...grade.m, ...grade.b].map(v => v.toFixed(4)).join(' ')
}

export function gradeColorMatrixValues(grade: Grade): string {
  const { m, b } = grade
  return `${m[0]} ${m[1]} ${m[2]} 0 ${b[0]} ${m[3]} ${m[4]} ${m[5]} 0 ${b[1]} ${m[6]} ${m[7]} ${m[8]} 0 ${b[2]} 0 0 0 1 0`
}
