import { MathText } from '../../util/math-text'
import style from '../styles/sqlJoin.scss'
import { registerMdxComponent, type QuartzMdxComponent } from './registry'

type Props = { caption?: string; mode?: Mode }
type Mode = 'inner' | 'left' | 'right' | 'full'

const MODES: { id: Mode; label: string }[] = [
  { id: 'inner', label: 'inner' },
  { id: 'left', label: 'left outer' },
  { id: 'right', label: 'right outer' },
  { id: 'full', label: 'full outer' },
]

const COURSE = [
  ['BIO-301', 'Genetics', 'Biology', '4'],
  ['CS-190', 'Game Design', 'Comp. Sci.', '4'],
  ['CS-315', 'Robotics', 'Comp. Sci.', '3'],
] as const

const PREREQ = [
  ['BIO-301', 'BIO-101'],
  ['CS-190', 'CS-101'],
  ['CS-347', 'CS-101'],
] as const

type Row = string[]

const joinRows = (mode: Mode): Row[] => {
  const courseById = new Map<string, Row>(COURSE.map(r => [r[0], [...r]]))
  const matched: Row[] = []
  const matchedCourse = new Set<string>()

  for (const [cid, pid] of PREREQ) {
    const c = courseById.get(cid)
    if (!c) continue
    matchedCourse.add(cid)
    matched.push([...c, pid])
  }

  const leftDangle = COURSE.filter(c => !matchedCourse.has(c[0])).map(c => [...c, 'null'] as Row)
  const rightDangle = PREREQ.filter(([cid]) => !courseById.has(cid)).map(
    ([cid, pid]) => [cid, 'null', 'null', 'null', pid] as Row,
  )

  if (mode === 'inner') return matched
  if (mode === 'left') return [...matched, ...leftDangle]
  if (mode === 'right') return [...matched, ...rightDangle]
  return [...matched, ...leftDangle, ...rightDangle]
}

const HEADERS = ['course_id', 'title', 'dept_name', 'credits', 'prereq_id'] as const

let instance = 0

const Table = ({
  name,
  headers,
  rows,
  tone,
}: {
  name: string
  headers: readonly string[]
  rows: readonly (readonly string[])[]
  tone: 'course' | 'prereq' | 'result'
}) => (
  <div class={`sj-table sj-table--${tone}`}>
    <div class="sj-name">
      <span class={`sj-swatch sj-swatch--${tone}`} aria-hidden="true" />
      <span>{name}</span>
    </div>
    <table>
      <thead>
        <tr>
          {headers.map(h => (
            <th key={h}>{h}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((row, i) => (
          <tr key={`${name}-${i}`}>
            {row.map((cell, j) => (
              <td key={j} class={cell === 'null' ? 'sj-null' : undefined}>
                {cell}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  </div>
)

const SqlJoinImpl: QuartzMdxComponent<Props> = ({ caption, mode = 'inner' }) => {
  const id = `sj-${++instance}`
  const initial = MODES.some(m => m.id === mode) ? mode : 'inner'

  return (
    <figure class="sql-join" data-sql-join data-sql-mode={initial}>
      <fieldset class="sj-segments">
        <legend class="sj-legend">join</legend>
        {MODES.map(m => (
          <label class="sj-segment" key={m.id}>
            <input
              type="radio"
              name={`${id}-mode`}
              value={m.id}
              checked={m.id === initial}
              data-sql-mode-input
            />
            <span>{m.label}</span>
          </label>
        ))}
      </fieldset>

      <div class="sj-sources">
        <Table
          name="course"
          tone="course"
          headers={['course_id', 'title', 'dept_name', 'credits']}
          rows={COURSE}
        />
        <Table name="prereq" tone="prereq" headers={['course_id', 'prereq_id']} rows={PREREQ} />
      </div>

      {MODES.map(m => (
        <div class="sj-result" data-sql-panel={m.id} key={m.id}>
          <p class="sj-op">
            course <strong>{m.label}</strong> join prereq on course.course_id = prereq.course_id
          </p>
          <Table name="result" tone="result" headers={HEADERS} rows={joinRows(m.id)} />
        </div>
      ))}

      {caption ? (
        <figcaption class="sj-caption">
          <MathText text={caption} mathClass="sj-math" />
        </figcaption>
      ) : null}
    </figure>
  )
}

const SqlJoinComponent = SqlJoinImpl as QuartzMdxComponent<Props>
SqlJoinComponent.css = style

export const SqlJoin = registerMdxComponent('SqlJoin', SqlJoinComponent)

export default (() => SqlJoin) satisfies (opts: undefined) => QuartzMdxComponent<Props>
