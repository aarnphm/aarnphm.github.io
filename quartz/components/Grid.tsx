import {
  QuartzComponent,
  QuartzComponentConstructor,
  QuartzComponentProps,
} from "../types/component"
import { concatenateResources } from "../util/resources"

type GridConfig = {
  components: {
    Component: QuartzComponent
    column?: string
    row?: string
    area?: string
    align?: "start" | "end" | "center" | "stretch"
    justify?: "start" | "end" | "center" | "stretch"
  }[]
  columns?: string
  rows?: string
  autoFlow?: "row" | "column" | "dense" | "row dense" | "column dense"
  gap?: string
  columnGap?: string
  rowGap?: string
}

export default ((config: GridConfig) => {
  const Grid: QuartzComponent = (props: QuartzComponentProps) => {
    const columns = config.columns ?? "repeat(auto-fit, minmax(0, 1fr))"
    const rows = config.rows ?? "auto"
    const autoFlow = config.autoFlow ?? "row"
    const gap = config.gap ?? "1rem"
    const columnGap = config.columnGap ?? config.gap
    const rowGap = config.rowGap ?? config.gap

    return (
      <div
        style={`display: grid; grid-template-columns: ${columns}; grid-template-rows: ${rows}; grid-auto-flow: ${autoFlow}; gap: ${gap}; column-gap: ${columnGap ?? gap}; row-gap: ${rowGap ?? gap};`}
      >
        {config.components.map((c) => {
          const column = c.column ?? "auto"
          const row = c.row ?? "auto"
          const area = c.area ?? "auto"
          const align = c.align ?? "stretch"
          const justify = c.justify ?? "stretch"

          return (
            <div
              style={`grid-column: ${column}; grid-row: ${row}; grid-area: ${area}; align-self: ${align}; justify-self: ${justify};`}
            >
              <c.Component {...props} />
            </div>
          )
        })}
      </div>
    )
  }

  Grid.afterDOMLoaded = concatenateResources(
    ...config.components.map((c) => c.Component.afterDOMLoaded),
  )
  Grid.beforeDOMLoaded = concatenateResources(
    ...config.components.map((c) => c.Component.beforeDOMLoaded),
  )
  Grid.css = concatenateResources(...config.components.map((c) => c.Component.css))
  return Grid
}) satisfies QuartzComponentConstructor<GridConfig>
