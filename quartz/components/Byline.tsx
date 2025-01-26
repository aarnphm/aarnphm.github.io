import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"

export default ((...components: QuartzComponent[]) => {
  const Components = Array.from(components)
  const Byline: QuartzComponent = (props: QuartzComponentProps) => {
    return (
      <section class="byline all-col grid">
        {Components.filter((Inner) => (props.ctx.argv.serve ? !Inner.skipDuringServe : true)).map(
          (Inner) => (
            <Inner {...props} />
          ),
        )}
      </section>
    )
  }

  Byline.css = Components.map((Inner) => Inner.css ?? "").join("\n")
  Byline.beforeDOMLoaded = Components.map((Inner) => Inner.beforeDOMLoaded ?? "").join("\n")
  Byline.afterDOMLoaded = Components.map((Inner) => Inner.afterDOMLoaded ?? "").join("\n")
  return Byline
}) satisfies QuartzComponentConstructor<any>
