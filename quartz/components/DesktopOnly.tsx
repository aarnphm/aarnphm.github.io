import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"

export default ((Component?: QuartzComponent) => {
  if (Component) {
    const DesktopOnly: QuartzComponent = (props: QuartzComponentProps) => {
      return <Component displayClass="desktop-only" {...props} />
    }

    DesktopOnly.displayName = Component.displayName
    DesktopOnly.afterDOMLoaded = Component?.afterDOMLoaded
    DesktopOnly.beforeDOMLoaded = Component?.beforeDOMLoaded
    DesktopOnly.css = Component?.css
    DesktopOnly.skipDuringServe = Component?.skipDuringServe
    return DesktopOnly
  }
  return () => <></>
}) satisfies QuartzComponentConstructor
