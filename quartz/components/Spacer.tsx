import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"
import { classNames } from "../util/lang"

interface Options {
  classNames?: string[]
}

const defaultOptions = { classNames: [] }

export default ((userOpts?: Options) => {
  const opts = { ...defaultOptions, ...userOpts }

  const Spacer: QuartzComponent = ({ displayClass }: QuartzComponentProps) => {
    return <div class={classNames(displayClass, "spacer", ...opts.classNames)}></div>
  }

  return Spacer
}) satisfies QuartzComponentConstructor
