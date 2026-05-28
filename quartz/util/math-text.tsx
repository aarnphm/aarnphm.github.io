import katex from 'katex'
import { type ComponentChildren, type FunctionalComponent } from 'preact'
import { customMacros, katexOptions } from '../cfg'

export const renderInlineMath = (tex: string, display = false): string =>
  katex.renderToString(tex, {
    ...katexOptions,
    displayMode: display,
    output: 'html',
    macros: customMacros,
    strict: false,
    throwOnError: false,
  })

export const InlineMath: FunctionalComponent<{
  tex: string
  display?: boolean
  className?: string
}> = ({ tex, display, className }) => (
  <span class={className} dangerouslySetInnerHTML={{ __html: renderInlineMath(tex, display) }} />
)

export const MathText: FunctionalComponent<{ text: string; mathClass?: string }> = ({
  text,
  mathClass,
}) => {
  const children: ComponentChildren[] = []
  const mathPattern = /\$([^$]+)\$/g
  let lastIndex = 0
  let partIndex = 0

  for (const match of text.matchAll(mathPattern)) {
    if (match.index > lastIndex) children.push(text.slice(lastIndex, match.index))
    children.push(<InlineMath tex={match[1]} className={mathClass} key={partIndex} />)
    partIndex += 1
    lastIndex = match.index + match[0].length
  }

  if (lastIndex < text.length) children.push(text.slice(lastIndex))
  return <>{children}</>
}
