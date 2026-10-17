import { useLayoutEffect, useRef } from 'preact/hooks'
import { MarkdownEditor } from '../scripts/markdown-editor'

interface EditorProps {
  label: string
  value: string
  limit: number
  autoFocus?: boolean
  onChange(body: string): void
  onExit(): void
}

export function DraftEditor(props: EditorProps) {
  const host = useRef<HTMLDivElement>(null)
  const editor = useRef<MarkdownEditor | null>(null)
  const callbacks = useRef(props)
  callbacks.current = props
  const replacing = useRef(false)
  const lastEmitted = useRef(props.value)
  useLayoutEffect(() => {
    if (!host.current) return
    const view = new MarkdownEditor({
      parent: host.current,
      initialContent: callbacks.current.value,
      ariaLabel: callbacks.current.label,
      placeholder: 'Write a thought, question, or proof. Markdown and $math$ work.',
      maxLength: callbacks.current.limit,
      changeDebounceMs: 120,
      onChange: body => {
        if (!replacing.current) {
          lastEmitted.current = body
          callbacks.current.onChange(body)
        }
      },
      onCancel: () => callbacks.current.onExit(),
      onSubmit: () => callbacks.current.onExit(),
    })
    editor.current = view
    if (callbacks.current.autoFocus) view.focus()
    return () => {
      editor.current = null
      view.destroy()
    }
  }, [])
  useLayoutEffect(() => {
    if (props.value === lastEmitted.current) return
    lastEmitted.current = props.value
    replacing.current = true
    editor.current?.setValue(props.value)
    replacing.current = false
  }, [props.value])
  return <div class="course-draft-editor" ref={host} />
}
