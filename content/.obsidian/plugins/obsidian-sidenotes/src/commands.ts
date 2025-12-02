import type { Plugin, MarkdownFileInfo } from "obsidian"
import { Editor, MarkdownView, EditorPosition } from "obsidian"

export function registerCommands(plugin: Plugin): void {
  plugin.addCommand({
    id: "insert-sidenote-template",
    name: "Insert sidenote template",
    editorCallback: (editor: Editor, _ctx: MarkdownView | MarkdownFileInfo) => {
      const selection = editor.getSelection()
      const content = selection || "<content here>"
      const template = `{{sidenotes[<items>]: ${content}}}`

      const from = editor.getCursor("from")
      editor.replaceSelection(template)

      if (!selection) {
        focusPlaceholder(editor, from, template)
      }
    },
  })
}

function focusPlaceholder(editor: Editor, from: EditorPosition, template: string): void {
  const placeholder = "<items>"
  const startOffset = editor.posToOffset(from)
  const idx = template.indexOf(placeholder)
  if (idx === -1) return

  const start = editor.offsetToPos(startOffset + idx)
  const end = editor.offsetToPos(startOffset + idx + placeholder.length)
  editor.setSelection(start, end)
}
