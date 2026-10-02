import { Plugin } from 'obsidian'
import { registerCommands } from './src/commands'
import { sidenoteEditorExtension } from './src/editor'
import { SidenoteLayout } from './src/layout'
import { processSidenotes } from './src/renderer'

export default class SidenotesPlugin extends Plugin {
  async onload(): Promise<void> {
    const layout = this.addChild(new SidenoteLayout(this.app.workspace.containerEl))
    this.registerEvent(this.app.workspace.on('resize', () => layout.schedule()))
    this.registerEvent(this.app.workspace.on('layout-change', () => layout.schedule()))
    this.registerMarkdownPostProcessor((el, ctx) => {
      return processSidenotes(el, ctx, this)
    })
    registerCommands(this)
    this.registerEditorExtension(sidenoteEditorExtension(this))
  }
}
