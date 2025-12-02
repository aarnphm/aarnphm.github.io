import { Plugin } from "obsidian"
import { processSidenotes } from "./src/renderer"
import { registerCommands } from "./src/commands"

export default class SidenotesPlugin extends Plugin {
  async onload(): Promise<void> {
    this.registerMarkdownPostProcessor((el, ctx) => {
      return processSidenotes(el, ctx, this)
    })
    registerCommands(this)
  }
}
