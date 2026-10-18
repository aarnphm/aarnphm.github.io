import { Plugin } from 'obsidian'
import { CoverCache } from './src/covers'
import { bookcaseOptions } from './src/options'
import { BookcaseView, BOOKCASE_VIEW } from './src/view'

export default class BookcasePlugin extends Plugin {
  covers!: CoverCache

  async onload(): Promise<void> {
    this.covers = this.addChild(new CoverCache(this))
    await this.covers.load()
    this.registerBasesView(BOOKCASE_VIEW, {
      name: 'Bookcase',
      icon: 'library',
      factory: (controller, containerEl) => new BookcaseView(controller, containerEl, this),
      options: bookcaseOptions,
    })
  }
}
