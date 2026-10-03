import { App, MarkdownView, Plugin, PluginSettingTab, Setting } from 'obsidian'
import type { CharacterMotion, LeapOptions } from './src/types'
import { EditorLeap } from './src/editor-leap'
import { ReadingLeap } from './src/reading-leap'

const defaultOptions: LeapOptions = { vimMotions: true, readingMotions: true, showLabels: true }

function optionsFromData(data: unknown): LeapOptions {
  if (!data || typeof data !== 'object') return { ...defaultOptions }
  return {
    vimMotions:
      'vimMotions' in data && typeof data.vimMotions === 'boolean'
        ? data.vimMotions
        : defaultOptions.vimMotions,
    readingMotions:
      'readingMotions' in data && typeof data.readingMotions === 'boolean'
        ? data.readingMotions
        : defaultOptions.readingMotions,
    showLabels:
      'showLabels' in data && typeof data.showLabels === 'boolean'
        ? data.showLabels
        : defaultOptions.showLabels,
  }
}

export default class GardenLeapPlugin extends Plugin {
  options: LeapOptions = { ...defaultOptions }
  private editor?: EditorLeap
  private reading?: ReadingLeap

  async onload(): Promise<void> {
    const saved: unknown = await this.loadData()
    this.options = optionsFromData(saved)
    this.editor = this.addChild(new EditorLeap(this.app, () => this.options))
    this.reading = this.addChild(new ReadingLeap(this.app, () => this.options))
    this.registerEditorExtension(this.editor.extension)
    for (const motion of ['f', 'F', 't', 'T']) {
      if (!isCharacterMotion(motion)) continue
      const name =
        motion === 'f'
          ? 'Leap forward to character'
          : motion === 'F'
            ? 'Leap backward to character'
            : motion === 't'
              ? 'Leap forward until character'
              : 'Leap backward until character'
      this.addCommand({
        id: `leap-${motion}`,
        name,
        checkCallback: checking => {
          const view = this.app.workspace.getActiveViewOfType(MarkdownView)
          if (!view) return false
          if (!checking) {
            if (view.getMode() === 'preview') this.reading?.startCharacter(motion)
            else this.editor?.startCharacter(motion)
          }
          return true
        },
      })
    }
    for (const linewise of [false, true]) {
      this.addCommand({
        id: linewise ? 'select-syntax-lines' : 'select-syntax-node',
        name: linewise ? 'Select syntax node by lines' : 'Select syntax node',
        checkCallback: checking => {
          const view = this.app.workspace.getActiveViewOfType(MarkdownView)
          if (!view) return false
          if (!checking) {
            if (view.getMode() === 'preview') this.reading?.startSyntax(linewise)
            else this.editor?.startSyntax(linewise)
          }
          return true
        },
      })
    }
    this.addSettingTab(new LeapSettings(this.app, this))
    this.app.workspace.onLayoutReady(() => this.editor?.refreshBindings())
  }

  async saveOptions(): Promise<void> {
    this.reading?.cancel()
    this.editor?.refreshBindings()
    await this.saveData(this.options)
  }

  onunload(): void {
    this.editor = undefined
    this.reading = undefined
  }
}

function isCharacterMotion(value: string): value is CharacterMotion {
  return value === 'f' || value === 'F' || value === 't' || value === 'T'
}

class LeapSettings extends PluginSettingTab {
  constructor(
    app: App,
    private plugin: GardenLeapPlugin,
  ) {
    super(app, plugin)
  }

  display(): void {
    const { containerEl } = this
    containerEl.empty()
    containerEl.classList.add('garden-leap-settings', 'garden-plugin-ui')
    new Setting(containerEl).setName('Garden Leap').setHeading()
    new Setting(containerEl)
      .setName('Editor Vim motions')
      .setDesc('Use f/F/t/T and ga/gA in Vim Normal, Visual, and operator modes.')
      .addToggle(toggle =>
        toggle.setValue(this.plugin.options.vimMotions).onChange(async value => {
          this.plugin.options.vimMotions = value
          await this.plugin.saveOptions()
        }),
      )
    new Setting(containerEl)
      .setName('Reading view motions')
      .setDesc('Use f/F/t/T and ga/gA to navigate visible text and rendered sections.')
      .addToggle(toggle =>
        toggle.setValue(this.plugin.options.readingMotions).onChange(async value => {
          this.plugin.options.readingMotions = value
          await this.plugin.saveOptions()
        }),
      )
    new Setting(containerEl)
      .setName('Target labels')
      .setDesc('Show the remaining jump targets after the first match.')
      .addToggle(toggle =>
        toggle.setValue(this.plugin.options.showLabels).onChange(async value => {
          this.plugin.options.showLabels = value
          await this.plugin.saveOptions()
        }),
      )
  }
}
