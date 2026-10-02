import { Plugin } from 'obsidian'
import { MarkerAnnotations } from './src/annotations'
import { markerEditorExtension } from './src/editor'
import { processMarkers } from './src/renderer'
import { loadSettings, MarkedSettingsTab, type MarkerSettings } from './src/settings'

export default class MarkedPlugin extends Plugin {
  settings: MarkerSettings = loadSettings(undefined)
  private annotations?: MarkerAnnotations

  async onload(): Promise<void> {
    this.settings = loadSettings(await this.loadData())
    const annotations = this.addChild(new MarkerAnnotations(this))
    this.annotations = annotations
    this.registerEvent(this.app.workspace.on('css-change', () => annotations.schedule()))
    this.registerEvent(this.app.workspace.on('layout-change', () => annotations.schedule()))
    this.registerMarkdownPostProcessor((el, ctx) => processMarkers(el, ctx, annotations), 100)
    this.registerEditorExtension(markerEditorExtension(annotations))
    this.addSettingTab(new MarkedSettingsTab(this))
  }

  async saveSettings(): Promise<void> {
    await this.saveData(this.settings)
    this.annotations?.schedule()
  }
}
