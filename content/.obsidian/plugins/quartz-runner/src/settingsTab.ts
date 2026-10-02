import { App, PluginSettingTab, Setting } from 'obsidian'
import type QuartzRunner from './main'
import { DEFAULT_SETTINGS, type QuartzRunnerSettings } from './settings'

export class QuartzRunnerSettingTab extends PluginSettingTab {
  plugin: QuartzRunner

  constructor(app: App, plugin: QuartzRunner) {
    super(app, plugin)
    this.plugin = plugin
  }

  display(): void {
    const { containerEl } = this
    containerEl.empty()
    containerEl.addClass('quartz-runner-settings', 'garden-plugin-ui')
    containerEl.createEl('h2', { text: 'Quartz Runner' })
    containerEl.createEl('p', {
      cls: 'quartz-runner-description',
      text: 'Start the local garden server and follow its logs from the command palette.',
    })

    new Setting(containerEl)
      .setName('Retry limit')
      .setDesc('How many times the server may restart after a failure.')
      .addText(text =>
        text
          .setPlaceholder(String(DEFAULT_SETTINGS.retryLimit))
          .setValue(String(this.plugin.settings.retryLimit))
          .onChange(async value => {
            const next = Number(value)
            const settings: QuartzRunnerSettings = {
              ...this.plugin.settings,
              retryLimit: Number.isInteger(next) && next >= 0 ? next : DEFAULT_SETTINGS.retryLimit,
            }
            await this.plugin.saveSettings(settings)
          }),
      )

    new Setting(containerEl)
      .setName('Log lines')
      .setDesc('How many recent lines to show when opening the server logs.')
      .addText(text =>
        text
          .setPlaceholder(String(DEFAULT_SETTINGS.tailLines))
          .setValue(String(this.plugin.settings.tailLines))
          .onChange(async value => {
            const next = Number(value)
            const settings: QuartzRunnerSettings = {
              ...this.plugin.settings,
              tailLines: Number.isInteger(next) && next > 0 ? next : DEFAULT_SETTINGS.tailLines,
            }
            await this.plugin.saveSettings(settings)
          }),
      )
  }
}
