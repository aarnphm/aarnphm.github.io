import { App, PluginSettingTab, Setting } from "obsidian"
import type ObsidianQuartzRunner from "./main"
import { DEFAULT_SETTINGS, type QuartzRunnerSettings } from "./settings"

export class QuartzRunnerSettingTab extends PluginSettingTab {
	plugin: ObsidianQuartzRunner

	constructor(app: App, plugin: ObsidianQuartzRunner) {
		super(app, plugin)
		this.plugin = plugin
	}

	display(): void {
		const { containerEl } = this
		containerEl.empty()

		new Setting(containerEl)
			.setName("retry limit")
			.setDesc("passed to quartz/scripts/dev.ts as --retry")
			.addText((text) =>
				text
					.setPlaceholder(String(DEFAULT_SETTINGS.retryLimit))
					.setValue(String(this.plugin.settings.retryLimit))
					.onChange(async (value) => {
						const next = Number(value)
						const settings: QuartzRunnerSettings = {
							...this.plugin.settings,
							retryLimit: Number.isInteger(next) && next >= 0 ? next : DEFAULT_SETTINGS.retryLimit,
						}
						await this.plugin.saveSettings(settings)
					}),
			)

		new Setting(containerEl)
			.setName("tail lines")
			.setDesc("lines to show when following .dev.log/.dev.err.log")
			.addText((text) =>
				text
					.setPlaceholder(String(DEFAULT_SETTINGS.tailLines))
					.setValue(String(this.plugin.settings.tailLines))
					.onChange(async (value) => {
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
