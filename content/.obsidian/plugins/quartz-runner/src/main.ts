import { Plugin } from "obsidian"
import { DEFAULT_SETTINGS, type QuartzRunnerSettings } from "./settings"
import { QuartzRunnerSettingTab } from "./settingsTab"
import { registerQuartzRunnerCommands } from "./registerCommands"

export default class ObsidianQuartzRunner extends Plugin {
	settings: QuartzRunnerSettings

	async onload(): Promise<void> {
		this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData())
		this.addSettingTab(new QuartzRunnerSettingTab(this.app, this))
		registerQuartzRunnerCommands(this)
	}

	async saveSettings(settings: QuartzRunnerSettings): Promise<void> {
		this.settings = settings
		await this.saveData(settings)
	}
}
