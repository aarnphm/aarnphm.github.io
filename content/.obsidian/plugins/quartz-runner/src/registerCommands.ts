import { Notice } from "obsidian"
import type QuartzRunner from "./main"
import { followQuartzLogs, startQuartzDev, stopQuartzDev } from "./runner/quartzDev"
import { resolveGitRoot } from "./runner/resolveGitRoot"

export function registerQuartzRunnerCommands(plugin: QuartzRunner): void {
	plugin.addCommand({
		id: "quartz-stop-dev",
		name: "quartz: stop dev server",
		callback: async () => {
			try {
				const gitRoot = await resolveGitRoot(plugin.app)
				const result = await stopQuartzDev(gitRoot)
				if (result.status === "not-running") {
					new Notice(result.pid ? `quartz dev not running (stale pid ${result.pid})` : "quartz dev not running")
					return
				}
				new Notice(`quartz dev stopped (pid ${result.pid})`)
			} catch (err) {
				new Notice(`quartz stop failed: ${(err as Error)?.message ?? String(err)}`)
			}
		},
	})

	plugin.addCommand({
		id: "quartz-start-dev",
		name: "quartz: start dev server",
		callback: async () => {
			try {
				const gitRoot = await resolveGitRoot(plugin.app)
				const result = await startQuartzDev(gitRoot, plugin.settings)
				if (result.status === "already-running") {
					new Notice(`quartz dev already running (pid ${result.pid})`)
					return
				}
				new Notice(result.pid ? `quartz dev started (pid ${result.pid})` : "quartz dev started")
			} catch (err) {
				new Notice(`quartz dev failed: ${(err as Error)?.message ?? String(err)}`)
			}
		},
	})

	plugin.addCommand({
		id: "quartz-follow-dev-logs",
		name: "quartz: follow dev logs",
		callback: async () => {
			try {
				const gitRoot = await resolveGitRoot(plugin.app)
				const pid = await followQuartzLogs(gitRoot, plugin.settings)
				new Notice(pid ? `following quartz logs (pid ${pid})` : "following quartz logs")
			} catch (err) {
				new Notice(`follow logs failed: ${(err as Error)?.message ?? String(err)}`)
			}
		},
	})

	plugin.addCommand({
		id: "quartz-start-dev-and-follow-logs",
		name: "quartz: start dev server and follow logs",
		callback: async () => {
			try {
				const gitRoot = await resolveGitRoot(plugin.app)
				const result = await startQuartzDev(gitRoot, plugin.settings)
				await followQuartzLogs(gitRoot, plugin.settings)
				if (result.status === "already-running") {
					new Notice(`quartz dev already running (pid ${result.pid})`)
					return
				}
				new Notice(result.pid ? `quartz dev started (pid ${result.pid})` : "quartz dev started")
			} catch (err) {
				new Notice(`quartz dev failed: ${(err as Error)?.message ?? String(err)}`)
			}
		},
	})
}
