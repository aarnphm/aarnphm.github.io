import { spawn } from "child_process"

export async function openTerminalWithCommand(command: string): Promise<void> {
	if (process.platform === "darwin") {
		const appleScript = [
			"on run argv",
			'tell application "Terminal"',
			"activate",
			"do script (item 1 of argv)",
			"end tell",
			"end run",
		].join("\n")
		const child = spawn("osascript", ["-e", appleScript, command], { detached: true, stdio: "ignore" })
		child.unref()
		return
	}

	if (process.platform === "win32") {
		const child = spawn("cmd.exe", ["/c", "start", "cmd.exe", "/k", command], { detached: true, stdio: "ignore" })
		child.unref()
		return
	}

	const child = spawn("bash", ["-lc", command], { detached: true, stdio: "ignore" })
	child.unref()
}
