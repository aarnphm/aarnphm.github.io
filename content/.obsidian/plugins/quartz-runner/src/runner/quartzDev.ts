import { spawn, spawnSync } from "child_process"
import { devFiles } from "./devFiles"
import { isPidRunning, readPidFile } from "./pid"
import { resolveLocalTsx } from "./tsx"
import type { QuartzRunnerSettings } from "../settings"
import { openTerminalWithCommand } from "../terminal/openTerminal"

async function waitForPidFile(pidFile: string, timeoutMs: number): Promise<number | null> {
	const startedAt = Date.now()
	for (;;) {
		const pid = await readPidFile(pidFile)
		if (pid) return pid
		if (Date.now() - startedAt > timeoutMs) return null
		await new Promise((r) => setTimeout(r, 100))
	}
}

function shellQuotePosix(value: string): string {
	if (value.length === 0) return "''"
	return `'${value.replace(/'/g, `'\"'\"'`)}'`
}

export type StartResult = { status: "started"; pid: number | null } | { status: "already-running"; pid: number }

function truncate(value: string, limit: number): string {
	const trimmed = value.trim()
	if (trimmed.length <= limit) return trimmed
	return `${trimmed.slice(0, limit)}…`
}

function buildSpawnEnv(gitRoot: string): NodeJS.ProcessEnv {
	const env: NodeJS.ProcessEnv = { ...process.env, GIT_ROOT: gitRoot }
	const delimiter = process.platform === "win32" ? ";" : ":"
	const home = env.HOME ?? env.USERPROFILE ?? ""

	const extraEntries: string[] = []
	if (process.platform === "darwin") {
		extraEntries.push("/opt/homebrew/bin", "/usr/local/bin")
	}
	if (process.platform === "linux") {
		extraEntries.push("/home/linuxbrew/.linuxbrew/bin", "/usr/local/bin")
	}
	if (home) {
		extraEntries.push(`${home}/.volta/bin`, `${home}/.asdf/shims`)
	}

	const basePath = env.PATH ?? ""
	const parts = basePath.split(delimiter).filter((part) => part.length > 0)
	const merged = [
		...extraEntries.filter((candidate) => candidate.length > 0 && !parts.includes(candidate)),
		...parts,
	]
	env.PATH = merged.join(delimiter)
	return env
}

async function spawnAndCapture(
	command: string,
	args: string[],
	opts: { cwd: string; env: NodeJS.ProcessEnv; timeoutMs: number },
): Promise<{ code: number | null; signal: NodeJS.Signals | null; stdout: string; stderr: string }> {
	return await new Promise((resolve, reject) => {
		const child = spawn(command, args, { cwd: opts.cwd, env: opts.env, stdio: ["ignore", "pipe", "pipe"] })

		const stdoutChunks: Buffer[] = []
		const stderrChunks: Buffer[] = []
		child.stdout.on("data", (chunk) => stdoutChunks.push(Buffer.from(chunk)))
		child.stderr.on("data", (chunk) => stderrChunks.push(Buffer.from(chunk)))

		const timer = setTimeout(() => {
			try {
				child.kill("SIGKILL")
			} catch {}
			reject(new Error(`command timed out: ${command}`))
		}, opts.timeoutMs)

		child.once("error", (err) => {
			clearTimeout(timer)
			reject(err)
		})
		child.once("exit", (code, signal) => {
			clearTimeout(timer)
			resolve({
				code,
				signal,
				stdout: Buffer.concat(stdoutChunks).toString("utf8"),
				stderr: Buffer.concat(stderrChunks).toString("utf8"),
			})
		})
	})
}

export async function startQuartzDev(gitRoot: string, settings: QuartzRunnerSettings): Promise<StartResult> {
	const { pidFile } = devFiles(gitRoot)
	const existingPid = await readPidFile(pidFile)
	if (existingPid && isPidRunning(existingPid)) {
		return { status: "already-running", pid: existingPid }
	}

	const tsx = await resolveLocalTsx(gitRoot)
	const env = buildSpawnEnv(gitRoot)
	const command = `${shellQuotePosix(tsx)} quartz/scripts/dev.ts --retry ${settings.retryLimit} --bg`
	const shell = process.platform === "darwin" ? "/bin/zsh" : "bash"
	const args =
		process.platform === "darwin" ? ["-lic", command] : process.platform === "win32" ? [] : ["-lc", command]

	const result =
		process.platform === "win32"
			? await spawnAndCapture(tsx, ["quartz/scripts/dev.ts", "--retry", String(settings.retryLimit), "--bg"], {
					cwd: gitRoot,
					env,
					timeoutMs: 20000,
				})
			: await spawnAndCapture(shell, args, { cwd: gitRoot, env, timeoutMs: 20000 })

	if (result.code !== 0) {
		const suffix = result.stderr.trim().length > 0 ? `: ${truncate(result.stderr, 400)}` : ""
		throw new Error(`quartz dev bootstrap failed (exit ${result.code ?? "null"})${suffix}`)
	}

	const pid = await waitForPidFile(pidFile, 4000)
	if (pid && !isPidRunning(pid)) {
		throw new Error(`quartz dev exited immediately (pid ${pid})`)
	}
	return { status: "started", pid }
}

function chooseLinuxTerminal(): { cmd: string; args: string[] } | null {
	const candidates = [
		{ cmd: "x-terminal-emulator", args: ["-e"] },
		{ cmd: "gnome-terminal", args: ["--"] },
		{ cmd: "konsole", args: ["-e"] },
		{ cmd: "xterm", args: ["-e"] },
	]
	for (const candidate of candidates) {
		const resolved = spawnSync("which", [candidate.cmd])
		if (resolved.status === 0) return candidate
	}
	return null
}

export async function followQuartzLogs(gitRoot: string, settings: QuartzRunnerSettings): Promise<number | null> {
	const { pidFile } = devFiles(gitRoot)
	const pid = await readPidFile(pidFile)
	const tail = `tail -n ${settings.tailLines} -F .dev.log .dev.err.log`
	const header = pid ? `echo quartz\\ dev\\ pid:\\ ${pid} && ` : ""
	const command = `cd ${shellQuotePosix(gitRoot)} && ${header}${tail}`

	if (process.platform === "linux") {
		const terminal = chooseLinuxTerminal()
		if (terminal) {
			const child = spawn(terminal.cmd, [...terminal.args, "bash", "-lc", command], {
				detached: true,
				stdio: "ignore",
			})
			child.unref()
			return pid
		}
	}

	await openTerminalWithCommand(command)
	return pid
}

async function waitForExit(pid: number, timeoutMs: number): Promise<boolean> {
	const startedAt = Date.now()
	for (;;) {
		if (!isPidRunning(pid)) return true
		if (Date.now() - startedAt > timeoutMs) return !isPidRunning(pid)
		await new Promise((r) => setTimeout(r, 100))
	}
}

function killPidOrGroup(pid: number, signal: NodeJS.Signals): void {
	const isNoSuchProcess = (err: unknown): boolean =>
		!!err && typeof err === "object" && (err as NodeJS.ErrnoException).code === "ESRCH"

	if (process.platform !== "win32") {
		try {
			process.kill(-pid, signal)
			return
		} catch (err) {
			if (!isNoSuchProcess(err)) {
				throw err
			}
		}
	}
	try {
		process.kill(pid, signal)
	} catch (err) {
		if (!isNoSuchProcess(err)) {
			throw err
		}
	}
}

export type StopResult = { status: "stopped"; pid: number } | { status: "not-running"; pid: number | null }

export async function stopQuartzDev(gitRoot: string): Promise<StopResult> {
	const { pidFile } = devFiles(gitRoot)
	const pid = await readPidFile(pidFile)
	if (!pid) return { status: "not-running", pid: null }
	if (!isPidRunning(pid)) return { status: "not-running", pid }

	if (process.platform === "win32") {
		const killed = spawnSync("taskkill", ["/PID", String(pid), "/T", "/F"])
		if (killed.status === 0) return { status: "stopped", pid }
		throw new Error(`taskkill failed (exit ${killed.status ?? "null"})`)
	}

	killPidOrGroup(pid, "SIGINT")
	if (await waitForExit(pid, 4000)) return { status: "stopped", pid }

	killPidOrGroup(pid, "SIGTERM")
	if (await waitForExit(pid, 2500)) return { status: "stopped", pid }

	killPidOrGroup(pid, "SIGKILL")
	await waitForExit(pid, 1500)
	return { status: "stopped", pid }
}
