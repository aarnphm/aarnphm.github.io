import { readFile } from "fs/promises"

export function parsePid(raw: string): number | null {
	const trimmed = raw.trim()
	if (!trimmed) return null
	const parsed = Number(trimmed)
	if (!Number.isInteger(parsed) || parsed <= 0) return null
	return parsed
}

export function isPidRunning(pid: number): boolean {
	try {
		process.kill(pid, 0)
		return true
	} catch (err) {
		if (err && typeof err === "object" && (err as NodeJS.ErrnoException).code === "EPERM") return true
		return false
	}
}

export async function readPidFile(pidFile: string): Promise<number | null> {
	try {
		const raw = await readFile(pidFile, "utf8")
		return parsePid(raw)
	} catch {
		return null
	}
}
