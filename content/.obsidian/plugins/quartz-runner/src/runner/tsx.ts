import { access } from "fs/promises"
import * as path from "path"

async function pathExists(filePath: string): Promise<boolean> {
	try {
		await access(filePath)
		return true
	} catch {
		return false
	}
}

export async function resolveLocalTsx(gitRoot: string): Promise<string> {
	const binName = process.platform === "win32" ? "tsx.cmd" : "tsx"
	const candidate = path.join(gitRoot, "node_modules", ".bin", binName)
	if (await pathExists(candidate)) return candidate
	return "tsx"
}
