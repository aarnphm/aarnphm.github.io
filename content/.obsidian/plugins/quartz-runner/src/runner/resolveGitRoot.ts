import { execFile } from "child_process"
import { access } from "fs/promises"
import * as path from "path"
import { FileSystemAdapter, type App } from "obsidian"

function execFileText(command: string, args: string[], options: { cwd?: string }): Promise<string> {
	return new Promise((resolve, reject) => {
		execFile(command, args, options, (err, stdout) => {
			if (err) {
				reject(err)
				return
			}
			resolve(String(stdout ?? ""))
		})
	})
}

async function pathExists(filePath: string): Promise<boolean> {
	try {
		await access(filePath)
		return true
	} catch {
		return false
	}
}

async function resolveGitRootFrom(basePath: string): Promise<string> {
	try {
		const stdout = await execFileText("git", ["rev-parse", "--show-toplevel"], { cwd: basePath })
		const candidate = stdout.trim()
		if (candidate) return candidate
	} catch {}

	let cursor = basePath
	for (let depth = 0; depth < 25; depth += 1) {
		if (await pathExists(path.join(cursor, ".git"))) return cursor
		const parent = path.dirname(cursor)
		if (parent === cursor) break
		cursor = parent
	}

	return basePath
}

export async function resolveGitRoot(app: App): Promise<string> {
	const adapter = app.vault.adapter
	if (!(adapter instanceof FileSystemAdapter)) {
		throw new Error("vault adapter is not filesystem-backed")
	}
	return await resolveGitRootFrom(adapter.getBasePath())
}
