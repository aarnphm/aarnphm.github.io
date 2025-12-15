import * as path from "path"

export interface DevFiles {
	pidFile: string
	logFile: string
	errFile: string
}

export function devFiles(gitRoot: string): DevFiles {
	return {
		pidFile: path.join(gitRoot, ".dev.pid"),
		logFile: path.join(gitRoot, ".dev.log"),
		errFile: path.join(gitRoot, ".dev.err.log"),
	}
}
