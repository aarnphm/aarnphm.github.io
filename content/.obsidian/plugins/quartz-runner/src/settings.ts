export interface QuartzRunnerSettings {
	retryLimit: number
	tailLines: number
}

export const DEFAULT_SETTINGS: QuartzRunnerSettings = {
	retryLimit: 9999,
	tailLines: 200,
}
