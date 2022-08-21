"use strict"

/** @type {Record<string, Set<string>>} */
const optionalDepsToPrune = {
  agents: new Set(["x402"]),
}

/** @type {Record<string, string>} */
const allowedPeerRanges = {
  "use-sync-external-store>react": "^19.0.0",
  zod: "^3.22.0 || ^4.0.0",
}

/**
 * Remove noisy optional dependencies we never install.
 * @param {import('@pnpm/pnpmfile').ReadPackageHookInput} pkg
 * @param {import('@pnpm/pnpmfile').ReadPackageContext} ctx
 */
const readPackage = (pkg, ctx) => {
  const disallowed = optionalDepsToPrune[pkg.name]
  if (!disallowed) {
    return pkg
  }

  for (const depName of disallowed) {
    if (pkg.optionalDependencies && depName in pkg.optionalDependencies) {
      ctx?.log?.(`removing optional dependency "${depName}" from ${pkg.name}`)
      delete pkg.optionalDependencies[depName]
    }
    if (pkg.dependencies && depName in pkg.dependencies) {
      delete pkg.dependencies[depName]
    }
    if (pkg.peerDependencies && depName in pkg.peerDependencies) {
      delete pkg.peerDependencies[depName]
    }
  }

  return pkg
}

/**
 * Mirror peer dependency rules from package.json so pnpm picks them up early.
 * Requires pnpm >= 10.8.0, older versions ignore this hook.
 * @param {import('@pnpm/pnpmfile').UpdateConfigHookInput} config
 */
const updateConfig = (config) => {
  config.peerDependencyRules ??= {}
  config.peerDependencyRules.allowedVersions ??= {}

  for (const [selector, range] of Object.entries(allowedPeerRanges)) {
    config.peerDependencyRules.allowedVersions[selector] ??= range
  }

  return config
}

module.exports = {
  hooks: {
    readPackage,
    updateConfig,
  },
}
