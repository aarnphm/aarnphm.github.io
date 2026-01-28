#!/usr/bin/env -S node --no-deprecation
import yargs from "yargs"
import { hideBin } from "yargs/helpers"
import { BuildArgv, StatsArgv } from "./cli/args.js"
import { version } from "./cli/constants.js"
import { handleBuild, handleStats } from "./cli/handlers.js"

yargs(hideBin(process.argv))
  .scriptName("quartz")
  .version(version)
  .usage("$0 <cmd> [args]")
  .command("build", "Build Quartz into a bundle of static HTML files", BuildArgv, async (argv) => {
    await handleBuild(argv)
  })
  .command("stats", "Show bundle and vault stats", StatsArgv, async (argv) => {
    await handleStats(argv)
  })
  .showHelpOnFail(true)
  .help()
  .strict()
  .demandCommand().argv
