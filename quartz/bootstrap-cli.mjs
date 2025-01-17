#!/usr/bin/env -S node --no-deprecation
import yargs from "yargs"
import { hideBin } from "yargs/helpers"
import { handleBuild } from "./cli/handlers.js"
import { BuildArgv } from "./cli/args.js"
import { version } from "./cli/constants.js"

yargs(hideBin(process.argv))
  .scriptName("quartz")
  .version(version)
  .usage("$0 <cmd> [args]")
  .command("build", "Build Quartz into a bundle of static HTML files", BuildArgv, async (argv) => {
    await handleBuild(argv)
  })
  .showHelpOnFail(true)
  .help()
  .strict()
  .demandCommand().argv
