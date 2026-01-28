export const CommonArgv = {
  directory: {
    string: true,
    alias: ["d"],
    default: "content",
    describe: "directory to look for content files",
  },
  verbose: {
    boolean: true,
    alias: ["v"],
    default: false,
    describe: "print out extra logging information",
  },
  force: { boolean: true, alias: ["f"], default: false, describe: "force-load all plugins" },
}

export const BuildArgv = {
  ...CommonArgv,
  output: { string: true, alias: ["o"], default: "public", describe: "output folder for files" },
  serve: {
    boolean: true,
    default: false,
    describe: "run a local server to live-preview your Quartz",
  },
  watch: { boolean: true, default: false, describe: "watch for changes and rebuild automatically" },
  baseDir: { string: true, default: "", describe: "base path to serve your local server on" },
  port: { number: true, default: 8080, describe: "port to serve Quartz on" },
  wsPort: {
    number: true,
    default: 3001,
    describe: "port to use for WebSocket-based hot-reload notifications",
  },
  remoteDevHost: {
    string: true,
    default: "",
    describe: "A URL override for the websocket connection if you are not developing on localhost",
  },
  bundleInfo: { boolean: true, default: false, describe: "show detailed bundle information" },
  concurrency: { number: true, describe: "how many threads to use to parse notes" },
}

export const StatsArgv = { ...CommonArgv }
