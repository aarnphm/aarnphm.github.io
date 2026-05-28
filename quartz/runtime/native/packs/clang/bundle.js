var __defProp = Object.defineProperty;
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};

// node_modules/@yowasp/runtime/lib/fetch.js
var fetch2;
if (typeof process === "object" && process.release?.name === "node") {
  fetch2 = async function(url, options) {
    if (url.protocol === "file:") {
      const { readFile } = await import("node:fs/promises");
      const data = await readFile(url);
      const isWasm = url.pathname.endsWith(".wasm");
      const headers = {
        "content-length": data.length,
        "content-type": isWasm ? "application/wasm" : "application/octet-stream"
      };
      return new Response(data, { headers });
    } else {
      return globalThis.fetch(url, options);
    }
  };
} else {
  fetch2 = globalThis.fetch;
}
var fetch_default = fetch2;

// node_modules/@yowasp/runtime/lib/wasi-virt.js
var Exit = class extends Error {
  constructor(code = 0) {
    super(`Exited with status ${code}`);
    this.code = code;
  }
};
function monotonicNow() {
  return BigInt(Math.floor(performance.now() * 1e6));
}
function wallClockNow() {
  let now = Date.now();
  const seconds = BigInt(Math.floor(now / 1e3));
  const nanoseconds = now % 1e3 * 1e6;
  return { seconds, nanoseconds };
}
var Xoroshiro128StarStar = class {
  constructor(seed) {
    if (BigInt(seed) === 0n) {
      throw new Error("xoroshiro128** must be seeded with a non-zero state");
    }
    this.s = [BigInt(seed) & 0xffffffffffffffffn, BigInt(seed) >> 64n & 0xffffffffffffffffn];
  }
  next() {
    function trunc64(x) {
      return x & 0xffffffffffffffffn;
    }
    function rotl(x, k) {
      return x << k | x >> 64n - k;
    }
    let [s0, s1] = this.s;
    const r = trunc64(rotl(s0 * 5n, 7n) * 9n);
    s1 ^= s0;
    s0 = trunc64(rotl(s0, 24n) ^ s1 ^ s1 << 16n);
    s1 = trunc64(rotl(s1, 37n));
    this.s = [s0, s1];
    return r;
  }
  getBytes(length) {
    return Uint8Array.from({ length }, () => Number(BigInt.asUintN(8, this.next() >> 32n)));
  }
};
var IoError = class extends Error {
};
var InputStream = class {
  read(_len) {
    throw { tag: "closed" };
  }
  blockingRead(len) {
    return this.read(len);
  }
};
var OutputStream = class {
  checkWrite() {
    throw { tag: "closed" };
  }
  write(_contents) {
    this.checkWrite();
  }
  flush() {
  }
  blockingFlush() {
    this.flush();
  }
  blockingWriteAndFlush(contents) {
    this.write(contents);
    this.blockingFlush();
  }
};
var CallbackInputStream = class extends InputStream {
  constructor(callback = null) {
    super();
    this.callback = callback;
  }
  read(len) {
    if (this.callback === null)
      throw { tag: "closed" };
    let contents = this.callback(Number(len));
    if (contents === null)
      throw { tag: "closed" };
    return contents;
  }
};
var CallbackOutputStream = class extends OutputStream {
  constructor(callback = null) {
    super();
    this.callback = callback;
  }
  checkWrite() {
    return 4096;
  }
  write(contents) {
    if (this.callback !== null)
      this.callback(contents);
  }
  flush() {
    if (this.callback !== null)
      this.callback(null);
  }
};
var TerminalInput = class {
};
var TerminalOutput = class {
};
var nextFilesystemId = /* @__PURE__ */ (function() {
  let id = 0;
  return () => id++;
})();
var File = class {
  constructor(data = "") {
    this.id = nextFilesystemId();
    if (data instanceof Uint8Array) {
      this.data = data;
    } else if (typeof data === "string") {
      this.data = new TextEncoder().encode(data);
    } else {
      throw new Error(`Cannot construct a file from ${typeof data}`);
    }
  }
  get size() {
    return this.data.length;
  }
};
var Directory = class _Directory {
  constructor(files = {}) {
    this.id = nextFilesystemId();
    this.files = files;
  }
  get size() {
    return Object.keys(this.files).length;
  }
  traverse(path, { create = null, remove = false } = {}) {
    let entry = this;
    let separatorAt = -1;
    do {
      if (entry instanceof File)
        throw "not-directory";
      const files = entry.files;
      separatorAt = path.indexOf("/");
      const segment = separatorAt === -1 ? path : path.substring(0, separatorAt);
      if (separatorAt === -1 && remove)
        delete files[segment];
      else if (segment === "" || segment === ".")
        ;
      else if (segment === "..")
        ;
      else if (Object.hasOwn(files, segment))
        entry = files[segment];
      else if (create === "directory" || create !== null && separatorAt !== -1)
        entry = files[segment] = new _Directory({});
      else if (create === "file")
        entry = files[segment] = new File(new Uint8Array());
      else if (create instanceof File || create instanceof _Directory)
        entry = files[segment] = create;
      else
        throw "no-entry";
      path = path.substring(separatorAt + 1);
    } while (separatorAt !== -1);
    return entry;
  }
};
var ReadStream = class extends InputStream {
  constructor(file, offset) {
    super();
    this.file = file;
    this.offset = offset;
  }
  read(len) {
    const data = this.file.data.subarray(Number(this.offset), Number(this.offset + len));
    this.offset += len;
    return data;
  }
};
var WriteStream = class extends OutputStream {
  constructor(file, offset) {
    super();
    this.file = file;
    this.offset = offset;
  }
  write(contents) {
    const offset = Number(this.offset);
    const newData = new Uint8Array(Math.max(this.file.data.length, offset + contents.length));
    newData.set(this.file.data);
    newData.subarray(offset).set(contents);
    this.file.data = newData;
    this.offset += BigInt(contents.length);
  }
};
var Descriptor = class _Descriptor {
  constructor(entry) {
    this.entry = entry;
  }
  getType() {
    if (this.entry instanceof Directory)
      return "directory";
    if (this.entry instanceof File)
      return "regular-file";
  }
  getFlags() {
    return {};
  }
  metadataHash() {
    return { upper: 0, lower: this.entry.id };
  }
  metadataHashAt(_pathFlags, path) {
    if (!(this.entry instanceof Directory))
      throw "invalid";
    const pathEntry = this.entry.traverse(path);
    return new _Descriptor(pathEntry).metadataHash();
  }
  stat() {
    let type;
    if (this.entry instanceof Directory)
      type = "directory";
    if (this.entry instanceof File)
      type = "regular-file";
    return {
      type,
      linkCount: 1,
      size: this.entry.size,
      dataAccessTimestamp: null,
      dataModificationTimestamp: null,
      statusChangeTimestamp: null
    };
  }
  statAt(_pathFlags, path) {
    if (!(this.entry instanceof Directory))
      throw "invalid";
    const pathEntry = this.entry.traverse(path);
    return new _Descriptor(pathEntry).stat();
  }
  openAt(_pathFlags, path, openFlags, _descriptorFlags) {
    if (!(this.entry instanceof Directory))
      throw "invalid";
    const openEntry = this.entry.traverse(path, openFlags.create ? { create: "file" } : {});
    if (openFlags.directory) {
      if (!(openEntry instanceof Directory))
        throw "not-directory";
    } else {
      if (openEntry instanceof Directory)
        throw "is-directory";
      if (openFlags.truncate)
        openEntry.data = new Uint8Array();
    }
    return new _Descriptor(openEntry);
  }
  read(length, offset) {
    if (this.entry instanceof Directory)
      throw "is-directory";
    [length, offset] = [Number(length), Number(offset)];
    return [this.entry.data.subarray(offset, offset + length), offset + length >= this.entry.data.byteLength];
  }
  readViaStream(offset) {
    return new ReadStream(this.entry, offset);
  }
  write(_buffer, _offset) {
    if (this.entry instanceof Directory)
      throw "is-directory";
    console.error("Descriptor.write not implemented");
    throw "unsupported";
  }
  writeViaStream(offset) {
    return new WriteStream(this.entry, offset);
  }
  setSize(size) {
    if (this.entry instanceof Directory)
      throw "is-directory";
    size = Number(size);
    if (size > this.entry.data.length) {
      const newData = new Uint8Array(size);
      newData.set(this.entry.data);
      this.entry.data = newData;
    } else if (size < this.entry.data.length) {
      this.entry.data = this.entry.data.subarray(0, size);
    }
  }
  readDirectory() {
    return new DirectoryEntryStream(this.entry);
  }
  createDirectoryAt(path) {
    this.entry.traverse(path, { create: "directory" });
  }
  unlinkFileAt(path) {
    const pathEntry = this.entry.traverse(path);
    if (pathEntry instanceof Directory)
      throw "is-directory";
    this.entry.traverse(path, { remove: true });
  }
  removeDirectoryAt(path) {
    const pathEntry = this.entry.traverse(path);
    if (!(pathEntry instanceof Directory))
      throw "not-directory";
    this.entry.traverse(path, { remove: true });
  }
  readlinkAt(path) {
    const _pathEntry = this.entry.traverse(path);
    throw "invalid";
  }
  renameAt(oldPath, newDescriptor, newPath) {
    if (!(this.entry instanceof Directory))
      throw "not-directory";
    if (!(newDescriptor.entry instanceof Directory))
      throw "not-directory";
    const oldEntry = this.entry.traverse(oldPath);
    this.entry.traverse(newPath, { create: oldEntry });
    this.entry.traverse(oldPath, { remove: true });
  }
};
var DirectoryEntryStream = class {
  constructor(directory) {
    this.entries = Object.entries(directory.files);
    this.index = 0;
  }
  readDirectoryEntry() {
    if (this.index === this.entries.length)
      return null;
    const [name, entry] = this.entries[this.index++];
    let type;
    if (entry instanceof Directory)
      type = "directory";
    if (entry instanceof File)
      type = "regular-file";
    return { name, type };
  }
};
function directoryFromTree(tree) {
  const files = {};
  for (const [filename, data] of Object.entries(tree)) {
    if (typeof data === "string" || data instanceof Uint8Array)
      files[filename] = new File(tree[filename]);
    else
      files[filename] = directoryFromTree(tree[filename]);
  }
  return new Directory(files);
}
function directoryIntoTree(directory, { decodeASCII = true } = {}) {
  function isASCII(buffer) {
    for (const byte of buffer)
      if (byte < 32 && byte !== 9 && byte !== 10 && byte !== 13 || byte >= 127)
        return false;
    return true;
  }
  const tree = {};
  for (const [filename, entry] of Object.entries(directory.files)) {
    if (entry instanceof File)
      tree[filename] = decodeASCII && isASCII(entry.data) ? new TextDecoder().decode(entry.data) : entry.data;
    if (entry instanceof Directory)
      tree[filename] = directoryIntoTree(entry, { decodeASCII });
  }
  return tree;
}
var Environment = class {
  vars = {};
  args = [];
  root = new Directory({});
  constructor() {
    this.prng = new Xoroshiro128StarStar(1n);
    this.standardInputStream = new CallbackInputStream();
    this.standardOutputStream = new CallbackOutputStream();
    this.standardErrorStream = new CallbackOutputStream();
    this.terminalInput = new TerminalInput();
    this.terminalOutput = new TerminalOutput();
    const $this = this;
    this.exports = {
      monotonicClock: {
        now: monotonicNow
      },
      wallClock: {
        now: wallClockNow
      },
      random: {
        getRandomBytes(length) {
          return $this.prng.getBytes(Number(length));
        }
      },
      io: {
        Error: IoError,
        InputStream,
        OutputStream
      },
      cli: {
        exit(status) {
          throw new Exit(status.tag === "ok" ? 0 : 1);
        },
        getEnvironment() {
          return $this.vars;
        },
        getArguments() {
          return $this.args;
        },
        getStdin() {
          return $this.standardInputStream;
        },
        getStdout() {
          return $this.standardOutputStream;
        },
        getStderr() {
          return $this.standardErrorStream;
        },
        getTerminalStdin() {
          return $this.terminalInput;
        },
        getTerminalStdout() {
          return $this.terminalOutput;
        },
        getTerminalStderr() {
          return $this.terminalOutput;
        },
        TerminalInput,
        TerminalOutput
      },
      fs: {
        Descriptor,
        DirectoryEntryStream,
        filesystemErrorCode() {
        },
        getDirectories() {
          if ($this.root === null) return [];
          return [[new Descriptor($this.root), "/"]];
        }
      }
    };
  }
  get stdin() {
    return this.standardInputStream.callback;
  }
  set stdin(callback) {
    this.standardInputStream.callback = callback;
  }
  get stdout() {
    return this.standardOutputStream.callback;
  }
  set stdout(callback) {
    this.standardOutputStream.callback = callback;
  }
  get stderr() {
    return this.standardErrorStream.callback;
  }
  set stderr(callback) {
    this.standardErrorStream.callback = callback;
  }
};

// node_modules/@yowasp/runtime/lib/util.js
function lineBuffered(processLine) {
  let buffer = new Uint8Array();
  return (bytes) => {
    if (bytes === null)
      return;
    let newBuffer = new Uint8Array(buffer.length + bytes.length);
    newBuffer.set(buffer);
    newBuffer.set(bytes, buffer.length);
    buffer = newBuffer;
    let newlineAt = -1;
    while (true) {
      const nextNewlineAt = buffer.indexOf(10, newlineAt + 1);
      if (nextNewlineAt === -1)
        break;
      processLine(new TextDecoder().decode(buffer.subarray(newlineAt + 1, nextNewlineAt)));
      newlineAt = nextNewlineAt;
    }
    buffer = buffer.subarray(newlineAt + 1);
  };
}

// node_modules/@yowasp/runtime/lib/api.js
var Application = class {
  #resourceModule;
  #resourceData;
  #instantiate;
  #argv0;
  constructor(resourceModule, instantiate2, argv0) {
    this.#resourceModule = resourceModule;
    this.#resourceData = null;
    this.#instantiate = instantiate2;
    this.#argv0 = argv0;
  }
  get argv0() {
    return this.#argv0;
  }
  async #fetchResources(fetchProgress) {
    const resourceModule = await this.#resourceModule;
    let fetchFn = fetch_default;
    if (fetchProgress !== void 0) {
      const status = { source: this, totalLength: resourceModule.totalSize, doneLength: 0 };
      fetchProgress(status);
      fetchFn = (input, init) => fetch_default(input, init).then((response) => {
        return new Response(response.body.pipeThrough(new TransformStream({
          transform(chunk, controller) {
            controller.enqueue(chunk);
            status.doneLength += chunk.length;
            fetchProgress(status);
          }
        })), response);
      });
    }
    const [modules2, filesystem2] = await Promise.all([
      resourceModule.modules(fetchFn),
      resourceModule.filesystem(fetchFn)
    ]);
    this.#resourceData = { modules: modules2, filesystem: filesystem2 };
  }
  // The `printLine` option is deprecated and not documented but still accepted for compatibility.
  run(args = null, files = {}, options = {}) {
    if (this.#resourceData === null) {
      if (options.synchronously)
        throw new Error("Cannot run application synchronously unless resources are prefetched first; use `await run()` to do so");
      const defaultFetchProgress = ({ source, totalLength, doneLength }) => {
        const percent = (100 * doneLength / totalLength).toFixed(0);
        console.log(`${source.argv0}: fetched ${percent}% (${doneLength} / ${totalLength})`);
      };
      return this.#fetchResources(options.fetchProgress ?? defaultFetchProgress).then(() => {
        return this.run(args, files, options);
      });
    }
    if (args === null)
      return;
    const environment = new Environment();
    environment.args = [this.#argv0].concat(args);
    environment.root = directoryFromTree(files);
    for (const [dirName, dirContents] of Object.entries(this.#resourceData.filesystem))
      environment.root.files[dirName] = directoryFromTree(dirContents);
    const lineBufferedConsole = lineBuffered(options.printLine ?? console.log);
    environment.stdin = options.stdin === void 0 ? null : options.stdin;
    environment.stdout = options.stdout === void 0 ? lineBufferedConsole : options.stdout;
    environment.stderr = options.stderr === void 0 ? lineBufferedConsole : options.stderr;
    const runCommand = (wasmCommand) => {
      let error = null;
      try {
        wasmCommand.run.run();
      } catch (e) {
        if (!(e instanceof Exit))
          throw e;
        if (e instanceof Exit && e.code !== 0)
          error = e;
      }
      for (const dirName of Object.keys(this.#resourceData.filesystem))
        delete environment.root.files[dirName];
      files = directoryIntoTree(environment.root, { decodeASCII: options.decodeASCII ?? true });
      if (error !== null) {
        error.files = files;
        throw error;
      } else {
        return files;
      }
    };
    const getCoreModule = (filename) => this.#resourceData.modules[filename];
    const imports = { runtime: environment.exports };
    if (options.synchronously) {
      const instantiateCore = (module, imports2) => new WebAssembly.Instance(module, imports2);
      return runCommand(this.#instantiate(getCoreModule, imports, instantiateCore));
    } else {
      return this.#instantiate(getCoreModule, imports).then(runCommand);
    }
  }
};

// gen/llvm-resources.js
var llvm_resources_exports = {};
__export(llvm_resources_exports, {
  filesystem: () => filesystem,
  modules: () => modules,
  totalSize: () => totalSize
});

// node_modules/nanotar/dist/index.mjs
var TAR_TYPE_FILE = 0;
var TAR_TYPE_DIR = 5;
function parseTar(data, opts) {
  const buffer = data.buffer || data;
  const files = [];
  let offset = 0;
  while (offset < buffer.byteLength - 512) {
    let name = _readString(buffer, offset, 100);
    if (name.length === 0) {
      break;
    }
    const mode = _readString(buffer, offset + 100, 8).trim();
    const uid = Number.parseInt(_readString(buffer, offset + 108, 8));
    const gid = Number.parseInt(_readString(buffer, offset + 116, 8));
    const size = _readNumber(buffer, offset + 124, 12);
    const seek = 512 + 512 * Math.trunc(size / 512) + (size % 512 ? 512 : 0);
    const mtime = _readNumber(buffer, offset + 136, 12);
    const _type = _readNumber(buffer, offset + 156, 1);
    const type = _type === TAR_TYPE_FILE ? "file" : _type === TAR_TYPE_DIR ? "directory" : _type;
    const user = _readString(buffer, offset + 265, 32);
    const group = _readString(buffer, offset + 297, 32);
    name = _sanitizePath(name);
    const meta = {
      name,
      type,
      size,
      attrs: {
        mode,
        uid,
        gid,
        mtime,
        user,
        group
      }
    };
    if (opts?.filter && !opts.filter(meta)) {
      offset += seek;
      continue;
    }
    if (opts?.metaOnly) {
      files.push(meta);
      offset += seek;
      continue;
    }
    const data2 = _type === TAR_TYPE_DIR ? void 0 : new Uint8Array(buffer, offset + 512, size);
    files.push({
      ...meta,
      data: data2,
      get text() {
        return new TextDecoder().decode(this.data);
      }
    });
    offset += seek;
  }
  return files;
}
function _sanitizePath(path) {
  let normalized = path.replace(/\\/g, "/");
  normalized = normalized.replace(/^[a-zA-Z]:\//, "");
  normalized = normalized.replace(/^\/+/, "");
  const hasLeadingDotSlash = normalized.startsWith("./");
  const parts = normalized.split("/");
  const resolved = [];
  for (const part of parts) {
    if (part === "..") {
      resolved.pop();
    } else if (part !== "." && part !== "") {
      resolved.push(part);
    }
  }
  let result = resolved.join("/");
  if (hasLeadingDotSlash && !result.startsWith("./")) {
    result = "./" + result;
  }
  if (path.endsWith("/") && !result.endsWith("/")) {
    result += "/";
  }
  return result;
}
function _readString(buffer, offset, size) {
  const view = new Uint8Array(buffer, offset, size);
  const i = view.indexOf(0);
  const td = new TextDecoder();
  return td.decode(i === -1 ? view : view.slice(0, i));
}
function _readNumber(buffer, offset, size) {
  const view = new Uint8Array(buffer, offset, size);
  let str = "";
  for (let i = 0; i < size; i++) {
    str += String.fromCodePoint(view[i]);
  }
  return Number.parseInt(str, 8);
}

// gen/llvm-resources.js
function compileWasmModule(response) {
  if (WebAssembly.compileStreaming !== void 0) {
    return WebAssembly.compileStreaming(response);
  } else {
    return WebAssembly.compile(response.arrayBuffer());
  }
}
function unpackTarFilesystem(buffer) {
  const root = {};
  for (const tarEntry of parseTar(buffer)) {
    const nameParts = tarEntry.name.split("/");
    const dirNames = nameParts.slice(0, -1);
    const fileName = nameParts[nameParts.length - 1];
    let dir = root;
    for (const dirName of dirNames)
      dir = dir[dirName];
    if (tarEntry.type === "directory") {
      dir[fileName] = {};
    } else {
      dir[fileName] = tarEntry.data;
    }
  }
  return root;
}
var modules = async (fetch3) => ({
  "llvm.core.wasm": await fetch3(new URL("./llvm.core.wasm", import.meta.url)).then(compileWasmModule),
  "llvm.core2.wasm": await fetch3(new URL("./llvm.core2.wasm", import.meta.url)).then(compileWasmModule),
  "llvm.core3.wasm": await fetch3(new URL("./llvm.core3.wasm", import.meta.url)).then(compileWasmModule),
  "llvm.core4.wasm": await fetch3(new URL("./llvm.core4.wasm", import.meta.url)).then(compileWasmModule)
});
var filesystem = async (fetch3) => ({
  "usr": await fetch3(new URL("./llvm-resources.tar", import.meta.url)).then((resp) => resp.arrayBuffer()).then(unpackTarFilesystem)
});
var totalSize = 105241285;

// gen/llvm.js
function instantiate(getCoreModule, imports, instantiateCore = WebAssembly.instantiate) {
  let dv = new DataView(new ArrayBuffer());
  const dataView = (mem) => dv.buffer === mem.buffer ? dv : dv = new DataView(mem.buffer);
  const toUint64 = (val) => BigInt.asUintN(64, BigInt(val));
  function toUint32(val) {
    return val >>> 0;
  }
  const utf8Decoder = new TextDecoder();
  const utf8Encoder = new TextEncoder();
  let utf8EncodedLen = 0;
  function utf8Encode(s, realloc, memory) {
    if (typeof s !== "string") throw new TypeError("expected a string");
    if (s.length === 0) {
      utf8EncodedLen = 0;
      return 1;
    }
    let buf = utf8Encoder.encode(s);
    let ptr = realloc(0, 0, 1, buf.length);
    new Uint8Array(memory.buffer).set(buf, ptr);
    utf8EncodedLen = buf.length;
    return ptr;
  }
  const T_FLAG = 1 << 30;
  function rscTableCreateOwn(table, rep) {
    const free2 = table[0] & ~T_FLAG;
    if (free2 === 0) {
      table.push(0);
      table.push(rep | T_FLAG);
      return (table.length >> 1) - 1;
    }
    table[0] = table[free2 << 1];
    table[free2 << 1] = 0;
    table[(free2 << 1) + 1] = rep | T_FLAG;
    return free2;
  }
  function rscTableRemove(table, handle) {
    const scope = table[handle << 1];
    const val = table[(handle << 1) + 1];
    const own = (val & T_FLAG) !== 0;
    const rep = val & ~T_FLAG;
    if (val === 0 || (scope & T_FLAG) !== 0) throw new TypeError("Invalid handle");
    table[handle << 1] = table[0] | T_FLAG;
    table[0] = handle | T_FLAG;
    return { rep, scope, own };
  }
  let curResourceBorrows = [];
  let NEXT_TASK_ID = 0n;
  function startCurrentTask(componentIdx, isAsync, entryFnName) {
    _debugLog("[startCurrentTask()] args", { componentIdx, isAsync });
    if (componentIdx === void 0 || componentIdx === null) {
      throw new Error("missing/invalid component instance index while starting task");
    }
    const tasks = ASYNC_TASKS_BY_COMPONENT_IDX.get(componentIdx);
    const nextId = ++NEXT_TASK_ID;
    const newTask = new AsyncTask({ id: nextId, componentIdx, isAsync, entryFnName });
    const newTaskMeta = { id: nextId, componentIdx, task: newTask };
    ASYNC_CURRENT_TASK_IDS.push(nextId);
    ASYNC_CURRENT_COMPONENT_IDXS.push(componentIdx);
    if (!tasks) {
      ASYNC_TASKS_BY_COMPONENT_IDX.set(componentIdx, [newTaskMeta]);
      return nextId;
    } else {
      tasks.push(newTaskMeta);
    }
    return nextId;
  }
  function endCurrentTask(componentIdx, taskId) {
    _debugLog("[endCurrentTask()] args", { componentIdx });
    componentIdx ??= ASYNC_CURRENT_COMPONENT_IDXS.at(-1);
    taskId ??= ASYNC_CURRENT_TASK_IDS.at(-1);
    if (componentIdx === void 0 || componentIdx === null) {
      throw new Error("missing/invalid component instance index while ending current task");
    }
    const tasks = ASYNC_TASKS_BY_COMPONENT_IDX.get(componentIdx);
    if (!tasks || !Array.isArray(tasks)) {
      throw new Error("missing/invalid tasks for component instance while ending task");
    }
    if (tasks.length == 0) {
      throw new Error("no current task(s) for component instance while ending task");
    }
    if (taskId) {
      const last = tasks[tasks.length - 1];
      if (last.id !== taskId) {
        throw new Error("current task does not match expected task ID");
      }
    }
    ASYNC_CURRENT_TASK_IDS.pop();
    ASYNC_CURRENT_COMPONENT_IDXS.pop();
    return tasks.pop();
  }
  const ASYNC_TASKS_BY_COMPONENT_IDX = /* @__PURE__ */ new Map();
  const ASYNC_CURRENT_TASK_IDS = [];
  const ASYNC_CURRENT_COMPONENT_IDXS = [];
  class AsyncTask {
    static State = {
      INITIAL: "initial",
      CANCELLED: "cancelled",
      CANCEL_PENDING: "cancel-pending",
      CANCEL_DELIVERED: "cancel-delivered",
      RESOLVED: "resolved"
    };
    static BlockResult = {
      CANCELLED: "block.cancelled",
      NOT_CANCELLED: "block.not-cancelled"
    };
    #id;
    #componentIdx;
    #state;
    #isAsync;
    #onResolve = null;
    #returnedResults = null;
    #entryFnName = null;
    cancelled = false;
    requested = false;
    alwaysTaskReturn = false;
    returnCalls = 0;
    storage = [0, 0];
    borrowedHandles = {};
    awaitableResume = null;
    awaitableCancel = null;
    constructor(opts) {
      if (opts?.id === void 0) {
        throw new TypeError("missing task ID during task creation");
      }
      this.#id = opts.id;
      if (opts?.componentIdx === void 0) {
        throw new TypeError("missing component id during task creation");
      }
      this.#componentIdx = opts.componentIdx;
      this.#state = AsyncTask.State.INITIAL;
      this.#isAsync = opts?.isAsync ?? false;
      this.#entryFnName = opts.entryFnName;
      this.#onResolve = (results) => {
        this.#returnedResults = results;
      };
    }
    taskState() {
      return this.#state.slice();
    }
    id() {
      return this.#id;
    }
    componentIdx() {
      return this.#componentIdx;
    }
    isAsync() {
      return this.#isAsync;
    }
    getEntryFnName() {
      return this.#entryFnName;
    }
    takeResults() {
      const results = this.#returnedResults;
      this.#returnedResults = null;
      return results;
    }
    mayEnter(task) {
      const cstate = getOrCreateAsyncState(this.#componentIdx);
      if (!cstate.backpressure) {
        _debugLog("[AsyncTask#mayEnter()] disallowed due to backpressure", { taskID: this.#id });
        return false;
      }
      if (!cstate.callingSyncImport()) {
        _debugLog("[AsyncTask#mayEnter()] disallowed due to sync import call", { taskID: this.#id });
        return false;
      }
      const callingSyncExportWithSyncPending = cstate.callingSyncExport && !task.isAsync;
      if (!callingSyncExportWithSyncPending) {
        _debugLog("[AsyncTask#mayEnter()] disallowed due to sync export w/ sync pending", { taskID: this.#id });
        return false;
      }
      return true;
    }
    async enter() {
      _debugLog("[AsyncTask#enter()] args", { taskID: this.#id });
      const cstate = getOrCreateAsyncState(this.#componentIdx);
      let mayNotEnter = !this.mayEnter(this);
      const componentHasPendingTasks = cstate.pendingTasks > 0;
      if (mayNotEnter || componentHasPendingTasks) {
        throw new Error("in enter()");
        cstate.pendingTasks.set(this.#id, new Awaitable(new Promise()));
        const blockResult = await this.onBlock(awaitable);
        if (blockResult) {
          const pendingTask = cstate.pendingTasks.get(this.#id);
          if (!pendingTask) {
            throw new Error("pending task [" + this.#id + "] not found for component instance");
          }
          cstate.pendingTasks.remove(this.#id);
          this.#onResolve([]);
          return false;
        }
        mayNotEnter = !this.mayEnter(this);
        if (!mayNotEnter || !cstate.startPendingTask) {
          throw new Error("invalid component entrance/pending task resolution");
        }
        cstate.startPendingTask = false;
      }
      if (!this.isAsync) {
        cstate.callingSyncExport = true;
      }
      return true;
    }
    async waitForEvent(opts) {
      const { waitableSetRep, isAsync } = opts;
      _debugLog("[AsyncTask#waitForEvent()] args", { taskID: this.#id, waitableSetRep, isAsync });
      if (this.#isAsync !== isAsync) {
        throw new Error("async waitForEvent called on non-async task");
      }
      if (this.status === AsyncTask.State.CANCEL_PENDING) {
        this.#state = AsyncTask.State.CANCEL_DELIVERED;
        return {
          code: ASYNC_EVENT_CODE.TASK_CANCELLED
        };
      }
      const state = getOrCreateAsyncState(this.#componentIdx);
      const waitableSet = state.waitableSets.get(waitableSetRep);
      if (!waitableSet) {
        throw new Error("missing/invalid waitable set");
      }
      waitableSet.numWaiting += 1;
      let event = null;
      while (event == null) {
        const awaitable2 = new Awaitable(waitableSet.getPendingEvent());
        const waited = await this.blockOn({ awaitable: awaitable2, isAsync, isCancellable: true });
        if (waited) {
          if (this.#state !== AsyncTask.State.INITIAL) {
            throw new Error("task should be in initial state found [" + this.#state + "]");
          }
          this.#state = AsyncTask.State.CANCELLED;
          return {
            code: ASYNC_EVENT_CODE.TASK_CANCELLED
          };
        }
        event = waitableSet.poll();
      }
      waitableSet.numWaiting -= 1;
      return event;
    }
    waitForEventSync(opts) {
      throw new Error("AsyncTask#yieldSync() not implemented");
    }
    async pollForEvent(opts) {
      const { waitableSetRep, isAsync } = opts;
      _debugLog("[AsyncTask#pollForEvent()] args", { taskID: this.#id, waitableSetRep, isAsync });
      if (this.#isAsync !== isAsync) {
        throw new Error("async pollForEvent called on non-async task");
      }
      throw new Error("AsyncTask#pollForEvent() not implemented");
    }
    pollForEventSync(opts) {
      throw new Error("AsyncTask#yieldSync() not implemented");
    }
    async blockOn(opts) {
      const { awaitable: awaitable2, isCancellable, forCallback } = opts;
      _debugLog("[AsyncTask#blockOn()] args", { taskID: this.#id, awaitable: awaitable2, isCancellable, forCallback });
      if (awaitable2.resolved() && !ASYNC_DETERMINISM && _coinFlip()) {
        return AsyncTask.BlockResult.NOT_CANCELLED;
      }
      const cstate = getOrCreateAsyncState(this.#componentIdx);
      if (forCallback) {
        cstate.exclusiveRelease();
      }
      let cancelled = await this.onBlock(awaitable2);
      if (cancelled === AsyncTask.BlockResult.CANCELLED && !isCancellable) {
        const secondCancel = await this.onBlock(awaitable2);
        if (secondCancel !== AsyncTask.BlockResult.NOT_CANCELLED) {
          throw new Error("uncancellable task was canceled despite second onBlock()");
        }
      }
      if (forCallback) {
        const acquired = new Awaitable(cstate.exclusiveLock());
        cancelled = await this.onBlock(acquired);
        if (cancelled === AsyncTask.BlockResult.CANCELLED) {
          const secondCancel = await this.onBlock(acquired);
          if (secondCancel !== AsyncTask.BlockResult.NOT_CANCELLED) {
            throw new Error("uncancellable callback task was canceled despite second onBlock()");
          }
        }
      }
      if (cancelled === AsyncTask.BlockResult.CANCELLED) {
        if (this.#state !== AsyncTask.State.INITIAL) {
          throw new Error("cancelled task is not at initial state");
        }
        if (isCancellable) {
          this.#state = AsyncTask.State.CANCELLED;
          return AsyncTask.BlockResult.CANCELLED;
        } else {
          this.#state = AsyncTask.State.CANCEL_PENDING;
          return AsyncTask.BlockResult.NOT_CANCELLED;
        }
      }
      return AsyncTask.BlockResult.NOT_CANCELLED;
    }
    async onBlock(awaitable2) {
      _debugLog("[AsyncTask#onBlock()] args", { taskID: this.#id, awaitable: awaitable2 });
      if (!(awaitable2 instanceof Awaitable)) {
        throw new Error("invalid awaitable during onBlock");
      }
      const { promise: promise2, resolve: resolve2, reject: reject2 } = Promise.withResolvers();
      this.awaitableResume = () => {
        _debugLog("[AsyncTask] resuming after onBlock", { taskID: this.#id });
        resolve2();
      };
      this.awaitableCancel = (err) => {
        _debugLog("[AsyncTask] rejecting after onBlock", { taskID: this.#id, err });
        reject2(err);
      };
      const state = getOrCreateAsyncState(this.#componentIdx);
      state.parkTaskOnAwaitable({ awaitable: awaitable2, task: this });
      try {
        await promise2;
        return AsyncTask.BlockResult.NOT_CANCELLED;
      } catch (err) {
        return AsyncTask.BlockResult.CANCELLED;
      }
    }
    // NOTE: this should likely be moved to a SubTask class
    async asyncOnBlock(awaitable2) {
      _debugLog("[AsyncTask#asyncOnBlock()] args", { taskID: this.#id, awaitable: awaitable2 });
      if (!(awaitable2 instanceof Awaitable)) {
        throw new Error("invalid awaitable during onBlock");
      }
      throw new Error("AsyncTask#asyncOnBlock() not yet implemented");
    }
    async yield(opts) {
      const { isCancellable, forCallback } = opts;
      _debugLog("[AsyncTask#yield()] args", { taskID: this.#id, isCancellable, forCallback });
      if (isCancellable && this.status === AsyncTask.State.CANCEL_PENDING) {
        this.#state = AsyncTask.State.CANCELLED;
        return {
          code: ASYNC_EVENT_CODE.TASK_CANCELLED,
          payload: [0, 0]
        };
      }
      const blockResult = await this.blockOn({
        awaitable: new Awaitable(new Promise((resolve2) => setTimeout(resolve2, 0))),
        isCancellable,
        forCallback
      });
      if (blockResult === AsyncTask.BlockResult.CANCELLED) {
        if (this.#state !== AsyncTask.State.INITIAL) {
          throw new Error("task should be in initial state found [" + this.#state + "]");
        }
        this.#state = AsyncTask.State.CANCELLED;
        return {
          code: ASYNC_EVENT_CODE.TASK_CANCELLED,
          payload: [0, 0]
        };
      }
      return {
        code: ASYNC_EVENT_CODE.NONE,
        payload: [0, 0]
      };
    }
    yieldSync(opts) {
      throw new Error("AsyncTask#yieldSync() not implemented");
    }
    cancel() {
      _debugLog("[AsyncTask#cancel()] args", {});
      if (!this.taskState() !== AsyncTask.State.CANCEL_DELIVERED) {
        throw new Error("invalid task state for cancellation");
      }
      if (this.borrowedHandles.length > 0) {
        throw new Error("task still has borrow handles");
      }
      this.#onResolve([]);
      this.#state = AsyncTask.State.RESOLVED;
    }
    resolve(result) {
      if (this.#state === AsyncTask.State.RESOLVED) {
        throw new Error("task is already resolved");
      }
      if (this.borrowedHandles.length > 0) {
        throw new Error("task still has borrow handles");
      }
      this.#onResolve(result);
      this.#state = AsyncTask.State.RESOLVED;
    }
    exit() {
      if (this.#state !== AsyncTask.State.RESOLVED) {
        throw new Error("task exited without resolution");
      }
      if (this.borrowedHandles > 0) {
        throw new Error("task exited without clearing borrowed handles");
      }
      const state = getOrCreateAsyncState(this.#componentIdx);
      if (!state) {
        throw new Error("missing async state for component [" + this.#componentIdx + "]");
      }
      if (!this.#isAsync && !state.inSyncExportCall) {
        throw new Error("sync task must be run from components known to be in a sync export call");
      }
      state.inSyncExportCall = false;
      this.startPendingTask();
    }
    startPendingTask(opts) {
    }
  }
  function unpackCallbackResult(result) {
    _debugLog("[unpackCallbackResult()] args", { result });
    if (!_typeCheckValidI32(result)) {
      throw new Error("invalid callback return value [" + result + "], not a valid i32");
    }
    const eventCode = result & 15;
    if (eventCode < 0 || eventCode > 3) {
      throw new Error("invalid async return value [" + eventCode + "], outside callback code range");
    }
    if (result < 0 || result >= 2 ** 32) {
      throw new Error("invalid callback result");
    }
    const waitableSetIdx = result >> 4;
    return [eventCode, waitableSetIdx];
  }
  const ASYNC_STATE = /* @__PURE__ */ new Map();
  function getOrCreateAsyncState(componentIdx, init) {
    if (!ASYNC_STATE.has(componentIdx)) {
      ASYNC_STATE.set(componentIdx, new ComponentAsyncState());
    }
    return ASYNC_STATE.get(componentIdx);
  }
  class ComponentAsyncState {
    #callingAsyncImport = false;
    #syncImportWait = Promise.withResolvers();
    #lock = null;
    mayLeave = false;
    waitableSets = new RepTable();
    waitables = new RepTable();
    #parkedTasks = /* @__PURE__ */ new Map();
    callingSyncImport(val) {
      if (val === void 0) {
        return this.#callingAsyncImport;
      }
      if (typeof val !== "boolean") {
        throw new TypeError("invalid setting for async import");
      }
      const prev = this.#callingAsyncImport;
      this.#callingAsyncImport = val;
      if (prev === true && this.#callingAsyncImport === false) {
        this.#notifySyncImportEnd();
      }
    }
    #notifySyncImportEnd() {
      const existing = this.#syncImportWait;
      this.#syncImportWait = Promise.withResolvers();
      existing.resolve();
    }
    async waitForSyncImportCallEnd() {
      await this.#syncImportWait.promise;
    }
    parkTaskOnAwaitable(args) {
      if (!args.awaitable) {
        throw new TypeError("missing awaitable when trying to park");
      }
      if (!args.task) {
        throw new TypeError("missing task when trying to park");
      }
      const { awaitable: awaitable2, task } = args;
      let taskList = this.#parkedTasks.get(awaitable2.id());
      if (!taskList) {
        taskList = [];
        this.#parkedTasks.set(awaitable2.id(), taskList);
      }
      taskList.push(task);
      this.wakeNextTaskForAwaitable(awaitable2);
    }
    wakeNextTaskForAwaitable(awaitable2) {
      if (!awaitable2) {
        throw new TypeError("missing awaitable when waking next task");
      }
      const awaitableID = awaitable2.id();
      const taskList = this.#parkedTasks.get(awaitableID);
      if (!taskList || taskList.length === 0) {
        _debugLog("[ComponentAsyncState] no tasks waiting for awaitable", { awaitableID: awaitable2.id() });
        return;
      }
      let task = taskList.shift();
      if (!task) {
        throw new Error("no task in parked list despite previous check");
      }
      if (!task.awaitableResume) {
        throw new Error("task ready due to awaitable is missing resume", { taskID: task.id(), awaitableID });
      }
      task.awaitableResume();
    }
    async exclusiveLock() {
      if (this.#lock === null) {
        this.#lock = { ticket: 0n };
      }
      const ticket = ++this.#lock.ticket;
      _debugLog("[ComponentAsyncState#exclusiveLock()] locking", {
        currentTicket: ticket - 1n,
        ticket
      });
      let finishedTicket;
      while (this.#lock.promise) {
        finishedTicket = await this.#lock.promise;
        if (finishedTicket === ticket - 1n) {
          break;
        }
      }
      const { promise: promise2, resolve: resolve2 } = Promise.withResolvers();
      this.#lock = {
        ticket,
        promise: promise2,
        resolve: resolve2
      };
      return this.#lock.promise;
    }
    exclusiveRelease() {
      _debugLog("[ComponentAsyncState#exclusiveRelease()] releasing", {
        currentTicket: this.#lock === null ? "none" : this.#lock.ticket
      });
      if (this.#lock === null) {
        return;
      }
      const existingLock = this.#lock;
      this.#lock = null;
      existingLock.resolve(existingLock.ticket);
    }
    isExclusivelyLocked() {
      return this.#lock !== null;
    }
  }
  if (!Promise.withResolvers) {
    Promise.withResolvers = () => {
      let resolve2;
      let reject2;
      const promise2 = new Promise((res, rej) => {
        resolve2 = res;
        reject2 = rej;
      });
      return { promise: promise2, resolve: resolve2, reject: reject2 };
    };
  }
  const _debugLog = (...args) => {
    if (!globalThis?.process?.env?.JCO_DEBUG) {
      return;
    }
    console.debug(...args);
  };
  const ASYNC_DETERMINISM = "random";
  const _coinFlip = () => {
    return Math.random() > 0.5;
  };
  const I32_MAX = 2147483647;
  const I32_MIN = -2147483648;
  const _typeCheckValidI32 = (n) => typeof n === "number" && n >= I32_MIN && n <= I32_MAX;
  const isNode = typeof process !== "undefined" && process.versions && process.versions.node;
  let _fs;
  async function fetchCompile(url) {
    if (isNode) {
      _fs = _fs || await import("node:fs/promises");
      return WebAssembly.compile(await _fs.readFile(url));
    }
    return fetch(url).then(WebAssembly.compileStreaming);
  }
  const symbolCabiDispose = Symbol.for("cabiDispose");
  const symbolRscHandle = Symbol("handle");
  const symbolRscRep = Symbol.for("cabiRep");
  const symbolDispose = Symbol.dispose || Symbol.for("dispose");
  const handleTables = [];
  class ComponentError extends Error {
    constructor(value) {
      const enumerable = typeof value !== "string";
      super(enumerable ? `${String(value)} (see error.payload)` : value);
      Object.defineProperty(this, "payload", { value, enumerable });
    }
  }
  function getErrorPayload(e) {
    if (e && hasOwnProperty.call(e, "payload")) return e.payload;
    if (e instanceof Error) throw e;
    return e;
  }
  class RepTable {
    #data = [0, null];
    insert(val) {
      _debugLog("[RepTable#insert()] args", { val });
      const freeIdx = this.#data[0];
      if (freeIdx === 0) {
        this.#data.push(val);
        this.#data.push(null);
        return (this.#data.length >> 1) - 1;
      }
      this.#data[0] = this.#data[freeIdx];
      const newFreeIdx = freeIdx << 1;
      this.#data[newFreeIdx] = val;
      this.#data[newFreeIdx + 1] = null;
      return free;
    }
    get(rep) {
      _debugLog("[RepTable#insert()] args", { rep });
      const baseIdx = idx << 1;
      const val = this.#data[baseIdx];
      return val;
    }
    contains(rep) {
      _debugLog("[RepTable#insert()] args", { rep });
      const baseIdx = idx << 1;
      return !!this.#data[baseIdx];
    }
    remove(rep) {
      _debugLog("[RepTable#insert()] args", { idx });
      if (this.#data.length === 2) {
        throw new Error("invalid");
      }
      const baseIdx = idx << 1;
      const val = this.#data[baseIdx];
      if (val === 0) {
        throw new Error("invalid resource rep (cannot be 0)");
      }
      this.#data[baseIdx] = this.#data[0];
      this.#data[0] = idx;
      return val;
    }
    clear() {
      this.#data = [0, null];
    }
  }
  const hasOwnProperty = Object.prototype.hasOwnProperty;
  if (!getCoreModule) getCoreModule = (name) => fetchCompile(new URL(`./${name}`, import.meta.url));
  const module0 = getCoreModule("llvm.core.wasm");
  const module1 = getCoreModule("llvm.core2.wasm");
  const module2 = getCoreModule("llvm.core3.wasm");
  const module3 = getCoreModule("llvm.core4.wasm");
  const { cli, fs, io, monotonicClock, random, wallClock } = imports.runtime;
  const {
    TerminalInput: TerminalInput2,
    TerminalOutput: TerminalOutput2,
    exit,
    getArguments,
    getEnvironment,
    getStderr,
    getStdin,
    getStdout,
    getTerminalStderr,
    getTerminalStdin,
    getTerminalStdout
  } = cli;
  const {
    Descriptor: Descriptor2,
    DirectoryEntryStream: DirectoryEntryStream2,
    filesystemErrorCode,
    getDirectories
  } = fs;
  const {
    Error: Error$1,
    InputStream: InputStream2,
    OutputStream: OutputStream2,
    Pollable,
    poll
  } = io;
  const {
    now,
    subscribeDuration,
    subscribeInstant
  } = monotonicClock;
  const { getRandomBytes } = random;
  const { now: now$1 } = wallClock;
  let gen = (function* init() {
    let exports0;
    let exports1;
    function trampoline0() {
      _debugLog('[iface="wasi:clocks/monotonic-clock@0.2.3", function="now"] [Instruction::CallInterface] (async? sync, @ enter)');
      const _interface_call_currentTaskID = startCurrentTask(0, false, "now");
      const ret = now();
      _debugLog('[iface="wasi:clocks/monotonic-clock@0.2.3", function="now"] [Instruction::CallInterface] (sync, @ post-call)');
      endCurrentTask(0);
      _debugLog('[iface="wasi:clocks/monotonic-clock@0.2.3", function="now"][Instruction::Return]', {
        funcName: "now",
        paramCount: 1,
        postReturn: false
      });
      return toUint64(ret);
    }
    const handleTable1 = [T_FLAG, 0];
    const captureTable1 = /* @__PURE__ */ new Map();
    let captureCnt1 = 0;
    handleTables[1] = handleTable1;
    function trampoline6(arg0) {
      _debugLog('[iface="wasi:clocks/monotonic-clock@0.2.3", function="subscribe-duration"] [Instruction::CallInterface] (async? sync, @ enter)');
      const _interface_call_currentTaskID = startCurrentTask(0, false, "subscribe-duration");
      const ret = subscribeDuration(BigInt.asUintN(64, arg0));
      _debugLog('[iface="wasi:clocks/monotonic-clock@0.2.3", function="subscribe-duration"] [Instruction::CallInterface] (sync, @ post-call)');
      endCurrentTask(0);
      if (!(ret instanceof Pollable)) {
        throw new TypeError('Resource error: Not a valid "Pollable" resource.');
      }
      var handle0 = ret[symbolRscHandle];
      if (!handle0) {
        const rep = ret[symbolRscRep] || ++captureCnt1;
        captureTable1.set(rep, ret);
        handle0 = rscTableCreateOwn(handleTable1, rep);
      }
      _debugLog('[iface="wasi:clocks/monotonic-clock@0.2.3", function="subscribe-duration"][Instruction::Return]', {
        funcName: "subscribe-duration",
        paramCount: 1,
        postReturn: false
      });
      return handle0;
    }
    function trampoline7(arg0) {
      _debugLog('[iface="wasi:clocks/monotonic-clock@0.2.3", function="subscribe-instant"] [Instruction::CallInterface] (async? sync, @ enter)');
      const _interface_call_currentTaskID = startCurrentTask(0, false, "subscribe-instant");
      const ret = subscribeInstant(BigInt.asUintN(64, arg0));
      _debugLog('[iface="wasi:clocks/monotonic-clock@0.2.3", function="subscribe-instant"] [Instruction::CallInterface] (sync, @ post-call)');
      endCurrentTask(0);
      if (!(ret instanceof Pollable)) {
        throw new TypeError('Resource error: Not a valid "Pollable" resource.');
      }
      var handle0 = ret[symbolRscHandle];
      if (!handle0) {
        const rep = ret[symbolRscRep] || ++captureCnt1;
        captureTable1.set(rep, ret);
        handle0 = rscTableCreateOwn(handleTable1, rep);
      }
      _debugLog('[iface="wasi:clocks/monotonic-clock@0.2.3", function="subscribe-instant"][Instruction::Return]', {
        funcName: "subscribe-instant",
        paramCount: 1,
        postReturn: false
      });
      return handle0;
    }
    const handleTable3 = [T_FLAG, 0];
    const captureTable3 = /* @__PURE__ */ new Map();
    let captureCnt3 = 0;
    handleTables[3] = handleTable3;
    function trampoline8(arg0) {
      var handle1 = arg0;
      var rep2 = handleTable3[(handle1 << 1) + 1] & ~T_FLAG;
      var rsc0 = captureTable3.get(rep2);
      if (!rsc0) {
        rsc0 = Object.create(OutputStream2.prototype);
        Object.defineProperty(rsc0, symbolRscHandle, { writable: true, value: handle1 });
        Object.defineProperty(rsc0, symbolRscRep, { writable: true, value: rep2 });
      }
      curResourceBorrows.push(rsc0);
      _debugLog('[iface="wasi:io/streams@0.2.3", function="[method]output-stream.subscribe"] [Instruction::CallInterface] (async? sync, @ enter)');
      const _interface_call_currentTaskID = startCurrentTask(0, false, "[method]output-stream.subscribe");
      const ret = rsc0.subscribe();
      _debugLog('[iface="wasi:io/streams@0.2.3", function="[method]output-stream.subscribe"] [Instruction::CallInterface] (sync, @ post-call)');
      for (const rsc of curResourceBorrows) {
        rsc[symbolRscHandle] = void 0;
      }
      curResourceBorrows = [];
      endCurrentTask(0);
      if (!(ret instanceof Pollable)) {
        throw new TypeError('Resource error: Not a valid "Pollable" resource.');
      }
      var handle3 = ret[symbolRscHandle];
      if (!handle3) {
        const rep = ret[symbolRscRep] || ++captureCnt1;
        captureTable1.set(rep, ret);
        handle3 = rscTableCreateOwn(handleTable1, rep);
      }
      _debugLog('[iface="wasi:io/streams@0.2.3", function="[method]output-stream.subscribe"][Instruction::Return]', {
        funcName: "[method]output-stream.subscribe",
        paramCount: 1,
        postReturn: false
      });
      return handle3;
    }
    const handleTable2 = [T_FLAG, 0];
    const captureTable2 = /* @__PURE__ */ new Map();
    let captureCnt2 = 0;
    handleTables[2] = handleTable2;
    function trampoline9(arg0) {
      var handle1 = arg0;
      var rep2 = handleTable2[(handle1 << 1) + 1] & ~T_FLAG;
      var rsc0 = captureTable2.get(rep2);
      if (!rsc0) {
        rsc0 = Object.create(InputStream2.prototype);
        Object.defineProperty(rsc0, symbolRscHandle, { writable: true, value: handle1 });
        Object.defineProperty(rsc0, symbolRscRep, { writable: true, value: rep2 });
      }
      curResourceBorrows.push(rsc0);
      _debugLog('[iface="wasi:io/streams@0.2.3", function="[method]input-stream.subscribe"] [Instruction::CallInterface] (async? sync, @ enter)');
      const _interface_call_currentTaskID = startCurrentTask(0, false, "[method]input-stream.subscribe");
      const ret = rsc0.subscribe();
      _debugLog('[iface="wasi:io/streams@0.2.3", function="[method]input-stream.subscribe"] [Instruction::CallInterface] (sync, @ post-call)');
      for (const rsc of curResourceBorrows) {
        rsc[symbolRscHandle] = void 0;
      }
      curResourceBorrows = [];
      endCurrentTask(0);
      if (!(ret instanceof Pollable)) {
        throw new TypeError('Resource error: Not a valid "Pollable" resource.');
      }
      var handle3 = ret[symbolRscHandle];
      if (!handle3) {
        const rep = ret[symbolRscRep] || ++captureCnt1;
        captureTable1.set(rep, ret);
        handle3 = rscTableCreateOwn(handleTable1, rep);
      }
      _debugLog('[iface="wasi:io/streams@0.2.3", function="[method]input-stream.subscribe"][Instruction::Return]', {
        funcName: "[method]input-stream.subscribe",
        paramCount: 1,
        postReturn: false
      });
      return handle3;
    }
    function trampoline11() {
      _debugLog('[iface="wasi:cli/stderr@0.2.3", function="get-stderr"] [Instruction::CallInterface] (async? sync, @ enter)');
      const _interface_call_currentTaskID = startCurrentTask(0, false, "get-stderr");
      const ret = getStderr();
      _debugLog('[iface="wasi:cli/stderr@0.2.3", function="get-stderr"] [Instruction::CallInterface] (sync, @ post-call)');
      endCurrentTask(0);
      if (!(ret instanceof OutputStream2)) {
        throw new TypeError('Resource error: Not a valid "OutputStream" resource.');
      }
      var handle0 = ret[symbolRscHandle];
      if (!handle0) {
        const rep = ret[symbolRscRep] || ++captureCnt3;
        captureTable3.set(rep, ret);
        handle0 = rscTableCreateOwn(handleTable3, rep);
      }
      _debugLog('[iface="wasi:cli/stderr@0.2.3", function="get-stderr"][Instruction::Return]', {
        funcName: "get-stderr",
        paramCount: 1,
        postReturn: false
      });
      return handle0;
    }
    function trampoline14() {
      _debugLog('[iface="wasi:cli/stdin@0.2.3", function="get-stdin"] [Instruction::CallInterface] (async? sync, @ enter)');
      const _interface_call_currentTaskID = startCurrentTask(0, false, "get-stdin");
      const ret = getStdin();
      _debugLog('[iface="wasi:cli/stdin@0.2.3", function="get-stdin"] [Instruction::CallInterface] (sync, @ post-call)');
      endCurrentTask(0);
      if (!(ret instanceof InputStream2)) {
        throw new TypeError('Resource error: Not a valid "InputStream" resource.');
      }
      var handle0 = ret[symbolRscHandle];
      if (!handle0) {
        const rep = ret[symbolRscRep] || ++captureCnt2;
        captureTable2.set(rep, ret);
        handle0 = rscTableCreateOwn(handleTable2, rep);
      }
      _debugLog('[iface="wasi:cli/stdin@0.2.3", function="get-stdin"][Instruction::Return]', {
        funcName: "get-stdin",
        paramCount: 1,
        postReturn: false
      });
      return handle0;
    }
    function trampoline15() {
      _debugLog('[iface="wasi:cli/stdout@0.2.3", function="get-stdout"] [Instruction::CallInterface] (async? sync, @ enter)');
      const _interface_call_currentTaskID = startCurrentTask(0, false, "get-stdout");
      const ret = getStdout();
      _debugLog('[iface="wasi:cli/stdout@0.2.3", function="get-stdout"] [Instruction::CallInterface] (sync, @ post-call)');
      endCurrentTask(0);
      if (!(ret instanceof OutputStream2)) {
        throw new TypeError('Resource error: Not a valid "OutputStream" resource.');
      }
      var handle0 = ret[symbolRscHandle];
      if (!handle0) {
        const rep = ret[symbolRscRep] || ++captureCnt3;
        captureTable3.set(rep, ret);
        handle0 = rscTableCreateOwn(handleTable3, rep);
      }
      _debugLog('[iface="wasi:cli/stdout@0.2.3", function="get-stdout"][Instruction::Return]', {
        funcName: "get-stdout",
        paramCount: 1,
        postReturn: false
      });
      return handle0;
    }
    function trampoline16(arg0) {
      let variant0;
      switch (arg0) {
        case 0: {
          variant0 = {
            tag: "ok",
            val: void 0
          };
          break;
        }
        case 1: {
          variant0 = {
            tag: "err",
            val: void 0
          };
          break;
        }
        default: {
          throw new TypeError("invalid variant discriminant for expected");
        }
      }
      _debugLog('[iface="wasi:cli/exit@0.2.3", function="exit"] [Instruction::CallInterface] (async? sync, @ enter)');
      const _interface_call_currentTaskID = startCurrentTask(0, false, "exit");
      exit(variant0);
      _debugLog('[iface="wasi:cli/exit@0.2.3", function="exit"] [Instruction::CallInterface] (sync, @ post-call)');
      endCurrentTask(0);
      _debugLog('[iface="wasi:cli/exit@0.2.3", function="exit"][Instruction::Return]', {
        funcName: "exit",
        paramCount: 0,
        postReturn: false
      });
    }
    let exports2;
    let memory0;
    let realloc0;
    function trampoline17(arg0) {
      _debugLog('[iface="wasi:cli/environment@0.2.3", function="get-arguments"] [Instruction::CallInterface] (async? sync, @ enter)');
      const _interface_call_currentTaskID = startCurrentTask(0, false, "get-arguments");
      const ret = getArguments();
      _debugLog('[iface="wasi:cli/environment@0.2.3", function="get-arguments"] [Instruction::CallInterface] (sync, @ post-call)');
      endCurrentTask(0);
      var vec1 = ret;
      var len1 = vec1.length;
      var result1 = realloc0(0, 0, 4, len1 * 8);
      for (let i = 0; i < vec1.length; i++) {
        const e = vec1[i];
        const base = result1 + i * 8;
        var ptr0 = utf8Encode(e, realloc0, memory0);
        var len0 = utf8EncodedLen;
        dataView(memory0).setUint32(base + 4, len0, true);
        dataView(memory0).setUint32(base + 0, ptr0, true);
      }
      dataView(memory0).setUint32(arg0 + 4, len1, true);
      dataView(memory0).setUint32(arg0 + 0, result1, true);
      _debugLog('[iface="wasi:cli/environment@0.2.3", function="get-arguments"][Instruction::Return]', {
        funcName: "get-arguments",
        paramCount: 0,
        postReturn: false
      });
    }
    function trampoline18(arg0) {
      _debugLog('[iface="wasi:cli/environment@0.2.3", function="get-environment"] [Instruction::CallInterface] (async? sync, @ enter)');
      const _interface_call_currentTaskID = startCurrentTask(0, false, "get-environment");
      const ret = getEnvironment();
      _debugLog('[iface="wasi:cli/environment@0.2.3", function="get-environment"] [Instruction::CallInterface] (sync, @ post-call)');
      endCurrentTask(0);
      var vec3 = ret;
      var len3 = vec3.length;
      var result3 = realloc0(0, 0, 4, len3 * 16);
      for (let i = 0; i < vec3.length; i++) {
        const e = vec3[i];
        const base = result3 + i * 16;
        var [tuple0_0, tuple0_1] = e;
        var ptr1 = utf8Encode(tuple0_0, realloc0, memory0);
        var len1 = utf8EncodedLen;
        dataView(memory0).setUint32(base + 4, len1, true);
        dataView(memory0).setUint32(base + 0, ptr1, true);
        var ptr2 = utf8Encode(tuple0_1, realloc0, memory0);
        var len2 = utf8EncodedLen;
        dataView(memory0).setUint32(base + 12, len2, true);
        dataView(memory0).setUint32(base + 8, ptr2, true);
      }
      dataView(memory0).setUint32(arg0 + 4, len3, true);
      dataView(memory0).setUint32(arg0 + 0, result3, true);
      _debugLog('[iface="wasi:cli/environment@0.2.3", function="get-environment"][Instruction::Return]', {
        funcName: "get-environment",
        paramCount: 0,
        postReturn: false
      });
    }
    function trampoline19(arg0) {
      _debugLog('[iface="wasi:clocks/wall-clock@0.2.3", function="now"] [Instruction::CallInterface] (async? sync, @ enter)');
      const _interface_call_currentTaskID = startCurrentTask(0, false, "now");
      const ret = now$1();
      _debugLog('[iface="wasi:clocks/wall-clock@0.2.3", function="now"] [Instruction::CallInterface] (sync, @ post-call)');
      endCurrentTask(0);
      var { seconds: v0_0, nanoseconds: v0_1 } = ret;
      dataView(memory0).setBigInt64(arg0 + 0, toUint64(v0_0), true);
      dataView(memory0).setInt32(arg0 + 8, toUint32(v0_1), true);
      _debugLog('[iface="wasi:clocks/wall-clock@0.2.3", function="now"][Instruction::Return]', {
        funcName: "now",
        paramCount: 0,
        postReturn: false
      });
    }
    const handleTable7 = [T_FLAG, 0];
    const captureTable7 = /* @__PURE__ */ new Map();
    let captureCnt7 = 0;
    handleTables[7] = handleTable7;
    function trampoline20(arg0, arg1) {
      var handle1 = arg0;
      var rep2 = handleTable7[(handle1 << 1) + 1] & ~T_FLAG;
      var rsc0 = captureTable7.get(rep2);
      if (!rsc0) {
        rsc0 = Object.create(Descriptor2.prototype);
        Object.defineProperty(rsc0, symbolRscHandle, { writable: true, value: handle1 });
        Object.defineProperty(rsc0, symbolRscRep, { writable: true, value: rep2 });
      }
      curResourceBorrows.push(rsc0);
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.get-flags"] [Instruction::CallInterface] (async? sync, @ enter)');
      const _interface_call_currentTaskID = startCurrentTask(0, false, "[method]descriptor.get-flags");
      let ret;
      try {
        ret = { tag: "ok", val: rsc0.getFlags() };
      } catch (e) {
        ret = { tag: "err", val: getErrorPayload(e) };
      }
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.get-flags"] [Instruction::CallInterface] (sync, @ post-call)');
      for (const rsc of curResourceBorrows) {
        rsc[symbolRscHandle] = void 0;
      }
      curResourceBorrows = [];
      endCurrentTask(0);
      var variant5 = ret;
      switch (variant5.tag) {
        case "ok": {
          const e = variant5.val;
          dataView(memory0).setInt8(arg1 + 0, 0, true);
          let flags3 = 0;
          if (typeof e === "object" && e !== null) {
            flags3 = Boolean(e.read) << 0 | Boolean(e.write) << 1 | Boolean(e.fileIntegritySync) << 2 | Boolean(e.dataIntegritySync) << 3 | Boolean(e.requestedWriteSync) << 4 | Boolean(e.mutateDirectory) << 5;
          } else if (e !== null && e !== void 0) {
            throw new TypeError("only an object, undefined or null can be converted to flags");
          }
          dataView(memory0).setInt8(arg1 + 1, flags3, true);
          break;
        }
        case "err": {
          const e = variant5.val;
          dataView(memory0).setInt8(arg1 + 0, 1, true);
          var val4 = e;
          let enum4;
          switch (val4) {
            case "access": {
              enum4 = 0;
              break;
            }
            case "would-block": {
              enum4 = 1;
              break;
            }
            case "already": {
              enum4 = 2;
              break;
            }
            case "bad-descriptor": {
              enum4 = 3;
              break;
            }
            case "busy": {
              enum4 = 4;
              break;
            }
            case "deadlock": {
              enum4 = 5;
              break;
            }
            case "quota": {
              enum4 = 6;
              break;
            }
            case "exist": {
              enum4 = 7;
              break;
            }
            case "file-too-large": {
              enum4 = 8;
              break;
            }
            case "illegal-byte-sequence": {
              enum4 = 9;
              break;
            }
            case "in-progress": {
              enum4 = 10;
              break;
            }
            case "interrupted": {
              enum4 = 11;
              break;
            }
            case "invalid": {
              enum4 = 12;
              break;
            }
            case "io": {
              enum4 = 13;
              break;
            }
            case "is-directory": {
              enum4 = 14;
              break;
            }
            case "loop": {
              enum4 = 15;
              break;
            }
            case "too-many-links": {
              enum4 = 16;
              break;
            }
            case "message-size": {
              enum4 = 17;
              break;
            }
            case "name-too-long": {
              enum4 = 18;
              break;
            }
            case "no-device": {
              enum4 = 19;
              break;
            }
            case "no-entry": {
              enum4 = 20;
              break;
            }
            case "no-lock": {
              enum4 = 21;
              break;
            }
            case "insufficient-memory": {
              enum4 = 22;
              break;
            }
            case "insufficient-space": {
              enum4 = 23;
              break;
            }
            case "not-directory": {
              enum4 = 24;
              break;
            }
            case "not-empty": {
              enum4 = 25;
              break;
            }
            case "not-recoverable": {
              enum4 = 26;
              break;
            }
            case "unsupported": {
              enum4 = 27;
              break;
            }
            case "no-tty": {
              enum4 = 28;
              break;
            }
            case "no-such-device": {
              enum4 = 29;
              break;
            }
            case "overflow": {
              enum4 = 30;
              break;
            }
            case "not-permitted": {
              enum4 = 31;
              break;
            }
            case "pipe": {
              enum4 = 32;
              break;
            }
            case "read-only": {
              enum4 = 33;
              break;
            }
            case "invalid-seek": {
              enum4 = 34;
              break;
            }
            case "text-file-busy": {
              enum4 = 35;
              break;
            }
            case "cross-device": {
              enum4 = 36;
              break;
            }
            default: {
              if (e instanceof Error) {
                console.error(e);
              }
              throw new TypeError(`"${val4}" is not one of the cases of error-code`);
            }
          }
          dataView(memory0).setInt8(arg1 + 1, enum4, true);
          break;
        }
        default: {
          throw new TypeError("invalid variant specified for result");
        }
      }
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.get-flags"][Instruction::Return]', {
        funcName: "[method]descriptor.get-flags",
        paramCount: 0,
        postReturn: false
      });
    }
    function trampoline21(arg0, arg1) {
      var handle1 = arg0;
      var rep2 = handleTable7[(handle1 << 1) + 1] & ~T_FLAG;
      var rsc0 = captureTable7.get(rep2);
      if (!rsc0) {
        rsc0 = Object.create(Descriptor2.prototype);
        Object.defineProperty(rsc0, symbolRscHandle, { writable: true, value: handle1 });
        Object.defineProperty(rsc0, symbolRscRep, { writable: true, value: rep2 });
      }
      curResourceBorrows.push(rsc0);
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.get-type"] [Instruction::CallInterface] (async? sync, @ enter)');
      const _interface_call_currentTaskID = startCurrentTask(0, false, "[method]descriptor.get-type");
      let ret;
      try {
        ret = { tag: "ok", val: rsc0.getType() };
      } catch (e) {
        ret = { tag: "err", val: getErrorPayload(e) };
      }
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.get-type"] [Instruction::CallInterface] (sync, @ post-call)');
      for (const rsc of curResourceBorrows) {
        rsc[symbolRscHandle] = void 0;
      }
      curResourceBorrows = [];
      endCurrentTask(0);
      var variant5 = ret;
      switch (variant5.tag) {
        case "ok": {
          const e = variant5.val;
          dataView(memory0).setInt8(arg1 + 0, 0, true);
          var val3 = e;
          let enum3;
          switch (val3) {
            case "unknown": {
              enum3 = 0;
              break;
            }
            case "block-device": {
              enum3 = 1;
              break;
            }
            case "character-device": {
              enum3 = 2;
              break;
            }
            case "directory": {
              enum3 = 3;
              break;
            }
            case "fifo": {
              enum3 = 4;
              break;
            }
            case "symbolic-link": {
              enum3 = 5;
              break;
            }
            case "regular-file": {
              enum3 = 6;
              break;
            }
            case "socket": {
              enum3 = 7;
              break;
            }
            default: {
              if (e instanceof Error) {
                console.error(e);
              }
              throw new TypeError(`"${val3}" is not one of the cases of descriptor-type`);
            }
          }
          dataView(memory0).setInt8(arg1 + 1, enum3, true);
          break;
        }
        case "err": {
          const e = variant5.val;
          dataView(memory0).setInt8(arg1 + 0, 1, true);
          var val4 = e;
          let enum4;
          switch (val4) {
            case "access": {
              enum4 = 0;
              break;
            }
            case "would-block": {
              enum4 = 1;
              break;
            }
            case "already": {
              enum4 = 2;
              break;
            }
            case "bad-descriptor": {
              enum4 = 3;
              break;
            }
            case "busy": {
              enum4 = 4;
              break;
            }
            case "deadlock": {
              enum4 = 5;
              break;
            }
            case "quota": {
              enum4 = 6;
              break;
            }
            case "exist": {
              enum4 = 7;
              break;
            }
            case "file-too-large": {
              enum4 = 8;
              break;
            }
            case "illegal-byte-sequence": {
              enum4 = 9;
              break;
            }
            case "in-progress": {
              enum4 = 10;
              break;
            }
            case "interrupted": {
              enum4 = 11;
              break;
            }
            case "invalid": {
              enum4 = 12;
              break;
            }
            case "io": {
              enum4 = 13;
              break;
            }
            case "is-directory": {
              enum4 = 14;
              break;
            }
            case "loop": {
              enum4 = 15;
              break;
            }
            case "too-many-links": {
              enum4 = 16;
              break;
            }
            case "message-size": {
              enum4 = 17;
              break;
            }
            case "name-too-long": {
              enum4 = 18;
              break;
            }
            case "no-device": {
              enum4 = 19;
              break;
            }
            case "no-entry": {
              enum4 = 20;
              break;
            }
            case "no-lock": {
              enum4 = 21;
              break;
            }
            case "insufficient-memory": {
              enum4 = 22;
              break;
            }
            case "insufficient-space": {
              enum4 = 23;
              break;
            }
            case "not-directory": {
              enum4 = 24;
              break;
            }
            case "not-empty": {
              enum4 = 25;
              break;
            }
            case "not-recoverable": {
              enum4 = 26;
              break;
            }
            case "unsupported": {
              enum4 = 27;
              break;
            }
            case "no-tty": {
              enum4 = 28;
              break;
            }
            case "no-such-device": {
              enum4 = 29;
              break;
            }
            case "overflow": {
              enum4 = 30;
              break;
            }
            case "not-permitted": {
              enum4 = 31;
              break;
            }
            case "pipe": {
              enum4 = 32;
              break;
            }
            case "read-only": {
              enum4 = 33;
              break;
            }
            case "invalid-seek": {
              enum4 = 34;
              break;
            }
            case "text-file-busy": {
              enum4 = 35;
              break;
            }
            case "cross-device": {
              enum4 = 36;
              break;
            }
            default: {
              if (e instanceof Error) {
                console.error(e);
              }
              throw new TypeError(`"${val4}" is not one of the cases of error-code`);
            }
          }
          dataView(memory0).setInt8(arg1 + 1, enum4, true);
          break;
        }
        default: {
          throw new TypeError("invalid variant specified for result");
        }
      }
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.get-type"][Instruction::Return]', {
        funcName: "[method]descriptor.get-type",
        paramCount: 0,
        postReturn: false
      });
    }
    function trampoline22(arg0, arg1) {
      var handle1 = arg0;
      var rep2 = handleTable7[(handle1 << 1) + 1] & ~T_FLAG;
      var rsc0 = captureTable7.get(rep2);
      if (!rsc0) {
        rsc0 = Object.create(Descriptor2.prototype);
        Object.defineProperty(rsc0, symbolRscHandle, { writable: true, value: handle1 });
        Object.defineProperty(rsc0, symbolRscRep, { writable: true, value: rep2 });
      }
      curResourceBorrows.push(rsc0);
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.metadata-hash"] [Instruction::CallInterface] (async? sync, @ enter)');
      const _interface_call_currentTaskID = startCurrentTask(0, false, "[method]descriptor.metadata-hash");
      let ret;
      try {
        ret = { tag: "ok", val: rsc0.metadataHash() };
      } catch (e) {
        ret = { tag: "err", val: getErrorPayload(e) };
      }
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.metadata-hash"] [Instruction::CallInterface] (sync, @ post-call)');
      for (const rsc of curResourceBorrows) {
        rsc[symbolRscHandle] = void 0;
      }
      curResourceBorrows = [];
      endCurrentTask(0);
      var variant5 = ret;
      switch (variant5.tag) {
        case "ok": {
          const e = variant5.val;
          dataView(memory0).setInt8(arg1 + 0, 0, true);
          var { lower: v3_0, upper: v3_1 } = e;
          dataView(memory0).setBigInt64(arg1 + 8, toUint64(v3_0), true);
          dataView(memory0).setBigInt64(arg1 + 16, toUint64(v3_1), true);
          break;
        }
        case "err": {
          const e = variant5.val;
          dataView(memory0).setInt8(arg1 + 0, 1, true);
          var val4 = e;
          let enum4;
          switch (val4) {
            case "access": {
              enum4 = 0;
              break;
            }
            case "would-block": {
              enum4 = 1;
              break;
            }
            case "already": {
              enum4 = 2;
              break;
            }
            case "bad-descriptor": {
              enum4 = 3;
              break;
            }
            case "busy": {
              enum4 = 4;
              break;
            }
            case "deadlock": {
              enum4 = 5;
              break;
            }
            case "quota": {
              enum4 = 6;
              break;
            }
            case "exist": {
              enum4 = 7;
              break;
            }
            case "file-too-large": {
              enum4 = 8;
              break;
            }
            case "illegal-byte-sequence": {
              enum4 = 9;
              break;
            }
            case "in-progress": {
              enum4 = 10;
              break;
            }
            case "interrupted": {
              enum4 = 11;
              break;
            }
            case "invalid": {
              enum4 = 12;
              break;
            }
            case "io": {
              enum4 = 13;
              break;
            }
            case "is-directory": {
              enum4 = 14;
              break;
            }
            case "loop": {
              enum4 = 15;
              break;
            }
            case "too-many-links": {
              enum4 = 16;
              break;
            }
            case "message-size": {
              enum4 = 17;
              break;
            }
            case "name-too-long": {
              enum4 = 18;
              break;
            }
            case "no-device": {
              enum4 = 19;
              break;
            }
            case "no-entry": {
              enum4 = 20;
              break;
            }
            case "no-lock": {
              enum4 = 21;
              break;
            }
            case "insufficient-memory": {
              enum4 = 22;
              break;
            }
            case "insufficient-space": {
              enum4 = 23;
              break;
            }
            case "not-directory": {
              enum4 = 24;
              break;
            }
            case "not-empty": {
              enum4 = 25;
              break;
            }
            case "not-recoverable": {
              enum4 = 26;
              break;
            }
            case "unsupported": {
              enum4 = 27;
              break;
            }
            case "no-tty": {
              enum4 = 28;
              break;
            }
            case "no-such-device": {
              enum4 = 29;
              break;
            }
            case "overflow": {
              enum4 = 30;
              break;
            }
            case "not-permitted": {
              enum4 = 31;
              break;
            }
            case "pipe": {
              enum4 = 32;
              break;
            }
            case "read-only": {
              enum4 = 33;
              break;
            }
            case "invalid-seek": {
              enum4 = 34;
              break;
            }
            case "text-file-busy": {
              enum4 = 35;
              break;
            }
            case "cross-device": {
              enum4 = 36;
              break;
            }
            default: {
              if (e instanceof Error) {
                console.error(e);
              }
              throw new TypeError(`"${val4}" is not one of the cases of error-code`);
            }
          }
          dataView(memory0).setInt8(arg1 + 8, enum4, true);
          break;
        }
        default: {
          throw new TypeError("invalid variant specified for result");
        }
      }
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.metadata-hash"][Instruction::Return]', {
        funcName: "[method]descriptor.metadata-hash",
        paramCount: 0,
        postReturn: false
      });
    }
    function trampoline23(arg0, arg1, arg2) {
      var handle1 = arg0;
      var rep2 = handleTable7[(handle1 << 1) + 1] & ~T_FLAG;
      var rsc0 = captureTable7.get(rep2);
      if (!rsc0) {
        rsc0 = Object.create(Descriptor2.prototype);
        Object.defineProperty(rsc0, symbolRscHandle, { writable: true, value: handle1 });
        Object.defineProperty(rsc0, symbolRscRep, { writable: true, value: rep2 });
      }
      curResourceBorrows.push(rsc0);
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.set-size"] [Instruction::CallInterface] (async? sync, @ enter)');
      const _interface_call_currentTaskID = startCurrentTask(0, false, "[method]descriptor.set-size");
      let ret;
      try {
        ret = { tag: "ok", val: rsc0.setSize(BigInt.asUintN(64, arg1)) };
      } catch (e) {
        ret = { tag: "err", val: getErrorPayload(e) };
      }
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.set-size"] [Instruction::CallInterface] (sync, @ post-call)');
      for (const rsc of curResourceBorrows) {
        rsc[symbolRscHandle] = void 0;
      }
      curResourceBorrows = [];
      endCurrentTask(0);
      var variant4 = ret;
      switch (variant4.tag) {
        case "ok": {
          const e = variant4.val;
          dataView(memory0).setInt8(arg2 + 0, 0, true);
          break;
        }
        case "err": {
          const e = variant4.val;
          dataView(memory0).setInt8(arg2 + 0, 1, true);
          var val3 = e;
          let enum3;
          switch (val3) {
            case "access": {
              enum3 = 0;
              break;
            }
            case "would-block": {
              enum3 = 1;
              break;
            }
            case "already": {
              enum3 = 2;
              break;
            }
            case "bad-descriptor": {
              enum3 = 3;
              break;
            }
            case "busy": {
              enum3 = 4;
              break;
            }
            case "deadlock": {
              enum3 = 5;
              break;
            }
            case "quota": {
              enum3 = 6;
              break;
            }
            case "exist": {
              enum3 = 7;
              break;
            }
            case "file-too-large": {
              enum3 = 8;
              break;
            }
            case "illegal-byte-sequence": {
              enum3 = 9;
              break;
            }
            case "in-progress": {
              enum3 = 10;
              break;
            }
            case "interrupted": {
              enum3 = 11;
              break;
            }
            case "invalid": {
              enum3 = 12;
              break;
            }
            case "io": {
              enum3 = 13;
              break;
            }
            case "is-directory": {
              enum3 = 14;
              break;
            }
            case "loop": {
              enum3 = 15;
              break;
            }
            case "too-many-links": {
              enum3 = 16;
              break;
            }
            case "message-size": {
              enum3 = 17;
              break;
            }
            case "name-too-long": {
              enum3 = 18;
              break;
            }
            case "no-device": {
              enum3 = 19;
              break;
            }
            case "no-entry": {
              enum3 = 20;
              break;
            }
            case "no-lock": {
              enum3 = 21;
              break;
            }
            case "insufficient-memory": {
              enum3 = 22;
              break;
            }
            case "insufficient-space": {
              enum3 = 23;
              break;
            }
            case "not-directory": {
              enum3 = 24;
              break;
            }
            case "not-empty": {
              enum3 = 25;
              break;
            }
            case "not-recoverable": {
              enum3 = 26;
              break;
            }
            case "unsupported": {
              enum3 = 27;
              break;
            }
            case "no-tty": {
              enum3 = 28;
              break;
            }
            case "no-such-device": {
              enum3 = 29;
              break;
            }
            case "overflow": {
              enum3 = 30;
              break;
            }
            case "not-permitted": {
              enum3 = 31;
              break;
            }
            case "pipe": {
              enum3 = 32;
              break;
            }
            case "read-only": {
              enum3 = 33;
              break;
            }
            case "invalid-seek": {
              enum3 = 34;
              break;
            }
            case "text-file-busy": {
              enum3 = 35;
              break;
            }
            case "cross-device": {
              enum3 = 36;
              break;
            }
            default: {
              if (e instanceof Error) {
                console.error(e);
              }
              throw new TypeError(`"${val3}" is not one of the cases of error-code`);
            }
          }
          dataView(memory0).setInt8(arg2 + 1, enum3, true);
          break;
        }
        default: {
          throw new TypeError("invalid variant specified for result");
        }
      }
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.set-size"][Instruction::Return]', {
        funcName: "[method]descriptor.set-size",
        paramCount: 0,
        postReturn: false
      });
    }
    const handleTable0 = [T_FLAG, 0];
    const captureTable0 = /* @__PURE__ */ new Map();
    let captureCnt0 = 0;
    handleTables[0] = handleTable0;
    function trampoline24(arg0, arg1) {
      var handle1 = arg0;
      var rep2 = handleTable0[(handle1 << 1) + 1] & ~T_FLAG;
      var rsc0 = captureTable0.get(rep2);
      if (!rsc0) {
        rsc0 = Object.create(Error$1.prototype);
        Object.defineProperty(rsc0, symbolRscHandle, { writable: true, value: handle1 });
        Object.defineProperty(rsc0, symbolRscRep, { writable: true, value: rep2 });
      }
      curResourceBorrows.push(rsc0);
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="filesystem-error-code"] [Instruction::CallInterface] (async? sync, @ enter)');
      const _interface_call_currentTaskID = startCurrentTask(0, false, "filesystem-error-code");
      const ret = filesystemErrorCode(rsc0);
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="filesystem-error-code"] [Instruction::CallInterface] (sync, @ post-call)');
      for (const rsc of curResourceBorrows) {
        rsc[symbolRscHandle] = void 0;
      }
      curResourceBorrows = [];
      endCurrentTask(0);
      var variant4 = ret;
      if (variant4 === null || variant4 === void 0) {
        dataView(memory0).setInt8(arg1 + 0, 0, true);
      } else {
        const e = variant4;
        dataView(memory0).setInt8(arg1 + 0, 1, true);
        var val3 = e;
        let enum3;
        switch (val3) {
          case "access": {
            enum3 = 0;
            break;
          }
          case "would-block": {
            enum3 = 1;
            break;
          }
          case "already": {
            enum3 = 2;
            break;
          }
          case "bad-descriptor": {
            enum3 = 3;
            break;
          }
          case "busy": {
            enum3 = 4;
            break;
          }
          case "deadlock": {
            enum3 = 5;
            break;
          }
          case "quota": {
            enum3 = 6;
            break;
          }
          case "exist": {
            enum3 = 7;
            break;
          }
          case "file-too-large": {
            enum3 = 8;
            break;
          }
          case "illegal-byte-sequence": {
            enum3 = 9;
            break;
          }
          case "in-progress": {
            enum3 = 10;
            break;
          }
          case "interrupted": {
            enum3 = 11;
            break;
          }
          case "invalid": {
            enum3 = 12;
            break;
          }
          case "io": {
            enum3 = 13;
            break;
          }
          case "is-directory": {
            enum3 = 14;
            break;
          }
          case "loop": {
            enum3 = 15;
            break;
          }
          case "too-many-links": {
            enum3 = 16;
            break;
          }
          case "message-size": {
            enum3 = 17;
            break;
          }
          case "name-too-long": {
            enum3 = 18;
            break;
          }
          case "no-device": {
            enum3 = 19;
            break;
          }
          case "no-entry": {
            enum3 = 20;
            break;
          }
          case "no-lock": {
            enum3 = 21;
            break;
          }
          case "insufficient-memory": {
            enum3 = 22;
            break;
          }
          case "insufficient-space": {
            enum3 = 23;
            break;
          }
          case "not-directory": {
            enum3 = 24;
            break;
          }
          case "not-empty": {
            enum3 = 25;
            break;
          }
          case "not-recoverable": {
            enum3 = 26;
            break;
          }
          case "unsupported": {
            enum3 = 27;
            break;
          }
          case "no-tty": {
            enum3 = 28;
            break;
          }
          case "no-such-device": {
            enum3 = 29;
            break;
          }
          case "overflow": {
            enum3 = 30;
            break;
          }
          case "not-permitted": {
            enum3 = 31;
            break;
          }
          case "pipe": {
            enum3 = 32;
            break;
          }
          case "read-only": {
            enum3 = 33;
            break;
          }
          case "invalid-seek": {
            enum3 = 34;
            break;
          }
          case "text-file-busy": {
            enum3 = 35;
            break;
          }
          case "cross-device": {
            enum3 = 36;
            break;
          }
          default: {
            if (e instanceof Error) {
              console.error(e);
            }
            throw new TypeError(`"${val3}" is not one of the cases of error-code`);
          }
        }
        dataView(memory0).setInt8(arg1 + 1, enum3, true);
      }
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="filesystem-error-code"][Instruction::Return]', {
        funcName: "filesystem-error-code",
        paramCount: 0,
        postReturn: false
      });
    }
    function trampoline25(arg0, arg1, arg2, arg3, arg4) {
      var handle1 = arg0;
      var rep2 = handleTable7[(handle1 << 1) + 1] & ~T_FLAG;
      var rsc0 = captureTable7.get(rep2);
      if (!rsc0) {
        rsc0 = Object.create(Descriptor2.prototype);
        Object.defineProperty(rsc0, symbolRscHandle, { writable: true, value: handle1 });
        Object.defineProperty(rsc0, symbolRscRep, { writable: true, value: rep2 });
      }
      curResourceBorrows.push(rsc0);
      if ((arg1 & 4294967294) !== 0) {
        throw new TypeError("flags have extraneous bits set");
      }
      var flags3 = {
        symlinkFollow: Boolean(arg1 & 1)
      };
      var ptr4 = arg2;
      var len4 = arg3;
      var result4 = utf8Decoder.decode(new Uint8Array(memory0.buffer, ptr4, len4));
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.metadata-hash-at"] [Instruction::CallInterface] (async? sync, @ enter)');
      const _interface_call_currentTaskID = startCurrentTask(0, false, "[method]descriptor.metadata-hash-at");
      let ret;
      try {
        ret = { tag: "ok", val: rsc0.metadataHashAt(flags3, result4) };
      } catch (e) {
        ret = { tag: "err", val: getErrorPayload(e) };
      }
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.metadata-hash-at"] [Instruction::CallInterface] (sync, @ post-call)');
      for (const rsc of curResourceBorrows) {
        rsc[symbolRscHandle] = void 0;
      }
      curResourceBorrows = [];
      endCurrentTask(0);
      var variant7 = ret;
      switch (variant7.tag) {
        case "ok": {
          const e = variant7.val;
          dataView(memory0).setInt8(arg4 + 0, 0, true);
          var { lower: v5_0, upper: v5_1 } = e;
          dataView(memory0).setBigInt64(arg4 + 8, toUint64(v5_0), true);
          dataView(memory0).setBigInt64(arg4 + 16, toUint64(v5_1), true);
          break;
        }
        case "err": {
          const e = variant7.val;
          dataView(memory0).setInt8(arg4 + 0, 1, true);
          var val6 = e;
          let enum6;
          switch (val6) {
            case "access": {
              enum6 = 0;
              break;
            }
            case "would-block": {
              enum6 = 1;
              break;
            }
            case "already": {
              enum6 = 2;
              break;
            }
            case "bad-descriptor": {
              enum6 = 3;
              break;
            }
            case "busy": {
              enum6 = 4;
              break;
            }
            case "deadlock": {
              enum6 = 5;
              break;
            }
            case "quota": {
              enum6 = 6;
              break;
            }
            case "exist": {
              enum6 = 7;
              break;
            }
            case "file-too-large": {
              enum6 = 8;
              break;
            }
            case "illegal-byte-sequence": {
              enum6 = 9;
              break;
            }
            case "in-progress": {
              enum6 = 10;
              break;
            }
            case "interrupted": {
              enum6 = 11;
              break;
            }
            case "invalid": {
              enum6 = 12;
              break;
            }
            case "io": {
              enum6 = 13;
              break;
            }
            case "is-directory": {
              enum6 = 14;
              break;
            }
            case "loop": {
              enum6 = 15;
              break;
            }
            case "too-many-links": {
              enum6 = 16;
              break;
            }
            case "message-size": {
              enum6 = 17;
              break;
            }
            case "name-too-long": {
              enum6 = 18;
              break;
            }
            case "no-device": {
              enum6 = 19;
              break;
            }
            case "no-entry": {
              enum6 = 20;
              break;
            }
            case "no-lock": {
              enum6 = 21;
              break;
            }
            case "insufficient-memory": {
              enum6 = 22;
              break;
            }
            case "insufficient-space": {
              enum6 = 23;
              break;
            }
            case "not-directory": {
              enum6 = 24;
              break;
            }
            case "not-empty": {
              enum6 = 25;
              break;
            }
            case "not-recoverable": {
              enum6 = 26;
              break;
            }
            case "unsupported": {
              enum6 = 27;
              break;
            }
            case "no-tty": {
              enum6 = 28;
              break;
            }
            case "no-such-device": {
              enum6 = 29;
              break;
            }
            case "overflow": {
              enum6 = 30;
              break;
            }
            case "not-permitted": {
              enum6 = 31;
              break;
            }
            case "pipe": {
              enum6 = 32;
              break;
            }
            case "read-only": {
              enum6 = 33;
              break;
            }
            case "invalid-seek": {
              enum6 = 34;
              break;
            }
            case "text-file-busy": {
              enum6 = 35;
              break;
            }
            case "cross-device": {
              enum6 = 36;
              break;
            }
            default: {
              if (e instanceof Error) {
                console.error(e);
              }
              throw new TypeError(`"${val6}" is not one of the cases of error-code`);
            }
          }
          dataView(memory0).setInt8(arg4 + 8, enum6, true);
          break;
        }
        default: {
          throw new TypeError("invalid variant specified for result");
        }
      }
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.metadata-hash-at"][Instruction::Return]', {
        funcName: "[method]descriptor.metadata-hash-at",
        paramCount: 0,
        postReturn: false
      });
    }
    function trampoline26(arg0, arg1, arg2, arg3) {
      var handle1 = arg0;
      var rep2 = handleTable7[(handle1 << 1) + 1] & ~T_FLAG;
      var rsc0 = captureTable7.get(rep2);
      if (!rsc0) {
        rsc0 = Object.create(Descriptor2.prototype);
        Object.defineProperty(rsc0, symbolRscHandle, { writable: true, value: handle1 });
        Object.defineProperty(rsc0, symbolRscRep, { writable: true, value: rep2 });
      }
      curResourceBorrows.push(rsc0);
      var ptr3 = arg1;
      var len3 = arg2;
      var result3 = utf8Decoder.decode(new Uint8Array(memory0.buffer, ptr3, len3));
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.create-directory-at"] [Instruction::CallInterface] (async? sync, @ enter)');
      const _interface_call_currentTaskID = startCurrentTask(0, false, "[method]descriptor.create-directory-at");
      let ret;
      try {
        ret = { tag: "ok", val: rsc0.createDirectoryAt(result3) };
      } catch (e) {
        ret = { tag: "err", val: getErrorPayload(e) };
      }
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.create-directory-at"] [Instruction::CallInterface] (sync, @ post-call)');
      for (const rsc of curResourceBorrows) {
        rsc[symbolRscHandle] = void 0;
      }
      curResourceBorrows = [];
      endCurrentTask(0);
      var variant5 = ret;
      switch (variant5.tag) {
        case "ok": {
          const e = variant5.val;
          dataView(memory0).setInt8(arg3 + 0, 0, true);
          break;
        }
        case "err": {
          const e = variant5.val;
          dataView(memory0).setInt8(arg3 + 0, 1, true);
          var val4 = e;
          let enum4;
          switch (val4) {
            case "access": {
              enum4 = 0;
              break;
            }
            case "would-block": {
              enum4 = 1;
              break;
            }
            case "already": {
              enum4 = 2;
              break;
            }
            case "bad-descriptor": {
              enum4 = 3;
              break;
            }
            case "busy": {
              enum4 = 4;
              break;
            }
            case "deadlock": {
              enum4 = 5;
              break;
            }
            case "quota": {
              enum4 = 6;
              break;
            }
            case "exist": {
              enum4 = 7;
              break;
            }
            case "file-too-large": {
              enum4 = 8;
              break;
            }
            case "illegal-byte-sequence": {
              enum4 = 9;
              break;
            }
            case "in-progress": {
              enum4 = 10;
              break;
            }
            case "interrupted": {
              enum4 = 11;
              break;
            }
            case "invalid": {
              enum4 = 12;
              break;
            }
            case "io": {
              enum4 = 13;
              break;
            }
            case "is-directory": {
              enum4 = 14;
              break;
            }
            case "loop": {
              enum4 = 15;
              break;
            }
            case "too-many-links": {
              enum4 = 16;
              break;
            }
            case "message-size": {
              enum4 = 17;
              break;
            }
            case "name-too-long": {
              enum4 = 18;
              break;
            }
            case "no-device": {
              enum4 = 19;
              break;
            }
            case "no-entry": {
              enum4 = 20;
              break;
            }
            case "no-lock": {
              enum4 = 21;
              break;
            }
            case "insufficient-memory": {
              enum4 = 22;
              break;
            }
            case "insufficient-space": {
              enum4 = 23;
              break;
            }
            case "not-directory": {
              enum4 = 24;
              break;
            }
            case "not-empty": {
              enum4 = 25;
              break;
            }
            case "not-recoverable": {
              enum4 = 26;
              break;
            }
            case "unsupported": {
              enum4 = 27;
              break;
            }
            case "no-tty": {
              enum4 = 28;
              break;
            }
            case "no-such-device": {
              enum4 = 29;
              break;
            }
            case "overflow": {
              enum4 = 30;
              break;
            }
            case "not-permitted": {
              enum4 = 31;
              break;
            }
            case "pipe": {
              enum4 = 32;
              break;
            }
            case "read-only": {
              enum4 = 33;
              break;
            }
            case "invalid-seek": {
              enum4 = 34;
              break;
            }
            case "text-file-busy": {
              enum4 = 35;
              break;
            }
            case "cross-device": {
              enum4 = 36;
              break;
            }
            default: {
              if (e instanceof Error) {
                console.error(e);
              }
              throw new TypeError(`"${val4}" is not one of the cases of error-code`);
            }
          }
          dataView(memory0).setInt8(arg3 + 1, enum4, true);
          break;
        }
        default: {
          throw new TypeError("invalid variant specified for result");
        }
      }
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.create-directory-at"][Instruction::Return]', {
        funcName: "[method]descriptor.create-directory-at",
        paramCount: 0,
        postReturn: false
      });
    }
    function trampoline27(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7) {
      var handle1 = arg0;
      var rep2 = handleTable7[(handle1 << 1) + 1] & ~T_FLAG;
      var rsc0 = captureTable7.get(rep2);
      if (!rsc0) {
        rsc0 = Object.create(Descriptor2.prototype);
        Object.defineProperty(rsc0, symbolRscHandle, { writable: true, value: handle1 });
        Object.defineProperty(rsc0, symbolRscRep, { writable: true, value: rep2 });
      }
      curResourceBorrows.push(rsc0);
      if ((arg1 & 4294967294) !== 0) {
        throw new TypeError("flags have extraneous bits set");
      }
      var flags3 = {
        symlinkFollow: Boolean(arg1 & 1)
      };
      var ptr4 = arg2;
      var len4 = arg3;
      var result4 = utf8Decoder.decode(new Uint8Array(memory0.buffer, ptr4, len4));
      var handle6 = arg4;
      var rep7 = handleTable7[(handle6 << 1) + 1] & ~T_FLAG;
      var rsc5 = captureTable7.get(rep7);
      if (!rsc5) {
        rsc5 = Object.create(Descriptor2.prototype);
        Object.defineProperty(rsc5, symbolRscHandle, { writable: true, value: handle6 });
        Object.defineProperty(rsc5, symbolRscRep, { writable: true, value: rep7 });
      }
      curResourceBorrows.push(rsc5);
      var ptr8 = arg5;
      var len8 = arg6;
      var result8 = utf8Decoder.decode(new Uint8Array(memory0.buffer, ptr8, len8));
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.link-at"] [Instruction::CallInterface] (async? sync, @ enter)');
      const _interface_call_currentTaskID = startCurrentTask(0, false, "[method]descriptor.link-at");
      let ret;
      try {
        ret = { tag: "ok", val: rsc0.linkAt(flags3, result4, rsc5, result8) };
      } catch (e) {
        ret = { tag: "err", val: getErrorPayload(e) };
      }
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.link-at"] [Instruction::CallInterface] (sync, @ post-call)');
      for (const rsc of curResourceBorrows) {
        rsc[symbolRscHandle] = void 0;
      }
      curResourceBorrows = [];
      endCurrentTask(0);
      var variant10 = ret;
      switch (variant10.tag) {
        case "ok": {
          const e = variant10.val;
          dataView(memory0).setInt8(arg7 + 0, 0, true);
          break;
        }
        case "err": {
          const e = variant10.val;
          dataView(memory0).setInt8(arg7 + 0, 1, true);
          var val9 = e;
          let enum9;
          switch (val9) {
            case "access": {
              enum9 = 0;
              break;
            }
            case "would-block": {
              enum9 = 1;
              break;
            }
            case "already": {
              enum9 = 2;
              break;
            }
            case "bad-descriptor": {
              enum9 = 3;
              break;
            }
            case "busy": {
              enum9 = 4;
              break;
            }
            case "deadlock": {
              enum9 = 5;
              break;
            }
            case "quota": {
              enum9 = 6;
              break;
            }
            case "exist": {
              enum9 = 7;
              break;
            }
            case "file-too-large": {
              enum9 = 8;
              break;
            }
            case "illegal-byte-sequence": {
              enum9 = 9;
              break;
            }
            case "in-progress": {
              enum9 = 10;
              break;
            }
            case "interrupted": {
              enum9 = 11;
              break;
            }
            case "invalid": {
              enum9 = 12;
              break;
            }
            case "io": {
              enum9 = 13;
              break;
            }
            case "is-directory": {
              enum9 = 14;
              break;
            }
            case "loop": {
              enum9 = 15;
              break;
            }
            case "too-many-links": {
              enum9 = 16;
              break;
            }
            case "message-size": {
              enum9 = 17;
              break;
            }
            case "name-too-long": {
              enum9 = 18;
              break;
            }
            case "no-device": {
              enum9 = 19;
              break;
            }
            case "no-entry": {
              enum9 = 20;
              break;
            }
            case "no-lock": {
              enum9 = 21;
              break;
            }
            case "insufficient-memory": {
              enum9 = 22;
              break;
            }
            case "insufficient-space": {
              enum9 = 23;
              break;
            }
            case "not-directory": {
              enum9 = 24;
              break;
            }
            case "not-empty": {
              enum9 = 25;
              break;
            }
            case "not-recoverable": {
              enum9 = 26;
              break;
            }
            case "unsupported": {
              enum9 = 27;
              break;
            }
            case "no-tty": {
              enum9 = 28;
              break;
            }
            case "no-such-device": {
              enum9 = 29;
              break;
            }
            case "overflow": {
              enum9 = 30;
              break;
            }
            case "not-permitted": {
              enum9 = 31;
              break;
            }
            case "pipe": {
              enum9 = 32;
              break;
            }
            case "read-only": {
              enum9 = 33;
              break;
            }
            case "invalid-seek": {
              enum9 = 34;
              break;
            }
            case "text-file-busy": {
              enum9 = 35;
              break;
            }
            case "cross-device": {
              enum9 = 36;
              break;
            }
            default: {
              if (e instanceof Error) {
                console.error(e);
              }
              throw new TypeError(`"${val9}" is not one of the cases of error-code`);
            }
          }
          dataView(memory0).setInt8(arg7 + 1, enum9, true);
          break;
        }
        default: {
          throw new TypeError("invalid variant specified for result");
        }
      }
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.link-at"][Instruction::Return]', {
        funcName: "[method]descriptor.link-at",
        paramCount: 0,
        postReturn: false
      });
    }
    function trampoline28(arg0, arg1, arg2, arg3) {
      var handle1 = arg0;
      var rep2 = handleTable7[(handle1 << 1) + 1] & ~T_FLAG;
      var rsc0 = captureTable7.get(rep2);
      if (!rsc0) {
        rsc0 = Object.create(Descriptor2.prototype);
        Object.defineProperty(rsc0, symbolRscHandle, { writable: true, value: handle1 });
        Object.defineProperty(rsc0, symbolRscRep, { writable: true, value: rep2 });
      }
      curResourceBorrows.push(rsc0);
      var ptr3 = arg1;
      var len3 = arg2;
      var result3 = utf8Decoder.decode(new Uint8Array(memory0.buffer, ptr3, len3));
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.readlink-at"] [Instruction::CallInterface] (async? sync, @ enter)');
      const _interface_call_currentTaskID = startCurrentTask(0, false, "[method]descriptor.readlink-at");
      let ret;
      try {
        ret = { tag: "ok", val: rsc0.readlinkAt(result3) };
      } catch (e) {
        ret = { tag: "err", val: getErrorPayload(e) };
      }
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.readlink-at"] [Instruction::CallInterface] (sync, @ post-call)');
      for (const rsc of curResourceBorrows) {
        rsc[symbolRscHandle] = void 0;
      }
      curResourceBorrows = [];
      endCurrentTask(0);
      var variant6 = ret;
      switch (variant6.tag) {
        case "ok": {
          const e = variant6.val;
          dataView(memory0).setInt8(arg3 + 0, 0, true);
          var ptr4 = utf8Encode(e, realloc0, memory0);
          var len4 = utf8EncodedLen;
          dataView(memory0).setUint32(arg3 + 8, len4, true);
          dataView(memory0).setUint32(arg3 + 4, ptr4, true);
          break;
        }
        case "err": {
          const e = variant6.val;
          dataView(memory0).setInt8(arg3 + 0, 1, true);
          var val5 = e;
          let enum5;
          switch (val5) {
            case "access": {
              enum5 = 0;
              break;
            }
            case "would-block": {
              enum5 = 1;
              break;
            }
            case "already": {
              enum5 = 2;
              break;
            }
            case "bad-descriptor": {
              enum5 = 3;
              break;
            }
            case "busy": {
              enum5 = 4;
              break;
            }
            case "deadlock": {
              enum5 = 5;
              break;
            }
            case "quota": {
              enum5 = 6;
              break;
            }
            case "exist": {
              enum5 = 7;
              break;
            }
            case "file-too-large": {
              enum5 = 8;
              break;
            }
            case "illegal-byte-sequence": {
              enum5 = 9;
              break;
            }
            case "in-progress": {
              enum5 = 10;
              break;
            }
            case "interrupted": {
              enum5 = 11;
              break;
            }
            case "invalid": {
              enum5 = 12;
              break;
            }
            case "io": {
              enum5 = 13;
              break;
            }
            case "is-directory": {
              enum5 = 14;
              break;
            }
            case "loop": {
              enum5 = 15;
              break;
            }
            case "too-many-links": {
              enum5 = 16;
              break;
            }
            case "message-size": {
              enum5 = 17;
              break;
            }
            case "name-too-long": {
              enum5 = 18;
              break;
            }
            case "no-device": {
              enum5 = 19;
              break;
            }
            case "no-entry": {
              enum5 = 20;
              break;
            }
            case "no-lock": {
              enum5 = 21;
              break;
            }
            case "insufficient-memory": {
              enum5 = 22;
              break;
            }
            case "insufficient-space": {
              enum5 = 23;
              break;
            }
            case "not-directory": {
              enum5 = 24;
              break;
            }
            case "not-empty": {
              enum5 = 25;
              break;
            }
            case "not-recoverable": {
              enum5 = 26;
              break;
            }
            case "unsupported": {
              enum5 = 27;
              break;
            }
            case "no-tty": {
              enum5 = 28;
              break;
            }
            case "no-such-device": {
              enum5 = 29;
              break;
            }
            case "overflow": {
              enum5 = 30;
              break;
            }
            case "not-permitted": {
              enum5 = 31;
              break;
            }
            case "pipe": {
              enum5 = 32;
              break;
            }
            case "read-only": {
              enum5 = 33;
              break;
            }
            case "invalid-seek": {
              enum5 = 34;
              break;
            }
            case "text-file-busy": {
              enum5 = 35;
              break;
            }
            case "cross-device": {
              enum5 = 36;
              break;
            }
            default: {
              if (e instanceof Error) {
                console.error(e);
              }
              throw new TypeError(`"${val5}" is not one of the cases of error-code`);
            }
          }
          dataView(memory0).setInt8(arg3 + 4, enum5, true);
          break;
        }
        default: {
          throw new TypeError("invalid variant specified for result");
        }
      }
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.readlink-at"][Instruction::Return]', {
        funcName: "[method]descriptor.readlink-at",
        paramCount: 0,
        postReturn: false
      });
    }
    function trampoline29(arg0, arg1, arg2, arg3) {
      var handle1 = arg0;
      var rep2 = handleTable7[(handle1 << 1) + 1] & ~T_FLAG;
      var rsc0 = captureTable7.get(rep2);
      if (!rsc0) {
        rsc0 = Object.create(Descriptor2.prototype);
        Object.defineProperty(rsc0, symbolRscHandle, { writable: true, value: handle1 });
        Object.defineProperty(rsc0, symbolRscRep, { writable: true, value: rep2 });
      }
      curResourceBorrows.push(rsc0);
      var ptr3 = arg1;
      var len3 = arg2;
      var result3 = utf8Decoder.decode(new Uint8Array(memory0.buffer, ptr3, len3));
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.remove-directory-at"] [Instruction::CallInterface] (async? sync, @ enter)');
      const _interface_call_currentTaskID = startCurrentTask(0, false, "[method]descriptor.remove-directory-at");
      let ret;
      try {
        ret = { tag: "ok", val: rsc0.removeDirectoryAt(result3) };
      } catch (e) {
        ret = { tag: "err", val: getErrorPayload(e) };
      }
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.remove-directory-at"] [Instruction::CallInterface] (sync, @ post-call)');
      for (const rsc of curResourceBorrows) {
        rsc[symbolRscHandle] = void 0;
      }
      curResourceBorrows = [];
      endCurrentTask(0);
      var variant5 = ret;
      switch (variant5.tag) {
        case "ok": {
          const e = variant5.val;
          dataView(memory0).setInt8(arg3 + 0, 0, true);
          break;
        }
        case "err": {
          const e = variant5.val;
          dataView(memory0).setInt8(arg3 + 0, 1, true);
          var val4 = e;
          let enum4;
          switch (val4) {
            case "access": {
              enum4 = 0;
              break;
            }
            case "would-block": {
              enum4 = 1;
              break;
            }
            case "already": {
              enum4 = 2;
              break;
            }
            case "bad-descriptor": {
              enum4 = 3;
              break;
            }
            case "busy": {
              enum4 = 4;
              break;
            }
            case "deadlock": {
              enum4 = 5;
              break;
            }
            case "quota": {
              enum4 = 6;
              break;
            }
            case "exist": {
              enum4 = 7;
              break;
            }
            case "file-too-large": {
              enum4 = 8;
              break;
            }
            case "illegal-byte-sequence": {
              enum4 = 9;
              break;
            }
            case "in-progress": {
              enum4 = 10;
              break;
            }
            case "interrupted": {
              enum4 = 11;
              break;
            }
            case "invalid": {
              enum4 = 12;
              break;
            }
            case "io": {
              enum4 = 13;
              break;
            }
            case "is-directory": {
              enum4 = 14;
              break;
            }
            case "loop": {
              enum4 = 15;
              break;
            }
            case "too-many-links": {
              enum4 = 16;
              break;
            }
            case "message-size": {
              enum4 = 17;
              break;
            }
            case "name-too-long": {
              enum4 = 18;
              break;
            }
            case "no-device": {
              enum4 = 19;
              break;
            }
            case "no-entry": {
              enum4 = 20;
              break;
            }
            case "no-lock": {
              enum4 = 21;
              break;
            }
            case "insufficient-memory": {
              enum4 = 22;
              break;
            }
            case "insufficient-space": {
              enum4 = 23;
              break;
            }
            case "not-directory": {
              enum4 = 24;
              break;
            }
            case "not-empty": {
              enum4 = 25;
              break;
            }
            case "not-recoverable": {
              enum4 = 26;
              break;
            }
            case "unsupported": {
              enum4 = 27;
              break;
            }
            case "no-tty": {
              enum4 = 28;
              break;
            }
            case "no-such-device": {
              enum4 = 29;
              break;
            }
            case "overflow": {
              enum4 = 30;
              break;
            }
            case "not-permitted": {
              enum4 = 31;
              break;
            }
            case "pipe": {
              enum4 = 32;
              break;
            }
            case "read-only": {
              enum4 = 33;
              break;
            }
            case "invalid-seek": {
              enum4 = 34;
              break;
            }
            case "text-file-busy": {
              enum4 = 35;
              break;
            }
            case "cross-device": {
              enum4 = 36;
              break;
            }
            default: {
              if (e instanceof Error) {
                console.error(e);
              }
              throw new TypeError(`"${val4}" is not one of the cases of error-code`);
            }
          }
          dataView(memory0).setInt8(arg3 + 1, enum4, true);
          break;
        }
        default: {
          throw new TypeError("invalid variant specified for result");
        }
      }
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.remove-directory-at"][Instruction::Return]', {
        funcName: "[method]descriptor.remove-directory-at",
        paramCount: 0,
        postReturn: false
      });
    }
    function trampoline30(arg0, arg1, arg2, arg3, arg4, arg5, arg6) {
      var handle1 = arg0;
      var rep2 = handleTable7[(handle1 << 1) + 1] & ~T_FLAG;
      var rsc0 = captureTable7.get(rep2);
      if (!rsc0) {
        rsc0 = Object.create(Descriptor2.prototype);
        Object.defineProperty(rsc0, symbolRscHandle, { writable: true, value: handle1 });
        Object.defineProperty(rsc0, symbolRscRep, { writable: true, value: rep2 });
      }
      curResourceBorrows.push(rsc0);
      var ptr3 = arg1;
      var len3 = arg2;
      var result3 = utf8Decoder.decode(new Uint8Array(memory0.buffer, ptr3, len3));
      var handle5 = arg3;
      var rep6 = handleTable7[(handle5 << 1) + 1] & ~T_FLAG;
      var rsc4 = captureTable7.get(rep6);
      if (!rsc4) {
        rsc4 = Object.create(Descriptor2.prototype);
        Object.defineProperty(rsc4, symbolRscHandle, { writable: true, value: handle5 });
        Object.defineProperty(rsc4, symbolRscRep, { writable: true, value: rep6 });
      }
      curResourceBorrows.push(rsc4);
      var ptr7 = arg4;
      var len7 = arg5;
      var result7 = utf8Decoder.decode(new Uint8Array(memory0.buffer, ptr7, len7));
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.rename-at"] [Instruction::CallInterface] (async? sync, @ enter)');
      const _interface_call_currentTaskID = startCurrentTask(0, false, "[method]descriptor.rename-at");
      let ret;
      try {
        ret = { tag: "ok", val: rsc0.renameAt(result3, rsc4, result7) };
      } catch (e) {
        ret = { tag: "err", val: getErrorPayload(e) };
      }
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.rename-at"] [Instruction::CallInterface] (sync, @ post-call)');
      for (const rsc of curResourceBorrows) {
        rsc[symbolRscHandle] = void 0;
      }
      curResourceBorrows = [];
      endCurrentTask(0);
      var variant9 = ret;
      switch (variant9.tag) {
        case "ok": {
          const e = variant9.val;
          dataView(memory0).setInt8(arg6 + 0, 0, true);
          break;
        }
        case "err": {
          const e = variant9.val;
          dataView(memory0).setInt8(arg6 + 0, 1, true);
          var val8 = e;
          let enum8;
          switch (val8) {
            case "access": {
              enum8 = 0;
              break;
            }
            case "would-block": {
              enum8 = 1;
              break;
            }
            case "already": {
              enum8 = 2;
              break;
            }
            case "bad-descriptor": {
              enum8 = 3;
              break;
            }
            case "busy": {
              enum8 = 4;
              break;
            }
            case "deadlock": {
              enum8 = 5;
              break;
            }
            case "quota": {
              enum8 = 6;
              break;
            }
            case "exist": {
              enum8 = 7;
              break;
            }
            case "file-too-large": {
              enum8 = 8;
              break;
            }
            case "illegal-byte-sequence": {
              enum8 = 9;
              break;
            }
            case "in-progress": {
              enum8 = 10;
              break;
            }
            case "interrupted": {
              enum8 = 11;
              break;
            }
            case "invalid": {
              enum8 = 12;
              break;
            }
            case "io": {
              enum8 = 13;
              break;
            }
            case "is-directory": {
              enum8 = 14;
              break;
            }
            case "loop": {
              enum8 = 15;
              break;
            }
            case "too-many-links": {
              enum8 = 16;
              break;
            }
            case "message-size": {
              enum8 = 17;
              break;
            }
            case "name-too-long": {
              enum8 = 18;
              break;
            }
            case "no-device": {
              enum8 = 19;
              break;
            }
            case "no-entry": {
              enum8 = 20;
              break;
            }
            case "no-lock": {
              enum8 = 21;
              break;
            }
            case "insufficient-memory": {
              enum8 = 22;
              break;
            }
            case "insufficient-space": {
              enum8 = 23;
              break;
            }
            case "not-directory": {
              enum8 = 24;
              break;
            }
            case "not-empty": {
              enum8 = 25;
              break;
            }
            case "not-recoverable": {
              enum8 = 26;
              break;
            }
            case "unsupported": {
              enum8 = 27;
              break;
            }
            case "no-tty": {
              enum8 = 28;
              break;
            }
            case "no-such-device": {
              enum8 = 29;
              break;
            }
            case "overflow": {
              enum8 = 30;
              break;
            }
            case "not-permitted": {
              enum8 = 31;
              break;
            }
            case "pipe": {
              enum8 = 32;
              break;
            }
            case "read-only": {
              enum8 = 33;
              break;
            }
            case "invalid-seek": {
              enum8 = 34;
              break;
            }
            case "text-file-busy": {
              enum8 = 35;
              break;
            }
            case "cross-device": {
              enum8 = 36;
              break;
            }
            default: {
              if (e instanceof Error) {
                console.error(e);
              }
              throw new TypeError(`"${val8}" is not one of the cases of error-code`);
            }
          }
          dataView(memory0).setInt8(arg6 + 1, enum8, true);
          break;
        }
        default: {
          throw new TypeError("invalid variant specified for result");
        }
      }
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.rename-at"][Instruction::Return]', {
        funcName: "[method]descriptor.rename-at",
        paramCount: 0,
        postReturn: false
      });
    }
    function trampoline31(arg0, arg1, arg2, arg3, arg4, arg5) {
      var handle1 = arg0;
      var rep2 = handleTable7[(handle1 << 1) + 1] & ~T_FLAG;
      var rsc0 = captureTable7.get(rep2);
      if (!rsc0) {
        rsc0 = Object.create(Descriptor2.prototype);
        Object.defineProperty(rsc0, symbolRscHandle, { writable: true, value: handle1 });
        Object.defineProperty(rsc0, symbolRscRep, { writable: true, value: rep2 });
      }
      curResourceBorrows.push(rsc0);
      var ptr3 = arg1;
      var len3 = arg2;
      var result3 = utf8Decoder.decode(new Uint8Array(memory0.buffer, ptr3, len3));
      var ptr4 = arg3;
      var len4 = arg4;
      var result4 = utf8Decoder.decode(new Uint8Array(memory0.buffer, ptr4, len4));
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.symlink-at"] [Instruction::CallInterface] (async? sync, @ enter)');
      const _interface_call_currentTaskID = startCurrentTask(0, false, "[method]descriptor.symlink-at");
      let ret;
      try {
        ret = { tag: "ok", val: rsc0.symlinkAt(result3, result4) };
      } catch (e) {
        ret = { tag: "err", val: getErrorPayload(e) };
      }
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.symlink-at"] [Instruction::CallInterface] (sync, @ post-call)');
      for (const rsc of curResourceBorrows) {
        rsc[symbolRscHandle] = void 0;
      }
      curResourceBorrows = [];
      endCurrentTask(0);
      var variant6 = ret;
      switch (variant6.tag) {
        case "ok": {
          const e = variant6.val;
          dataView(memory0).setInt8(arg5 + 0, 0, true);
          break;
        }
        case "err": {
          const e = variant6.val;
          dataView(memory0).setInt8(arg5 + 0, 1, true);
          var val5 = e;
          let enum5;
          switch (val5) {
            case "access": {
              enum5 = 0;
              break;
            }
            case "would-block": {
              enum5 = 1;
              break;
            }
            case "already": {
              enum5 = 2;
              break;
            }
            case "bad-descriptor": {
              enum5 = 3;
              break;
            }
            case "busy": {
              enum5 = 4;
              break;
            }
            case "deadlock": {
              enum5 = 5;
              break;
            }
            case "quota": {
              enum5 = 6;
              break;
            }
            case "exist": {
              enum5 = 7;
              break;
            }
            case "file-too-large": {
              enum5 = 8;
              break;
            }
            case "illegal-byte-sequence": {
              enum5 = 9;
              break;
            }
            case "in-progress": {
              enum5 = 10;
              break;
            }
            case "interrupted": {
              enum5 = 11;
              break;
            }
            case "invalid": {
              enum5 = 12;
              break;
            }
            case "io": {
              enum5 = 13;
              break;
            }
            case "is-directory": {
              enum5 = 14;
              break;
            }
            case "loop": {
              enum5 = 15;
              break;
            }
            case "too-many-links": {
              enum5 = 16;
              break;
            }
            case "message-size": {
              enum5 = 17;
              break;
            }
            case "name-too-long": {
              enum5 = 18;
              break;
            }
            case "no-device": {
              enum5 = 19;
              break;
            }
            case "no-entry": {
              enum5 = 20;
              break;
            }
            case "no-lock": {
              enum5 = 21;
              break;
            }
            case "insufficient-memory": {
              enum5 = 22;
              break;
            }
            case "insufficient-space": {
              enum5 = 23;
              break;
            }
            case "not-directory": {
              enum5 = 24;
              break;
            }
            case "not-empty": {
              enum5 = 25;
              break;
            }
            case "not-recoverable": {
              enum5 = 26;
              break;
            }
            case "unsupported": {
              enum5 = 27;
              break;
            }
            case "no-tty": {
              enum5 = 28;
              break;
            }
            case "no-such-device": {
              enum5 = 29;
              break;
            }
            case "overflow": {
              enum5 = 30;
              break;
            }
            case "not-permitted": {
              enum5 = 31;
              break;
            }
            case "pipe": {
              enum5 = 32;
              break;
            }
            case "read-only": {
              enum5 = 33;
              break;
            }
            case "invalid-seek": {
              enum5 = 34;
              break;
            }
            case "text-file-busy": {
              enum5 = 35;
              break;
            }
            case "cross-device": {
              enum5 = 36;
              break;
            }
            default: {
              if (e instanceof Error) {
                console.error(e);
              }
              throw new TypeError(`"${val5}" is not one of the cases of error-code`);
            }
          }
          dataView(memory0).setInt8(arg5 + 1, enum5, true);
          break;
        }
        default: {
          throw new TypeError("invalid variant specified for result");
        }
      }
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.symlink-at"][Instruction::Return]', {
        funcName: "[method]descriptor.symlink-at",
        paramCount: 0,
        postReturn: false
      });
    }
    function trampoline32(arg0, arg1, arg2, arg3) {
      var handle1 = arg0;
      var rep2 = handleTable7[(handle1 << 1) + 1] & ~T_FLAG;
      var rsc0 = captureTable7.get(rep2);
      if (!rsc0) {
        rsc0 = Object.create(Descriptor2.prototype);
        Object.defineProperty(rsc0, symbolRscHandle, { writable: true, value: handle1 });
        Object.defineProperty(rsc0, symbolRscRep, { writable: true, value: rep2 });
      }
      curResourceBorrows.push(rsc0);
      var ptr3 = arg1;
      var len3 = arg2;
      var result3 = utf8Decoder.decode(new Uint8Array(memory0.buffer, ptr3, len3));
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.unlink-file-at"] [Instruction::CallInterface] (async? sync, @ enter)');
      const _interface_call_currentTaskID = startCurrentTask(0, false, "[method]descriptor.unlink-file-at");
      let ret;
      try {
        ret = { tag: "ok", val: rsc0.unlinkFileAt(result3) };
      } catch (e) {
        ret = { tag: "err", val: getErrorPayload(e) };
      }
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.unlink-file-at"] [Instruction::CallInterface] (sync, @ post-call)');
      for (const rsc of curResourceBorrows) {
        rsc[symbolRscHandle] = void 0;
      }
      curResourceBorrows = [];
      endCurrentTask(0);
      var variant5 = ret;
      switch (variant5.tag) {
        case "ok": {
          const e = variant5.val;
          dataView(memory0).setInt8(arg3 + 0, 0, true);
          break;
        }
        case "err": {
          const e = variant5.val;
          dataView(memory0).setInt8(arg3 + 0, 1, true);
          var val4 = e;
          let enum4;
          switch (val4) {
            case "access": {
              enum4 = 0;
              break;
            }
            case "would-block": {
              enum4 = 1;
              break;
            }
            case "already": {
              enum4 = 2;
              break;
            }
            case "bad-descriptor": {
              enum4 = 3;
              break;
            }
            case "busy": {
              enum4 = 4;
              break;
            }
            case "deadlock": {
              enum4 = 5;
              break;
            }
            case "quota": {
              enum4 = 6;
              break;
            }
            case "exist": {
              enum4 = 7;
              break;
            }
            case "file-too-large": {
              enum4 = 8;
              break;
            }
            case "illegal-byte-sequence": {
              enum4 = 9;
              break;
            }
            case "in-progress": {
              enum4 = 10;
              break;
            }
            case "interrupted": {
              enum4 = 11;
              break;
            }
            case "invalid": {
              enum4 = 12;
              break;
            }
            case "io": {
              enum4 = 13;
              break;
            }
            case "is-directory": {
              enum4 = 14;
              break;
            }
            case "loop": {
              enum4 = 15;
              break;
            }
            case "too-many-links": {
              enum4 = 16;
              break;
            }
            case "message-size": {
              enum4 = 17;
              break;
            }
            case "name-too-long": {
              enum4 = 18;
              break;
            }
            case "no-device": {
              enum4 = 19;
              break;
            }
            case "no-entry": {
              enum4 = 20;
              break;
            }
            case "no-lock": {
              enum4 = 21;
              break;
            }
            case "insufficient-memory": {
              enum4 = 22;
              break;
            }
            case "insufficient-space": {
              enum4 = 23;
              break;
            }
            case "not-directory": {
              enum4 = 24;
              break;
            }
            case "not-empty": {
              enum4 = 25;
              break;
            }
            case "not-recoverable": {
              enum4 = 26;
              break;
            }
            case "unsupported": {
              enum4 = 27;
              break;
            }
            case "no-tty": {
              enum4 = 28;
              break;
            }
            case "no-such-device": {
              enum4 = 29;
              break;
            }
            case "overflow": {
              enum4 = 30;
              break;
            }
            case "not-permitted": {
              enum4 = 31;
              break;
            }
            case "pipe": {
              enum4 = 32;
              break;
            }
            case "read-only": {
              enum4 = 33;
              break;
            }
            case "invalid-seek": {
              enum4 = 34;
              break;
            }
            case "text-file-busy": {
              enum4 = 35;
              break;
            }
            case "cross-device": {
              enum4 = 36;
              break;
            }
            default: {
              if (e instanceof Error) {
                console.error(e);
              }
              throw new TypeError(`"${val4}" is not one of the cases of error-code`);
            }
          }
          dataView(memory0).setInt8(arg3 + 1, enum4, true);
          break;
        }
        default: {
          throw new TypeError("invalid variant specified for result");
        }
      }
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.unlink-file-at"][Instruction::Return]', {
        funcName: "[method]descriptor.unlink-file-at",
        paramCount: 0,
        postReturn: false
      });
    }
    function trampoline33(arg0, arg1, arg2) {
      var handle1 = arg0;
      var rep2 = handleTable7[(handle1 << 1) + 1] & ~T_FLAG;
      var rsc0 = captureTable7.get(rep2);
      if (!rsc0) {
        rsc0 = Object.create(Descriptor2.prototype);
        Object.defineProperty(rsc0, symbolRscHandle, { writable: true, value: handle1 });
        Object.defineProperty(rsc0, symbolRscRep, { writable: true, value: rep2 });
      }
      curResourceBorrows.push(rsc0);
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.read-via-stream"] [Instruction::CallInterface] (async? sync, @ enter)');
      const _interface_call_currentTaskID = startCurrentTask(0, false, "[method]descriptor.read-via-stream");
      let ret;
      try {
        ret = { tag: "ok", val: rsc0.readViaStream(BigInt.asUintN(64, arg1)) };
      } catch (e) {
        ret = { tag: "err", val: getErrorPayload(e) };
      }
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.read-via-stream"] [Instruction::CallInterface] (sync, @ post-call)');
      for (const rsc of curResourceBorrows) {
        rsc[symbolRscHandle] = void 0;
      }
      curResourceBorrows = [];
      endCurrentTask(0);
      var variant5 = ret;
      switch (variant5.tag) {
        case "ok": {
          const e = variant5.val;
          dataView(memory0).setInt8(arg2 + 0, 0, true);
          if (!(e instanceof InputStream2)) {
            throw new TypeError('Resource error: Not a valid "InputStream" resource.');
          }
          var handle3 = e[symbolRscHandle];
          if (!handle3) {
            const rep = e[symbolRscRep] || ++captureCnt2;
            captureTable2.set(rep, e);
            handle3 = rscTableCreateOwn(handleTable2, rep);
          }
          dataView(memory0).setInt32(arg2 + 4, handle3, true);
          break;
        }
        case "err": {
          const e = variant5.val;
          dataView(memory0).setInt8(arg2 + 0, 1, true);
          var val4 = e;
          let enum4;
          switch (val4) {
            case "access": {
              enum4 = 0;
              break;
            }
            case "would-block": {
              enum4 = 1;
              break;
            }
            case "already": {
              enum4 = 2;
              break;
            }
            case "bad-descriptor": {
              enum4 = 3;
              break;
            }
            case "busy": {
              enum4 = 4;
              break;
            }
            case "deadlock": {
              enum4 = 5;
              break;
            }
            case "quota": {
              enum4 = 6;
              break;
            }
            case "exist": {
              enum4 = 7;
              break;
            }
            case "file-too-large": {
              enum4 = 8;
              break;
            }
            case "illegal-byte-sequence": {
              enum4 = 9;
              break;
            }
            case "in-progress": {
              enum4 = 10;
              break;
            }
            case "interrupted": {
              enum4 = 11;
              break;
            }
            case "invalid": {
              enum4 = 12;
              break;
            }
            case "io": {
              enum4 = 13;
              break;
            }
            case "is-directory": {
              enum4 = 14;
              break;
            }
            case "loop": {
              enum4 = 15;
              break;
            }
            case "too-many-links": {
              enum4 = 16;
              break;
            }
            case "message-size": {
              enum4 = 17;
              break;
            }
            case "name-too-long": {
              enum4 = 18;
              break;
            }
            case "no-device": {
              enum4 = 19;
              break;
            }
            case "no-entry": {
              enum4 = 20;
              break;
            }
            case "no-lock": {
              enum4 = 21;
              break;
            }
            case "insufficient-memory": {
              enum4 = 22;
              break;
            }
            case "insufficient-space": {
              enum4 = 23;
              break;
            }
            case "not-directory": {
              enum4 = 24;
              break;
            }
            case "not-empty": {
              enum4 = 25;
              break;
            }
            case "not-recoverable": {
              enum4 = 26;
              break;
            }
            case "unsupported": {
              enum4 = 27;
              break;
            }
            case "no-tty": {
              enum4 = 28;
              break;
            }
            case "no-such-device": {
              enum4 = 29;
              break;
            }
            case "overflow": {
              enum4 = 30;
              break;
            }
            case "not-permitted": {
              enum4 = 31;
              break;
            }
            case "pipe": {
              enum4 = 32;
              break;
            }
            case "read-only": {
              enum4 = 33;
              break;
            }
            case "invalid-seek": {
              enum4 = 34;
              break;
            }
            case "text-file-busy": {
              enum4 = 35;
              break;
            }
            case "cross-device": {
              enum4 = 36;
              break;
            }
            default: {
              if (e instanceof Error) {
                console.error(e);
              }
              throw new TypeError(`"${val4}" is not one of the cases of error-code`);
            }
          }
          dataView(memory0).setInt8(arg2 + 4, enum4, true);
          break;
        }
        default: {
          throw new TypeError("invalid variant specified for result");
        }
      }
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.read-via-stream"][Instruction::Return]', {
        funcName: "[method]descriptor.read-via-stream",
        paramCount: 0,
        postReturn: false
      });
    }
    function trampoline34(arg0, arg1, arg2) {
      var handle1 = arg0;
      var rep2 = handleTable7[(handle1 << 1) + 1] & ~T_FLAG;
      var rsc0 = captureTable7.get(rep2);
      if (!rsc0) {
        rsc0 = Object.create(Descriptor2.prototype);
        Object.defineProperty(rsc0, symbolRscHandle, { writable: true, value: handle1 });
        Object.defineProperty(rsc0, symbolRscRep, { writable: true, value: rep2 });
      }
      curResourceBorrows.push(rsc0);
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.write-via-stream"] [Instruction::CallInterface] (async? sync, @ enter)');
      const _interface_call_currentTaskID = startCurrentTask(0, false, "[method]descriptor.write-via-stream");
      let ret;
      try {
        ret = { tag: "ok", val: rsc0.writeViaStream(BigInt.asUintN(64, arg1)) };
      } catch (e) {
        ret = { tag: "err", val: getErrorPayload(e) };
      }
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.write-via-stream"] [Instruction::CallInterface] (sync, @ post-call)');
      for (const rsc of curResourceBorrows) {
        rsc[symbolRscHandle] = void 0;
      }
      curResourceBorrows = [];
      endCurrentTask(0);
      var variant5 = ret;
      switch (variant5.tag) {
        case "ok": {
          const e = variant5.val;
          dataView(memory0).setInt8(arg2 + 0, 0, true);
          if (!(e instanceof OutputStream2)) {
            throw new TypeError('Resource error: Not a valid "OutputStream" resource.');
          }
          var handle3 = e[symbolRscHandle];
          if (!handle3) {
            const rep = e[symbolRscRep] || ++captureCnt3;
            captureTable3.set(rep, e);
            handle3 = rscTableCreateOwn(handleTable3, rep);
          }
          dataView(memory0).setInt32(arg2 + 4, handle3, true);
          break;
        }
        case "err": {
          const e = variant5.val;
          dataView(memory0).setInt8(arg2 + 0, 1, true);
          var val4 = e;
          let enum4;
          switch (val4) {
            case "access": {
              enum4 = 0;
              break;
            }
            case "would-block": {
              enum4 = 1;
              break;
            }
            case "already": {
              enum4 = 2;
              break;
            }
            case "bad-descriptor": {
              enum4 = 3;
              break;
            }
            case "busy": {
              enum4 = 4;
              break;
            }
            case "deadlock": {
              enum4 = 5;
              break;
            }
            case "quota": {
              enum4 = 6;
              break;
            }
            case "exist": {
              enum4 = 7;
              break;
            }
            case "file-too-large": {
              enum4 = 8;
              break;
            }
            case "illegal-byte-sequence": {
              enum4 = 9;
              break;
            }
            case "in-progress": {
              enum4 = 10;
              break;
            }
            case "interrupted": {
              enum4 = 11;
              break;
            }
            case "invalid": {
              enum4 = 12;
              break;
            }
            case "io": {
              enum4 = 13;
              break;
            }
            case "is-directory": {
              enum4 = 14;
              break;
            }
            case "loop": {
              enum4 = 15;
              break;
            }
            case "too-many-links": {
              enum4 = 16;
              break;
            }
            case "message-size": {
              enum4 = 17;
              break;
            }
            case "name-too-long": {
              enum4 = 18;
              break;
            }
            case "no-device": {
              enum4 = 19;
              break;
            }
            case "no-entry": {
              enum4 = 20;
              break;
            }
            case "no-lock": {
              enum4 = 21;
              break;
            }
            case "insufficient-memory": {
              enum4 = 22;
              break;
            }
            case "insufficient-space": {
              enum4 = 23;
              break;
            }
            case "not-directory": {
              enum4 = 24;
              break;
            }
            case "not-empty": {
              enum4 = 25;
              break;
            }
            case "not-recoverable": {
              enum4 = 26;
              break;
            }
            case "unsupported": {
              enum4 = 27;
              break;
            }
            case "no-tty": {
              enum4 = 28;
              break;
            }
            case "no-such-device": {
              enum4 = 29;
              break;
            }
            case "overflow": {
              enum4 = 30;
              break;
            }
            case "not-permitted": {
              enum4 = 31;
              break;
            }
            case "pipe": {
              enum4 = 32;
              break;
            }
            case "read-only": {
              enum4 = 33;
              break;
            }
            case "invalid-seek": {
              enum4 = 34;
              break;
            }
            case "text-file-busy": {
              enum4 = 35;
              break;
            }
            case "cross-device": {
              enum4 = 36;
              break;
            }
            default: {
              if (e instanceof Error) {
                console.error(e);
              }
              throw new TypeError(`"${val4}" is not one of the cases of error-code`);
            }
          }
          dataView(memory0).setInt8(arg2 + 4, enum4, true);
          break;
        }
        default: {
          throw new TypeError("invalid variant specified for result");
        }
      }
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.write-via-stream"][Instruction::Return]', {
        funcName: "[method]descriptor.write-via-stream",
        paramCount: 0,
        postReturn: false
      });
    }
    function trampoline35(arg0, arg1) {
      var handle1 = arg0;
      var rep2 = handleTable7[(handle1 << 1) + 1] & ~T_FLAG;
      var rsc0 = captureTable7.get(rep2);
      if (!rsc0) {
        rsc0 = Object.create(Descriptor2.prototype);
        Object.defineProperty(rsc0, symbolRscHandle, { writable: true, value: handle1 });
        Object.defineProperty(rsc0, symbolRscRep, { writable: true, value: rep2 });
      }
      curResourceBorrows.push(rsc0);
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.append-via-stream"] [Instruction::CallInterface] (async? sync, @ enter)');
      const _interface_call_currentTaskID = startCurrentTask(0, false, "[method]descriptor.append-via-stream");
      let ret;
      try {
        ret = { tag: "ok", val: rsc0.appendViaStream() };
      } catch (e) {
        ret = { tag: "err", val: getErrorPayload(e) };
      }
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.append-via-stream"] [Instruction::CallInterface] (sync, @ post-call)');
      for (const rsc of curResourceBorrows) {
        rsc[symbolRscHandle] = void 0;
      }
      curResourceBorrows = [];
      endCurrentTask(0);
      var variant5 = ret;
      switch (variant5.tag) {
        case "ok": {
          const e = variant5.val;
          dataView(memory0).setInt8(arg1 + 0, 0, true);
          if (!(e instanceof OutputStream2)) {
            throw new TypeError('Resource error: Not a valid "OutputStream" resource.');
          }
          var handle3 = e[symbolRscHandle];
          if (!handle3) {
            const rep = e[symbolRscRep] || ++captureCnt3;
            captureTable3.set(rep, e);
            handle3 = rscTableCreateOwn(handleTable3, rep);
          }
          dataView(memory0).setInt32(arg1 + 4, handle3, true);
          break;
        }
        case "err": {
          const e = variant5.val;
          dataView(memory0).setInt8(arg1 + 0, 1, true);
          var val4 = e;
          let enum4;
          switch (val4) {
            case "access": {
              enum4 = 0;
              break;
            }
            case "would-block": {
              enum4 = 1;
              break;
            }
            case "already": {
              enum4 = 2;
              break;
            }
            case "bad-descriptor": {
              enum4 = 3;
              break;
            }
            case "busy": {
              enum4 = 4;
              break;
            }
            case "deadlock": {
              enum4 = 5;
              break;
            }
            case "quota": {
              enum4 = 6;
              break;
            }
            case "exist": {
              enum4 = 7;
              break;
            }
            case "file-too-large": {
              enum4 = 8;
              break;
            }
            case "illegal-byte-sequence": {
              enum4 = 9;
              break;
            }
            case "in-progress": {
              enum4 = 10;
              break;
            }
            case "interrupted": {
              enum4 = 11;
              break;
            }
            case "invalid": {
              enum4 = 12;
              break;
            }
            case "io": {
              enum4 = 13;
              break;
            }
            case "is-directory": {
              enum4 = 14;
              break;
            }
            case "loop": {
              enum4 = 15;
              break;
            }
            case "too-many-links": {
              enum4 = 16;
              break;
            }
            case "message-size": {
              enum4 = 17;
              break;
            }
            case "name-too-long": {
              enum4 = 18;
              break;
            }
            case "no-device": {
              enum4 = 19;
              break;
            }
            case "no-entry": {
              enum4 = 20;
              break;
            }
            case "no-lock": {
              enum4 = 21;
              break;
            }
            case "insufficient-memory": {
              enum4 = 22;
              break;
            }
            case "insufficient-space": {
              enum4 = 23;
              break;
            }
            case "not-directory": {
              enum4 = 24;
              break;
            }
            case "not-empty": {
              enum4 = 25;
              break;
            }
            case "not-recoverable": {
              enum4 = 26;
              break;
            }
            case "unsupported": {
              enum4 = 27;
              break;
            }
            case "no-tty": {
              enum4 = 28;
              break;
            }
            case "no-such-device": {
              enum4 = 29;
              break;
            }
            case "overflow": {
              enum4 = 30;
              break;
            }
            case "not-permitted": {
              enum4 = 31;
              break;
            }
            case "pipe": {
              enum4 = 32;
              break;
            }
            case "read-only": {
              enum4 = 33;
              break;
            }
            case "invalid-seek": {
              enum4 = 34;
              break;
            }
            case "text-file-busy": {
              enum4 = 35;
              break;
            }
            case "cross-device": {
              enum4 = 36;
              break;
            }
            default: {
              if (e instanceof Error) {
                console.error(e);
              }
              throw new TypeError(`"${val4}" is not one of the cases of error-code`);
            }
          }
          dataView(memory0).setInt8(arg1 + 4, enum4, true);
          break;
        }
        default: {
          throw new TypeError("invalid variant specified for result");
        }
      }
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.append-via-stream"][Instruction::Return]', {
        funcName: "[method]descriptor.append-via-stream",
        paramCount: 0,
        postReturn: false
      });
    }
    function trampoline36(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7) {
      var handle1 = arg0;
      var rep2 = handleTable7[(handle1 << 1) + 1] & ~T_FLAG;
      var rsc0 = captureTable7.get(rep2);
      if (!rsc0) {
        rsc0 = Object.create(Descriptor2.prototype);
        Object.defineProperty(rsc0, symbolRscHandle, { writable: true, value: handle1 });
        Object.defineProperty(rsc0, symbolRscRep, { writable: true, value: rep2 });
      }
      curResourceBorrows.push(rsc0);
      let variant3;
      switch (arg1) {
        case 0: {
          variant3 = {
            tag: "no-change"
          };
          break;
        }
        case 1: {
          variant3 = {
            tag: "now"
          };
          break;
        }
        case 2: {
          variant3 = {
            tag: "timestamp",
            val: {
              seconds: BigInt.asUintN(64, arg2),
              nanoseconds: arg3 >>> 0
            }
          };
          break;
        }
        default: {
          throw new TypeError("invalid variant discriminant for NewTimestamp");
        }
      }
      let variant4;
      switch (arg4) {
        case 0: {
          variant4 = {
            tag: "no-change"
          };
          break;
        }
        case 1: {
          variant4 = {
            tag: "now"
          };
          break;
        }
        case 2: {
          variant4 = {
            tag: "timestamp",
            val: {
              seconds: BigInt.asUintN(64, arg5),
              nanoseconds: arg6 >>> 0
            }
          };
          break;
        }
        default: {
          throw new TypeError("invalid variant discriminant for NewTimestamp");
        }
      }
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.set-times"] [Instruction::CallInterface] (async? sync, @ enter)');
      const _interface_call_currentTaskID = startCurrentTask(0, false, "[method]descriptor.set-times");
      let ret;
      try {
        ret = { tag: "ok", val: rsc0.setTimes(variant3, variant4) };
      } catch (e) {
        ret = { tag: "err", val: getErrorPayload(e) };
      }
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.set-times"] [Instruction::CallInterface] (sync, @ post-call)');
      for (const rsc of curResourceBorrows) {
        rsc[symbolRscHandle] = void 0;
      }
      curResourceBorrows = [];
      endCurrentTask(0);
      var variant6 = ret;
      switch (variant6.tag) {
        case "ok": {
          const e = variant6.val;
          dataView(memory0).setInt8(arg7 + 0, 0, true);
          break;
        }
        case "err": {
          const e = variant6.val;
          dataView(memory0).setInt8(arg7 + 0, 1, true);
          var val5 = e;
          let enum5;
          switch (val5) {
            case "access": {
              enum5 = 0;
              break;
            }
            case "would-block": {
              enum5 = 1;
              break;
            }
            case "already": {
              enum5 = 2;
              break;
            }
            case "bad-descriptor": {
              enum5 = 3;
              break;
            }
            case "busy": {
              enum5 = 4;
              break;
            }
            case "deadlock": {
              enum5 = 5;
              break;
            }
            case "quota": {
              enum5 = 6;
              break;
            }
            case "exist": {
              enum5 = 7;
              break;
            }
            case "file-too-large": {
              enum5 = 8;
              break;
            }
            case "illegal-byte-sequence": {
              enum5 = 9;
              break;
            }
            case "in-progress": {
              enum5 = 10;
              break;
            }
            case "interrupted": {
              enum5 = 11;
              break;
            }
            case "invalid": {
              enum5 = 12;
              break;
            }
            case "io": {
              enum5 = 13;
              break;
            }
            case "is-directory": {
              enum5 = 14;
              break;
            }
            case "loop": {
              enum5 = 15;
              break;
            }
            case "too-many-links": {
              enum5 = 16;
              break;
            }
            case "message-size": {
              enum5 = 17;
              break;
            }
            case "name-too-long": {
              enum5 = 18;
              break;
            }
            case "no-device": {
              enum5 = 19;
              break;
            }
            case "no-entry": {
              enum5 = 20;
              break;
            }
            case "no-lock": {
              enum5 = 21;
              break;
            }
            case "insufficient-memory": {
              enum5 = 22;
              break;
            }
            case "insufficient-space": {
              enum5 = 23;
              break;
            }
            case "not-directory": {
              enum5 = 24;
              break;
            }
            case "not-empty": {
              enum5 = 25;
              break;
            }
            case "not-recoverable": {
              enum5 = 26;
              break;
            }
            case "unsupported": {
              enum5 = 27;
              break;
            }
            case "no-tty": {
              enum5 = 28;
              break;
            }
            case "no-such-device": {
              enum5 = 29;
              break;
            }
            case "overflow": {
              enum5 = 30;
              break;
            }
            case "not-permitted": {
              enum5 = 31;
              break;
            }
            case "pipe": {
              enum5 = 32;
              break;
            }
            case "read-only": {
              enum5 = 33;
              break;
            }
            case "invalid-seek": {
              enum5 = 34;
              break;
            }
            case "text-file-busy": {
              enum5 = 35;
              break;
            }
            case "cross-device": {
              enum5 = 36;
              break;
            }
            default: {
              if (e instanceof Error) {
                console.error(e);
              }
              throw new TypeError(`"${val5}" is not one of the cases of error-code`);
            }
          }
          dataView(memory0).setInt8(arg7 + 1, enum5, true);
          break;
        }
        default: {
          throw new TypeError("invalid variant specified for result");
        }
      }
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.set-times"][Instruction::Return]', {
        funcName: "[method]descriptor.set-times",
        paramCount: 0,
        postReturn: false
      });
    }
    function trampoline37(arg0, arg1, arg2, arg3) {
      var handle1 = arg0;
      var rep2 = handleTable7[(handle1 << 1) + 1] & ~T_FLAG;
      var rsc0 = captureTable7.get(rep2);
      if (!rsc0) {
        rsc0 = Object.create(Descriptor2.prototype);
        Object.defineProperty(rsc0, symbolRscHandle, { writable: true, value: handle1 });
        Object.defineProperty(rsc0, symbolRscRep, { writable: true, value: rep2 });
      }
      curResourceBorrows.push(rsc0);
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.read"] [Instruction::CallInterface] (async? sync, @ enter)');
      const _interface_call_currentTaskID = startCurrentTask(0, false, "[method]descriptor.read");
      let ret;
      try {
        ret = { tag: "ok", val: rsc0.read(BigInt.asUintN(64, arg1), BigInt.asUintN(64, arg2)) };
      } catch (e) {
        ret = { tag: "err", val: getErrorPayload(e) };
      }
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.read"] [Instruction::CallInterface] (sync, @ post-call)');
      for (const rsc of curResourceBorrows) {
        rsc[symbolRscHandle] = void 0;
      }
      curResourceBorrows = [];
      endCurrentTask(0);
      var variant6 = ret;
      switch (variant6.tag) {
        case "ok": {
          const e = variant6.val;
          dataView(memory0).setInt8(arg3 + 0, 0, true);
          var [tuple3_0, tuple3_1] = e;
          var val4 = tuple3_0;
          var len4 = val4.byteLength;
          var ptr4 = realloc0(0, 0, 1, len4 * 1);
          var src4 = new Uint8Array(val4.buffer || val4, val4.byteOffset, len4 * 1);
          new Uint8Array(memory0.buffer, ptr4, len4 * 1).set(src4);
          dataView(memory0).setUint32(arg3 + 8, len4, true);
          dataView(memory0).setUint32(arg3 + 4, ptr4, true);
          dataView(memory0).setInt8(arg3 + 12, tuple3_1 ? 1 : 0, true);
          break;
        }
        case "err": {
          const e = variant6.val;
          dataView(memory0).setInt8(arg3 + 0, 1, true);
          var val5 = e;
          let enum5;
          switch (val5) {
            case "access": {
              enum5 = 0;
              break;
            }
            case "would-block": {
              enum5 = 1;
              break;
            }
            case "already": {
              enum5 = 2;
              break;
            }
            case "bad-descriptor": {
              enum5 = 3;
              break;
            }
            case "busy": {
              enum5 = 4;
              break;
            }
            case "deadlock": {
              enum5 = 5;
              break;
            }
            case "quota": {
              enum5 = 6;
              break;
            }
            case "exist": {
              enum5 = 7;
              break;
            }
            case "file-too-large": {
              enum5 = 8;
              break;
            }
            case "illegal-byte-sequence": {
              enum5 = 9;
              break;
            }
            case "in-progress": {
              enum5 = 10;
              break;
            }
            case "interrupted": {
              enum5 = 11;
              break;
            }
            case "invalid": {
              enum5 = 12;
              break;
            }
            case "io": {
              enum5 = 13;
              break;
            }
            case "is-directory": {
              enum5 = 14;
              break;
            }
            case "loop": {
              enum5 = 15;
              break;
            }
            case "too-many-links": {
              enum5 = 16;
              break;
            }
            case "message-size": {
              enum5 = 17;
              break;
            }
            case "name-too-long": {
              enum5 = 18;
              break;
            }
            case "no-device": {
              enum5 = 19;
              break;
            }
            case "no-entry": {
              enum5 = 20;
              break;
            }
            case "no-lock": {
              enum5 = 21;
              break;
            }
            case "insufficient-memory": {
              enum5 = 22;
              break;
            }
            case "insufficient-space": {
              enum5 = 23;
              break;
            }
            case "not-directory": {
              enum5 = 24;
              break;
            }
            case "not-empty": {
              enum5 = 25;
              break;
            }
            case "not-recoverable": {
              enum5 = 26;
              break;
            }
            case "unsupported": {
              enum5 = 27;
              break;
            }
            case "no-tty": {
              enum5 = 28;
              break;
            }
            case "no-such-device": {
              enum5 = 29;
              break;
            }
            case "overflow": {
              enum5 = 30;
              break;
            }
            case "not-permitted": {
              enum5 = 31;
              break;
            }
            case "pipe": {
              enum5 = 32;
              break;
            }
            case "read-only": {
              enum5 = 33;
              break;
            }
            case "invalid-seek": {
              enum5 = 34;
              break;
            }
            case "text-file-busy": {
              enum5 = 35;
              break;
            }
            case "cross-device": {
              enum5 = 36;
              break;
            }
            default: {
              if (e instanceof Error) {
                console.error(e);
              }
              throw new TypeError(`"${val5}" is not one of the cases of error-code`);
            }
          }
          dataView(memory0).setInt8(arg3 + 4, enum5, true);
          break;
        }
        default: {
          throw new TypeError("invalid variant specified for result");
        }
      }
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.read"][Instruction::Return]', {
        funcName: "[method]descriptor.read",
        paramCount: 0,
        postReturn: false
      });
    }
    const handleTable6 = [T_FLAG, 0];
    const captureTable6 = /* @__PURE__ */ new Map();
    let captureCnt6 = 0;
    handleTables[6] = handleTable6;
    function trampoline38(arg0, arg1) {
      var handle1 = arg0;
      var rep2 = handleTable7[(handle1 << 1) + 1] & ~T_FLAG;
      var rsc0 = captureTable7.get(rep2);
      if (!rsc0) {
        rsc0 = Object.create(Descriptor2.prototype);
        Object.defineProperty(rsc0, symbolRscHandle, { writable: true, value: handle1 });
        Object.defineProperty(rsc0, symbolRscRep, { writable: true, value: rep2 });
      }
      curResourceBorrows.push(rsc0);
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.read-directory"] [Instruction::CallInterface] (async? sync, @ enter)');
      const _interface_call_currentTaskID = startCurrentTask(0, false, "[method]descriptor.read-directory");
      let ret;
      try {
        ret = { tag: "ok", val: rsc0.readDirectory() };
      } catch (e) {
        ret = { tag: "err", val: getErrorPayload(e) };
      }
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.read-directory"] [Instruction::CallInterface] (sync, @ post-call)');
      for (const rsc of curResourceBorrows) {
        rsc[symbolRscHandle] = void 0;
      }
      curResourceBorrows = [];
      endCurrentTask(0);
      var variant5 = ret;
      switch (variant5.tag) {
        case "ok": {
          const e = variant5.val;
          dataView(memory0).setInt8(arg1 + 0, 0, true);
          if (!(e instanceof DirectoryEntryStream2)) {
            throw new TypeError('Resource error: Not a valid "DirectoryEntryStream" resource.');
          }
          var handle3 = e[symbolRscHandle];
          if (!handle3) {
            const rep = e[symbolRscRep] || ++captureCnt6;
            captureTable6.set(rep, e);
            handle3 = rscTableCreateOwn(handleTable6, rep);
          }
          dataView(memory0).setInt32(arg1 + 4, handle3, true);
          break;
        }
        case "err": {
          const e = variant5.val;
          dataView(memory0).setInt8(arg1 + 0, 1, true);
          var val4 = e;
          let enum4;
          switch (val4) {
            case "access": {
              enum4 = 0;
              break;
            }
            case "would-block": {
              enum4 = 1;
              break;
            }
            case "already": {
              enum4 = 2;
              break;
            }
            case "bad-descriptor": {
              enum4 = 3;
              break;
            }
            case "busy": {
              enum4 = 4;
              break;
            }
            case "deadlock": {
              enum4 = 5;
              break;
            }
            case "quota": {
              enum4 = 6;
              break;
            }
            case "exist": {
              enum4 = 7;
              break;
            }
            case "file-too-large": {
              enum4 = 8;
              break;
            }
            case "illegal-byte-sequence": {
              enum4 = 9;
              break;
            }
            case "in-progress": {
              enum4 = 10;
              break;
            }
            case "interrupted": {
              enum4 = 11;
              break;
            }
            case "invalid": {
              enum4 = 12;
              break;
            }
            case "io": {
              enum4 = 13;
              break;
            }
            case "is-directory": {
              enum4 = 14;
              break;
            }
            case "loop": {
              enum4 = 15;
              break;
            }
            case "too-many-links": {
              enum4 = 16;
              break;
            }
            case "message-size": {
              enum4 = 17;
              break;
            }
            case "name-too-long": {
              enum4 = 18;
              break;
            }
            case "no-device": {
              enum4 = 19;
              break;
            }
            case "no-entry": {
              enum4 = 20;
              break;
            }
            case "no-lock": {
              enum4 = 21;
              break;
            }
            case "insufficient-memory": {
              enum4 = 22;
              break;
            }
            case "insufficient-space": {
              enum4 = 23;
              break;
            }
            case "not-directory": {
              enum4 = 24;
              break;
            }
            case "not-empty": {
              enum4 = 25;
              break;
            }
            case "not-recoverable": {
              enum4 = 26;
              break;
            }
            case "unsupported": {
              enum4 = 27;
              break;
            }
            case "no-tty": {
              enum4 = 28;
              break;
            }
            case "no-such-device": {
              enum4 = 29;
              break;
            }
            case "overflow": {
              enum4 = 30;
              break;
            }
            case "not-permitted": {
              enum4 = 31;
              break;
            }
            case "pipe": {
              enum4 = 32;
              break;
            }
            case "read-only": {
              enum4 = 33;
              break;
            }
            case "invalid-seek": {
              enum4 = 34;
              break;
            }
            case "text-file-busy": {
              enum4 = 35;
              break;
            }
            case "cross-device": {
              enum4 = 36;
              break;
            }
            default: {
              if (e instanceof Error) {
                console.error(e);
              }
              throw new TypeError(`"${val4}" is not one of the cases of error-code`);
            }
          }
          dataView(memory0).setInt8(arg1 + 4, enum4, true);
          break;
        }
        default: {
          throw new TypeError("invalid variant specified for result");
        }
      }
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.read-directory"][Instruction::Return]', {
        funcName: "[method]descriptor.read-directory",
        paramCount: 0,
        postReturn: false
      });
    }
    function trampoline39(arg0, arg1) {
      var handle1 = arg0;
      var rep2 = handleTable7[(handle1 << 1) + 1] & ~T_FLAG;
      var rsc0 = captureTable7.get(rep2);
      if (!rsc0) {
        rsc0 = Object.create(Descriptor2.prototype);
        Object.defineProperty(rsc0, symbolRscHandle, { writable: true, value: handle1 });
        Object.defineProperty(rsc0, symbolRscRep, { writable: true, value: rep2 });
      }
      curResourceBorrows.push(rsc0);
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.stat"] [Instruction::CallInterface] (async? sync, @ enter)');
      const _interface_call_currentTaskID = startCurrentTask(0, false, "[method]descriptor.stat");
      let ret;
      try {
        ret = { tag: "ok", val: rsc0.stat() };
      } catch (e) {
        ret = { tag: "err", val: getErrorPayload(e) };
      }
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.stat"] [Instruction::CallInterface] (sync, @ post-call)');
      for (const rsc of curResourceBorrows) {
        rsc[symbolRscHandle] = void 0;
      }
      curResourceBorrows = [];
      endCurrentTask(0);
      var variant12 = ret;
      switch (variant12.tag) {
        case "ok": {
          const e = variant12.val;
          dataView(memory0).setInt8(arg1 + 0, 0, true);
          var { type: v3_0, linkCount: v3_1, size: v3_2, dataAccessTimestamp: v3_3, dataModificationTimestamp: v3_4, statusChangeTimestamp: v3_5 } = e;
          var val4 = v3_0;
          let enum4;
          switch (val4) {
            case "unknown": {
              enum4 = 0;
              break;
            }
            case "block-device": {
              enum4 = 1;
              break;
            }
            case "character-device": {
              enum4 = 2;
              break;
            }
            case "directory": {
              enum4 = 3;
              break;
            }
            case "fifo": {
              enum4 = 4;
              break;
            }
            case "symbolic-link": {
              enum4 = 5;
              break;
            }
            case "regular-file": {
              enum4 = 6;
              break;
            }
            case "socket": {
              enum4 = 7;
              break;
            }
            default: {
              if (v3_0 instanceof Error) {
                console.error(v3_0);
              }
              throw new TypeError(`"${val4}" is not one of the cases of descriptor-type`);
            }
          }
          dataView(memory0).setInt8(arg1 + 8, enum4, true);
          dataView(memory0).setBigInt64(arg1 + 16, toUint64(v3_1), true);
          dataView(memory0).setBigInt64(arg1 + 24, toUint64(v3_2), true);
          var variant6 = v3_3;
          if (variant6 === null || variant6 === void 0) {
            dataView(memory0).setInt8(arg1 + 32, 0, true);
          } else {
            const e2 = variant6;
            dataView(memory0).setInt8(arg1 + 32, 1, true);
            var { seconds: v5_0, nanoseconds: v5_1 } = e2;
            dataView(memory0).setBigInt64(arg1 + 40, toUint64(v5_0), true);
            dataView(memory0).setInt32(arg1 + 48, toUint32(v5_1), true);
          }
          var variant8 = v3_4;
          if (variant8 === null || variant8 === void 0) {
            dataView(memory0).setInt8(arg1 + 56, 0, true);
          } else {
            const e2 = variant8;
            dataView(memory0).setInt8(arg1 + 56, 1, true);
            var { seconds: v7_0, nanoseconds: v7_1 } = e2;
            dataView(memory0).setBigInt64(arg1 + 64, toUint64(v7_0), true);
            dataView(memory0).setInt32(arg1 + 72, toUint32(v7_1), true);
          }
          var variant10 = v3_5;
          if (variant10 === null || variant10 === void 0) {
            dataView(memory0).setInt8(arg1 + 80, 0, true);
          } else {
            const e2 = variant10;
            dataView(memory0).setInt8(arg1 + 80, 1, true);
            var { seconds: v9_0, nanoseconds: v9_1 } = e2;
            dataView(memory0).setBigInt64(arg1 + 88, toUint64(v9_0), true);
            dataView(memory0).setInt32(arg1 + 96, toUint32(v9_1), true);
          }
          break;
        }
        case "err": {
          const e = variant12.val;
          dataView(memory0).setInt8(arg1 + 0, 1, true);
          var val11 = e;
          let enum11;
          switch (val11) {
            case "access": {
              enum11 = 0;
              break;
            }
            case "would-block": {
              enum11 = 1;
              break;
            }
            case "already": {
              enum11 = 2;
              break;
            }
            case "bad-descriptor": {
              enum11 = 3;
              break;
            }
            case "busy": {
              enum11 = 4;
              break;
            }
            case "deadlock": {
              enum11 = 5;
              break;
            }
            case "quota": {
              enum11 = 6;
              break;
            }
            case "exist": {
              enum11 = 7;
              break;
            }
            case "file-too-large": {
              enum11 = 8;
              break;
            }
            case "illegal-byte-sequence": {
              enum11 = 9;
              break;
            }
            case "in-progress": {
              enum11 = 10;
              break;
            }
            case "interrupted": {
              enum11 = 11;
              break;
            }
            case "invalid": {
              enum11 = 12;
              break;
            }
            case "io": {
              enum11 = 13;
              break;
            }
            case "is-directory": {
              enum11 = 14;
              break;
            }
            case "loop": {
              enum11 = 15;
              break;
            }
            case "too-many-links": {
              enum11 = 16;
              break;
            }
            case "message-size": {
              enum11 = 17;
              break;
            }
            case "name-too-long": {
              enum11 = 18;
              break;
            }
            case "no-device": {
              enum11 = 19;
              break;
            }
            case "no-entry": {
              enum11 = 20;
              break;
            }
            case "no-lock": {
              enum11 = 21;
              break;
            }
            case "insufficient-memory": {
              enum11 = 22;
              break;
            }
            case "insufficient-space": {
              enum11 = 23;
              break;
            }
            case "not-directory": {
              enum11 = 24;
              break;
            }
            case "not-empty": {
              enum11 = 25;
              break;
            }
            case "not-recoverable": {
              enum11 = 26;
              break;
            }
            case "unsupported": {
              enum11 = 27;
              break;
            }
            case "no-tty": {
              enum11 = 28;
              break;
            }
            case "no-such-device": {
              enum11 = 29;
              break;
            }
            case "overflow": {
              enum11 = 30;
              break;
            }
            case "not-permitted": {
              enum11 = 31;
              break;
            }
            case "pipe": {
              enum11 = 32;
              break;
            }
            case "read-only": {
              enum11 = 33;
              break;
            }
            case "invalid-seek": {
              enum11 = 34;
              break;
            }
            case "text-file-busy": {
              enum11 = 35;
              break;
            }
            case "cross-device": {
              enum11 = 36;
              break;
            }
            default: {
              if (e instanceof Error) {
                console.error(e);
              }
              throw new TypeError(`"${val11}" is not one of the cases of error-code`);
            }
          }
          dataView(memory0).setInt8(arg1 + 8, enum11, true);
          break;
        }
        default: {
          throw new TypeError("invalid variant specified for result");
        }
      }
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.stat"][Instruction::Return]', {
        funcName: "[method]descriptor.stat",
        paramCount: 0,
        postReturn: false
      });
    }
    function trampoline40(arg0, arg1, arg2, arg3, arg4) {
      var handle1 = arg0;
      var rep2 = handleTable7[(handle1 << 1) + 1] & ~T_FLAG;
      var rsc0 = captureTable7.get(rep2);
      if (!rsc0) {
        rsc0 = Object.create(Descriptor2.prototype);
        Object.defineProperty(rsc0, symbolRscHandle, { writable: true, value: handle1 });
        Object.defineProperty(rsc0, symbolRscRep, { writable: true, value: rep2 });
      }
      curResourceBorrows.push(rsc0);
      if ((arg1 & 4294967294) !== 0) {
        throw new TypeError("flags have extraneous bits set");
      }
      var flags3 = {
        symlinkFollow: Boolean(arg1 & 1)
      };
      var ptr4 = arg2;
      var len4 = arg3;
      var result4 = utf8Decoder.decode(new Uint8Array(memory0.buffer, ptr4, len4));
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.stat-at"] [Instruction::CallInterface] (async? sync, @ enter)');
      const _interface_call_currentTaskID = startCurrentTask(0, false, "[method]descriptor.stat-at");
      let ret;
      try {
        ret = { tag: "ok", val: rsc0.statAt(flags3, result4) };
      } catch (e) {
        ret = { tag: "err", val: getErrorPayload(e) };
      }
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.stat-at"] [Instruction::CallInterface] (sync, @ post-call)');
      for (const rsc of curResourceBorrows) {
        rsc[symbolRscHandle] = void 0;
      }
      curResourceBorrows = [];
      endCurrentTask(0);
      var variant14 = ret;
      switch (variant14.tag) {
        case "ok": {
          const e = variant14.val;
          dataView(memory0).setInt8(arg4 + 0, 0, true);
          var { type: v5_0, linkCount: v5_1, size: v5_2, dataAccessTimestamp: v5_3, dataModificationTimestamp: v5_4, statusChangeTimestamp: v5_5 } = e;
          var val6 = v5_0;
          let enum6;
          switch (val6) {
            case "unknown": {
              enum6 = 0;
              break;
            }
            case "block-device": {
              enum6 = 1;
              break;
            }
            case "character-device": {
              enum6 = 2;
              break;
            }
            case "directory": {
              enum6 = 3;
              break;
            }
            case "fifo": {
              enum6 = 4;
              break;
            }
            case "symbolic-link": {
              enum6 = 5;
              break;
            }
            case "regular-file": {
              enum6 = 6;
              break;
            }
            case "socket": {
              enum6 = 7;
              break;
            }
            default: {
              if (v5_0 instanceof Error) {
                console.error(v5_0);
              }
              throw new TypeError(`"${val6}" is not one of the cases of descriptor-type`);
            }
          }
          dataView(memory0).setInt8(arg4 + 8, enum6, true);
          dataView(memory0).setBigInt64(arg4 + 16, toUint64(v5_1), true);
          dataView(memory0).setBigInt64(arg4 + 24, toUint64(v5_2), true);
          var variant8 = v5_3;
          if (variant8 === null || variant8 === void 0) {
            dataView(memory0).setInt8(arg4 + 32, 0, true);
          } else {
            const e2 = variant8;
            dataView(memory0).setInt8(arg4 + 32, 1, true);
            var { seconds: v7_0, nanoseconds: v7_1 } = e2;
            dataView(memory0).setBigInt64(arg4 + 40, toUint64(v7_0), true);
            dataView(memory0).setInt32(arg4 + 48, toUint32(v7_1), true);
          }
          var variant10 = v5_4;
          if (variant10 === null || variant10 === void 0) {
            dataView(memory0).setInt8(arg4 + 56, 0, true);
          } else {
            const e2 = variant10;
            dataView(memory0).setInt8(arg4 + 56, 1, true);
            var { seconds: v9_0, nanoseconds: v9_1 } = e2;
            dataView(memory0).setBigInt64(arg4 + 64, toUint64(v9_0), true);
            dataView(memory0).setInt32(arg4 + 72, toUint32(v9_1), true);
          }
          var variant12 = v5_5;
          if (variant12 === null || variant12 === void 0) {
            dataView(memory0).setInt8(arg4 + 80, 0, true);
          } else {
            const e2 = variant12;
            dataView(memory0).setInt8(arg4 + 80, 1, true);
            var { seconds: v11_0, nanoseconds: v11_1 } = e2;
            dataView(memory0).setBigInt64(arg4 + 88, toUint64(v11_0), true);
            dataView(memory0).setInt32(arg4 + 96, toUint32(v11_1), true);
          }
          break;
        }
        case "err": {
          const e = variant14.val;
          dataView(memory0).setInt8(arg4 + 0, 1, true);
          var val13 = e;
          let enum13;
          switch (val13) {
            case "access": {
              enum13 = 0;
              break;
            }
            case "would-block": {
              enum13 = 1;
              break;
            }
            case "already": {
              enum13 = 2;
              break;
            }
            case "bad-descriptor": {
              enum13 = 3;
              break;
            }
            case "busy": {
              enum13 = 4;
              break;
            }
            case "deadlock": {
              enum13 = 5;
              break;
            }
            case "quota": {
              enum13 = 6;
              break;
            }
            case "exist": {
              enum13 = 7;
              break;
            }
            case "file-too-large": {
              enum13 = 8;
              break;
            }
            case "illegal-byte-sequence": {
              enum13 = 9;
              break;
            }
            case "in-progress": {
              enum13 = 10;
              break;
            }
            case "interrupted": {
              enum13 = 11;
              break;
            }
            case "invalid": {
              enum13 = 12;
              break;
            }
            case "io": {
              enum13 = 13;
              break;
            }
            case "is-directory": {
              enum13 = 14;
              break;
            }
            case "loop": {
              enum13 = 15;
              break;
            }
            case "too-many-links": {
              enum13 = 16;
              break;
            }
            case "message-size": {
              enum13 = 17;
              break;
            }
            case "name-too-long": {
              enum13 = 18;
              break;
            }
            case "no-device": {
              enum13 = 19;
              break;
            }
            case "no-entry": {
              enum13 = 20;
              break;
            }
            case "no-lock": {
              enum13 = 21;
              break;
            }
            case "insufficient-memory": {
              enum13 = 22;
              break;
            }
            case "insufficient-space": {
              enum13 = 23;
              break;
            }
            case "not-directory": {
              enum13 = 24;
              break;
            }
            case "not-empty": {
              enum13 = 25;
              break;
            }
            case "not-recoverable": {
              enum13 = 26;
              break;
            }
            case "unsupported": {
              enum13 = 27;
              break;
            }
            case "no-tty": {
              enum13 = 28;
              break;
            }
            case "no-such-device": {
              enum13 = 29;
              break;
            }
            case "overflow": {
              enum13 = 30;
              break;
            }
            case "not-permitted": {
              enum13 = 31;
              break;
            }
            case "pipe": {
              enum13 = 32;
              break;
            }
            case "read-only": {
              enum13 = 33;
              break;
            }
            case "invalid-seek": {
              enum13 = 34;
              break;
            }
            case "text-file-busy": {
              enum13 = 35;
              break;
            }
            case "cross-device": {
              enum13 = 36;
              break;
            }
            default: {
              if (e instanceof Error) {
                console.error(e);
              }
              throw new TypeError(`"${val13}" is not one of the cases of error-code`);
            }
          }
          dataView(memory0).setInt8(arg4 + 8, enum13, true);
          break;
        }
        default: {
          throw new TypeError("invalid variant specified for result");
        }
      }
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.stat-at"][Instruction::Return]', {
        funcName: "[method]descriptor.stat-at",
        paramCount: 0,
        postReturn: false
      });
    }
    function trampoline41(arg0, arg1, arg2, arg3, arg4, arg5, arg6) {
      var handle1 = arg0;
      var rep2 = handleTable7[(handle1 << 1) + 1] & ~T_FLAG;
      var rsc0 = captureTable7.get(rep2);
      if (!rsc0) {
        rsc0 = Object.create(Descriptor2.prototype);
        Object.defineProperty(rsc0, symbolRscHandle, { writable: true, value: handle1 });
        Object.defineProperty(rsc0, symbolRscRep, { writable: true, value: rep2 });
      }
      curResourceBorrows.push(rsc0);
      if ((arg1 & 4294967294) !== 0) {
        throw new TypeError("flags have extraneous bits set");
      }
      var flags3 = {
        symlinkFollow: Boolean(arg1 & 1)
      };
      var ptr4 = arg2;
      var len4 = arg3;
      var result4 = utf8Decoder.decode(new Uint8Array(memory0.buffer, ptr4, len4));
      if ((arg4 & 4294967280) !== 0) {
        throw new TypeError("flags have extraneous bits set");
      }
      var flags5 = {
        create: Boolean(arg4 & 1),
        directory: Boolean(arg4 & 2),
        exclusive: Boolean(arg4 & 4),
        truncate: Boolean(arg4 & 8)
      };
      if ((arg5 & 4294967232) !== 0) {
        throw new TypeError("flags have extraneous bits set");
      }
      var flags6 = {
        read: Boolean(arg5 & 1),
        write: Boolean(arg5 & 2),
        fileIntegritySync: Boolean(arg5 & 4),
        dataIntegritySync: Boolean(arg5 & 8),
        requestedWriteSync: Boolean(arg5 & 16),
        mutateDirectory: Boolean(arg5 & 32)
      };
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.open-at"] [Instruction::CallInterface] (async? sync, @ enter)');
      const _interface_call_currentTaskID = startCurrentTask(0, false, "[method]descriptor.open-at");
      let ret;
      try {
        ret = { tag: "ok", val: rsc0.openAt(flags3, result4, flags5, flags6) };
      } catch (e) {
        ret = { tag: "err", val: getErrorPayload(e) };
      }
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.open-at"] [Instruction::CallInterface] (sync, @ post-call)');
      for (const rsc of curResourceBorrows) {
        rsc[symbolRscHandle] = void 0;
      }
      curResourceBorrows = [];
      endCurrentTask(0);
      var variant9 = ret;
      switch (variant9.tag) {
        case "ok": {
          const e = variant9.val;
          dataView(memory0).setInt8(arg6 + 0, 0, true);
          if (!(e instanceof Descriptor2)) {
            throw new TypeError('Resource error: Not a valid "Descriptor" resource.');
          }
          var handle7 = e[symbolRscHandle];
          if (!handle7) {
            const rep = e[symbolRscRep] || ++captureCnt7;
            captureTable7.set(rep, e);
            handle7 = rscTableCreateOwn(handleTable7, rep);
          }
          dataView(memory0).setInt32(arg6 + 4, handle7, true);
          break;
        }
        case "err": {
          const e = variant9.val;
          dataView(memory0).setInt8(arg6 + 0, 1, true);
          var val8 = e;
          let enum8;
          switch (val8) {
            case "access": {
              enum8 = 0;
              break;
            }
            case "would-block": {
              enum8 = 1;
              break;
            }
            case "already": {
              enum8 = 2;
              break;
            }
            case "bad-descriptor": {
              enum8 = 3;
              break;
            }
            case "busy": {
              enum8 = 4;
              break;
            }
            case "deadlock": {
              enum8 = 5;
              break;
            }
            case "quota": {
              enum8 = 6;
              break;
            }
            case "exist": {
              enum8 = 7;
              break;
            }
            case "file-too-large": {
              enum8 = 8;
              break;
            }
            case "illegal-byte-sequence": {
              enum8 = 9;
              break;
            }
            case "in-progress": {
              enum8 = 10;
              break;
            }
            case "interrupted": {
              enum8 = 11;
              break;
            }
            case "invalid": {
              enum8 = 12;
              break;
            }
            case "io": {
              enum8 = 13;
              break;
            }
            case "is-directory": {
              enum8 = 14;
              break;
            }
            case "loop": {
              enum8 = 15;
              break;
            }
            case "too-many-links": {
              enum8 = 16;
              break;
            }
            case "message-size": {
              enum8 = 17;
              break;
            }
            case "name-too-long": {
              enum8 = 18;
              break;
            }
            case "no-device": {
              enum8 = 19;
              break;
            }
            case "no-entry": {
              enum8 = 20;
              break;
            }
            case "no-lock": {
              enum8 = 21;
              break;
            }
            case "insufficient-memory": {
              enum8 = 22;
              break;
            }
            case "insufficient-space": {
              enum8 = 23;
              break;
            }
            case "not-directory": {
              enum8 = 24;
              break;
            }
            case "not-empty": {
              enum8 = 25;
              break;
            }
            case "not-recoverable": {
              enum8 = 26;
              break;
            }
            case "unsupported": {
              enum8 = 27;
              break;
            }
            case "no-tty": {
              enum8 = 28;
              break;
            }
            case "no-such-device": {
              enum8 = 29;
              break;
            }
            case "overflow": {
              enum8 = 30;
              break;
            }
            case "not-permitted": {
              enum8 = 31;
              break;
            }
            case "pipe": {
              enum8 = 32;
              break;
            }
            case "read-only": {
              enum8 = 33;
              break;
            }
            case "invalid-seek": {
              enum8 = 34;
              break;
            }
            case "text-file-busy": {
              enum8 = 35;
              break;
            }
            case "cross-device": {
              enum8 = 36;
              break;
            }
            default: {
              if (e instanceof Error) {
                console.error(e);
              }
              throw new TypeError(`"${val8}" is not one of the cases of error-code`);
            }
          }
          dataView(memory0).setInt8(arg6 + 4, enum8, true);
          break;
        }
        default: {
          throw new TypeError("invalid variant specified for result");
        }
      }
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]descriptor.open-at"][Instruction::Return]', {
        funcName: "[method]descriptor.open-at",
        paramCount: 0,
        postReturn: false
      });
    }
    function trampoline42(arg0, arg1) {
      var handle1 = arg0;
      var rep2 = handleTable6[(handle1 << 1) + 1] & ~T_FLAG;
      var rsc0 = captureTable6.get(rep2);
      if (!rsc0) {
        rsc0 = Object.create(DirectoryEntryStream2.prototype);
        Object.defineProperty(rsc0, symbolRscHandle, { writable: true, value: handle1 });
        Object.defineProperty(rsc0, symbolRscRep, { writable: true, value: rep2 });
      }
      curResourceBorrows.push(rsc0);
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]directory-entry-stream.read-directory-entry"] [Instruction::CallInterface] (async? sync, @ enter)');
      const _interface_call_currentTaskID = startCurrentTask(0, false, "[method]directory-entry-stream.read-directory-entry");
      let ret;
      try {
        ret = { tag: "ok", val: rsc0.readDirectoryEntry() };
      } catch (e) {
        ret = { tag: "err", val: getErrorPayload(e) };
      }
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]directory-entry-stream.read-directory-entry"] [Instruction::CallInterface] (sync, @ post-call)');
      for (const rsc of curResourceBorrows) {
        rsc[symbolRscHandle] = void 0;
      }
      curResourceBorrows = [];
      endCurrentTask(0);
      var variant8 = ret;
      switch (variant8.tag) {
        case "ok": {
          const e = variant8.val;
          dataView(memory0).setInt8(arg1 + 0, 0, true);
          var variant6 = e;
          if (variant6 === null || variant6 === void 0) {
            dataView(memory0).setInt8(arg1 + 4, 0, true);
          } else {
            const e2 = variant6;
            dataView(memory0).setInt8(arg1 + 4, 1, true);
            var { type: v3_0, name: v3_1 } = e2;
            var val4 = v3_0;
            let enum4;
            switch (val4) {
              case "unknown": {
                enum4 = 0;
                break;
              }
              case "block-device": {
                enum4 = 1;
                break;
              }
              case "character-device": {
                enum4 = 2;
                break;
              }
              case "directory": {
                enum4 = 3;
                break;
              }
              case "fifo": {
                enum4 = 4;
                break;
              }
              case "symbolic-link": {
                enum4 = 5;
                break;
              }
              case "regular-file": {
                enum4 = 6;
                break;
              }
              case "socket": {
                enum4 = 7;
                break;
              }
              default: {
                if (v3_0 instanceof Error) {
                  console.error(v3_0);
                }
                throw new TypeError(`"${val4}" is not one of the cases of descriptor-type`);
              }
            }
            dataView(memory0).setInt8(arg1 + 8, enum4, true);
            var ptr5 = utf8Encode(v3_1, realloc0, memory0);
            var len5 = utf8EncodedLen;
            dataView(memory0).setUint32(arg1 + 16, len5, true);
            dataView(memory0).setUint32(arg1 + 12, ptr5, true);
          }
          break;
        }
        case "err": {
          const e = variant8.val;
          dataView(memory0).setInt8(arg1 + 0, 1, true);
          var val7 = e;
          let enum7;
          switch (val7) {
            case "access": {
              enum7 = 0;
              break;
            }
            case "would-block": {
              enum7 = 1;
              break;
            }
            case "already": {
              enum7 = 2;
              break;
            }
            case "bad-descriptor": {
              enum7 = 3;
              break;
            }
            case "busy": {
              enum7 = 4;
              break;
            }
            case "deadlock": {
              enum7 = 5;
              break;
            }
            case "quota": {
              enum7 = 6;
              break;
            }
            case "exist": {
              enum7 = 7;
              break;
            }
            case "file-too-large": {
              enum7 = 8;
              break;
            }
            case "illegal-byte-sequence": {
              enum7 = 9;
              break;
            }
            case "in-progress": {
              enum7 = 10;
              break;
            }
            case "interrupted": {
              enum7 = 11;
              break;
            }
            case "invalid": {
              enum7 = 12;
              break;
            }
            case "io": {
              enum7 = 13;
              break;
            }
            case "is-directory": {
              enum7 = 14;
              break;
            }
            case "loop": {
              enum7 = 15;
              break;
            }
            case "too-many-links": {
              enum7 = 16;
              break;
            }
            case "message-size": {
              enum7 = 17;
              break;
            }
            case "name-too-long": {
              enum7 = 18;
              break;
            }
            case "no-device": {
              enum7 = 19;
              break;
            }
            case "no-entry": {
              enum7 = 20;
              break;
            }
            case "no-lock": {
              enum7 = 21;
              break;
            }
            case "insufficient-memory": {
              enum7 = 22;
              break;
            }
            case "insufficient-space": {
              enum7 = 23;
              break;
            }
            case "not-directory": {
              enum7 = 24;
              break;
            }
            case "not-empty": {
              enum7 = 25;
              break;
            }
            case "not-recoverable": {
              enum7 = 26;
              break;
            }
            case "unsupported": {
              enum7 = 27;
              break;
            }
            case "no-tty": {
              enum7 = 28;
              break;
            }
            case "no-such-device": {
              enum7 = 29;
              break;
            }
            case "overflow": {
              enum7 = 30;
              break;
            }
            case "not-permitted": {
              enum7 = 31;
              break;
            }
            case "pipe": {
              enum7 = 32;
              break;
            }
            case "read-only": {
              enum7 = 33;
              break;
            }
            case "invalid-seek": {
              enum7 = 34;
              break;
            }
            case "text-file-busy": {
              enum7 = 35;
              break;
            }
            case "cross-device": {
              enum7 = 36;
              break;
            }
            default: {
              if (e instanceof Error) {
                console.error(e);
              }
              throw new TypeError(`"${val7}" is not one of the cases of error-code`);
            }
          }
          dataView(memory0).setInt8(arg1 + 4, enum7, true);
          break;
        }
        default: {
          throw new TypeError("invalid variant specified for result");
        }
      }
      _debugLog('[iface="wasi:filesystem/types@0.2.3", function="[method]directory-entry-stream.read-directory-entry"][Instruction::Return]', {
        funcName: "[method]directory-entry-stream.read-directory-entry",
        paramCount: 0,
        postReturn: false
      });
    }
    function trampoline43(arg0, arg1, arg2) {
      var handle1 = arg0;
      var rep2 = handleTable2[(handle1 << 1) + 1] & ~T_FLAG;
      var rsc0 = captureTable2.get(rep2);
      if (!rsc0) {
        rsc0 = Object.create(InputStream2.prototype);
        Object.defineProperty(rsc0, symbolRscHandle, { writable: true, value: handle1 });
        Object.defineProperty(rsc0, symbolRscRep, { writable: true, value: rep2 });
      }
      curResourceBorrows.push(rsc0);
      _debugLog('[iface="wasi:io/streams@0.2.3", function="[method]input-stream.read"] [Instruction::CallInterface] (async? sync, @ enter)');
      const _interface_call_currentTaskID = startCurrentTask(0, false, "[method]input-stream.read");
      let ret;
      try {
        ret = { tag: "ok", val: rsc0.read(BigInt.asUintN(64, arg1)) };
      } catch (e) {
        ret = { tag: "err", val: getErrorPayload(e) };
      }
      _debugLog('[iface="wasi:io/streams@0.2.3", function="[method]input-stream.read"] [Instruction::CallInterface] (sync, @ post-call)');
      for (const rsc of curResourceBorrows) {
        rsc[symbolRscHandle] = void 0;
      }
      curResourceBorrows = [];
      endCurrentTask(0);
      var variant6 = ret;
      switch (variant6.tag) {
        case "ok": {
          const e = variant6.val;
          dataView(memory0).setInt8(arg2 + 0, 0, true);
          var val3 = e;
          var len3 = val3.byteLength;
          var ptr3 = realloc0(0, 0, 1, len3 * 1);
          var src3 = new Uint8Array(val3.buffer || val3, val3.byteOffset, len3 * 1);
          new Uint8Array(memory0.buffer, ptr3, len3 * 1).set(src3);
          dataView(memory0).setUint32(arg2 + 8, len3, true);
          dataView(memory0).setUint32(arg2 + 4, ptr3, true);
          break;
        }
        case "err": {
          const e = variant6.val;
          dataView(memory0).setInt8(arg2 + 0, 1, true);
          var variant5 = e;
          switch (variant5.tag) {
            case "last-operation-failed": {
              const e2 = variant5.val;
              dataView(memory0).setInt8(arg2 + 4, 0, true);
              if (!(e2 instanceof Error$1)) {
                throw new TypeError('Resource error: Not a valid "Error" resource.');
              }
              var handle4 = e2[symbolRscHandle];
              if (!handle4) {
                const rep = e2[symbolRscRep] || ++captureCnt0;
                captureTable0.set(rep, e2);
                handle4 = rscTableCreateOwn(handleTable0, rep);
              }
              dataView(memory0).setInt32(arg2 + 8, handle4, true);
              break;
            }
            case "closed": {
              dataView(memory0).setInt8(arg2 + 4, 1, true);
              break;
            }
            default: {
              throw new TypeError(`invalid variant tag value \`${JSON.stringify(variant5.tag)}\` (received \`${variant5}\`) specified for \`StreamError\``);
            }
          }
          break;
        }
        default: {
          throw new TypeError("invalid variant specified for result");
        }
      }
      _debugLog('[iface="wasi:io/streams@0.2.3", function="[method]input-stream.read"][Instruction::Return]', {
        funcName: "[method]input-stream.read",
        paramCount: 0,
        postReturn: false
      });
    }
    function trampoline44(arg0, arg1, arg2) {
      var handle1 = arg0;
      var rep2 = handleTable2[(handle1 << 1) + 1] & ~T_FLAG;
      var rsc0 = captureTable2.get(rep2);
      if (!rsc0) {
        rsc0 = Object.create(InputStream2.prototype);
        Object.defineProperty(rsc0, symbolRscHandle, { writable: true, value: handle1 });
        Object.defineProperty(rsc0, symbolRscRep, { writable: true, value: rep2 });
      }
      curResourceBorrows.push(rsc0);
      _debugLog('[iface="wasi:io/streams@0.2.3", function="[method]input-stream.blocking-read"] [Instruction::CallInterface] (async? sync, @ enter)');
      const _interface_call_currentTaskID = startCurrentTask(0, false, "[method]input-stream.blocking-read");
      let ret;
      try {
        ret = { tag: "ok", val: rsc0.blockingRead(BigInt.asUintN(64, arg1)) };
      } catch (e) {
        ret = { tag: "err", val: getErrorPayload(e) };
      }
      _debugLog('[iface="wasi:io/streams@0.2.3", function="[method]input-stream.blocking-read"] [Instruction::CallInterface] (sync, @ post-call)');
      for (const rsc of curResourceBorrows) {
        rsc[symbolRscHandle] = void 0;
      }
      curResourceBorrows = [];
      endCurrentTask(0);
      var variant6 = ret;
      switch (variant6.tag) {
        case "ok": {
          const e = variant6.val;
          dataView(memory0).setInt8(arg2 + 0, 0, true);
          var val3 = e;
          var len3 = val3.byteLength;
          var ptr3 = realloc0(0, 0, 1, len3 * 1);
          var src3 = new Uint8Array(val3.buffer || val3, val3.byteOffset, len3 * 1);
          new Uint8Array(memory0.buffer, ptr3, len3 * 1).set(src3);
          dataView(memory0).setUint32(arg2 + 8, len3, true);
          dataView(memory0).setUint32(arg2 + 4, ptr3, true);
          break;
        }
        case "err": {
          const e = variant6.val;
          dataView(memory0).setInt8(arg2 + 0, 1, true);
          var variant5 = e;
          switch (variant5.tag) {
            case "last-operation-failed": {
              const e2 = variant5.val;
              dataView(memory0).setInt8(arg2 + 4, 0, true);
              if (!(e2 instanceof Error$1)) {
                throw new TypeError('Resource error: Not a valid "Error" resource.');
              }
              var handle4 = e2[symbolRscHandle];
              if (!handle4) {
                const rep = e2[symbolRscRep] || ++captureCnt0;
                captureTable0.set(rep, e2);
                handle4 = rscTableCreateOwn(handleTable0, rep);
              }
              dataView(memory0).setInt32(arg2 + 8, handle4, true);
              break;
            }
            case "closed": {
              dataView(memory0).setInt8(arg2 + 4, 1, true);
              break;
            }
            default: {
              throw new TypeError(`invalid variant tag value \`${JSON.stringify(variant5.tag)}\` (received \`${variant5}\`) specified for \`StreamError\``);
            }
          }
          break;
        }
        default: {
          throw new TypeError("invalid variant specified for result");
        }
      }
      _debugLog('[iface="wasi:io/streams@0.2.3", function="[method]input-stream.blocking-read"][Instruction::Return]', {
        funcName: "[method]input-stream.blocking-read",
        paramCount: 0,
        postReturn: false
      });
    }
    function trampoline45(arg0, arg1) {
      var handle1 = arg0;
      var rep2 = handleTable3[(handle1 << 1) + 1] & ~T_FLAG;
      var rsc0 = captureTable3.get(rep2);
      if (!rsc0) {
        rsc0 = Object.create(OutputStream2.prototype);
        Object.defineProperty(rsc0, symbolRscHandle, { writable: true, value: handle1 });
        Object.defineProperty(rsc0, symbolRscRep, { writable: true, value: rep2 });
      }
      curResourceBorrows.push(rsc0);
      _debugLog('[iface="wasi:io/streams@0.2.3", function="[method]output-stream.check-write"] [Instruction::CallInterface] (async? sync, @ enter)');
      const _interface_call_currentTaskID = startCurrentTask(0, false, "[method]output-stream.check-write");
      let ret;
      try {
        ret = { tag: "ok", val: rsc0.checkWrite() };
      } catch (e) {
        ret = { tag: "err", val: getErrorPayload(e) };
      }
      _debugLog('[iface="wasi:io/streams@0.2.3", function="[method]output-stream.check-write"] [Instruction::CallInterface] (sync, @ post-call)');
      for (const rsc of curResourceBorrows) {
        rsc[symbolRscHandle] = void 0;
      }
      curResourceBorrows = [];
      endCurrentTask(0);
      var variant5 = ret;
      switch (variant5.tag) {
        case "ok": {
          const e = variant5.val;
          dataView(memory0).setInt8(arg1 + 0, 0, true);
          dataView(memory0).setBigInt64(arg1 + 8, toUint64(e), true);
          break;
        }
        case "err": {
          const e = variant5.val;
          dataView(memory0).setInt8(arg1 + 0, 1, true);
          var variant4 = e;
          switch (variant4.tag) {
            case "last-operation-failed": {
              const e2 = variant4.val;
              dataView(memory0).setInt8(arg1 + 8, 0, true);
              if (!(e2 instanceof Error$1)) {
                throw new TypeError('Resource error: Not a valid "Error" resource.');
              }
              var handle3 = e2[symbolRscHandle];
              if (!handle3) {
                const rep = e2[symbolRscRep] || ++captureCnt0;
                captureTable0.set(rep, e2);
                handle3 = rscTableCreateOwn(handleTable0, rep);
              }
              dataView(memory0).setInt32(arg1 + 12, handle3, true);
              break;
            }
            case "closed": {
              dataView(memory0).setInt8(arg1 + 8, 1, true);
              break;
            }
            default: {
              throw new TypeError(`invalid variant tag value \`${JSON.stringify(variant4.tag)}\` (received \`${variant4}\`) specified for \`StreamError\``);
            }
          }
          break;
        }
        default: {
          throw new TypeError("invalid variant specified for result");
        }
      }
      _debugLog('[iface="wasi:io/streams@0.2.3", function="[method]output-stream.check-write"][Instruction::Return]', {
        funcName: "[method]output-stream.check-write",
        paramCount: 0,
        postReturn: false
      });
    }
    function trampoline46(arg0, arg1, arg2, arg3) {
      var handle1 = arg0;
      var rep2 = handleTable3[(handle1 << 1) + 1] & ~T_FLAG;
      var rsc0 = captureTable3.get(rep2);
      if (!rsc0) {
        rsc0 = Object.create(OutputStream2.prototype);
        Object.defineProperty(rsc0, symbolRscHandle, { writable: true, value: handle1 });
        Object.defineProperty(rsc0, symbolRscRep, { writable: true, value: rep2 });
      }
      curResourceBorrows.push(rsc0);
      var ptr3 = arg1;
      var len3 = arg2;
      var result3 = new Uint8Array(memory0.buffer.slice(ptr3, ptr3 + len3 * 1));
      _debugLog('[iface="wasi:io/streams@0.2.3", function="[method]output-stream.write"] [Instruction::CallInterface] (async? sync, @ enter)');
      const _interface_call_currentTaskID = startCurrentTask(0, false, "[method]output-stream.write");
      let ret;
      try {
        ret = { tag: "ok", val: rsc0.write(result3) };
      } catch (e) {
        ret = { tag: "err", val: getErrorPayload(e) };
      }
      _debugLog('[iface="wasi:io/streams@0.2.3", function="[method]output-stream.write"] [Instruction::CallInterface] (sync, @ post-call)');
      for (const rsc of curResourceBorrows) {
        rsc[symbolRscHandle] = void 0;
      }
      curResourceBorrows = [];
      endCurrentTask(0);
      var variant6 = ret;
      switch (variant6.tag) {
        case "ok": {
          const e = variant6.val;
          dataView(memory0).setInt8(arg3 + 0, 0, true);
          break;
        }
        case "err": {
          const e = variant6.val;
          dataView(memory0).setInt8(arg3 + 0, 1, true);
          var variant5 = e;
          switch (variant5.tag) {
            case "last-operation-failed": {
              const e2 = variant5.val;
              dataView(memory0).setInt8(arg3 + 4, 0, true);
              if (!(e2 instanceof Error$1)) {
                throw new TypeError('Resource error: Not a valid "Error" resource.');
              }
              var handle4 = e2[symbolRscHandle];
              if (!handle4) {
                const rep = e2[symbolRscRep] || ++captureCnt0;
                captureTable0.set(rep, e2);
                handle4 = rscTableCreateOwn(handleTable0, rep);
              }
              dataView(memory0).setInt32(arg3 + 8, handle4, true);
              break;
            }
            case "closed": {
              dataView(memory0).setInt8(arg3 + 4, 1, true);
              break;
            }
            default: {
              throw new TypeError(`invalid variant tag value \`${JSON.stringify(variant5.tag)}\` (received \`${variant5}\`) specified for \`StreamError\``);
            }
          }
          break;
        }
        default: {
          throw new TypeError("invalid variant specified for result");
        }
      }
      _debugLog('[iface="wasi:io/streams@0.2.3", function="[method]output-stream.write"][Instruction::Return]', {
        funcName: "[method]output-stream.write",
        paramCount: 0,
        postReturn: false
      });
    }
    function trampoline47(arg0, arg1, arg2, arg3) {
      var handle1 = arg0;
      var rep2 = handleTable3[(handle1 << 1) + 1] & ~T_FLAG;
      var rsc0 = captureTable3.get(rep2);
      if (!rsc0) {
        rsc0 = Object.create(OutputStream2.prototype);
        Object.defineProperty(rsc0, symbolRscHandle, { writable: true, value: handle1 });
        Object.defineProperty(rsc0, symbolRscRep, { writable: true, value: rep2 });
      }
      curResourceBorrows.push(rsc0);
      var ptr3 = arg1;
      var len3 = arg2;
      var result3 = new Uint8Array(memory0.buffer.slice(ptr3, ptr3 + len3 * 1));
      _debugLog('[iface="wasi:io/streams@0.2.3", function="[method]output-stream.blocking-write-and-flush"] [Instruction::CallInterface] (async? sync, @ enter)');
      const _interface_call_currentTaskID = startCurrentTask(0, false, "[method]output-stream.blocking-write-and-flush");
      let ret;
      try {
        ret = { tag: "ok", val: rsc0.blockingWriteAndFlush(result3) };
      } catch (e) {
        ret = { tag: "err", val: getErrorPayload(e) };
      }
      _debugLog('[iface="wasi:io/streams@0.2.3", function="[method]output-stream.blocking-write-and-flush"] [Instruction::CallInterface] (sync, @ post-call)');
      for (const rsc of curResourceBorrows) {
        rsc[symbolRscHandle] = void 0;
      }
      curResourceBorrows = [];
      endCurrentTask(0);
      var variant6 = ret;
      switch (variant6.tag) {
        case "ok": {
          const e = variant6.val;
          dataView(memory0).setInt8(arg3 + 0, 0, true);
          break;
        }
        case "err": {
          const e = variant6.val;
          dataView(memory0).setInt8(arg3 + 0, 1, true);
          var variant5 = e;
          switch (variant5.tag) {
            case "last-operation-failed": {
              const e2 = variant5.val;
              dataView(memory0).setInt8(arg3 + 4, 0, true);
              if (!(e2 instanceof Error$1)) {
                throw new TypeError('Resource error: Not a valid "Error" resource.');
              }
              var handle4 = e2[symbolRscHandle];
              if (!handle4) {
                const rep = e2[symbolRscRep] || ++captureCnt0;
                captureTable0.set(rep, e2);
                handle4 = rscTableCreateOwn(handleTable0, rep);
              }
              dataView(memory0).setInt32(arg3 + 8, handle4, true);
              break;
            }
            case "closed": {
              dataView(memory0).setInt8(arg3 + 4, 1, true);
              break;
            }
            default: {
              throw new TypeError(`invalid variant tag value \`${JSON.stringify(variant5.tag)}\` (received \`${variant5}\`) specified for \`StreamError\``);
            }
          }
          break;
        }
        default: {
          throw new TypeError("invalid variant specified for result");
        }
      }
      _debugLog('[iface="wasi:io/streams@0.2.3", function="[method]output-stream.blocking-write-and-flush"][Instruction::Return]', {
        funcName: "[method]output-stream.blocking-write-and-flush",
        paramCount: 0,
        postReturn: false
      });
    }
    function trampoline48(arg0, arg1) {
      var handle1 = arg0;
      var rep2 = handleTable3[(handle1 << 1) + 1] & ~T_FLAG;
      var rsc0 = captureTable3.get(rep2);
      if (!rsc0) {
        rsc0 = Object.create(OutputStream2.prototype);
        Object.defineProperty(rsc0, symbolRscHandle, { writable: true, value: handle1 });
        Object.defineProperty(rsc0, symbolRscRep, { writable: true, value: rep2 });
      }
      curResourceBorrows.push(rsc0);
      _debugLog('[iface="wasi:io/streams@0.2.3", function="[method]output-stream.blocking-flush"] [Instruction::CallInterface] (async? sync, @ enter)');
      const _interface_call_currentTaskID = startCurrentTask(0, false, "[method]output-stream.blocking-flush");
      let ret;
      try {
        ret = { tag: "ok", val: rsc0.blockingFlush() };
      } catch (e) {
        ret = { tag: "err", val: getErrorPayload(e) };
      }
      _debugLog('[iface="wasi:io/streams@0.2.3", function="[method]output-stream.blocking-flush"] [Instruction::CallInterface] (sync, @ post-call)');
      for (const rsc of curResourceBorrows) {
        rsc[symbolRscHandle] = void 0;
      }
      curResourceBorrows = [];
      endCurrentTask(0);
      var variant5 = ret;
      switch (variant5.tag) {
        case "ok": {
          const e = variant5.val;
          dataView(memory0).setInt8(arg1 + 0, 0, true);
          break;
        }
        case "err": {
          const e = variant5.val;
          dataView(memory0).setInt8(arg1 + 0, 1, true);
          var variant4 = e;
          switch (variant4.tag) {
            case "last-operation-failed": {
              const e2 = variant4.val;
              dataView(memory0).setInt8(arg1 + 4, 0, true);
              if (!(e2 instanceof Error$1)) {
                throw new TypeError('Resource error: Not a valid "Error" resource.');
              }
              var handle3 = e2[symbolRscHandle];
              if (!handle3) {
                const rep = e2[symbolRscRep] || ++captureCnt0;
                captureTable0.set(rep, e2);
                handle3 = rscTableCreateOwn(handleTable0, rep);
              }
              dataView(memory0).setInt32(arg1 + 8, handle3, true);
              break;
            }
            case "closed": {
              dataView(memory0).setInt8(arg1 + 4, 1, true);
              break;
            }
            default: {
              throw new TypeError(`invalid variant tag value \`${JSON.stringify(variant4.tag)}\` (received \`${variant4}\`) specified for \`StreamError\``);
            }
          }
          break;
        }
        default: {
          throw new TypeError("invalid variant specified for result");
        }
      }
      _debugLog('[iface="wasi:io/streams@0.2.3", function="[method]output-stream.blocking-flush"][Instruction::Return]', {
        funcName: "[method]output-stream.blocking-flush",
        paramCount: 0,
        postReturn: false
      });
    }
    function trampoline49(arg0, arg1, arg2) {
      var len3 = arg1;
      var base3 = arg0;
      var result3 = [];
      for (let i = 0; i < len3; i++) {
        const base = base3 + i * 4;
        var handle1 = dataView(memory0).getInt32(base + 0, true);
        var rep2 = handleTable1[(handle1 << 1) + 1] & ~T_FLAG;
        var rsc0 = captureTable1.get(rep2);
        if (!rsc0) {
          rsc0 = Object.create(Pollable.prototype);
          Object.defineProperty(rsc0, symbolRscHandle, { writable: true, value: handle1 });
          Object.defineProperty(rsc0, symbolRscRep, { writable: true, value: rep2 });
        }
        curResourceBorrows.push(rsc0);
        result3.push(rsc0);
      }
      _debugLog('[iface="wasi:io/poll@0.2.3", function="poll"] [Instruction::CallInterface] (async? sync, @ enter)');
      const _interface_call_currentTaskID = startCurrentTask(0, false, "poll");
      const ret = poll(result3);
      _debugLog('[iface="wasi:io/poll@0.2.3", function="poll"] [Instruction::CallInterface] (sync, @ post-call)');
      for (const rsc of curResourceBorrows) {
        rsc[symbolRscHandle] = void 0;
      }
      curResourceBorrows = [];
      endCurrentTask(0);
      var val4 = ret;
      var len4 = val4.length;
      var ptr4 = realloc0(0, 0, 4, len4 * 4);
      var src4 = new Uint8Array(val4.buffer, val4.byteOffset, len4 * 4);
      new Uint8Array(memory0.buffer, ptr4, len4 * 4).set(src4);
      dataView(memory0).setUint32(arg2 + 4, len4, true);
      dataView(memory0).setUint32(arg2 + 0, ptr4, true);
      _debugLog('[iface="wasi:io/poll@0.2.3", function="poll"][Instruction::Return]', {
        funcName: "poll",
        paramCount: 0,
        postReturn: false
      });
    }
    function trampoline50(arg0, arg1) {
      _debugLog('[iface="wasi:random/random@0.2.3", function="get-random-bytes"] [Instruction::CallInterface] (async? sync, @ enter)');
      const _interface_call_currentTaskID = startCurrentTask(0, false, "get-random-bytes");
      const ret = getRandomBytes(BigInt.asUintN(64, arg0));
      _debugLog('[iface="wasi:random/random@0.2.3", function="get-random-bytes"] [Instruction::CallInterface] (sync, @ post-call)');
      endCurrentTask(0);
      var val0 = ret;
      var len0 = val0.byteLength;
      var ptr0 = realloc0(0, 0, 1, len0 * 1);
      var src0 = new Uint8Array(val0.buffer || val0, val0.byteOffset, len0 * 1);
      new Uint8Array(memory0.buffer, ptr0, len0 * 1).set(src0);
      dataView(memory0).setUint32(arg1 + 4, len0, true);
      dataView(memory0).setUint32(arg1 + 0, ptr0, true);
      _debugLog('[iface="wasi:random/random@0.2.3", function="get-random-bytes"][Instruction::Return]', {
        funcName: "get-random-bytes",
        paramCount: 0,
        postReturn: false
      });
    }
    function trampoline51(arg0) {
      _debugLog('[iface="wasi:filesystem/preopens@0.2.3", function="get-directories"] [Instruction::CallInterface] (async? sync, @ enter)');
      const _interface_call_currentTaskID = startCurrentTask(0, false, "get-directories");
      const ret = getDirectories();
      _debugLog('[iface="wasi:filesystem/preopens@0.2.3", function="get-directories"] [Instruction::CallInterface] (sync, @ post-call)');
      endCurrentTask(0);
      var vec3 = ret;
      var len3 = vec3.length;
      var result3 = realloc0(0, 0, 4, len3 * 12);
      for (let i = 0; i < vec3.length; i++) {
        const e = vec3[i];
        const base = result3 + i * 12;
        var [tuple0_0, tuple0_1] = e;
        if (!(tuple0_0 instanceof Descriptor2)) {
          throw new TypeError('Resource error: Not a valid "Descriptor" resource.');
        }
        var handle1 = tuple0_0[symbolRscHandle];
        if (!handle1) {
          const rep = tuple0_0[symbolRscRep] || ++captureCnt7;
          captureTable7.set(rep, tuple0_0);
          handle1 = rscTableCreateOwn(handleTable7, rep);
        }
        dataView(memory0).setInt32(base + 0, handle1, true);
        var ptr2 = utf8Encode(tuple0_1, realloc0, memory0);
        var len2 = utf8EncodedLen;
        dataView(memory0).setUint32(base + 8, len2, true);
        dataView(memory0).setUint32(base + 4, ptr2, true);
      }
      dataView(memory0).setUint32(arg0 + 4, len3, true);
      dataView(memory0).setUint32(arg0 + 0, result3, true);
      _debugLog('[iface="wasi:filesystem/preopens@0.2.3", function="get-directories"][Instruction::Return]', {
        funcName: "get-directories",
        paramCount: 0,
        postReturn: false
      });
    }
    const handleTable4 = [T_FLAG, 0];
    const captureTable4 = /* @__PURE__ */ new Map();
    let captureCnt4 = 0;
    handleTables[4] = handleTable4;
    function trampoline52(arg0) {
      _debugLog('[iface="wasi:cli/terminal-stdin@0.2.3", function="get-terminal-stdin"] [Instruction::CallInterface] (async? sync, @ enter)');
      const _interface_call_currentTaskID = startCurrentTask(0, false, "get-terminal-stdin");
      const ret = getTerminalStdin();
      _debugLog('[iface="wasi:cli/terminal-stdin@0.2.3", function="get-terminal-stdin"] [Instruction::CallInterface] (sync, @ post-call)');
      endCurrentTask(0);
      var variant1 = ret;
      if (variant1 === null || variant1 === void 0) {
        dataView(memory0).setInt8(arg0 + 0, 0, true);
      } else {
        const e = variant1;
        dataView(memory0).setInt8(arg0 + 0, 1, true);
        if (!(e instanceof TerminalInput2)) {
          throw new TypeError('Resource error: Not a valid "TerminalInput" resource.');
        }
        var handle0 = e[symbolRscHandle];
        if (!handle0) {
          const rep = e[symbolRscRep] || ++captureCnt4;
          captureTable4.set(rep, e);
          handle0 = rscTableCreateOwn(handleTable4, rep);
        }
        dataView(memory0).setInt32(arg0 + 4, handle0, true);
      }
      _debugLog('[iface="wasi:cli/terminal-stdin@0.2.3", function="get-terminal-stdin"][Instruction::Return]', {
        funcName: "get-terminal-stdin",
        paramCount: 0,
        postReturn: false
      });
    }
    const handleTable5 = [T_FLAG, 0];
    const captureTable5 = /* @__PURE__ */ new Map();
    let captureCnt5 = 0;
    handleTables[5] = handleTable5;
    function trampoline53(arg0) {
      _debugLog('[iface="wasi:cli/terminal-stdout@0.2.3", function="get-terminal-stdout"] [Instruction::CallInterface] (async? sync, @ enter)');
      const _interface_call_currentTaskID = startCurrentTask(0, false, "get-terminal-stdout");
      const ret = getTerminalStdout();
      _debugLog('[iface="wasi:cli/terminal-stdout@0.2.3", function="get-terminal-stdout"] [Instruction::CallInterface] (sync, @ post-call)');
      endCurrentTask(0);
      var variant1 = ret;
      if (variant1 === null || variant1 === void 0) {
        dataView(memory0).setInt8(arg0 + 0, 0, true);
      } else {
        const e = variant1;
        dataView(memory0).setInt8(arg0 + 0, 1, true);
        if (!(e instanceof TerminalOutput2)) {
          throw new TypeError('Resource error: Not a valid "TerminalOutput" resource.');
        }
        var handle0 = e[symbolRscHandle];
        if (!handle0) {
          const rep = e[symbolRscRep] || ++captureCnt5;
          captureTable5.set(rep, e);
          handle0 = rscTableCreateOwn(handleTable5, rep);
        }
        dataView(memory0).setInt32(arg0 + 4, handle0, true);
      }
      _debugLog('[iface="wasi:cli/terminal-stdout@0.2.3", function="get-terminal-stdout"][Instruction::Return]', {
        funcName: "get-terminal-stdout",
        paramCount: 0,
        postReturn: false
      });
    }
    function trampoline54(arg0) {
      _debugLog('[iface="wasi:cli/terminal-stderr@0.2.3", function="get-terminal-stderr"] [Instruction::CallInterface] (async? sync, @ enter)');
      const _interface_call_currentTaskID = startCurrentTask(0, false, "get-terminal-stderr");
      const ret = getTerminalStderr();
      _debugLog('[iface="wasi:cli/terminal-stderr@0.2.3", function="get-terminal-stderr"] [Instruction::CallInterface] (sync, @ post-call)');
      endCurrentTask(0);
      var variant1 = ret;
      if (variant1 === null || variant1 === void 0) {
        dataView(memory0).setInt8(arg0 + 0, 0, true);
      } else {
        const e = variant1;
        dataView(memory0).setInt8(arg0 + 0, 1, true);
        if (!(e instanceof TerminalOutput2)) {
          throw new TypeError('Resource error: Not a valid "TerminalOutput" resource.');
        }
        var handle0 = e[symbolRscHandle];
        if (!handle0) {
          const rep = e[symbolRscRep] || ++captureCnt5;
          captureTable5.set(rep, e);
          handle0 = rscTableCreateOwn(handleTable5, rep);
        }
        dataView(memory0).setInt32(arg0 + 4, handle0, true);
      }
      _debugLog('[iface="wasi:cli/terminal-stderr@0.2.3", function="get-terminal-stderr"][Instruction::Return]', {
        funcName: "get-terminal-stderr",
        paramCount: 0,
        postReturn: false
      });
    }
    let exports3;
    function trampoline1(handle) {
      const handleEntry = rscTableRemove(handleTable6, handle);
      if (handleEntry.own) {
        const rsc = captureTable6.get(handleEntry.rep);
        if (rsc) {
          if (rsc[symbolDispose]) rsc[symbolDispose]();
          captureTable6.delete(handleEntry.rep);
        } else if (DirectoryEntryStream2[symbolCabiDispose]) {
          DirectoryEntryStream2[symbolCabiDispose](handleEntry.rep);
        }
      }
    }
    function trampoline2(handle) {
      const handleEntry = rscTableRemove(handleTable3, handle);
      if (handleEntry.own) {
        const rsc = captureTable3.get(handleEntry.rep);
        if (rsc) {
          if (rsc[symbolDispose]) rsc[symbolDispose]();
          captureTable3.delete(handleEntry.rep);
        } else if (OutputStream2[symbolCabiDispose]) {
          OutputStream2[symbolCabiDispose](handleEntry.rep);
        }
      }
    }
    function trampoline3(handle) {
      const handleEntry = rscTableRemove(handleTable0, handle);
      if (handleEntry.own) {
        const rsc = captureTable0.get(handleEntry.rep);
        if (rsc) {
          if (rsc[symbolDispose]) rsc[symbolDispose]();
          captureTable0.delete(handleEntry.rep);
        } else if (Error$1[symbolCabiDispose]) {
          Error$1[symbolCabiDispose](handleEntry.rep);
        }
      }
    }
    function trampoline4(handle) {
      const handleEntry = rscTableRemove(handleTable2, handle);
      if (handleEntry.own) {
        const rsc = captureTable2.get(handleEntry.rep);
        if (rsc) {
          if (rsc[symbolDispose]) rsc[symbolDispose]();
          captureTable2.delete(handleEntry.rep);
        } else if (InputStream2[symbolCabiDispose]) {
          InputStream2[symbolCabiDispose](handleEntry.rep);
        }
      }
    }
    function trampoline5(handle) {
      const handleEntry = rscTableRemove(handleTable7, handle);
      if (handleEntry.own) {
        const rsc = captureTable7.get(handleEntry.rep);
        if (rsc) {
          if (rsc[symbolDispose]) rsc[symbolDispose]();
          captureTable7.delete(handleEntry.rep);
        } else if (Descriptor2[symbolCabiDispose]) {
          Descriptor2[symbolCabiDispose](handleEntry.rep);
        }
      }
    }
    function trampoline10(handle) {
      const handleEntry = rscTableRemove(handleTable1, handle);
      if (handleEntry.own) {
        const rsc = captureTable1.get(handleEntry.rep);
        if (rsc) {
          if (rsc[symbolDispose]) rsc[symbolDispose]();
          captureTable1.delete(handleEntry.rep);
        } else if (Pollable[symbolCabiDispose]) {
          Pollable[symbolCabiDispose](handleEntry.rep);
        }
      }
    }
    function trampoline12(handle) {
      const handleEntry = rscTableRemove(handleTable4, handle);
      if (handleEntry.own) {
        const rsc = captureTable4.get(handleEntry.rep);
        if (rsc) {
          if (rsc[symbolDispose]) rsc[symbolDispose]();
          captureTable4.delete(handleEntry.rep);
        } else if (TerminalInput2[symbolCabiDispose]) {
          TerminalInput2[symbolCabiDispose](handleEntry.rep);
        }
      }
    }
    function trampoline13(handle) {
      const handleEntry = rscTableRemove(handleTable5, handle);
      if (handleEntry.own) {
        const rsc = captureTable5.get(handleEntry.rep);
        if (rsc) {
          if (rsc[symbolDispose]) rsc[symbolDispose]();
          captureTable5.delete(handleEntry.rep);
        } else if (TerminalOutput2[symbolCabiDispose]) {
          TerminalOutput2[symbolCabiDispose](handleEntry.rep);
        }
      }
    }
    Promise.all([module0, module1, module2, module3]).catch(() => {
    });
    ({ exports: exports0 } = yield instantiateCore(yield module2));
    ({ exports: exports1 } = yield instantiateCore(yield module0, {
      wasi_snapshot_preview1: {
        args_get: exports0["0"],
        args_sizes_get: exports0["1"],
        clock_time_get: exports0["4"],
        environ_get: exports0["2"],
        environ_sizes_get: exports0["3"],
        fd_close: exports0["5"],
        fd_fdstat_get: exports0["6"],
        fd_filestat_get: exports0["7"],
        fd_filestat_set_size: exports0["8"],
        fd_filestat_set_times: exports0["9"],
        fd_pread: exports0["10"],
        fd_prestat_dir_name: exports0["12"],
        fd_prestat_get: exports0["11"],
        fd_read: exports0["13"],
        fd_readdir: exports0["14"],
        fd_seek: exports0["15"],
        fd_write: exports0["16"],
        path_create_directory: exports0["17"],
        path_filestat_get: exports0["18"],
        path_link: exports0["19"],
        path_open: exports0["20"],
        path_readlink: exports0["21"],
        path_remove_directory: exports0["22"],
        path_rename: exports0["23"],
        path_symlink: exports0["24"],
        path_unlink_file: exports0["25"],
        poll_oneoff: exports0["26"],
        proc_exit: exports0["27"],
        random_get: exports0["28"]
      }
    }));
    ({ exports: exports2 } = yield instantiateCore(yield module1, {
      __main_module__: {
        _start: exports1._start
      },
      env: {
        memory: exports1.memory
      },
      "wasi:cli/environment@0.2.3": {
        "get-arguments": exports0["29"],
        "get-environment": exports0["30"]
      },
      "wasi:cli/exit@0.2.3": {
        exit: trampoline16
      },
      "wasi:cli/stderr@0.2.3": {
        "get-stderr": trampoline11
      },
      "wasi:cli/stdin@0.2.3": {
        "get-stdin": trampoline14
      },
      "wasi:cli/stdout@0.2.3": {
        "get-stdout": trampoline15
      },
      "wasi:cli/terminal-input@0.2.3": {
        "[resource-drop]terminal-input": trampoline12
      },
      "wasi:cli/terminal-output@0.2.3": {
        "[resource-drop]terminal-output": trampoline13
      },
      "wasi:cli/terminal-stderr@0.2.3": {
        "get-terminal-stderr": exports0["66"]
      },
      "wasi:cli/terminal-stdin@0.2.3": {
        "get-terminal-stdin": exports0["64"]
      },
      "wasi:cli/terminal-stdout@0.2.3": {
        "get-terminal-stdout": exports0["65"]
      },
      "wasi:clocks/monotonic-clock@0.2.3": {
        now: trampoline0,
        "subscribe-duration": trampoline6,
        "subscribe-instant": trampoline7
      },
      "wasi:clocks/wall-clock@0.2.3": {
        now: exports0["31"]
      },
      "wasi:filesystem/preopens@0.2.3": {
        "get-directories": exports0["63"]
      },
      "wasi:filesystem/types@0.2.3": {
        "[method]descriptor.append-via-stream": exports0["47"],
        "[method]descriptor.create-directory-at": exports0["38"],
        "[method]descriptor.get-flags": exports0["32"],
        "[method]descriptor.get-type": exports0["33"],
        "[method]descriptor.link-at": exports0["39"],
        "[method]descriptor.metadata-hash": exports0["34"],
        "[method]descriptor.metadata-hash-at": exports0["37"],
        "[method]descriptor.open-at": exports0["53"],
        "[method]descriptor.read": exports0["49"],
        "[method]descriptor.read-directory": exports0["50"],
        "[method]descriptor.read-via-stream": exports0["45"],
        "[method]descriptor.readlink-at": exports0["40"],
        "[method]descriptor.remove-directory-at": exports0["41"],
        "[method]descriptor.rename-at": exports0["42"],
        "[method]descriptor.set-size": exports0["35"],
        "[method]descriptor.set-times": exports0["48"],
        "[method]descriptor.stat": exports0["51"],
        "[method]descriptor.stat-at": exports0["52"],
        "[method]descriptor.symlink-at": exports0["43"],
        "[method]descriptor.unlink-file-at": exports0["44"],
        "[method]descriptor.write-via-stream": exports0["46"],
        "[method]directory-entry-stream.read-directory-entry": exports0["54"],
        "[resource-drop]descriptor": trampoline5,
        "[resource-drop]directory-entry-stream": trampoline1,
        "filesystem-error-code": exports0["36"]
      },
      "wasi:io/error@0.2.3": {
        "[resource-drop]error": trampoline3
      },
      "wasi:io/poll@0.2.3": {
        "[resource-drop]pollable": trampoline10,
        poll: exports0["61"]
      },
      "wasi:io/streams@0.2.3": {
        "[method]input-stream.blocking-read": exports0["56"],
        "[method]input-stream.read": exports0["55"],
        "[method]input-stream.subscribe": trampoline9,
        "[method]output-stream.blocking-flush": exports0["60"],
        "[method]output-stream.blocking-write-and-flush": exports0["59"],
        "[method]output-stream.check-write": exports0["57"],
        "[method]output-stream.subscribe": trampoline8,
        "[method]output-stream.write": exports0["58"],
        "[resource-drop]input-stream": trampoline4,
        "[resource-drop]output-stream": trampoline2
      },
      "wasi:random/random@0.2.3": {
        "get-random-bytes": exports0["62"]
      }
    }));
    memory0 = exports1.memory;
    realloc0 = exports2.cabi_import_realloc;
    ({ exports: exports3 } = yield instantiateCore(yield module3, {
      "": {
        $imports: exports0.$imports,
        "0": exports2.args_get,
        "1": exports2.args_sizes_get,
        "10": exports2.fd_pread,
        "11": exports2.fd_prestat_get,
        "12": exports2.fd_prestat_dir_name,
        "13": exports2.fd_read,
        "14": exports2.fd_readdir,
        "15": exports2.fd_seek,
        "16": exports2.fd_write,
        "17": exports2.path_create_directory,
        "18": exports2.path_filestat_get,
        "19": exports2.path_link,
        "2": exports2.environ_get,
        "20": exports2.path_open,
        "21": exports2.path_readlink,
        "22": exports2.path_remove_directory,
        "23": exports2.path_rename,
        "24": exports2.path_symlink,
        "25": exports2.path_unlink_file,
        "26": exports2.poll_oneoff,
        "27": exports2.proc_exit,
        "28": exports2.random_get,
        "29": trampoline17,
        "3": exports2.environ_sizes_get,
        "30": trampoline18,
        "31": trampoline19,
        "32": trampoline20,
        "33": trampoline21,
        "34": trampoline22,
        "35": trampoline23,
        "36": trampoline24,
        "37": trampoline25,
        "38": trampoline26,
        "39": trampoline27,
        "4": exports2.clock_time_get,
        "40": trampoline28,
        "41": trampoline29,
        "42": trampoline30,
        "43": trampoline31,
        "44": trampoline32,
        "45": trampoline33,
        "46": trampoline34,
        "47": trampoline35,
        "48": trampoline36,
        "49": trampoline37,
        "5": exports2.fd_close,
        "50": trampoline38,
        "51": trampoline39,
        "52": trampoline40,
        "53": trampoline41,
        "54": trampoline42,
        "55": trampoline43,
        "56": trampoline44,
        "57": trampoline45,
        "58": trampoline46,
        "59": trampoline47,
        "6": exports2.fd_fdstat_get,
        "60": trampoline48,
        "61": trampoline49,
        "62": trampoline50,
        "63": trampoline51,
        "64": trampoline52,
        "65": trampoline53,
        "66": trampoline54,
        "7": exports2.fd_filestat_get,
        "8": exports2.fd_filestat_set_size,
        "9": exports2.fd_filestat_set_times
      }
    }));
    let run023Run;
    function run() {
      _debugLog('[iface="wasi:cli/run@0.2.3", function="run"] [Instruction::CallWasm] (async? false, @ enter)');
      const _wasm_call_currentTaskID = startCurrentTask(0, false, "run023Run");
      const ret = run023Run();
      endCurrentTask(0);
      let variant0;
      switch (ret) {
        case 0: {
          variant0 = {
            tag: "ok",
            val: void 0
          };
          break;
        }
        case 1: {
          variant0 = {
            tag: "err",
            val: void 0
          };
          break;
        }
        default: {
          throw new TypeError("invalid variant discriminant for expected");
        }
      }
      _debugLog('[iface="wasi:cli/run@0.2.3", function="run"][Instruction::Return]', {
        funcName: "run",
        paramCount: 1,
        postReturn: false
      });
      const retCopy = variant0;
      if (typeof retCopy === "object" && retCopy.tag === "err") {
        throw new ComponentError(retCopy.val);
      }
      return retCopy.val;
    }
    run023Run = exports2["wasi:cli/run@0.2.3#run"];
    const run023 = {
      run
    };
    return { run: run023, "wasi:cli/run@0.2.3": run023 };
  })();
  let promise, resolve, reject;
  function runNext(value) {
    try {
      let done;
      do {
        ({ value, done } = gen.next(value));
      } while (!(value instanceof Promise) && !done);
      if (done) {
        if (resolve) return resolve(value);
        else return value;
      }
      if (!promise) promise = new Promise((_resolve, _reject) => (resolve = _resolve, reject = _reject));
      value.then((nextVal) => done ? resolve() : runNext(nextVal), reject);
    } catch (e) {
      if (reject) reject(e);
      else throw e;
    }
  }
  const maybeSyncReturn = runNext(null);
  return promise || maybeSyncReturn;
}

// lib/api.js
var llvm = new Application(llvm_resources_exports, instantiate, "yowasp-llvm");
var runLLVM = llvm.run.bind(llvm);
function subcommand(command, subcommandName) {
  return function(args = null, files = {}, options = {}) {
    if (args === null)
      return command(args, files, options);
    return command([subcommandName, ...args], files, options);
  };
}
function runClang(args = null, files = {}, options = {}) {
  if (args === null)
    return runLLVM(args, files, options);
  if (args.includes("-###"))
    return runLLVM(args, files, options);
  if (args.includes(`--version`) || args.includes(`-help`) || args.includes(`--help`) || args.includes(`--help-hidden`))
    return runLLVM(args, files, options);
  function writeStderr(output) {
    if (options.stderr === void 0) {
      console.log(output);
    } else {
      options.stderr(new TextEncoder().encode(output));
      options.stderr(null);
    }
  }
  function unquoteClangArgs(line) {
    return Array.from(line.matchAll(/ (?:([^ "]+)|"((?:[^"\\$]|\\["\\$])+)")/g), (match) => {
      if (match[1] !== void 0) {
        return match[1];
      } else if (match[2] !== void 0) {
        return match[2].replaceAll(/\\["$\\]/g, (m) => m[1]);
      }
    });
  }
  let gen = (function* () {
    const [arg0, ...argsRest] = args;
    const outputSubarrays = [];
    function captureOutput(bytes) {
      if (bytes !== null)
        outputSubarrays.push(new Uint8Array(bytes));
    }
    let hash3Error = void 0;
    try {
      yield runLLVM([arg0, "-###", ...argsRest], files, {
        stdout: captureOutput,
        stderr: captureOutput,
        synchronously: options.synchronously
      });
    } catch (err) {
      hash3Error = err;
    }
    const outputArray = new Uint8Array(outputSubarrays.reduce((a, b) => a + b.length, 0));
    let outputLength = 0;
    for (const outputSubarray of outputSubarrays) {
      outputArray.subarray(outputLength, outputLength + outputSubarray.length).set(outputSubarray);
      outputLength += outputSubarray.length;
    }
    const output = new TextDecoder().decode(outputArray);
    if (hash3Error !== void 0) {
      writeStderr(output);
      throw hash3Error;
    }
    let state = 0;
    const commands2 = [];
    for (const line of output.split("\n")) {
      if (state === 0) {
        if (!(line.startsWith("clang") || line.startsWith("Target:") || line.startsWith("Thread model:") || line.startsWith("InstalledDir:") || line.startsWith("Build config:"))) {
          state = 1;
        }
      }
      if (state === 1) {
        if (line === " (in-process)") {
        } else if (line.startsWith(' "')) {
          commands2.push(unquoteClangArgs(line));
        } else if (line === "") {
          state = 2;
        } else {
          state = 3;
        }
        continue;
      }
      if (state === 2) {
        state = 3;
      }
      if (state === 3) {
        break;
      }
    }
    if (state !== 2) {
      writeStderr(output);
    } else {
      if (args.includes("-v"))
        writeStderr(output);
      for (const command of commands2) {
        if (command[0] === "") {
          command.shift();
        }
        try {
          files = yield runLLVM(command, files, options);
        } catch (err) {
          if (err instanceof Exit)
            delete err.files.tmp;
          throw err;
        }
      }
    }
    delete files.tmp;
    return files;
  })();
  let promise, resolve, reject;
  function runNext(value) {
    try {
      let done;
      do {
        ({ value, done } = gen.next(value));
      } while (!(value instanceof Promise) && !done);
      if (done) {
        if (resolve) resolve(value);
        else return value;
      }
      if (!promise) promise = new Promise((_resolve, _reject) => (resolve = _resolve, reject = _reject));
      value.then(
        (nextVal) => done ? resolve() : runNext(nextVal),
        (error) => {
          try {
            ({ value, done } = gen.throw(error));
          } catch (e) {
            reject(e);
          }
        }
      );
    } catch (e) {
      if (reject) reject(e);
      else throw e;
    }
  }
  const maybeSyncReturn = runNext(null);
  return promise || maybeSyncReturn;
}
var commands = {
  // LLVM tools
  "addr2line": subcommand(runLLVM, "addr2line"),
  // actually `symbolizer`
  "ar": subcommand(runLLVM, "ar"),
  "c++filt": subcommand(runLLVM, "c++filt"),
  "dwarfdump": subcommand(runLLVM, "dwarfdump"),
  "nm": subcommand(runLLVM, "nm"),
  "objcopy": subcommand(runLLVM, "objcopy"),
  "objdump": subcommand(runLLVM, "objdump"),
  "readobj": subcommand(runLLVM, "readobj"),
  "ranlib": subcommand(runLLVM, "ranlib"),
  // actually `ar`
  "size": subcommand(runLLVM, "size"),
  "strip": subcommand(runLLVM, "strip"),
  // actually `objcopy`
  "symbolizer": subcommand(runLLVM, "symbolizer"),
  // Compiler and linker
  "wasm-ld": subcommand(runLLVM, "wasm-ld"),
  "clang": subcommand(runClang, "clang"),
  "clang++": subcommand(runClang, "clang++")
};
var version = "22.0.0-git20542-10";
export {
  Exit,
  commands,
  runClang,
  runLLVM,
  version
};
