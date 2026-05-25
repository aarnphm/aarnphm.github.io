The Mojo runtime pack is intentionally unavailable until Mojo can produce or run browser-hosted WebAssembly.

Current gate:

- `mojo build --print-supported-targets` in this checkout lists CPU and accelerator targets only: AArch64, ARM, AMD GCN, Hexagon, NVIDIA PTX, RISC-V, X86, and Xtensa.
- The command output does not list a wasm or WebAssembly target.
- Modular documents cross-compilation as object, assembly, LLVM IR, and LLVM bitcode output. Linked cross-compiled executables still require a target linker and runtime libraries.

The implementation path is the same native runtime-pack ABI used by Go, Haskell, OCaml, and Rust:

1. Vendor a Mojo compiler, interpreter, or runtime artifact that runs in a browser worker.
2. Emit any sysroot or runtime-library assets through `manifest.json`.
3. Make `mojo/worker.js` accept notebook source, execute it inside the worker sandbox, and return notebook stream/error/done events.
4. Flip the manifest entry from `available: false` to a worker/assets entry only after a browser or worker smoke test proves execution.
