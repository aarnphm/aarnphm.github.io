The Rust runtime pack vendors the Rubri browser Miri artifacts from `LyonSyonII/rubri` at `main`, pushed on 2025-11-19.

Runtime artifacts:

- `bin/miri.opt.1718474653.wasm.gz`: `https://github.com/LyonSyonII/rubri/tree/main/example/public/wasm-rustc/bin`
- `lib/rustlib/x86_64-unknown-linux-gnu/lib/*.gz`: `https://github.com/LyonSyonII/rubri/tree/main/example/public/wasm-rustc/lib/rustlib/x86_64-unknown-linux-gnu/lib`
- `browser-wasi-shim.mjs`: `https://esm.sh/gh/haskell-wasm/browser_wasi_shim@2f86b49/es2022/browser_wasi_shim.bundle.mjs`

The worker runs Rust snippets through Miri with the x86_64 sysroot instead of invoking `rustc` and a linker. The checked-in Miri and sysroot files stay gzip-compressed in the source tree and are decompressed in the worker before instantiation.
