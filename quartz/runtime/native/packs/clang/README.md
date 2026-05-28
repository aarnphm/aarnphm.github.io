The C and C++ runtime pack vendors YoWASP Clang `22.0.0-git20542-10` from the npm artifact `@yowasp/clang`.

Runtime artifacts:

- `bundle.js`, `llvm-resources.tar`, `llvm.core*.wasm`: `https://registry.npmjs.org/@yowasp/clang/-/clang-22.0.0-git20542-10.tgz`
- `browser-wasi-shim.mjs`: `https://esm.sh/gh/haskell-wasm/browser_wasi_shim@2f86b49/es2022/browser_wasi_shim.bundle.mjs`

The worker compiles C and C++ cells to WASI WebAssembly with Clang, then runs the emitted module through `browser-wasi-shim`.
