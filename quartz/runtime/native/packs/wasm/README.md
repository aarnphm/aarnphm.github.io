The Wasm runtime pack vendors WABT `1.0.39` from the repo dependency and the shared browser WASI shim used by the native runtime packs.

Runtime artifacts:

- `wabt.mjs`: bundled from `node_modules/.pnpm/wabt@1.0.39/node_modules/wabt/index.js`
- `browser-wasi-shim.mjs`: `https://esm.sh/gh/haskell-wasm/browser_wasi_shim@2f86b49/es2022/browser_wasi_shim.bundle.mjs`

The worker accepts WAT source, wasm base64, and wasm hex, then instantiates the module in the browser. WASI modules run through the shim, and plain modules call `_start`, `main`, `run`, `start`, or the single exported function when one exists.
