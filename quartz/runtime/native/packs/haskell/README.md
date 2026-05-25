The Haskell runtime pack vendors the GHC-in-browser artifacts published from `haskell-wasm/ghc-in-browser` at the `gh-pages` branch pushed on 2025-11-01.

Runtime artifacts:

- `dyld.mjs`, `post-link.mjs`, `prelude.mjs`, `rootfs.tar.zst`: `https://github.com/haskell-wasm/ghc-in-browser/tree/gh-pages`
- `bsdtar.wasm`: `https://haskell-wasm.github.io/bsdtar-wasm/bsdtar.wasm`
- `browser-wasi-shim.mjs`: `https://esm.sh/gh/haskell-wasm/browser_wasi_shim@2f86b49/es2022/browser_wasi_shim.bundle.mjs`

The root filesystem contains GHC `9.14.0.20251031-inplace` browser bytecode interpreter assets. The checked-in tarball is zstd-compressed and should stay compressed in the source tree.
