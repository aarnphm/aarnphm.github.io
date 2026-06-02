The C and C++ runtime pack vendors YoWASP Clang `22.0.0-git20542-10` (LLVM 22.0.x) from the npm artifact `@yowasp/clang`, with its WASI sysroot replaced by the exceptions-enabled sysroot from `wasi-sdk-33` (LLVM 22.1.0).

Runtime artifacts:

- `bundle.js`, `llvm.core*.wasm`: `https://registry.npmjs.org/@yowasp/clang/-/clang-22.0.0-git20542-10.tgz`
- `browser-wasi-shim.mjs`: `https://esm.sh/gh/haskell-wasm/browser_wasi_shim@2f86b49/es2022/browser_wasi_shim.bundle.mjs`
- `llvm-resources.tar`: a merge of two upstreams (see below).

`llvm-resources.tar` keeps YoWASP's compiler-side files (clang builtin/intrinsic headers under `include/`, and `lib/wasm32-unknown-wasip1/libclang_rt.*`) and replaces the WASI sysroot with `wasi-sdk-33`'s `wasi-sysroot-33.0+m.tar.gz` (`https://github.com/WebAssembly/wasi-sdk/releases/download/wasi-sdk-33`):

- `include/c++/v1` ← `include/wasm32-wasip1/eh/c++/v1` (exceptions-enabled libc++ headers)
- `include/wasm32-wasip1` ← wasi-libc headers (without the `eh`/`noeh` libc++ trees)
- `lib/wasm32-wasip1` ← the base libc/crt plus the flattened `eh/` variant of `libc++.a`, `libc++abi.a`, `libunwind.a`, `libc++experimental.a`

Both sides are LLVM 22 with `_LIBCPP_VERSION 220100` / `_LIBCPP_ABI_VERSION 2`, and the prebuilt libc++abi/libunwind use the standardized exnref exception model (`try_table`/`throw_ref`).

The tar is packed as plain USTAR (no PAX/GNU extended headers, directory entries listed parent-first without a trailing slash) so the `nanotar` reader in `bundle.js` mounts it correctly.

The worker compiles C and C++ cells to WASI WebAssembly with Clang, then runs the emitted module through `browser-wasi-shim`. C++ cells compile with `-fwasm-exceptions -mllvm -wasm-use-legacy-eh=false -lc++abi -lunwind`: the `-wasm-use-legacy-eh=false` is required because YoWASP Clang 22.0.x still defaults `-fwasm-exceptions` to the legacy EH instructions, which would otherwise mix with the exnref-form prebuilt libraries and be rejected by the engine. The exnref model needs V8 ≥ Chrome 137 (mid-2025).
