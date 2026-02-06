# Subdirectory Context for RFC Generation

Map from command argument to filesystem path and component context for ~/workspace/modular

## Top-level components

| Argument     | Path          | Scope                                        |
| ------------ | ------------- | -------------------------------------------- |
| `max`        | `max/`        | MAX platform (broad, cross-cutting)          |
| `Kernels`    | `Kernels/`    | High-performance compute kernels (Mojo, GPU) |
| `KGEN`       | `KGEN/`       | Mojo compiler / kernel generator             |
| `GenericML`  | `GenericML/`  | ML graph compiler                            |
| `SDK`        | `SDK/`        | MAX SDK APIs and tools                       |
| `Support`    | `Support/`    | Core utilities and infrastructure            |
| `CloudInfra` | `CloudInfra/` | Cloud infrastructure                         |

## MAX subcomponents (shorthand)

These resolve to paths under `max/python/max/`:

| Argument    | Path                        | Scope                                             |
| ----------- | --------------------------- | ------------------------------------------------- |
| `nn`        | `max/python/max/nn/`        | Neural network modules (linear, conv, norm, rope) |
| `kv_cache`  | `max/python/max/kv_cache/`  | KV cache management, paged attention, connectors  |
| `serve`     | `max/python/max/serve/`     | Serving infra (scheduler, router, API server)     |
| `pipelines` | `max/python/max/pipelines/` | Model pipeline architectures and configs          |
| `graph`     | `max/python/max/graph/`     | Graph API, ops, weight management                 |
| `engine`    | `max/python/max/engine/`    | Inference session management                      |
| `driver`    | `max/python/max/driver/`    | Low-level device and tensor management            |
| `config`    | `max/python/max/config/`    | Configuration management                          |
| `compiler`  | `max/compiler/`             | MAX compiler                                      |

## Kernel subcomponents

| Argument       | Path            | Scope                          |
| -------------- | --------------- | ------------------------------ |
| `kernels/lib`  | `Kernels/lib/`  | Kernel library implementations |
| `kernels/test` | `Kernels/test/` | Kernel test suites             |
| `max/kernels`  | `max/kernels/`  | MAX-specific kernel wrappers   |

## Resolution rules

1. Exact match against Argument column (case-sensitive for top-level, case-insensitive for MAX subcomponents)
2. If argument contains `/`, treat as relative path from repo root
3. If no match, search for directory name in common locations: `max/python/max/`, `max/`, top-level
4. If still no match, ask the user to clarify

## Context gathering per component

After resolving the path, gather context by:

1. Read the component's CLAUDE.md if it exists
2. List the directory structure (1-2 levels deep)
3. Check for existing design docs in `docs/internal/projects/` related to the component
4. Identify adjacent components that will be affected (e.g., KV cache changes affect serve/scheduler)
5. Check recent git log for the component to understand velocity and active contributors:
   ```bash
   git log --oneline -20 -- <resolved-path>
   ```
