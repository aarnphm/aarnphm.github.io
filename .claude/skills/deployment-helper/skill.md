# Deployment Helper

Manage Cloudflare Worker deployments and ensure code quality before deployment.

## When to Use

- User wants to deploy to Cloudflare
- User needs to validate code before deployment
- User wants to check deployment status
- User needs help troubleshooting deployment issues
- User wants to understand the deployment pipeline

## Instructions

### Pre-Deployment Validation

**ALWAYS run the complete validation pipeline before deployment:**

```bash
pnpm check
```

This command runs:
1. `pnpm format` - Format code with Prettier and organize References.bib
2. `pnpm convert` - Convert and process content
3. `wrangler types` - Generate Cloudflare Worker types
4. `prettier . --check` - Verify formatting
5. `tsc --noEmit` - TypeScript type checking
6. `tsx --test` - Run tests

**Do not proceed with deployment if any step fails.**

### Deploying to Cloudflare

Once validation passes:

```bash
pnpm cf:deploy
```

This command:
1. Runs `pnpm check` first (validation)
2. Runs `wrangler deploy` to deploy the worker

### Understanding the Deployment Configuration

**wrangler.toml** contains:
- Worker name: "portfolio"
- Main entry: `./worker/index.ts`
- KV namespaces: OAUTH_KV, FONTS, STACKED_CACHE
- R2 buckets: LFS_BUCKET (for large files)
- Browser binding: for Puppeteer
- AI binding: for Cloudflare AI
- Durable Objects: Garden (for MCP)
- Assets: `./public/` directory

**Custom domains:**
- aarnphm.xyz
- notes.aarnphm.xyz
- stream.aarnphm.xyz
- lyd.aarnphm.xyz

### Deployment Checklist

Before deploying:
- [ ] Run `pnpm check` and ensure all checks pass
- [ ] Verify no secrets in commits (use `.env` locally)
- [ ] Check that TypeScript has no errors
- [ ] Ensure tests pass
- [ ] Verify References.bib is properly formatted
- [ ] Check that `public/` directory is up to date

After deploying:
- [ ] Verify deployment succeeded (check wrangler output)
- [ ] Test the deployed site on all domains
- [ ] Check browser console for errors
- [ ] Verify static assets are loading correctly

### Troubleshooting Common Issues

**Type Errors:**
```bash
tsc --noEmit
```
Review and fix TypeScript errors before deployment.

**Format Issues:**
```bash
pnpm format
```
Auto-fixes formatting issues.

**Wrangler Errors:**
```bash
wrangler whoami  # Check authentication
wrangler tail    # View live logs
```

**Build Constraints:**
- Never run `pnpm bundle` or `pnpm build` directly
- The dev server (`dev.ts`) is always running
- Only inspect the running process, don't spawn new builds

### Local Development vs Production

**Local Dev:**
```bash
# Dev server is always running via dev.ts
# Changes are watched automatically
```

**Production Deployment:**
```bash
pnpm cf:deploy
```

### Secrets Management

**Never commit secrets!**

**Local development:**
- Use `.env` file (gitignored)
- Load with `process.env.SECRET_NAME`

**Production (Cloudflare):**
```bash
# Set secrets via wrangler
wrangler secret put SECRET_NAME

# List secrets
wrangler secret list
```

### Checking Deployment Status

**View live logs:**
```bash
wrangler tail
```

**View deployments:**
```bash
wrangler deployments list
```

**Check worker status:**
```bash
wrangler whoami  # Verify authentication
```

## Examples

### Example 1: Full Deployment Workflow

```bash
# 1. Validate everything
pnpm check

# If validation passes:
# 2. Deploy
pnpm cf:deploy

# 3. Monitor deployment
wrangler tail

# 4. Test the site
# Visit aarnphm.xyz and verify functionality
```

### Example 2: Fix Validation Errors

```bash
# Run validation
pnpm check

# If format errors:
pnpm format

# If TypeScript errors:
tsc --noEmit
# (Fix reported errors)

# If test failures:
tsx --test
# (Fix failing tests)

# Retry validation
pnpm check
```

### Example 3: Emergency Rollback

```bash
# List recent deployments
wrangler deployments list

# Rollback to previous version (if needed)
wrangler rollback <deployment-id>
```

### Example 4: Check Production Logs

```bash
# Tail live logs
wrangler tail

# Filter for errors
wrangler tail --format json | grep -i error

# View specific deployment logs
wrangler tail --deployment-id <id>
```

### Example 5: Update Secrets

```bash
# Add or update a secret
wrangler secret put API_KEY
# (Enter value when prompted)

# Verify secret exists (won't show value)
wrangler secret list

# Delete a secret
wrangler secret delete OLD_SECRET
```

## Notes

### Cloudflare Worker Features

**Bindings available:**
- `env.OAUTH_KV` - OAuth state storage
- `env.FONTS` - Font files cache
- `env.STACKED_CACHE` - Stacked notes cache
- `env.LFS_BUCKET` - R2 bucket for large files
- `env.BROWSER` - Puppeteer browser
- `env.AI` - Cloudflare AI models
- `env.MCP_OBJECT` - Durable Object for MCP
- `env.ASSETS` - Static assets from `public/`

**Asset Handling:**
- Static files served from `public/` directory
- Run worker first, then fall back to assets
- 404 handling via custom 404-page
- Assets are bound to worker for dynamic routing

**Performance:**
- Smart placement mode enabled
- Observability logs enabled
- Preview URLs available for testing

### Important Constraints

1. **Never run bundle/build directly** - dev server handles this
2. **No fs operations in transformers** - breaks parallelization
3. **Keep public/ reproducible** - via `pnpm bundle`
4. **Use Git LFS** - for large binaries
5. **Environment-specific behavior** - check for CF_PAGES and NODE_ENV

### Worker Entry Point

Main file: `worker/index.ts`

Related files in `worker/`:
- `auth.ts` - OAuth authentication
- `arxiv.ts` - ArXiv paper handling
- `github-handler.ts` - GitHub webhooks
- `semantic.ts` - Semantic search
- `stacked.tsx` - Stacked notes rendering
- `mcp.ts` - Model Context Protocol
- `curius.ts` - Curius integration

### Development Best Practices

1. Test locally first (dev server)
2. Run full validation before deployment
3. Check logs after deployment
4. Keep secrets in Cloudflare (not in code)
5. Use environment variables for configuration
6. Monitor performance and errors
7. Document any new bindings or features
