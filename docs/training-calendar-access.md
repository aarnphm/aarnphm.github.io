# TrainingPeaks calendar access

The password protects the imported TrainingPeaks calendar: workout titles, prescriptions, notes, planned and actual metrics, zones, and activity matches. The separate reference plans, race calendar, and recorded activity pages remain public.

The Strava emitter encrypts the calendar with AES-256-GCM and a random 96-bit nonce. `/static/training-calendar.json` contains only authenticated ciphertext, or `null` when the build key is missing or invalid. This replaces any older plaintext file on the next successful emission. The HTML contains only a password form. The calendar never enters render data, inline JSON, or a browser bundle. The encryption key is independent of the password, so publicly reachable ciphertext does not provide a password-guessing oracle.

The Worker rejects direct requests to that asset before redirects, host rewrites, and generic static/CORS handling. `/api/training-calendar` authenticates every GET or HEAD, reads the asset internally, and decrypts it on the server. It fails closed if the key, password hash, Durable Object binding, or encrypted asset is absent, invalid, or incompatible. There is no fallback to plaintext. Keep `[assets].run_worker_first = true`.

## Local setup

Run `pnpm exec tsx quartz/scripts/setup-training-calendar.ts`. Set `TRAININGPEAKS_CALENDAR_PASSWORD` in the process environment or ignored `.env` to run without a prompt; the process environment takes precedence. When neither source defines it, the command asks for the password twice in an interactive terminal with hidden input. Use a unique password or password-manager-generated passphrase (15–128 characters). An empty or invalid environment value stops setup before any writes.

Setup derives `TRAINING_CALENDAR_PASSWORD_HASH` from that password. It writes only the salted scrypt verifier and random 256-bit data key to ignored `.env` and an existing `.dev.vars`, preserves other settings, and retains an existing data key. It never prints the password or adds it to generated secrets; a password entry you put in `.env` remains there as an existing setting. The Worker uses the generated verifier, so changing `TRAININGPEAKS_CALENDAR_PASSWORD` requires rerunning setup and uploading the new verifier. An unreadable settings file stops setup.

The script also writes the two secrets to ignored `.quartz-cache/training-calendar-secrets.json` with mode `0600`, ready for an explicit secrets upload. Treat this file and both environment files as secrets. No real credentials belong in test fixtures or verification evidence.

The build environment must load `TRAINING_CALENDAR_DATA_KEY`. The existing container workflow loads `.env`; a shell launch can use `node --env-file=.env node_modules/tsx/dist/cli.mjs quartz/scripts/dev.ts` when starting the watcher. Preserve an already running watcher; arrange its environment at the next intended restart. A watcher started without the key safely emits `null` until its environment is updated. Never add the key to esbuild defines or client configuration.

Wrangler reads `.dev.vars` when that file exists, otherwise `.env`. Setup preserves this choice. There is no development authentication bypass. Localhost HTTP is allowed for local testing; the session cookie retains `Secure`. Public hosts require HTTPS.

## Server controls

- Password hashing uses native `node:crypto` scrypt, `N=32768`, `r=8`, `p=3`, a random 128-bit salt, and a constant-time comparison. These parameters follow an [OWASP scrypt profile](https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html). Cloudflare [supports scrypt but excludes Argon2 from its Node crypto API](https://developers.cloudflare.com/workers/runtime-apis/nodejs/crypto/).
- A single SQLite Durable Object stores hashed 256-bit session tokens, their origin, credential generation, and expiry. New sessions last seven days from login, including time away from the calendar. Reads preserve that deadline. The open calendar clears itself when its session deadline arrives. Logout deletes the session immediately. Changing the verifier or data key invalidates all old sessions on the next request. Existing sessions retain their original absolute expiry; log in again to receive a full week. The object removes the obsolete idle-expiry column when it opens existing storage.
- Cookies use `__Host-`, `Secure`, `HttpOnly`, `SameSite=Strict`, and `Path=/`, with no `Domain`. Tokens and passwords never enter local/session storage or URLs. The browser clears private data on lock, expiry, page exit, and controller disposal; a BroadcastChannel clears other open instances on logout.
- Login and logout require an exact matching Origin and reject a cross-origin Fetch Metadata header. Login accepts only JSON, reads at most 4096 bytes, and rejects oversized passwords.
- The shared object limits attempts to five per client IP and thirty across the calendar per fifteen minutes. Counters include successful attempts, survive Worker restarts, and are not reset by changing hostnames. IP values are hashed before storage. A blocked request returns HTTP 429 with `Retry-After`.
- Every authentication/data response uses `Cache-Control: private, no-store`, explicit CDN no-store headers, `Vary: Cookie`, `nosniff`, and same-origin resource policy. No permissive CORS headers are added. Session handling follows the [OWASP session guidance](https://cheatsheetseries.owasp.org/cheatsheets/Session_Management_Cheat_Sheet.html).

## Deployment

This change alone does not alter the live site. Deployment requires the same `TRAINING_CALENDAR_DATA_KEY` in the build environment and the Worker, plus `TRAINING_CALENDAR_PASSWORD_HASH` in the Worker. After local setup, the explicit upload command is `pnpm exec wrangler secret bulk .quartz-cache/training-calendar-secrets.json`. The Worker deployment also applies the new `training-calendar-001` Durable Object migration declared in `wrangler.toml`.

Publish a newly emitted site together with the updated Worker. Verify that the build asset is ciphertext and public HTML has no `data-training-payload`. Purge any old plaintext JSON and HTML from external caches and remove access to old public deployment versions or mirrors that still contain the calendar. A new password cannot revoke copies someone already downloaded. The encrypted artifact stays confidential on a static mirror as long as its key remains secret.

Use HTTPS on the published site and verify all exposed hostnames, including the apex, `t.aarnphm.xyz`, and `workers.dev`. Avoid cache rules that override these response headers. If the data key is rotated, rebuild the encrypted asset with the new key before publishing the matching Worker secret. Password rotation needs only the new hash; it does not require re-encrypting data.

## Verification

Run `pnpm test worker/training-calendar.test.ts`. This starts the complete Worker in workerd against synthetic assets and a real SQLite Durable Object. It saves response status, headers (cookies redacted), bodies, and a repeat command under the printed `garden-training-access-*` directory. It verifies direct asset denial, encoded paths and multiple hosts, wrong passwords, a successful read, cookie flags and tampering, origin binding, duplicate cookies, cross-origin mutations, body limits, session expiry, credential changes, revocation, and rate limiting. It also checks the generated password form and encrypted build output.

An authenticated person can download the calendar they are authorized to view. The access control protects unauthenticated requests; it does not provide DRM or a guarantee against a compromised browser or server. This implementation uses the cited controls and is not a formal compliance certification.
