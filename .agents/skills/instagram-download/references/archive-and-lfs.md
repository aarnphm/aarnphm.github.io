# Garden archive and LFS

Run from the Garden checkout. Commands also accept `--repo /absolute/checkout`, so a second chat can use the same entry points against the publishing checkout. Use the current checkout's scripts, not files under a previous chat's Documents directory.

## Archive downloaded media

Preview one or more verified download folders:

```sh
python3 .agents/skills/instagram-download/scripts/archive-media.py \
  --source ~/Downloads/instagram-SHORTCODE
```

Repeat `--source` for additional folders. Add `--apply` to copy the files, or `--apply --move` to remove verified source files after the attribution map is saved. Preview performs no writes. Applying serializes concurrent imports with `.quartz-cache/instagram-archive.lock`; a changed attribution map stops the operation instead of overwriting another writer's changes.

Inputs need `verification.json` or `manifest.json` with an Instagram post URL, posting account, and listed output files. Failed captures, missing outputs, incorrect hashes, unresolved LFS pointers, and images at or above 25 MiB are rejected before destination writes. Keep incomplete historical captures separate rather than labeling them successful.

The script preserves image bytes and writes unpadded base64url SHA-256 filenames with their original extensions. Images go directly in `content/thoughts/images/`. Video, audio, ZIPs, and raw acquisition reports go flat in `content/thoughts/images/instagram/`. Existing Garden ignore rules keep those non-image files local.

`content/thoughts/images/instagram/attribution.json` maps vault-relative wikilinks to author handles, original post URL, media kind, slide position, and SHA-256. Existing entries remain intact. Identical content with distinct attribution gets a deterministic suffix; repeating the same import reuses the existing files and map entry. This records posting-account provenance; it does not establish upstream authorship of reposted work. Do not create Markdown author indexes.

## LFS lifecycle

The shared CLI is `node quartz/scripts/lfs-assets.mjs COMMAND`. Git LFS and the checkout's existing R2 authentication must already work; these commands do not configure credentials.

| Command            | Behavior                                                                                                                                                                                                   |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `audit`            | Read-only source size/pointer check; verifies every eligible image's working bytes against its indexed LFS pointer. New or changed images need staging first.                                              |
| `stage-images`     | Validates source assets, stages `.gitattributes`, eligible images and attribution JSON, then forces clean-filter conversion. Preserves unrelated staged blobs.                                             |
| `upload-images`    | Audits the index and uploads only its distinct image object IDs through the configured LFS endpoint. Does not commit or push Git refs.                                                                     |
| `hydrate`          | Explicitly pulls native packs, wind archives, images and `content/hinterland/app/**`, clearing configured fetch exclusions for this invocation, then audits images.                                        |
| `verify-hydration` | Requires committed image/app/attribute changes. Creates a temporary checkout of HEAD, clears its temporary LFS cache, downloads images and app files, verifies hashes, and removes the temporary checkout. |

Every command accepts `--repo PATH` and `--report PATH`. Reports omit credentials. An operation exceeding 280 seconds fails. Use mutation commands only within the current task's authorization for Git, local file, or R2 writes. These commands never rewrite history, commit, push Git refs, or deploy the site.

A normal authorized update is:

```sh
node quartz/scripts/lfs-assets.mjs stage-images
node quartz/scripts/lfs-assets.mjs upload-images
# Commit/push only when requested, then verify the committed snapshot:
node quartz/scripts/lfs-assets.mjs verify-hydration --report /absolute/path/hydration.json
```

For the publishing checkout:

```sh
node quartz/scripts/lfs-assets.mjs hydrate --repo ~/workspace/aarnphm.github.io
```

`deploy.sh` already hydrates these required paths before building. It runs `validate-deployment-assets.mjs` before the build and against `public/` afterward. The validator rejects unresolved image/Markdown LFS pointers and images at or above 25 MiB. LFS does not reduce deployed asset size; retain this validation even when all Git entries are pointers.

## Verification

```sh
python3 .agents/skills/instagram-download/scripts/archive-media.test.py
node --test quartz/scripts/lfs-assets.test.mjs quartz/scripts/validate-deployment-assets.test.mjs
node quartz/scripts/lfs-assets.mjs audit
```

Archive tests use temporary files and actual JSON maps. LFS staging tests use a real temporary Git repository and Git LFS, verify unchanged image bytes, and check preservation of unrelated staging. `verify-hydration` is the real network/R2 check; run it when download verification is requested or a transfer change needs evidence.
