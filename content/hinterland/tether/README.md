# Tether

A native macOS companion for reading activity FIT files from Garmin watches and syncing them to Garmin Connect.

```sh
./scripts/build.sh
open dist/Tether.app
```

Requires macOS 14+, Xcode's Swift toolchain, an installed code-signing identity, and the installed Apple Silicon OpenMTP transport at `/Applications/OpenMTP.app`. OpenMTP does not need to run. The build adds no package dependencies. The bundle is signed for local use and is not notarized for distribution.

The build selects the sole valid identity reported by `security find-identity -v -p codesigning`. Set `TETHER_SIGN_IDENTITY` to an identity name or certificate hash when several are installed. Missing or ambiguous identities stop the build; ad-hoc signing is rejected. Both the helper and app retain fixed signing identifiers, and the previous app remains available if packaging fails.

Keep the same signing identity across rebuilds so macOS can recognize the app's Keychain authorization. The first certificate-signed build may need approval for a session saved by an older ad-hoc build. Choose **Always Allow** in the macOS Keychain prompt to authorize subsequent access by that signing identity. This authorizes local credential access; the Garmin session can still expire and need refresh. Changing signing identities or distribution channels can require another approval. See Apple's [code-signing requirements](https://developer.apple.com/documentation/technotes/tn3127-inside-code-signing-requirements).

## Current behavior

- Native SwiftUI window, Garden paper/sage palette, light/dark appearance, and a generated Garden mascot icon.
- Connection, sync/cancel and device options share the header with FIT import. The person icon opens Garmin Connect in the default browser; Device → Account opens Tether's login settings.
- Event-driven USB discovery. Known devices survive reconnect; vendor-specific Garmin USB interfaces do not become saved MTP devices. Readiness notifications act immediately, with 50 ms fallback checks during the first five seconds and 250 ms checks afterward while preparing. Ready devices have no polling timer; wake and app activation refresh the registry.
- Automatic sync on each stable USB attachment, including launch with a watch attached. Repeated notifications do not start another transfer; cancellation stays cancelled until reconnect or an explicit retry.
- Sync progress shows connection, reading, saving, upload and verification stages, then actual activity counts. Numeric changes flip over 120 ms; Reduce Motion keeps them static. The progress strip closes after completion, cancellation or failure.
- Multiple saved devices, with one MTP device accessed at a time. Every session checks the selected serial before reading.
- Read-only activity transfer from `GARMIN/Activity` and `GARMIN/Activity/PendingHD`. Files remain on the watch.
- FIT structure, activity type, checksum and SHA-256 validation before upload. An atomic local archive retains bytes and receipts.
- Incremental sync reuses a file only when its device, path, size, timestamp and local hash match.
- Native authenticated uploads, duplicate recognition and separate remote activity verification.
- Uncertain uploads are reconciled by comparing exact bytes with Garmin's original downloads. An uncertain POST is never automatically repeated.
- Import local FIT files with Command-O; sync with Command-R. The activity menu reveals local files; confirmed rows link to Connect.

`~/Library/Application Support/app.aarnphm.tether/` contains private device and transfer records. Credentials use a non-synchronizing Keychain item. Device → Account → Login opens Garmin Connect in the default browser. Tether does not host the login form. Browser sign-in and Tether's stored API session are separate; opening Connect does not import or refresh that session. Existing imported credentials remain access-only, and their original files and refresh owners remain untouched.

## Boundaries

The tested device is a Forerunner 970 over USB MTP. Bluetooth, mounted-volume devices, Garmin account device registration, wellness/health sync, firmware, maps, music and Connect IQ installation are not implemented. Adding a device registers it locally in Tether.

Uploads use Garmin's private Connect protocol. The documented Personal API currently exposes reading, not FIT upload. This adapter may need updates when Garmin changes its private service.

USB detection never waits on Keychain, network or MTP. The app starts syncing as soon as macOS publishes a usable MTP identity. Garmin's vendor-specific USB interface remains in the connecting state because it does not expose activity files through MTP. On a Forerunner 970, select Watch Settings → System → Advanced → USB Mode → MTP. Cable, watch-mode and firmware delays before that interface appears are separate from app scheduling.

The transport is isolated in `tether-mtp`: each read owns a serial-checked process with a 20-second deadline. It dynamically loads the already-installed OpenMTP library. Distribution needs a separately reviewed, bundled transport and its license notices.

## Verify

```sh
swift build
# From the Garden root:
python3 docs/verification/tether/replay.py
content/hinterland/tether/.build/out/Products/Debug/tether-probe devices
content/hinterland/tether/.build/out/Products/Debug/tether-probe latency
```

The archive replay uses a real repository FIT, exercises duplicate import and process-restart persistence, rejects corrupt/truncated/non-activity files, and verifies missing/corrupt local archive recovery. It uses a temporary private archive and sends nothing to Garmin.

`docs/verification/tether/README.md` records hardware, network and rendered-app evidence separately. `tether-probe upload` performs a real provider write; it is deliberately absent from the offline replay.

## Icon

The icon extends Garden's existing pen-drawn mascot with a watch strap and cable. Sources and renditions are in `Resources/`. Regenerate from the Garden root:

```sh
pnpm exec tsx content/hinterland/tether/scripts/build-icon.ts
```

The script writes SVG, a 1024px PNG and a multi-resolution ICNS using the repository's existing Sharp installation and macOS `iconutil`.
