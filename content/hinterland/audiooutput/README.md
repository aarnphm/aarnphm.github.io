# Audio Output

macOS menu-bar audio routing. Native apps use Core Audio process taps. Websites use the included Chromium extension. The system default output stays unchanged; BlackHole is only a test fixture.

## Run

Requires macOS 14.2+ and Xcode command-line tools.

```sh
cd /Users/aarnphm/workspace/garden/content/hinterland/audiooutput
just run
```

`just install` adds a Finder alias at `/Applications/Audio Output.app` and registers the app for Raycast and Spotlight. The alias follows the bundle in this checkout across rebuilds.

Choose an output beside an application or site. **Normal output** removes the override. **Pause** releases routes without deleting assignments. Disconnected devices fall back to normal playback and resume when reconnected. Grant System Audio Recording access for native-app routes.

Each application and site has a 0–300% volume slider in 5% steps and a 100% reset. Volume is saved independently of the output selection; native applications can follow the system output with a volume override. 0% mutes, 200% doubles sample amplitude, and 300% triples it. Native output clamps peaks to full scale, so boosting loud material can distort. Pause restores normal playback.

Website volume requires local blob media or explicitly CORS-enabled media, an unprotected player, and browser support for AudioContext output selection. Click the player if the browser requires a playback gesture. Reload the extension and player tabs after upgrading. Players that already own their media element through Web Audio cannot be boosted by a second graph.

**Volume-key target** selects the output controlled by the keyboard without changing macOS’s system output. Choose STR-DH190 to control its audio while keeping Yamaha AG03MK2 as the system default. The selection and each device’s master level persist. A disconnected target resumes when reconnected; keys pass through while it is unavailable. Explicit targets use native device volume when available, including STR-DH190. Selecting such a target leaves its playback route and current hardware volume unchanged. Outputs without hardware volume require a single stereo stream and use the software master. **System output** follows macOS’s default and enables the software master only when that device has no hardware volume control. Click **Allow volume keys…** and grant Audio Output Accessibility access. Volume-up/down change the master by 1/16; Shift–Option uses 1/64 steps. Mute toggles without losing the saved level. The software master multiplies app gain, so 50% master with 200% app gain yields 100% effective gain. Native device volume is applied by the output itself; app and site gains are not multiplied a second time. Its percentage appears above the volume-key target. Per-device levels persist. Pause or disabling **Use volume keys** releases the master tap and passes keys through. In **System output** mode, switching to an output with hardware volume also releases the tap.

The master covers the target device’s single stereo stream, including browser sites routed to it. Sites routed to another output stay independent. Apps already routed by Audio Output receive the master in their existing renderer; the device tap excludes those processes and its own renderer to prevent duplicate gain or feedback. Device-scoped capture covers other apps and browser media without changing the extension.

The gear menu contains the standalone window, login-item switch, extension folder, and capture-permission settings. Keep the app at a stable path before enabling login startup.

## YouTube Music and other websites

Chromium web apps share their browser's audio process. The extension selects the output on the site's `<audio>` and `<video>` elements, keeping other tabs on their normal output.

```sh
just install-bridge
```

1. Open `chrome://extensions` in Helium or Chrome.
2. Enable Developer mode, choose **Load unpacked**, and select this project's `Extension` directory.
3. Reload the YouTube Music web app.
4. Select its output in Audio Output.
5. If device access is requested, approve it on the page. On Chromium versions without `selectAudioOutput`, output discovery requires microphone permission for that site. The extension immediately stops the microphone stream.

YouTube Music is enabled by default. For another HTTPS site, open the extension popup and choose **Enable on this site**. Permission is requested for that origin only. Web Audio graphs and cross-origin embedded players are not supported by this media-element implementation.

Sites return to normal output when paused, when the bridge disconnects, or within ten seconds of the desktop app exiting. Browser device permission remains under the browser's site settings.

## Development

```sh
just check           # Swift build and plist validation
just extension-check # JavaScript syntax
just build           # release app and stable local signature
just install         # build and add the Applications launcher
just icons           # regenerate the SVG, PNG, ICNS and menu-bar artwork
just probe           # real tap + render check through two installed BlackHole devices
python3 scripts/verify-bridge.py
```

The app is at `dist/Audio Output.app`. The build selects the sole available code-signing identity, or uses `AUDIOOUTPUT_SIGN_IDENTITY` when specified. A stable identity preserves the designated requirement across rebuilds. With multiple identities, select one explicitly. `AUDIOOUTPUT_SIGN_IDENTITY=-` opts into ad-hoc signing; each changed binary then needs renewed permissions. Local builds are not notarized.

`Resources/logo.svg` is the transparent Garden headphone mascot; `app-icon.svg` adds the macOS icon background. The menu-bar artwork uses a monochrome template and shows pause bars when routing is paused. Regenerate artwork with the repository's pnpm dependencies installed.

`AudioOutput --inventory` lists connected outputs and audio processes. `AudioOutput --probe PID OUTPUT_UID 5` temporarily routes one live process, reports rendered/nonzero frames, and verifies resource cleanup. Integration fixtures and results go to `docs/verification/audiooutput/artifacts/` at the repository root.

## Ownership

- `audio-hardware.swift`: devices, process identity, web-app metadata.
- `audio-route.swift` and `AudioRender`: tap, aggregate, format negotiation, allocation-free render callback, physical-input exclusion, teardown.
- `router-model.swift`: assignments, device/app reconciliation, sleep and login state.
- `site-routing.swift` and `native-host.swift`: local site settings, status, native-messaging protocol.
- `Extension`: site permission, device matching, media-element routing, reconnects.

Native assignments use UserDefaults (`xyz.aarnphm.audiooutput`). Site settings/status live in `~/Library/Application Support/AudioOutput`. The bridge installer registers only this extension's ID with installed Chromium browsers. It does not change browser profiles, default audio devices, or install an audio driver.

[Core Audio taps](https://developer.apple.com/documentation/coreaudio/capturing-system-audio-with-core-audio-taps) · [Native messaging](https://developer.chrome.com/docs/extensions/develop/concepts/native-messaging) · [Audio output specification](https://www.w3.org/TR/audio-output/)

The running app writes `~/Library/Application Support/AudioOutput/master-status.json` every refresh. Check its PID and update timestamp before using it: `trusted` reports Accessibility trust, `keyTap` reports event-tap creation, `handlesKeys` reports whether a master route is attached, and frame counters report audio received by that route. Zero nonzero frames do not establish audible gain control.
