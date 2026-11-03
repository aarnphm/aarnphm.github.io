# Audio Output

macOS menu-bar audio routing. Native apps use Core Audio process taps. Websites use the included Chromium extension. The system default output stays unchanged; BlackHole is only a test fixture.

## Run

Requires macOS 14.2+ and Xcode command-line tools.

```sh
cd /Users/aarnphm/workspace/garden/content/hinterland/audiooutput
just run
```

Choose an output beside an application or site. **Normal output** removes the override. **Pause** releases routes without deleting assignments. Disconnected devices fall back to normal playback and resume when reconnected. Grant System Audio Recording access for native-app routes.

Each application and site has a 0–300% volume slider in 5% steps and a 100% reset. Volume is saved independently of the output selection; native applications can follow the system output with a volume override. 0% mutes, 200% doubles sample amplitude, and 300% triples it. Native output clamps peaks to full scale, so boosting loud material can distort. Pause restores normal playback.

Website volume requires local blob media or explicitly CORS-enabled media, an unprotected player, and browser support for AudioContext output selection. Click the player if the browser requires a playback gesture. Reload the extension and player tabs after upgrading. Players that already own their media element through Web Audio cannot be boosted by a second graph.

When the selected system output has no hardware volume control (including the Yamaha AG03MK2), **Use volume keys** enables a software master from 0–100%. Click **Allow volume keys…** and grant Audio Output Accessibility access. Volume-up/down change the master by 1/16; Shift–Option uses 1/64 steps. Mute toggles without losing the saved level. The master multiplies app gain, so 50% master with 200% app gain yields 100% effective gain. Its percentage appears beside the system output. Per-device levels persist. Pause, disabling the option, or switching to a hardware-volume-capable output releases the master tap and passes keys through.

The master covers the selected device's single stereo stream. Sites routed to another output stay independent. Apps already routed by Audio Output receive the master in their existing renderer; the device tap excludes those processes and its own renderer to prevent duplicate gain or feedback. Device-scoped capture covers other apps and browser media without changing the extension.

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
just build           # release app and ad-hoc signature
just probe           # real tap + render check through two installed BlackHole devices
python3 scripts/verify-bridge.py
```

The app is at `dist/Audio Output.app`. Set `AUDIOOUTPUT_SIGN_IDENTITY` to use a stable signing identity. Ad-hoc builds are local and not notarized; rebuilding may require renewed audio-capture permission.

`AudioOutput --inventory` lists connected outputs and audio processes. `AudioOutput --probe PID OUTPUT_UID 5` temporarily routes one live process, reports rendered/nonzero frames, and verifies resource cleanup. Integration fixtures and results go to `docs/verification/audiooutput/artifacts/` at the repository root.

## Ownership

- `audio-hardware.swift`: devices, process identity, web-app metadata.
- `audio-route.swift` and `AudioRender`: tap, aggregate, format negotiation, allocation-free render callback, physical-input exclusion, teardown.
- `router-model.swift`: assignments, device/app reconciliation, sleep and login state.
- `site-routing.swift` and `native-host.swift`: local site settings, status, native-messaging protocol.
- `Extension`: site permission, device matching, media-element routing, reconnects.

Native assignments use UserDefaults (`xyz.aarnphm.audiooutput`). Site settings/status live in `~/Library/Application Support/AudioOutput`. The bridge installer registers only this extension's ID with installed Chromium browsers. It does not change browser profiles, default audio devices, or install an audio driver.

[Core Audio taps](https://developer.apple.com/documentation/coreaudio/capturing-system-audio-with-core-audio-taps) · [Native messaging](https://developer.chrome.com/docs/extensions/develop/concepts/native-messaging) · [Audio output specification](https://www.w3.org/TR/audio-output/)
