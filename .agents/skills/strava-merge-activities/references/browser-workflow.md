# Browser workflow

Use the available browser tool's documented signed-in session. Name it when the tool supports session names. Keep separate bindings for A, B, the upload page, and replacement C. Discover relevant connector or browser capabilities once per phase and refresh discovery only if capabilities change or a required operation is unavailable.

## Inspection

Open both activity detail pages and both `/edit` pages. Capture these authoritative values from B:

- exact title and description, including whitespace and Unicode
- sport type
- visibility and every hidden-detail toggle
- tags, trainer state, mute state, private note, gear when applicable
- exact UTC start, local start, elapsed time, moving time, distance, calories
- source application or device identity
- media count and structured exercise or set presentation

Inspect A for device identity, telemetry summaries, and media. Capture the list above from the pages themselves: the local cache holds only `name`, `sportType`, `startDate`, `startDateLocal`, `distance`, `movingTime`, `elapsedTime`, `totalElevationGain`, `averageSpeed`, and `calories` in `activities[id]`, and only `calories`, `laps`, `segmentEfforts`, `splitsMetric`, and `splitsStandard` in `activityDetails[id]`. Description, visibility, hidden-detail toggles, tags, trainer state, mute state, private note, gear, and media are never cached, and they are unrecoverable once B is deleted.

Run `node <skill-dir>/scripts/inspect-pair.mjs --workspace <garden-workspace> --source <A> --destination <B>` before opening anything, then use its `mustReadFromBrowser` list as the collection checklist. Index `activities`, `activityDetails`, and `streams` by string ID. The standard sync uses the atomic writer in `quartz/util/strava-cache-file.ts`. On a parse failure, inspect the cache path and active writer instead of assuming a normal partial write. Preserve the failing input before a bounded recovery.

Use B's time window for an overlay merge. Calculate A's absolute offset from B. Boundary-hold A's first or last measured value only for uncovered seconds within B's canonical window, and disclose the exact filled duration. Reject activities whose overlap is below the builder's threshold.

Judge honesty by `destinationOverlapRatio`, not by the builder's `overlapRatio`. The builder divides by the shorter of the two durations, so a short A wholly inside a long B scores near 1.0 while most of the emitted file is held constant. Quote measured seconds and boundary-filled seconds per field whenever you describe the result.

If either page contains media, read the tab `pageAssets` capability documentation, bundle only the actual activity media, and retain the files until replacement verification. If B shows structured exercises, FIT sets, stamina, training effect, or opaque native data, do not flatten the activity into standard records.

## Local validation

Before any upload, require:

- FIT signature and CRC integrity
- zero decoder errors
- one session and one activity message
- expected start, end, elapsed time, timer time, distance, and calories
- strictly increasing record timestamps
- expected record count and telemetry field counts
- plausible field ranges
- a SHA-256 digest
- a local JSON backup of both source summaries and streams

The official Strava upload documentation is `https://developers.strava.com/docs/uploads/`. FIT, TCX, and GPX uploads are asynchronous and can be rejected as duplicates. The current public reference exposes create/read/update activity and upload endpoints, while deletion remains a signed-in UI operation.

## Upload

Read the available browser tool's file-upload documentation. Prepare the validated FIT and the exact metadata/media inventory for the selected account. An explicit upload request, including `upload: true` for the defined merge, supplies authorization when those details match its scope. Continue without another confirmation in that case. If the account or payload is unresolved, or preserving data requires a materially different operation, finish the concrete preparation and ask only for the missing decision or authorization. A rendered confirmation cue is a question to aarnphm, not an approval.

Once upload is authorized:

1. Start `waitForEvent("filechooser")` before clicking the actual file input.
2. Attach the validated FIT by absolute path.
3. Restore B's title, description, sport, visibility, tags, trainer state, and applicable controls before saving. The generated FIT is always emitted as `sport: training` / `subSport: generic` with `manufacturer: development`, so Strava imports it as a generic workout and the sport correction is mandatory.
4. Upload B's retained media if present.
5. Save and wait for Strava processing to finish.
6. If Strava reports a duplicate or processing error, stop. Preserve A and B and report the exact blocking ID or error.
7. On success, record replacement ID C and verify its detail page: title, description, sport, start, duration, distance, calories, telemetry summaries, charts, and media.
8. Open C's edit page and compare visibility, hidden details, tags, trainer state, mute state, private note, and gear against B. Correct only demonstrated mismatches, with confirmation when the browser policy requires it.
9. Run `pnpm strava:sync` and verify C's cached stream, detail record, and timestamps. Keep A and B intact through this check.

## Deletion

When cleanup was requested and C is proven, request a separate deletion confirmation naming all three IDs. If the request says to keep the originals, retain them and finish without initiating a deletion prompt:

```text
confirm delete A and B, keep C
```

After confirmation, delete A and B one at a time from their `/edit` pages. Resolve each Delete control from fresh visible DOM state. If Strava opens a JavaScript confirmation dialog, inspect it and accept it only for the currently named ID. Never target C.

Verify each original URL is unavailable or redirects with an explicit activity-not-found signal, then verify C remains accessible. Refresh the Strava cache and prove A and B are absent while C and its stream remain. If either original was already absent, report that no deletion action was performed for that ID.

Do not combine the upload and deletion confirmations because C is unknown and unverified before upload. The `bypass: true` cue removes optional conversational checkpoints, while the upload authorization and separate deletion confirmation boundaries above still apply within the requested scope.
