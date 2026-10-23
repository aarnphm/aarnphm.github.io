---
name: instagram-download
description: Download Instagram photos, carousels, reels, and available audio from supplied post links into local files. Use when an Instagram link is supplied for media retrieval, including separate audio extraction or compact video downloads.
---

# Instagram Download

Save the media from the supplied Instagram post and return usable local file links. Default to `~/Downloads/instagram-<shortcode>/`. Respect a requested destination or media subset.

## Acquisition

Use the browser controls available in the current session and read their live documentation. Open the actual post, establish its media type, and capture its own loaded files. The public browser often works for reels. A signed-in browser can expose additional media; photo-post music may remain unavailable on the web even while signed in.

Read [references/browser-capture.md](references/browser-capture.md) for carousel ordering, DASH streams, the capture schema, and recovery from failed asset downloads.

- For a carousel, visit every slide and retain its rendered position. Complete the traversal when the Next control disappears. Exclude suggested posts, avatars, covers for unrelated reels, and advertisements.
- For video, identify the actual player's streams. Pair separate video and audio through their matching asset identifier and the player's duration. Inventory alone is insufficient, because Instagram preloads unrelated media.
- For a photo with music, save the photo and the available attached audio. A suggested reel's soundtrack is unrelated. If the track is a full music asset, use the post's documented excerpt interval.
- Use `pageAssets.bundle()` or the documented media downloader when possible. For a public CDN file observed in the page, the helper can download it directly. It removes only `bytestart` and `byteend`, which otherwise yield fragments rather than complete files.
- If audio is absent from the accessible web view, report **audio not exposed**. Assert **no audio** only when the requested post's own media metadata or downloaded video establishes that fact. Do not invent a soundtrack or substitute another post's audio.
- Stop on a login, access, or browser restriction that cannot be resolved within the requested access. Deliver the files obtained and name the missing media.

Keep browser authentication in the browser. Do not export cookies or store credentials in capture manifests. Refresh expired CDN links through the post rather than retrying guessed URLs.

## Local processing

Write a capture JSON using the schema in the browser reference, then run:

```sh
python3 <skill-directory>/scripts/process-media.py \
  --manifest /absolute/path/capture.json \
  --output-dir /absolute/path/instagram-<shortcode>
```

The helper uses Python's standard library plus existing `curl`, `ffmpeg`, and `ffprobe`. It accepts either observed CDN URLs or already downloaded local files. It creates:

- Original image bytes, numbered in slide order, with the extension derived from the actual file. A `.heic` CDN URL can serve JPEG bytes.
- A compact H.264 MP4, capped at 1080p without upscaling, with CRF 23, a 5 Mb/s rate cap, and fast-start metadata. It preserves the source audio when its codec supports M4A. Use the original downloaded stream as input when available.
- Separate M4A and MP3 audio when present. Preserve the attached excerpt interval. Ordinary video exports retain the full source duration.
- `verification.json` with commands, dimensions, durations, frame counts, file hashes, and full-decode results. Multiple media items also get a CRC-checked ZIP.

Existing output folders are refused to prevent overwrites. A failed run retains its report and partial files, marks its status failed, and prints the temporary source directory for recovery.

If a software encode stalls under memory pressure on macOS, recover from the downloaded sources with input decoder and filter threads limited to one. The existing `h264_videotoolbox` encoder at about 3 Mb/s is a tested fallback for a 1080p export. Verify the complete output before moving failed partial files out of the delivery folder. Record the fallback command and results in the final verification report.

## Verification and handoff

Verify through real downloaded files. Check that all carousel positions are present, the MP4 has the expected audio, and its duration matches the source. Inspect a frame when video quality matters; a decode pass proves readability rather than visual fidelity. If an encoded file is unexpectedly large, adjust the compression from the source.

If repetition is suspected, compare the decoded first and second occurrences. A source repetition belongs to the source; create a clearly named single-copy version only when requested. Browser looping alone does not change downloaded duration.

Return clickable file links and, when useful, an inline image, video, or audio player. State the number of items, basic media details, and any unavailable audio. Report completion only after the final files pass verification.
