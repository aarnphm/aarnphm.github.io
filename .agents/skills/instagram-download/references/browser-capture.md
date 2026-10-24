# Browser capture

## Failure modes to verify

- A carousel inventory can omit slides not visited yet.
- Preloaded recommendations and ads can contribute unrelated image, video, and audio URLs.
- DASH URLs with byte-range parameters return incomplete MP4 fragments.
- The CDN filename extension can differ from the served image format.
- A silent video stream can require a separate audio stream.
- Public and signed-in web views can omit a photo post's music.
- Full-track music URLs can differ from the attached excerpt's start and duration.
- A high-resolution CRF 18 H.264 transcode can be several times larger than Instagram's VP9 stream.
- A browser player can loop; only the downloaded media establishes its actual content and duration.

## Observe the requested media

Use supported read-only DOM inspection to record `currentSrc`, `naturalWidth`, `naturalHeight`, and the player's duration. Scope selection to the requested post. Large image dimensions alone do not distinguish the post from suggested posts.

On a carousel, the owning `ul` contains positioned `li` elements. Their rendered `translateX` positions can establish order when Instagram mounts only the current slide and its neighbors. Save each new slide's file URL and position before navigating. Use the visible Next control and inspect the resulting state. Do not assume the carousel has a fixed length or that every item is an image.

Read the browser's `pageAssets` documentation before use. An inventory includes observed resources even after their elements leave the DOM. If bundling fails for a previously visible slide, bring it into view, refresh the inventory, and retry its own asset. Direct downloads of an observed public CDN URL are also supported by the helper.

For blob-backed video, inspect loaded media assets rather than treating the blob URL as a complete file. The CDN's `efg` query value has sometimes contained base64 JSON with `vencode_tag`, `xpv_asset_id`, `duration_s`, and `bitrate`. Decode it as data, not code. Group renditions by asset identifier, distinguish the `_audio` tag, and match the group to the requested player. Do not select the largest asset from the whole page without that match. If metadata is insufficient, inspect the player's own network requests through the documented browser tools.

Use the public app view first when it exposes the requested media. If a photo's music is missing, inspect the post's own music controls or an already signed-in browser. Treat missing web audio as an access limitation. Never download an unrelated preloaded soundtrack as a substitute.

## Capture schema

Write the captured URLs or bundled file paths into this JSON shape. Positions are positive integers in post order. Each item supplies a `path` or `url`; use `audio_path` or `audio_url` for a video's separate audio. An audio item represents a track attached to a photo or the post as a whole.

```json
{
  "source_url": "https://www.instagram.com/p/SHORTCODE/",
  "account": "account-name",
  "audio_status": "present",
  "items": [
    { "position": 1, "kind": "image", "path": "/absolute/path/downloaded-image.jpg" },
    {
      "position": 2,
      "kind": "video",
      "url": "https://scontent-example.cdninstagram.com/observed-video.mp4?observed_parameters",
      "audio_url": "https://scontent-example.cdninstagram.com/observed-audio.mp4?observed_parameters",
      "expected_duration": 9.87
    }
  ]
}
```

`audio_status` is `present`, `none`, or `not_exposed`. For a photo with an available full music asset, add an audio item with `start_seconds` and `duration_seconds` from the post's own excerpt metadata. Do not infer these values from the track's full duration.

Sources may be local paths or observed HTTPS URLs on Instagram/Meta media CDN hosts. The helper does not authenticate or discover media on its own. It strips only byte-range parameters; preserve the rest of a signed CDN URL.

## Replay

`verification.json` records the exact helper invocation, source paths or CDN host/path, processing commands, outputs, and status. Replay with the original capture manifest while its URLs remain valid, or replace its URL fields with retained local source paths. Browser discovery is refreshed from the actual post each time; never retain another person's browser cookies as a fixture.
