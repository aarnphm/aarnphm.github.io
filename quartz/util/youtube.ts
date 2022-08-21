export type YouTubeEmbedKind = "video" | "playlist" | "video-with-playlist"

export type YouTubeEmbedSpec = {
  src: string
  kind: YouTubeEmbedKind
  videoId?: string
  playlistId?: string
  startSeconds?: string
  endSeconds?: string
  index?: string
}

const YOUTUBE_VIDEO_ID_REGEX = /^[0-9A-Za-z_-]{11}$/
const DURATION_REGEX = /(\d+)([hms])/gi

const stripHost = (value: string): string => (value.startsWith("www.") ? value.slice(4) : value)

const isKnownYouTubeHost = (host: string): boolean =>
  host === "youtu.be" || host.endsWith("youtube.com") || host.endsWith("youtube-nocookie.com")

const parseYouTubeTimestamp = (value: string | null): string | undefined => {
  if (!value) return undefined

  const trimmed = value.trim()
  if (trimmed.length === 0) return undefined

  if (/^\d+$/.test(trimmed)) {
    return trimmed.replace(/^0+/, "") || "0"
  }

  let matched = false
  let totalSeconds = 0

  trimmed.replace(DURATION_REGEX, (_match, amount, unit) => {
    matched = true
    const numeric = Number(amount)
    if (Number.isNaN(numeric)) return ""
    switch (unit.toLowerCase()) {
      case "h":
        totalSeconds += numeric * 3600
        break
      case "m":
        totalSeconds += numeric * 60
        break
      case "s":
        totalSeconds += numeric
        break
    }
    return ""
  })

  if (matched) {
    return totalSeconds > 0 ? String(totalSeconds) : "0"
  }

  if (/^\d{1,2}(:\d{2}){1,2}$/.test(trimmed)) {
    const parts = trimmed.split(":").map((part) => Number(part))
    if (parts.some((part) => Number.isNaN(part))) {
      return undefined
    }
    let seconds = 0
    for (const part of parts) {
      seconds = seconds * 60 + part
    }
    return seconds > 0 ? String(seconds) : "0"
  }

  return undefined
}

const extractYouTubeStart = (url: URL): string | undefined => {
  const searchStart =
    parseYouTubeTimestamp(url.searchParams.get("start")) ??
    parseYouTubeTimestamp(url.searchParams.get("t"))
  if (searchStart) return searchStart

  const hash = url.hash.startsWith("#") ? url.hash.slice(1) : url.hash
  if (!hash) return undefined

  const hashParams = hash.includes("=")
    ? new URLSearchParams(hash)
    : new URLSearchParams(`t=${hash}`)
  return (
    parseYouTubeTimestamp(hashParams.get("start")) ?? parseYouTubeTimestamp(hashParams.get("t"))
  )
}

export const buildYouTubeEmbed = (rawUrl: string): YouTubeEmbedSpec | undefined => {
  try {
    const url = new URL(rawUrl)
    const host = stripHost(url.hostname.toLowerCase())
    if (!isKnownYouTubeHost(host)) return undefined

    const segments = url.pathname.split("/").filter(Boolean)
    let videoId: string | undefined
    let playlistId: string | undefined

    if (host === "youtu.be") {
      videoId = segments[0]
    } else if (segments[0] === "watch") {
      videoId = url.searchParams.get("v") ?? undefined
    } else if (segments[0] === "embed" && segments[1]) {
      videoId = segments[1]
    } else if (segments[0] === "shorts" && segments[1]) {
      videoId = segments[1]
    } else if (segments[0] === "live" && segments[1]) {
      videoId = segments[1]
    }

    playlistId = url.searchParams.get("list") ?? undefined
    if (!videoId && segments[0] === "playlist") {
      playlistId = playlistId ?? segments[1]
    }

    if (!videoId) {
      const candidate = url.searchParams.get("v")
      if (candidate) {
        videoId = candidate
      }
    }

    if (videoId && !YOUTUBE_VIDEO_ID_REGEX.test(videoId)) {
      videoId = undefined
    }

    const start = extractYouTubeStart(url)
    const indexParam = url.searchParams.get("index") ?? url.searchParams.get("i") ?? undefined
    const endParam = url.searchParams.get("end") ?? undefined

    const params = new URLSearchParams()
    if (playlistId) {
      params.set("list", playlistId)
      if (indexParam && /^\d+$/.test(indexParam)) {
        params.set("index", indexParam)
      }
    }
    if (start && /^\d+$/.test(start)) {
      params.set("start", start)
    }
    if (endParam && /^\d+$/.test(endParam)) {
      params.set("end", endParam)
    }

    if (!videoId && !playlistId) {
      return undefined
    }

    const hasVideo = Boolean(videoId)
    const hasPlaylist = Boolean(playlistId)
    const kind: YouTubeEmbedKind = hasVideo
      ? hasPlaylist
        ? "video-with-playlist"
        : "video"
      : "playlist"

    const path = hasVideo ? `/embed/${videoId}` : "/embed/videoseries"
    const query = params.toString()
    const src = `https://www.youtube-nocookie.com${path}${query ? `?${query}` : ""}`

    return {
      src,
      kind,
      videoId,
      playlistId,
      startSeconds: params.get("start") ?? undefined,
      endSeconds: params.get("end") ?? undefined,
      index: params.get("index") ?? undefined,
    }
  } catch {
    return undefined
  }
}
