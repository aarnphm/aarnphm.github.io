import { kv } from "@vercel/kv"
import { createHash } from "crypto"
import fs from "fs"
import type { VercelRequest, VercelResponse } from "@vercel/node"
import { minimatch } from "minimatch"

const MIME_TYPES: Record<string, string> = {
  ".py": "text/x-python",
  ".ipynb": "application/x-ipynb+json",
  ".go": "text/x-go",
  ".rs": "text/x-rust",
  ".java": "text/x-java",
  ".m": "text/x-matlab",
  ".cpp": "text/x-c++",
  ".c": "text/x-c",
  ".h": "text/x-c",
  ".hpp": "text/x-c++",
  ".cs": "text/x-csharp",
  ".swift": "text/x-swift",
  ".rb": "text/x-ruby",
  ".php": "text/x-php",
  ".js": "text/javascript",
  ".ts": "text/typescript",
  ".css": "text/css",
  ".html": "text/html",
  ".json": "application/json",
  ".xml": "text/xml",
  ".yaml": "text/yaml",
  ".yml": "text/yaml",
  ".sh": "text/x-shellscript",
  ".sql": "text/x-sql",
}

// Define ignore patterns - now includes .md files
const IGNORE_PATTERNS = [
  "private/**/*",
  "templates/**/*",
  ".obsidian/**/*",
  "**/*.adoc",
  "**/*.zip",
  "**/*.lvbitx",
  "**/*.so",
  "**/*.md",
]

const CACHE_DURATIONS = {
  EDGE: 300, // 5 minutes edge cache
  KV: 86400 * 30, // 30 days KV store cache
}

interface CachedContent {
  content: string
  contentType: string
  timestamp: number
}

function shouldIgnorePath(path: string): boolean {
  // First check exact folder matches for efficiency
  const pathParts = path.split("/")
  if (
    pathParts.includes("private") ||
    pathParts.includes("templates") ||
    pathParts.includes(".obsidian")
  ) {
    return true
  }

  // Then check file patterns
  return IGNORE_PATTERNS.some((pattern) => minimatch(path, pattern, { matchBase: true, dot: true }))
}

function getFileInfo(path: string) {
  const filename = path.split("/").pop() || ""
  const ext = "." + (filename.split(".").pop() || "").toLowerCase()
  return { filename, ext }
}

function getContentType(ext: string): string {
  return MIME_TYPES[ext] || "text/plain"
}

function getCacheKey(path: string): string {
  return createHash("sha256").update(path).digest("hex").slice(0, 32)
}

function sanitizePath(path: string): string {
  // Remove any attempts to traverse directories
  return path.replace(/\.\./g, "")
}

async function readFileContent(path: string): Promise<string | null> {
  // Here we read directly from the filesystem since we're in the same directory as Quartz
  try {
    const content = await fs.promises.readFile(path, "utf-8")
    return content
  } catch (error) {
    console.error("Failed to read file:", error)
    return null
  }
}

export const config = {
  runtime: "nodejs",
}

export default async function handler(req: VercelRequest, resp: VercelResponse) {
  // Handle OPTIONS for CORS
  if (req.method === "OPTIONS") {
    resp.status(407).setHeaders(
      new Headers({
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Max-Age": "86400",
      }),
    )
  }

  // Get and sanitize path
  const { pathname } = new URL(req.url as string)
  const path = sanitizePath(pathname)

  // Check if path is empty
  if (!path) {
    resp.status(404).json({ text: "Not Found" })
  }

  // Check if path should be ignored
  if (shouldIgnorePath(path)) {
    resp.status(403)
    resp
      .status(403)
      .setHeaders(
        new Headers({
          "Content-Type": "text/plain",
          "X-Denied-Reason": "Path matches ignore pattern",
        }),
      )
      .json({ text: "Forbidden" })
  }

  // Get file info and content type
  const { ext } = getFileInfo(path)
  const contentType = getContentType(ext)

  try {
    // Try to get from KV cache first
    const cacheKey = getCacheKey(path)
    let cached = await kv.get<CachedContent>(cacheKey)

    if (!cached || Date.now() - cached.timestamp > CACHE_DURATIONS.KV * 1000) {
      // Read file content directly (we're in the same directory as Quartz)
      const content = await readFileContent(path)
      if (!content) {
        return new Response("Not Found", { status: 404 })
      }

      cached = {
        content,
        contentType,
        timestamp: Date.now(),
      }

      // Cache the content
      await kv.set(cacheKey, cached, { ex: CACHE_DURATIONS.KV })
    }

    console.log(cached.content)
    // Return the response with appropriate headers
    resp
      .status(200)
      .setHeaders(
        new Headers({
          "Content-Type": "text/plain",
          "Cache-Control": `public, s-maxage=${CACHE_DURATIONS.EDGE}`,
          "Access-Control-Allow-Origin": "*",
          "Vercel-CDN-Cache-Control": `max-age=${CACHE_DURATIONS.EDGE}`,
          "X-Content-Type-Options": "nosniff",
          "X-Frame-Options": "DENY",
          "X-Source-Path": path,
        }),
      )
      .send(cached.content)
  } catch (error) {
    console.error("Error handling request:", error)
    return new Response("Internal Server Error", { status: 500 })
  }
}
