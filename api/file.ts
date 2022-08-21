import path from "path"
import type { VercelRequest, VercelResponse } from "."

const ALLOWED_EXTENSIONS = new Set([
  ".py",
  ".go",
  ".java",
  ".c",
  ".cpp",
  ".cxx",
  ".cu",
  ".cuh",
  ".h",
  ".hpp",
  ".ts",
  ".js",
  ".yaml",
  ".yml",
  ".rs",
  ".m",
  ".sql",
  ".sh",
  ".txt",
])

function sanitizePath(filePath: string): string {
  // Remove any leading/trailing whitespace
  let cleaned = filePath.trim()

  // Normalize path separators
  cleaned = path.normalize(cleaned).replace(/\\/g, "/")

  // Remove any attempts at path traversal
  cleaned = cleaned
    .split("/")
    .filter((segment) => {
      return (
        segment !== ".." &&
        segment !== "." &&
        !segment.includes("%2e") && // URL encoded dots
        !segment.includes("%2E")
      )
    })
    .join("/")

  // Remove any double slashes and leading/trailing slashes
  return cleaned.replace(/\/+/g, "/").replace(/^\/+|\/+$/g, "")
}

function isValidPath(filePath: string): boolean {
  // Check for null bytes
  if (filePath.includes("\0")) return false

  // Check extension is allowed
  const ext = path.extname(filePath).toLowerCase()
  if (!ALLOWED_EXTENSIONS.has(ext)) return false

  // Basic path validation
  if (!filePath || filePath.includes("..")) return false

  // Check for any sneaky encoded characters
  try {
    decodeURIComponent(filePath)
  } catch {
    return false
  }

  return true
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  try {
    // Only allow GET requests
    if (req.method !== "GET") {
      return res.status(405).json({ error: "Method not allowed" })
    }

    const { query } = req
    if (!query.path || typeof query.path !== "string") {
      return res.status(400).json({ error: "Invalid file path" })
    }

    // Decode and sanitize the path
    const rawPath = decodeURIComponent(query.path)
    const cleanPath = sanitizePath(rawPath)

    // Validate the path
    if (!isValidPath(cleanPath)) {
      console.warn("Invalid path detected:", rawPath)
      return res.status(403).json({ error: "Invalid file path" })
    }

    console.log("Accessing file:", cleanPath)

    const response = await fetch(`https://cdn.aarnphm.xyz/${cleanPath}`, {
      headers: {
        Accept: "text/plain",
        "User-Agent": "Vercel Serverless Function",
      },
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const content = await response.text()

    // Set security headers
    res.setHeader("X-Content-Type-Options", "nosniff")
    res.setHeader("Content-Security-Policy", "default-src 'none'; script-src 'none';")
    res.setHeader("X-Frame-Options", "DENY")
    res.setHeader("Cache-Control", "no-store, no-cache, must-revalidate")
    res.setHeader("Content-Type", "text/plain; charset=utf-8")
    return res.send(content)
  } catch (error) {
    console.error("Error:", error instanceof Error ? error.message : "Unknown error")
    return res.status(500).json({ error: "Internal server error" })
  }
}
