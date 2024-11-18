import path from "path"
import type { VercelRequest, VercelResponse } from "."
import { spawn } from "child_process"

// NOTE: make sure to also update vercel.json{redirects[0].source}
const ALLOWED_EXTENSIONS = [
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
]

function joinSegments(...args: string[]): string {
  return args
    .filter((segment) => segment !== "")
    .join("/")
    .replace(/\/\/+/g, "/")
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  const { query } = req
  const filePath = decodeURIComponent(query.path as string)

  const baseDir = process.cwd() // Root directory of the project

  // Resolve the full file path safely to prevent directory traversal attacks
  const basePath = process.env.VERCEL_ENV === "production" ? "./" : "public"
  const fullPath = path.resolve(baseDir, joinSegments(basePath, filePath))

  const ls = spawn("ls", ["-lhR", "/usr"])

  ls.stdout.on("data", (data) => {
    console.log(`stdout: ${data}`)
  })

  ls.stderr.on("data", (data) => {
    console.error(`stderr: ${data}`)
  })

  ls.on("close", (code) => {
    console.log(`child process exited with code ${code}`)
  })

  console.log(fullPath)

  try {
    // Ensure the requested file is within the base directory
    if (!fullPath.startsWith(baseDir)) {
      res.status(400).send("Invalid file path")
      return
    }

    const ext = path.extname(fullPath).toLowerCase()
    if (!ALLOWED_EXTENSIONS.includes(ext)) {
      return res.status(403).json({ text: "File type not allowed" })
    }

    const resp = await fetch(joinSegments("https://aarnphm.xyz", fullPath))
    const content = await resp.text()

    // Set the appropriate headers and return the file content
    res.setHeader("Content-Type", "text/plain")
    res.send(content)
  } catch (error) {
    console.error(error)
    res.status(404).send("File not found")
  }
}
