import { execSync } from "child_process"
import { promises as fs } from "fs"
import path from "path"
import { globby } from "globby"
import { styleText } from "node:util"

async function convertMedia(contentDir: string) {
  try {
    const mediaFiles = await globby(["**/*.{png,jpeg,jpg}"], {
      cwd: contentDir,
      ignore: [
        "**/*.ignore.{png,jpeg,jpg}",
        "android-chrome-*",
        "apple-touch-icon.png",
        "favicon-*",
        "movies/",
        "library/",
      ],
      absolute: true,
    })

    if (mediaFiles.length === 0) {
      console.log(styleText("yellow", "No media files found to convert."))
      return
    }

    console.log(styleText("blue", `Found ${mediaFiles.length} media files to convert.`))

    for (const mediaFile of mediaFiles) {
      const ext = path.extname(mediaFile).toLowerCase()
      let outputFile: string
      let ffmpegCmd: string

      switch (ext) {
        case ".png":
        case ".jpeg":
        case ".jpg":
          outputFile = mediaFile.replace(/\.(png|jpe?g)$/i, ".webp")
          ffmpegCmd = `ffmpeg -y -i "${mediaFile}" -c:v libwebp -quality 90 -compression_level 6 "${outputFile}"`
          break
        default:
          continue
      }

      try {
        execSync(ffmpegCmd, { stdio: "inherit" })
        await fs.unlink(mediaFile)
        console.log(
          styleText(
            "green",
            `Converted: ${path.basename(mediaFile)} -> ${path.basename(outputFile)}`,
          ),
        )
      } catch (error) {
        console.error(styleText("red", `Failed to convert ${mediaFile}:`), error)
      }
    }

    const mdFiles = await globby(["**/*.md"], {
      cwd: contentDir,
      absolute: true,
    })

    console.log(styleText("blue", `\nUpdating ${mdFiles.length} markdown files...`))

    for (const mdFile of mdFiles) {
      let content = await fs.readFile(mdFile, "utf-8")
      const originalContent = content

      // Replace both png and jpeg/jpg references, skipping ignored files
      content = content.replace(/(?<!\.ignore)\.(png|jpe?g)([^\w]|$)/gi, ".webp$2")

      // Handle both file types in wikilinks
      content = content.replace(
        /\[\[([^\]]+(?<!\.ignore)\.(png|jpe?g))(\|[^\]]+)?\]\]/gi,
        (_match, p1, _ext, p2) => {
          return `[[${p1.replace(/\.(png|jpe?g)$/i, ".webp")}${p2 || ""}]]`
        },
      )

      if (content !== originalContent) {
        await fs.writeFile(mdFile, content, "utf-8")
        console.log(styleText("green", `Updated references in: ${path.basename(mdFile)}`))
      }
    }

    console.log(styleText("green", "\nMedia conversion and markdown updates completed!"))
  } catch (error) {
    console.error(styleText("red", "Error during conversion:"), error)
    process.exit(1)
  }
}

const contentDir = process.argv[2] || path.join(process.cwd(), "content")
convertMedia(contentDir).catch(console.error)
