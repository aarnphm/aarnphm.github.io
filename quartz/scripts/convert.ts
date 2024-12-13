import { execSync } from "child_process"
import { promises as fs } from "fs"
import path from "path"
import { globby } from "globby"
import chalk from "chalk"

async function convertMedia(contentDir: string) {
  try {
    const mediaFiles = await globby(["**/*.{png,jpeg,jpg}"], {
      cwd: contentDir,
      ignore: ["**/*.ignore.{png,jpeg,jpg}", "**/[0-9]*"],
      absolute: true,
    })

    if (mediaFiles.length === 0) {
      console.log(chalk.yellow("No media files found to convert."))
      return
    }

    console.log(chalk.blue(`Found ${mediaFiles.length} media files to convert.`))

    for (const mediaFile of mediaFiles) {
      const ext = path.extname(mediaFile).toLowerCase()
      let outputFile: string
      let ffmpegCmd: string

      switch (ext) {
        case ".png":
        case ".jpeg":
        case ".jpg":
          outputFile = mediaFile.replace(/\.(png|jpe?g)$/i, ".webp")
          ffmpegCmd = `ffmpeg -i "${mediaFile}" -c:v libwebp -quality 90 -compression_level 6 "${outputFile}"`
          break
        case ".mp4":
          outputFile = mediaFile.replace(/\.mp4$/, ".avif")
          ffmpegCmd = `ffmpeg -i "${mediaFile}" -c:v libaom-av1 "${outputFile}"`
          break
        default:
          continue
      }

      try {
        execSync(ffmpegCmd, { stdio: "inherit" })
        await fs.unlink(mediaFile)
        console.log(
          chalk.green(`Converted: ${path.basename(mediaFile)} -> ${path.basename(outputFile)}`),
        )
      } catch (error) {
        console.error(chalk.red(`Failed to convert ${mediaFile}:`), error)
      }
    }

    const mdFiles = await globby(["**/*.md"], {
      cwd: contentDir,
      absolute: true,
      ignore: ["**/posts/**/*.md"],
    })

    console.log(chalk.blue(`\nUpdating ${mdFiles.length} markdown files...`))

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
        console.log(chalk.green(`Updated references in: ${path.basename(mdFile)}`))
      }
    }

    console.log(chalk.green("\nMedia conversion and markdown updates completed!"))
  } catch (error) {
    console.error(chalk.red("Error during conversion:"), error)
    process.exit(1)
  }
}

const contentDir = process.argv[2] || path.join(process.cwd(), "content")
convertMedia(contentDir).catch(console.error)
