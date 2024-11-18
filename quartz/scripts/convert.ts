import { execSync } from "child_process"
import { promises as fs } from "fs"
import path from "path"
import { globby } from "globby"
import chalk from "chalk"

async function convertMedia(contentDir: string) {
  try {
    const mediaFiles = await globby(["**/*.png"], {
      cwd: contentDir,
      absolute: true,
    })

    if (mediaFiles.length === 0) {
      console.log(chalk.yellow("No media files found to convert."))
      return
    }

    console.log(chalk.blue(`Found ${mediaFiles.length} media files to convert.`))

    for (const mediaFile of mediaFiles) {
      const ext = path.extname(mediaFile)
      let outputFile: string
      let ffmpegCmd: string

      switch (ext) {
        case ".png":
          outputFile = mediaFile.replace(/\.png$/, ".jpeg")
          ffmpegCmd = `ffmpeg -i "${mediaFile}" -quality 100 -compression_level 9 "${outputFile}"`
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
    })

    console.log(chalk.blue(`\nUpdating ${mdFiles.length} markdown files...`))

    for (const mdFile of mdFiles) {
      let content = await fs.readFile(mdFile, "utf-8")
      const originalContent = content

      content = content.replace(/\.(png|mp4)([^\w]|$)/g, (_match, ext) => {
        return ext === "png" ? ".jpeg$2" : ".avif$2"
      })
      content = content.replace(
        /\[\[([^\]]+\.(png|mp4))(\|[^\]]+)?\]\]/g,
        (_match, p1, ext, p3) => {
          const newExt = ext === "png" ? "jpeg" : "avif"
          return `[[${p1.replace(`.${ext}`, `.${newExt}`)}${p3 || ""}]]`
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
