import { execSync } from "child_process"
import { promises as fs } from "fs"
import path from "path"
import { globby } from "globby"
import chalk from "chalk"

async function convertImages(contentDir: string) {
  try {
    // Find all PNG files in content directory
    const pngFiles = await globby(["**/*.png"], {
      cwd: contentDir,
      absolute: true,
    })

    if (pngFiles.length === 0) {
      console.log(chalk.yellow("No PNG files found to convert."))
      return
    }

    console.log(chalk.blue(`Found ${pngFiles.length} PNG files to convert.`))

    // Convert each PNG to JPEG
    for (const pngFile of pngFiles) {
      const jpegFile = pngFile.replace(/\.png$/, ".jpeg")

      // Convert using ffmpeg with high compression
      try {
        execSync(`ffmpeg -i "${pngFile}" -quality 100 -compression_level 9 "${jpegFile}"`, {
          stdio: "inherit",
        })

        // Delete original PNG after successful conversion
        await fs.unlink(pngFile)
        console.log(
          chalk.green(`Converted: ${path.basename(pngFile)} -> ${path.basename(jpegFile)}`),
        )
      } catch (error) {
        console.error(chalk.red(`Failed to convert ${pngFile}:`), error)
      }
    }

    // Find all markdown files
    const mdFiles = await globby(["**/*.md"], {
      cwd: contentDir,
      absolute: true,
    })

    console.log(chalk.blue(`\nUpdating ${mdFiles.length} markdown files...`))

    // Update markdown files to reference jpeg instead of png
    for (const mdFile of mdFiles) {
      let content = await fs.readFile(mdFile, "utf-8")
      const originalContent = content

      // Replace both standard markdown image syntax and wikilink image syntax
      content = content.replace(/\.(png)([^\w]|$)/g, ".jpeg$2")
      content = content.replace(/\[\[([^\]]+\.png)(\|[^\]]+)?\]\]/g, (match, p1, p2) => {
        return `[[${p1.replace(".png", ".jpeg")}${p2 || ""}]]`
      })

      if (content !== originalContent) {
        await fs.writeFile(mdFile, content, "utf-8")
        console.log(chalk.green(`Updated references in: ${path.basename(mdFile)}`))
      }
    }

    console.log(chalk.green("\nImage conversion and markdown updates completed!"))
  } catch (error) {
    console.error(chalk.red("Error during conversion:"), error)
    process.exit(1)
  }
}

// Execute the script
const contentDir = process.argv[2] || path.join(process.cwd(), "content")
convertImages(contentDir).catch(console.error)
