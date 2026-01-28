import sharp from "sharp"
import { QuartzEmitterPlugin } from "../../types/plugin"
import { joinSegments, QUARTZ, FilePath } from "../../util/path"

export const Favicon: QuartzEmitterPlugin = () => ({
  name: "Favicon",
  async *emit({ argv }) {
    const iconPath = joinSegments(QUARTZ, "static", "icon.png")

    const files = [
      { path: "favicon.ico", size: 48 },
      { size: 16, path: "favicon-16x16.png" },
      { size: 32, path: "favicon-32x32.png" },
    ]
    for (const f of files) {
      const dest = joinSegments(argv.output, f.path) as FilePath
      await sharp(iconPath).resize(f.size, f.size).toFormat("png").toFile(dest)
      yield dest
    }
  },
  async *partialEmit() {},
})
